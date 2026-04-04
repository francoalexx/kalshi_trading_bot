"""
Main entry point — wires all components and runs the trading loop.

Architecture flow (per 1-minute bar close):
  1. BinanceFeed delivers a closed candle → Feature Engine updates indicators
  2. OFI module flushes bar metrics (OFI, VPIN, aggressor_ratio)
  3. AlphaEngine evaluates CNN-LSTM + XGBoost → TradeSignal
  4. KalshiFeed provides current MarketState (price, expiry, spread)
  5. RiskManager checks position/loss/rate limits
  6. ExecutionController sizes position (Kelly) and places limit order
  7. Order is monitored until fill or T-90s cancellation
  8. Settlement monitor polls Kalshi for results and updates all systems

Paper mode (--paper flag): all steps run but no real orders are placed.
Live mode: real orders via Kalshi API; settlement tracked identically.

Usage:
    python -m src.main --paper                     # paper trading mode
    python -m src.main                             # live trading mode
    python -m src.main --cnn models/artifacts/cnn_lstm_best.pt \
                       --xgb models/artifacts/xgboost_best.pkl
"""

import argparse
import ast
import asyncio
import os
import signal as _signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.data.binance_feed import BinanceFeed
from src.data.kalshi_feed import KalshiFeed
from src.execution.controller import ExecutionController
from src.execution.kalshi_client import KalshiClient
from src.features.indicators import compute_indicators, extract_feature_vector, build_sequence_tensor
from src.features.ofi import OFICalculator
from src.engine.alpha import AlphaEngine, Direction
from src.risk.manager import RiskManager
from src.paper_tracker import PaperTracker
from src.learning.trade_journal import TradeJournal
from src.learning.heuristics import HeuristicsEngine
from src.learning.postmortem import PostmortemEngine
from src.learning.attribution import AttributionAnalyzer
from src.notifications import send_settlement_email

load_dotenv()

# ─── Logging setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="00:00")


# ─── Pending trade: unified tracking for both paper and live ─────────────────

@dataclass
class PendingTrade:
    """Tracks an open trade awaiting settlement (paper or live)."""
    ticker:            str
    side:              str        # "yes" or "no"
    contracts:         int
    entry_price:       float     # 0-1 scale
    expiry_ts:         float     # unix epoch
    signal_edge:       float
    journal_trade_id:  Optional[int] = None
    paper_trade:       object = None   # PaperTrade (paper mode only)
    settled:           bool = False


SETTLEMENT_POLL_INTERVAL = 30   # seconds between settlement checks
SETTLEMENT_TIMEOUT       = 600  # give up after 10 minutes post-expiry


async def run_bot(
    cnn_model_path: str,
    xgb_model_path: str,
    paper_mode: bool = False,
):
    bankroll        = float(os.getenv("INITIAL_BANKROLL", "500"))
    max_open        = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    daily_loss_pct  = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.10"))
    max_tph         = int(os.getenv("MAX_TRADES_PER_HOUR", "20"))
    min_edge        = float(os.getenv("MIN_NET_EDGE", "0.01"))

    # ── Component initialisation ─────────────────────────────────────────
    risk = RiskManager(
        initial_bankroll=bankroll,
        max_open_positions=max_open,
        daily_loss_limit_pct=daily_loss_pct,
        max_trades_per_hour=max_tph,
    )

    kalshi = KalshiClient(env=os.getenv("KALSHI_ENV", "demo"))

    # Validate credentials for live trading
    if not paper_mode and kalshi._private_key is None:
        logger.critical(
            "LIVE MODE requires a valid Kalshi private key. "
            "Set KALSHI_PRIVATE_KEY_PATH in .env or use --paper."
        )
        sys.exit(1)

    binance_feed = BinanceFeed()
    # Always use real Kalshi markets (even in paper mode) so we test on the
    # exact same market data as live trading.  paper_mode only affects whether
    # real orders are sent to the exchange.
    market_feed = KalshiFeed(client=kalshi)
    ofi_calc = OFICalculator()

    alpha = AlphaEngine(
        cnn_lstm_path=cnn_model_path,
        xgb_path=xgb_model_path,
        min_net_edge=min_edge,
    )

    controller = ExecutionController(
        client=kalshi,
        risk_manager=risk,
        bankroll=bankroll,
        paper_mode=paper_mode,
    )

    paper = PaperTracker(starting_bankroll=bankroll) if paper_mode else None

    # ── Learning system ──────────────────────────────────────────────────
    journal    = TradeJournal()
    heuristics = HeuristicsEngine()
    analyzer   = AttributionAnalyzer(journal)
    postmortem = PostmortemEngine(journal, analyzer)
    # Inject heuristics engine into alpha so rules fire on every signal
    alpha.heuristics_engine = heuristics

    # ── Unified pending-trade list (paper + live) ────────────────────────
    pending_trades: List[PendingTrade] = []

    # ── SIGINT handler → graceful kill switch ────────────────────────────
    def _handle_sigint(sig, frame):
        logger.warning("SIGINT received. Shutting down.")
        os._exit(0)

    _signal.signal(_signal.SIGINT, _handle_sigint)

    mode_label = "PAPER" if paper_mode else "LIVE"
    logger.info(f"Bot starting — mode={mode_label} bankroll=${bankroll:.2f}")

    # ── Settlement monitor: polls Kalshi for market results ──────────────
    async def monitor_settlements():
        """
        Every 30s check expired trades and settle them using the real
        Kalshi market result ("yes" / "no"). Works for both paper and
        live trades. Updates risk manager, journal, postmortem, and
        sends email notifications.
        """
        while True:
            await asyncio.sleep(SETTLEMENT_POLL_INTERVAL)
            now = time.time()

            for pt in pending_trades:
                if pt.settled:
                    continue
                if pt.expiry_ts > now:
                    continue   # not yet expired

                if now - pt.expiry_ts > SETTLEMENT_TIMEOUT:
                    logger.warning(
                        f"Settlement timeout for {pt.ticker} — marking as loss"
                    )
                    _settle_trade(pt, won=False)
                    continue

                try:
                    result = _fetch_market_result(pt.ticker)
                    if result in ("yes", "no"):
                        logger.info(
                            f"Kalshi result for {pt.ticker}: {result.upper()}"
                        )
                        won = (pt.side == result)
                        _settle_trade(pt, won=won)
                except Exception as exc:
                    logger.debug(
                        f"Settlement poll error for {pt.ticker}: {exc}"
                    )

    def _fetch_market_result(ticker: str) -> str:
        """Fetch and normalise the market result from Kalshi API."""
        raw = kalshi._read_request(f"/markets/{ticker}")
        mkt = raw.get("market", raw)
        result = mkt.get("result", "")
        # Kalshi sometimes wraps scalar values in a list e.g. '[9.56E-2]'
        if isinstance(result, list):
            result = result[0] if result else ""
        if isinstance(result, str) and result.startswith("["):
            try:
                parsed = ast.literal_eval(result)
                result = parsed[0] if isinstance(parsed, list) else str(parsed)
            except Exception:
                result = result.strip("[]")
        return str(result).strip().lower()

    def _settle_trade(pt: PendingTrade, won: bool) -> None:
        """
        Handle settlement for both paper and live trades:
        1. Compute P&L and fees
        2. Update paper tracker (paper mode) or risk manager (live mode)
        3. Log to trade journal
        4. Run postmortem analysis
        5. Send email notification
        6. Update controller bankroll
        """
        # ── P&L calculation (same formula for paper and live) ────────
        if won:
            gross_pnl = (1.0 - pt.entry_price) * pt.contracts
        else:
            gross_pnl = -pt.entry_price * pt.contracts

        fee = 0.07 * pt.entry_price * (1 - pt.entry_price) * pt.contracts
        net_pnl = gross_pnl - fee
        outcome = "WIN" if won else "LOSS"

        pt.settled = True

        # ── Paper tracker (paper mode) ───────────────────────────────
        if paper is not None and pt.paper_trade is not None:
            paper.settle_by_result(
                pt.paper_trade,
                pt.side if won else ("no" if pt.side == "yes" else "yes"),
            )

        # ── Risk manager update (both modes) ─────────────────────────
        risk.record_settlement(
            contracts_won=pt.contracts if won else 0,
            contracts_lost=pt.contracts if not won else 0,
            gross_pnl=net_pnl,
            fees_paid=fee,
        )

        # ── Update controller bankroll for correct Kelly sizing ──────
        controller.update_bankroll(risk.bankroll)

        # ── Journal + postmortem ─────────────────────────────────────
        if pt.journal_trade_id is not None:
            journal.log_settlement(
                pt.journal_trade_id, net_pnl, fee, outcome
            )
            postmortem.on_settlement(
                pt.journal_trade_id, net_pnl, fee, outcome
            )

        # ── Email notification (live mode — paper uses PaperTracker) ─
        if paper is None:
            _send_live_email(pt, won, net_pnl, fee)

        result_str = "WIN" if won else "LOSS"
        logger.info(
            f"[{mode_label}] Trade settled — {result_str} | "
            f"pnl={net_pnl:+.2f} fee={fee:.2f} | "
            f"bankroll=${risk.bankroll:.2f}"
        )

    def _send_live_email(pt: PendingTrade, won: bool, pnl: float, fee: float):
        """Send email notification for live trade settlement."""
        from types import SimpleNamespace

        current = SimpleNamespace(
            trade_id=pt.journal_trade_id or 0,
            side=pt.side,
            contracts=pt.contracts,
            entry_price=pt.entry_price,
            won=won,
            pnl=pnl,
            fee=fee,
        )

        try:
            send_settlement_email(
                trade=current,
                bankroll=risk.bankroll,
                start_bankroll=bankroll,
                settled_trades=[current],
            )
        except Exception as exc:
            logger.warning(f"Email notification failed: {exc}")

    # ── Start background tasks ───────────────────────────────────────────
    asyncio.create_task(binance_feed.run(),       name="binance_feed")
    asyncio.create_task(market_feed.run(),        name="market_feed")
    asyncio.create_task(monitor_settlements(),    name="settlement_monitor")

    # ── Candle buffer for indicator computation ──────────────────────────
    candle_df_rows = []

    # ── Order book consumer (feeds OFI) ─────────────────────────────────
    async def consume_books():
        while True:
            snap = await binance_feed.book_queue.get()
            ofi_calc.update(snap)

    asyncio.create_task(consume_books(), name="book_consumer")

    # Wait until we have a full 60-bar buffer
    logger.info("Waiting for 60-candle warm-up...")
    while not binance_feed.is_buffer_ready():
        await asyncio.sleep(1)

    logger.info("Warm-up complete. Trading loop started.")

    # ── Main trading loop ────────────────────────────────────────────────
    while True:
        # Block until a closed candle arrives
        candle = await binance_feed.candle_queue.get()
        if not candle.is_closed:
            continue  # skip in-progress candles; only act on bar close

        if risk.kill_switch.is_set():
            logger.warning("Kill switch active. Sleeping 60s...")
            await asyncio.sleep(60)
            continue

        # ── Build feature DataFrame ──────────────────────────────────
        candle_df_rows.append({
            "open":   candle.open,
            "high":   candle.high,
            "low":    candle.low,
            "close":  candle.close,
            "volume": candle.volume,
        })
        # Keep last 120 bars (2x seq_len for safety)
        if len(candle_df_rows) > 120:
            candle_df_rows = candle_df_rows[-120:]

        df = pd.DataFrame(candle_df_rows)
        df = compute_indicators(df)

        if len(df) < 60:
            continue  # not enough bars for sequence model

        # ── OFI flush ────────────────────────────────────────────────
        ofi_metrics = ofi_calc.flush()

        # ── Feature extraction ───────────────────────────────────────
        features = extract_feature_vector(df)
        features["ofi_norm"]        = ofi_metrics.ofi_norm
        features["aggressor_ratio"] = ofi_metrics.aggressor_ratio
        features["spread_pct"]      = ofi_metrics.spread_pct

        sequence = build_sequence_tensor(df, seq_len=60)
        if sequence is None:
            continue

        # ── Market state (simulated or live) ─────────────────────────
        market = market_feed.get_active_market()
        if market is None:
            logger.debug("No active Kalshi market. Skipping bar.")
            continue

        # ── Alpha evaluation ─────────────────────────────────────────
        signal = alpha.evaluate(
            sequence=sequence,
            features=features,
            market_price=market.yes_ask,
            spread=market.spread_yes,
            strike_type=market.strike_type,
        )

        # ── Log bar to journal (regardless of trade) ─────────────────
        bar_id = journal.log_bar(
            features=features,
            signal=signal,
            market=market,
            btc_close=candle.close,
            traded=(signal.direction != Direction.NO_TRADE),
        )

        if signal.direction == Direction.NO_TRADE:
            continue

        # ── Execution ────────────────────────────────────────────────
        result = await controller.execute(signal, market)
        if result is not None:
            logger.info(
                f"Trade complete: {result.side.upper()} {result.filled}/{result.contracts} "
                f"contracts filled @ avg {result.avg_fill_price or 'N/A'} "
                f"slippage={result.slippage:.4f}"
            )

            # ── Journal: log trade ───────────────────────────────────
            trade_id = journal.log_trade(
                bar_id=bar_id,
                result=result,
                kelly_fraction=getattr(result, "kelly_fraction", 0.0),
            )

            # ── Track for settlement (both paper and live) ───────────
            if result.filled > 0:
                entry = result.avg_fill_price or market.yes_ask
                pt = PendingTrade(
                    ticker=result.ticker,
                    side=result.side,
                    contracts=result.filled,
                    entry_price=entry,
                    expiry_ts=market.expiry_ts,
                    signal_edge=signal.net_edge,
                    journal_trade_id=trade_id,
                )

                # Paper mode: also record in PaperTracker for dashboard
                if paper is not None:
                    paper_trade = paper.record_trade(
                        ticker=result.ticker,
                        side=result.side,
                        contracts=result.filled,
                        entry_price=entry,
                        expiry_ts=market.expiry_ts,
                        signal_edge=signal.net_edge,
                    )
                    pt.paper_trade = paper_trade

                pending_trades.append(pt)
                # Settlement handled by monitor_settlements() coroutine


def main():
    parser = argparse.ArgumentParser(description="BTC Prediction Market Trading Bot")
    parser.add_argument(
        "--cnn", default="models/artifacts/cnn_lstm_best.pt",
        help="Path to CNN-LSTM checkpoint (.pt)"
    )
    parser.add_argument(
        "--xgb", default="models/artifacts/xgboost_best.pkl",
        help="Path to XGBoost model (.pkl)"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run in paper trading mode (no real orders)"
    )
    args = parser.parse_args()

    # Validate model paths exist (warn if not — DRY RUN mode will activate)
    for label, path in [("CNN-LSTM", args.cnn), ("XGBoost", args.xgb)]:
        if not Path(path).exists():
            logger.warning(
                f"{label} model not found at '{path}'. "
                "Run src/models/train.py first, or operate in paper mode."
            )

    asyncio.run(run_bot(
        cnn_model_path=args.cnn,
        xgb_model_path=args.xgb,
        paper_mode=args.paper,
    ))


if __name__ == "__main__":
    main()
