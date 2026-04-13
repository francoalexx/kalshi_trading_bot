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
from src.learning.shadow_model import ShadowModelManager
from src.data.kalshi_price_logger import KalshiPriceLogger
from src.notifications import send_settlement_email

load_dotenv()

# ─── Logging setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="00:00")


# ─── Pending trade: unified tracking for both paper and live ─────────────────

@dataclass
class PendingTrade:
    """Tracks an open trade awaiting settlement or scalp exit (paper or live)."""
    ticker:            str
    side:              str        # "yes" or "no"
    contracts:         int
    entry_price:       float     # 0-1 scale
    expiry_ts:         float     # unix epoch
    signal_edge:       float
    journal_trade_id:  Optional[int] = None
    paper_trade:       object = None   # PaperTrade (paper mode only)
    settled:           bool = False
    # Scalp-mode fields (unused in settlement mode)
    mode:              str = "settlement"   # "settlement" or "scalp"
    exit_price:        Optional[float] = None
    exit_ts:           Optional[float] = None
    sell_order_id:     Optional[str] = None
    sell_fee:          float = 0.0


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
    max_edge        = float(os.getenv("MAX_NET_EDGE", "0.10"))

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
        max_net_edge=max_edge,
    )

    controller = ExecutionController(
        client=kalshi,
        risk_manager=risk,
        bankroll=bankroll,
        paper_mode=paper_mode,
    )

    paper = PaperTracker(starting_bankroll=bankroll) if paper_mode else None

    # ── Learning system ──────────────────────────────────────────────────
    # Separate DB files so paper and live bots can run simultaneously
    db_path = Path("data/trade_journal_paper.db") if paper_mode else Path("data/trade_journal.db")
    journal    = TradeJournal(db_path=db_path)
    heuristics = HeuristicsEngine()
    analyzer   = AttributionAnalyzer(journal)
    postmortem = PostmortemEngine(journal, analyzer)
    # Inject heuristics engine into alpha so rules fire on every signal
    alpha.heuristics_engine = heuristics

    # ── Shadow model (paper mode only) ───────────────────────────────────
    # Runs a parallel XGBoost with alternative hyperparameters on every bar.
    # Never places orders; logs observations to the trade journal so a
    # statistically-tested promotion candidate is ready to compare against
    # the live model after enough observations accumulate. Observations
    # persist across bot restarts via the shadow_observations SQLite table.
    shadow_mgr: Optional[ShadowModelManager] = None
    if paper_mode:
        shadow_mgr = ShadowModelManager(
            live_model_path=xgb_model_path,
            # Candidate params: deeper trees + slower learning rate vs. the
            # live model's defaults. First principled A/B experiment — change
            # these once Optuna starts feeding real candidates.
            candidate_params={
                "max_depth":     6,
                "learning_rate": 0.03,
                "n_estimators":  300,
                "subsample":     0.85,
            },
            # Slightly wider confidence band than live (0.55/0.45) to test
            # whether stricter entry raises win rate enough to offset lower
            # trade frequency.
            xgb_yes_thresh=0.56,
            xgb_no_thresh=0.44,
            journal=journal,
        )
        if not shadow_mgr.train_from_parquet("data/btc_1m.parquet"):
            logger.warning(
                "Shadow model failed to train from parquet. "
                "Paper run will continue without shadow observations."
            )
            shadow_mgr = None
        else:
            stats = journal.get_shadow_stats()
            logger.info(
                f"Shadow model active. Existing observations in journal: "
                f"{stats['n_observations']:,} "
                f"(would-trade: {stats['n_would_trade']:,}, "
                f"settled: {stats['n_settled']:,}, "
                f"win_rate: {stats['shadow_win_rate']:.1%})"
            )

    # ── Unified pending-trade list (paper + live) ────────────────────────
    pending_trades: List[PendingTrade] = []
    # Tickers with an unsettled trade already placed. Prevents the bot from
    # stacking multiple orders on the same 15-min expiry (see Lessons Learned:
    # "Same-market stacking" 2026-04-10).
    traded_tickers: set = set()
    # Tickers where a "too good to be true" signal fired. Once tainted, the
    # bot skips the ticker for its entire lifecycle — the model-market
    # disagreement doesn't go away just because market_price moves closer
    # to p_yes (see 2026-04-13 session notes).
    tainted_tickers: set = set()

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
                        # Update any shadow observations tied to this market.
                        if shadow_mgr is not None:
                            n = shadow_mgr.record_settlement(pt.ticker, result)
                            if n > 0:
                                logger.debug(
                                    f"Shadow: settled {n} observation(s) for {pt.ticker}"
                                )
                except Exception as exc:
                    logger.debug(
                        f"Settlement poll error for {pt.ticker}: {exc}"
                    )

            # ── Shadow-only sweep ───────────────────────────────────────
            # Many shadow observations sit on markets where live did not
            # trade. Sweep them here: find distinct expired tickers with
            # unsettled shadow rows and fetch each result once.
            if shadow_mgr is not None:
                try:
                    rows = journal._conn.execute(
                        """
                        SELECT DISTINCT market_ticker
                        FROM shadow_observations
                        WHERE settled = 0
                          AND expiry_ts IS NOT NULL
                          AND expiry_ts < ?
                        LIMIT 25
                        """,
                        (now,),
                    ).fetchall()
                    for r in rows:
                        ticker = r["market_ticker"]
                        if not ticker:
                            continue
                        try:
                            result = _fetch_market_result(ticker)
                            if result in ("yes", "no"):
                                shadow_mgr.record_settlement(ticker, result)
                        except Exception as exc:
                            logger.debug(
                                f"Shadow sweep poll error for {ticker}: {exc}"
                            )
                except Exception as exc:
                    logger.debug(f"Shadow sweep query error: {exc}")

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
        # Release the per-ticker dedup slot so the next expiry can trade.
        traded_tickers.discard(pt.ticker)
        tainted_tickers.discard(pt.ticker)

        # ── Paper tracker (paper mode) ───────────────────────────────
        if paper is not None and pt.paper_trade is not None:
            paper.settle_by_result(
                pt.paper_trade,
                pt.side if won else ("no" if pt.side == "yes" else "yes"),
            )

        # ── Release position slot now that market has settled ─────────
        risk.register_order_closed(pt.contracts, pt.entry_price)

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

    # Track all settled trades for cumulative email dashboard
    settled_history: List[object] = []

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
        settled_history.append(current)

        try:
            send_settlement_email(
                trade=current,
                bankroll=risk.bankroll,
                start_bankroll=bankroll,
                settled_trades=settled_history,
            )
        except Exception as exc:
            logger.warning(f"Email notification failed: {exc}")

    # ── Contract price logger (runs in all modes to collect scalp training data)
    scalp_poll = float(os.getenv("SCALP_POLL_INTERVAL", "5"))
    price_logger = KalshiPriceLogger(
        kalshi_client=kalshi,
        binance_feed=binance_feed,
        db_path=Path("data/contract_prices.db"),
        poll_interval=scalp_poll,
    )

    # ── Start background tasks ───────────────────────────────────────────
    asyncio.create_task(binance_feed.run(),       name="binance_feed")
    asyncio.create_task(market_feed.run(),        name="market_feed")
    asyncio.create_task(monitor_settlements(),    name="settlement_monitor")
    asyncio.create_task(price_logger.run(),       name="price_logger")

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

        # ── Shadow model observation ─────────────────────────────────
        # Run the shadow model on the same bar, regardless of whether
        # live will trade. The observation — and its eventual outcome —
        # is persisted to the journal via log_shadow_observation().
        if shadow_mgr is not None:
            try:
                shadow_mgr.record_bar(
                    bar_id=bar_id,
                    market=market,
                    features=features,
                    live_prob=signal.xgb_prob,
                    live_direction=signal.direction.value,
                )
            except Exception as exc:
                logger.debug(f"Shadow record_bar error: {exc}")

        # ── TGTBT taint tracking ────────────────────────────────────
        if signal.tgtbt_filtered:
            tainted_tickers.add(market.ticker)

        if signal.direction == Direction.NO_TRADE:
            continue

        if market.ticker in tainted_tickers:
            logger.info(
                f"Skipping {market.ticker}: previously flagged as 'too good "
                f"to be true' — model-market disagreement is structural."
            )
            continue

        # ── Per-market dedup ────────────────────────────────────────
        # Never place more than one trade per market expiry. Averaging
        # into a losing position is the single largest source of drawdown
        # on live (see Lessons Learned: "Same-market stacking" 2026-04-10).
        if market.ticker in traded_tickers:
            logger.debug(
                f"Skipping {market.ticker}: already have an unsettled trade "
                f"on this expiry."
            )
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
                traded_tickers.add(result.ticker)
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
