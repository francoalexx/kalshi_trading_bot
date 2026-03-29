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

Paper mode (--paper flag): all steps run but no real orders are placed.
  Set KALSHI_PRIVATE_KEY_PATH to a non-existent path to auto-enter dry-run.

Usage:
    python -m src.main --paper                     # paper trading mode
    python -m src.main --cnn models/artifacts/cnn_lstm_best.pt \
                       --xgb models/artifacts/xgboost_best.pkl
"""

import argparse
import asyncio
import os
import signal as _signal
import sys
from pathlib import Path

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

load_dotenv()

# ─── Logging setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/bot_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="00:00")


async def run_bot(
    cnn_model_path: str,
    xgb_model_path: str,
    paper_mode: bool = False,
):
    bankroll        = float(os.getenv("INITIAL_BANKROLL", "500"))
    max_open        = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    daily_loss_pct  = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.10"))
    max_tph         = int(os.getenv("MAX_TRADES_PER_HOUR", "8"))
    min_edge        = float(os.getenv("MIN_NET_EDGE", "0.02"))

    # ── Component initialisation ─────────────────────────────────────────
    risk = RiskManager(
        initial_bankroll=bankroll,
        max_open_positions=max_open,
        daily_loss_limit_pct=daily_loss_pct,
        max_trades_per_hour=max_tph,
    )

    kalshi = KalshiClient(env="demo" if paper_mode else os.getenv("KALSHI_ENV", "demo"))

    kalshi_feed = KalshiFeed(client=kalshi)
    binance_feed = BinanceFeed()
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
    )

    paper = PaperTracker(starting_bankroll=bankroll) if paper_mode else None

    # ── SIGINT handler → graceful kill switch ────────────────────────────
    def _handle_sigint(sig, frame):
        logger.warning("SIGINT received. Triggering kill switch.")
        risk.trigger_kill_switch("SIGINT")

    _signal.signal(_signal.SIGINT, _handle_sigint)

    logger.info(f"Bot starting — paper={paper_mode} bankroll=${bankroll:.2f}")

    # ── Start background tasks ───────────────────────────────────────────
    asyncio.create_task(binance_feed.run(),   name="binance_feed")
    asyncio.create_task(kalshi_feed.run(),    name="kalshi_feed")

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

        # ── Kalshi market state ──────────────────────────────────────
        market = kalshi_feed.get_active_market()
        if market is None:
            logger.debug("No active Kalshi market. Skipping bar.")
            continue

        # ── Alpha evaluation ─────────────────────────────────────────
        signal = alpha.evaluate(
            sequence=sequence,
            features=features,
            market_price=market.yes_ask,
            spread=market.spread_yes,
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

            # ── Paper P&L tracking ───────────────────────────────────
            if paper is not None and result.filled > 0:
                entry = result.avg_fill_price or market.yes_ask
                paper_trade = paper.record_trade(
                    ticker=result.ticker,
                    side=result.side,
                    contracts=result.filled,
                    entry_price=entry,
                    expiry_ts=market.expiry_ts,
                    signal_edge=signal.net_edge,
                )
                # Schedule settlement check after market expiry
                async def _settle(pt=paper_trade, mkt=market):
                    wait = max(0, mkt.expiry_ts - __import__('time').time() + 5)
                    await asyncio.sleep(wait)
                    # Fetch final BTC price direction from Kalshi market result
                    try:
                        settled_mkt = kalshi._request("GET", f"/markets/{pt.ticker}")
                        result_val = settled_mkt.get("market", {}).get("result", "")
                        btc_went_up = result_val.lower() == "yes"
                        paper.settle_trade(pt, btc_went_up=btc_went_up)
                    except Exception as exc:
                        logger.warning(f"Could not fetch settlement for {pt.ticker}: {exc}")
                asyncio.create_task(_settle())


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
