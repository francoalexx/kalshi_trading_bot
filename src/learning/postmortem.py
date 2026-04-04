"""
Post-Mortem Engine — classifies trade failures and extracts lessons.

Triggered every ANALYSIS_INTERVAL settled trades (default 50).

Failure categories:
  ALPHA_DECAY    — edge was real but small, eaten by fees/timing
  FALSE_POSITIVE — high confidence signal, wrong outcome
  MARKET_NOISE   — low volatility + low OFI, signal was random
  WIN            — profitable trade
  UNCLASSIFIED   — doesn't fit other categories
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", "50"))

# Thresholds for failure classification
VOL_NOISE_THRESHOLD = 0.003   # rolling_vol_15 below this = noisy bar
ALPHA_DECAY_EDGE    = 0.03    # signal_edge below this = thin edge
FP_ENSEMBLE_THRESH  = 0.58    # ensemble_prob above this = high confidence


def classify_failure(trade_row: pd.Series, bar_row: pd.Series) -> str:
    """
    Classify a single loss (or win) into a failure category.

    Parameters
    ----------
    trade_row : row from the trades table
    bar_row   : row from the bars table (joined on bar_id)
    """
    outcome = trade_row.get("outcome", "")

    if outcome == "WIN":
        return "WIN"

    if outcome != "LOSS":
        return "UNCLASSIFIED"

    signal_edge    = float(trade_row.get("signal_edge") or 0.0)
    ensemble_prob  = float(bar_row.get("ensemble_prob") or 0.5)
    rolling_vol_15 = float(bar_row.get("rolling_vol_15") or 0.005)
    ofi_norm       = float(bar_row.get("ofi_norm") or 0.0)

    # ALPHA_DECAY: entered with thin edge → fees consumed it
    if signal_edge < ALPHA_DECAY_EDGE:
        return "ALPHA_DECAY"

    # FALSE_POSITIVE: model was highly confident but wrong
    if ensemble_prob > FP_ENSEMBLE_THRESH:
        return "FALSE_POSITIVE"

    # MARKET_NOISE: low volatility and near-zero order flow imbalance
    if rolling_vol_15 < VOL_NOISE_THRESHOLD and abs(ofi_norm) < 0.5:
        return "MARKET_NOISE"

    return "UNCLASSIFIED"


class PostmortemEngine:
    """
    Runs post-mortem analysis every ANALYSIS_INTERVAL settled trades.

    Usage:
        pm = PostmortemEngine(journal, attribution_analyzer)
        # Call after each settlement:
        pm.on_settlement(trade_id, pnl, fee, outcome)
    """

    def __init__(self, journal, analyzer=None):
        self._journal  = journal
        self._analyzer = analyzer
        self._since_last_analysis = 0

    def on_settlement(
        self,
        trade_id: int,
        pnl: float,
        fee: float,
        outcome: str,
    ) -> None:
        """
        Call this after every settlement (after journal.log_settlement has already run).
        Adds failure classification and triggers full analysis every N trades.
        """
        self._since_last_analysis += 1

        # Classify failure and write failure_class back to the trade row
        try:
            df = self._journal.get_recent_trades(n=5)
            if not df.empty:
                row = df.iloc[0]
                failure_class = classify_failure(row, row)
                # Update just the failure_class (settlement already logged)
                self._journal._conn.execute(
                    "UPDATE trades SET failure_class=? WHERE trade_id=?",
                    (failure_class, trade_id),
                )
                self._journal._conn.commit()
        except Exception as exc:
            logger.debug(f"PostmortemEngine classification error: {exc}")

        if self._since_last_analysis >= ANALYSIS_INTERVAL:
            self._run_analysis()
            self._since_last_analysis = 0

    def _run_analysis(self) -> None:
        logger.info("PostmortemEngine: running full analysis cycle...")
        total = self._journal.count_settled()
        df = self._journal.get_recent_trades(n=200)

        if df.empty:
            return

        # Failure category breakdown
        if "failure_class" in df.columns:
            counts = df["failure_class"].value_counts()
            logger.info(f"Failure breakdown (last 200): {counts.to_dict()}")

        # Win rate trend
        if "outcome" in df.columns:
            settled = df[df["outcome"].isin(["WIN", "LOSS"])]
            if len(settled) >= 20:
                win_rate = (settled["outcome"] == "WIN").mean()
                recent_20 = settled.head(20)
                recent_wr = (recent_20["outcome"] == "WIN").mean()
                logger.info(
                    f"Win rate: {win_rate:.1%} overall | {recent_wr:.1%} last 20 trades"
                )

        # Attribution cycle
        if self._analyzer is not None:
            reports = self._analyzer.run()
            if reports:
                degrading = self._analyzer.get_degrading_features(reports)
                if degrading:
                    logger.warning(
                        f"Degrading alpha features detected: {degrading}. "
                        "Consider retraining or disabling in HeuristicsEngine."
                    )

        logger.info(f"PostmortemEngine: analysis complete ({total} total settled trades).")
