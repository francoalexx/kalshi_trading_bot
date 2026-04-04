"""
Shadow Model Manager — runs a parallel XGBoost with candidate hyperparameters.

The shadow model watches every bar alongside the live model but never places
real or paper orders. It logs hypothetical trade outcomes to evaluate whether
the candidate params outperform the live model before promotion.

Lifecycle:
  1. Candidate params arrive from Optuna (adaptation.py)
  2. Shadow model is retrained from scratch on journal data
  3. On every bar: shadow model produces a probability and logs what it would do
  4. After SHADOW_MIN_TRADES hypothetical outcomes, compare_performance() runs
  5. If promotion gate passes → return candidate params to caller for live promotion
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

SHADOW_MIN_TRADES = int(os.getenv("SHADOW_MIN_TRADES", "50"))

FEATURE_COLS = [
    "log_return", "log_return_2", "log_return_5",
    "rolling_vol_15", "macd", "macd_hist", "rsi_14",
    "bb_width", "ema_delta", "volume_z",
    "ofi_norm", "aggressor_ratio", "spread_pct",
]


class PromotionDecision(Enum):
    PROMOTE        = "PROMOTE"
    KEEP_LIVE      = "KEEP_LIVE"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class ShadowOutcome:
    ts:           float
    would_trade:  bool   # did the shadow model generate a signal above threshold?
    shadow_prob:  float
    live_prob:    float
    actual_win:   Optional[bool] = None   # filled in after settlement
    pnl:          float = 0.0


class ShadowModelManager:
    """
    Manages a shadow XGBoost model running in parallel with the live model.

    Parameters
    ----------
    live_model_path  : path to current live XGBoost .pkl
    candidate_params : dict of XGBoost params from Optuna
    xgb_yes_thresh   : candidate YES threshold (from Optuna)
    xgb_no_thresh    : candidate NO threshold (from Optuna)
    """

    def __init__(
        self,
        live_model_path: str,
        candidate_params: Optional[Dict] = None,
        xgb_yes_thresh: float = 0.52,
        xgb_no_thresh: float = 0.48,
    ):
        self._live_path       = live_model_path
        self._candidate_params = candidate_params or {}
        self._xgb_yes_thresh  = xgb_yes_thresh
        self._xgb_no_thresh   = xgb_no_thresh
        self._shadow_model    = None
        self._outcomes: List[ShadowOutcome] = []
        self._active = False

    def is_active(self) -> bool:
        return self._active and self._shadow_model is not None

    def train_shadow(self, journal) -> bool:
        """
        Retrain shadow model from scratch using journal data + candidate params.
        Returns True if training succeeded.
        """
        try:
            from xgboost import XGBClassifier as _XGB
        except ImportError:
            logger.warning("ShadowModel: xgboost not installed.")
            return False

        df = journal.get_feature_outcome_matrix(min_trades=SHADOW_MIN_TRADES)
        if df is None:
            logger.info("ShadowModel: not enough data to train shadow model.")
            return False

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y = df["y"].values

        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        params.update(self._candidate_params)

        clf = _XGB(**params)
        clf.fit(X, y)
        self._shadow_model = clf
        self._active = True
        self._outcomes = []
        logger.info(f"ShadowModel: trained on {len(X)} samples with params={self._candidate_params}")
        return True

    def evaluate_both(
        self, live_model, features: Dict[str, float]
    ) -> Tuple[float, Optional[float]]:
        """
        Run both live and shadow model on current features.
        Returns (live_prob, shadow_prob). shadow_prob is None if not active.
        """
        live_prob = live_model.predict_proba(features)

        if not self.is_active():
            return live_prob, None

        try:
            feature_cols = [c for c in FEATURE_COLS if c in features]
            x = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
            shadow_prob = float(self._shadow_model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.debug(f"ShadowModel inference error: {exc}")
            shadow_prob = None

        return live_prob, shadow_prob

    def record_bar(
        self,
        live_prob: float,
        shadow_prob: float,
        market_price: float,
    ) -> None:
        """Log what the shadow model would have done on this bar."""
        if not self.is_active():
            return

        would_trade = (
            shadow_prob > self._xgb_yes_thresh or
            shadow_prob < self._xgb_no_thresh
        )
        self._outcomes.append(ShadowOutcome(
            ts=time.time(),
            would_trade=would_trade,
            shadow_prob=shadow_prob,
            live_prob=live_prob,
        ))

    def record_shadow_outcome(self, actual_win: bool, pnl: float) -> None:
        """
        Mark the most recent tradeable shadow outcome as settled.
        Called from the settlement callback.
        """
        for outcome in reversed(self._outcomes):
            if outcome.would_trade and outcome.actual_win is None:
                outcome.actual_win = actual_win
                outcome.pnl = pnl
                break

    def compare_performance(self, live_outcomes: List[float]) -> PromotionDecision:
        """
        Compare shadow vs live performance and return a promotion decision.

        Parameters
        ----------
        live_outcomes : list of per-trade P&L for the live model over same period
        """
        shadow_settled = [
            o for o in self._outcomes
            if o.would_trade and o.actual_win is not None
        ]

        if len(shadow_settled) < SHADOW_MIN_TRADES:
            return PromotionDecision.NEEDS_MORE_DATA

        try:
            from scipy import stats
        except ImportError:
            logger.warning("ShadowModel: scipy required for promotion gate.")
            return PromotionDecision.NEEDS_MORE_DATA

        shadow_pnls = [o.pnl for o in shadow_settled]
        shadow_win_rate = np.mean([o.actual_win for o in shadow_settled])

        if len(live_outcomes) < 10:
            return PromotionDecision.NEEDS_MORE_DATA

        shadow_sharpe = _sharpe(shadow_pnls)
        live_sharpe   = _sharpe(live_outcomes)

        stat, p_value = stats.mannwhitneyu(
            shadow_pnls, live_outcomes, alternative="greater"
        )

        shadow_dd = _max_drawdown(shadow_pnls)
        live_dd   = _max_drawdown(live_outcomes)

        logger.info(
            f"ShadowModel: shadow_sharpe={shadow_sharpe:.2f} live_sharpe={live_sharpe:.2f} "
            f"p={p_value:.3f} shadow_win={shadow_win_rate:.1%} "
            f"n_shadow={len(shadow_pnls)} n_live={len(live_outcomes)}"
        )

        if (
            p_value < 0.05
            and shadow_sharpe > live_sharpe
            and shadow_dd < live_dd * 1.2
        ):
            logger.info("ShadowModel: PROMOTE — shadow model passes all gates.")
            return PromotionDecision.PROMOTE

        logger.info("ShadowModel: KEEP_LIVE — shadow model did not outperform.")
        return PromotionDecision.KEEP_LIVE


# ── Utility functions ─────────────────────────────────────────────────────────

def _sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def _max_drawdown(pnls: List[float]) -> float:
    arr = np.array(pnls)
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(drawdowns.max()) if len(drawdowns) else 0.0
