"""
Shadow Model Manager — runs a parallel XGBoost with candidate hyperparameters.

The shadow model watches every bar alongside the live model but never places
real or paper orders. It logs hypothetical trade observations to SQLite (via
TradeJournal) so that every observation — and its eventual outcome — survives
bot restarts.

Lifecycle:
  1. Candidate params arrive from Optuna (adaptation.py) OR are hand-supplied
  2. Shadow model is trained:
     - train_from_parquet(): bootstrap from historical btc_1m.parquet
     - train_shadow():       retrain from existing journal trade data
  3. On every bar: shadow model produces a probability, decides whether it
     would trade, and logs the observation row via the journal
  4. On settlement: journal.settle_shadow_observations(ticker, result) fills
     in shadow_win and shadow_pnl for every observation tied to that market
  5. After enough settled observations, compare_performance() runs the
     Mann-Whitney promotion gate
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

SHADOW_MIN_TRADES = int(os.getenv("SHADOW_MIN_TRADES", "50"))

# Tabular features used by the shadow XGBoost. Must match the features
# consumed by the live model (see src/models/xgboost_clf.py FEATURE_NAMES).
FEATURE_COLS = [
    "log_return", "log_return_2", "log_return_5", "rolling_vol_15",
    "macd", "macd_hist", "rsi_14", "bb_width", "ema_delta", "volume_z",
]

# OTM guard — identical to alpha.py. Shadow should not pretend to trade
# markets the live bot would have skipped outright.
OTM_LOW  = 0.30
OTM_HIGH = 0.70


class PromotionDecision(Enum):
    PROMOTE         = "PROMOTE"
    KEEP_LIVE       = "KEEP_LIVE"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class ShadowDecision:
    """What the shadow model would do on a single bar."""
    shadow_prob:  float    # P(BTC up) from shadow model
    direction:    str      # "BUY_YES" | "BUY_NO" | "NO_TRADE"
    would_trade:  bool


class ShadowModelManager:
    """
    Manages a shadow XGBoost model running in parallel with the live model.

    Parameters
    ----------
    live_model_path  : path to current live XGBoost .pkl (reference only)
    candidate_params : dict of XGBoost params to try (from Optuna or hand-tuned)
    xgb_yes_thresh   : candidate YES threshold
    xgb_no_thresh    : candidate NO threshold
    journal          : TradeJournal instance — required for persistence
    """

    def __init__(
        self,
        live_model_path: str,
        candidate_params: Optional[Dict] = None,
        xgb_yes_thresh: float = 0.56,
        xgb_no_thresh:  float = 0.44,
        journal=None,
    ):
        self._live_path        = live_model_path
        self._candidate_params = candidate_params or {}
        self._xgb_yes_thresh   = xgb_yes_thresh
        self._xgb_no_thresh    = xgb_no_thresh
        self._shadow_model     = None
        self._journal          = journal
        self._active           = False

    def is_active(self) -> bool:
        return self._active and self._shadow_model is not None

    # ── Training paths ───────────────────────────────────────────────────

    def train_from_parquet(self, parquet_path: str) -> bool:
        """
        Bootstrap the shadow model by training on the historical OHLCV parquet
        (same data the live model was trained on) with candidate_params.

        This is the preferred startup path when there is not yet enough journal
        data for train_shadow() to be meaningful.
        """
        try:
            from xgboost import XGBClassifier as _XGB
            import pandas as pd
        except ImportError:
            logger.warning("ShadowModel: xgboost / pandas not installed.")
            return False

        path = Path(parquet_path)
        if not path.exists():
            logger.warning(f"ShadowModel: parquet not found at {path}")
            return False

        try:
            from src.models.train import load_and_label, train_val_split
            df = load_and_label(str(path))
            train_df, _ = train_val_split(df)
        except Exception as exc:
            logger.warning(f"ShadowModel: parquet preparation failed: {exc}")
            return False

        available = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[available].fillna(0.0).values.astype("float32")
        y = train_df["label"].values.astype(int)

        params = {
            "n_estimators":    300,
            "max_depth":       6,
            "learning_rate":   0.03,
            "subsample":       0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "eval_metric":     "logloss",
            "verbosity":       0,
            "n_jobs":          -1,
            "tree_method":     "hist",
        }
        params.update(self._candidate_params)

        try:
            clf = _XGB(**params)
            clf.fit(X, y)
        except Exception as exc:
            logger.warning(f"ShadowModel: training failed: {exc}")
            return False

        self._shadow_model = clf
        self._active = True
        logger.info(
            f"ShadowModel: trained from parquet on {len(X):,} samples "
            f"with params={self._candidate_params or '(defaults)'}, "
            f"thresholds=({self._xgb_no_thresh}, {self._xgb_yes_thresh})"
        )
        return True

    def train_shadow(self, journal) -> bool:
        """
        Retrain shadow model from journal data + candidate params.
        Falls back to returning False if journal has fewer than
        SHADOW_MIN_TRADES settled trades.
        """
        try:
            from xgboost import XGBClassifier as _XGB
        except ImportError:
            logger.warning("ShadowModel: xgboost not installed.")
            return False

        df = journal.get_feature_outcome_matrix(min_trades=SHADOW_MIN_TRADES)
        if df is None:
            logger.info("ShadowModel: not enough journal data to train shadow.")
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
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        params.update(self._candidate_params)

        clf = _XGB(**params)
        clf.fit(X, y)
        self._shadow_model = clf
        self._active = True
        logger.info(
            f"ShadowModel: trained on {len(X)} journal samples "
            f"with params={self._candidate_params}"
        )
        return True

    # ── Inference + observation logging ──────────────────────────────────

    def predict_decision(
        self,
        features: Dict[str, float],
        market_price: float,
        strike_type: str,
    ) -> ShadowDecision:
        """
        Run the shadow model on current features and produce a decision that
        mirrors alpha.evaluate()'s logic (OTM guard, bearish inversion,
        direction thresholds).
        """
        # OTM guard — shadow refuses to pretend to trade OTM markets.
        if market_price <= OTM_LOW or market_price >= OTM_HIGH:
            return ShadowDecision(shadow_prob=0.5, direction="NO_TRADE", would_trade=False)

        try:
            x = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]], dtype="float32")
            shadow_prob = float(self._shadow_model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.debug(f"ShadowModel inference error: {exc}")
            return ShadowDecision(shadow_prob=0.5, direction="NO_TRADE", would_trade=False)

        bearish = strike_type in ("less", "less_or_equal")

        if shadow_prob > self._xgb_yes_thresh:
            direction = "BUY_NO" if bearish else "BUY_YES"
        elif shadow_prob < self._xgb_no_thresh:
            direction = "BUY_YES" if bearish else "BUY_NO"
        else:
            direction = "NO_TRADE"

        return ShadowDecision(
            shadow_prob=shadow_prob,
            direction=direction,
            would_trade=(direction != "NO_TRADE"),
        )

    def record_bar(
        self,
        bar_id: int,
        market,
        features: Dict[str, float],
        live_prob: float,
        live_direction: str,
    ) -> Optional[ShadowDecision]:
        """
        Compute the shadow decision for this bar and persist it to the journal.
        Returns the ShadowDecision, or None if the shadow is inactive.
        """
        if not self.is_active() or self._journal is None:
            return None

        decision = self.predict_decision(
            features=features,
            market_price=getattr(market, "yes_ask", 0.5),
            strike_type=getattr(market, "strike_type", "greater"),
        )

        try:
            self._journal.log_shadow_observation(
                bar_id=bar_id,
                market_ticker=getattr(market, "ticker", ""),
                expiry_ts=getattr(market, "expiry_ts", 0.0),
                market_yes_ask=getattr(market, "yes_ask", 0.5),
                market_spread=getattr(market, "spread_yes", 0.0),
                strike_type=getattr(market, "strike_type", "greater"),
                live_prob=live_prob,
                shadow_prob=decision.shadow_prob,
                live_direction=live_direction,
                shadow_direction=decision.direction,
                would_trade_live=(live_direction != "NO_TRADE"),
                would_trade_shadow=decision.would_trade,
            )
        except Exception as exc:
            logger.debug(f"ShadowModel: failed to log observation: {exc}")

        return decision

    def record_settlement(self, ticker: str, market_result: str) -> int:
        """
        Called from the settlement monitor when a market's YES/NO result is
        known. Updates every observation for that ticker with shadow_win
        and shadow_pnl. Returns the number of observations updated.
        """
        if self._journal is None:
            return 0
        try:
            return self._journal.settle_shadow_observations(ticker, market_result)
        except Exception as exc:
            logger.debug(f"ShadowModel: failed to settle {ticker}: {exc}")
            return 0

    # ── Performance comparison ───────────────────────────────────────────

    def compare_performance(self, live_outcomes: List[float]) -> PromotionDecision:
        """
        Compare shadow vs live performance from persisted journal observations.
        """
        if self._journal is None:
            return PromotionDecision.NEEDS_MORE_DATA

        try:
            from scipy import stats
            import pandas as pd
        except ImportError:
            logger.warning("ShadowModel: scipy required for promotion gate.")
            return PromotionDecision.NEEDS_MORE_DATA

        conn = self._journal._conn
        df = pd.read_sql_query(
            """
            SELECT shadow_pnl, shadow_win
            FROM shadow_observations
            WHERE would_trade_shadow = 1 AND settled = 1
            """,
            conn,
        )

        if len(df) < SHADOW_MIN_TRADES:
            return PromotionDecision.NEEDS_MORE_DATA
        if len(live_outcomes) < 10:
            return PromotionDecision.NEEDS_MORE_DATA

        shadow_pnls = df["shadow_pnl"].tolist()
        shadow_win_rate = float(df["shadow_win"].mean())

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
