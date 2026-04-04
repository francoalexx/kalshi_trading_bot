"""
Strategy Adaptation — Optuna hyperparameter tuning + promotion gate orchestration.

Two responsibilities:
  1. run_optuna_study()   — weekly background thread that finds better XGBoost params
  2. PromotionOrchestrator — wires shadow model + promotion gate + revert logic

Promotion requires:
  - n_shadow_trades >= SHADOW_MIN_TRADES (default 50)
  - Mann-Whitney p < 0.05 (shadow better than live)
  - shadow_sharpe > live_sharpe
  - shadow_max_drawdown < live_max_drawdown × 1.2

After promotion:
  - 20-trade probation period
  - Auto-revert if rolling Sharpe drops >30% below pre-promotion baseline
  - Model checkpoints: xgboost_best.pkl, xgboost_prev.pkl, xgboost_prev2.pkl
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from src.learning.shadow_model import PromotionDecision, ShadowModelManager

SHADOW_MIN_TRADES    = int(os.getenv("SHADOW_MIN_TRADES", "50"))
PROBATION_TRADES     = int(os.getenv("PROBATION_TRADES", "20"))
REVERT_SHARPE_DROP   = 0.30   # auto-revert if Sharpe drops >30% post-promotion
OPTUNA_INTERVAL_DAYS = 7      # run Optuna every 7 days

MODEL_DIR = Path("models/artifacts")


class PromotionOrchestrator:
    """
    Monitors shadow model performance and promotes or reverts models.

    Usage:
        orch = PromotionOrchestrator(shadow_mgr, journal)
        # Call after each live trade settles:
        orch.on_live_settlement(pnl)
    """

    def __init__(self, shadow_manager: ShadowModelManager, journal):
        self._shadow     = shadow_manager
        self._journal    = journal
        self._live_pnls: List[float] = []
        self._probation_pnls: List[float] = []
        self._in_probation  = False
        self._pre_promo_sharpe = 0.0
        self._promoted_params: Optional[Dict] = None

    def on_live_settlement(self, pnl: float) -> None:
        """Call after every live/paper trade settles."""
        self._live_pnls.append(pnl)
        # Keep last 200 for comparison window
        self._live_pnls = self._live_pnls[-200:]

        if self._in_probation:
            self._probation_pnls.append(pnl)
            self._check_regression()
            return

        if not self._shadow.is_active():
            return

        decision = self._shadow.compare_performance(self._live_pnls)

        if decision == PromotionDecision.PROMOTE:
            self._promote()
        elif decision == PromotionDecision.KEEP_LIVE:
            logger.info("Promotion gate: KEEP_LIVE — shadow model not promoted.")

    def _promote(self) -> None:
        """Swap live model with shadow model and enter probation."""
        logger.info("Promoting shadow model to live...")

        live_path = MODEL_DIR / "xgboost_best.pkl"
        prev_path = MODEL_DIR / "xgboost_prev.pkl"
        prev2_path = MODEL_DIR / "xgboost_prev2.pkl"

        # Rotate checkpoints
        if prev_path.exists():
            shutil.copy2(prev_path, prev2_path)
        if live_path.exists():
            shutil.copy2(live_path, prev_path)

        # Save shadow model as new live
        try:
            import pickle
            with open(live_path, "wb") as f:
                pickle.dump(self._shadow._shadow_model, f)
            logger.success(f"Shadow model saved to {live_path}")
        except Exception as exc:
            logger.error(f"Failed to save promoted model: {exc}")
            return

        self._pre_promo_sharpe = _sharpe(self._live_pnls[-50:]) if len(self._live_pnls) >= 10 else 0.0
        self._in_probation     = True
        self._probation_pnls   = []
        self._promoted_params  = self._shadow._candidate_params

        self._journal.log_heuristic_change(
            rule_name="model_promotion",
            action="PROMOTED",
            old_value="xgboost_prev.pkl",
            new_value="xgboost_best.pkl",
            n_trades_basis=len(self._live_pnls),
            p_value=0.05,  # already passed gate
        )

    def _check_regression(self) -> None:
        """Auto-revert if post-promotion Sharpe drops too much."""
        if len(self._probation_pnls) < PROBATION_TRADES:
            return

        post_sharpe = _sharpe(self._probation_pnls)
        threshold   = self._pre_promo_sharpe * (1.0 - REVERT_SHARPE_DROP)

        if post_sharpe < threshold:
            logger.warning(
                f"Promotion regression detected: post_sharpe={post_sharpe:.2f} < "
                f"threshold={threshold:.2f}. Reverting."
            )
            self._revert()
        else:
            logger.info(
                f"Probation passed: post_sharpe={post_sharpe:.2f} >= {threshold:.2f}. "
                "Promoted model is now stable."
            )
            self._in_probation = False

    def _revert(self) -> None:
        live_path = MODEL_DIR / "xgboost_best.pkl"
        prev_path = MODEL_DIR / "xgboost_prev.pkl"

        if prev_path.exists():
            shutil.copy2(prev_path, live_path)
            logger.success("Reverted to previous model.")
        else:
            logger.error("No previous model checkpoint to revert to.")

        self._in_probation = False
        self._journal.log_heuristic_change(
            rule_name="model_promotion",
            action="REGRESSION_REVERT",
            old_value=str(self._promoted_params),
            new_value="xgboost_prev.pkl",
            n_trades_basis=len(self._probation_pnls),
            p_value=0.0,
        )


def run_optuna_study(journal, n_trials: int = 50) -> Optional[Dict]:
    """
    Use Optuna to find better XGBoost hyperparameters.

    Objective: maximize Sharpe ratio on out-of-sample journal data
    using time-series cross-validation.

    Returns best_params dict or None if optimization failed.
    """
    try:
        import optuna
        from xgboost import XGBClassifier as _XGB
    except ImportError:
        logger.warning("Optuna study requires 'optuna' and 'xgboost' packages.")
        return None

    df = journal.get_feature_outcome_matrix(min_trades=100)
    if df is None:
        logger.info("Optuna: not enough data for hyperparameter tuning.")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feature_cols = [
        c for c in [
            "log_return", "log_return_2", "log_return_5",
            "rolling_vol_15", "macd", "macd_hist", "rsi_14",
            "bb_width", "ema_delta", "volume_z",
            "ofi_norm", "aggressor_ratio", "spread_pct",
        ] if c in df.columns
    ]
    X = df[feature_cols].fillna(0).values
    y = df["y"].values

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        return _ts_cv_sharpe(X, y, params)

    Path("data").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///data/optuna.db",
        study_name="xgb_tuning",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Optuna study complete. Best params: {best} (value={study.best_value:.3f})")
    return best


def schedule_weekly_optuna(journal, shadow_manager: ShadowModelManager) -> None:
    """
    Start a background thread that runs Optuna every OPTUNA_INTERVAL_DAYS days,
    then trains a new shadow model with the best params.
    """
    def _worker():
        while True:
            time.sleep(OPTUNA_INTERVAL_DAYS * 86400)
            logger.info("Weekly Optuna study starting...")
            best_params = run_optuna_study(journal)
            if best_params:
                shadow_manager._candidate_params = best_params
                shadow_manager.train_shadow(journal)

    t = threading.Thread(target=_worker, daemon=True, name="optuna_scheduler")
    t.start()
    logger.info(f"Weekly Optuna scheduler started (interval={OPTUNA_INTERVAL_DAYS}d).")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ts_cv_sharpe(X, y, params, n_splits: int = 5) -> float:
    """Time-series cross-validation Sharpe ratio (with 10-bar embargo)."""
    from xgboost import XGBClassifier as _XGB

    n = len(X)
    fold_size = n // (n_splits + 1)
    embargo   = 10
    pnls = []

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_start = train_end + embargo
        test_end   = min(test_start + fold_size, n)

        if test_start >= n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        clf = _XGB(**params)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        # Simple P&L simulation: trade if prob > 0.52 or < 0.48
        for prob, actual in zip(probs, y_te):
            if prob > 0.52:
                pnls.append(1.0 if actual == 1 else -1.0)
            elif prob < 0.48:
                pnls.append(1.0 if actual == 0 else -1.0)

    return _sharpe(pnls) if pnls else 0.0


def _sharpe(pnls: List[float]) -> float:
    arr = np.array(pnls)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))
