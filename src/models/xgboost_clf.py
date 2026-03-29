"""
XGBoost binary classifier for BTC price direction prediction.

Uses tabular features (technical indicators + OFI) rather than the raw
sequence consumed by CNN-LSTM. Acts as an independent second opinion in
the ensemble, particularly strong at 5-minute interval predictions
(83% accuracy, F1=82%, per 2024 benchmark studies).

Features (must match FEATURE_NAMES in indicators.py):
    log_return, log_return_2, log_return_5, rolling_vol_15,
    macd, macd_hist, rsi_14, bb_width, ema_delta, volume_z,
    aggressor_ratio, ofi_norm, spread_pct
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

from src.features.indicators import FEATURE_NAMES


class XGBoostClassifier:
    """
    Thin wrapper around xgb.XGBClassifier with a consistent interface
    matching the CNN-LSTM model (predict_proba → scalar float).
    """

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    def __init__(self, params: Optional[Dict] = None):
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        # early_stopping_rounds lives in the constructor across all XGBoost versions
        p.setdefault("early_stopping_rounds", 30)
        self._model = xgb.XGBClassifier(**p)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary labels (1 = up, 0 = down)
        """
        fit_kwargs = {"verbose": False}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Return P(BTC_up) for a single feature dict.

        Parameters
        ----------
        features : dict mapping feature name → float value
                   Keys must match FEATURE_NAMES exactly.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        x = np.array([[features.get(k, 0.0) for k in FEATURE_NAMES]], dtype="float32")
        proba = self._model.predict_proba(x)[0, 1]
        return float(proba)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Return P(up) for a (n_samples, n_features) array."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)[:, 1]

    def feature_importances(self) -> Dict[str, float]:
        """Return a dict of feature name → importance score."""
        if not self._fitted:
            return {}
        scores = self._model.feature_importances_
        return dict(zip(FEATURE_NAMES, scores.tolist()))

    def save(self, path: str) -> None:
        """Persist model to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "feature_names": FEATURE_NAMES}, f)

    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        """Load a saved model from a pickle file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        with open(p, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._model = data["model"]
        obj._fitted = True
        return obj
