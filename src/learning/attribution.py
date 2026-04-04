"""
SHAP-based feature attribution for trade outcomes.

Every analysis cycle (every 50 settled trades):
  1. Pull last 200 trades with bar features
  2. Fit a lightweight XGBClassifier (outcome: WIN=1 / LOSS=0)
  3. Compute SHAP values via TreeExplainer
  4. Rank features by mean |SHAP| — high = alpha, low = noise
  5. Detect degrading alpha (>50% SHAP drop vs prior 3-cycle mean)

Results are returned as a list of FeatureReport dicts and also logged
to the console for visibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


SHAP_HISTORY_PATH = Path("data/shap_history.json")

FEATURE_COLS = [
    "log_return", "log_return_2", "log_return_5",
    "rolling_vol_15", "macd", "macd_hist", "rsi_14",
    "bb_width", "ema_delta", "volume_z",
    "ofi_norm", "aggressor_ratio", "spread_pct",
    "cnn_prob", "xgb_prob", "ensemble_prob",
    "p_yes", "net_edge", "market_yes_ask", "market_spread",
]


@dataclass
class FeatureReport:
    feature_name: str
    mean_shap: float
    alpha_rank: int
    trend_direction: str          # "improving" | "declining" | "stable"
    last_3_cycles_shap: List[float] = field(default_factory=list)
    degrading: bool = False       # True if SHAP dropped >50% vs 3-cycle mean


class AttributionAnalyzer:
    """
    Computes SHAP-based feature importance on the trade outcome dataset.

    Usage:
        analyzer = AttributionAnalyzer(journal)
        reports = analyzer.run()
    """

    def __init__(self, journal, shap_history_path: Path = SHAP_HISTORY_PATH):
        self._journal = journal
        self._history_path = shap_history_path
        self._history: Dict[str, List[float]] = self._load_history()

    def run(self, min_trades: int = 50) -> Optional[List[FeatureReport]]:
        """
        Run full attribution cycle.
        Returns None if not enough trades for meaningful analysis.
        """
        df = self._journal.get_feature_outcome_matrix(min_trades=min_trades)
        if df is None:
            logger.debug("Attribution: not enough settled trades yet.")
            return None

        try:
            import shap
            from xgboost import XGBClassifier as _XGB
        except ImportError:
            logger.warning("Attribution requires 'shap' and 'xgboost' packages.")
            return None

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y = df["y"].values

        # Fit a lightweight outcome-prediction model (not the price-direction model)
        clf = _XGB(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X, y)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Build ranked reports
        ranked = sorted(
            zip(feature_cols, mean_shap), key=lambda x: x[1], reverse=True
        )
        reports = []
        for rank, (fname, ms) in enumerate(ranked, start=1):
            history = self._history.get(fname, [])
            trend = self._compute_trend(history, ms)
            degrading = self._is_degrading(history, ms)
            report = FeatureReport(
                feature_name=fname,
                mean_shap=float(ms),
                alpha_rank=rank,
                trend_direction=trend,
                last_3_cycles_shap=history[-3:],
                degrading=degrading,
            )
            reports.append(report)

        # Update history
        for fname, ms in zip(feature_cols, mean_shap):
            if fname not in self._history:
                self._history[fname] = []
            self._history[fname].append(float(ms))
            self._history[fname] = self._history[fname][-10:]  # keep last 10 cycles

        self._save_history()
        self._log_report(reports)
        return reports

    def get_degrading_features(self, reports: List[FeatureReport]) -> List[str]:
        return [r.feature_name for r in reports if r.degrading]

    # ── Internal helpers ─────────────────────────────────────────────────

    def _compute_trend(self, history: List[float], current: float) -> str:
        if len(history) < 2:
            return "stable"
        prior = history[-1]
        if prior == 0:
            return "stable"
        pct_change = (current - prior) / prior
        if pct_change > 0.10:
            return "improving"
        if pct_change < -0.10:
            return "declining"
        return "stable"

    def _is_degrading(self, history: List[float], current: float) -> bool:
        """True if current SHAP is >50% below the 3-cycle mean."""
        if len(history) < 3:
            return False
        three_cycle_mean = np.mean(history[-3:])
        if three_cycle_mean == 0:
            return False
        return (three_cycle_mean - current) / three_cycle_mean > 0.50

    def _load_history(self) -> Dict[str, List[float]]:
        if self._history_path.exists():
            try:
                return json.loads(self._history_path.read_text())
            except Exception:
                pass
        return {}

    def _save_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.write_text(json.dumps(self._history, indent=2))

    def _log_report(self, reports: List[FeatureReport]) -> None:
        logger.info("── SHAP Attribution Report ──────────────────────────────")
        for r in reports[:10]:
            flag = " ⚠ DEGRADING" if r.degrading else ""
            logger.info(
                f"  #{r.alpha_rank:>2}  {r.feature_name:<22} "
                f"mean|SHAP|={r.mean_shap:.4f}  trend={r.trend_direction}{flag}"
            )
        logger.info("─────────────────────────────────────────────────────────")
