"""
Unit tests for src/features/indicators.py

Validates:
  - Technical indicator values match expected ranges
  - Lookahead prevention (shift(1) applied correctly)
  - Feature vector completeness
  - Sequence tensor shape
"""

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    FEATURE_NAMES,
    build_sequence_tensor,
    compute_indicators,
    extract_feature_vector,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    price = 50_000 + np.cumsum(rng.normal(0, 100, n))
    price = np.abs(price)
    opens  = price
    closes = price + rng.normal(0, 50, n)
    highs  = np.maximum(opens, closes) + rng.uniform(0, 100, n)
    lows   = np.minimum(opens, closes) - rng.uniform(0, 100, n)
    vols   = rng.uniform(10, 100, n)
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": vols,
    })


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestComputeIndicators:
    def test_output_has_expected_columns(self):
        df = make_ohlcv()
        result = compute_indicators(df)
        expected_cols = [
            "log_return", "log_return_2", "log_return_5", "rolling_vol_15",
            "macd", "macd_hist", "rsi_14", "bb_width", "ema_delta", "volume_z",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_normalised_range(self):
        df = make_ohlcv(200)
        result = compute_indicators(df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= -1.0 - 1e-6, "RSI below -1"
        assert rsi.max() <= 1.0 + 1e-6, "RSI above 1"

    def test_rolling_vol_is_positive(self):
        df = make_ohlcv(200)
        result = compute_indicators(df)
        vol = result["rolling_vol_15"].dropna()
        assert (vol >= 0).all(), "Rolling vol must be non-negative"

    def test_lookahead_prevention_via_shift(self):
        """
        The shifted indicator at row i must equal the un-shifted indicator at row i-1.
        We verify this by comparing a manually shifted series with compute_indicators output.
        """
        df = make_ohlcv(200)
        result = compute_indicators(df)
        # log_return at row 10 should equal raw log_return at row 9 (before shift)
        raw_log_return = np.log(df["close"] / df["close"].shift(1))
        # After shift(1), result["log_return"][i] == raw_log_return[i-1]
        for i in range(5, 50):
            expected = raw_log_return.iloc[i - 1]
            actual   = result["log_return"].iloc[i]
            assert abs(actual - expected) < 1e-10, (
                f"Lookahead detected at row {i}: expected {expected:.6f}, got {actual:.6f}"
            )

    def test_bb_width_positive(self):
        df = make_ohlcv(200)
        result = compute_indicators(df)
        bb = result["bb_width"].dropna()
        assert (bb > 0).all(), "Bollinger Band width must be positive"


class TestExtractFeatureVector:
    def test_returns_all_feature_names(self):
        df = make_ohlcv(120)
        df = compute_indicators(df)
        fv = extract_feature_vector(df)
        for name in FEATURE_NAMES:
            assert name in fv, f"Feature missing from vector: {name}"

    def test_all_values_are_finite(self):
        df = make_ohlcv(120)
        df = compute_indicators(df)
        fv = extract_feature_vector(df)
        for name, val in fv.items():
            assert np.isfinite(val), f"Non-finite value for feature {name}: {val}"

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError):
            extract_feature_vector(pd.DataFrame())


class TestBuildSequenceTensor:
    def test_correct_shape(self):
        df = make_ohlcv(120)
        df = compute_indicators(df)
        seq = build_sequence_tensor(df, seq_len=60)
        assert seq is not None
        assert seq.shape[0] == 60
        assert seq.shape[1] > 0

    def test_returns_none_if_not_enough_rows(self):
        df = make_ohlcv(30)
        df = compute_indicators(df)
        seq = build_sequence_tensor(df, seq_len=60)
        assert seq is None

    def test_normalised_mean_approx_zero(self):
        df = make_ohlcv(120)
        df = compute_indicators(df)
        seq = build_sequence_tensor(df, seq_len=60)
        assert seq is not None
        col_means = seq.mean(axis=0)
        # After z-score normalisation, column means should be near 0
        assert np.allclose(col_means, 0, atol=0.5), (
            f"Sequence column means not near 0: {col_means}"
        )

    def test_dtype_float32(self):
        df = make_ohlcv(120)
        df = compute_indicators(df)
        seq = build_sequence_tensor(df, seq_len=60)
        assert seq.dtype == np.float32
