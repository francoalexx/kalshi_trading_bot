"""
Technical indicator computation for the Alpha Engine.

All indicators are computed on a pandas DataFrame with columns:
    open, high, low, close, volume

Key design rules to avoid lookahead bias:
  - All indicator values are .shift(1) before being used as model features,
    so the model only sees indicator values that were knowable at bar close-1.
  - The feature vector returned always corresponds to the PREVIOUS closed bar.

Uses the `ta` library (pip install ta) — Python 3.10+ compatible.
"""

from typing import Dict

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility


# ─── Feature names (must match training) ─────────────────────────────────────
FEATURE_NAMES = [
    "log_return",
    "log_return_2",
    "log_return_5",
    "rolling_vol_15",
    "macd",
    "macd_hist",
    "rsi_14",
    "bb_width",
    "ema_delta",
    "volume_z",
    "aggressor_ratio",   # filled externally from OFI module
    "ofi_norm",          # filled externally from OFI module
    "spread_pct",        # filled externally from book snapshot
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        Rows are sorted oldest→newest.

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with indicator columns.
        All indicator columns are shifted by 1 to prevent lookahead.
    """
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    # ── Log returns ──────────────────────────────────────────────────────
    df["log_return"]   = np.log(close / close.shift(1))
    df["log_return_2"] = np.log(close / close.shift(2))
    df["log_return_5"] = np.log(close / close.shift(5))

    # ── Rolling volatility (15-bar std of log returns) ───────────────────
    df["rolling_vol_15"] = df["log_return"].rolling(15).std()

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]      = macd_indicator.macd()
    df["macd_hist"] = macd_indicator.macd_diff()

    # ── RSI (14) — normalised to [-1, 1] ─────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_14"] = (rsi - 50) / 50

    # ── Bollinger Band Width ──────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    upper  = bb.bollinger_hband()
    lower  = bb.bollinger_lband()
    middle = bb.bollinger_mavg()
    df["bb_width"] = (upper - lower) / middle.replace(0, np.nan)

    # ── EMA delta: EMA(9) - EMA(21), normalised by close ─────────────────
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema_delta"] = (ema9 - ema21) / close.replace(0, np.nan)

    # ── Volume z-score (30-bar rolling) ──────────────────────────────────
    vol_mean = volume.rolling(30).mean()
    vol_std  = volume.rolling(30).std()
    df["volume_z"] = (volume - vol_mean) / (vol_std + 1e-9)

    # ── Shift all indicator columns by 1 to prevent lookahead ────────────
    indicator_cols = [
        "log_return", "log_return_2", "log_return_5", "rolling_vol_15",
        "macd", "macd_hist", "rsi_14", "bb_width", "ema_delta", "volume_z",
    ]
    for col in indicator_cols:
        df[col] = df[col].shift(1)

    return df


def extract_feature_vector(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract the feature vector for the most recent row.

    Columns not computed here (aggressor_ratio, ofi_norm, spread_pct)
    must be injected by the caller after this function returns.

    Returns a dict mapping feature name → float value.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    row = df.iloc[-1]
    return {
        "log_return":    float(row.get("log_return", 0.0)    or 0.0),
        "log_return_2":  float(row.get("log_return_2", 0.0)  or 0.0),
        "log_return_5":  float(row.get("log_return_5", 0.0)  or 0.0),
        "rolling_vol_15": float(row.get("rolling_vol_15", 0.0) or 0.0),
        "macd":          float(row.get("macd", 0.0)          or 0.0),
        "macd_hist":     float(row.get("macd_hist", 0.0)     or 0.0),
        "rsi_14":        float(row.get("rsi_14", 0.0)        or 0.0),
        "bb_width":      float(row.get("bb_width", 0.0)      or 0.0),
        "ema_delta":     float(row.get("ema_delta", 0.0)     or 0.0),
        "volume_z":      float(row.get("volume_z", 0.0)      or 0.0),
        # Externally injected — defaults to 0 until overwritten
        "aggressor_ratio": 0.0,
        "ofi_norm":        0.0,
        "spread_pct":      0.0,
    }


def build_sequence_tensor(df: pd.DataFrame, seq_len: int = 60):
    """
    Build the (seq_len, n_features) numpy array consumed by the CNN-LSTM model.

    Uses OHLCV columns normalised by rolling z-score over the sequence window,
    plus the computed indicator columns.

    Returns None if the DataFrame has fewer than seq_len rows.
    """
    sequence_cols = [
        "open", "high", "low", "close", "volume",
        "log_return", "rolling_vol_15", "macd", "macd_hist",
        "rsi_14", "bb_width", "ema_delta", "volume_z",
    ]
    if len(df) < seq_len:
        return None

    window = df[sequence_cols].iloc[-seq_len:].copy()

    # Z-score normalise each column over the window
    for col in sequence_cols:
        mu  = window[col].mean()
        std = window[col].std() + 1e-9
        window[col] = (window[col] - mu) / std

    window.fillna(0.0, inplace=True)
    return window.values.astype("float32")   # shape: (seq_len, n_features)
