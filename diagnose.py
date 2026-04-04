"""
Diagnostic script — run this while the bot is stopped to identify why no trades fire.
Checks: Kalshi market discovery, model output distribution, and edge filter.
"""
import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── 1. Check Kalshi market discovery ─────────────────────────────────────────
print("\n═══ 1. KALSHI MARKET DISCOVERY (via signed prod client) ═══")
print("Checking PEM file:")
pem_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
with open(pem_path, "r") as f:
    pem_contents = f.read()
print(f"  First line : {pem_contents.splitlines()[0]}")
print(f"  Last line  : {pem_contents.splitlines()[-1]}")
print(f"  Total lines: {len(pem_contents.splitlines())}")

from src.execution.kalshi_client import KalshiClient
client = KalshiClient(env=os.getenv("KALSHI_ENV", "demo"))
print(f"  Private key loaded: {client._private_key is not None}")
try:
    from datetime import datetime, timezone
    markets = client.get_btc_markets()
    print(f"BTC markets found on PROD: {len(markets)}")
    now = time.time()

    def _parse_close(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    # Show markets nearest to 50¢ YES mid-price (at-the-money = most interesting for trading)
    tradeable = [
        m for m in markets
        if 90 < (_parse_close(m.get("close_time", "")) - now) < 108_000
        and m.get("status") == "active"
    ]
    def _yes_mid(m):
        ask = float(m.get("yes_ask_dollars") or 1.0)
        bid = float(m.get("yes_bid_dollars") or 0.0)
        return (ask + bid) / 2 if bid > 0 else ask / 2

    # Prefer liquid markets (YES bid > 0), fall back to illiquid if none
    liquid = [m for m in tradeable if float(m.get("yes_bid_dollars") or 0.0) > 0]
    pool   = liquid if liquid else tradeable
    pool.sort(key=lambda m: abs(_yes_mid(m) - 0.50))

    print(f"\nTradeable: {len(tradeable)} total, {len(liquid)} with YES bid > 0")
    print(f"Top markets (liquid-first, sorted by closeness to 50¢):")
    for m in pool[:10]:
        expiry  = _parse_close(m.get("close_time", ""))
        yes_ask = float(m.get("yes_ask_dollars") or 1.0)
        yes_bid = float(m.get("yes_bid_dollars") or 0.0)
        mid     = _yes_mid(m)
        st      = m.get("strike_type", "?")
        strike  = m.get("floor_strike") or m.get("cap_strike") or "?"
        print(f"  {m.get('ticker'):<45} YES {yes_bid:.2f}/{yes_ask:.2f} mid={mid:.2f} | {st} ${strike} | {expiry-now:.0f}s")

    if pool:
        best = pool[0]
        print(f"\nBest market (bot will trade): {best.get('ticker')}")
        print(f"  YES {float(best.get('yes_bid_dollars',0)):.2f}/{float(best.get('yes_ask_dollars',1)):.2f} "
              f"| strike_type={best.get('strike_type')} strike=${best.get('floor_strike') or best.get('cap_strike')}")
        ob = client.get_orderbook(best["ticker"])
        print(f"  Orderbook: {ob}")
except Exception as e:
    print(f"ERROR: {e}")

# ── 2. Check model output distribution ───────────────────────────────────────
print("\n═══ 2. MODEL OUTPUT DISTRIBUTION ═══")
try:
    import torch
    from src.models.cnn_lstm import load_cnn_lstm
    from src.models.xgboost_clf import XGBoostClassifier
    from src.features.indicators import FEATURE_NAMES

    cnn = load_cnn_lstm("models/artifacts/cnn_lstm_best.pt")
    xgb = XGBoostClassifier.load("models/artifacts/xgboost_best.pkl")

    # Generate random-ish inputs to sample output distribution
    cnn_probs, xgb_probs = [], []
    rng = np.random.default_rng(42)
    for _ in range(200):
        seq = rng.normal(0, 1, (1, 60, 13)).astype("float32")
        cnn_p = cnn.predict_proba(torch.tensor(seq))
        cnn_probs.append(cnn_p)

        feat = {k: float(rng.normal(0, 1)) for k in FEATURE_NAMES}
        xgb_p = xgb.predict_proba(feat)
        xgb_probs.append(xgb_p)

    cnn_probs = np.array(cnn_probs)
    xgb_probs = np.array(xgb_probs)

    print(f"CNN-LSTM: min={cnn_probs.min():.3f} max={cnn_probs.max():.3f} mean={cnn_probs.mean():.3f}")
    print(f"XGBoost:  min={xgb_probs.min():.3f} max={xgb_probs.max():.3f} mean={xgb_probs.mean():.3f}")
    # Use actual thresholds from alpha.py (CNN=0.50, XGB YES=0.54, XGB NO=0.46)
    CNN_YES = 0.50; CNN_NO = 0.50
    XGB_YES = 0.54; XGB_NO = 0.46
    print(f"CNN > {CNN_YES}: {(cnn_probs > CNN_YES).mean():.1%} of samples")
    print(f"CNN < {CNN_NO}: {(cnn_probs < CNN_NO).mean():.1%} of samples")
    print(f"XGB > {XGB_YES}: {(xgb_probs > XGB_YES).mean():.1%} of samples")
    print(f"XGB < {XGB_NO}: {(xgb_probs < XGB_NO).mean():.1%} of samples")
    both_yes = ((cnn_probs > CNN_YES) & (xgb_probs > XGB_YES)).mean()
    both_no  = ((cnn_probs < CNN_NO)  & (xgb_probs < XGB_NO)).mean()
    print(f"Both agree UP:   {both_yes:.1%} → BUY_NO  on B-markets, BUY_YES on T-markets")
    print(f"Both agree DOWN: {both_no:.1%}  → BUY_YES on B-markets, BUY_NO  on T-markets")
    print(f"Total signal rate: {both_yes + both_no:.1%}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n═══ DIAGNOSIS COMPLETE ═══\n")
