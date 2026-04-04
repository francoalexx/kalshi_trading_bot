"""
Export settled trades from SQLite to data/trade_history.csv.

Fixes from the original manual export:
  - Adds trade_time (bar timestamp) and settled_time as human-readable UTC
  - Uses b.net_edge instead of leaving signal_edge null
  - Includes all bar features available in the DB
  - Computes running_bankroll and won columns
  - Rerunnable: always regenerates from the full DB
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/trade_journal.db")
CSV_PATH = Path("data/trade_history.csv")
STARTING_BANKROLL = 500.0


def export():
    conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT
            t.trade_id,
            b.ts           AS trade_time,
            t.settled_ts   AS settled_time,
            t.expiry_ts,
            t.side,
            t.contracts,
            t.entry_price,
            t.signal_edge,
            t.kelly_fraction,
            t.slippage,
            t.outcome,
            t.pnl,
            t.fee,
            t.failure_class,
            b.btc_close,
            b.cnn_prob,
            b.xgb_prob,
            b.ensemble_prob,
            b.p_yes,
            b.net_edge     AS bar_net_edge,
            b.direction,
            b.regime,
            b.strike_type,
            b.strike_price,
            b.market_yes_ask,
            b.market_spread,
            b.seconds_to_expiry,
            b.log_return,
            b.log_return_2,
            b.log_return_5,
            b.rolling_vol_15,
            b.rsi_14,
            b.macd,
            b.macd_hist,
            b.bb_width,
            b.ema_delta,
            b.volume_z,
            b.ofi_norm,
            b.aggressor_ratio,
            b.spread_pct
        FROM trades t
        JOIN bars b ON t.bar_id = b.bar_id
        WHERE t.outcome IS NOT NULL
        ORDER BY t.trade_id ASC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert unix timestamps to human-readable UTC
    for col in ("trade_time", "settled_time", "expiry_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
            df[col] = df[col].where(df[col].notna(), None)

    # Compute derived columns
    df["won"] = (df["outcome"] == "WIN").astype(int)
    df["running_bankroll"] = STARTING_BANKROLL + df["pnl"].cumsum()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Exported {len(df)} trades to {CSV_PATH}")

    # Quick summary
    wins = df["won"].sum()
    losses = len(df) - wins
    wr = wins / len(df) if len(df) > 0 else 0
    final = df["running_bankroll"].iloc[-1] if len(df) > 0 else STARTING_BANKROLL
    print(f"  Record: {wins}W / {losses}L ({wr:.1%})")
    print(f"  Bankroll: ${STARTING_BANKROLL:.2f} -> ${final:.2f}")


if __name__ == "__main__":
    export()
