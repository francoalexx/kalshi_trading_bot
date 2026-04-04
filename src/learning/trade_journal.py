"""
Trade Journal — SQLite-backed log of every bar, every trade, and every settlement.

Three tables:
  bars         — one row per 1-minute bar close (with features + signal + market context)
  trades       — one row per executed trade (execution quality + outcome after settlement)
  heuristics_log — audit trail of rule changes from the learning system

The journal is the single source of truth for the post-mortem engine, SHAP
attribution, shadow model evaluation, and Optuna hyperparameter tuning.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger


DB_PATH = Path("data/trade_journal.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS bars (
        bar_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts               REAL NOT NULL,
        btc_close        REAL,
        log_return       REAL,
        log_return_2     REAL,
        log_return_5     REAL,
        rolling_vol_15   REAL,
        macd             REAL,
        macd_hist        REAL,
        rsi_14           REAL,
        bb_width         REAL,
        ema_delta        REAL,
        volume_z         REAL,
        ofi_norm         REAL,
        aggressor_ratio  REAL,
        spread_pct       REAL,
        cnn_prob         REAL,
        xgb_prob         REAL,
        ensemble_prob    REAL,
        p_yes            REAL,
        net_edge         REAL,
        direction        TEXT,
        regime           TEXT,
        market_ticker    TEXT,
        strike_type      TEXT,
        strike_price     REAL,
        market_yes_ask   REAL,
        market_spread    REAL,
        seconds_to_expiry REAL,
        traded           INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS trades (
        trade_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        bar_id           INTEGER REFERENCES bars(bar_id),
        ticker           TEXT NOT NULL,
        side             TEXT NOT NULL,
        contracts        INTEGER,
        entry_price      REAL,
        kelly_fraction   REAL,
        signal_edge      REAL,
        slippage         REAL,
        time_to_fill_s   REAL,
        cancelled        INTEGER DEFAULT 0,
        outcome          TEXT,
        pnl              REAL,
        fee              REAL,
        failure_class    TEXT,
        expiry_ts        REAL,
        settled_ts       REAL
    );

    CREATE TABLE IF NOT EXISTS heuristics_log (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        ts               REAL,
        rule_name        TEXT,
        action           TEXT,
        old_value        TEXT,
        new_value        TEXT,
        n_trades_basis   INTEGER,
        p_value          REAL
    );
    """)
    conn.commit()


class TradeJournal:
    """
    Logs bars, trades, and settlements to SQLite.

    Thread-safe via a single connection (asyncio is single-threaded; all
    calls happen in the same event loop thread).
    """

    def __init__(self, db_path: Path = DB_PATH):
        global DB_PATH
        DB_PATH = db_path
        self._conn = _get_conn()
        _init_db(self._conn)
        logger.info(f"TradeJournal initialised at {db_path}")

    # ── Write methods ────────────────────────────────────────────────────

    def log_bar(
        self,
        features: Dict[str, float],
        signal,           # TradeSignal
        market,           # MarketState
        btc_close: float,
        traded: bool = False,
    ) -> int:
        """Insert one bar row; return its bar_id."""
        p_yes = signal.ensemble_prob
        if hasattr(market, "strike_type") and market.strike_type in ("less", "less_or_equal"):
            p_yes = 1.0 - signal.ensemble_prob

        cur = self._conn.execute(
            """
            INSERT INTO bars (
                ts, btc_close,
                log_return, log_return_2, log_return_5,
                rolling_vol_15, macd, macd_hist, rsi_14,
                bb_width, ema_delta, volume_z,
                ofi_norm, aggressor_ratio, spread_pct,
                cnn_prob, xgb_prob, ensemble_prob, p_yes, net_edge,
                direction, regime,
                market_ticker, strike_type, strike_price,
                market_yes_ask, market_spread, seconds_to_expiry,
                traded
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (
                time.time(), btc_close,
                features.get("log_return"),
                features.get("log_return_2"),
                features.get("log_return_5"),
                features.get("rolling_vol_15"),
                features.get("macd"),
                features.get("macd_hist"),
                features.get("rsi_14"),
                features.get("bb_width"),
                features.get("ema_delta"),
                features.get("volume_z"),
                features.get("ofi_norm"),
                features.get("aggressor_ratio"),
                features.get("spread_pct"),
                signal.cnn_lstm_prob,
                signal.xgb_prob,
                signal.ensemble_prob,
                p_yes,
                signal.net_edge,
                signal.direction.value,
                signal.regime,
                getattr(market, "ticker", None),
                getattr(market, "strike_type", None),
                getattr(market, "strike_price", None),
                getattr(market, "yes_ask", None),
                getattr(market, "spread_yes", None),
                getattr(market, "seconds_to_expiry", None),
                int(traded),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_trade(
        self,
        bar_id: int,
        result,           # ExecutionResult from controller
        kelly_fraction: float = 0.0,
    ) -> int:
        """Insert one trade row; return trade_id."""
        cur = self._conn.execute(
            """
            INSERT INTO trades (
                bar_id, ticker, side, contracts, entry_price,
                kelly_fraction, signal_edge, slippage,
                cancelled, expiry_ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                bar_id,
                getattr(result, "ticker", None),
                getattr(result, "side", None),
                getattr(result, "filled", 0),
                getattr(result, "avg_fill_price", None),
                kelly_fraction,
                getattr(result, "signal_edge", None),
                getattr(result, "slippage", 0.0),
                int(getattr(result, "filled", 0) == 0),
                getattr(result, "expiry_ts", None),
            ),
        )
        self._conn.commit()
        # Mark the bar as having generated a trade
        self._conn.execute("UPDATE bars SET traded=1 WHERE bar_id=?", (bar_id,))
        self._conn.commit()
        return cur.lastrowid

    def log_settlement(
        self,
        trade_id: int,
        pnl: float,
        fee: float,
        outcome: str,          # "WIN" / "LOSS" / "CANCELLED"
        failure_class: Optional[str] = None,
    ) -> None:
        """Update a trade row with its settlement outcome."""
        self._conn.execute(
            """
            UPDATE trades
            SET outcome=?, pnl=?, fee=?, failure_class=?, settled_ts=?
            WHERE trade_id=?
            """,
            (outcome, pnl, fee, failure_class, time.time(), trade_id),
        )
        self._conn.commit()

    def log_heuristic_change(
        self,
        rule_name: str,
        action: str,
        old_value: str,
        new_value: str,
        n_trades_basis: int,
        p_value: float,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO heuristics_log
              (ts, rule_name, action, old_value, new_value, n_trades_basis, p_value)
            VALUES (?,?,?,?,?,?,?)
            """,
            (time.time(), rule_name, action, old_value, new_value, n_trades_basis, p_value),
        )
        self._conn.commit()

    # ── Read methods ─────────────────────────────────────────────────────

    def get_recent_trades(self, n: int = 100) -> pd.DataFrame:
        """Return last N settled trades joined with their bar features."""
        query = """
            SELECT
                t.trade_id, t.bar_id, t.ticker, t.side, t.contracts,
                t.entry_price, t.kelly_fraction, t.signal_edge,
                t.slippage, t.outcome, t.pnl, t.fee, t.failure_class,
                t.expiry_ts, t.settled_ts,
                b.ts, b.btc_close,
                b.log_return, b.log_return_2, b.log_return_5,
                b.rolling_vol_15, b.macd, b.macd_hist, b.rsi_14,
                b.bb_width, b.ema_delta, b.volume_z,
                b.ofi_norm, b.aggressor_ratio, b.spread_pct,
                b.cnn_prob, b.xgb_prob, b.ensemble_prob,
                b.p_yes, b.net_edge, b.direction, b.regime,
                b.strike_type, b.market_yes_ask, b.market_spread
            FROM trades t
            JOIN bars b ON t.bar_id = b.bar_id
            WHERE t.outcome IS NOT NULL
            ORDER BY t.trade_id DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self._conn, params=(n,))

    def get_feature_outcome_matrix(self, min_trades: int = 50) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of bar features with binary WIN label.
        Returns None if fewer than min_trades settled trades available.
        """
        df = self.get_recent_trades(n=10_000)
        settled = df[df["outcome"].isin(["WIN", "LOSS"])]
        if len(settled) < min_trades:
            return None
        settled = settled.copy()
        settled["y"] = (settled["outcome"] == "WIN").astype(int)
        feature_cols = [
            "log_return", "log_return_2", "log_return_5",
            "rolling_vol_15", "macd", "macd_hist", "rsi_14",
            "bb_width", "ema_delta", "volume_z",
            "ofi_norm", "aggressor_ratio", "spread_pct",
            "cnn_prob", "xgb_prob", "ensemble_prob",
            "p_yes", "net_edge", "market_yes_ask", "market_spread",
        ]
        cols = [c for c in feature_cols if c in settled.columns]
        return settled[cols + ["y"]].dropna()

    def count_settled(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL"
        ).fetchone()
        return row[0] if row else 0
