"""
Risk Manager — enforces all position limits, daily loss limits, and the kill switch.

Controls:
  ┌────────────────────────────┬──────────────────────────────────────────────┐
  │ Kill Switch                │ Thread-safe Event; halts ALL new orders       │
  │ Max Position per Trade     │ 5% of current bankroll (enforced in Kelly)    │
  │ Max Open Positions         │ 3 simultaneous Kalshi contracts               │
  │ Daily Loss Limit           │ 10% of start-of-day bankroll                 │
  │ Max Trades per Hour        │ 8 (prevents over-trading in choppy regimes)  │
  │ Minimum Net Edge           │ 2% after fees + spread (enforced in Alpha)   │
  └────────────────────────────┴──────────────────────────────────────────────┘

The kill switch is triggered automatically when:
  - Daily loss limit is breached
  - An unhandled exception propagates to the main loop
  - The user sends SIGINT (Ctrl+C)

When the kill switch is active, the bot logs all events but places no orders.
Prediction market positions settle automatically — no manual close needed.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from loguru import logger


@dataclass
class DailyStats:
    start_bankroll:  float
    realized_pnl:    float = 0.0
    trades_today:    int   = 0
    wins:            int   = 0
    losses:          int   = 0
    gross_fees_paid: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def daily_loss_pct(self) -> float:
        if self.start_bankroll == 0:
            return 0.0
        return -self.realized_pnl / self.start_bankroll


class RiskManager:
    """
    Central risk enforcement module. All trading decisions pass through here.

    Thread-safety: methods are protected by a reentrant lock.
    """

    def __init__(
        self,
        initial_bankroll: float,
        max_open_positions: int = 3,
        daily_loss_limit_pct: float = 0.10,
        max_trades_per_hour: int = 8,
    ):
        self._bankroll             = initial_bankroll
        self._max_open             = max_open_positions
        self._daily_loss_limit_pct = daily_loss_limit_pct
        self._max_trades_per_hour  = max_trades_per_hour

        self._open_positions: int = 0
        self._lock = threading.RLock()

        # Kill switch — set this Event to halt the bot immediately
        self.kill_switch = threading.Event()

        # Trade timestamp ring buffer for rate-limiting
        self._trade_times: Deque[float] = deque(maxlen=max_trades_per_hour)

        # Daily stats (reset at UTC midnight)
        self._daily: DailyStats = DailyStats(start_bankroll=initial_bankroll)
        self._day_start: float  = self._today_start_ts()

    # ─── Core Guard ──────────────────────────────────────────────────────────

    def allow_trade(self) -> bool:
        """
        Returns True if a new trade is permitted.
        Checks all risk controls in order.
        """
        with self._lock:
            self._maybe_reset_daily()

            if self.kill_switch.is_set():
                logger.warning("KILL SWITCH ACTIVE — trade blocked.")
                return False

            if self._open_positions >= self._max_open:
                logger.debug(
                    f"Max open positions reached ({self._open_positions}/{self._max_open})."
                )
                return False

            if self._is_rate_limited():
                logger.debug("Rate limit: max trades per hour reached.")
                return False

            if self._daily.daily_loss_pct >= self._daily_loss_limit_pct:
                logger.critical(
                    f"Daily loss limit breached: {self._daily.daily_loss_pct:.2%} "
                    f">= {self._daily_loss_limit_pct:.2%}. Triggering kill switch."
                )
                self._trigger_kill_switch("Daily loss limit breached.")
                return False

            return True

    # ─── Order lifecycle hooks ───────────────────────────────────────────────

    def register_order_open(self) -> None:
        """Call when an order has been placed (not yet filled)."""
        with self._lock:
            self._open_positions += 1
            self._trade_times.append(time.time())
            logger.debug(f"Open positions: {self._open_positions}")

    def register_order_closed(self, filled_contracts: int, avg_fill_price: float) -> None:
        """
        Call when an order is fully settled (filled or cancelled with no fill).
        At this point we don't yet know the P&L — call record_settlement() when
        the market settles.
        """
        with self._lock:
            self._open_positions = max(0, self._open_positions - 1)
            if filled_contracts > 0:
                self._daily.trades_today += 1
            logger.debug(f"Open positions: {self._open_positions}")

    def record_settlement(
        self,
        contracts_won: int,
        contracts_lost: int,
        gross_pnl: float,
        fees_paid: float,
    ) -> None:
        """
        Record the outcome of a settled prediction market position.

        Parameters
        ----------
        contracts_won  : number of contracts that paid out $1
        contracts_lost : number of contracts that paid out $0
        gross_pnl      : net cash change (positive = profit, negative = loss)
        fees_paid      : total platform fees for this trade
        """
        with self._lock:
            self._daily.realized_pnl    += gross_pnl
            self._daily.gross_fees_paid += fees_paid
            self._daily.wins            += (1 if gross_pnl > 0 else 0)
            self._daily.losses          += (1 if gross_pnl <= 0 else 0)
            self._bankroll              += gross_pnl

            logger.info(
                f"Settlement: pnl={gross_pnl:+.2f} | fees={fees_paid:.2f} | "
                f"bankroll=${self._bankroll:.2f} | daily_pnl={self._daily.realized_pnl:+.2f} "
                f"({self._daily.daily_loss_pct:.2%} of day start)"
            )

            # Re-check daily loss after settlement
            if self._daily.daily_loss_pct >= self._daily_loss_limit_pct:
                self._trigger_kill_switch(
                    f"Daily loss limit hit post-settlement: "
                    f"{self._daily.daily_loss_pct:.2%}"
                )

    # ─── Kill switch ─────────────────────────────────────────────────────────

    def _trigger_kill_switch(self, reason: str) -> None:
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
        self.kill_switch.set()

    def trigger_kill_switch(self, reason: str = "Manual trigger") -> None:
        """Public API for external kill switch activation (e.g., SIGINT handler)."""
        self._trigger_kill_switch(reason)

    def reset_kill_switch(self) -> None:
        """Reset kill switch (e.g., after daily reset or manual override)."""
        self.kill_switch.clear()
        logger.info("Kill switch reset.")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _is_rate_limited(self) -> bool:
        """True if we have placed >= max_trades_per_hour in the last 60 minutes."""
        now = time.time()
        cutoff = now - 3600
        recent = [t for t in self._trade_times if t > cutoff]
        return len(recent) >= self._max_trades_per_hour

    @staticmethod
    def _today_start_ts() -> float:
        """Unix timestamp of today's UTC midnight."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def _maybe_reset_daily(self) -> None:
        """Reset daily stats at UTC midnight."""
        if time.time() >= self._day_start + 86400:
            logger.info(
                f"Daily reset. Yesterday: pnl={self._daily.realized_pnl:+.2f} "
                f"win_rate={self._daily.win_rate:.2%} trades={self._daily.trades_today}"
            )
            self._daily = DailyStats(start_bankroll=self._bankroll)
            self._day_start = self._today_start_ts()
            # Do NOT auto-reset kill switch on daily reset — require manual reset
            if self.kill_switch.is_set():
                logger.warning(
                    "Kill switch remains active after daily reset. "
                    "Call reset_kill_switch() to resume trading."
                )

    @property
    def bankroll(self) -> float:
        with self._lock:
            return self._bankroll

    @property
    def daily_stats(self) -> DailyStats:
        with self._lock:
            return self._daily
