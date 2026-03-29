"""
Execution Controller — translates TradeSignals into Kalshi limit orders.

Responsibilities:
  1. Position sizing via 0.5× fractional Kelly Criterion
  2. Limit order placement (never market orders)
  3. Fill monitoring with 5-second polling
  4. Stale order cancellation at T-90 seconds before market expiry
  5. Slippage tracking and logging

Kelly Criterion (0.5× fractional):
    b     = (1 - p_market) / p_market   ← payout odds on YES
    f_full = (b × p_model - (1 - p_model)) / b
    f_half = max(0, 0.5 × f_full)
    position_usd = min(f_half × bankroll, MAX_POSITION_USD)

    Using 0.5× Kelly to reduce drawdown risk due to probability estimation
    uncertainty in volatile crypto markets.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.data.kalshi_feed import MarketState
from src.engine.alpha import Direction, TradeSignal
from src.execution.kalshi_client import KalshiClient
from src.risk.manager import RiskManager


MAX_POSITION_USD  = None   # set from env in Controller.__init__
FILL_POLL_SECONDS = 5
CANCEL_BEFORE_EXPIRY = 90  # cancel any unfilled order this many seconds before expiry


@dataclass
class OrderResult:
    order_id:       str
    ticker:         str
    side:           str     # "yes" or "no"
    contracts:      int
    limit_price:    float   # 0-1 scale
    filled:         int
    avg_fill_price: Optional[float]
    slippage:       float   # actual_fill - expected_fill
    cancelled:      bool
    outcome_ts:     Optional[float]  # unix seconds when market settles


class ExecutionController:
    """
    Manages the full lifecycle of a single trade on a prediction market.

    Usage:
        ctrl = ExecutionController(client, risk_manager, bankroll=500.0)
        result = await ctrl.execute(signal, market_state)
    """

    def __init__(
        self,
        client: KalshiClient,
        risk_manager: RiskManager,
        bankroll: float,
        max_position_usd: Optional[float] = None,
    ):
        self._client       = client
        self._risk         = risk_manager
        self._bankroll     = bankroll
        self._max_pos      = max_position_usd or (bankroll * 0.05)

    # ─── Public API ──────────────────────────────────────────────────────────

    async def execute(
        self,
        signal: TradeSignal,
        market: MarketState,
    ) -> Optional[OrderResult]:
        """
        Execute a trade based on a TradeSignal and current MarketState.

        Returns OrderResult if an order was placed, None if skipped by risk.
        """
        if signal.direction == Direction.NO_TRADE:
            return None

        # ── Risk checks ──────────────────────────────────────────────────
        if not self._risk.allow_trade():
            logger.warning("Trade blocked by risk manager.")
            return None

        if not market.is_tradeable:
            logger.warning(
                f"Market {market.ticker} not tradeable: "
                f"{market.seconds_to_expiry:.0f}s to expiry."
            )
            return None

        # ── Determine side and limit price ───────────────────────────────
        if signal.direction == Direction.BUY_YES:
            side        = "yes"
            limit_float = market.yes_ask   # pay the ask to get filled
        else:
            side        = "no"
            limit_float = market.no_ask

        limit_cents = int(round(limit_float * 100))
        limit_cents = max(1, min(99, limit_cents))   # Kalshi bounds

        # ── Position sizing ──────────────────────────────────────────────
        position_usd = self._kelly_size(
            p_model=signal.ensemble_prob,
            p_market=limit_float,
        )
        if position_usd < 1.0:
            logger.debug(f"Position too small (${position_usd:.2f}), skipping.")
            return None

        # Number of contracts: $1 per contract at limit price
        contracts = max(1, int(position_usd / limit_float))

        # ── Place order ──────────────────────────────────────────────────
        logger.info(
            f"Placing {side.upper()} {contracts} contracts on {market.ticker} "
            f"@ {limit_cents}¢ | kelly_size=${position_usd:.2f}"
        )
        self._risk.register_order_open()

        order = self._client.place_order(
            ticker=market.ticker,
            side=side,
            count=contracts,
            limit_price=limit_cents,
            expiration_ts=int(market.expiry_ts) - 5,  # expire 5s before market expiry
        )
        order_id = order.get("order_id", "UNKNOWN")

        # ── Monitor fill until T-90s deadline ────────────────────────────
        result = await self._monitor_fill(
            order_id=order_id,
            ticker=market.ticker,
            side=side,
            contracts=contracts,
            limit_float=limit_float,
            expiry_ts=market.expiry_ts,
        )

        self._risk.register_order_closed(result.filled, result.avg_fill_price or 0.0)
        return result

    # ─── Position Sizing ─────────────────────────────────────────────────────

    def _kelly_size(self, p_model: float, p_market: float) -> float:
        """
        Fractional Kelly position size in USD.

        b = net payout ratio (e.g., at 50¢ market price: b = 0.50/0.50 = 1.0)
        f* = (b × p_model - q) / b  where q = 1 - p_model
        size = 0.5 × f* × bankroll, capped at max_position_usd
        """
        if p_market <= 0 or p_market >= 1:
            return 0.0

        b = (1 - p_market) / p_market   # payout odds
        q = 1 - p_model
        f_full = (b * p_model - q) / b

        if f_full <= 0:
            return 0.0  # no edge

        f_half = 0.5 * f_full
        position = f_half * self._bankroll
        return min(position, self._max_pos)

    # ─── Fill Monitoring ─────────────────────────────────────────────────────

    async def _monitor_fill(
        self,
        order_id: str,
        ticker: str,
        side: str,
        contracts: int,
        limit_float: float,
        expiry_ts: float,
    ) -> OrderResult:
        """
        Poll order status every FILL_POLL_SECONDS.
        Cancel any remaining quantity at CANCEL_BEFORE_EXPIRY seconds before expiry.
        """
        filled = 0
        fill_prices = []
        cancelled = False

        while True:
            await asyncio.sleep(FILL_POLL_SECONDS)

            now = time.time()
            deadline = expiry_ts - CANCEL_BEFORE_EXPIRY

            if now >= deadline:
                # Cancel stale order before market expires
                try:
                    cancelled_order = self._client.cancel_order(order_id)
                    remaining = cancelled_order.get("remaining_count", 0)
                    filled_now = contracts - remaining
                    filled += filled_now
                    cancelled = True
                    logger.info(
                        f"Order {order_id} cancelled at T-{expiry_ts - now:.0f}s: "
                        f"filled={filled}/{contracts}"
                    )
                except Exception as exc:
                    logger.error(f"Cancel failed for {order_id}: {exc}")
                break

            # Check fill status
            try:
                order = self._client.get_order(order_id)
            except Exception as exc:
                logger.warning(f"Fill check failed for {order_id}: {exc}")
                continue

            status = order.get("status", "")
            new_fills = order.get("filled_count", 0) - filled
            if new_fills > 0:
                fill_price = order.get("yes_price" if side == "yes" else "no_price", 0) / 100
                fill_prices.extend([fill_price] * new_fills)
                filled = order.get("filled_count", 0)
                logger.debug(f"Fill update: {filled}/{contracts} @ {fill_price:.3f}")

            if status in ("filled", "canceled", "expired"):
                logger.info(
                    f"Order {order_id} terminal status={status} filled={filled}/{contracts}"
                )
                break

        avg_fill = sum(fill_prices) / len(fill_prices) if fill_prices else None
        slippage = (avg_fill - limit_float) if avg_fill is not None else 0.0

        if abs(slippage) > 0.01:
            logger.warning(f"High slippage detected: {slippage:.4f} on order {order_id}")

        return OrderResult(
            order_id=order_id,
            ticker=ticker,
            side=side,
            contracts=contracts,
            limit_price=limit_float,
            filled=filled,
            avg_fill_price=avg_fill,
            slippage=slippage,
            cancelled=cancelled,
            outcome_ts=None,  # populated by settlement monitor
        )

    def update_bankroll(self, new_bankroll: float) -> None:
        """Called by RiskManager after P&L updates."""
        self._bankroll = new_bankroll
        self._max_pos  = new_bankroll * 0.05
