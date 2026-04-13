"""
Kalshi market feed — polls the REST API for active BTC Up/Down markets
and subscribes to their order books via WebSocket.

Responsibilities:
  - Discover the currently active BTC 15-min Up/Down market ticker and expiry
  - Maintain a live YES/NO bid-ask snapshot for that market
  - Enforce the T-90s deadline: flag when it is too late to enter

The Kalshi auth client is imported from execution.kalshi_client to avoid
duplicating RSA-PSS logic.
"""

import asyncio
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

# KalshiClient is imported at runtime to avoid circular imports at module load.
# It is injected via KalshiFeed.__init__(client=...).


@dataclass
class MarketState:
    ticker: str
    event_ticker: str
    expiry_ts: float          # unix epoch seconds
    yes_bid: float = 0.0
    yes_ask: float = 1.0
    no_bid: float = 0.0
    no_ask: float = 1.0
    strike_type: str = "greater"   # "greater" = BTC above X, "less" = BTC below X
    strike_price: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def mid_yes(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def seconds_to_expiry(self) -> float:
        return self.expiry_ts - time.time()

    @property
    def is_tradeable(self) -> bool:
        """True if there is more than 90 seconds until expiry."""
        return self.seconds_to_expiry > 90

    @property
    def spread_yes(self) -> float:
        return self.yes_ask - self.yes_bid


class KalshiFeed:
    """
    Polls Kalshi REST every 10s to track active BTC Up/Down markets and
    refresh their order book prices.

    Usage:
        feed = KalshiFeed(client=kalshi_client_instance)
        asyncio.create_task(feed.run())
        state = feed.get_active_market()
    """

    POLL_INTERVAL = 10  # seconds
    # Kalshi market series ticker for BTC 15-min up/down
    BTC_SERIES_TICKER = "BTCUSD"

    def __init__(self, client, scalp_mode: bool = False):
        self._client = client
        self._active: Optional[MarketState] = None
        self._all_markets: List[MarketState] = []
        self._running = False
        self._scalp_mode = scalp_mode
        if scalp_mode:
            self.POLL_INTERVAL = 3

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._refresh()
            except Exception as exc:
                logger.warning(f"KalshiFeed refresh error: {exc}")
            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self):
        self._running = False

    # Markets priced below this or above this have drifted too far from ATM.
    # The ML model's P(direction) is not a valid proxy for P(YES wins) on OTM
    # markets. Band tightened from [0.15, 0.85] → [0.30, 0.70] on 2026-04-10
    # after live data showed 21% win rate on entries below 0.30.
    OTM_REJECT_LOW  = 0.30
    OTM_REJECT_HIGH = 0.70

    def get_active_market(self) -> Optional[MarketState]:
        """
        Return the current active BTC market state, or None if none found.
        Returns None if the best available market has drifted outside the ATM
        band — waiting for the next 15-minute market to open is always better
        than trading an OTM contract. Checks yes_ask (the price the bot will
        actually pay) rather than mid so this agrees with the alpha-engine
        OTM guard, which uses market_price = yes_ask.
        """
        if self._active is None:
            return None
        ask = self._active.yes_ask
        if ask <= self.OTM_REJECT_LOW or ask >= self.OTM_REJECT_HIGH:
            logger.debug(
                f"KalshiFeed: market {self._active.ticker} yes_ask={ask:.3f} is OTM "
                f"(outside [{self.OTM_REJECT_LOW},{self.OTM_REJECT_HIGH}]) — withholding."
            )
            return None
        return self._active

    def get_all_markets(self) -> List[MarketState]:
        """Return all tracked markets (no OTM filtering). Used by scalp mode."""
        return list(self._all_markets)

    def get_market_by_ticker(self, ticker: str) -> Optional[MarketState]:
        """Look up a specific market by ticker. Used to monitor scalp positions."""
        for m in self._all_markets:
            if m.ticker == ticker:
                return m
        return None

    @staticmethod
    def _parse_close_time(close_time_str: str) -> float:
        """Parse ISO 8601 close_time string to Unix epoch seconds."""
        try:
            dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0

    async def _refresh(self):
        """
        Fetch active BTC markets from Kalshi and select the most price-sensitive one.

        Selects the market whose YES price (yes_ask_dollars) is closest to 0.50,
        meaning it is at-the-money and most sensitive to BTC price moves.
        Only considers markets expiring within the next 30 hours (excludes weeklies).
        """
        markets = await asyncio.get_event_loop().run_in_executor(
            None, self._client.get_btc_markets
        )
        now = time.time()

        def _expiry(m):
            return self._parse_close_time(m.get("close_time", ""))

        # Filter: tradeable window
        # For 15-min markets: expire within 16 minutes (one contract at a time)
        # For daily markets: allow up to 30 hours
        has_15m = any("15M" in m.get("ticker", "") for m in markets)
        max_window = 960 if has_15m else 108_000   # 16 min vs 30 hours
        candidates = [
            m for m in markets
            if 90 < (_expiry(m) - now) < max_window
            and m.get("status") == "active"
        ]

        if not candidates:
            logger.debug("No active BTC markets in trading window.")
            self._active = None
            self._all_markets = []
            return

        # Build MarketState for every candidate (used by scalp mode + price logger)
        self._all_markets = [self._build_market_state(m) for m in candidates]

        # Prefer markets with real two-sided liquidity (YES bid > 0).
        # Among liquid markets, pick the one with YES mid price closest to 50¢.
        # Fall back to illiquid markets (bid=0) only if no liquid ones exist.
        def _yes_mid(m):
            ask = float(m.get("yes_ask_dollars") or 1.0)
            bid = float(m.get("yes_bid_dollars") or 0.0)
            return (ask + bid) / 2 if bid > 0 else ask / 2

        def _atm_dist(m):
            return abs(_yes_mid(m) - 0.50)

        liquid = [m for m in candidates if float(m.get("yes_bid_dollars") or 0.0) > 0]
        pool   = liquid if liquid else candidates
        best   = min(pool, key=_atm_dist)
        state  = self._build_market_state(best)
        self._active = state
        logger.debug(
            f"Market {state.ticker} | YES {state.yes_bid:.2f}/{state.yes_ask:.2f} "
            f"| expires in {state.seconds_to_expiry:.0f}s"
        )

    def _build_market_state(self, m: dict) -> MarketState:
        """Parse a raw Kalshi market dict into a MarketState."""
        expiry       = self._parse_close_time(m.get("close_time", ""))
        yes_ask      = float(m.get("yes_ask_dollars") or 1.0)
        yes_bid      = float(m.get("yes_bid_dollars") or 0.0)
        no_ask_raw   = m.get("no_ask_dollars")
        no_ask       = float(no_ask_raw) if no_ask_raw else (1.0 - yes_bid if yes_bid > 0 else 1.0)
        no_bid_raw   = m.get("no_bid_dollars")
        no_bid       = float(no_bid_raw) if no_bid_raw else (1.0 - yes_ask if yes_ask < 1 else 0.0)
        strike_type  = m.get("strike_type", "greater")
        strike_price = float(m.get("floor_strike") or m.get("cap_strike") or 0.0)

        return MarketState(
            ticker=m.get("ticker", ""),
            event_ticker=m.get("event_ticker", ""),
            expiry_ts=expiry,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            strike_type=strike_type,
            strike_price=strike_price,
            last_updated=time.time(),
        )
