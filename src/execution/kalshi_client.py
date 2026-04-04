"""
Kalshi REST API v2 client with RSA-PSS authentication.

Authentication:
  - Kalshi uses RSA-PSS (SHA-256) signed request headers.
  - Each request is signed with: method + path + timestamp.
  - The API key ID is sent as a header; the signature proves ownership
    of the corresponding private key.

Endpoints used:
  GET  /markets                — list active BTC Up/Down markets
  GET  /markets/{ticker}/orderbook — order book for a market
  POST /orders                 — place a new order
  GET  /orders/{order_id}      — check order status
  DELETE /orders/{order_id}    — cancel an order
  GET  /portfolio/balance      — fetch available balance

Configuration (from .env):
  KALSHI_API_KEY_ID        — API key identifier (UUID)
  KALSHI_PRIVATE_KEY_PATH  — path to RSA private key PEM file
  KALSHI_ENV               — "demo" or "prod"
"""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    """
    Thread-safe Kalshi REST API v2 client.

    Handles RSA-PSS request signing, automatic 429 retry (once),
    and JSON response parsing.
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        env: Optional[str] = None,
    ):
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID", "")
        key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        env = env or os.getenv("KALSHI_ENV", "demo")

        # Orders go to demo (paper-safe) or prod; market data always from prod
        self.base_url  = DEMO_BASE if env == "demo" else PROD_BASE
        self.read_url  = PROD_BASE   # BTC markets only exist on prod
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # Load RSA private key
        pem = Path(key_path)
        if not pem.exists():
            logger.warning(
                f"Kalshi private key not found at '{key_path}'. "
                "Client will operate in mock/dry-run mode."
            )
            self._private_key = None
        else:
            with open(pem, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.info(f"Kalshi client initialised ({env.upper()}) key={self.api_key_id[:8]}...")

    # ─── Authentication ──────────────────────────────────────────────────────

    def _sign(self, method: str, path: str) -> Dict[str, str]:
        """Return signed auth headers for a request."""
        if self._private_key is None:
            return {}  # dry-run mode
        ts = str(int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    def _read_request(self, path: str, params: Optional[Dict] = None) -> Any:
        """Signed GET against prod (for market data — only exists on prod)."""
        url = self.read_url + path
        # Prod requires the full path including version prefix in the signature
        full_path = "/trade-api/v2" + path
        headers = self._sign("GET", full_path)
        for attempt in range(2):
            resp = self._session.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 429:
                logger.warning("Kalshi rate-limited (429) on read. Retrying in 1s...")
                # Use non-blocking sleep when inside an async event loop,
                # fall back to blocking sleep for synchronous callers.
                try:
                    import asyncio
                    loop = asyncio.get_running_loop()
                    # Can't await here (sync method), but at least yield
                    # control briefly via a short blocking sleep. The 1s
                    # pause is unavoidable for retries.
                except RuntimeError:
                    pass
                time.sleep(1.0)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Kalshi GET {path} failed after retry.")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Any:
        """Make a signed request against the configured env (demo or prod); retry once on 429."""
        url = self.base_url + path
        full_path = "/trade-api/v2" + path
        headers = self._sign(method, full_path)

        for attempt in range(2):
            resp = self._session.request(
                method, url, headers=headers, params=params, json=json, timeout=10
            )
            if resp.status_code == 429:
                logger.warning(f"Kalshi rate-limited (429) on {method} {path}. Retrying in 1s...")
                time.sleep(1.0)
                continue
            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"Kalshi {method} {path} failed after retry.")

    # ─── Market Data ─────────────────────────────────────────────────────────

    def get_btc_markets(self) -> List[Dict]:
        """
        Return active BTC prediction markets from prod, prioritising near-ATM markets.

        Paginates through the KXBTC series until it finds markets with YES price
        in the 0.35–0.65 band (at-the-money).  Falls back to all found markets if
        none are ATM (e.g. very directional market days).
        """
        ATM_LO, ATM_HI = 0.30, 0.70   # acceptable YES price window
        all_markets: List[Dict] = []
        cursor: Optional[str] = None
        pages = 0

        # Try 15-minute markets first (KXBTC15M), fall back to daily (KXBTC)
        for series in ("KXBTC15M", "KXBTC"):
            cursor = None
            pages  = 0
            series_markets: List[Dict] = []

            while pages < 10:
                params: Dict = {"status": "open", "series_ticker": series, "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = self._read_request("/markets", params=params)
                except Exception as exc:
                    logger.warning(f"Kalshi {series} fetch failed: {exc}")
                    break

                page = data.get("markets", [])
                if not page:
                    break

                series_markets.extend(page)
                pages += 1

                # Stop early once ATM markets found
                if any(ATM_LO <= float(m.get("yes_ask_dollars") or 0) <= ATM_HI for m in page):
                    break

                cursor = data.get("cursor")
                if not cursor:
                    break

            if series_markets:
                logger.debug(
                    f"Found {len(series_markets)} open BTC markets "
                    f"via series_ticker={series} ({pages} page(s))."
                )
                all_markets = series_markets
                break   # prefer 15-min; don't fall through to daily

        logger.debug(f"Using {len(all_markets)} BTC markets total.")
        return all_markets

    def get_orderbook(self, ticker: str, depth: int = 10) -> Dict:
        """
        Return parsed order book for a market ticker.

        Kalshi returns: {'orderbook_fp': {'yes_dollars': [[price, size], ...],
                                          'no_dollars':  [[price, size], ...]}}
        Prices are 0-1 scale strings.  We normalise to the legacy format:
            {'yes': {'bid': [float, ...], 'ask': [float, ...]},
             'no':  {'bid': [float, ...], 'ask': [float, ...]}}
        """
        raw = self._read_request(f"/markets/{ticker}/orderbook", params={"depth": depth})
        ob  = raw.get("orderbook_fp", {})

        # yes_dollars: sorted best-ask first (lowest price = cheapest YES)
        yes_prices = [float(row[0]) for row in ob.get("yes_dollars", [])]
        # no_dollars: sorted best-bid first (highest price = most expensive NO)
        no_prices  = [float(row[0]) for row in ob.get("no_dollars",  [])]

        return {
            "yes": {"ask": yes_prices, "bid": []},
            "no":  {"ask": [],         "bid": no_prices},
        }

    def get_balance(self) -> float:
        """Return available balance in USD (cents → dollars)."""
        data = self._request("GET", "/portfolio/balance")
        balance_cents = data.get("balance", 0)
        return balance_cents / 100.0

    # ─── Order Management ────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,           # "yes" or "no"
        count: int,          # number of contracts
        limit_price: int,    # cents (1–99)
        expiration_ts: Optional[int] = None,  # unix seconds; None = GTC
    ) -> Dict:
        """
        Place a limit order.

        Returns the Kalshi order object with fields:
            order_id, status, filled_count, remaining_count, etc.
        """
        if self._private_key is None:
            logger.warning(f"[DRY RUN] Would place: {side} {count}x {ticker} @ {limit_price}¢")
            return {
                "order_id": "DRY_RUN",
                "status": "resting",
                "filled_count": 0,
                "remaining_count": count,
            }

        body: Dict[str, Any] = {
            "ticker": ticker,
            "action": "buy",
            "side":   side,
            "count":  count,
            "type":   "limit",
            "yes_price" if side == "yes" else "no_price": limit_price,
        }
        if expiration_ts is not None:
            body["expiration_time"] = expiration_ts
            body["expiration_ts_type"] = "unix_ts"

        result = self._request("POST", "/orders", json=body)
        order = result.get("order", result)
        logger.info(
            f"Order placed: {order.get('order_id')} | {side} {count}x {ticker} "
            f"@ {limit_price}¢ | status={order.get('status')}"
        )
        return order

    def get_order(self, order_id: str) -> Dict:
        """Fetch current status of an order."""
        data = self._request("GET", f"/orders/{order_id}")
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel a resting order. Returns the cancelled order object."""
        if order_id == "DRY_RUN":
            logger.debug("[DRY RUN] Cancel skipped.")
            return {"order_id": "DRY_RUN", "status": "canceled"}
        data = self._request("DELETE", f"/orders/{order_id}")
        order = data.get("order", data)
        logger.info(f"Order cancelled: {order_id} | status={order.get('status')}")
        return order
