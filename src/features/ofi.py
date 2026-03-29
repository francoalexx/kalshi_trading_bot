"""
Order Flow Imbalance (OFI) and VPIN computation.

OFI measures the net directional pressure in the order book:
    OFI = Σ_{i=1}^{10} (bid_volume_i - ask_volume_i)

Positive OFI → buy-side pressure → price likely to rise.
Negative OFI → sell-side pressure → price likely to fall.

Empirically explains 65-87% of short-term BTC price variance.

VPIN (Volume-Synchronized Probability of Informed Trading):
    VPIN_bucket = |OFI_bucket| / total_volume_in_bucket

High VPIN signals informed-trader activity and precedes price jumps.

Usage:
    ofi_calc = OFICalculator()

    # On each order book snapshot (100ms):
    ofi_calc.update(snapshot)

    # On each 1-minute bar close, collect the bar's metrics:
    metrics = ofi_calc.flush()
    # metrics.ofi_norm: normalised OFI for the model feature vector
    # metrics.vpin:     VPIN estimate
    # metrics.aggressor_ratio: fraction of volume that was buy-aggressor
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.data.binance_feed import OrderBookSnapshot


DEPTH_LEVELS = 10   # how many order book levels to aggregate
VPIN_WINDOW  = 50   # number of volume buckets for rolling VPIN


@dataclass
class OFIMetrics:
    ofi_raw:          float   # raw signed imbalance
    ofi_norm:         float   # normalised by rolling volume std
    vpin:             float   # VPIN estimate [0, 1]
    aggressor_ratio:  float   # buy aggressor fraction [0, 1]
    spread_pct:       float   # best bid-ask spread as % of mid
    bar_volume:       float   # total volume observed in this bar
    n_snapshots:      int     # how many book updates were seen


class OFICalculator:
    """
    Accumulates order book snapshots within a 1-minute bar and computes
    OFI and VPIN at bar close.

    Thread-safety note: feed these from a single asyncio task.
    """

    def __init__(self):
        # Accumulator for the current bar
        self._bar_ofi: List[float]    = []
        self._bar_volume: float       = 0.0
        self._bar_spreads: List[float] = []
        self._n_snapshots: int        = 0

        # Rolling history for normalisation
        self._ofi_history: Deque[float] = deque(maxlen=60)  # 60-bar rolling
        self._vpin_buckets: Deque[float] = deque(maxlen=VPIN_WINDOW)

        # Previous book snapshot for delta-OFI computation
        self._prev_bids: Optional[List[Tuple[float, float]]] = None
        self._prev_asks: Optional[List[Tuple[float, float]]] = None

        # Aggressor volume tracking (estimated from book changes)
        self._buy_volume:  float = 0.0
        self._sell_volume: float = 0.0

    def update(self, snapshot: OrderBookSnapshot) -> None:
        """Ingest a new order book snapshot (called ~every 100ms)."""
        bids = snapshot.bids[:DEPTH_LEVELS]
        asks = snapshot.asks[:DEPTH_LEVELS]

        # ── OFI: signed volume imbalance at top N levels ─────────────────
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        instantaneous_ofi = bid_vol - ask_vol
        self._bar_ofi.append(instantaneous_ofi)

        # ── Delta-OFI: changes in bid/ask quantities (more informative) ──
        if self._prev_bids is not None and self._prev_asks is not None:
            delta = self._compute_delta_ofi(bids, asks, self._prev_bids, self._prev_asks)
            self._buy_volume  += max(0,  delta)
            self._sell_volume += max(0, -delta)
            self._bar_volume  += abs(delta)

        self._prev_bids = bids
        self._prev_asks = asks

        # ── Spread ────────────────────────────────────────────────────────
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid = (best_bid + best_ask) / 2
            if mid > 0:
                self._bar_spreads.append((best_ask - best_bid) / mid)

        self._n_snapshots += 1

    def flush(self) -> OFIMetrics:
        """
        Called at the end of each 1-minute bar. Returns OFIMetrics for
        this bar and resets internal accumulators.
        """
        # ── Aggregate bar OFI ─────────────────────────────────────────────
        raw_ofi = float(np.sum(self._bar_ofi)) if self._bar_ofi else 0.0

        # Normalise by rolling std of bar-level OFI
        self._ofi_history.append(raw_ofi)
        if len(self._ofi_history) >= 5:
            ofi_std = float(np.std(self._ofi_history)) + 1e-9
        else:
            ofi_std = 1.0
        ofi_norm = raw_ofi / ofi_std

        # ── VPIN ─────────────────────────────────────────────────────────
        total_vol = self._buy_volume + self._sell_volume
        if total_vol > 0:
            vpin_bucket = abs(self._buy_volume - self._sell_volume) / total_vol
        else:
            vpin_bucket = 0.0
        self._vpin_buckets.append(vpin_bucket)
        vpin = float(np.mean(self._vpin_buckets)) if self._vpin_buckets else 0.0

        # ── Aggressor ratio ───────────────────────────────────────────────
        if total_vol > 0:
            aggressor_ratio = self._buy_volume / total_vol
        else:
            aggressor_ratio = 0.5

        # ── Spread ────────────────────────────────────────────────────────
        avg_spread = float(np.mean(self._bar_spreads)) if self._bar_spreads else 0.0

        metrics = OFIMetrics(
            ofi_raw=raw_ofi,
            ofi_norm=float(np.clip(ofi_norm, -5, 5)),   # clip outliers
            vpin=float(np.clip(vpin, 0, 1)),
            aggressor_ratio=float(np.clip(aggressor_ratio, 0, 1)),
            spread_pct=avg_spread,
            bar_volume=total_vol,
            n_snapshots=self._n_snapshots,
        )

        # Reset for next bar
        self._bar_ofi.clear()
        self._bar_spreads.clear()
        self._buy_volume   = 0.0
        self._sell_volume  = 0.0
        self._bar_volume   = 0.0
        self._n_snapshots  = 0
        self._prev_bids    = None
        self._prev_asks    = None

        return metrics

    @staticmethod
    def _compute_delta_ofi(
        bids:      List[Tuple[float, float]],
        asks:      List[Tuple[float, float]],
        prev_bids: List[Tuple[float, float]],
        prev_asks: List[Tuple[float, float]],
    ) -> float:
        """
        Compute change in net order book pressure between two snapshots.

        Positive → more buy pressure added (or sell pressure removed).
        Negative → more sell pressure added (or buy pressure removed).
        """
        def vol_at_level(levels, idx):
            return levels[idx][1] if idx < len(levels) else 0.0

        n = min(DEPTH_LEVELS, len(bids), len(asks), len(prev_bids), len(prev_asks))
        delta = 0.0
        for i in range(n):
            # Bid changes: increase = more buy pressure
            delta += vol_at_level(bids, i) - vol_at_level(prev_bids, i)
            # Ask changes: increase = more sell pressure
            delta -= vol_at_level(asks, i) - vol_at_level(prev_asks, i)
        return delta
