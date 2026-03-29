"""
Unit tests for src/features/ofi.py

Validates:
  - OFI sign reflects correct buy/sell pressure
  - VPIN range is [0, 1]
  - Aggressor ratio range is [0, 1]
  - flush() correctly resets internal state
  - Delta-OFI responds correctly to order book changes
"""

import time
from typing import List, Tuple

import pytest

from src.data.binance_feed import OrderBookSnapshot
from src.features.ofi import OFICalculator, OFIMetrics


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_snapshot(
    bid_qty: float = 10.0,
    ask_qty: float = 10.0,
    bid_price: float = 50000.0,
    ask_price: float = 50001.0,
    n_levels: int = 10,
) -> OrderBookSnapshot:
    """Create a uniform order book snapshot for testing."""
    bids = [(bid_price - i * 0.5, bid_qty) for i in range(n_levels)]
    asks = [(ask_price + i * 0.5, ask_qty) for i in range(n_levels)]
    return OrderBookSnapshot(timestamp=time.time(), bids=bids, asks=asks)


def feed_snapshots(calc: OFICalculator, snapshots: List[OrderBookSnapshot]) -> None:
    for snap in snapshots:
        calc.update(snap)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestOFISign:
    def test_positive_ofi_when_bid_heavy(self):
        """Heavy bids → positive OFI → expected price increase."""
        calc = OFICalculator()
        snaps = [make_snapshot(bid_qty=50.0, ask_qty=5.0) for _ in range(10)]
        feed_snapshots(calc, snaps)
        metrics = calc.flush()
        assert metrics.ofi_raw > 0, "Expected positive OFI with heavy bid side"

    def test_negative_ofi_when_ask_heavy(self):
        """Heavy asks → negative OFI → expected price decrease."""
        calc = OFICalculator()
        snaps = [make_snapshot(bid_qty=5.0, ask_qty=50.0) for _ in range(10)]
        feed_snapshots(calc, snaps)
        metrics = calc.flush()
        assert metrics.ofi_raw < 0, "Expected negative OFI with heavy ask side"

    def test_near_zero_ofi_when_balanced(self):
        """Balanced book → OFI near zero."""
        calc = OFICalculator()
        snaps = [make_snapshot(bid_qty=10.0, ask_qty=10.0) for _ in range(10)]
        feed_snapshots(calc, snaps)
        metrics = calc.flush()
        assert abs(metrics.ofi_raw) < 1e-6, "Expected OFI ≈ 0 for balanced book"


class TestVPIN:
    def test_vpin_range(self):
        calc = OFICalculator()
        # Feed directional book updates (bid-heavy) to generate volume
        for _ in range(20):
            calc.update(make_snapshot(bid_qty=30.0, ask_qty=5.0))
            calc.update(make_snapshot(bid_qty=5.0,  ask_qty=30.0))
        metrics = calc.flush()
        assert 0.0 <= metrics.vpin <= 1.0, f"VPIN out of range: {metrics.vpin}"

    def test_vpin_accumulates_across_bars(self):
        """VPIN should be non-zero after multiple bars of directional activity."""
        calc = OFICalculator()
        # Simulate 5 bars of one-directional flow
        for _ in range(5):
            for _ in range(10):
                calc.update(make_snapshot(bid_qty=50.0, ask_qty=2.0))
            calc.flush()
        # 6th bar flush should show accumulated VPIN
        for _ in range(5):
            calc.update(make_snapshot(bid_qty=50.0, ask_qty=2.0))
        metrics = calc.flush()
        assert metrics.vpin > 0.0, "VPIN should be > 0 after directional flow"


class TestAggressorRatio:
    def test_aggressor_ratio_range(self):
        calc = OFICalculator()
        for _ in range(15):
            calc.update(make_snapshot(bid_qty=20.0, ask_qty=5.0))
        metrics = calc.flush()
        assert 0.0 <= metrics.aggressor_ratio <= 1.0

    def test_aggressor_ratio_buy_heavy(self):
        """Bid-heavy book should produce aggressor_ratio > 0.5."""
        calc = OFICalculator()
        # Provide initial snapshot then change to bid-heavy
        calc.update(make_snapshot(bid_qty=10.0, ask_qty=10.0))
        for _ in range(20):
            calc.update(make_snapshot(bid_qty=40.0, ask_qty=5.0))
        metrics = calc.flush()
        assert metrics.aggressor_ratio > 0.5, (
            f"Expected buy-dominant aggressor ratio, got {metrics.aggressor_ratio:.3f}"
        )


class TestFlushReset:
    def test_flush_resets_accumulators(self):
        """After flush(), the next flush() on empty input should return zeros."""
        calc = OFICalculator()
        for _ in range(10):
            calc.update(make_snapshot(bid_qty=30.0, ask_qty=5.0))
        first = calc.flush()
        assert first.n_snapshots == 10

        # No new updates — flush again immediately
        second = calc.flush()
        assert second.n_snapshots == 0
        assert second.ofi_raw == 0.0

    def test_ofi_norm_clipped(self):
        """ofi_norm should always be in [-5, 5]."""
        calc = OFICalculator()
        # Build up rolling history
        for _ in range(10):
            for _ in range(5):
                calc.update(make_snapshot(bid_qty=1000.0, ask_qty=1.0))
            calc.flush()
        # Extreme bid pressure
        for _ in range(20):
            calc.update(make_snapshot(bid_qty=1e6, ask_qty=1.0))
        metrics = calc.flush()
        assert -5.0 <= metrics.ofi_norm <= 5.0, (
            f"ofi_norm not clipped: {metrics.ofi_norm}"
        )

    def test_spread_pct_non_negative(self):
        calc = OFICalculator()
        for _ in range(5):
            calc.update(make_snapshot(bid_price=49999.0, ask_price=50001.0))
        metrics = calc.flush()
        assert metrics.spread_pct >= 0.0
