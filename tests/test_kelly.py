"""
Unit tests for Kelly Criterion position sizing in src/execution/controller.py

Validates:
  - Kelly formula output matches analytical calculation
  - Fractional (0.5x) Kelly is applied
  - Position is capped at max_position_usd
  - Zero or negative edge produces zero position
  - Extreme probability inputs are handled safely
"""

import pytest

# We test the private method directly via a thin controller instance
from src.execution.controller import ExecutionController


# ─── Minimal stub for dependencies ───────────────────────────────────────────

class _StubClient:
    """KalshiClient stub — no network calls needed for Kelly tests."""
    def get_btc_markets(self): return []
    def get_orderbook(self, t): return {}
    def get_balance(self): return 1000.0
    def place_order(self, **kw): return {"order_id": "stub", "status": "resting"}
    def get_order(self, oid): return {"status": "resting", "filled_count": 0}
    def cancel_order(self, oid): return {"order_id": oid, "status": "canceled"}


class _StubRisk:
    """RiskManager stub."""
    kill_switch_is_set = False
    def allow_trade(self): return True
    def register_order_open(self): pass
    def register_order_closed(self, *a): pass
    @property
    def kill_switch(self):
        import threading
        e = threading.Event()
        if self.kill_switch_is_set:
            e.set()
        return e


def make_controller(bankroll: float = 1000.0, max_pos: float = 50.0) -> ExecutionController:
    return ExecutionController(
        client=_StubClient(),
        risk_manager=_StubRisk(),
        bankroll=bankroll,
        max_position_usd=max_pos,
    )


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestKellyFormula:
    def test_zero_edge_produces_zero_position(self):
        """If model probability equals market price, edge is zero."""
        ctrl = make_controller(bankroll=1000.0, max_pos=50.0)
        # p_model = p_market → no edge
        pos = ctrl._kelly_size(p_model=0.50, p_market=0.50)
        assert pos == 0.0 or pos < 0.01, f"Expected ~0 position, got {pos}"

    def test_negative_edge_produces_zero_position(self):
        """Model probability below market price → no bet."""
        ctrl = make_controller(bankroll=1000.0, max_pos=50.0)
        pos = ctrl._kelly_size(p_model=0.45, p_market=0.60)
        assert pos == 0.0, f"Expected 0 for negative edge, got {pos}"

    def test_positive_edge_produces_positive_position(self):
        """Model probability above market price → positive position."""
        ctrl = make_controller(bankroll=1000.0, max_pos=50.0)
        pos = ctrl._kelly_size(p_model=0.70, p_market=0.50)
        assert pos > 0, f"Expected positive position, got {pos}"

    def test_formula_correctness(self):
        """
        Manual verification:
            p_model = 0.70, p_market = 0.50
            b = (1 - 0.50) / 0.50 = 1.0
            q = 1 - 0.70 = 0.30
            f_full = (1.0 * 0.70 - 0.30) / 1.0 = 0.40
            f_half = 0.5 * 0.40 = 0.20
            position = 0.20 * 1000 = 200.0, capped at 50.0
        """
        ctrl = make_controller(bankroll=1000.0, max_pos=50.0)
        pos = ctrl._kelly_size(p_model=0.70, p_market=0.50)
        # Should be capped at max_pos=50
        assert pos == pytest.approx(50.0, abs=0.01)

    def test_fractional_kelly_is_half(self):
        """With a small max_pos that doesn't cap, verify 0.5× factor."""
        ctrl = make_controller(bankroll=100.0, max_pos=1000.0)  # high cap to avoid capping
        pos = ctrl._kelly_size(p_model=0.70, p_market=0.50)
        # f_full = 0.40, f_half = 0.20, position = 0.20 * 100 = 20
        assert pos == pytest.approx(20.0, abs=0.01)

    def test_capped_at_max_position(self):
        """Position never exceeds max_position_usd."""
        ctrl = make_controller(bankroll=10_000.0, max_pos=25.0)
        pos = ctrl._kelly_size(p_model=0.90, p_market=0.50)
        assert pos <= 25.0 + 1e-9, f"Position exceeded cap: {pos}"

    def test_extreme_probability_safe(self):
        """p_market at extremes (0 or 1) should not crash."""
        ctrl = make_controller()
        assert ctrl._kelly_size(p_model=0.9, p_market=0.0) == 0.0
        assert ctrl._kelly_size(p_model=0.9, p_market=1.0) == 0.0

    def test_position_scales_with_edge(self):
        """Higher model confidence → larger position (before cap)."""
        ctrl = make_controller(bankroll=100.0, max_pos=1000.0)
        pos_low  = ctrl._kelly_size(p_model=0.55, p_market=0.50)
        pos_high = ctrl._kelly_size(p_model=0.75, p_market=0.50)
        assert pos_high > pos_low, (
            f"Higher edge should produce larger position: {pos_low:.2f} vs {pos_high:.2f}"
        )

    def test_bankroll_update_reflected(self):
        """After update_bankroll(), position sizing uses new bankroll."""
        ctrl = make_controller(bankroll=100.0, max_pos=1000.0)
        pos_before = ctrl._kelly_size(p_model=0.70, p_market=0.50)
        ctrl.update_bankroll(200.0)
        pos_after = ctrl._kelly_size(p_model=0.70, p_market=0.50)
        assert pytest.approx(pos_after, rel=0.01) == 2 * pos_before


class TestKalshiFeeFormula:
    def test_fee_at_50_cents(self):
        """At p=0.50: fee = 0.07 × 0.5 × 0.5 = 0.0175"""
        from src.engine.alpha import kalshi_taker_fee
        fee = kalshi_taker_fee(0.50)
        assert fee == pytest.approx(0.0175, abs=1e-9)

    def test_fee_at_extremes_is_small(self):
        """Fee should approach 0 as p → 0 or p → 1."""
        from src.engine.alpha import kalshi_taker_fee
        assert kalshi_taker_fee(0.01) < 0.001
        assert kalshi_taker_fee(0.99) < 0.001

    def test_fee_symmetric(self):
        """Fee is symmetric: f(p) == f(1-p)."""
        from src.engine.alpha import kalshi_taker_fee
        for p in [0.2, 0.3, 0.4]:
            assert kalshi_taker_fee(p) == pytest.approx(kalshi_taker_fee(1 - p), abs=1e-9)
