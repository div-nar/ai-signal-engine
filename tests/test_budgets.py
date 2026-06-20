import pytest
from strategy.layers import BASELINE_BUDGETS
from strategy.budgets import apply_layer_tilt, LAYER_FLOOR, LAYER_CEILING


def test_zero_tilt_returns_baseline():
    tilt = {k: 0.0 for k in BASELINE_BUDGETS}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    for k in BASELINE_BUDGETS:
        assert out[k] == pytest.approx(BASELINE_BUDGETS[k])


def test_tilt_sums_to_one():
    tilt = {"power": 0.10, "compute": 0.05, "platform": -0.15,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert sum(out.values()) == pytest.approx(1.0)


def test_tilt_must_sum_to_zero():
    bad = {"power": 0.10, "compute": 0.0, "platform": 0.0,
           "fabrication": 0.0, "infrastructure": 0.0}
    with pytest.raises(ValueError):
        apply_layer_tilt(BASELINE_BUDGETS, bad)


def test_ceiling_enforced():
    # Huge tilt into compute is clamped at the ceiling, not allowed to run away.
    tilt = {"compute": 0.30, "platform": -0.30, "power": 0.0,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert out["compute"] <= LAYER_CEILING + 1e-9
    assert sum(out.values()) == pytest.approx(1.0)


def test_floor_enforced():
    # Draining platform below the floor is clamped up to the floor.
    tilt = {"platform": -0.18, "power": 0.18, "compute": 0.0,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert out["platform"] >= LAYER_FLOOR - 1e-9
    assert sum(out.values()) == pytest.approx(1.0)


def test_raises_when_bounds_infeasible(monkeypatch):
    # Force infeasibility: 5 layers * 0.10 max = 0.5 < 1.0, impossible to satisfy.
    import strategy.budgets as b
    monkeypatch.setattr(b, "LAYER_CEILING", 0.10)
    tilt = {k: 0.0 for k in BASELINE_BUDGETS}
    with pytest.raises(RuntimeError):
        b.apply_layer_tilt(BASELINE_BUDGETS, tilt)
