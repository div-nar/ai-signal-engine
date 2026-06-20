import pytest
from strategy.layers import LAYERS
from scoring.thesis_scorer import normalize_tilt, parse_thesis_response


def test_normalize_fills_missing_layers_and_sums_zero():
    out = normalize_tilt({"power": 0.10})
    assert set(out) == set(LAYERS)
    assert sum(out.values()) == pytest.approx(0.0)


def test_normalize_drops_unknown_keys():
    out = normalize_tilt({"power": 0.10, "crypto": 0.5})
    assert "crypto" not in out
    assert sum(out.values()) == pytest.approx(0.0)


def test_normalize_recenters_nonzero_sum():
    # raw sums to +0.10; after recentering it must sum to 0
    out = normalize_tilt({"power": 0.10, "compute": 0.0, "platform": 0.0,
                          "fabrication": 0.0, "infrastructure": 0.0})
    assert sum(out.values()) == pytest.approx(0.0)
    # power should remain the most-positive layer after recentering
    assert max(out, key=out.get) == "power"


def test_parse_strips_code_fence():
    text = '```json\n{"market_regime": "balanced", "layer_tilt": {"power": 0.1}}\n```'
    out = parse_thesis_response(text)
    assert out["market_regime"] == "balanced"
    assert out["layer_tilt"]["power"] == 0.1
