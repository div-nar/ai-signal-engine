import pytest
from unittest.mock import patch

_SUPPLY_CLEAN = {
    "shipping_pressure": 0.3,
    "semis_inventory_trend": "neutral",
    "pmi": 52.0,
    "pmi_trend": "expanding",
}
_CROSS_CLEAN = {
    "power_compute_lead": 0.5,
    "copper_infra_lead": 0.2,
    "credit_stress": False,
    "vix_level": 16.0,
}
_ZERO_MODIFIER = (0.0, {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": 0})
_MAX_MODIFIER = (-0.25, {"stress_score": 1.0, "modifier": -0.25, "cache_age_days": 0})


def test_clean_macro_gives_compute_constrained():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "compute_constrained"
    assert result["net_exposure_target"] == pytest.approx(0.80)


def test_credit_stress_overrides_all():
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "credit_stress"
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_credit_stress_exempt_from_modifier():
    """Composite modifier must not be applied when regime is credit_stress."""
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed), \
         patch("macro.regime.load_composite_modifier", return_value=_MAX_MODIFIER):
        result = compute_macro_signal()
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_contracting_pmi_and_weak_copper_gives_balanced():
    from macro.regime import compute_macro_signal
    supply_weak = {**_SUPPLY_CLEAN, "pmi": 48.0, "pmi_trend": "contracting"}
    cross_weak = {**_CROSS_CLEAN, "copper_infra_lead": -0.8}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_weak), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_weak), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "balanced"
    assert result["net_exposure_target"] == pytest.approx(0.65)


def test_shipping_pressure_no_longer_triggers_own_regime():
    """High shipping pressure alone must NOT produce a shipping_bottleneck regime."""
    from macro.regime import compute_macro_signal
    supply_ship = {**_SUPPLY_CLEAN, "shipping_pressure": 0.90}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_ship), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] != "shipping_bottleneck"


def test_composite_modifier_reduces_net_exposure():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=(-0.15, {"stress_score": 0.6, "modifier": -0.15, "cache_age_days": 2})):
        result = compute_macro_signal()
    assert result["net_exposure_target"] == pytest.approx(0.80 - 0.15)
    assert result["composite_modifier"]["stress_score"] == pytest.approx(0.6)


def test_net_exposure_floored_at_015():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=(-0.25, {"stress_score": 1.0, "modifier": -0.25, "cache_age_days": 0})):
        result = compute_macro_signal()
    assert result["net_exposure_target"] >= 0.15


def test_macro_signal_has_all_required_keys():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    required = {
        "computed_at", "regime", "regime_confidence",
        "net_exposure_target", "supply_chain", "cross_sector",
        "notes", "composite_modifier",
    }
    assert required.issubset(result.keys())
