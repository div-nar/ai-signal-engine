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


def test_clean_macro_gives_compute_constrained():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert result["regime"] == "compute_constrained"
    assert result["net_exposure_target"] == pytest.approx(0.80)


def test_credit_stress_overrides_all_other_signals():
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    supply_shipping = {**_SUPPLY_CLEAN, "shipping_pressure": 0.9}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_shipping), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed):
        result = compute_macro_signal()
    assert result["regime"] == "credit_stress"
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_high_shipping_without_credit_stress_gives_shipping_bottleneck():
    from macro.regime import compute_macro_signal
    supply_ship = {**_SUPPLY_CLEAN, "shipping_pressure": 0.80}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_ship), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert result["regime"] == "shipping_bottleneck"
    assert result["net_exposure_target"] == pytest.approx(0.55)


def test_contracting_pmi_and_weak_copper_gives_balanced():
    from macro.regime import compute_macro_signal
    supply_weak = {**_SUPPLY_CLEAN, "pmi": 48.0, "pmi_trend": "contracting"}
    cross_weak = {**_CROSS_CLEAN, "copper_infra_lead": -0.8}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_weak), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_weak):
        result = compute_macro_signal()
    assert result["regime"] == "balanced"
    assert result["net_exposure_target"] == pytest.approx(0.65)


def test_macro_signal_has_all_required_keys():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert set(result.keys()) == {
        "computed_at", "regime", "regime_confidence",
        "net_exposure_target", "supply_chain", "cross_sector", "notes",
    }
    assert 0.0 <= result["regime_confidence"] <= 1.0
