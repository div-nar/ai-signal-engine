from strategy.layers import (
    LAYERS, LAYER_MAP, BASELINE_BUDGETS, layer_of, tickers_in_layer,
)


def test_layers_canonical_order():
    assert LAYERS == ["power", "fabrication", "compute", "infrastructure", "platform"]


def test_baseline_budgets_sum_to_one():
    assert abs(sum(BASELINE_BUDGETS.values()) - 1.0) < 1e-9
    assert set(BASELINE_BUDGETS) == set(LAYERS)


def test_known_tickers_mapped_to_expected_layers():
    assert layer_of("VST") == "power"
    assert layer_of("TSM") == "fabrication"
    assert layer_of("NVDA") == "compute"
    assert layer_of("MU") == "compute"
    assert layer_of("VRT") == "infrastructure"
    assert layer_of("MSFT") == "platform"


def test_non_ai_names_are_unmapped():
    # Healthcare / financials / staples are not part of the thesis universe.
    assert layer_of("JNJ") is None
    assert layer_of("JPM") is None


def test_every_layer_has_members():
    for layer in LAYERS:
        assert len(tickers_in_layer(layer)) >= 3, layer


def test_layer_map_values_are_valid_layers():
    assert all(v in LAYERS for v in LAYER_MAP.values())
