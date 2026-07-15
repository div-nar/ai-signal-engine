"""Bounded LLM name emphasis on momentum scores + per-layer concentration dial."""
import pytest
from strategy.factors import apply_name_adjustments
from strategy.assemble import assemble_portfolio, _layer_top_n


def test_boost_and_dampen_positive_scores():
    out = apply_name_adjustments({"NVDA": 0.2, "AMD": 0.2}, {"NVDA": 1.5, "AMD": 0.5})
    assert out["NVDA"] == pytest.approx(0.3)
    assert out["AMD"] == pytest.approx(0.1)


def test_boost_improves_negative_scores_too():
    # A boost must never push a name further down the rank.
    out = apply_name_adjustments({"MU": -0.2}, {"MU": 1.5})
    assert out["MU"] == pytest.approx(-0.2 / 1.5)
    out = apply_name_adjustments({"MU": -0.2}, {"MU": 0.5})
    assert out["MU"] == pytest.approx(-0.4)


def test_veto_removes_name():
    out = apply_name_adjustments({"NVDA": 0.2, "MU": 0.4}, {"MU": 0})
    assert "MU" not in out
    assert out["NVDA"] == 0.2


def test_multiplier_clamped():
    out = apply_name_adjustments({"NVDA": 0.1}, {"NVDA": 99.0})
    assert out["NVDA"] == pytest.approx(0.15)  # clamped to 1.5x


def test_unknown_adjustments_ignored():
    out = apply_name_adjustments({"NVDA": 0.1}, {"ZZZ": 1.5})
    assert out == {"NVDA": 0.1}


def test_layer_top_n_dict_clamped_int_passthrough():
    assert _layer_top_n({"compute": 9}, "compute") == 4
    assert _layer_top_n({"compute": 1}, "compute") == 2
    assert _layer_top_n({}, "compute") == 3
    assert _layer_top_n({"compute": "bad"}, "compute") == 3
    assert _layer_top_n(1, "compute") == 1  # programmatic int not clamped


def test_assemble_respects_per_layer_top_n():
    layer_map = {"A1": "compute", "A2": "compute", "A3": "compute",
                 "B1": "power", "B2": "power", "B3": "power"}
    scores = {t: s for t, s in zip(["A1", "A2", "A3", "B1", "B2", "B3"],
                                   [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])}
    budgets = {"compute": 0.5, "power": 0.5}
    w = assemble_portfolio(budgets, scores, layer_map,
                           top_n={"compute": 2, "power": 3}, name_cap=1.0)
    assert set(w) == {"A1", "A2", "B1", "B2", "B3"}  # A3 cut by top_n=2
