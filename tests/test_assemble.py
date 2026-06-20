import pytest
from strategy.assemble import assemble_portfolio

LAYER_MAP = {
    "P1": "power", "P2": "power",
    "C1": "compute", "C2": "compute", "C3": "compute", "C4": "compute",
}
BUDGETS = {"power": 0.4, "compute": 0.6,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def test_weights_sum_to_one():
    scores = {"P1": 0.3, "P2": 0.1, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": -0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=3, name_cap=0.5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_only_top_n_per_layer_selected():
    scores = {"P1": 0.3, "P2": 0.1, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": -0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=2, name_cap=0.5)
    # compute layer: only C1, C2 (top 2) selected; C3, C4 excluded
    assert set(w) == {"P1", "P2", "C1", "C2"}


def test_name_cap_enforced():
    # cap=0.30 is feasible with 5 selected names (5*0.30=1.5>=1.0) AND binds:
    # P1 would naturally take ~0.396 of power's budget, so the cap must pull it down.
    scores = {"P1": 0.9, "P2": 0.01, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": 0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=3, name_cap=0.30)
    assert max(w.values()) <= 0.30 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_empty_layer_skipped():
    budgets = {"power": 1.0, "compute": 0.0,
               "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}
    scores = {"P1": 0.3, "P2": 0.1}
    w = assemble_portfolio(budgets, scores, LAYER_MAP, top_n=3, name_cap=0.9)
    assert set(w) == {"P1", "P2"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_nonpositive_scores_fall_back_to_equal_weight():
    budgets = {"compute": 1.0, "power": 0.0,
               "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}
    scores = {"C1": -0.1, "C2": -0.2}
    w = assemble_portfolio(budgets, scores, LAYER_MAP, top_n=2, name_cap=0.9)
    assert w["C1"] == pytest.approx(w["C2"])
