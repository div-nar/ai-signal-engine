import json
from db import init_targets_table, insert_target, get_latest_target


def test_insert_and_get_latest(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    assert get_latest_target(db) is None
    rid = insert_target(db, {
        "layer_tilt": {"power": 0.05, "compute": 0.0, "platform": -0.05,
                       "fabrication": 0.0, "infrastructure": 0.0},
        "layer_budgets": {"power": 0.25, "compute": 0.25, "platform": 0.15,
                          "fabrication": 0.20, "infrastructure": 0.15},
        "target_weights": {"VST": 0.12, "NVDA": 0.12},
        "market_regime": "compute_constrained",
        "thesis_update": "power is the binding constraint",
        "regime_shift": True,
    })
    assert rid >= 1
    got = get_latest_target(db)
    assert got["market_regime"] == "compute_constrained"
    assert got["regime_shift"] is True
    assert got["target_weights"]["VST"] == 0.12
    assert got["layer_budgets"]["power"] == 0.25


def test_get_latest_returns_most_recent(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {},
                       "market_regime": "balanced", "thesis_update": "first",
                       "regime_shift": False})
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {},
                       "market_regime": "stalling", "thesis_update": "second",
                       "regime_shift": False})
    assert get_latest_target(db)["thesis_update"] == "second"
