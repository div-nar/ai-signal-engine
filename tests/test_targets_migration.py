"""New target fields round-trip + in-place migration of the pre-expansion schema."""
import sqlite3
from db import init_targets_table, insert_target, get_latest_target


def test_new_fields_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {
        "layer_tilt": {}, "layer_budgets": {}, "target_weights": {"NVDA": 1.0},
        "market_regime": "balanced", "thesis_update": "x", "regime_shift": False,
        "layer_top_n": {"compute": 2},
        "name_adjustments": {"NVDA": 1.3, "MU": 0},
        "cash_buffer": 0.15,
        "rebalance_urgency": "hold",
        "retrieval_log": [{"round": 1, "query": "HBM", "hits": 4}],
    })
    got = get_latest_target(db)
    assert got["layer_top_n"] == {"compute": 2}
    assert got["name_adjustments"] == {"NVDA": 1.3, "MU": 0}
    assert got["cash_buffer"] == 0.15
    assert got["rebalance_urgency"] == "hold"
    assert got["retrieval_log"][0]["query"] == "HBM"
    assert got["trade_gate"] == ""


def test_defaults_when_fields_absent(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {},
                       "market_regime": "balanced", "thesis_update": "x",
                       "regime_shift": False})
    got = get_latest_target(db)
    assert got["layer_top_n"] == {}
    assert got["name_adjustments"] == {}
    assert got["cash_buffer"] == 0.0
    assert got["rebalance_urgency"] == "normal"
    assert got["retrieval_log"] == []


def test_migrates_pre_expansion_schema_in_place(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE targets (
            id             INTEGER PRIMARY KEY,
            computed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            layer_tilt     TEXT,
            layer_budgets  TEXT,
            target_weights TEXT,
            market_regime  TEXT,
            thesis_update  TEXT,
            regime_shift   INTEGER DEFAULT 0
        )
    """)
    conn.execute("""INSERT INTO targets
                    (layer_tilt, layer_budgets, target_weights, market_regime,
                     thesis_update, regime_shift)
                    VALUES ('{}', '{}', '{"MU": 1.0}', 'balanced', 'old row', 0)""")
    conn.commit()
    conn.close()

    init_targets_table(db)  # must ALTER in the new columns
    got = get_latest_target(db)
    assert got["thesis_update"] == "old row"          # old data intact
    assert got["target_weights"] == {"MU": 1.0}
    assert got["cash_buffer"] == 0.0                  # new fields defaulted
    assert got["rebalance_urgency"] == "normal"
    # and new-style inserts now work on the migrated table
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {"NVDA": 1.0},
                       "market_regime": "balanced", "thesis_update": "new row",
                       "regime_shift": False, "cash_buffer": 0.1})
    assert get_latest_target(db)["cash_buffer"] == 0.1
