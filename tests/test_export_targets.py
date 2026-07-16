import json

from db import init_targets_table, insert_target
from export_targets import build_payload, export


def _insert(db, **over):
    data = {
        "layer_tilt": {}, "layer_budgets": {},
        "target_weights": {"MU": 0.6, "GEV": 0.4},
        "market_regime": "compute_constrained",
        "thesis_update": "x", "regime_shift": False,
    }
    data.update(over)
    return insert_target(db, data)


def test_build_payload_maps_db_row_to_dashboard_shape():
    target = {
        "id": 11, "computed_at": "2026-07-03 13:00:38",
        "target_weights": {"MU": 0.09}, "market_regime": "shipping_bottleneck",
        "thesis_update": "ignored", "cash_buffer": 0.1,
    }
    assert build_payload(target) == {
        "id": 11,
        "computed_at": "2026-07-03 13:00:38",
        "weights": {"MU": 0.09},
        "regime": "shipping_bottleneck",
    }


def test_export_writes_latest_target(tmp_path):
    db = str(tmp_path / "t.db")
    out = tmp_path / "targets.json"
    init_targets_table(db)
    _insert(db, market_regime="first")
    _insert(db, market_regime="second", target_weights={"ARM": 1.0})

    payload = export(db_path=db, out_path=str(out))

    assert payload["regime"] == "second"
    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["weights"] == {"ARM": 1.0}


def test_export_raises_when_no_target(tmp_path):
    db = str(tmp_path / "t.db")
    out = tmp_path / "targets.json"
    init_targets_table(db)

    try:
        export(db_path=db, out_path=str(out))
    except SystemExit as e:
        assert e.code != 0
    else:
        raise AssertionError("expected SystemExit when targets table is empty")

    assert not out.exists()


def test_export_leaves_existing_file_untouched_on_empty_db(tmp_path):
    db = str(tmp_path / "t.db")
    out = tmp_path / "targets.json"
    init_targets_table(db)
    out.write_text('{"id": 11, "weights": {"MU": 0.09}}')

    try:
        export(db_path=db, out_path=str(out))
    except SystemExit:
        pass

    assert json.loads(out.read_text())["id"] == 11
