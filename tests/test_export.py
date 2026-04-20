# ai-signal-engine/tests/test_export.py
import json
import pytest
from pathlib import Path
from export import export_signal


SAMPLE_SIGNAL = {
    "p_final": 0.82,
    "stock_conviction": json.dumps({"NVDA": 0.90, "ASML": 0.85, "TSM": 0.80}),
    "stock_weights": json.dumps({"NVDA": 0.10, "ASML": 0.09, "TSM": 0.08}),
    "stock_reasoning": json.dumps({"NVDA": "ASML backlog signals GPU tightening.", "ASML": "Record backlog.", "TSM": "Node utilization rising."}),
    "sector_tilt": json.dumps({}),
    "supply_demand_balance": 0.34,
    "market_regime": "compute_constrained",
    "signal_confidence": 0.76,
    "thesis_stress": False,
    "signal_age_days": 0,
    "sources_ingested": 3,
    "signal_breakdown": json.dumps({"compute": 0.25, "infrastructure": 0.20}),
    "thesis_update": "ASML backlog up 18% QoQ.",
}


def test_export_signal_writes_three_files(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))

    assert (tmp_path / "p_estimate.json").exists()
    assert (tmp_path / "stock_signals.json").exists()
    assert (tmp_path / "market_regime.json").exists()


def test_export_signal_p_estimate_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "p_estimate.json").read_text())
    assert data["p"] == 0.82
    assert "generated_at" in data


def test_export_signal_stock_signals_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "stock_signals.json").read_text())
    assert data["conviction"]["NVDA"] == 0.90
    assert "ASML backlog" in data["reasoning"]["NVDA"]
    assert "generated_at" in data


def test_export_signal_market_regime_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "market_regime.json").read_text())
    assert data["market_regime"] == "compute_constrained"
    assert data["supply_demand_balance"] == 0.34
    assert data["signal_confidence"] == 0.76
    assert data["thesis_stress"] is False
    assert "generated_at" in data
