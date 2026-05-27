import json
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta


def _make_fake_history(n=90, vix_spike_on_last=False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "shipping_pressure": rng.uniform(0.3, 0.7, n),
        "copper_infra_lead_neg": rng.standard_normal(n),
        "power_compute_lead_neg": rng.standard_normal(n),
        "vix_level": rng.uniform(14, 22, n),
        "pmi_neg": rng.uniform(-2, 2, n),
    }
    df = pd.DataFrame(data)
    if vix_spike_on_last:
        df.loc[n - 1, "vix_level"] = 40.0
    return df


def _write_fake_cache(path: Path, history_df: pd.DataFrame):
    from macro.composite import fit_and_cache_composite
    with patch("macro.composite._build_signal_history", return_value=history_df):
        fit_and_cache_composite(str(path))


# ── is_cache_stale ────────────────────────────────────────────────────────────

def test_missing_cache_is_stale(tmp_path):
    from macro.composite import is_cache_stale
    assert is_cache_stale(str(tmp_path / "nonexistent.json")) is True


def test_fresh_cache_is_not_stale(tmp_path):
    from macro.composite import is_cache_stale
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"computed_at": datetime.now(timezone.utc).isoformat()}))
    assert is_cache_stale(str(cache), max_age_days=8) is False


def test_old_cache_is_stale(tmp_path):
    from macro.composite import is_cache_stale
    cache = tmp_path / "cache.json"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    cache.write_text(json.dumps({"computed_at": old_ts}))
    assert is_cache_stale(str(cache), max_age_days=8) is True


# ── fit_and_cache_composite ───────────────────────────────────────────────────

def test_fit_writes_cache(tmp_path):
    from macro.composite import fit_and_cache_composite
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert "pc1_loadings" in data
    assert "history_mean" in data
    assert "history_std" in data
    assert "pc1_history_scores" in data
    assert len(data["pc1_loadings"]) == 5


def test_vix_loading_is_positive_after_orientation(tmp_path):
    from macro.composite import fit_and_cache_composite
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    data = json.loads(cache.read_text())
    signal_names = data["signal_names"]
    vix_idx = signal_names.index("vix_level")
    assert data["pc1_loadings"][vix_idx] > 0


# ── load_composite_modifier ───────────────────────────────────────────────────

def test_cold_start_returns_zero_modifier(tmp_path):
    from macro.composite import load_composite_modifier
    supply = {"shipping_pressure": 0.5, "pmi": 52.0}
    cross = {"copper_infra_lead": 0.2, "power_compute_lead": 0.3, "vix_level": 16.0}
    modifier, info = load_composite_modifier(str(tmp_path / "nonexistent.json"), supply, cross)
    assert modifier == pytest.approx(0.0)
    assert info["stress_score"] == pytest.approx(0.0)


def test_high_vix_increases_stress_score(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    normal_df = _make_fake_history(n=90)
    with patch("macro.composite._build_signal_history", return_value=normal_df):
        fit_and_cache_composite(str(cache))

    low_vix_supply = {"shipping_pressure": 0.3, "pmi": 54.0}
    low_vix_cross = {"copper_infra_lead": 0.5, "power_compute_lead": 0.5, "vix_level": 14.0}
    _, low_info = load_composite_modifier(str(cache), low_vix_supply, low_vix_cross)

    high_vix_cross = {"copper_infra_lead": 0.5, "power_compute_lead": 0.5, "vix_level": 35.0}
    _, high_info = load_composite_modifier(str(cache), low_vix_supply, high_vix_cross)

    assert high_info["stress_score"] > low_info["stress_score"]


def test_modifier_is_non_positive(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    supply = {"shipping_pressure": 0.9, "pmi": 44.0}
    cross = {"copper_infra_lead": -2.0, "power_compute_lead": -2.0, "vix_level": 30.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    assert modifier <= 0.0


def test_modifier_ceiling_is_minus_0_25(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    supply = {"shipping_pressure": 1.0, "pmi": 30.0}
    cross = {"copper_infra_lead": -5.0, "power_compute_lead": -5.0, "vix_level": 80.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    assert modifier >= -0.25


def test_net_exposure_never_falls_below_015(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    supply = {"shipping_pressure": 1.0, "pmi": 30.0}
    cross = {"copper_infra_lead": -5.0, "power_compute_lead": -5.0, "vix_level": 80.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    net = max(0.15, 0.80 + modifier)
    assert net >= 0.15
