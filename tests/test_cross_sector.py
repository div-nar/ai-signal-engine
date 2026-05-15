import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


def _make_close_df(tickers, n=90, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-11-01", periods=n, freq="B")
    data = {t: 100 + rng.normal(0, 2, n).cumsum() for t in tickers}
    return pd.DataFrame(data, index=idx)


ALL_TICKERS = ["NEE", "ETN", "PWR", "NVDA", "TSM", "HYG", "^VIX", "FCX"]


def test_fetch_cross_sector_signal_returns_required_keys():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert set(result.keys()) == {"power_compute_lead", "copper_infra_lead", "credit_stress", "vix_level"}


def test_high_vix_and_falling_hyg_triggers_credit_stress():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    # Override VIX to be high, HYG to be falling
    df["^VIX"] = 30.0
    df["HYG"] = [100.0 - i * 0.15 for i in range(len(df))]
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["credit_stress"] is True
    assert result["vix_level"] == pytest.approx(30.0)


def test_low_vix_does_not_trigger_credit_stress():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    df["^VIX"] = 15.0
    df["HYG"] = [100.0 + i * 0.05 for i in range(len(df))]
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["credit_stress"] is False


def test_power_leading_compute_gives_positive_lead_score():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    n = len(df)
    # Power basket rising strongly, compute flat
    for t in ["NEE", "ETN", "PWR"]:
        df[t] = [100.0 + i * 0.5 for i in range(n)]
    for t in ["NVDA", "TSM"]:
        df[t] = 100.0
    df["^VIX"] = 15.0
    df["HYG"] = 100.0
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["power_compute_lead"] > 0


def test_copper_rising_gives_positive_infra_lead():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    n = len(df)
    df["FCX"] = [100.0 + i * 0.4 for i in range(n)]
    df["^VIX"] = 15.0
    df["HYG"] = 100.0
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["copper_infra_lead"] > 0
