import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_fred_series(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="MS"))


def _make_yf_close(values, ticker="BDRY"):
    idx = pd.date_range("2025-11-01", periods=len(values), freq="B")
    return pd.DataFrame({"Close": values}, index=idx)


def test_fetch_supply_chain_signal_returns_required_keys():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch.dict(os.environ, {"FRED_API_KEY": "test"}), \
         patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        # DGORDER: flat orders, IPG3344S: falling semis
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [280000.0, 280000.0, 280500.0] if s == "DGORDER" else [98.0, 97.5, 97.0]
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 40 + [115] * 20)

        result = fetch_supply_chain_signal()

    assert set(result.keys()) == {"shipping_pressure", "semis_inventory_trend", "pmi", "pmi_trend"}
    assert 0.0 <= result["shipping_pressure"] <= 1.0
    assert result["pmi_trend"] in {"expanding", "contracting", "stable"}
    assert result["semis_inventory_trend"] in {"drawing_down", "building", "neutral"}


def test_pmi_below_50_and_falling_is_contracting():
    # DGORDER MoM < -0.5% → contracting; pmi = 50 + mom_pct*500
    # prev=280000, latest=277000 → mom=-1.07% → pmi≈44.6
    from macro.supply_chain import fetch_supply_chain_signal
    with patch.dict(os.environ, {"FRED_API_KEY": "test"}), \
         patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [280000.0, 277000.0] if s == "DGORDER" else [98.0] * 5
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["pmi_trend"] == "contracting"
    assert result["pmi"] < 50.0


def test_pmi_above_50_is_expanding():
    # DGORDER MoM > +0.5% → expanding
    # prev=280000, latest=283000 → mom=+1.07% → expanding
    from macro.supply_chain import fetch_supply_chain_signal
    with patch.dict(os.environ, {"FRED_API_KEY": "test"}), \
         patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [280000.0, 283000.0] if s == "DGORDER" else [98.0] * 5
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["pmi_trend"] == "expanding"
    assert result["pmi"] > 50.0


def test_high_shipping_momentum_gives_high_pressure():
    from macro.supply_chain import fetch_supply_chain_signal
    # BDRY up 25% over 30 days → shipping_pressure should be 1.0 (capped)
    base = [100.0] * 30
    high = [125.0] * 30
    with patch.dict(os.environ, {"FRED_API_KEY": "test"}), \
         patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series([280000.0, 282000.0])
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close(base + high)

        result = fetch_supply_chain_signal()

    assert result["shipping_pressure"] == pytest.approx(1.0)


def test_falling_semis_ip_is_drawing_down():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch.dict(os.environ, {"FRED_API_KEY": "test"}), \
         patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        # DGORDER flat, IPG3344S falling
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [280000.0, 280000.0] if s == "DGORDER" else [100.0, 99.0, 98.0]
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["semis_inventory_trend"] == "drawing_down"


def test_missing_fred_api_key_returns_default_signal(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from macro.supply_chain import fetch_supply_chain_signal
    result = fetch_supply_chain_signal()
    assert result["pmi_trend"] == "stable"
    assert result["shipping_pressure"] == 0.5
