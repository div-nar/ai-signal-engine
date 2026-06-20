import pandas as pd
import pytest
from backtest.panel import load_price_panel, to_weekly_fridays, forward_returns


def _daily():
    idx = pd.date_range("2025-01-01", periods=20, freq="B")
    return pd.DataFrame({"AAA": [100 + i for i in range(20)],
                         "BBB": [200 + 2 * i for i in range(20)]}, index=idx)


def test_weekly_fridays_are_all_fridays():
    wk = to_weekly_fridays(_daily())
    assert all(ts.weekday() == 4 for ts in wk.index)
    assert not wk.empty


def test_forward_returns_between_rebalances():
    wk = to_weekly_fridays(_daily())
    fr = forward_returns(_daily(), wk.index)
    # one fewer row than rebalance dates (last has no forward period)
    assert len(fr) == len(wk.index) - 1
    # AAA rises each day -> all forward returns positive
    assert (fr["AAA"] > 0).all()


def test_load_price_panel_single_ticker_returns_close_only(mocker):
    import backtest.panel as bp
    idx = pd.date_range("2025-01-01", periods=3, freq="B")
    flat = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3], "Close": [10.0, 11.0, 12.0]},
        index=idx,
    )
    mocker.patch.object(bp.yf, "download", return_value=flat)
    out = bp.load_price_panel(["AAPL"], "2025-01-01", "2025-01-10")
    assert list(out.columns) == ["AAPL"]
    assert out["AAPL"].tolist() == [10.0, 11.0, 12.0]


def test_load_price_panel_multi_ticker_selects_close(mocker):
    import backtest.panel as bp
    idx = pd.date_range("2025-01-01", periods=2, freq="B")
    cols = pd.MultiIndex.from_product([["Open", "Close"], ["AAA", "BBB"]])
    data = pd.DataFrame(
        [[1, 1, 10.0, 20.0], [2, 2, 11.0, 21.0]], index=idx, columns=cols
    )
    mocker.patch.object(bp.yf, "download", return_value=data)
    out = bp.load_price_panel(["AAA", "BBB"], "2025-01-01", "2025-01-10")
    assert set(out.columns) == {"AAA", "BBB"}
    assert out["AAA"].tolist() == [10.0, 11.0]
