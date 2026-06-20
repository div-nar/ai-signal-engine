import pandas as pd
import pytest
from backtest.panel import to_weekly_fridays, forward_returns


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
