import datetime as dt
import pandas as pd
from pricing.history import fetch_recent_closes


class _Bar:
    def __init__(self, ts, close):
        self.timestamp = ts
        self.close = close


class FakeDataClient:
    def __init__(self, data):
        self._data = data
        self.last_request = None

    def get_stock_bars(self, request):
        self.last_request = request
        class R:
            pass
        r = R()
        r.data = self._data
        return r


def test_builds_close_panel():
    d0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data = {
        "AAA": [_Bar(d0, 10.0), _Bar(d0 + dt.timedelta(days=1), 11.0)],
        "BBB": [_Bar(d0, 20.0), _Bar(d0 + dt.timedelta(days=1), 21.0)],
    }
    panel = fetch_recent_closes(["AAA", "BBB"], lookback_days=30,
                                client=FakeDataClient(data),
                                now=dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc))
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel["AAA"].tolist() == [10.0, 11.0]
    assert list(panel.index) == sorted(panel.index)


def test_empty_data_returns_empty_frame():
    panel = fetch_recent_closes(["AAA"], client=FakeDataClient({}),
                                now=dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc))
    assert panel.empty
