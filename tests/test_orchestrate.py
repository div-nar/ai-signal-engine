import datetime as dt
import json
import pandas as pd
import pytest
from db import init_targets_table, get_latest_target
from orchestrate import compute_weekly_target


class FakeThesisClient:
    def generate(self, prompt):
        return json.dumps({
            "layer_tilt": {"compute": 0.10, "platform": -0.10},
            "market_regime": "compute_constrained",
            "regime_shift": False,
            "signal_confidence": 0.8,
            "thesis_update": "compute bottleneck",
        })


class _Bar:
    def __init__(self, ts, close):
        self.timestamp = ts
        self.close = close


class FakeDataClient:
    """Returns an uptrending 200-day panel for two real-universe tickers per layer."""
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {}
        for j, s in enumerate(syms):
            data[s] = [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 0.0001) ** i)
                       for i in range(200)]
        class R: pass
        r = R(); r.data = data
        return r


def test_computes_and_persists_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    out = compute_weekly_target(
        docs=[{"id": 1, "source": "rss", "title": "t", "content": "c"}],
        db_path=db, thesis_client=FakeThesisClient(), data_client=FakeDataClient(),
        now=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
    )
    assert out["market_regime"] == "compute_constrained"
    assert sum(out["layer_budgets"].values()) == pytest.approx(1.0)
    assert out["target_weights"]  # non-empty
    assert sum(out["target_weights"].values()) == pytest.approx(1.0)
    # persisted
    assert get_latest_target(db)["market_regime"] == "compute_constrained"


def test_persist_false_does_not_write(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    compute_weekly_target(
        docs=[], db_path=db, thesis_client=FakeThesisClient(),
        data_client=FakeDataClient(), persist=False,
        now=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
    )
    assert get_latest_target(db) is None
