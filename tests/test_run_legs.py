import datetime as dt
import json
from unittest.mock import MagicMock
from db import init_targets_table, insert_target
from orchestrate import run_sell, run_buy


class FakeThesis:
    def generate(self, prompt):
        return json.dumps({"layer_tilt": {"compute": 0.1, "platform": -0.1},
                           "market_regime": "compute_constrained", "regime_shift": False,
                           "signal_confidence": 0.8, "thesis_update": "x"})


class _Bar:
    def __init__(self, ts, close): self.timestamp = ts; self.close = close


class FakeData:
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {s: [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 1e-4) ** i)
                    for i in range(260)] for j, s in enumerate(syms)}
        r = MagicMock(); r.data = data; return r


def test_run_sell_computes_and_sells(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = MagicMock()
    out = run_sell(docs=[], db_path=db, thesis_client=FakeThesis(),
                   data_client=FakeData(), exec_client=exec_client,
                   now=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc))
    assert out["target_weights"]
    # execute_sells reads account/positions from the injected client
    exec_client.get_account.assert_called()


def test_run_buy_uses_latest_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {},
                       "target_weights": {"NVDA": 1.0},
                       "market_regime": "balanced", "thesis_update": "x",
                       "regime_shift": False})
    exec_client = MagicMock()
    exec_client.get_account.return_value = MagicMock(portfolio_value="100000", cash="100000")
    exec_client.get_all_positions.return_value = []
    exec_client.submit_order.return_value = MagicMock(id="oid")
    exec_client.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    out = run_buy(db_path=db, exec_client=exec_client)
    assert out["target_weights"]["NVDA"] == 1.0
    exec_client.submit_order.assert_called()


def test_run_buy_skips_when_no_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    assert run_buy(db_path=db, exec_client=MagicMock()) is None
