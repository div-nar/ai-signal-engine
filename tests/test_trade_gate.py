"""LLM rebalance urgency + drift bands gate the weekly trading legs."""
import datetime as dt
import json
from unittest.mock import MagicMock
from db import init_targets_table, insert_target, get_latest_target, update_target_gate
from orchestrate import run_sell, run_buy


def _thesis(urgency="normal", cash_buffer=0.0):
    class C:
        def generate(self, prompt):
            return json.dumps({
                "layer_tilt": {"compute": 0.1, "platform": -0.1},
                "rebalance_urgency": urgency,
                "cash_buffer": cash_buffer,
                "market_regime": "compute_constrained", "regime_shift": False,
                "signal_confidence": 0.8, "thesis_update": "x",
            })
    return C()


class _Bar:
    def __init__(self, ts, close): self.timestamp = ts; self.close = close


class FakeData:
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {s: [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 1e-4) ** i)
                    for i in range(260)] for j, s in enumerate(syms)}
        r = MagicMock(); r.data = data; return r


_NOW = dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc)


def test_hold_skips_sells_and_records_gate(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = MagicMock()
    out = run_sell(docs=[], db_path=db, thesis_client=_thesis("hold"),
                   data_client=FakeData(), exec_client=exec_client, now=_NOW)
    assert out["trade_gate"] == "skipped_hold"
    exec_client.submit_order.assert_not_called()
    assert get_latest_target(db)["trade_gate"] == "skipped_hold"


def test_within_bands_skips_when_book_matches_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    # First compute the target with a throwaway client to learn the weights...
    probe = run_sell(docs=[], db_path=db, thesis_client=_thesis("normal"),
                     data_client=FakeData(), exec_client=MagicMock(), now=_NOW)
    weights = probe["target_weights"]
    # ...then run again with a book already sitting exactly on target.
    exec_client = MagicMock()
    exec_client.get_account.return_value = MagicMock(portfolio_value="100000")
    positions = []
    for sym, w in weights.items():
        p = MagicMock(); p.symbol = sym; p.market_value = str(w * 100000)
        positions.append(p)
    exec_client.get_all_positions.return_value = positions
    out = run_sell(docs=[], db_path=db, thesis_client=_thesis("normal"),
                   data_client=FakeData(), exec_client=exec_client, now=_NOW)
    assert out["trade_gate"] == "skipped_within_bands"
    exec_client.submit_order.assert_not_called()


def test_urgent_trades_even_on_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = MagicMock()
    exec_client.get_account.return_value = MagicMock(portfolio_value="100000")
    exec_client.get_all_positions.return_value = []
    out = run_sell(docs=[], db_path=db, thesis_client=_thesis("urgent"),
                   data_client=FakeData(), exec_client=exec_client, now=_NOW)
    assert out["trade_gate"] == "traded"


def test_buy_leg_honours_skip_gate(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    rid = insert_target(db, {"layer_tilt": {}, "layer_budgets": {},
                             "target_weights": {"NVDA": 1.0},
                             "market_regime": "balanced", "thesis_update": "x",
                             "regime_shift": False})
    update_target_gate(db, rid, "skipped_hold")
    exec_client = MagicMock()
    out = run_buy(db_path=db, exec_client=exec_client)
    assert out["trade_gate"] == "skipped_hold"
    exec_client.submit_order.assert_not_called()


def test_llm_cash_buffer_scales_buys(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {},
                       "target_weights": {"NVDA": 1.0},
                       "market_regime": "balanced", "thesis_update": "x",
                       "regime_shift": False, "cash_buffer": 0.25})
    exec_client = MagicMock()
    exec_client.get_account.return_value = MagicMock(portfolio_value="100000", cash="100000")
    exec_client.get_all_positions.return_value = []
    exec_client.submit_order.return_value = MagicMock(id="oid")
    exec_client.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    run_buy(db_path=db, exec_client=exec_client)
    # target scaled by (1 - 0.25): NVDA notional 75k, not 100k
    notional = exec_client.submit_order.call_args[0][0].notional
    assert abs(notional - 75000.0) < 1.0
