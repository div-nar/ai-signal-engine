"""On-demand same-day trade mode: sell + buy in one session, --force bypass."""
import datetime as dt
import json
from unittest.mock import MagicMock
from db import init_targets_table, get_latest_target
from main import parse_args, dispatch
from orchestrate import run_trade


def _thesis(urgency="normal"):
    class C:
        def generate(self, prompt):
            return json.dumps({
                "layer_tilt": {"compute": 0.1, "platform": -0.1},
                "rebalance_urgency": urgency,
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


def _exec_client(cash="100000"):
    c = MagicMock()
    c.get_account.return_value = MagicMock(portfolio_value="100000", cash=cash)
    c.get_all_positions.return_value = []
    c.submit_order.return_value = MagicMock(id="oid")
    c.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    return c


def test_parse_trade_mode_and_force():
    args = parse_args(["--mode", "trade", "--force"])
    assert args.mode == "trade" and args.force is True
    assert parse_args(["--mode", "trade"]).force is False


def test_trade_runs_sells_then_buys_same_session(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = _exec_client()
    out = run_trade(docs=[], db_path=db, thesis_client=_thesis("normal"),
                    data_client=FakeData(), exec_client=exec_client, now=_NOW)
    assert out["trade_gate"] == "traded"
    # empty book, all-cash account -> no sells needed, but buys submitted
    assert exec_client.submit_order.call_count >= 1
    sides = {str(c[0][0].side) for c in exec_client.submit_order.call_args_list}
    assert any("BUY" in s.upper() for s in sides)


def test_trade_respects_hold_gate_without_force(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = _exec_client()
    out = run_trade(docs=[], db_path=db, thesis_client=_thesis("hold"),
                    data_client=FakeData(), exec_client=exec_client, now=_NOW)
    assert out["trade_gate"] == "skipped_hold"
    exec_client.submit_order.assert_not_called()


def test_trade_force_bypasses_hold_gate(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = _exec_client()
    out = run_trade(docs=[], db_path=db, thesis_client=_thesis("hold"),
                    data_client=FakeData(), exec_client=exec_client, now=_NOW,
                    force=True)
    assert out["trade_gate"] == "traded"
    assert get_latest_target(db)["trade_gate"] == "traded"
    assert exec_client.submit_order.call_count >= 1


def test_dispatch_routes_trade_with_force_and_snapshot():
    docs = [{"id": 1}]
    trade_fn, snap, chroma = MagicMock(), MagicMock(), MagicMock()
    dispatch("trade", run_passive=MagicMock(), run_sell_fn=MagicMock(),
             run_buy_fn=MagicMock(), run_trade_fn=trade_fn,
             gather_docs=lambda: docs, record_snapshot=snap,
             init_chroma_fn=lambda: chroma, force=True)
    trade_fn.assert_called_once_with(docs, chroma_client=chroma, force=True)
    snap.assert_called_once()
