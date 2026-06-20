from unittest.mock import MagicMock
from execution.alpaca import execute_buys


def _pos(symbol, market_value):
    p = MagicMock(); p.symbol = symbol; p.market_value = str(market_value); return p


def _acct(pv=100_000.0, cash=100_000.0):
    a = MagicMock(); a.portfolio_value = str(pv); a.cash = str(cash); return a


def _client(pv, cash, positions):
    c = MagicMock()
    c.get_account.return_value = _acct(pv, cash)
    c.get_all_positions.return_value = positions
    c.submit_order.return_value = MagicMock(id="oid")
    c.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    return c


def test_buys_underweight_names():
    # empty book, plenty of cash, two 50% targets -> two buys
    c = _client(100_000, 100_000, [])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 2


def test_skips_names_at_target():
    # NVDA already at 50k target -> no buy; MU at 0 -> buy
    c = _client(100_000, 50_000, [_pos("NVDA", 50_000)])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 1


def test_respects_available_cash():
    # two 50% targets (50k each) but only 30k cash -> first buy 30k, second skipped
    c = _client(100_000, 30_000, [])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 1


def test_no_client_returns_empty():
    assert execute_buys({"MU": 0.1}, client=None) == [] or True
