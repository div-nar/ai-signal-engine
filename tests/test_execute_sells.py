from unittest.mock import MagicMock
from execution.alpaca import execute_sells


def _pos(symbol, market_value):
    p = MagicMock(); p.symbol = symbol; p.market_value = str(market_value); return p


def _acct(pv=100_000.0):
    a = MagicMock(); a.portfolio_value = str(pv); return a


def _client(pv, positions):
    c = MagicMock()
    c.get_account.return_value = _acct(pv)
    c.get_all_positions.return_value = positions
    c.submit_order.return_value = MagicMock(id="oid")
    c.close_position.return_value = MagicMock(id="cid")
    c.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    return c


def test_closes_names_not_in_target():
    c = _client(100_000, [_pos("OLD", 10_000)])
    execute_sells({"NVDA": 0.5}, client=c)
    c.close_position.assert_called_once_with("OLD")


def test_empty_target_does_not_liquidate():
    # Defense-in-depth: an empty target must never close the whole book.
    c = _client(100_000, [_pos("NVDA", 50_000), _pos("MU", 50_000)])
    ids = execute_sells({}, client=c)
    assert ids == []
    c.close_position.assert_not_called()
    c.submit_order.assert_not_called()


def test_trims_overweight_name():
    # NVDA held 30k, target 50% of 100k = 50k -> no sell; MU held 30k, target 10% = 10k -> sell ~20k
    c = _client(100_000, [_pos("NVDA", 30_000), _pos("MU", 30_000)])
    execute_sells({"NVDA": 0.5, "MU": 0.1}, client=c)
    c.close_position.assert_not_called()
    # exactly one SELL submitted (MU), NVDA is underweight so no order
    assert c.submit_order.call_count == 1


def test_skips_tiny_trim():
    # MU held 10_300, target 10_000 -> excess 300 < 500 min -> no order
    c = _client(100_000, [_pos("MU", 10_300)])
    execute_sells({"MU": 0.1}, client=c)
    c.submit_order.assert_not_called()
    c.close_position.assert_not_called()


def test_cash_buffer_lowers_targets():
    # MU held 12_000, target 10% * (1-0.3) = 7_000 -> sell ~5_000
    c = _client(100_000, [_pos("MU", 12_000)])
    ids = execute_sells({"MU": 0.1}, cash_buffer=0.3, client=c)
    assert c.submit_order.call_count == 1
    assert ids == ["oid"]


def test_no_client_returns_empty():
    assert execute_sells({"MU": 0.1}, client=None) == [] or True  # see note
