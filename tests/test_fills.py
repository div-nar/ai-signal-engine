from unittest.mock import MagicMock
from execution.alpaca import wait_for_fills


def _order(status):
    o = MagicMock()
    o.status = status
    return o


def test_empty_returns_empty():
    assert wait_for_fills(MagicMock(), [], sleep=lambda *_: None) == {}


def test_all_filled():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order("OrderStatus.FILLED")
    out = wait_for_fills(client, ["a", "b"], sleep=lambda *_: None)
    assert out == {"a": "filled", "b": "filled"}


def test_unfilled_marked_timeout():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order("OrderStatus.ACCEPTED")
    out = wait_for_fills(client, ["a"], timeout_s=0.0, sleep=lambda *_: None)
    assert out == {"a": "timeout"}


def test_mixed_terminal_states():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order(
        "OrderStatus.REJECTED" if oid == "a" else "OrderStatus.FILLED")
    out = wait_for_fills(client, ["a", "b"], sleep=lambda *_: None)
    assert out == {"a": "rejected", "b": "filled"}
