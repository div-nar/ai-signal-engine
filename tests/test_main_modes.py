import pytest
from unittest.mock import MagicMock
from main import parse_args, dispatch


def test_parse_requires_mode():
    args = parse_args(["--mode", "sell"])
    assert args.mode == "sell"


def test_parse_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "bogus"])


def test_dispatch_routes_to_buy():
    passive, sell_fn, buy_fn = MagicMock(), MagicMock(), MagicMock()
    dispatch("buy", run_passive=passive, run_sell_fn=sell_fn, run_buy_fn=buy_fn)
    buy_fn.assert_called_once()
    sell_fn.assert_not_called()
    passive.assert_not_called()


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError):
        dispatch("nope", run_passive=MagicMock(), run_sell_fn=MagicMock(), run_buy_fn=MagicMock())
