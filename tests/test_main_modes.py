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
    dispatch("buy", run_passive=passive, run_sell_fn=sell_fn, run_buy_fn=buy_fn,
             gather_docs=MagicMock(), record_snapshot=MagicMock())
    buy_fn.assert_called_once()
    sell_fn.assert_not_called()
    passive.assert_not_called()


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError):
        dispatch("nope", run_passive=MagicMock(), run_sell_fn=MagicMock(),
                 run_buy_fn=MagicMock(), gather_docs=MagicMock(), record_snapshot=MagicMock())


def test_dispatch_passive_passes_gathered_docs():
    docs = [{"id": 1}]
    passive = MagicMock()
    dispatch("passive", run_passive=passive, run_sell_fn=MagicMock(), run_buy_fn=MagicMock(),
             gather_docs=lambda: docs, record_snapshot=MagicMock())
    passive.assert_called_once_with(docs)


def test_dispatch_sell_passes_gathered_docs():
    docs = [{"id": 2}]
    sell_fn = MagicMock()
    dispatch("sell", run_passive=MagicMock(), run_sell_fn=sell_fn, run_buy_fn=MagicMock(),
             gather_docs=lambda: docs, record_snapshot=MagicMock())
    sell_fn.assert_called_once_with(docs)


def test_dispatch_buy_records_snapshot():
    buy_fn, snap = MagicMock(), MagicMock()
    dispatch("buy", run_passive=MagicMock(), run_sell_fn=MagicMock(), run_buy_fn=buy_fn,
             gather_docs=MagicMock(), record_snapshot=snap)
    buy_fn.assert_called_once()
    snap.assert_called_once()
