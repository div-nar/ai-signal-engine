import pytest
from unittest.mock import patch, MagicMock


def _make_position(symbol, market_value, side="long"):
    p = MagicMock()
    p.symbol = symbol
    p.market_value = str(market_value)
    p.side = side
    return p


def _make_account(portfolio_value=100_000.0):
    acc = MagicMock()
    acc.portfolio_value = str(portfolio_value)
    return acc


def test_get_alpaca_positions_returns_longs():
    from execution.alpaca import get_alpaca_positions
    mock_client = MagicMock()
    mock_client.get_account.return_value = _make_account(100_000)
    mock_client.get_all_positions.return_value = [
        _make_position("NVDA", 12_000, "long"),
        _make_position("MU",   9_000, "long"),
    ]

    with patch("execution.alpaca.TradingClient", return_value=mock_client), \
         patch.dict("os.environ", {"ALPACA_API_KEY": "fake", "ALPACA_SECRET_KEY": "fake"}):
        result = get_alpaca_positions()

    assert result["longs"]["NVDA"] == pytest.approx(0.12)
    assert result["longs"]["MU"] == pytest.approx(0.09)
    assert result["net_exposure"] == pytest.approx(0.21)
    assert result["portfolio_value"] == pytest.approx(100_000)


def test_get_alpaca_positions_returns_empty_when_no_credentials():
    from execution.alpaca import get_alpaca_positions
    with patch.dict("os.environ", {}, clear=True):
        result = get_alpaca_positions()
    assert result == {"longs": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}


def test_rebalance_skips_tiny_orders():
    from execution.alpaca import rebalance
    mock_client = MagicMock()
    mock_client.get_account.return_value = _make_account(100_000)
    mock_client.get_all_positions.return_value = []
    mock_client.get_orders.return_value = []

    # Weight of 0.003 → $300 long notional at net=0.80, below $500 min threshold
    with patch("execution.alpaca.TradingClient", return_value=mock_client), \
         patch.dict("os.environ", {"ALPACA_API_KEY": "fake", "ALPACA_SECRET_KEY": "fake"}):
        rebalance(long_weights={"NVDA": 0.003}, net_exposure_target=0.80)

    mock_client.submit_order.assert_not_called()


def test_rebalance_executes_without_error_at_boundary_gross():
    from execution.alpaca import rebalance
    long_weights = {f"T{i}": 0.1 for i in range(10)}   # sums to 1.0
    with patch("execution.alpaca.TradingClient") as mock_cls, \
         patch.dict("os.environ", {"ALPACA_API_KEY": "fake", "ALPACA_SECRET_KEY": "fake"}):
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_account(100_000)
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = []
        mock_cls.return_value = mock_client

        rebalance(long_weights=long_weights, net_exposure_target=0.80)
        # Should complete without raising
