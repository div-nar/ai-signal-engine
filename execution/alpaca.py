import os
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

_MIN_ORDER_VALUE = 500.0
_ORDER_DELAY_S = 0.3
_CANCEL_POLL_INTERVAL_S = 1.0
_CANCEL_POLL_TIMEOUT_S = 30.0


def _get_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return TradingClient(api_key, secret_key, paper=True)


def _wait_for_cancels(client):
    """Poll until no open orders remain, up to _CANCEL_POLL_TIMEOUT_S."""
    deadline = time.monotonic() + _CANCEL_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        open_orders = client.get_orders()
        if not open_orders:
            return
        time.sleep(_CANCEL_POLL_INTERVAL_S)
    print("  WARNING: open orders still present after cancel timeout — proceeding anyway")


def get_alpaca_positions() -> dict:
    """Return current paper portfolio as {longs, net_exposure, gross_exposure, portfolio_value}."""
    client = _get_client()
    if not client:
        return {"longs": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}

    try:
        account = client.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            return {"longs": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}

        positions = client.get_all_positions()
        longs = {}
        for p in positions:
            if float(p.market_value) >= 0:
                longs[p.symbol] = abs(float(p.market_value)) / portfolio_value

        net = sum(longs.values())
        return {
            "longs": longs,
            "net_exposure": net,
            "gross_exposure": net,
            "portfolio_value": portfolio_value,
        }
    except Exception as e:
        print(f"  WARNING: Could not fetch Alpaca positions: {e}")
        return {"longs": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}


def rebalance(long_weights: dict, net_exposure_target: float) -> None:
    """Rebalance Alpaca paper account to target long weights.

    long_weights values sum to 1.0; dollar notional = portfolio_value * net_exposure_target.
    """
    client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping rebalance")
        return

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    long_notional = portfolio_value * net_exposure_target
    long_targets = {t: w * long_notional for t, w in long_weights.items()}

    # Cancel all open orders and wait until the account is clear before reading positions.
    try:
        client.cancel_orders()
    except Exception as e:
        print(f"  WARNING: Could not cancel open orders: {e}")
    _wait_for_cancels(client)

    # Re-fetch positions after cancels settle so held_for_orders qty is released.
    current_positions = client.get_all_positions()
    current_longs = {
        p.symbol: float(p.market_value)
        for p in current_positions
        if float(p.market_value) >= 0
    }

    def _submit(symbol, side, notional):
        if notional < _MIN_ORDER_VALUE:
            return
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            client.submit_order(req)
        except Exception as e:
            print(f"  WARNING: Order failed {side.value} {symbol} ${notional:,.0f}: {e}")
        time.sleep(_ORDER_DELAY_S)

    # Step 1: Close positions no longer in the long book.
    for sym in list(current_longs):
        if sym not in long_targets:
            try:
                client.close_position(sym)
                time.sleep(_ORDER_DELAY_S)
            except Exception as e:
                print(f"  WARNING: Could not close {sym}: {e}")

    # Step 2: Adjust/open long positions.
    for sym, target_val in long_targets.items():
        current_val = current_longs.get(sym, 0.0)
        diff = target_val - current_val
        if diff > _MIN_ORDER_VALUE:
            _submit(sym, OrderSide.BUY, diff)
        elif diff < -_MIN_ORDER_VALUE:
            _submit(sym, OrderSide.SELL, abs(diff))

    print(
        f"  Rebalance complete | long_notional=${long_notional:,.0f} "
        f"| net_exposure={net_exposure_target:.0%}"
    )
