import os
import time
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

_MIN_ORDER_VALUE = 500.0
_ORDER_DELAY_S = 0.3
_CANCEL_POLL_INTERVAL_S = 1.0
_CANCEL_POLL_TIMEOUT_S = 30.0
_FILL_POLL_S = 2.0
_FILL_TIMEOUT_S = 120.0
_TERMINAL_STATES = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}


def _status_str(order) -> str:
    """Normalize an order status (enum or str) to a bare lowercase token."""
    return str(getattr(order, "status", "")).lower().split(".")[-1]


def wait_for_fills(client, order_ids, timeout_s: float = _FILL_TIMEOUT_S,
                   poll_s: float = _FILL_POLL_S, sleep=time.sleep) -> dict:
    """Poll each order to a terminal state; return {order_id: status}.

    Orders that do not settle within timeout_s are recorded as 'timeout' with a warning.
    `sleep` is injectable so tests don't block.
    """
    statuses: dict = {}
    if not order_ids:
        return statuses
    pending = set(order_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for oid in list(pending):
            st = _status_str(client.get_order_by_id(oid))
            if st in _TERMINAL_STATES:
                statuses[oid] = st
                pending.discard(oid)
        if pending:
            sleep(poll_s)
    for oid in pending:
        statuses[oid] = "timeout"
        print(f"  WARNING: order {oid} did not reach a terminal state within {timeout_s:.0f}s")
    return statuses


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


def get_account_snapshot(net_deposits: float = 100_000.0) -> Optional[dict]:
    """Mark-to-market account snapshot for performance accounting.

    Returns None if credentials are unset. realized_to_date is derived as
    (total P&L − open unrealized): equity already nets cash + holdings, so
    total P&L is equity − net_deposits, and subtracting open unrealized leaves
    profit already booked into cash. It's an identity, not a per-trade ledger.
    """
    client = _get_client()
    if not client:
        return None

    account = client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    long_market_value = float(account.long_market_value or 0.0)
    unrealized_pl = sum(float(p.unrealized_pl) for p in client.get_all_positions())

    total_pnl = equity - net_deposits
    return {
        "equity": equity,
        "cash": cash,
        "long_market_value": long_market_value,
        "unrealized_pl": unrealized_pl,
        "realized_to_date": total_pnl - unrealized_pl,
        "net_deposits": net_deposits,
        "total_return_pct": (total_pnl / net_deposits * 100) if net_deposits else 0.0,
    }


def execute_sells(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list:
    """Friday leg: trim/close positions down to (1-cash_buffer)-scaled targets."""
    if client is None:
        client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping sells")
        return []

    # Defense-in-depth: an empty target would mark every held name "absent" and
    # close the entire book. Refuse, independent of the caller's own guard.
    if not target_weights:
        print("  WARNING: empty target_weights — refusing to sell (would liquidate the book)")
        return []

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    scale = 1.0 - cash_buffer
    targets = {t: w * portfolio_value * scale for t, w in target_weights.items()}

    order_ids = []
    for p in client.get_all_positions():
        sym = p.symbol
        current = float(p.market_value)
        target_val = targets.get(sym, 0.0)
        excess = current - target_val
        if excess <= _MIN_ORDER_VALUE:
            continue
        try:
            if sym not in target_weights:
                order = client.close_position(sym)
            else:
                order = client.submit_order(MarketOrderRequest(
                    symbol=sym, notional=round(excess, 2),
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            oid = getattr(order, "id", None)
            if oid:
                order_ids.append(oid)
            time.sleep(_ORDER_DELAY_S)
        except Exception as e:
            print(f"  WARNING: sell failed for {sym}: {e}")

    wait_for_fills(client, order_ids)
    print(f"  Sells complete | {len(order_ids)} orders submitted")
    return order_ids


def execute_buys(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list:
    """Monday leg: buy underweight names toward (1-cash_buffer)-scaled targets, cash-capped."""
    if client is None:
        client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping buys")
        return []

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    remaining_cash = float(account.cash)
    scale = 1.0 - cash_buffer
    targets = {t: w * portfolio_value * scale for t, w in target_weights.items()}
    current = {p.symbol: float(p.market_value) for p in client.get_all_positions()}

    order_ids = []
    for sym, target_val in targets.items():
        deficit = target_val - current.get(sym, 0.0)
        if deficit <= _MIN_ORDER_VALUE:
            continue
        notional = min(deficit, remaining_cash)
        if notional < _MIN_ORDER_VALUE:
            continue
        try:
            order = client.submit_order(MarketOrderRequest(
                symbol=sym, notional=round(notional, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            remaining_cash -= notional
            oid = getattr(order, "id", None)
            if oid:
                order_ids.append(oid)
            time.sleep(_ORDER_DELAY_S)
        except Exception as e:
            print(f"  WARNING: buy failed for {sym}: {e}")

    wait_for_fills(client, order_ids)
    print(f"  Buys complete | {len(order_ids)} orders submitted")
    return order_ids


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
