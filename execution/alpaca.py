import os
import time

from alpaca.trading.client import TradingClient        # module-level so tests patch execution.alpaca.TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import MAX_GROSS_EXPOSURE

_MIN_ORDER_VALUE = 500.0   # skip orders smaller than this dollar amount
_ORDER_DELAY_S = 0.3       # delay between order submissions


def _get_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return TradingClient(api_key, secret_key, paper=True)


def get_alpaca_positions() -> dict:
    """Return current paper portfolio as {longs, shorts, net_exposure, gross_exposure, portfolio_value}."""
    client = _get_client()
    if not client:
        return {"longs": {}, "shorts": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}

    try:
        account = client.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            return {"longs": {}, "shorts": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}

        positions = client.get_all_positions()
        longs, shorts = {}, {}
        for p in positions:
            weight = abs(float(p.market_value)) / portfolio_value
            if float(p.market_value) >= 0:
                longs[p.symbol] = weight
            else:
                shorts[p.symbol] = weight

        net = sum(longs.values()) - sum(shorts.values())
        gross = sum(longs.values()) + sum(shorts.values())
        return {
            "longs": longs,
            "shorts": shorts,
            "net_exposure": net,
            "gross_exposure": gross,
            "portfolio_value": portfolio_value,
        }
    except Exception as e:
        print(f"  WARNING: Could not fetch Alpaca positions: {e}")
        return {"longs": {}, "shorts": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}


def rebalance(long_weights: dict, short_weights: dict, net_exposure_target: float) -> None:
    """Rebalance Alpaca paper account to target long/short weights.

    Both long_weights and short_weights are expected to sum to 1.0.
    Dollar notional for each book is derived from net_exposure_target and portfolio value.
    """
    client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping rebalance")
        return

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)

    # Compute per-book notional from net_exposure_target
    long_notional = portfolio_value * (MAX_GROSS_EXPOSURE + net_exposure_target) / 2
    short_notional = long_notional - portfolio_value * net_exposure_target
    gross = (long_notional + short_notional) / portfolio_value

    if gross > MAX_GROSS_EXPOSURE + 0.01:
        print(f"  ERROR: Computed gross exposure {gross:.2f} exceeds cap {MAX_GROSS_EXPOSURE} — aborting rebalance")
        return

    # Target dollar amounts per position
    long_targets = {t: w * long_notional for t, w in long_weights.items()}
    short_targets = {t: w * short_notional for t, w in short_weights.items()}

    current = client.get_all_positions()
    current_longs = {p.symbol: float(p.market_value) for p in current if float(p.market_value) >= 0}
    current_shorts = {p.symbol: abs(float(p.market_value)) for p in current if float(p.market_value) < 0}

    def _submit(symbol, side, notional):
        if notional < _MIN_ORDER_VALUE:
            return
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        client.submit_order(req)
        time.sleep(_ORDER_DELAY_S)

    # Step 1: Close stale positions not in either target book
    for sym, val in current_longs.items():
        if sym not in long_targets:
            _submit(sym, OrderSide.SELL, val)
    for sym, val in current_shorts.items():
        if sym not in short_targets:
            _submit(sym, OrderSide.BUY, val)

    # Step 2: Close any side conflicts (currently long but now short, and vice versa)
    for sym in short_targets:
        if sym in current_longs:
            _submit(sym, OrderSide.SELL, current_longs[sym])
    for sym in long_targets:
        if sym in current_shorts:
            _submit(sym, OrderSide.BUY, current_shorts[sym])

    # Step 3: Open/adjust longs
    for sym, target_val in long_targets.items():
        current_val = current_longs.get(sym, 0.0)
        diff = target_val - current_val
        if diff > _MIN_ORDER_VALUE:
            _submit(sym, OrderSide.BUY, diff)
        elif diff < -_MIN_ORDER_VALUE:
            _submit(sym, OrderSide.SELL, abs(diff))

    # Step 4: Open/adjust shorts
    for sym, target_val in short_targets.items():
        current_val = current_shorts.get(sym, 0.0)
        diff = target_val - current_val
        if diff > _MIN_ORDER_VALUE:
            _submit(sym, OrderSide.SELL, diff)
        elif diff < -_MIN_ORDER_VALUE:
            _submit(sym, OrderSide.BUY, abs(diff))

    net_actual = long_notional - short_notional
    print(f"  Rebalance complete | long_notional=${long_notional:,.0f} | short_notional=${short_notional:,.0f} | net=${net_actual:,.0f}")
