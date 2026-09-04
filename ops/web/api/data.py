"""Vercel Python serverless function: live layercake portfolio data.

Pulls account, positions, and inception-to-date equity history straight from
Alpaca (no local DB needed), merges in the bundled target snapshot for
target-vs-actual, and returns JSON. Alpaca keys stay server-side (env vars);
an optional DASH_TOKEN gates access via ?token= .

Env vars (set in Vercel project settings):
  ALPACA_API_KEY, ALPACA_SECRET_KEY   — required
  DASH_TOKEN                          — optional; if set, ?token= must match
"""
import datetime as dt
import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

NET_DEPOSITS = 100_000.0
MIGRATION = "2026-06-29"
# Book fully liquidated and rebaselined this date (Claude-CLI thesis LLM cutover) —
# the equity CHART starts here so it isn't dominated by the pre-reset history, and
# the Equity tile's total-return figure is measured from this fixed baseline too.
# CAGR still measures true since-inception performance off NET_DEPOSITS, unaffected.
RESET_DATE = "2026-09-03"
RESET_EQUITY = 108_157.88
HERE = os.path.dirname(os.path.abspath(__file__))


def _target():
    for p in (os.path.join(HERE, "..", "targets.json"), os.path.join(HERE, "targets.json")):
        try:
            with open(p) as f:
                return json.load(f)
        except OSError:
            continue
    return {"id": None, "weights": {}, "regime": None, "computed_at": None}


def _benchmarks(history, start_equity=NET_DEPOSITS):
    """SPY & QQQ normalized to `start_equity` at history[0]'s date, forward-filled
    onto the portfolio's own dates — so the page can plot the alpha and headline
    vs-QQQ. `start_equity` must match whatever the Layercake line itself starts
    at over this same `history` window, or the lines aren't on equal footing."""
    if not history:
        return {}
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    start = dt.date.fromisoformat(history[0]["date"])
    dcli = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    try:
        bars = dcli.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=["SPY", "QQQ"], timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start, dt.time()), feed=DataFeed.IEX)).data
    except Exception as e:
        return {"error": str(e)[:80]}
    out = {}
    for sym in ("SPY", "QQQ"):
        closes = {str(b.timestamp)[:10]: float(b.close) for b in (bars.get(sym) or [])}
        if not closes:
            continue
        first = closes[min(closes)]
        series, last = [], start_equity
        for h in history:
            if h["date"] in closes:
                last = closes[h["date"]] / first * start_equity
            series.append(round(last, 2))
        out[sym] = {"equity": series,
                    "return_pct": (series[-1] - start_equity) / start_equity * 100}
    return out


def _recent_trades(client, limit=12):
    """Last filled orders from Alpaca — the live trade log."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=60))
    except Exception:
        return []
    out = []
    for o in orders:
        if str(getattr(o, "status", "")).split(".")[-1] != "FILLED":
            continue
        out.append({
            "symbol": o.symbol,
            "side": str(o.side).split(".")[-1].lower(),
            "notional": float(o.notional) if o.notional else (
                float(o.filled_avg_price) * float(o.filled_qty)
                if o.filled_avg_price and o.filled_qty else None),
            "at": o.filled_at.isoformat() if getattr(o, "filled_at", None) else None,
        })
        if len(out) >= limit:
            break
    return out


def _build():
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    c = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)

    acct = c.get_account()
    clock = c.get_clock()
    eq = float(acct.equity)
    pv = float(acct.portfolio_value)
    tgt = _target()
    tw = tgt.get("weights", {})

    positions = []
    for p in c.get_all_positions():
        mv = float(p.market_value)
        positions.append({
            "symbol": p.symbol, "mv": mv, "weight": mv / pv if pv else 0.0,
            "target": tw.get(p.symbol, 0.0),
            "unrealized_pl": float(p.unrealized_pl),
            "intraday_pl": float(p.unrealized_intraday_pl),
        })
    positions.sort(key=lambda r: -r["mv"])

    h = c.get_portfolio_history(GetPortfolioHistoryRequest(period="all", timeframe="1D"))
    by_day = {}
    for t, e in zip(h.timestamp, h.equity):
        if e is not None:
            by_day[dt.datetime.fromtimestamp(t).date().isoformat()] = float(e)
    by_day[dt.date.today().isoformat()] = eq  # freshest point
    history = [{"date": d, "equity": by_day[d],
                "return_pct": (by_day[d] - NET_DEPOSITS) / NET_DEPOSITS * 100}
               for d in sorted(by_day)]

    base = next((x for x in reversed(history) if x["date"] < MIGRATION), None)
    layercake = None
    if base:
        layercake = {"since": MIGRATION, "base_equity": base["equity"],
                     "gain": eq - base["equity"],
                     "pct": (eq - base["equity"]) / base["equity"] * 100}

    # Chart shows only since the reset (visual, matches the liquidation) — CAGR
    # below keeps using the full `history` (true since-inception, unaffected).
    chart_history = [x for x in history if x["date"] >= RESET_DATE] or history[-1:]

    return {
        "generated_at": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "migration": MIGRATION,
        "reset_date": RESET_DATE,
        "history": history,
        "chart_history": chart_history,
        "benchmarks": _benchmarks(history),
        # Same starting dollar amount as the Layercake line itself over this
        # window, so SPY/QQQ/Layercake are all "if you'd put $X in at t=0".
        "chart_benchmarks": _benchmarks(chart_history, start_equity=chart_history[0]["equity"]),
        "layercake": layercake,
        "target": {"id": tgt.get("id"), "regime": tgt.get("regime"),
                   "computed_at": tgt.get("computed_at"),
                   "urgency": tgt.get("urgency", ""), "trade_gate": tgt.get("trade_gate", ""),
                   "thesis": tgt.get("thesis", ""),
                   "weights": tw},  # full intended book — dashboard shows it even when unfilled
        "recent_trades": _recent_trades(c),
        "live": {
            "ok": True, "market_open": bool(clock.is_open),
            "equity": eq, "cash": float(acct.cash),
            "long_market_value": float(acct.long_market_value or 0.0),
            "last_equity": float(acct.last_equity),
            "net_deposits": NET_DEPOSITS,
            "total_return_pct": (eq - RESET_EQUITY) / RESET_EQUITY * 100,
            "positions": positions,
        },
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        token = os.environ.get("DASH_TOKEN")
        if token:
            q = parse_qs(urlparse(self.path).query)
            if q.get("token", [""])[0] != token:
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
                return
        try:
            body = json.dumps(_build()).encode()
            code = 200
        except Exception as e:
            body = json.dumps({"live": {"ok": False, "error": str(e)}}).encode()
            code = 200
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
