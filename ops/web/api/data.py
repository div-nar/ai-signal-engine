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
HERE = os.path.dirname(os.path.abspath(__file__))


def _target():
    for p in (os.path.join(HERE, "..", "targets.json"), os.path.join(HERE, "targets.json")):
        try:
            with open(p) as f:
                return json.load(f)
        except OSError:
            continue
    return {"id": None, "weights": {}, "regime": None, "computed_at": None}


def _benchmarks(history):
    """SPY & QQQ normalized to the account's $100k start, forward-filled onto the
    portfolio's own dates — so the page can plot the alpha and headline vs-QQQ."""
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
        series, last = [], NET_DEPOSITS
        for h in history:
            if h["date"] in closes:
                last = closes[h["date"]] / first * NET_DEPOSITS
            series.append(round(last, 2))
        out[sym] = {"equity": series,
                    "return_pct": (series[-1] - NET_DEPOSITS) / NET_DEPOSITS * 100}
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

    return {
        "generated_at": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "migration": MIGRATION,
        "history": history,
        "benchmarks": _benchmarks(history),
        "layercake": layercake,
        "target": {"id": tgt.get("id"), "regime": tgt.get("regime"),
                   "computed_at": tgt.get("computed_at")},
        "live": {
            "ok": True, "market_open": bool(clock.is_open),
            "equity": eq, "cash": float(acct.cash),
            "long_market_value": float(acct.long_market_value or 0.0),
            "last_equity": float(acct.last_equity),
            "net_deposits": NET_DEPOSITS,
            "total_return_pct": (eq - NET_DEPOSITS) / NET_DEPOSITS * 100,
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
