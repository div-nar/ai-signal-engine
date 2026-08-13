"""Structured macro + earnings context (keyless, via yfinance).

Feeds the thesis pass hard numbers — rates, vol, dollar, commodities, and
upcoming earnings for held names — alongside the news it reads. Every call is
best-effort: yfinance is flaky, and a macro hiccup must never break a run.
"""
import datetime as dt
import warnings

_MACRO = {
    "10Y_yield": "^TNX",
    "VIX": "^VIX",
    "dollar_DXY": "DX-Y.NYB",
    "oil_WTI": "CL=F",
    "gold": "GC=F",
}


def fetch_macro_indicators() -> dict:
    """Latest level + 5-day % change for a handful of macro series."""
    warnings.filterwarnings("ignore")
    try:
        import yfinance as yf
    except Exception:
        return {}
    out = {}
    for name, ticker in _MACRO.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            if len(h):
                last = float(h["Close"].iloc[-1])
                chg = (last / float(h["Close"].iloc[0]) - 1) * 100
                out[name] = {"value": round(last, 2), "chg_5d_pct": round(chg, 2)}
        except Exception:
            continue
    return out


def upcoming_earnings(tickers: list[str], within_days: int = 14) -> list[dict]:
    """Held names reporting within `within_days` — earnings is event risk the
    LLM should weigh before sizing into a name."""
    warnings.filterwarnings("ignore")
    try:
        import yfinance as yf
    except Exception:
        return []
    today = dt.date.today()
    horizon = today + dt.timedelta(days=within_days)
    out = []
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            for d in (dates or []):
                if isinstance(d, dt.datetime):
                    d = d.date()
                if today <= d <= horizon:
                    out.append({"ticker": t, "date": d.isoformat()})
                    break
        except Exception:
            continue
    return sorted(out, key=lambda r: r["date"])
