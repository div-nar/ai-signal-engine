"""Macro + earnings context: best-effort, never raises; renders in the prompt."""
import datetime as dt
import ingestion.macro as macro
from scoring.thesis_scorer import _format_portfolio_context


def test_fetch_macro_never_raises(monkeypatch):
    # simulate yfinance being unavailable -> empty dict, no exception
    monkeypatch.setattr(macro, "fetch_macro_indicators", macro.fetch_macro_indicators)
    import builtins
    real_import = builtins.__import__

    def no_yf(name, *a, **k):
        if name == "yfinance":
            raise ImportError("no yfinance")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_yf)
    assert macro.fetch_macro_indicators() == {}
    assert macro.upcoming_earnings(["NVDA"]) == []


def test_context_renders_macro_and_earnings():
    ctx = {
        "positions": {"NVDA": 0.1},
        "macro": {"VIX": {"value": 14.7, "chg_5d_pct": -1.4},
                  "10Y_yield": {"value": 4.64, "chg_5d_pct": -0.4}},
        "earnings": [{"ticker": "NVDA", "date": "2026-08-27"}],
    }
    out = _format_portfolio_context(ctx)
    assert "MACRO" in out and "VIX 14.7" in out
    assert "EARNINGS WITHIN 14 DAYS" in out and "NVDA 2026-08-27" in out
