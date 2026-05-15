import os
import yfinance as yf
import pandas as pd
from fredapi import Fred  # must be at module level so tests can patch macro.supply_chain.Fred

_DEFAULT_SIGNAL = {
    "shipping_pressure": 0.5,
    "semis_inventory_trend": "neutral",
    "pmi": 50.0,
    "pmi_trend": "stable",
}


def fetch_supply_chain_signal() -> dict:
    """Fetch supply-chain lead indicators from FRED and yfinance.

    Returns default neutral signal if FRED_API_KEY is not set.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("  WARNING: FRED_API_KEY not set — using neutral supply chain defaults")
        return dict(_DEFAULT_SIGNAL)

    fred = Fred(api_key=api_key)

    # ── PMI ────────────────────────────────────────────────────────────────
    pmi_series = fred.get_series("NAPM", observation_start="2024-01-01")
    latest_pmi = float(pmi_series.iloc[-1])
    prev_pmi = float(pmi_series.iloc[-2]) if len(pmi_series) >= 2 else latest_pmi

    if latest_pmi >= 50:
        pmi_trend = "expanding"
    elif latest_pmi < 50 and latest_pmi < prev_pmi:
        pmi_trend = "contracting"
    else:
        pmi_trend = "stable"

    # ── Semiconductor industrial production ────────────────────────────────
    semis_series = fred.get_series("IPG3344S", observation_start="2024-01-01")
    semis_latest = float(semis_series.iloc[-1])
    semis_prev = float(semis_series.iloc[-2]) if len(semis_series) >= 2 else semis_latest

    if semis_latest < semis_prev:
        semis_inventory_trend = "drawing_down"
    elif semis_latest > semis_prev:
        semis_inventory_trend = "building"
    else:
        semis_inventory_trend = "neutral"

    # ── Shipping proxy: BDRY ETF 30-day momentum ──────────────────────────
    bdry = yf.download("BDRY", period="65d", auto_adjust=True, progress=False)["Close"]
    if len(bdry) < 31:
        shipping_pressure = 0.5
    else:
        ret_30d = float((bdry.iloc[-1] - bdry.iloc[-31]) / bdry.iloc[-31])
        # Map [-10%, +20%] → [0.0, 1.0]
        shipping_pressure = float(min(max((ret_30d + 0.10) / 0.30, 0.0), 1.0))

    return {
        "shipping_pressure": shipping_pressure,
        "semis_inventory_trend": semis_inventory_trend,
        "pmi": latest_pmi,
        "pmi_trend": pmi_trend,
    }
