import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SIGNAL_NAMES = [
    "shipping_pressure",
    "copper_infra_lead_neg",
    "power_compute_lead_neg",
    "vix_level",
    "pmi_neg",
]

_POWER_TICKERS = ["NEE", "ETN", "PWR"]
_COMPUTE_TICKERS = ["NVDA", "TSM"]


def _build_signal_history(lookback: int = 90, fred_api_key: Optional[str] = None) -> pd.DataFrame:
    """Pull daily time series for all 5 signals. Returns (lookback, 5) DataFrame."""
    period = f"{lookback + 40}d"

    tickers = ["BDRY", "FCX", "^VIX"] + _POWER_TICKERS + _COMPUTE_TICKERS
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    try:
        prices = raw.xs("Close", axis=1, level=0)
    except (KeyError, TypeError):
        prices = raw

    # shipping_pressure: BDRY 30d rolling return → [0,1]
    bdry = prices["BDRY"].dropna()
    sp = ((bdry - bdry.shift(30)) / bdry.shift(30) + 0.10) / 0.30
    shipping = sp.clip(0.0, 1.0)

    # copper_infra_lead: FCX rolling 30d return z-score (negated)
    fcx_ret = prices["FCX"].pct_change(30)
    copper_z = (fcx_ret - fcx_ret.rolling(60).mean()) / (fcx_ret.rolling(60).std() + 1e-9)
    copper_neg = -copper_z

    # power_compute_lead: (power - compute) 30d spread z-score (negated)
    power_ser = prices[_POWER_TICKERS].mean(axis=1)
    compute_ser = prices[_COMPUTE_TICKERS].mean(axis=1)
    spread_ret = (power_ser - compute_ser).diff(30)
    pcl_z = (spread_ret - spread_ret.rolling(60).mean()) / (spread_ret.rolling(60).std() + 1e-9)
    pcl_neg = -pcl_z

    # vix_level: raw VIX
    vix = prices["^VIX"]

    # pmi_neg: FRED DGORDER, monthly forward-filled to daily (negated)
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if api_key:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        dgorder = fred.get_series("DGORDER", observation_start="2024-01-01")
        pmi_raw = (50.0 + (dgorder.pct_change() * 500)).clip(30.0, 70.0)
        # DGORDER is monthly and lags ~2 months. resample("D") only spans up to
        # the last print, so reindex onto the price calendar and forward-fill the
        # last known release through today — otherwise recent dates are all-NaN
        # and the .dropna() below wipes the entire frame (empty PCA fit → crash).
        pmi_daily = (
            pmi_raw.resample("D").interpolate("linear").reindex(vix.index, method="ffill")
        )
    else:
        pmi_daily = pd.Series(50.0, index=vix.index)

    df = pd.DataFrame({
        "shipping_pressure": shipping,
        "copper_infra_lead_neg": copper_neg,
        "power_compute_lead_neg": pcl_neg,
        "vix_level": vix,
        "pmi_neg": -pmi_daily,
    }).dropna()

    return df.iloc[-lookback:]


def fit_and_cache_composite(
    cache_path: str,
    lookback: int = 90,
    fred_api_key: Optional[str] = None,
) -> None:
    """Weekly fit: pull 90-day signal history, run PCA, write cache."""
    history = _build_signal_history(lookback=lookback, fred_api_key=fred_api_key)

    scaler = StandardScaler()
    X = scaler.fit_transform(history.values)

    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = pca.components_[0].copy()

    # Orient so VIX loading is always positive (stress direction)
    vix_idx = _SIGNAL_NAMES.index("vix_level")
    if loadings[vix_idx] < 0:
        loadings = -loadings

    pc1_scores = X @ loadings

    cache = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "signal_names": _SIGNAL_NAMES,
        "pc1_loadings": loadings.tolist(),
        "history_mean": scaler.mean_.tolist(),
        "history_std": scaler.scale_.tolist(),
        "pc1_history_scores": pc1_scores.tolist(),
    }
    Path(cache_path).write_text(json.dumps(cache))


def load_composite_modifier(
    cache_path: str,
    supply: dict,
    cross: dict,
) -> tuple[float, dict]:
    """Daily apply: project current signals onto cached PC1 → modifier ∈ [-0.25, 0].
    Returns (modifier, info_dict).
    """
    if not Path(cache_path).exists():
        return 0.0, {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": None}

    cache = json.loads(Path(cache_path).read_text())
    loadings = np.array(cache["pc1_loadings"])
    mean = np.array(cache["history_mean"])
    std = np.array(cache["history_std"])
    history_scores = np.array(cache["pc1_history_scores"])

    computed_at = datetime.fromisoformat(cache["computed_at"])
    cache_age_days = (datetime.now(timezone.utc) - computed_at).days

    sp = supply.get("shipping_pressure", 0.5)
    copper_neg = -cross.get("copper_infra_lead", 0.0)
    pcl_neg = -cross.get("power_compute_lead", 0.0)
    vix = cross.get("vix_level", 16.0)
    pmi_neg = -(supply.get("pmi", 50.0) - 50.0) / 10.0

    current = np.array([sp, copper_neg, pcl_neg, vix, pmi_neg])
    current_std = (current - mean) / (std + 1e-9)
    score = float(current_std @ loadings)

    stress_score = float(np.mean(history_scores <= score))
    modifier = round(-0.25 * stress_score, 4)

    return modifier, {
        "stress_score": round(stress_score, 4),
        "modifier": modifier,
        "cache_age_days": cache_age_days,
    }


def is_cache_stale(cache_path: str, max_age_days: int = 8) -> bool:
    if not Path(cache_path).exists():
        return True
    try:
        data = json.loads(Path(cache_path).read_text())
        computed_at = datetime.fromisoformat(data["computed_at"])
        age = (datetime.now(timezone.utc) - computed_at).days
        return age > max_age_days
    except Exception:
        return True
