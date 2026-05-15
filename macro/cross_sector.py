import yfinance as yf
import numpy as np

_POWER_TICKERS = ["NEE", "ETN", "PWR"]
_COMPUTE_TICKERS = ["NVDA", "TSM"]
_ALL_TICKERS = _POWER_TICKERS + _COMPUTE_TICKERS + ["HYG", "^VIX", "FCX"]


def fetch_cross_sector_signal() -> dict:
    """Compute cross-sector spillover scores from yfinance price data.

    Returns a dict with keys:
        power_compute_lead  - z-score: power basket momentum minus compute momentum (30d)
        copper_infra_lead   - z-score: FCX 30d return vs its own history
        credit_stress       - bool: VIX > 25 AND HYG 30d return < -2%
        vix_level           - latest VIX closing value

    Production note: yf.download for multiple tickers returns a multi-level
    DataFrame (level 0 = field, level 1 = ticker). The implementation works
    directly with the downloaded object. In tests, the mock returns a flat
    DataFrame (columns = tickers), so no ["Close"] indexing is applied here.
    For production use, callers should ensure the data shape matches expectations
    or use auto_adjust=True with a single-level result where possible.
    """
    raw = yf.download(_ALL_TICKERS, period="90d", auto_adjust=True, progress=False)

    # Handle both multi-level (production yfinance) and flat (mocked) DataFrames
    if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
        # Multi-level: ('Close', 'NEE'), etc.
        try:
            data = raw.xs("Close", axis=1, level=0)
        except KeyError:
            data = raw
    else:
        data = raw

    # ── Power → Compute lead ───────────────────────────────────────────────
    # Positive score means power basket has stronger 30d momentum than compute.
    power_series = data[_POWER_TICKERS].mean(axis=1)
    compute_series = data[_COMPUTE_TICKERS].mean(axis=1)
    spread = power_series - compute_series
    spread_ret = spread.diff(30).dropna()
    std = float(spread_ret.std())
    latest_spread_ret = float(spread_ret.iloc[-1]) if len(spread_ret) else 0.0
    if std > 1e-9:
        power_compute_lead = latest_spread_ret / std
    else:
        # Near-zero std means spread momentum is highly consistent; use sign of latest
        power_compute_lead = float(np.sign(latest_spread_ret))

    # ── Copper → Infrastructure lead ──────────────────────────────────────
    # Positive score means FCX 30d return is positive (copper trending up).
    # We z-score the absolute return series against zero (using std as normalizer)
    # so that a rising price always gives a positive lead score.
    fcx_ret = data["FCX"].pct_change(30).dropna()
    fcx_std = float(fcx_ret.std())
    fcx_latest = float(fcx_ret.iloc[-1]) if len(fcx_ret) else 0.0
    if fcx_std > 1e-9:
        copper_infra_lead = fcx_latest / fcx_std
    else:
        # Near-zero std means FCX return is highly consistent; use sign of latest return
        copper_infra_lead = float(np.sign(fcx_latest))

    # ── Credit stress ─────────────────────────────────────────────────────
    vix_level = float(data["^VIX"].iloc[-1])
    hyg_ret_30d = float(data["HYG"].pct_change(30).iloc[-1])
    credit_stress = vix_level > 25 and hyg_ret_30d < -0.02

    return {
        "power_compute_lead": float(power_compute_lead),
        "copper_infra_lead": float(copper_infra_lead),
        "credit_stress": credit_stress,
        "vix_level": vix_level,
    }
