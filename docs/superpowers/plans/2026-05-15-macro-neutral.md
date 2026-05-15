# Macro Module + Market-Neutral Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add supply-chain lead indicators, cross-sector spillover detection, and two-sided Alpaca execution to reduce portfolio beta from ~1.45 to ~η_target × 1.45.

**Architecture:** A new `macro/` module fetches FRED + yfinance data and computes a `MacroSignal` dict (regime label + `net_exposure_target`). This is injected as a structured block into the Gemini prompt before document context. Gemini outputs both a long book and a short book (weights each summing to 1.0). A new `execution/alpaca.py` sizes each book in dollar terms from `net_exposure_target` and places market orders via Alpaca paper account.

**Tech Stack:** `fredapi`, `yfinance` (new); `alpaca-py`, `google-genai`, `sqlite3`, `pytest` (existing)

---

## File Map

**Create:**
- `macro/__init__.py`
- `macro/supply_chain.py` — FRED + yfinance supply chain signal
- `macro/cross_sector.py` — power/copper/credit spillover signal
- `macro/regime.py` — combines into MacroSignal, computes net_exposure_target
- `execution/__init__.py`
- `execution/alpaca.py` — reads positions + two-sided rebalance
- `tests/test_supply_chain.py`
- `tests/test_cross_sector.py`
- `tests/test_regime.py`
- `tests/test_alpaca_execution.py`

**Modify:**
- `requirements.txt` — add `fredapi`, `yfinance`
- `config.py` — add `MAX_SHORT_WEIGHT = 0.08`, `MAX_GROSS_EXPOSURE = 1.80`
- `db.py` — add `short_weights TEXT`, `macro_signal TEXT` columns; update `insert_signal`
- `scoring/gemini_scorer.py` — macro block injection, long/short output schema, updated guardrails
- `main.py` — call macro module, pass MacroSignal to scorer, call executor
- `export.py` — add `short_weights` and `macro_signal` to `stock_signals.json` output
- `tests/test_db.py` — extend for new columns
- `tests/test_gemini_scorer.py` — extend for new schema

---

## Task 1: Dependencies + Config + DB Migration

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`
- Modify: `db.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

```text
feedparser==6.0.11
arxiv==2.1.3
httpx==0.28.1
google-genai==1.9.0
pytest==8.2.0
pytest-mock==3.14.0
alpaca-py==0.38.0
fredapi==3.1.0
yfinance==0.2.51
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install fredapi yfinance`
Expected: both packages install without error.

- [ ] **Step 3: Add guardrail constants to config.py**

Open `config.py`. After the `MAX_TURNOVER_VS_PREV` line, add:

```python
MAX_SHORT_WEIGHT = 0.08
MAX_GROSS_EXPOSURE = 1.80
```

- [ ] **Step 4: Write failing DB migration test**

Add to `tests/test_db.py` (append to end of file):

```python
def test_signals_table_has_short_weights_column(db_path):
    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
    conn.close()
    assert "short_weights" in cols
    assert "macro_signal" in cols


def test_insert_signal_roundtrip_with_short_weights(db_path):
    signal_id = insert_signal(db_path, {
        "p_final": 0.91,
        "stock_conviction": '{"NVDA": 0.95}',
        "stock_weights": '{"NVDA": 1.0}',
        "stock_reasoning": '{"NVDA": "test"}',
        "short_weights": '{"AMD": 0.6, "QCOM": 0.4}',
        "macro_signal": '{"regime": "shipping_bottleneck", "net_exposure_target": 0.55}',
        "sector_tilt": '{}',
        "supply_demand_balance": 0.3,
        "market_regime": "shipping_bottleneck",
        "signal_confidence": 0.88,
        "thesis_stress": False,
        "signal_age_days": 0,
        "sources_ingested": 50,
        "signal_breakdown": '{}',
        "thesis_update": "test",
    })
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT short_weights, macro_signal FROM signals WHERE id = ?", (signal_id,)
    ).fetchone()
    conn.close()
    assert json.loads(row[0]) == {"AMD": 0.6, "QCOM": 0.4}
    assert json.loads(row[1])["regime"] == "shipping_bottleneck"
```

Add `import json` to the top of `tests/test_db.py` if not already present.

- [ ] **Step 5: Run tests to verify they fail**

Run: `cd /Users/div-nar/sideproj/ai-signal-engine && python -m pytest tests/test_db.py::test_signals_table_has_short_weights_column tests/test_db.py::test_insert_signal_roundtrip_with_short_weights -v`
Expected: FAIL — columns don't exist yet.

- [ ] **Step 6: Add columns to db.py schema and insert_signal**

In `db.py`, in `init_db()`, add two lines to the migration block after the existing `for col in (...)` loop:

```python
    for col in ("stock_weights", "stock_reasoning", "raw_response", "prompt_context_doc_ids",
                "short_weights", "macro_signal"):
        if col not in existing:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} TEXT")
```

Replace the existing `for col in (...)` block entirely with the above.

Then update `insert_signal` — replace the INSERT statement and values tuple:

```python
def insert_signal(db_path: str, data: dict) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO signals
           (p_final, stock_conviction, stock_weights, stock_reasoning,
            sector_tilt, supply_demand_balance, market_regime, signal_confidence,
            thesis_stress, signal_age_days, sources_ingested, signal_breakdown,
            thesis_update, raw_response, prompt_context_doc_ids,
            short_weights, macro_signal)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["p_final"],
            data["stock_conviction"],
            data.get("stock_weights"),
            data.get("stock_reasoning"),
            data["sector_tilt"],
            data["supply_demand_balance"],
            data["market_regime"],
            data["signal_confidence"],
            int(data["thesis_stress"]),
            data["signal_age_days"],
            data["sources_ingested"],
            data["signal_breakdown"],
            data["thesis_update"],
            data.get("raw_response"),
            data.get("prompt_context_doc_ids"),
            data.get("short_weights"),
            data.get("macro_signal"),
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py -v`
Expected: all PASS including the two new tests.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt config.py db.py tests/test_db.py
git commit -m "feat: add short_weights/macro_signal columns, fredapi/yfinance deps"
```

---

## Task 2: macro/supply_chain.py

**Files:**
- Create: `macro/__init__.py`
- Create: `macro/supply_chain.py`
- Create: `tests/test_supply_chain.py`

- [ ] **Step 1: Create macro/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Write failing tests**

Create `tests/test_supply_chain.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_fred_series(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), freq="MS"))


def _make_yf_close(values, ticker="BDRY"):
    idx = pd.date_range("2025-11-01", periods=len(values), freq="B")
    return pd.DataFrame({ticker: values}, index=idx)


def test_fetch_supply_chain_signal_returns_required_keys():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [49.1, 49.5, 50.1, 48.9, 48.5] if s == "NAPM" else [98.0, 97.5, 97.0, 96.5, 96.0]
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 40 + [115] * 20)

        result = fetch_supply_chain_signal()

    assert set(result.keys()) == {"shipping_pressure", "semis_inventory_trend", "pmi", "pmi_trend"}
    assert 0.0 <= result["shipping_pressure"] <= 1.0
    assert result["pmi_trend"] in {"expanding", "contracting", "stable"}
    assert result["semis_inventory_trend"] in {"drawing_down", "building", "neutral"}


def test_pmi_below_50_and_falling_is_contracting():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [51.0, 50.0, 49.0, 48.5, 47.9] if s == "NAPM" else [98.0] * 5
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["pmi_trend"] == "contracting"
    assert result["pmi"] == pytest.approx(47.9)


def test_pmi_above_50_is_expanding():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series([52.0, 53.0, 54.0])
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["pmi_trend"] == "expanding"


def test_high_shipping_momentum_gives_high_pressure():
    from macro.supply_chain import fetch_supply_chain_signal
    # BDRY up 25% over 30 days → shipping_pressure should be 1.0 (capped)
    base = [100.0] * 30
    high = [125.0] * 30
    with patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series([52.0, 53.0, 52.5])
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close(base + high)

        result = fetch_supply_chain_signal()

    assert result["shipping_pressure"] == pytest.approx(1.0)


def test_falling_semis_ip_is_drawing_down():
    from macro.supply_chain import fetch_supply_chain_signal
    with patch("macro.supply_chain.Fred") as mock_fred_cls, \
         patch("macro.supply_chain.yf.download") as mock_dl:
        mock_fred = MagicMock()
        # NAPM stable above 50, IPG3344S falling
        mock_fred.get_series.side_effect = lambda s, **kw: _make_fred_series(
            [52.0, 52.0, 52.0] if s == "NAPM" else [100.0, 99.0, 98.0]
        )
        mock_fred_cls.return_value = mock_fred
        mock_dl.return_value = _make_yf_close([100] * 60)

        result = fetch_supply_chain_signal()

    assert result["semis_inventory_trend"] == "drawing_down"


def test_missing_fred_api_key_returns_default_signal(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from macro.supply_chain import fetch_supply_chain_signal
    result = fetch_supply_chain_signal()
    assert result["pmi_trend"] == "stable"
    assert result["shipping_pressure"] == 0.5
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_supply_chain.py -v`
Expected: FAIL — `macro.supply_chain` not found.

- [ ] **Step 4: Implement macro/supply_chain.py**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_supply_chain.py -v`
Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add macro/__init__.py macro/supply_chain.py tests/test_supply_chain.py
git commit -m "feat: macro/supply_chain — FRED PMI, semis IP, BDRY shipping proxy"
```

---

## Task 3: macro/cross_sector.py

**Files:**
- Create: `macro/cross_sector.py`
- Create: `tests/test_cross_sector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cross_sector.py`:

```python
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


def _make_close_df(tickers, n=90, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-11-01", periods=n, freq="B")
    data = {t: 100 + rng.normal(0, 2, n).cumsum() for t in tickers}
    return pd.DataFrame(data, index=idx)


ALL_TICKERS = ["NEE", "ETN", "PWR", "NVDA", "TSM", "HYG", "^VIX", "FCX"]


def test_fetch_cross_sector_signal_returns_required_keys():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert set(result.keys()) == {"power_compute_lead", "copper_infra_lead", "credit_stress", "vix_level"}


def test_high_vix_and_falling_hyg_triggers_credit_stress():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    # Override VIX to be high, HYG to be falling
    df["^VIX"] = 30.0
    df["HYG"] = [100.0 - i * 0.15 for i in range(len(df))]
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["credit_stress"] is True
    assert result["vix_level"] == pytest.approx(30.0)


def test_low_vix_does_not_trigger_credit_stress():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    df["^VIX"] = 15.0
    df["HYG"] = [100.0 + i * 0.05 for i in range(len(df))]
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["credit_stress"] is False


def test_power_leading_compute_gives_positive_lead_score():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    n = len(df)
    # Power basket rising strongly, compute flat
    for t in ["NEE", "ETN", "PWR"]:
        df[t] = [100.0 + i * 0.5 for i in range(n)]
    for t in ["NVDA", "TSM"]:
        df[t] = 100.0
    df["^VIX"] = 15.0
    df["HYG"] = 100.0
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["power_compute_lead"] > 0


def test_copper_rising_gives_positive_infra_lead():
    from macro.cross_sector import fetch_cross_sector_signal
    df = _make_close_df(ALL_TICKERS)
    n = len(df)
    df["FCX"] = [100.0 + i * 0.4 for i in range(n)]
    df["^VIX"] = 15.0
    df["HYG"] = 100.0
    with patch("macro.cross_sector.yf.download", return_value=df):
        result = fetch_cross_sector_signal()
    assert result["copper_infra_lead"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cross_sector.py -v`
Expected: FAIL — `macro.cross_sector` not found.

- [ ] **Step 3: Implement macro/cross_sector.py**

```python
import yfinance as yf
import numpy as np

_POWER_TICKERS = ["NEE", "ETN", "PWR"]
_COMPUTE_TICKERS = ["NVDA", "TSM"]
_ALL_TICKERS = _POWER_TICKERS + _COMPUTE_TICKERS + ["HYG", "^VIX", "FCX"]


def fetch_cross_sector_signal() -> dict:
    """Compute cross-sector spillover scores from yfinance price data."""
    data = yf.download(_ALL_TICKERS, period="90d", auto_adjust=True, progress=False)["Close"]

    # ── Power → Compute lead ───────────────────────────────────────────────
    power_series = data[_POWER_TICKERS].mean(axis=1)
    compute_series = data[_COMPUTE_TICKERS].mean(axis=1)
    spread = power_series - compute_series
    spread_ret = spread.diff(30).dropna()
    std = float(spread_ret.std())
    latest_spread_ret = float(spread_ret.iloc[-1]) if len(spread_ret) else 0.0
    power_compute_lead = (latest_spread_ret / std) if std > 1e-9 else 0.0

    # ── Copper → Infrastructure lead ──────────────────────────────────────
    fcx_ret = data["FCX"].pct_change(30).dropna()
    fcx_std = float(fcx_ret.std())
    fcx_latest = float(fcx_ret.iloc[-1]) if len(fcx_ret) else 0.0
    fcx_mean = float(fcx_ret.mean())
    copper_infra_lead = ((fcx_latest - fcx_mean) / fcx_std) if fcx_std > 1e-9 else 0.0

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cross_sector.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add macro/cross_sector.py tests/test_cross_sector.py
git commit -m "feat: macro/cross_sector — power/compute lead, copper, credit stress"
```

---

## Task 4: macro/regime.py

**Files:**
- Create: `macro/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_regime.py`:

```python
import pytest
from unittest.mock import patch

_SUPPLY_CLEAN = {
    "shipping_pressure": 0.3,
    "semis_inventory_trend": "neutral",
    "pmi": 52.0,
    "pmi_trend": "expanding",
}
_CROSS_CLEAN = {
    "power_compute_lead": 0.5,
    "copper_infra_lead": 0.2,
    "credit_stress": False,
    "vix_level": 16.0,
}


def test_clean_macro_gives_compute_constrained():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert result["regime"] == "compute_constrained"
    assert result["net_exposure_target"] == pytest.approx(0.80)


def test_credit_stress_overrides_all_other_signals():
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    supply_shipping = {**_SUPPLY_CLEAN, "shipping_pressure": 0.9}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_shipping), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed):
        result = compute_macro_signal()
    assert result["regime"] == "credit_stress"
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_high_shipping_without_credit_stress_gives_shipping_bottleneck():
    from macro.regime import compute_macro_signal
    supply_ship = {**_SUPPLY_CLEAN, "shipping_pressure": 0.80}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_ship), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert result["regime"] == "shipping_bottleneck"
    assert result["net_exposure_target"] == pytest.approx(0.55)


def test_contracting_pmi_and_weak_copper_gives_balanced():
    from macro.regime import compute_macro_signal
    supply_weak = {**_SUPPLY_CLEAN, "pmi": 48.0, "pmi_trend": "contracting"}
    cross_weak = {**_CROSS_CLEAN, "copper_infra_lead": -0.8}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_weak), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_weak):
        result = compute_macro_signal()
    assert result["regime"] == "balanced"
    assert result["net_exposure_target"] == pytest.approx(0.65)


def test_macro_signal_has_all_required_keys():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN):
        result = compute_macro_signal()
    assert set(result.keys()) == {
        "computed_at", "regime", "regime_confidence",
        "net_exposure_target", "supply_chain", "cross_sector", "notes",
    }
    assert 0.0 <= result["regime_confidence"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_regime.py -v`
Expected: FAIL — `macro.regime` not found.

- [ ] **Step 3: Implement macro/regime.py**

```python
from datetime import datetime, timezone

from macro.supply_chain import fetch_supply_chain_signal
from macro.cross_sector import fetch_cross_sector_signal


def compute_macro_signal() -> dict:
    """Combine supply chain and cross-sector signals into a single MacroSignal dict."""
    supply = fetch_supply_chain_signal()
    cross = fetch_cross_sector_signal()

    # Priority-ordered regime rules
    if cross["credit_stress"]:
        regime = "credit_stress"
        net_exposure_target = 0.20
        confidence = 0.90
        notes = (
            f"Credit stress: VIX {cross['vix_level']:.1f} > 25, HYG falling. "
            "Defensive positioning — minimal net long exposure."
        )
    elif supply["shipping_pressure"] > 0.65:
        regime = "shipping_bottleneck"
        net_exposure_target = 0.55
        confidence = 0.75
        notes = (
            f"Shipping pressure {supply['shipping_pressure']:.2f} elevated. "
            "Reduce net exposure, rotate toward supply bottleneck names."
        )
    elif supply["pmi_trend"] == "contracting" and cross["copper_infra_lead"] < -0.5:
        regime = "balanced"
        net_exposure_target = 0.65
        confidence = 0.70
        notes = (
            f"PMI contracting ({supply['pmi']:.1f}) and copper weak "
            f"({cross['copper_infra_lead']:.2f}σ). Cautious positioning."
        )
    else:
        regime = "compute_constrained"
        net_exposure_target = 0.80
        confidence = 0.85
        notes = (
            f"PMI {supply['pmi']:.1f} ({supply['pmi_trend']}), credit clean, "
            "compute thesis intact. Full net long exposure."
        )

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_confidence": confidence,
        "net_exposure_target": net_exposure_target,
        "supply_chain": supply,
        "cross_sector": cross,
        "notes": notes,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_regime.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add macro/regime.py tests/test_regime.py
git commit -m "feat: macro/regime — MacroSignal with priority-ordered regime rules"
```

---

## Task 5: gemini_scorer.py — Macro Injection + Long/Short Schema

**Files:**
- Modify: `scoring/gemini_scorer.py`
- Modify: `tests/test_gemini_scorer.py`

- [ ] **Step 1: Write failing tests for new scorer behaviour**

Append to `tests/test_gemini_scorer.py`:

```python
_MACRO_SIGNAL = {
    "regime": "shipping_bottleneck",
    "regime_confidence": 0.82,
    "net_exposure_target": 0.55,
    "supply_chain": {
        "shipping_pressure": 0.74, "semis_inventory_trend": "drawing_down",
        "pmi": 49.2, "pmi_trend": "contracting",
    },
    "cross_sector": {
        "power_compute_lead": 1.3, "copper_infra_lead": -0.2,
        "credit_stress": False, "vix_level": 18.4,
    },
    "notes": "Freight pressure elevated but credit clean.",
}

VALID_GEMINI_OUTPUT_V2 = {
    "p_score": 0.91,
    "market_regime": "shipping_bottleneck",
    "supply_demand_balance": 0.3,
    "portfolio": [
        {"ticker": "NVDA", "weight": 0.15, "conviction": 0.95, "reasoning": "GPU supply tight."},
        {"ticker": "MU",   "weight": 0.12, "conviction": 0.90, "reasoning": "HBM demand rising."},
        {"ticker": "TSM",  "weight": 0.10, "conviction": 0.88, "reasoning": "Advanced node full."},
        {"ticker": "VRT",  "weight": 0.10, "conviction": 0.85, "reasoning": "Cooling bottleneck."},
        {"ticker": "CEG",  "weight": 0.10, "conviction": 0.83, "reasoning": "Power contracts."},
        {"ticker": "AVGO", "weight": 0.09, "conviction": 0.80, "reasoning": "Custom ASIC ramp."},
        {"ticker": "AMZN", "weight": 0.08, "conviction": 0.78, "reasoning": "AWS capex cycle."},
        {"ticker": "MSFT", "weight": 0.08, "conviction": 0.75, "reasoning": "Azure AI revenue."},
        {"ticker": "META", "weight": 0.08, "conviction": 0.72, "reasoning": "Llama infra spend."},
        {"ticker": "PWR",  "weight": 0.10, "conviction": 0.70, "reasoning": "Grid build."},
    ],
    "short_portfolio": [
        {"ticker": "AMD",  "weight": 0.40, "conviction": 0.42, "reasoning": "Lower GPU conviction vs NVDA."},
        {"ticker": "QCOM", "weight": 0.35, "conviction": 0.38, "reasoning": "Mobile-heavy, low AI infra exposure."},
        {"ticker": "ON",   "weight": 0.25, "conviction": 0.35, "reasoning": "EV exposure, not AI buildout."},
    ],
    "net_exposure": 0.54,
    "signal_confidence": 0.88,
    "thesis_stress": False,
    "thesis_update": "Shipping elevated — rotate to bottleneck names.",
}


def test_build_signal_context_includes_macro_block():
    context = build_signal_context(SAMPLE_DOCS, macro_signal=_MACRO_SIGNAL)
    assert "MACRO REGIME SIGNAL" in context
    assert "shipping_bottleneck" in context
    assert "net_exposure_target: 0.55" in context
    assert "Freight pressure elevated" in context


def test_build_signal_context_macro_precedes_documents():
    context = build_signal_context(SAMPLE_DOCS, macro_signal=_MACRO_SIGNAL)
    macro_pos = context.index("MACRO REGIME SIGNAL")
    doc_pos = context.index("ASML Q1 Backlog Surge")
    assert macro_pos < doc_pos


def test_apply_guardrails_caps_short_weight():
    output = dict(VALID_GEMINI_OUTPUT_V2)
    output["short_portfolio"] = [
        {"ticker": "AMD", "weight": 0.50, "conviction": 0.4, "reasoning": "x"},
    ]
    guarded = apply_guardrails(output, prev_weights={})
    amd = next(p for p in guarded["short_portfolio"] if p["ticker"] == "AMD")
    assert amd["weight"] <= 0.08


def test_apply_guardrails_short_book_sums_to_one():
    output = dict(VALID_GEMINI_OUTPUT_V2)
    guarded = apply_guardrails(output, prev_weights={})
    total = sum(p["weight"] for p in guarded["short_portfolio"])
    assert abs(total - 1.0) < 0.01


def test_score_documents_returns_short_weights(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_GEMINI_OUTPUT_V2)

    with patch("scoring.gemini_scorer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = score_documents(
            docs=SAMPLE_DOCS, db_path=db_path, prev_weights={},
            macro_signal=_MACRO_SIGNAL,
        )

    assert result["short_weights"] is not None
    short = json.loads(result["short_weights"])
    assert "AMD" in short
    assert abs(sum(short.values()) - 1.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gemini_scorer.py::test_build_signal_context_includes_macro_block tests/test_gemini_scorer.py::test_apply_guardrails_caps_short_weight tests/test_gemini_scorer.py::test_score_documents_returns_short_weights -v`
Expected: FAIL.

- [ ] **Step 3: Update _SYSTEM_PROMPT in gemini_scorer.py**

Replace the `_SYSTEM_PROMPT` string. The new version adds a `short_portfolio` field to the output schema and instructs Gemini to respect the net_exposure_target from the macro block:

```python
_SYSTEM_PROMPT = """You are a portfolio manager for an AI infrastructure long/short equity fund.
Your thesis is Aschenbrenner's: AI is on an exponential compute trajectory toward AGI,
driven by a physical buildout supercycle — chips, memory, power, cooling, datacenters, networking.
Your job is to identify which stocks will benefit most from the NEXT 1-4 quarters of AI
infrastructure spending: capex commitments, supply chain bottlenecks, and compute expansion.

Focus on BOTH sides of the AI buildout — supply bottlenecks AND demand-side compute consumers:

SUPPLY-SIDE (physical bottlenecks the buildout is constrained by):
- Compute: GPUs, ASICs, networking chips, foundry, HBM memory, advanced packaging
- Power & cooling: utilities, thermal management, grid equipment, power electronics
- Infrastructure: datacenter REITs, interconnects, fiber, structural components

DEMAND-SIDE (hyperscalers and platforms whose earnings scale with AI compute consumption):
- Hyperscalers driving the capex cycle: their own AI revenue (Cloud AI, model APIs, AI ad/search uplift)
- AI software platforms whose revenue scales directly with compute consumption
- Companies monetising AI products at scale (Gemini, Copilot, ChatGPT-class APIs, AI ad units)

Universe (you may pick any publicly traded stock globally): {universe}

PORTFOLIO STRUCTURE:
- Long book (portfolio): highest-conviction AI buildout beneficiaries. Weights sum to 1.0. Max 10% per stock.
- Short book (short_portfolio): lowest-conviction names in the same factor space as longs — genuine
  pairs, not random hedges. Example: long NVDA, short AMD (same sector, lower conviction).
  Weights sum to 1.0. Max 8% per stock.
- Net exposure is controlled by the MACRO REGIME SIGNAL above — follow net_exposure_target exactly.
- Every position must be directly tied to AI buildout.

Output ONLY valid JSON matching this schema exactly:
{{
  "p_score": <float 0-1, Aschenbrenner probability this week>,
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling"|"shipping_bottleneck"|"credit_stress">,
  "supply_demand_balance": <float, positive=demand>supply>,
  "portfolio": [
    {{"ticker": <str>, "weight": <float>, "conviction": <float 0-1>, "reasoning": <str 1-2 sentences>}}
  ],
  "short_portfolio": [
    {{"ticker": <str>, "weight": <float>, "conviction": <float 0-1>, "reasoning": <str 1-2 sentences>}}
  ],
  "net_exposure": <float, should match net_exposure_target from macro signal>,
  "signal_confidence": <float 0-1>,
  "thesis_stress": <bool>,
  "thesis_update": <str, what changed vs last run>
}}"""
```

- [ ] **Step 4: Update build_signal_context to inject macro block**

Replace the `build_signal_context` function signature and add macro injection at the top:

```python
def build_signal_context(
    docs: list[dict],
    current_portfolio: dict = None,
    macro_signal: dict = None,
) -> str:
    """Assemble documents into structured prompt context organised by value chain layer."""
    sections = []

    # Prepend macro regime block if available
    if macro_signal:
        sc = macro_signal.get("supply_chain", {})
        cs = macro_signal.get("cross_sector", {})
        macro_block = (
            "### MACRO REGIME SIGNAL [computed by quant module — treat as ground truth]\n"
            f"Regime: {macro_signal['regime']} (confidence: {macro_signal['regime_confidence']:.2f})\n"
            f"net_exposure_target: {macro_signal['net_exposure_target']:.2f} "
            f"({int(macro_signal['net_exposure_target']*100)}% long / "
            f"{int((1-macro_signal['net_exposure_target'])*100)}% short notional)\n"
            f"Supply chain: PMI {sc.get('pmi', 'N/A')} ({sc.get('pmi_trend', 'N/A')}), "
            f"shipping pressure {sc.get('shipping_pressure', 'N/A'):.2f}, "
            f"semis inventory {sc.get('semis_inventory_trend', 'N/A')}\n"
            f"Cross-sector: power→compute lead {cs.get('power_compute_lead', 0):.1f}σ, "
            f"copper→infra {cs.get('copper_infra_lead', 0):.1f}σ, "
            f"credit stress: {cs.get('credit_stress', False)}, "
            f"VIX {cs.get('vix_level', 'N/A'):.1f}\n"
            f"Notes: {macro_signal.get('notes', '')}\n\n"
            "[Your portfolio must reflect the net_exposure_target above]\n"
        )
        sections.append(macro_block)

    # Current portfolio context
    if current_portfolio:
        sorted_positions = sorted(current_portfolio.items(), key=lambda x: -x[1])
        lines = [f"  {ticker}: {weight:.1%}" for ticker, weight in sorted_positions[:20]]
        sections.append(
            "### CURRENT PORTFOLIO POSITIONS\n"
            "You currently hold these positions. Factor them into your recommendations —\n"
            "avoid large rotations unless the signal strongly justifies it.\n"
            + "\n".join(lines) + "\n"
        )

    # Documents by value chain layer
    by_layer = defaultdict(list)
    for doc in docs:
        layer = doc.get("value_chain_layer", "application")
        by_layer[layer].append(doc)

    for layer in VALUE_CHAIN_LAYERS:
        layer_docs = by_layer.get(layer, [])
        if not layer_docs:
            continue
        header = f"\n### {layer.upper()} LAYER SIGNALS\n"
        entries = []
        for d in layer_docs:
            entries.append(
                f"Source: {d['source'].upper()} | Date: {d.get('published_at', 'unknown')}\n"
                f"Title: {d['title']}\n"
                f"Content: {d['content'][:2000]}\n"
            )
        sections.append(header + "\n---\n".join(entries))

    return "\n".join(sections)
```

- [ ] **Step 5: Update apply_guardrails to handle short book**

Replace `apply_guardrails` entirely:

```python
def apply_guardrails(output: dict, prev_weights: dict) -> dict:
    """Apply hard constraints to Gemini long and short portfolio output."""
    output = copy.deepcopy(output)

    # ── Long book: cap + normalize ────────────────────────────────────────
    portfolio = output["portfolio"]
    _EPS = 1e-6
    for _ in range(50):
        for p in portfolio:
            p["weight"] = min(p["weight"], MAX_STOCK_WEIGHT)
        total = sum(p["weight"] for p in portfolio)
        if total <= 0:
            break
        if len(portfolio) > 1:
            for p in portfolio:
                p["weight"] = p["weight"] / total
        if all(p["weight"] <= MAX_STOCK_WEIGHT + _EPS for p in portfolio):
            break
    output["portfolio"] = portfolio

    # ── Short book: cap + normalize ───────────────────────────────────────
    short_portfolio = output.get("short_portfolio", [])
    for p in short_portfolio:
        p["weight"] = min(p["weight"], MAX_SHORT_WEIGHT)
    short_total = sum(p["weight"] for p in short_portfolio)
    if short_total > 0 and len(short_portfolio) > 1:
        for p in short_portfolio:
            p["weight"] = p["weight"] / short_total
    output["short_portfolio"] = short_portfolio

    return output
```

Add `from config import MAX_SHORT_WEIGHT` to the imports at the top of `gemini_scorer.py` (alongside the existing config imports).

- [ ] **Step 6: Update score_documents to accept macro_signal and return short_weights**

Add `macro_signal: Optional[dict] = None` parameter to `score_documents`. Pass it to `build_signal_context`:

```python
def score_documents(
    docs: list[dict],
    db_path: str = str(DEFAULT_DB),
    prev_weights: Optional[dict] = None,
    current_portfolio: Optional[dict] = None,
    macro_signal: Optional[dict] = None,
) -> dict:
    if prev_weights is None:
        prev_weights = {}

    guardrail_baseline = current_portfolio if current_portfolio else prev_weights
    context = build_signal_context(docs, current_portfolio=current_portfolio, macro_signal=macro_signal)
    universe_str = ", ".join(TICKER_UNIVERSE)
    system = _SYSTEM_PROMPT.format(universe=universe_str)

    user_prompt = f"""Given these forward-looking signals, output portfolio weights for next week.
Weight stocks that will benefit from what is being *committed to* today, not what has already happened.

{context}

[TASK]
Output your portfolio JSON now. Remember: long weights sum to 1.0 (max 10%), short weights sum to 1.0 (max 8%)."""

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    last_exc = None
    raw = None
    raw_response_text = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user_prompt}",
                config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
            )
            raw_response_text = response.text
            raw = parse_gemini_response(response.text)
            break
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  Gemini call failed (attempt {attempt + 1}/3): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    if raw is None:
        raise RuntimeError(f"Gemini failed after 3 attempts: {last_exc}") from last_exc

    guarded = apply_guardrails(raw, guardrail_baseline)

    final_sum = sum(p["weight"] for p in guarded["portfolio"])
    if abs(final_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Long portfolio weights sum to {final_sum:.4f} after guardrails")

    conviction_map = {p["ticker"]: p["conviction"] for p in guarded["portfolio"]}
    reasoning_map = {p["ticker"]: p["reasoning"] for p in guarded["portfolio"]}
    weight_map = {p["ticker"]: p["weight"] for p in guarded["portfolio"]}
    short_weight_map = {p["ticker"]: p["weight"] for p in guarded.get("short_portfolio", [])}

    doc_ids = [d["id"] for d in docs if "id" in d]

    return {
        "p_final": guarded["p_score"],
        "stock_conviction": json.dumps(conviction_map),
        "stock_weights": json.dumps(weight_map),
        "stock_reasoning": json.dumps(reasoning_map),
        "short_weights": json.dumps(short_weight_map) if short_weight_map else None,
        "macro_signal": json.dumps(macro_signal) if macro_signal else None,
        "sector_tilt": json.dumps({}),
        "supply_demand_balance": guarded.get("supply_demand_balance", 0.0),
        "market_regime": guarded["market_regime"],
        "signal_confidence": guarded.get("signal_confidence", 0.5),
        "thesis_stress": guarded.get("thesis_stress", False),
        "signal_age_days": 0,
        "sources_ingested": len(docs),
        "signal_breakdown": json.dumps({}),
        "thesis_update": guarded.get("thesis_update", ""),
        "raw_response": raw_response_text,
        "prompt_context_doc_ids": json.dumps(doc_ids),
    }
```

- [ ] **Step 7: Run full scorer test suite**

Run: `python -m pytest tests/test_gemini_scorer.py -v`
Expected: all tests PASS including the 5 new ones.

- [ ] **Step 8: Commit**

```bash
git add scoring/gemini_scorer.py tests/test_gemini_scorer.py
git commit -m "feat: scorer — macro block injection, long/short schema, short guardrails"
```

---

## Task 6: execution/alpaca.py

**Files:**
- Create: `execution/__init__.py`
- Create: `execution/alpaca.py`
- Create: `tests/test_alpaca_execution.py`

- [ ] **Step 1: Create execution/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Write failing tests**

Create `tests/test_alpaca_execution.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def _make_position(symbol, market_value, side="long"):
    p = MagicMock()
    p.symbol = symbol
    p.market_value = str(market_value)
    p.side = side
    return p


def _make_account(portfolio_value=100_000.0):
    acc = MagicMock()
    acc.portfolio_value = str(portfolio_value)
    return acc


def test_get_alpaca_positions_returns_longs_and_shorts():
    from execution.alpaca import get_alpaca_positions
    mock_client = MagicMock()
    mock_client.get_account.return_value = _make_account(100_000)
    mock_client.get_all_positions.return_value = [
        _make_position("NVDA", 12_000, "long"),
        _make_position("MU",   9_000, "long"),
        _make_position("AMD", -7_000, "short"),
    ]

    with patch("execution.alpaca.TradingClient", return_value=mock_client), \
         patch("execution.alpaca.os.environ.get", side_effect=lambda k, d=None: "fake" if k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") else d):
        result = get_alpaca_positions()

    assert result["longs"]["NVDA"] == pytest.approx(0.12)
    assert result["longs"]["MU"] == pytest.approx(0.09)
    assert result["shorts"]["AMD"] == pytest.approx(0.07)
    assert result["net_exposure"] == pytest.approx(0.21 - 0.07)
    assert result["portfolio_value"] == pytest.approx(100_000)


def test_get_alpaca_positions_returns_empty_when_no_credentials():
    from execution.alpaca import get_alpaca_positions
    with patch("execution.alpaca.os.environ.get", return_value=None):
        result = get_alpaca_positions()
    assert result == {"longs": {}, "shorts": {}, "net_exposure": 0.0, "gross_exposure": 0.0, "portfolio_value": 0.0}


def test_rebalance_skips_tiny_orders():
    from execution.alpaca import rebalance
    mock_client = MagicMock()
    mock_client.get_account.return_value = _make_account(100_000)
    mock_client.get_all_positions.return_value = []

    # Weight of 0.003 → $300, below $500 min threshold
    with patch("execution.alpaca.TradingClient", return_value=mock_client), \
         patch("execution.alpaca.os.environ.get", side_effect=lambda k, d=None: "fake" if k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") else d):
        rebalance(
            long_weights={"NVDA": 0.003},
            short_weights={},
            net_exposure_target=0.80,
        )

    mock_client.submit_order.assert_not_called()


def test_rebalance_executes_without_error_at_boundary_gross():
    from execution.alpaca import rebalance
    # long_notional=1.2, short_notional=0.7 → gross=1.9 > 1.80
    long_weights = {f"T{i}": 0.1 for i in range(10)}   # sums to 1.0, notional = 1.2 at η=0.80... 
    # Force a case: net=0.55 → long_notional=1.175, short_notional=0.625
    # Both books sum to 1.0 so notional is determined by net_exposure_target.
    # Gross = 1.80 exactly at design point — test extreme case instead.
    with patch("execution.alpaca.TradingClient") as mock_cls, \
         patch("execution.alpaca.os.environ.get", side_effect=lambda k, d=None: "fake" if k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") else d):
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_account(100_000)
        mock_client.get_all_positions.return_value = []
        mock_cls.return_value = mock_client

        # net_exposure_target=0.05 → long_notional=0.925, short_notional=0.875 → gross=1.80 ok
        # net_exposure_target=-0.1 would be invalid → use valid boundary
        rebalance(long_weights={"NVDA": 1.0}, short_weights={"AMD": 1.0}, net_exposure_target=0.55)
        # Should complete without raising
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_alpaca_execution.py -v`
Expected: FAIL — `execution.alpaca` not found.

- [ ] **Step 4: Implement execution/alpaca.py**

```python
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
    # w_gross_L = (MAX_GROSS_EXPOSURE + net_exposure_target) / 2
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
    for sym, target_val in short_targets.items():
        if sym in current_longs:
            _submit(sym, OrderSide.SELL, current_longs[sym])
    for sym, target_val in long_targets.items():
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_alpaca_execution.py -v`
Expected: all 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add execution/__init__.py execution/alpaca.py tests/test_alpaca_execution.py
git commit -m "feat: execution/alpaca — two-sided rebalance with gross exposure guardrail"
```

---

## Task 7: main.py Orchestration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update get_alpaca_positions import**

`main.py` currently defines `get_alpaca_positions()` inline. Remove that function entirely and import from `execution.alpaca` instead. Also import `rebalance` and `compute_macro_signal`.

Replace the two import blocks at the top of `main.py`:

```python
import argparse
import json
from datetime import datetime, timezone

from config import (
    DB_PATH, RSS_FEEDS, ARXIV_CATEGORIES, ARXIV_MAX_RESULTS,
    EDGAR_TICKERS,
)
from db import init_db, get_unscored_documents, get_recent_documents, mark_scored, insert_signal
from ingestion.rss import ingest_rss
from ingestion.arxiv import ingest_arxiv
from ingestion.transcripts import ingest_edgar
from macro.regime import compute_macro_signal
from scoring.gemini_scorer import score_documents
from execution.alpaca import get_alpaca_positions, rebalance
from export import export_signal
```

- [ ] **Step 2: Remove the old get_alpaca_positions function**

Delete the entire `get_alpaca_positions()` function definition from `main.py` (it's now in `execution/alpaca.py`).

- [ ] **Step 3: Update get_prev_weights to use stock_weights not stock_conviction**

Replace `get_prev_weights`:

```python
def get_prev_weights(db_path: str) -> dict:
    """Load stock weights from most recent signal row."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT stock_weights FROM signals ORDER BY computed_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row and row[0]:
        return json.loads(row[0])
    return {}
```

- [ ] **Step 4: Add macro signal computation + execution call in main()**

In `main()`, replace the scoring block and add orchestration steps 2 and 5:

```python
def main():
    parser = argparse.ArgumentParser(description="AI Signal Engine")
    parser.add_argument("--force", action="store_true",
                        help="Score even if no new documents were ingested")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest and score but don't write output JSON files or execute trades")
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine starting...")

    # 1. Init DB
    init_db(DB_PATH)

    # 2. Ingest
    print("\n--- Ingestion ---")
    total_new = 0
    for feed in RSS_FEEDS:
        n = ingest_rss(feed["url"], feed["value_chain_layer"], DB_PATH)
        print(f"  RSS [{feed['value_chain_layer']}]: {n} new documents")
        total_new += n
    n = ingest_arxiv(ARXIV_CATEGORIES, ARXIV_MAX_RESULTS, DB_PATH)
    print(f"  arXiv: {n} new documents")
    total_new += n
    n = ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=DB_PATH)
    print(f"  EDGAR: {n} new documents")
    total_new += n
    print(f"\nTotal new documents: {total_new}")

    unscored = get_unscored_documents(DB_PATH)
    if not unscored and not args.force:
        print("No unscored documents — nothing to do. Use --force to override.")
        return
    if not unscored and args.force:
        unscored = get_recent_documents(DB_PATH, days=30)
        print(f"Force mode: re-scoring {len(unscored)} documents from last 30 days")
    if not unscored:
        print("No documents available to score. Run ingestion first.")
        return

    # 3. Compute macro signal
    print("\n--- Macro Signal ---")
    macro_signal = compute_macro_signal()
    print(f"  Regime: {macro_signal['regime']} (confidence: {macro_signal['regime_confidence']:.2f})")
    print(f"  Net exposure target: {macro_signal['net_exposure_target']:.2f}")
    print(f"  {macro_signal['notes']}")

    # 4. Fetch current Alpaca positions
    print("\nFetching current Alpaca portfolio...")
    positions = get_alpaca_positions()
    current_portfolio = positions["longs"]
    if current_portfolio:
        top = sorted(current_portfolio.items(), key=lambda x: -x[1])[:5]
        print(f"  {len(current_portfolio)} long positions | top: " + ", ".join(f"{t} {w:.1%}" for t, w in top))
    if positions["shorts"]:
        top_s = sorted(positions["shorts"].items(), key=lambda x: -x[1])[:3]
        print(f"  {len(positions['shorts'])} short positions | top: " + ", ".join(f"{t} {w:.1%}" for t, w in top_s))

    # 5. Score
    print(f"\n--- Scoring {len(unscored)} documents via Gemini ---")
    prev_weights = get_prev_weights(DB_PATH)
    signal = score_documents(
        docs=unscored,
        db_path=DB_PATH,
        prev_weights=prev_weights,
        current_portfolio=current_portfolio,
        macro_signal=macro_signal,
    )

    # 6. Persist
    doc_ids = [d["id"] for d in unscored]
    mark_scored(DB_PATH, doc_ids)
    insert_signal(DB_PATH, signal)
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | confidence={signal['signal_confidence']:.2f}")
    print(f"  {signal['thesis_update']}")

    # 7. Execute rebalance
    if args.dry_run:
        print("\n[DRY-RUN] Skipping rebalance and export")
        return

    print("\n--- Executing Rebalance ---")
    long_weights = json.loads(signal.get("stock_weights") or "{}")
    short_weights = json.loads(signal.get("short_weights") or "{}")
    rebalance(
        long_weights=long_weights,
        short_weights=short_weights,
        net_exposure_target=macro_signal["net_exposure_target"],
    )

    # 8. Export
    print("\n--- Exporting ---")
    export_signal(signal)

    print("\nDone.")
```

- [ ] **Step 5: Run full test suite to confirm nothing regressed**

Run: `python -m pytest tests/ -v`
Expected: all existing tests PASS. The suite should not import `main.py` directly so no test failures expected from the orchestration changes.

- [ ] **Step 6: Smoke test with dry-run**

Run: `python main.py --dry-run --force 2>&1 | head -40`
Expected: see "Macro Signal" section print with a regime and net_exposure_target, then "[DRY-RUN] Skipping rebalance and export".

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: main — macro signal + two-sided rebalance orchestration"
```

---

## Task 8: export.py — Add short_weights + macro_signal

**Files:**
- Modify: `export.py`
- Modify: `tests/test_export.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_export.py`:

```python
import json as _json

SAMPLE_SIGNAL_V2 = {
    **SAMPLE_SIGNAL,
    "short_weights": _json.dumps({"AMD": 0.55, "QCOM": 0.45}),
    "macro_signal": _json.dumps({
        "regime": "shipping_bottleneck",
        "net_exposure_target": 0.55,
        "notes": "Freight elevated.",
    }),
}


def test_export_includes_short_weights(tmp_path):
    export_signal(SAMPLE_SIGNAL_V2, output_dir=str(tmp_path))
    data = _json.loads((tmp_path / "stock_signals.json").read_text())
    assert "short_weights" in data
    assert data["short_weights"]["AMD"] == pytest.approx(0.55)


def test_export_includes_macro_signal(tmp_path):
    export_signal(SAMPLE_SIGNAL_V2, output_dir=str(tmp_path))
    data = _json.loads((tmp_path / "market_regime.json").read_text())
    assert "macro_signal" in data
    assert data["macro_signal"]["regime"] == "shipping_bottleneck"
    assert data["macro_signal"]["net_exposure_target"] == pytest.approx(0.55)
```

Add `import pytest` to the top of `tests/test_export.py` if not present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export.py::test_export_includes_short_weights tests/test_export.py::test_export_includes_macro_signal -v`
Expected: FAIL.

- [ ] **Step 3: Update export.py**

In `export_signal`, update the `stock_signals.json` write block to include `short_weights`, and the `market_regime.json` block to include `macro_signal`:

```python
def export_signal(signal: dict, output_dir: str = str(BACKTEST_DATA_DIR)) -> None:
    """Write p_estimate.json, stock_signals.json, and market_regime.json."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    conviction = json.loads(signal["stock_conviction"])
    weights = json.loads(signal.get("stock_weights") or "{}")
    reasoning = json.loads(signal.get("stock_reasoning") or "{}")
    short_weights = json.loads(signal.get("short_weights") or "{}")
    macro_signal = json.loads(signal.get("macro_signal") or "{}")
    signal_breakdown = json.loads(signal["signal_breakdown"])

    if datetime.now(timezone.utc).weekday() == 0:
        p_file = output / "p_estimate.json"
        p_file.write_text(json.dumps({"p": signal["p_final"], "generated_at": now}, indent=2))
        print("  p_estimate.json updated (Monday)")

    stock_signals_file = output / "stock_signals.json"
    stock_signals_file.write_text(json.dumps({
        "generated_at": now,
        "conviction": conviction,
        "weights": weights,
        "short_weights": short_weights,
        "reasoning": reasoning,
    }, indent=2))

    market_regime_file = output / "market_regime.json"
    market_regime_file.write_text(json.dumps({
        "generated_at": now,
        "market_regime": signal["market_regime"],
        "supply_demand_balance": signal["supply_demand_balance"],
        "sector_tilt": json.loads(signal["sector_tilt"]),
        "signal_confidence": signal["signal_confidence"],
        "thesis_stress": bool(signal["thesis_stress"]),
        "thesis_update": signal["thesis_update"],
        "signal_breakdown": signal_breakdown,
        "macro_signal": macro_signal,
    }, indent=2))

    print(f"Exported signals to {output}/")
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | sources={signal['sources_ingested']}")
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add export.py tests/test_export.py
git commit -m "feat: export — include short_weights and macro_signal in output files"
```

---

## Final Verification

- [ ] **Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: all tests PASS, no warnings about missing modules.

- [ ] **Dry-run end-to-end**

Run: `python main.py --dry-run --force`
Expected output includes:
```
--- Macro Signal ---
  Regime: <one of compute_constrained|shipping_bottleneck|balanced|credit_stress>
  Net exposure target: <0.20–0.80>
--- Scoring N documents via Gemini ---
  p=0.XX | regime=... | confidence=0.XX
[DRY-RUN] Skipping rebalance and export
```
