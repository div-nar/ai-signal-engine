# Layer-Cake Strategy Core + Ablation Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the mechanical strategy core (five-layer budgets, momentum factor selection, portfolio assembly, risk + rebalance rules) and an ablation backtest that establishes whether the thesis tilt and momentum factor beat QQQ — the go/no-go gate before any live integration.

**Architecture:** Pure, side-effect-free functions in a new `strategy/` package, each independently unit-tested. A `backtest/` package loads a historical price panel and runs three mechanical variants (baseline budgets / +momentum / vs QQQ) through the same assembler, producing a return/Sharpe/max-drawdown scorecard. No live trading, no LLM, no Alpaca writes in this plan.

**Tech Stack:** Python 3.14, pandas, numpy, pytest + pytest-mock, yfinance (already in requirements), Alpaca data client (already used) for price history.

## Global Constraints

- Python interpreter: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` (the repo's `run.sh` uses this).
- Run tests with `pytest` from repo root `/Users/div-nar/sideproj/ai-signal-engine`.
- All new strategy functions are **pure** — no I/O, no global state, no network. Data is passed in.
- The five layers, in canonical order: `["power", "fabrication", "compute", "infrastructure", "platform"]`.
- Per-layer guardrails: floor `0.08`, ceiling `0.35`. Layer budgets always sum to `1.0`.
- Per-name cap: `0.12`. Final assembled weights always sum to `1.0` (fully invested).
- Top names per layer: `3`.
- Do not modify `scoring/gemini_scorer.py`, `execution/alpaca.py`, or `main.py` in this plan — those belong to Plan 2.
- Follow existing repo style: module-level constants in `config.py`, plain functions, type hints, docstrings.

---

### Task 1: Layer map and baseline budgets

**Files:**
- Create: `strategy/__init__.py`
- Create: `strategy/layers.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Produces:
  - `LAYERS: list[str]` — canonical layer order.
  - `LAYER_MAP: dict[str, str]` — ticker → layer (subset of `config.TICKER_UNIVERSE`; non-AI names omitted).
  - `BASELINE_BUDGETS: dict[str, float]` — layer → weight, sums to 1.0.
  - `layer_of(ticker: str) -> str | None`
  - `tickers_in_layer(layer: str) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layers.py
from strategy.layers import (
    LAYERS, LAYER_MAP, BASELINE_BUDGETS, layer_of, tickers_in_layer,
)


def test_layers_canonical_order():
    assert LAYERS == ["power", "fabrication", "compute", "infrastructure", "platform"]


def test_baseline_budgets_sum_to_one():
    assert abs(sum(BASELINE_BUDGETS.values()) - 1.0) < 1e-9
    assert set(BASELINE_BUDGETS) == set(LAYERS)


def test_known_tickers_mapped_to_expected_layers():
    assert layer_of("VST") == "power"
    assert layer_of("TSM") == "fabrication"
    assert layer_of("NVDA") == "compute"
    assert layer_of("MU") == "compute"
    assert layer_of("VRT") == "infrastructure"
    assert layer_of("MSFT") == "platform"


def test_non_ai_names_are_unmapped():
    # Healthcare / financials / staples are not part of the thesis universe.
    assert layer_of("JNJ") is None
    assert layer_of("JPM") is None


def test_every_layer_has_members():
    for layer in LAYERS:
        assert len(tickers_in_layer(layer)) >= 3, layer


def test_layer_map_values_are_valid_layers():
    assert all(v in LAYERS for v in LAYER_MAP.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy'`

- [ ] **Step 3: Create the package init**

```python
# strategy/__init__.py
```
(empty file)

- [ ] **Step 4: Write the implementation**

```python
# strategy/layers.py
"""Five-layer AI value-chain taxonomy and baseline allocation.

The "cake", physical -> value-capture:
  1 power           electrons: grid, generation, electrical gear
  2 fabrication     making silicon: foundry, semicap, EDA, materials
  3 compute         accelerators & memory
  4 infrastructure  datacenters, REITs, cooling, interconnect
  5 platform        hyperscalers & software (QQQ's core)

Tickers outside the AI-infra thesis (healthcare, financials, staples, energy
majors, defense) are intentionally unmapped -> ineligible for the portfolio.
"""

LAYERS = ["power", "fabrication", "compute", "infrastructure", "platform"]

LAYER_MAP = {
    # power & energy
    "VST": "power", "CEG": "power", "NRG": "power", "NEE": "power",
    "ETN": "power", "PWR": "power", "EIX": "power", "AES": "power",
    "GEV": "power", "GE": "power",
    # fabrication & materials
    "TSM": "fabrication", "ASML": "fabrication", "AMAT": "fabrication",
    "LRCX": "fabrication", "KLAC": "fabrication", "SNPS": "fabrication",
    "CDNS": "fabrication", "ON": "fabrication", "TOELY": "fabrication",
    "SHECY": "fabrication", "LIN": "fabrication",
    # compute & silicon
    "NVDA": "compute", "AMD": "compute", "AVGO": "compute", "MU": "compute",
    "MRVL": "compute", "QCOM": "compute", "TXN": "compute", "ADI": "compute",
    "ASX": "compute", "ARM": "compute",
    # infrastructure & networking
    "VRT": "infrastructure", "EQIX": "infrastructure", "DLR": "infrastructure",
    "IRM": "infrastructure", "AMT": "infrastructure", "CSCO": "infrastructure",
    "FCX": "infrastructure",
    # platform & application
    "MSFT": "platform", "GOOGL": "platform", "AMZN": "platform",
    "META": "platform", "ORCL": "platform", "PLTR": "platform",
    "NOW": "platform", "CRWD": "platform", "DDOG": "platform",
    "NFLX": "platform", "CRM": "platform", "ADBE": "platform",
    "INTU": "platform", "IBM": "platform", "SAP": "platform",
    "AAPL": "platform", "BIDU": "platform", "BABA": "platform",
    "TCEHY": "platform",
}

# Structural thesis tilt: overweight the layers QQQ underweights (1-3 + infra),
# underweight platform (QQQ's concentrated core). Calibration dials.
BASELINE_BUDGETS = {
    "power": 0.20,
    "fabrication": 0.20,
    "compute": 0.25,
    "infrastructure": 0.15,
    "platform": 0.20,
}


def layer_of(ticker: str) -> str | None:
    """Return the layer for a ticker, or None if it is outside the thesis universe."""
    return LAYER_MAP.get(ticker)


def tickers_in_layer(layer: str) -> list[str]:
    """Return all tickers assigned to a layer."""
    return [t for t, lyr in LAYER_MAP.items() if lyr == layer]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_layers.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add strategy/__init__.py strategy/layers.py tests/test_layers.py
git commit -m "feat(strategy): five-layer taxonomy and baseline budgets"
```

---

### Task 2: Layer-budget tilt application

**Files:**
- Create: `strategy/budgets.py`
- Test: `tests/test_budgets.py`

**Interfaces:**
- Consumes: `strategy.layers.LAYERS`, `BASELINE_BUDGETS`
- Produces:
  - `LAYER_FLOOR = 0.08`, `LAYER_CEILING = 0.35`
  - `apply_layer_tilt(baseline: dict[str, float], tilt: dict[str, float]) -> dict[str, float]`
    — adds `tilt` to `baseline`, clamps each layer to `[LAYER_FLOOR, LAYER_CEILING]`,
    renormalizes to sum 1.0. Raises `ValueError` if `tilt` does not sum to ~0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budgets.py
import pytest
from strategy.layers import BASELINE_BUDGETS
from strategy.budgets import apply_layer_tilt, LAYER_FLOOR, LAYER_CEILING


def test_zero_tilt_returns_baseline():
    tilt = {k: 0.0 for k in BASELINE_BUDGETS}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    for k in BASELINE_BUDGETS:
        assert out[k] == pytest.approx(BASELINE_BUDGETS[k])


def test_tilt_sums_to_one():
    tilt = {"power": 0.10, "compute": 0.05, "platform": -0.15,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert sum(out.values()) == pytest.approx(1.0)


def test_tilt_must_sum_to_zero():
    bad = {"power": 0.10, "compute": 0.0, "platform": 0.0,
           "fabrication": 0.0, "infrastructure": 0.0}
    with pytest.raises(ValueError):
        apply_layer_tilt(BASELINE_BUDGETS, bad)


def test_ceiling_enforced():
    # Huge tilt into compute is clamped at the ceiling, not allowed to run away.
    tilt = {"compute": 0.30, "platform": -0.30, "power": 0.0,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert out["compute"] <= LAYER_CEILING + 1e-9
    assert sum(out.values()) == pytest.approx(1.0)


def test_floor_enforced():
    # Draining platform below the floor is clamped up to the floor.
    tilt = {"platform": -0.18, "power": 0.18, "compute": 0.0,
            "fabrication": 0.0, "infrastructure": 0.0}
    out = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    assert out["platform"] >= LAYER_FLOOR - 1e-9
    assert sum(out.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_budgets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy.budgets'`

- [ ] **Step 3: Write the implementation**

```python
# strategy/budgets.py
"""Apply the LLM's layer tilt to the baseline budgets, under hard guardrails."""

LAYER_FLOOR = 0.08
LAYER_CEILING = 0.35
_TILT_SUM_TOL = 1e-6


def apply_layer_tilt(baseline: dict[str, float], tilt: dict[str, float]) -> dict[str, float]:
    """Return baseline + tilt, clamped to [floor, ceiling] and renormalized to 1.0.

    `tilt` must sum to ~0 (reallocation, not leverage). After clamping, the
    result is renormalized so the layer budgets sum to exactly 1.0; a second
    clamp pass keeps the ceiling honest after renormalization.
    """
    if abs(sum(tilt.values())) > _TILT_SUM_TOL:
        raise ValueError(f"tilt must sum to ~0, got {sum(tilt.values()):.6f}")

    budgets = {k: baseline[k] + tilt.get(k, 0.0) for k in baseline}

    # Clamp, then renormalize, repeating a few times so both bounds hold while
    # the total stays 1.0.
    for _ in range(50):
        budgets = {k: min(max(v, LAYER_FLOOR), LAYER_CEILING) for k, v in budgets.items()}
        total = sum(budgets.values())
        budgets = {k: v / total for k, v in budgets.items()}
        if all(LAYER_FLOOR - 1e-9 <= v <= LAYER_CEILING + 1e-9 for v in budgets.values()):
            break
    return budgets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_budgets.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add strategy/budgets.py tests/test_budgets.py
git commit -m "feat(strategy): bounded layer-tilt application with floor/ceiling"
```

---

### Task 3: Momentum factor

**Files:**
- Create: `strategy/factors.py`
- Test: `tests/test_factors.py`

**Interfaces:**
- Produces:
  - `momentum_scores(prices: "pd.DataFrame", asof, lookback: int = 126, skip: int = 21) -> dict[str, float]`
    — `prices` is a DataFrame indexed by date (ascending), columns are tickers.
    Returns each ticker's total return from `lookback`+`skip` rows ago to `skip`
    rows ago (classic 12-1-style momentum, skipping the most recent month).
    Tickers without enough history are omitted.
  - `rank_within_layer(layer_tickers: list[str], factor_scores: dict[str, float], top_n: int) -> list[str]`
    — the `top_n` tickers from `layer_tickers` with the highest scores (those with
    no score are excluded).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factors.py
import pandas as pd
import pytest
from strategy.factors import momentum_scores, rank_within_layer


def _panel():
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    # AAA rises 0.5%/day, BBB flat, CCC falls 0.3%/day
    aaa = [100 * (1.005 ** i) for i in range(200)]
    bbb = [100.0 for _ in range(200)]
    ccc = [100 * (0.997 ** i) for i in range(200)]
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc}, index=dates)


def test_momentum_orders_winners_above_losers():
    p = _panel()
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert scores["AAA"] > scores["BBB"] > scores["CCC"]


def test_momentum_skips_recent_window():
    # With skip=21 the score ends 21 rows before asof, so it ignores the last month.
    p = _panel()
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert "AAA" in scores and isinstance(scores["AAA"], float)


def test_momentum_omits_insufficient_history():
    p = _panel().iloc[:50]  # fewer rows than lookback+skip
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert scores == {}


def test_rank_within_layer_takes_top_n():
    scores = {"AAA": 0.5, "BBB": 0.0, "CCC": -0.3, "DDD": 0.2}
    ranked = rank_within_layer(["AAA", "BBB", "CCC", "DDD"], scores, top_n=2)
    assert ranked == ["AAA", "DDD"]


def test_rank_excludes_unscored():
    scores = {"AAA": 0.5}
    ranked = rank_within_layer(["AAA", "ZZZ"], scores, top_n=3)
    assert ranked == ["AAA"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy.factors'`

- [ ] **Step 3: Write the implementation**

```python
# strategy/factors.py
"""Mechanical within-layer ranking factors.

Phase 1: price momentum (no new data dependency — uses the price panel we
already pull). Phase 2 (separate plan) adds fundamental factors behind the
same dict-returning contract.
"""
import pandas as pd


def momentum_scores(prices: pd.DataFrame, asof, lookback: int = 126, skip: int = 21) -> dict[str, float]:
    """Total return from (lookback+skip) rows ago to (skip) rows ago, as of `asof`.

    Skipping the most recent `skip` rows (~1 month) avoids short-term reversal,
    the classic 12-1 momentum construction. Columns without enough history are
    omitted from the result.
    """
    window = prices.loc[:asof]
    needed = lookback + skip + 1
    if len(window) < needed:
        return {}
    start = window.iloc[-(lookback + skip + 1)]
    end = window.iloc[-(skip + 1)]
    ratio = end / start - 1.0
    return {t: float(v) for t, v in ratio.items() if pd.notna(v)}


def rank_within_layer(layer_tickers: list[str], factor_scores: dict[str, float], top_n: int) -> list[str]:
    """Return the top_n tickers in `layer_tickers` by score (unscored excluded)."""
    scored = [t for t in layer_tickers if t in factor_scores]
    scored.sort(key=lambda t: factor_scores[t], reverse=True)
    return scored[:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_factors.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add strategy/factors.py tests/test_factors.py
git commit -m "feat(strategy): price-momentum factor and within-layer ranking"
```

---

### Task 4: Portfolio assembler

**Files:**
- Create: `strategy/assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `strategy.factors.rank_within_layer`
- Produces:
  - `assemble_portfolio(budgets: dict[str, float], factor_scores: dict[str, float], layer_map: dict[str, str], top_n: int = 3, name_cap: float = 0.12) -> dict[str, float]`
    — for each layer, take the top_n names by factor score, split that layer's
    budget across them in proportion to (positive-shifted) factor score, then
    apply `name_cap` and renormalize so the final book sums to 1.0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_assemble.py
import pytest
from strategy.assemble import assemble_portfolio

LAYER_MAP = {
    "P1": "power", "P2": "power",
    "C1": "compute", "C2": "compute", "C3": "compute", "C4": "compute",
}
BUDGETS = {"power": 0.4, "compute": 0.6,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def test_weights_sum_to_one():
    scores = {"P1": 0.3, "P2": 0.1, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": -0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=3, name_cap=0.5)
    assert sum(w.values()) == pytest.approx(1.0)


def test_only_top_n_per_layer_selected():
    scores = {"P1": 0.3, "P2": 0.1, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": -0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=2, name_cap=0.5)
    # compute layer: only C1, C2 (top 2) selected; C3, C4 excluded
    assert set(w) == {"P1", "P2", "C1", "C2"}


def test_name_cap_enforced():
    scores = {"P1": 0.9, "P2": 0.01, "C1": 0.5, "C2": 0.4, "C3": 0.2, "C4": 0.1}
    w = assemble_portfolio(BUDGETS, scores, LAYER_MAP, top_n=3, name_cap=0.12)
    assert max(w.values()) <= 0.12 + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_empty_layer_skipped():
    budgets = {"power": 1.0, "compute": 0.0,
               "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}
    scores = {"P1": 0.3, "P2": 0.1}
    w = assemble_portfolio(budgets, scores, LAYER_MAP, top_n=3, name_cap=0.9)
    assert set(w) == {"P1", "P2"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_nonpositive_scores_fall_back_to_equal_weight():
    budgets = {"compute": 1.0, "power": 0.0,
               "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}
    scores = {"C1": -0.1, "C2": -0.2}
    w = assemble_portfolio(budgets, scores, LAYER_MAP, top_n=2, name_cap=0.9)
    assert w["C1"] == pytest.approx(w["C2"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy.assemble'`

- [ ] **Step 3: Write the implementation**

```python
# strategy/assemble.py
"""Assemble final portfolio weights from layer budgets + within-layer factor ranks."""
from strategy.factors import rank_within_layer


def _cap_and_normalize(weights: dict[str, float], name_cap: float) -> dict[str, float]:
    """Cap each name at name_cap and renormalize to sum 1.0 (iterative)."""
    w = dict(weights)
    for _ in range(50):
        total = sum(w.values())
        if total <= 0:
            return w
        w = {k: v / total for k, v in w.items()}
        if all(v <= name_cap + 1e-9 for v in w.values()):
            break
        w = {k: min(v, name_cap) for k, v in w.items()}
    return w


def assemble_portfolio(
    budgets: dict[str, float],
    factor_scores: dict[str, float],
    layer_map: dict[str, str],
    top_n: int = 3,
    name_cap: float = 0.12,
) -> dict[str, float]:
    """Fully-invested weights: top_n per layer, budget split by factor score, name-capped."""
    weights: dict[str, float] = {}
    for layer, budget in budgets.items():
        if budget <= 0:
            continue
        layer_tickers = [t for t, lyr in layer_map.items() if lyr == layer]
        ranked = rank_within_layer(layer_tickers, factor_scores, top_n)
        if not ranked:
            continue
        shifted = {t: max(factor_scores[t], 0.0) for t in ranked}
        s = sum(shifted.values())
        if s <= 0:
            alloc = {t: budget / len(ranked) for t in ranked}
        else:
            alloc = {t: budget * shifted[t] / s for t in ranked}
        for t, a in alloc.items():
            weights[t] = weights.get(t, 0.0) + a

    if not weights:
        return {}
    return _cap_and_normalize(weights, name_cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assemble.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add strategy/assemble.py tests/test_assemble.py
git commit -m "feat(strategy): portfolio assembler with name cap and full investment"
```

---

### Task 5: Risk-off switch and rebalance bands

**Files:**
- Create: `strategy/risk.py`
- Test: `tests/test_risk.py`

**Interfaces:**
- Produces:
  - `risk_off_cash(credit_stress: bool, vix: float, vix_threshold: float = 30.0, buffer: float = 0.30) -> float`
    — returns `buffer` only when `credit_stress and vix > vix_threshold`, else `0.0`.
  - `needs_rebalance(current: dict[str, float], target: dict[str, float], layer_map: dict[str, str], layer_band: float = 0.03, name_band: float = 0.03) -> bool`
    — True if any layer's aggregate weight drifts > `layer_band` from target, or
    any single name drifts > `name_band`. `current`/`target` are weight dicts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk.py
from strategy.risk import risk_off_cash, needs_rebalance

LAYER_MAP = {"A": "power", "B": "power", "C": "compute"}


def test_risk_off_only_on_extreme():
    assert risk_off_cash(credit_stress=False, vix=45.0) == 0.0
    assert risk_off_cash(credit_stress=True, vix=20.0) == 0.0
    assert risk_off_cash(credit_stress=True, vix=45.0) == 0.30


def test_no_rebalance_when_within_bands():
    cur = {"A": 0.50, "B": 0.10, "C": 0.40}
    tgt = {"A": 0.51, "B": 0.10, "C": 0.39}
    assert needs_rebalance(cur, tgt, LAYER_MAP) is False


def test_rebalance_on_name_drift():
    cur = {"A": 0.50, "B": 0.10, "C": 0.40}
    tgt = {"A": 0.40, "B": 0.20, "C": 0.40}  # A drifts 0.10 > 0.03
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True


def test_rebalance_on_layer_drift():
    # Names within band individually but the layer aggregate shifts.
    cur = {"A": 0.30, "B": 0.30, "C": 0.40}  # power = 0.60
    tgt = {"A": 0.32, "B": 0.32, "C": 0.36}  # power = 0.64 -> 0.04 > 0.03
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True


def test_rebalance_when_name_enters_or_exits():
    cur = {"A": 1.0}
    tgt = {"C": 1.0}
    assert needs_rebalance(cur, tgt, LAYER_MAP) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_risk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy.risk'`

- [ ] **Step 3: Write the implementation**

```python
# strategy/risk.py
"""Narrow extreme-only risk-off switch and turnover-throttling rebalance bands."""
from collections import defaultdict


def risk_off_cash(credit_stress: bool, vix: float, vix_threshold: float = 30.0,
                  buffer: float = 0.30) -> float:
    """Raise a fixed cash buffer only on an extreme objective trigger; else stay invested."""
    if credit_stress and vix > vix_threshold:
        return buffer
    return 0.0


def _layer_totals(weights: dict[str, float], layer_map: dict[str, str]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for t, w in weights.items():
        totals[layer_map.get(t, "_unmapped")] += w
    return totals


def needs_rebalance(current: dict[str, float], target: dict[str, float],
                    layer_map: dict[str, str], layer_band: float = 0.03,
                    name_band: float = 0.03) -> bool:
    """True if any name or any layer aggregate drifts beyond its band."""
    names = set(current) | set(target)
    for t in names:
        if abs(current.get(t, 0.0) - target.get(t, 0.0)) > name_band:
            return True
    cur_l = _layer_totals(current, layer_map)
    tgt_l = _layer_totals(target, layer_map)
    for layer in set(cur_l) | set(tgt_l):
        if abs(cur_l.get(layer, 0.0) - tgt_l.get(layer, 0.0)) > layer_band:
            return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_risk.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add strategy/risk.py tests/test_risk.py
git commit -m "feat(strategy): extreme-only risk-off switch and rebalance bands"
```

---

### Task 6: Backtest price panel loader

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: `strategy.layers.LAYER_MAP`
- Produces:
  - `load_price_panel(tickers: list[str], start: str, end: str) -> "pd.DataFrame"`
    — daily close panel (index=dates ascending, columns=tickers) via yfinance.
    Network-dependent; not unit-tested directly.
  - `to_weekly_fridays(panel: "pd.DataFrame") -> "pd.DataFrame"`
    — resample to Friday closes (rebalance dates). Pure; unit-tested.
  - `forward_returns(panel: "pd.DataFrame", rebal_dates) -> "pd.DataFrame"`
    — per-ticker simple return between consecutive rebalance dates. Pure; unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py
import pandas as pd
import pytest
from backtest.panel import to_weekly_fridays, forward_returns


def _daily():
    idx = pd.date_range("2025-01-01", periods=20, freq="B")
    return pd.DataFrame({"AAA": [100 + i for i in range(20)],
                         "BBB": [200 + 2 * i for i in range(20)]}, index=idx)


def test_weekly_fridays_are_all_fridays():
    wk = to_weekly_fridays(_daily())
    assert all(ts.weekday() == 4 for ts in wk.index)
    assert not wk.empty


def test_forward_returns_between_rebalances():
    wk = to_weekly_fridays(_daily())
    fr = forward_returns(_daily(), wk.index)
    # one fewer row than rebalance dates (last has no forward period)
    assert len(fr) == len(wk.index) - 1
    # AAA rises each day -> all forward returns positive
    assert (fr["AAA"] > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest'`

- [ ] **Step 3: Create the package init**

```python
# backtest/__init__.py
```
(empty file)

- [ ] **Step 4: Write the implementation**

```python
# backtest/panel.py
"""Historical price panel loading and rebalance-date utilities for the backtest."""
import pandas as pd
import yfinance as yf


def load_price_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close panel via yfinance, index ascending, columns=tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return close.dropna(how="all").sort_index()


def to_weekly_fridays(panel: pd.DataFrame) -> pd.DataFrame:
    """Resample daily panel to Friday closes (the weekly rebalance grid)."""
    return panel.resample("W-FRI").last().dropna(how="all")


def forward_returns(panel: pd.DataFrame, rebal_dates) -> pd.DataFrame:
    """Per-ticker simple return between consecutive rebalance dates."""
    dates = list(rebal_dates)
    rows = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        p0 = panel.loc[:d0].iloc[-1]
        p1 = panel.loc[:d1].iloc[-1]
        rows[d0] = (p1 / p0 - 1.0)
    return pd.DataFrame(rows).T
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_panel.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backtest/__init__.py backtest/panel.py tests/test_panel.py
git commit -m "feat(backtest): price panel loader and rebalance-date utilities"
```

---

### Task 7: Backtest engine and metrics

**Files:**
- Create: `backtest/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `strategy.assemble.assemble_portfolio`, `strategy.factors.momentum_scores`,
  `strategy.layers.LAYER_MAP`, `strategy.layers.BASELINE_BUDGETS`,
  `backtest.panel.to_weekly_fridays`, `backtest.panel.forward_returns`
- Produces:
  - `equal_weight_scores(tickers: list[str]) -> dict[str, float]` — all-zero scores
    (drives equal-weight within layer for the no-factor baseline variant).
  - `run_variant(daily_panel, layer_map, budgets, variant: str, top_n=3, name_cap=0.12, lookback=126, skip=21) -> "pd.Series"`
    — weekly-rebalanced equity curve (starts at 1.0). `variant` is `"baseline"`
    (equal-weight within layer) or `"momentum"` (momentum-ranked within layer).
  - `metrics(equity: "pd.Series", periods_per_year: int = 52) -> dict` — keys
    `total_return_pct`, `sharpe`, `max_drawdown_pct`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
import pytest
from backtest.engine import equal_weight_scores, run_variant, metrics

LAYER_MAP = {"A": "power", "B": "power", "C": "compute", "D": "compute"}
BUDGETS = {"power": 0.5, "compute": 0.5,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def _panel(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "A": [100 * 1.004 ** i for i in range(n)],
        "B": [100 * 1.001 ** i for i in range(n)],
        "C": [100 * 1.003 ** i for i in range(n)],
        "D": [100 * 0.999 ** i for i in range(n)],
    }, index=idx)


def test_equal_weight_scores_all_zero():
    s = equal_weight_scores(["A", "B"])
    assert s == {"A": 0.0, "B": 0.0}


def test_run_variant_returns_growing_equity_in_uptrend():
    eq = run_variant(_panel(), LAYER_MAP, BUDGETS, variant="baseline")
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] > 1.0


def test_momentum_beats_baseline_when_winners_persist():
    p = _panel()
    base = run_variant(p, LAYER_MAP, BUDGETS, variant="baseline")
    mom = run_variant(p, LAYER_MAP, BUDGETS, variant="momentum")
    # momentum concentrates in the persistent winners (A, C) -> ends higher
    assert mom.iloc[-1] >= base.iloc[-1]


def test_metrics_shape_and_drawdown_sign():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    m = metrics(eq)
    assert set(m) == {"total_return_pct", "sharpe", "max_drawdown_pct"}
    assert m["max_drawdown_pct"] <= 0
    assert m["total_return_pct"] == pytest.approx(20.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.engine'`

- [ ] **Step 3: Write the implementation**

```python
# backtest/engine.py
"""Weekly-rebalanced ablation backtest over the strategy core."""
import numpy as np
import pandas as pd

from strategy.assemble import assemble_portfolio
from strategy.factors import momentum_scores
from backtest.panel import to_weekly_fridays, forward_returns


def equal_weight_scores(tickers: list[str]) -> dict[str, float]:
    """Zero scores -> assembler falls back to equal weight within each layer."""
    return {t: 0.0 for t in tickers}


def run_variant(daily_panel, layer_map, budgets, variant: str = "baseline",
                top_n: int = 3, name_cap: float = 0.12,
                lookback: int = 126, skip: int = 21) -> pd.Series:
    """Return the weekly-rebalanced equity curve (starts at 1.0) for a variant."""
    weekly = to_weekly_fridays(daily_panel)
    fwd = forward_returns(daily_panel, weekly.index)
    universe = [t for t in daily_panel.columns if t in layer_map]

    equity = [1.0]
    index = [weekly.index[0]]
    for d0 in fwd.index:
        if variant == "momentum":
            scores = momentum_scores(daily_panel, d0, lookback=lookback, skip=skip)
            scores = {t: scores[t] for t in universe if t in scores}
        else:
            scores = equal_weight_scores(universe)
        weights = assemble_portfolio(budgets, scores, layer_map, top_n=top_n, name_cap=name_cap)
        period = fwd.loc[d0]
        port_ret = sum(w * period.get(t, 0.0) for t, w in weights.items())
        equity.append(equity[-1] * (1.0 + port_ret))
        index.append(d0)
    return pd.Series(equity, index=index)


def metrics(equity: pd.Series, periods_per_year: int = 52) -> dict:
    """Total return %, annualized Sharpe (rf=0), and max drawdown %."""
    rets = equity.pct_change().dropna()
    total = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min() * 100
    return {
        "total_return_pct": float(total),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(mdd),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_engine.py
git commit -m "feat(backtest): weekly ablation engine and performance metrics"
```

---

### Task 8: Ablation runner and the go/no-go report

**Files:**
- Create: `backtest/run_ablation.py`
- Test: `tests/test_run_ablation.py`

**Interfaces:**
- Consumes: everything above, plus `backtest.panel.load_price_panel`
- Produces:
  - `build_scorecard(daily_panel, benchmark: "pd.Series", layer_map, budgets) -> "pd.DataFrame"`
    — one row per variant (`baseline`, `momentum`) plus `QQQ`, columns =
    metrics keys. Pure given panels; unit-tested with synthetic data.
  - `main()` — CLI: loads the real panel (thesis universe + QQQ, inception→today),
    prints the scorecard. Not unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_ablation.py
import pandas as pd
import pytest
from backtest.run_ablation import build_scorecard

LAYER_MAP = {"A": "power", "B": "power", "C": "compute", "D": "compute"}
BUDGETS = {"power": 0.5, "compute": 0.5,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def _panel(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "A": [100 * 1.004 ** i for i in range(n)],
        "B": [100 * 1.001 ** i for i in range(n)],
        "C": [100 * 1.003 ** i for i in range(n)],
        "D": [100 * 0.999 ** i for i in range(n)],
    }, index=idx)


def test_scorecard_has_all_variants():
    p = _panel()
    bench = p["A"] / p["A"].iloc[0]  # any benchmark series
    sc = build_scorecard(p, bench, LAYER_MAP, BUDGETS)
    assert set(sc.index) == {"baseline", "momentum", "QQQ"}
    assert "sharpe" in sc.columns
    assert "total_return_pct" in sc.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_ablation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.run_ablation'`

- [ ] **Step 3: Write the implementation**

```python
# backtest/run_ablation.py
"""Run the ablation backtest and print the go/no-go scorecard.

Variants that ARE backtestable mechanically:
  - baseline : thesis layer budgets, equal weight within layer
  - momentum : thesis layer budgets, momentum-ranked within layer
The LLM layer-tilt variant is NOT historically backtestable (no historical
thesis-pass outputs); it is evaluated in the forward/paper phase in Plan 2 by
replaying recorded thesis passes. This runner establishes the mechanical floor.
"""
import pandas as pd

from strategy.layers import LAYER_MAP, BASELINE_BUDGETS
from backtest.panel import load_price_panel, to_weekly_fridays, forward_returns
from backtest.engine import run_variant, metrics


def _benchmark_equity(daily_panel, weekly_index, bench_col: str) -> pd.Series:
    weekly = daily_panel[bench_col].resample("W-FRI").last().dropna()
    weekly = weekly.loc[weekly.index.intersection(weekly_index)]
    return weekly / weekly.iloc[0]


def build_scorecard(daily_panel, benchmark, layer_map, budgets) -> pd.DataFrame:
    """One row per variant + QQQ, columns = metrics."""
    rows = {}
    for variant in ("baseline", "momentum"):
        eq = run_variant(daily_panel, layer_map, budgets, variant=variant)
        rows[variant] = metrics(eq)
    rows["QQQ"] = metrics(benchmark)
    return pd.DataFrame(rows).T[["total_return_pct", "sharpe", "max_drawdown_pct"]]


def main():
    tickers = sorted(LAYER_MAP) + ["QQQ"]
    panel = load_price_panel(tickers, start="2026-03-25", end="2026-06-20")
    qqq = panel["QQQ"]
    thesis_panel = panel.drop(columns=["QQQ"])
    weekly = to_weekly_fridays(thesis_panel)
    bench = _benchmark_equity(panel, weekly.index, "QQQ")
    scorecard = build_scorecard(thesis_panel, bench, LAYER_MAP, BASELINE_BUDGETS)
    print("\n=== ABLATION SCORECARD (weekly rebal, inception -> today) ===")
    print(scorecard.round(2).to_string())
    print("\nShip gate (Plan 2): momentum variant must beat baseline on Sharpe,")
    print("and at least one variant must beat QQQ on Sharpe, to justify going live.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_ablation.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full unit suite**

Run: `pytest tests/test_layers.py tests/test_budgets.py tests/test_factors.py tests/test_assemble.py tests/test_risk.py tests/test_panel.py tests/test_engine.py tests/test_run_ablation.py -v`
Expected: all PASS

- [ ] **Step 6: Run the real ablation (network) and capture the verdict**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m backtest.run_ablation`
Expected: a printed scorecard with `baseline`, `momentum`, `QQQ` rows. Record the
numbers — this is the go/no-go evidence for whether the thesis tilt + momentum
have edge, and whether Plan 2 (live integration) is justified.

- [ ] **Step 7: Commit**

```bash
git add backtest/run_ablation.py tests/test_run_ablation.py
git commit -m "feat(backtest): ablation runner and go/no-go scorecard"
```

---

## Self-review notes

- **Spec coverage:** five-layer cake (Task 1), bounded layer tilt with floor/ceiling (Task 2),
  mechanical momentum factor (Task 3), top-2/3-per-layer assembler with 12% cap + full
  investment (Task 4), extreme-only risk-off + rebalance bands (Task 5), ablation backtest
  with return/Sharpe/max-DD scorecard and ship gate (Tasks 6-8). Deferred to Plan 2 by
  design: LLM thesis-scorer rewrite, Friday/Monday execution + fill verification, weekly
  cadence orchestration, Phase-2 fundamental factors. The LLM-tilt ablation variant is
  explicitly documented as forward-tested (not historically backtestable).
- **Decision gate lives in Task 8 Step 6** — run the real ablation before committing to Plan 2.
- **Type consistency:** weight dicts `dict[str, float]` and score dicts `dict[str, float]`
  throughout; `layer_map`/`budgets` dict shapes consistent across strategy and backtest;
  `run_variant`/`metrics`/`build_scorecard` signatures match their consumers.
