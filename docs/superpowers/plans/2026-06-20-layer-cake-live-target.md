# Plan 2a — Live Target Computation (LLM layer-thesis + momentum pipeline)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a weekly target portfolio from research + live prices — the LLM sets only bounded layer tilts, the mechanical pipeline (momentum within layers) picks and weights names — and persist it. No trading in this plan.

**Architecture:** A new LLM `thesis_scorer` outputs five bounded layer tilts + a regime/thesis narrative (never per-name picks). `apply_layer_tilt` turns those into layer budgets; a `pipeline` runs the Plan-1 momentum factor + assembler within each layer to get fully-invested target weights from live Alpaca price history; the target is persisted to a new `targets` table. Network clients (Gemini, Alpaca data) are dependency-injected so every unit is testable with fakes.

**Tech Stack:** Python 3.14, google-genai, alpaca-py (data), pandas, pytest + pytest-mock. Reuses the Plan-1 `strategy/` package (already on this branch).

## Global Constraints

- Python interpreter: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`; run tests with `-m pytest` from repo root `/Users/div-nar/sideproj/ai-signal-engine`.
- The LLM produces ONLY layer-level output: 5 layer tilts + regime + regime_shift flag + thesis text. It never returns per-name weights or conviction.
- Layer tilts are normalized to sum exactly 0, then `strategy.budgets.apply_layer_tilt` clamps to [0.08, 0.35] and renormalizes to 1.0.
- Within-layer selection is mechanical: `strategy.factors.momentum_scores` + `strategy.assemble.assemble_portfolio` (top 3/layer, 12% name cap, fully invested).
- All network access (Gemini, Alpaca) goes through an injected client parameter defaulting to a real constructor, so tests pass fakes — no live calls in tests.
- Reuse Plan-1 modules; do not duplicate momentum/assembly logic.
- Follow existing repo style (module constants, type hints, docstrings, the soft-fail logging pattern in main.py).

---

### Task 1: `targets` table and accessors

**Files:**
- Modify: `db.py`
- Test: `tests/test_targets_db.py`

**Interfaces:**
- Produces:
  - `init_targets_table(db_path: str = str(DEFAULT_DB)) -> None` — idempotent `CREATE TABLE IF NOT EXISTS targets`.
  - `insert_target(db_path: str, data: dict) -> int` — columns: `layer_tilt`, `layer_budgets`, `target_weights`, `market_regime`, `thesis_update` (all TEXT; JSON for the dict fields), `regime_shift` (INT 0/1). Returns row id.
  - `get_latest_target(db_path: str = str(DEFAULT_DB)) -> dict | None` — most recent row as a dict with parsed JSON for the three dict fields, or None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_targets_db.py
import json
from db import init_targets_table, insert_target, get_latest_target


def test_insert_and_get_latest(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    assert get_latest_target(db) is None
    rid = insert_target(db, {
        "layer_tilt": {"power": 0.05, "compute": 0.0, "platform": -0.05,
                       "fabrication": 0.0, "infrastructure": 0.0},
        "layer_budgets": {"power": 0.25, "compute": 0.25, "platform": 0.15,
                          "fabrication": 0.20, "infrastructure": 0.15},
        "target_weights": {"VST": 0.12, "NVDA": 0.12},
        "market_regime": "compute_constrained",
        "thesis_update": "power is the binding constraint",
        "regime_shift": True,
    })
    assert rid >= 1
    got = get_latest_target(db)
    assert got["market_regime"] == "compute_constrained"
    assert got["regime_shift"] is True
    assert got["target_weights"]["VST"] == 0.12
    assert got["layer_budgets"]["power"] == 0.25


def test_get_latest_returns_most_recent(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {},
                       "market_regime": "balanced", "thesis_update": "first",
                       "regime_shift": False})
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {}, "target_weights": {},
                       "market_regime": "stalling", "thesis_update": "second",
                       "regime_shift": False})
    assert get_latest_target(db)["thesis_update"] == "second"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_targets_db.py -v`
Expected: FAIL with `ImportError: cannot import name 'init_targets_table'`

- [ ] **Step 3: Write the implementation**

Add to `db.py` (after the existing `insert_signal`):

```python
def init_targets_table(db_path: str = str(DEFAULT_DB)) -> None:
    """Create the weekly-target table (idempotent)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS targets (
            id             INTEGER PRIMARY KEY,
            computed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            layer_tilt     TEXT,
            layer_budgets  TEXT,
            target_weights TEXT,
            market_regime  TEXT,
            thesis_update  TEXT,
            regime_shift   INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def insert_target(db_path: str, data: dict) -> int:
    """Persist one weekly target. Dict fields are JSON-encoded."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO targets
           (layer_tilt, layer_budgets, target_weights, market_regime,
            thesis_update, regime_shift)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            json.dumps(data.get("layer_tilt", {})),
            json.dumps(data.get("layer_budgets", {})),
            json.dumps(data.get("target_weights", {})),
            data.get("market_regime", ""),
            data.get("thesis_update", ""),
            int(bool(data.get("regime_shift", False))),
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


def get_latest_target(db_path: str = str(DEFAULT_DB)) -> dict | None:
    """Most recent target row with JSON fields parsed, or None."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM targets ORDER BY computed_at DESC, id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "computed_at": row["computed_at"],
        "layer_tilt": json.loads(row["layer_tilt"] or "{}"),
        "layer_budgets": json.loads(row["layer_budgets"] or "{}"),
        "target_weights": json.loads(row["target_weights"] or "{}"),
        "market_regime": row["market_regime"],
        "thesis_update": row["thesis_update"],
        "regime_shift": bool(row["regime_shift"]),
    }
```

Confirm `import json` and `import sqlite3` are already present at the top of `db.py` (they are used by existing functions); if not, add them.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_targets_db.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_targets_db.py
git commit -m "feat(db): targets table for weekly layer-cake target persistence"
```

---

### Task 2: Tilt normalization and response parsing

**Files:**
- Create: `scoring/thesis_scorer.py`
- Test: `tests/test_thesis_parse.py`

**Interfaces:**
- Consumes: `strategy.layers.LAYERS`
- Produces:
  - `normalize_tilt(raw_tilt: dict) -> dict[str, float]` — restrict to the 5 `LAYERS`
    (missing → 0.0, unknown keys dropped), then subtract the mean so the result sums
    to exactly 0.0. Always returns all five layer keys.
  - `parse_thesis_response(text: str) -> dict` — strip markdown fences, `json.loads`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_thesis_parse.py
import pytest
from strategy.layers import LAYERS
from scoring.thesis_scorer import normalize_tilt, parse_thesis_response


def test_normalize_fills_missing_layers_and_sums_zero():
    out = normalize_tilt({"power": 0.10})
    assert set(out) == set(LAYERS)
    assert sum(out.values()) == pytest.approx(0.0)


def test_normalize_drops_unknown_keys():
    out = normalize_tilt({"power": 0.10, "crypto": 0.5})
    assert "crypto" not in out
    assert sum(out.values()) == pytest.approx(0.0)


def test_normalize_recenters_nonzero_sum():
    # raw sums to +0.10; after recentering it must sum to 0
    out = normalize_tilt({"power": 0.10, "compute": 0.0, "platform": 0.0,
                          "fabrication": 0.0, "infrastructure": 0.0})
    assert sum(out.values()) == pytest.approx(0.0)
    # power should remain the most-positive layer after recentering
    assert max(out, key=out.get) == "power"


def test_parse_strips_code_fence():
    text = '```json\n{"market_regime": "balanced", "layer_tilt": {"power": 0.1}}\n```'
    out = parse_thesis_response(text)
    assert out["market_regime"] == "balanced"
    assert out["layer_tilt"]["power"] == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_thesis_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.thesis_scorer'`

- [ ] **Step 3: Write the implementation**

```python
# scoring/thesis_scorer.py
"""LLM layer-thesis scorer: the model outputs ONLY bounded layer tilts + a regime
narrative. Per-name selection is mechanical (Plan-1 momentum + assembler).
"""
import json
import re

from strategy.layers import LAYERS


def normalize_tilt(raw_tilt: dict) -> dict[str, float]:
    """Coerce an LLM tilt dict to all five layers summing to exactly 0.0.

    Missing layers default to 0; unknown keys are dropped; the result is
    recentered (subtract the mean) so it is a pure reallocation.
    """
    vals = {layer: float(raw_tilt.get(layer, 0.0)) for layer in LAYERS}
    mean = sum(vals.values()) / len(vals)
    return {layer: v - mean for layer, v in vals.items()}


def parse_thesis_response(text: str) -> dict:
    """Parse the LLM response, stripping markdown code fences if present."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```.*$", "", text.strip(), flags=re.DOTALL)
    return json.loads(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_thesis_parse.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scoring/thesis_scorer.py tests/test_thesis_parse.py
git commit -m "feat(scoring): layer-tilt normalization and thesis response parsing"
```

---

### Task 3: `score_layer_thesis` (LLM → guardrailed layer budgets)

**Files:**
- Modify: `scoring/thesis_scorer.py`
- Test: `tests/test_thesis_scorer.py`

**Interfaces:**
- Consumes: `normalize_tilt`, `parse_thesis_response`, `strategy.layers.BASELINE_BUDGETS`,
  `strategy.budgets.apply_layer_tilt`
- Produces:
  - `THESIS_SYSTEM_PROMPT: str`
  - `build_thesis_prompt(docs: list[dict], prev_budgets: dict, macro_signal: dict | None) -> str`
  - `score_layer_thesis(docs: list[dict], prev_budgets: dict | None = None, macro_signal: dict | None = None, client=None) -> dict`
    — calls `client.generate(prompt) -> str` (injected; defaults to a real Gemini
    wrapper), parses, normalizes the tilt, applies it to `BASELINE_BUDGETS`, and returns
    `{"layer_tilt", "layer_budgets", "market_regime", "regime_shift", "signal_confidence",
    "thesis_update", "raw_response"}`. Retries the client up to 3 times; raises
    `RuntimeError` after exhausting retries.

The injected `client` is any object with `.generate(prompt: str) -> str`. A thin default
wrapper around `google.genai` is provided so production callers pass nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_thesis_scorer.py
import json
import pytest
from strategy.budgets import LAYER_FLOOR, LAYER_CEILING
from scoring.thesis_scorer import score_layer_thesis


class FakeClient:
    def __init__(self, response, fail_times=0):
        self.response = response
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return self.response


def _resp(tilt, regime="compute_constrained", shift=True):
    return json.dumps({
        "layer_tilt": tilt,
        "market_regime": regime,
        "regime_shift": shift,
        "signal_confidence": 0.8,
        "thesis_update": "memory + power are the binding constraints",
    })


def test_budgets_within_bounds_and_sum_one():
    client = FakeClient(_resp({"compute": 0.30, "platform": -0.30}))
    out = score_layer_thesis([{"id": 1, "content": "x", "title": "t", "source": "rss"}],
                             client=client)
    b = out["layer_budgets"]
    assert sum(b.values()) == pytest.approx(1.0)
    assert all(LAYER_FLOOR - 1e-9 <= v <= LAYER_CEILING + 1e-9 for v in b.values())
    assert out["market_regime"] == "compute_constrained"
    assert out["regime_shift"] is True


def test_retries_then_succeeds():
    client = FakeClient(_resp({"power": 0.05, "platform": -0.05}), fail_times=2)
    out = score_layer_thesis([], client=client)
    assert client.calls == 3
    assert "layer_budgets" in out


def test_raises_after_exhausting_retries():
    class AlwaysFails:
        def generate(self, prompt):
            raise RuntimeError("down")
    with pytest.raises(RuntimeError):
        score_layer_thesis([], client=AlwaysFails())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_thesis_scorer.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_layer_thesis'`

- [ ] **Step 3: Write the implementation**

Append to `scoring/thesis_scorer.py`:

```python
import os
import time

from strategy.layers import LAYERS, BASELINE_BUDGETS
from strategy.budgets import apply_layer_tilt
from config import GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS


THESIS_SYSTEM_PROMPT = """You are the macro strategist for an AI-infrastructure long-only fund.
The portfolio is organised as a five-layer value chain ("the cake"):
  power          - grid, generation, electrical gear (the electrons)
  fabrication    - foundry, semicap, EDA, materials (making the silicon)
  compute        - accelerators & memory (the chips)
  infrastructure - datacenters, REITs, cooling, interconnect
  platform       - hyperscalers & software (value capture)

Your ONLY job is to decide how to TILT capital across these five layers relative to a
neutral baseline, based on where the binding bottleneck of the AI buildout is right now.
You do NOT pick individual stocks — a mechanical momentum model selects names within each
layer. Tilts are a reallocation: they should sum to roughly zero. Lean hard when the thesis
is strong; the system will clamp extremes.

Output ONLY valid JSON:
{{
  "layer_tilt": {{"power": <float>, "fabrication": <float>, "compute": <float>,
                 "infrastructure": <float>, "platform": <float>}},
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling"|"shipping_bottleneck"|"credit_stress">,
  "regime_shift": <bool, true ONLY if the regime/bottleneck changed vs the prior thesis>,
  "signal_confidence": <float 0-1>,
  "thesis_update": <str, 1-3 sentences on the current bottleneck and what changed>
}}"""


def build_thesis_prompt(docs: list[dict], prev_budgets: dict, macro_signal: dict | None) -> str:
    """Assemble research + prior state into the user prompt for the thesis pass."""
    parts = []
    if macro_signal:
        parts.append(
            "### MACRO REGIME [quant module — ground truth]\n"
            f"regime: {macro_signal.get('regime')}, "
            f"confidence: {macro_signal.get('regime_confidence')}\n"
            f"notes: {macro_signal.get('notes', '')}\n"
        )
    if prev_budgets:
        parts.append("### PRIOR LAYER BUDGETS\n"
                     + ", ".join(f"{k}={v:.2f}" for k, v in prev_budgets.items()) + "\n")
    parts.append("### RESEARCH SIGNALS")
    for d in docs[:40]:
        parts.append(
            f"[{d.get('source', '?').upper()}] {d.get('title', '')}\n"
            f"{(d.get('content') or '')[:1500]}\n---"
        )
    parts.append("\n[TASK] Output the layer_tilt JSON now. Tilts should sum to ~0.")
    return "\n".join(parts)


class _GeminiClient:
    """Default production client: thin wrapper exposing .generate(prompt) -> str."""

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
        )
        return resp.text


def score_layer_thesis(docs: list[dict], prev_budgets: dict | None = None,
                       macro_signal: dict | None = None, client=None) -> dict:
    """Run the LLM thesis pass and return guardrailed layer budgets + narrative."""
    if client is None:
        client = _GeminiClient()
    prompt = f"{THESIS_SYSTEM_PROMPT}\n\n{build_thesis_prompt(docs, prev_budgets or {}, macro_signal)}"

    last_exc = None
    raw_text = None
    for attempt in range(3):
        try:
            raw_text = client.generate(prompt)
            parsed = parse_thesis_response(raw_text)
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"thesis scoring failed after 3 attempts: {last_exc}") from last_exc

    tilt = normalize_tilt(parsed.get("layer_tilt", {}))
    budgets = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    return {
        "layer_tilt": tilt,
        "layer_budgets": budgets,
        "market_regime": parsed.get("market_regime", "balanced"),
        "regime_shift": bool(parsed.get("regime_shift", False)),
        "signal_confidence": float(parsed.get("signal_confidence", 0.5)),
        "thesis_update": parsed.get("thesis_update", ""),
        "raw_response": raw_text,
    }
```

Note: keep the `import` lines at the top of the file per PEP 8 when you add them — the
appended `import os`/`import time` and `from ...` lines should be moved up with the existing
imports rather than left mid-file. Functionally either works; move them for cleanliness.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_thesis_scorer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scoring/thesis_scorer.py tests/test_thesis_scorer.py
git commit -m "feat(scoring): score_layer_thesis -> guardrailed layer budgets"
```

---

### Task 4: Live price history fetch

**Files:**
- Create: `pricing/__init__.py`
- Create: `pricing/history.py`
- Test: `tests/test_pricing_history.py`

**Interfaces:**
- Produces:
  - `fetch_recent_closes(tickers: list[str], lookback_days: int = 320, client=None, now=None) -> "pd.DataFrame"`
    — daily close panel (index ascending dates, columns = tickers) for the trailing
    `lookback_days`. `client` is any object with
    `.get_stock_bars(request) -> object` whose `.data` is `{symbol: [bar, ...]}` and each
    bar has `.timestamp` and `.close`; defaults to a real Alpaca data client. Returns an
    empty DataFrame if the client yields no data.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing_history.py
import datetime as dt
import pandas as pd
from pricing.history import fetch_recent_closes


class _Bar:
    def __init__(self, ts, close):
        self.timestamp = ts
        self.close = close


class FakeDataClient:
    def __init__(self, data):
        self._data = data
        self.last_request = None

    def get_stock_bars(self, request):
        self.last_request = request
        class R:
            pass
        r = R()
        r.data = self._data
        return r


def test_builds_close_panel():
    d0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data = {
        "AAA": [_Bar(d0, 10.0), _Bar(d0 + dt.timedelta(days=1), 11.0)],
        "BBB": [_Bar(d0, 20.0), _Bar(d0 + dt.timedelta(days=1), 21.0)],
    }
    panel = fetch_recent_closes(["AAA", "BBB"], lookback_days=30,
                                client=FakeDataClient(data),
                                now=dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc))
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel["AAA"].tolist() == [10.0, 11.0]
    assert list(panel.index) == sorted(panel.index)


def test_empty_data_returns_empty_frame():
    panel = fetch_recent_closes(["AAA"], client=FakeDataClient({}),
                                now=dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc))
    assert panel.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_pricing_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pricing'`

- [ ] **Step 3: Create the package init**

```python
# pricing/__init__.py
```
(empty file)

- [ ] **Step 4: Write the implementation**

```python
# pricing/history.py
"""Live daily price history for the momentum factor, via Alpaca market data."""
import datetime as dt

import pandas as pd


def _default_client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    )


def fetch_recent_closes(tickers: list[str], lookback_days: int = 320,
                        client=None, now=None) -> pd.DataFrame:
    """Daily close panel for the trailing lookback_days (index ascending, cols=tickers).

    Default 320 calendar days (~228 trading days) leaves comfortable headroom over the
    momentum window's 148-bar floor so a live run never silently returns too few rows.
    """
    if client is None:
        client = _default_client()
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=lookback_days)

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    request = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
                               start=start, end=now)
    bars = client.get_stock_bars(request).data

    series = {}
    for sym, rows in bars.items():
        if not rows:
            continue
        series[sym] = pd.Series(
            {str(b.timestamp)[:10]: float(b.close) for b in rows}
        )
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_pricing_history.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add pricing/__init__.py pricing/history.py tests/test_pricing_history.py
git commit -m "feat(pricing): live daily close panel via Alpaca data client"
```

---

### Task 5: Target-portfolio pipeline

**Files:**
- Create: `strategy/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `strategy.factors.momentum_scores`, `strategy.assemble.assemble_portfolio`
- Produces:
  - `build_target_portfolio(layer_budgets: dict, price_history: "pd.DataFrame", layer_map: dict, asof=None, top_n: int = 3, name_cap: float = 0.12, lookback: int = 126, skip: int = 21) -> dict[str, float]`
    — momentum-rank within layers and assemble fully-invested target weights. `asof`
    defaults to the last index date of `price_history`. Returns `{}` if momentum cannot be
    computed (insufficient history).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pandas as pd
import pytest
from strategy.pipeline import build_target_portfolio

LAYER_MAP = {"A": "power", "B": "power", "C": "compute", "D": "compute"}
BUDGETS = {"power": 0.5, "compute": 0.5,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def _panel(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "A": [100 * 1.004 ** i for i in range(n)],
        "B": [100 * 1.001 ** i for i in range(n)],
        "C": [100 * 1.003 ** i for i in range(n)],
        "D": [100 * 0.999 ** i for i in range(n)],
    }, index=idx)


def test_target_is_fully_invested():
    w = build_target_portfolio(BUDGETS, _panel(), LAYER_MAP,
                               top_n=3, name_cap=0.5, lookback=20, skip=5)
    assert sum(w.values()) == pytest.approx(1.0)
    assert max(w.values()) <= 0.5 + 1e-9


def test_target_favours_momentum_winners():
    w = build_target_portfolio(BUDGETS, _panel(), LAYER_MAP,
                               top_n=1, name_cap=1.0, lookback=20, skip=5)
    # strongest in each layer: A (power), C (compute)
    assert set(w) == {"A", "C"}


def test_empty_when_insufficient_history():
    w = build_target_portfolio(BUDGETS, _panel(n=10), LAYER_MAP,
                               lookback=126, skip=21)
    assert w == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'strategy.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# strategy/pipeline.py
"""Compose the mechanical target portfolio from layer budgets + live prices."""
from strategy.factors import momentum_scores
from strategy.assemble import assemble_portfolio


def build_target_portfolio(layer_budgets: dict, price_history, layer_map: dict,
                           asof=None, top_n: int = 3, name_cap: float = 0.12,
                           lookback: int = 126, skip: int = 21) -> dict[str, float]:
    """Momentum-rank within each layer and assemble fully-invested target weights."""
    if price_history is None or price_history.empty:
        return {}
    if asof is None:
        asof = price_history.index[-1]
    scores = momentum_scores(price_history, asof, lookback=lookback, skip=skip)
    if not scores:
        return {}
    return assemble_portfolio(layer_budgets, scores, layer_map,
                              top_n=top_n, name_cap=name_cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add strategy/pipeline.py tests/test_pipeline.py
git commit -m "feat(strategy): build_target_portfolio pipeline (momentum -> assemble)"
```

---

### Task 6: `compute_weekly_target` orchestration

**Files:**
- Create: `orchestrate.py`
- Test: `tests/test_orchestrate.py`

**Interfaces:**
- Consumes: `scoring.thesis_scorer.score_layer_thesis`, `pricing.history.fetch_recent_closes`,
  `strategy.pipeline.build_target_portfolio`, `strategy.layers.LAYER_MAP`,
  `db.init_targets_table`, `db.insert_target`, `db.get_latest_target`
- Produces:
  - `compute_weekly_target(docs: list[dict], db_path: str, thesis_client=None, data_client=None, persist: bool = True, now=None) -> dict`
    — runs the thesis pass → budgets, fetches prices for `LAYER_MAP` tickers, builds the
    target, and (if `persist`) writes it to the `targets` table. Returns the full target
    dict (`layer_tilt`, `layer_budgets`, `target_weights`, `market_regime`, `regime_shift`,
    `thesis_update`). Reads the prior budgets via `get_latest_target` to feed the thesis pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrate.py
import datetime as dt
import json
import pandas as pd
import pytest
from db import init_targets_table, get_latest_target
from orchestrate import compute_weekly_target


class FakeThesisClient:
    def generate(self, prompt):
        return json.dumps({
            "layer_tilt": {"compute": 0.10, "platform": -0.10},
            "market_regime": "compute_constrained",
            "regime_shift": False,
            "signal_confidence": 0.8,
            "thesis_update": "compute bottleneck",
        })


class _Bar:
    def __init__(self, ts, close):
        self.timestamp = ts
        self.close = close


class FakeDataClient:
    """Returns an uptrending 200-day panel for two real-universe tickers per layer."""
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {}
        for j, s in enumerate(syms):
            data[s] = [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 0.0001) ** i)
                       for i in range(200)]
        class R: pass
        r = R(); r.data = data
        return r


def test_computes_and_persists_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    out = compute_weekly_target(
        docs=[{"id": 1, "source": "rss", "title": "t", "content": "c"}],
        db_path=db, thesis_client=FakeThesisClient(), data_client=FakeDataClient(),
        now=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
    )
    assert out["market_regime"] == "compute_constrained"
    assert sum(out["layer_budgets"].values()) == pytest.approx(1.0)
    assert out["target_weights"]  # non-empty
    assert sum(out["target_weights"].values()) == pytest.approx(1.0)
    # persisted
    assert get_latest_target(db)["market_regime"] == "compute_constrained"


def test_persist_false_does_not_write(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    compute_weekly_target(
        docs=[], db_path=db, thesis_client=FakeThesisClient(),
        data_client=FakeDataClient(), persist=False,
        now=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
    )
    assert get_latest_target(db) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_orchestrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrate'`

- [ ] **Step 3: Write the implementation**

```python
# orchestrate.py
"""Weekly target computation: thesis -> budgets -> momentum -> target, persisted."""
from config import DB_PATH
from db import insert_target, get_latest_target
from strategy.layers import LAYER_MAP
from scoring.thesis_scorer import score_layer_thesis
from pricing.history import fetch_recent_closes
from strategy.pipeline import build_target_portfolio


def compute_weekly_target(docs: list[dict], db_path: str = DB_PATH,
                          thesis_client=None, data_client=None,
                          persist: bool = True, now=None) -> dict:
    """Run thesis -> budgets -> momentum pipeline and (optionally) persist the target."""
    prior = get_latest_target(db_path)
    prev_budgets = prior["layer_budgets"] if prior else {}

    thesis = score_layer_thesis(docs, prev_budgets=prev_budgets, client=thesis_client)

    tickers = sorted(LAYER_MAP)
    prices = fetch_recent_closes(tickers, client=data_client, now=now)
    weights = build_target_portfolio(thesis["layer_budgets"], prices, LAYER_MAP)

    target = {
        "layer_tilt": thesis["layer_tilt"],
        "layer_budgets": thesis["layer_budgets"],
        "target_weights": weights,
        "market_regime": thesis["market_regime"],
        "regime_shift": thesis["regime_shift"],
        "thesis_update": thesis["thesis_update"],
    }
    # Never persist a degenerate (empty) target: an empty book would become next
    # week's "prior" and would be what a Plan-2b executor reads. Fail loud instead.
    if not weights:
        print("  WARNING: empty target_weights (insufficient price history?) — not persisting")
    elif persist:
        insert_target(db_path, target)
    return target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_orchestrate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/ -q`
Expected: all PASS (Plan-1 suite + the 6 new test files).

- [ ] **Step 6: Commit**

```bash
git add orchestrate.py tests/test_orchestrate.py
git commit -m "feat: compute_weekly_target orchestration (thesis -> target, persisted)"
```

---

## Self-review notes

- **Spec coverage:** LLM constrained to layer tilts only (Tasks 2-3); tilts normalized + clamped via Plan-1 `apply_layer_tilt` (Tasks 2-3); mechanical momentum-within-layer selection (Task 5 reusing Plan-1 factors/assembler); live prices (Task 4); persisted weekly target (Tasks 1, 6). Deferred to **Plan 2b**: Friday-sell/Monday-buy execution with fill verification, weekly launchd/cron cadence, passive daily learning loop, risk-off switch wiring at execution, logging overhaul, and forward-test recording of LLM tilts. Phase-2 fundamental factors remain deferred (Plan 1 noted).
- **Dependency injection** keeps every unit testable with fakes — no live Gemini/Alpaca calls in tests.
- **Type consistency:** target dict shape (`layer_tilt`/`layer_budgets`/`target_weights`/`market_regime`/`regime_shift`/`thesis_update`) is identical across Tasks 1, 3, and 6; `score_layer_thesis` returns the keys `compute_weekly_target` reads; `fetch_recent_closes` returns the DataFrame shape `build_target_portfolio`/`momentum_scores` expect.
- **No live trading** anywhere in this plan — `targets` is written but never sent to a broker. That is Plan 2b.
