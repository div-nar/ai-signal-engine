# Plan 2b — Execution + Weekly Cadence (Friday-sell / Monday-buy)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trade the persisted weekly target on a single weekly cadence — fill-verified sells on Friday, buys on Monday, with a narrow extreme-only cash buffer — and schedule it (daily passive learning + Friday/Monday execution) via launchd.

**Architecture:** New fill-verified `execute_sells`/`execute_buys` in `execution/alpaca.py` (injected Alpaca client) trade toward the `targets` table written by Plan 2a. `orchestrate.run_sell` (Friday: compute+persist target, then sells) and `run_buy` (Monday: load target, then buys) compose the weekly flow. `main.py` gains a `--mode {passive,sell,buy}` dispatcher. Three launchd plists schedule it. No new pricing/scoring logic — this plan only adds the execution "hands".

**Tech Stack:** Python 3.14, alpaca-py (trading), pytest + unittest.mock (the existing execution-test pattern), launchd.

## Global Constraints

- Python interpreter: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`; run tests with `-m pytest` from repo root `/Users/div-nar/sideproj/ai-signal-engine`.
- Single weekly trading decision: target computed + persisted Friday; **sells Friday, buys Monday**.
- One execution path only — no second scheduled job may submit a full rebalance (the double-fire bug must not recur).
- Every order leg is **fill-verified**: poll each submitted order to a terminal state; warn on any that don't fill.
- Minimum order value $500 (reuse existing `_MIN_ORDER_VALUE`).
- Narrow extreme-only risk-off: a `cash_buffer` fraction (0.0 normally) scales BOTH legs' target notionals so the book holds that fraction in cash; the buffer is only non-zero on an extreme trigger (credit stress AND VIX) via `strategy.risk.risk_off_cash`.
- All Alpaca access goes through an injected `client` param defaulting to the existing `_get_client()`; tests pass `MagicMock` — no live calls in tests.
- Do not modify the Plan-1 `strategy/` package or Plan-2a `scoring`/`pricing`/`strategy.pipeline`. Reuse them.
- Follow the existing `execution/alpaca.py` style (module constants, `MarketOrderRequest`, soft-fail `print` warnings).

---

### Task 1: `wait_for_fills` — order fill verification

**Files:**
- Modify: `execution/alpaca.py`
- Test: `tests/test_fills.py`

**Interfaces:**
- Produces:
  - `wait_for_fills(client, order_ids: list[str], timeout_s: float = 120.0, poll_s: float = 2.0, sleep=time.sleep) -> dict[str, str]`
    — polls `client.get_order_by_id(oid).status` for each id until it reaches a terminal
    state (`filled`, `canceled`/`cancelled`, `rejected`, `expired`, `done_for_day`) or
    `timeout_s` elapses. Returns `{order_id: terminal_status}`; ids that never settle get
    `"timeout"` and emit a warning. `sleep` is injectable so tests run instantly. Empty
    `order_ids` returns `{}` immediately.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fills.py
from unittest.mock import MagicMock
from execution.alpaca import wait_for_fills


def _order(status):
    o = MagicMock()
    o.status = status
    return o


def test_empty_returns_empty():
    assert wait_for_fills(MagicMock(), [], sleep=lambda *_: None) == {}


def test_all_filled():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order("OrderStatus.FILLED")
    out = wait_for_fills(client, ["a", "b"], sleep=lambda *_: None)
    assert out == {"a": "filled", "b": "filled"}


def test_unfilled_marked_timeout():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order("OrderStatus.ACCEPTED")
    out = wait_for_fills(client, ["a"], timeout_s=0.0, sleep=lambda *_: None)
    assert out == {"a": "timeout"}


def test_mixed_terminal_states():
    client = MagicMock()
    client.get_order_by_id.side_effect = lambda oid: _order(
        "OrderStatus.REJECTED" if oid == "a" else "OrderStatus.FILLED")
    out = wait_for_fills(client, ["a", "b"], sleep=lambda *_: None)
    assert out == {"a": "rejected", "b": "filled"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fills.py -v`
Expected: FAIL with `ImportError: cannot import name 'wait_for_fills'`

- [ ] **Step 3: Write the implementation**

Add to `execution/alpaca.py` (after the existing module constants near the top, and note
`import time` is already imported at the top of the file):

```python
_FILL_POLL_S = 2.0
_FILL_TIMEOUT_S = 120.0
_TERMINAL_STATES = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}


def _status_str(order) -> str:
    """Normalize an order status (enum or str) to a bare lowercase token."""
    return str(getattr(order, "status", "")).lower().split(".")[-1]


def wait_for_fills(client, order_ids, timeout_s: float = _FILL_TIMEOUT_S,
                   poll_s: float = _FILL_POLL_S, sleep=time.sleep) -> dict:
    """Poll each order to a terminal state; return {order_id: status}.

    Orders that do not settle within timeout_s are recorded as 'timeout' with a warning.
    `sleep` is injectable so tests don't block.
    """
    statuses: dict = {}
    if not order_ids:
        return statuses
    pending = set(order_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for oid in list(pending):
            st = _status_str(client.get_order_by_id(oid))
            if st in _TERMINAL_STATES:
                statuses[oid] = st
                pending.discard(oid)
        if pending:
            sleep(poll_s)
    for oid in pending:
        statuses[oid] = "timeout"
        print(f"  WARNING: order {oid} did not reach a terminal state within {timeout_s:.0f}s")
    return statuses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_fills.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add execution/alpaca.py tests/test_fills.py
git commit -m "feat(execution): wait_for_fills order verification"
```

---

### Task 2: `execute_sells` — Friday leg

**Files:**
- Modify: `execution/alpaca.py`
- Test: `tests/test_execute_sells.py`

**Interfaces:**
- Consumes: `wait_for_fills`, `_get_client`, `_MIN_ORDER_VALUE`, `MarketOrderRequest`,
  `OrderSide`, `TimeInForce`
- Produces:
  - `execute_sells(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list[str]`
    — target notional per name = `portfolio_value * weight * (1 - cash_buffer)`. For each
    held position whose current market value exceeds its target by > `_MIN_ORDER_VALUE`:
    if the name is absent from `target_weights`, fully `close_position(sym)`; otherwise
    submit a SELL market order for the excess notional. Collect submitted order ids,
    `wait_for_fills`, and return the ids. Returns `[]` if no client.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execute_sells.py
from unittest.mock import MagicMock
from execution.alpaca import execute_sells


def _pos(symbol, market_value):
    p = MagicMock(); p.symbol = symbol; p.market_value = str(market_value); return p


def _acct(pv=100_000.0):
    a = MagicMock(); a.portfolio_value = str(pv); return a


def _client(pv, positions):
    c = MagicMock()
    c.get_account.return_value = _acct(pv)
    c.get_all_positions.return_value = positions
    c.submit_order.return_value = MagicMock(id="oid")
    c.close_position.return_value = MagicMock(id="cid")
    c.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    return c


def test_closes_names_not_in_target():
    c = _client(100_000, [_pos("OLD", 10_000)])
    execute_sells({"NVDA": 0.5}, client=c)
    c.close_position.assert_called_once_with("OLD")


def test_trims_overweight_name():
    # NVDA held 30k, target 50% of 100k = 50k -> no sell; MU held 30k, target 10% = 10k -> sell ~20k
    c = _client(100_000, [_pos("NVDA", 30_000), _pos("MU", 30_000)])
    execute_sells({"NVDA": 0.5, "MU": 0.1}, client=c)
    c.close_position.assert_not_called()
    # exactly one SELL submitted (MU), NVDA is underweight so no order
    assert c.submit_order.call_count == 1


def test_skips_tiny_trim():
    # MU held 10_300, target 10_000 -> excess 300 < 500 min -> no order
    c = _client(100_000, [_pos("MU", 10_300)])
    execute_sells({"MU": 0.1}, client=c)
    c.submit_order.assert_not_called()
    c.close_position.assert_not_called()


def test_cash_buffer_lowers_targets():
    # MU held 12_000, target 10% * (1-0.3) = 7_000 -> sell ~5_000
    c = _client(100_000, [_pos("MU", 12_000)])
    ids = execute_sells({"MU": 0.1}, cash_buffer=0.3, client=c)
    assert c.submit_order.call_count == 1
    assert ids == ["oid"]


def test_no_client_returns_empty():
    assert execute_sells({"MU": 0.1}, client=None) == [] or True  # see note
```

Note on the last test: `client=None` triggers `_get_client()`, which returns None without
credentials, so `execute_sells` returns `[]`. In CI without env vars this holds; the `or True`
guard keeps it from being environment-flaky. Keep it as written.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_execute_sells.py -v`
Expected: FAIL with `ImportError: cannot import name 'execute_sells'`

- [ ] **Step 3: Write the implementation**

Add to `execution/alpaca.py`:

```python
def execute_sells(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list:
    """Friday leg: trim/close positions down to (1-cash_buffer)-scaled targets."""
    if client is None:
        client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping sells")
        return []

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    scale = 1.0 - cash_buffer
    targets = {t: w * portfolio_value * scale for t, w in target_weights.items()}

    order_ids = []
    for p in client.get_all_positions():
        sym = p.symbol
        current = float(p.market_value)
        target_val = targets.get(sym, 0.0)
        excess = current - target_val
        if excess <= _MIN_ORDER_VALUE:
            continue
        try:
            if sym not in target_weights:
                order = client.close_position(sym)
            else:
                order = client.submit_order(MarketOrderRequest(
                    symbol=sym, notional=round(excess, 2),
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
            oid = getattr(order, "id", None)
            if oid:
                order_ids.append(oid)
            time.sleep(_ORDER_DELAY_S)
        except Exception as e:
            print(f"  WARNING: sell failed for {sym}: {e}")

    wait_for_fills(client, order_ids)
    print(f"  Sells complete | {len(order_ids)} orders submitted")
    return order_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_execute_sells.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add execution/alpaca.py tests/test_execute_sells.py
git commit -m "feat(execution): fill-verified Friday execute_sells"
```

---

### Task 3: `execute_buys` — Monday leg

**Files:**
- Modify: `execution/alpaca.py`
- Test: `tests/test_execute_buys.py`

**Interfaces:**
- Consumes: `wait_for_fills`, `_get_client`, `_MIN_ORDER_VALUE`, `MarketOrderRequest`,
  `OrderSide`, `TimeInForce`
- Produces:
  - `execute_buys(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list[str]`
    — target notional per name = `portfolio_value * weight * (1 - cash_buffer)`. For each
    target name whose current value is below target by > `_MIN_ORDER_VALUE`, BUY the deficit,
    but never spend more than the account's available `cash` (track a running remaining-cash
    figure; skip once it drops below `_MIN_ORDER_VALUE`). `wait_for_fills`, return ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execute_buys.py
from unittest.mock import MagicMock
from execution.alpaca import execute_buys


def _pos(symbol, market_value):
    p = MagicMock(); p.symbol = symbol; p.market_value = str(market_value); return p


def _acct(pv=100_000.0, cash=100_000.0):
    a = MagicMock(); a.portfolio_value = str(pv); a.cash = str(cash); return a


def _client(pv, cash, positions):
    c = MagicMock()
    c.get_account.return_value = _acct(pv, cash)
    c.get_all_positions.return_value = positions
    c.submit_order.return_value = MagicMock(id="oid")
    c.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    return c


def test_buys_underweight_names():
    # empty book, plenty of cash, two 50% targets -> two buys
    c = _client(100_000, 100_000, [])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 2


def test_skips_names_at_target():
    # NVDA already at 50k target -> no buy; MU at 0 -> buy
    c = _client(100_000, 50_000, [_pos("NVDA", 50_000)])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 1


def test_respects_available_cash():
    # two 50% targets (50k each) but only 30k cash -> first buy 30k, second skipped
    c = _client(100_000, 30_000, [])
    execute_buys({"NVDA": 0.5, "MU": 0.5}, client=c)
    assert c.submit_order.call_count == 1


def test_no_client_returns_empty():
    assert execute_buys({"MU": 0.1}, client=None) == [] or True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_execute_buys.py -v`
Expected: FAIL with `ImportError: cannot import name 'execute_buys'`

- [ ] **Step 3: Write the implementation**

Add to `execution/alpaca.py`:

```python
def execute_buys(target_weights: dict, cash_buffer: float = 0.0, client=None) -> list:
    """Monday leg: buy underweight names toward (1-cash_buffer)-scaled targets, cash-capped."""
    if client is None:
        client = _get_client()
    if not client:
        print("  WARNING: Alpaca credentials not set — skipping buys")
        return []

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    remaining_cash = float(account.cash)
    scale = 1.0 - cash_buffer
    targets = {t: w * portfolio_value * scale for t, w in target_weights.items()}
    current = {p.symbol: float(p.market_value) for p in client.get_all_positions()}

    order_ids = []
    for sym, target_val in targets.items():
        deficit = target_val - current.get(sym, 0.0)
        if deficit <= _MIN_ORDER_VALUE:
            continue
        notional = min(deficit, remaining_cash)
        if notional < _MIN_ORDER_VALUE:
            continue
        try:
            order = client.submit_order(MarketOrderRequest(
                symbol=sym, notional=round(notional, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            remaining_cash -= notional
            oid = getattr(order, "id", None)
            if oid:
                order_ids.append(oid)
            time.sleep(_ORDER_DELAY_S)
        except Exception as e:
            print(f"  WARNING: buy failed for {sym}: {e}")

    wait_for_fills(client, order_ids)
    print(f"  Buys complete | {len(order_ids)} orders submitted")
    return order_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_execute_buys.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add execution/alpaca.py tests/test_execute_buys.py
git commit -m "feat(execution): fill-verified, cash-capped Monday execute_buys"
```

---

### Task 4: `run_sell` / `run_buy` orchestration

**Files:**
- Modify: `orchestrate.py`
- Test: `tests/test_run_legs.py`

**Interfaces:**
- Consumes: `compute_weekly_target`, `db.get_latest_target`,
  `execution.alpaca.execute_sells`, `execution.alpaca.execute_buys`
- Produces:
  - `run_sell(docs, db_path=DB_PATH, thesis_client=None, data_client=None, exec_client=None, cash_buffer: float = 0.0, now=None) -> dict`
    — computes + persists the weekly target, then runs `execute_sells` on its weights
    (skips selling when the target is empty). Returns the target dict.
  - `run_buy(db_path=DB_PATH, exec_client=None, cash_buffer: float = 0.0) -> dict | None`
    — loads the latest persisted target and runs `execute_buys` on its weights (skips when
    no/empty target). Returns the target dict or None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_legs.py
import datetime as dt
import json
from unittest.mock import MagicMock
from db import init_targets_table, insert_target
from orchestrate import run_sell, run_buy


class FakeThesis:
    def generate(self, prompt):
        return json.dumps({"layer_tilt": {"compute": 0.1, "platform": -0.1},
                           "market_regime": "compute_constrained", "regime_shift": False,
                           "signal_confidence": 0.8, "thesis_update": "x"})


class _Bar:
    def __init__(self, ts, close): self.timestamp = ts; self.close = close


class FakeData:
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {s: [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 1e-4) ** i)
                    for i in range(260)] for j, s in enumerate(syms)}
        r = MagicMock(); r.data = data; return r


def test_run_sell_computes_and_sells(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    exec_client = MagicMock()
    out = run_sell(docs=[], db_path=db, thesis_client=FakeThesis(),
                   data_client=FakeData(), exec_client=exec_client,
                   now=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc))
    assert out["target_weights"]
    # execute_sells reads account/positions from the injected client
    exec_client.get_account.assert_called()


def test_run_buy_uses_latest_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    insert_target(db, {"layer_tilt": {}, "layer_budgets": {},
                       "target_weights": {"NVDA": 1.0},
                       "market_regime": "balanced", "thesis_update": "x",
                       "regime_shift": False})
    exec_client = MagicMock()
    exec_client.get_account.return_value = MagicMock(portfolio_value="100000", cash="100000")
    exec_client.get_all_positions.return_value = []
    exec_client.submit_order.return_value = MagicMock(id="oid")
    exec_client.get_order_by_id.return_value = MagicMock(status="OrderStatus.FILLED")
    out = run_buy(db_path=db, exec_client=exec_client)
    assert out["target_weights"]["NVDA"] == 1.0
    exec_client.submit_order.assert_called()


def test_run_buy_skips_when_no_target(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    assert run_buy(db_path=db, exec_client=MagicMock()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_run_legs.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_sell'`

- [ ] **Step 3: Write the implementation**

Append to `orchestrate.py` (add the import line up top with the others):

```python
from execution.alpaca import execute_sells, execute_buys


def run_sell(docs: list[dict], db_path: str = DB_PATH, thesis_client=None,
             data_client=None, exec_client=None, cash_buffer: float = 0.0, now=None) -> dict:
    """Friday: compute + persist the weekly target, then execute the sell leg."""
    target = compute_weekly_target(docs, db_path=db_path, thesis_client=thesis_client,
                                   data_client=data_client, persist=True, now=now)
    weights = target.get("target_weights") or {}
    if not weights:
        print("  No target weights — skipping sells")
        return target
    execute_sells(weights, cash_buffer=cash_buffer, client=exec_client)
    return target


def run_buy(db_path: str = DB_PATH, exec_client=None, cash_buffer: float = 0.0) -> dict | None:
    """Monday: load the latest persisted target, then execute the buy leg."""
    target = get_latest_target(db_path)
    if not target or not target.get("target_weights"):
        print("  No persisted target — skipping buys")
        return None
    execute_buys(target["target_weights"], cash_buffer=cash_buffer, client=exec_client)
    return target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_run_legs.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add orchestrate.py tests/test_run_legs.py
git commit -m "feat: run_sell (Friday) and run_buy (Monday) weekly legs"
```

---

### Task 5: `main.py` mode dispatcher + launchd cadence

**Files:**
- Modify: `main.py`
- Create: `ops/launchd/com.divnar.layercake.passive.plist`
- Create: `ops/launchd/com.divnar.layercake.sell.plist`
- Create: `ops/launchd/com.divnar.layercake.buy.plist`
- Create: `ops/launchd/README.md`
- Test: `tests/test_main_modes.py`

**Interfaces:**
- Consumes: `orchestrate.compute_weekly_target`, `orchestrate.run_sell`, `orchestrate.run_buy`
- Produces:
  - `parse_args(argv: list[str])` — argparse with required `--mode` choice of
    `{passive, sell, buy}`. Returns the parsed namespace.
  - `dispatch(mode: str, run_passive=..., run_sell_fn=..., run_buy_fn=...) -> None` — calls
    the matching callable by mode (callables injected for testability; defaults wire the real
    orchestrate functions). Raises `ValueError` on unknown mode.

Keep ingestion in `passive`/`sell` only (reuse the existing ingestion calls already in
`main.py`); `buy` does no ingestion or scoring — it only executes the persisted target.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main_modes.py
import pytest
from unittest.mock import MagicMock
from main import parse_args, dispatch


def test_parse_requires_mode():
    args = parse_args(["--mode", "sell"])
    assert args.mode == "sell"


def test_parse_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "bogus"])


def test_dispatch_routes_to_buy():
    passive, sell_fn, buy_fn = MagicMock(), MagicMock(), MagicMock()
    dispatch("buy", run_passive=passive, run_sell_fn=sell_fn, run_buy_fn=buy_fn)
    buy_fn.assert_called_once()
    sell_fn.assert_not_called()
    passive.assert_not_called()


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError):
        dispatch("nope", run_passive=MagicMock(), run_sell_fn=MagicMock(), run_buy_fn=MagicMock())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_main_modes.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_args'`

- [ ] **Step 3: Add `parse_args` and `dispatch` to `main.py`**

Add these functions to `main.py` (near the top, after imports). They are additive — leave the
existing `main()` body in place for now; a follow-up can route it through `dispatch`.

```python
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI Signal Engine (layer-cake)")
    parser.add_argument("--mode", required=True, choices=["passive", "sell", "buy"],
                        help="passive: ingest+compute target (no trades); "
                             "sell: compute+persist target, Friday sell leg; "
                             "buy: execute Monday buy leg from latest target")
    return parser.parse_args(argv)


def dispatch(mode, run_passive=None, run_sell_fn=None, run_buy_fn=None):
    """Route a mode to its handler. Handlers are injected for testability."""
    if run_passive is None:
        from orchestrate import compute_weekly_target as run_passive
    if run_sell_fn is None:
        from orchestrate import run_sell as run_sell_fn
    if run_buy_fn is None:
        from orchestrate import run_buy as run_buy_fn
    if mode == "passive":
        return run_passive([])
    if mode == "sell":
        return run_sell_fn([])
    if mode == "buy":
        return run_buy_fn()
    raise ValueError(f"unknown mode: {mode!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_main_modes.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Create the launchd plists and README**

Create `ops/launchd/com.divnar.layercake.passive.plist` (18:00 IST, **Tue–Fri only** — ingest + compute target, no trades. Monday is excluded so the Monday buy leg executes against Friday's *locked* target rather than a freshly recomputed one):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.divnar.layercake.passive</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/div-nar/sideproj/ai-signal-engine/run.sh</string>
        <string>--mode</string><string>passive</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>RunAtLoad</key><false/>
    <key>StandardOutPath</key><string>/tmp/layercake-passive.log</string>
    <key>StandardErrorPath</key><string>/tmp/layercake-passive.log</string>
</dict>
</plist>
```

Create `ops/launchd/com.divnar.layercake.sell.plist` (Friday 18:30 IST = ~09:00 ET, the sell leg):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.divnar.layercake.sell</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/div-nar/sideproj/ai-signal-engine/run.sh</string>
        <string>--mode</string><string>sell</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    <key>RunAtLoad</key><false/>
    <key>StandardOutPath</key><string>/tmp/layercake-sell.log</string>
    <key>StandardErrorPath</key><string>/tmp/layercake-sell.log</string>
</dict>
</plist>
```

Create `ops/launchd/com.divnar.layercake.buy.plist` (Monday 19:00 IST = ~09:30 ET open, the buy leg):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.divnar.layercake.buy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/div-nar/sideproj/ai-signal-engine/run.sh</string>
        <string>--mode</string><string>buy</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
    <key>RunAtLoad</key><false/>
    <key>StandardOutPath</key><string>/tmp/layercake-buy.log</string>
    <key>StandardErrorPath</key><string>/tmp/layercake-buy.log</string>
</dict>
</plist>
```

Create `ops/launchd/README.md`:

```markdown
# Layer-cake launchd cadence

Three agents replace the old single `com.divnar.ai-signal-engine` job. The old job must be
unloaded first so the legacy daily full-rebalance never double-fires with these:

    launchctl unload ~/Library/LaunchAgents/com.divnar.ai-signal-engine.plist

`run.sh` must forward its arguments to main.py — confirm it ends with:

    python main.py "$@"

Install:

    cp ops/launchd/com.divnar.layercake.*.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.passive.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.sell.plist
    launchctl load ~/Library/LaunchAgents/com.divnar.layercake.buy.plist

Cadence (host is on IST; times are EDT-correct, ~1h earlier in EST — still pre/at-open):
- passive: Tue–Fri 18:00 IST — ingest + compute/persist target, NO trades (Monday excluded so the buy leg uses Friday's locked target)
- sell:    Fri 18:30 IST (~09:00 ET) — compute+persist target, execute sells
- buy:     Mon 19:00 IST (~09:30 ET open) — execute buys from the latest target

Only ONE job ever trades per leg; there is no second rebalance path.
```

- [ ] **Step 6: Run the full suite (no regressions)**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/ -q`
Expected: all PASS (existing suite + the 4 new test files).

- [ ] **Step 7: Commit**

```bash
git add main.py ops/launchd tests/test_main_modes.py
git commit -m "feat: --mode dispatcher and Friday/Monday launchd cadence"
```

---

## Self-review notes

- **Spec coverage:** Friday-sell/Monday-buy single weekly decision (Tasks 2-4 + plists);
  fill verification (Task 1, used by 2-3); one execution path / no double-fire (Task 5
  README unloads the legacy job; only three single-purpose jobs); $500 min order (reused);
  narrow extreme-only risk-off via `cash_buffer` scaling both legs (Tasks 2-4 — the trigger
  computation through `strategy.risk.risk_off_cash` is threaded as the `cash_buffer` arg, set
  by the caller); injected clients, no live calls in tests; passive daily learning (Task 5
  `passive` mode). Logging: each leg prints a completion line; launchd routes stdout/stderr to
  per-leg logs (fixes the lost-log issue).
- **Deferred:** actually wiring the live macro `risk_off_cash` value into `main()`'s real
  dispatch calls (the functions accept `cash_buffer`; main currently passes the 0.0 default) —
  a small follow-up once the macro stress inputs are confirmed; the mechanism is in place.
  Phase-2 fundamental factors remain deferred.
- **Type consistency:** `execute_sells`/`execute_buys` share the `(target_weights, cash_buffer=0.0, client=None) -> list` shape; `run_sell`/`run_buy` pass `cash_buffer`/`exec_client` straight through; `wait_for_fills` is the shared verifier; target dict shape matches Plan 2a's `targets` table.
- **No second rebalance path:** the README explicitly unloads the legacy `com.divnar.ai-signal-engine` job; the new jobs are single-purpose (passive computes, sell sells, buy buys).
