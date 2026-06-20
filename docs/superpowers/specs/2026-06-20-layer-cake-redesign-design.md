# AI Signal Engine — Five-Layer Cake Redesign

**Date:** 2026-06-20
**Status:** Design (approved in brainstorming, pending written-spec review)

## Background & motivation

A full RCA of the live paper-trading record (inception 2026-03-25) found:

- **No demonstrable stock-selection edge.** Over the full reconstructed history, the
  fully-invested book ex-MU returned ~+23.5% vs QQQ +26.0% (−2.5 pts), daily win rate
  49%, correlation 0.74 — i.e. **closet-indexing QQQ with extra churn**.
- **The headline +16% was one lucky name.** MU contributed 58% of all P&L, yet was held
  at a *below*-equal weight (avg 7.9% vs 10% baseline). It was never conviction-sized →
  luck, not signal.
- **Gemini conviction scores carry zero information.** Spearman IC ≈ −0.008 over 125
  name-period observations; top-conviction tercile did *worse* than bottom.
- **~20 pts lost to implementation drag:** ~73% average net exposure (cash drag),
  14×/quarter turnover (churn), and execution failures (double-firing, stuck/unfilled
  orders, lost logs).

**Conclusion:** the AI-infra *universe* is sound (it tracks QQQ), but the LLM stock-picker
and the implementation are not. This redesign keeps the LLM only for the job it might do
well — reading the value-chain thesis at the layer level — and makes everything below that
mechanical, fully invested, low-churn, and reliably executed.

## Goals

1. **Beat QQQ on a risk-adjusted basis** via a repeatable, testable edge — not LLM whim.
2. **Capture the AI-infra thesis cheaply**: fully invested, low turnover, clean execution
   (recover the ~20 pts of implementation drag).
3. A thesis-tilted (not index-hugging) portfolio is the *consequence* of 1 + 2.

Non-goal: free-form LLM stock picking (proven to add no edge).

## Architecture

```
Ingest (RSS / HF papers / EDGAR)
        │
        ▼
LLM Thesis Pass  ── LAYER-LEVEL ONLY ──►  • thesis narrative (the "five-layer cake")
        │                                  • current bottleneck layer
        │                                  • regime-shift flag
        │                                  • layer tilts (agent-decided magnitude)
        ▼
Layer Budgets = baseline ± LLM tilt  (clamped: floor 8% / ceiling 35%, sum-to-zero)
        │
        ▼
Factor Engine  ── MECHANICAL ──►  rank names WITHIN each layer, take top 2–3
        │                          (momentum now; fundamentals Phase 2)
        ▼
Portfolio Assembler  ──►  position cap 12%, fully invested, rebalance bands
        │
        ▼
Execution  ──►  weekly: Friday-AM sells, Monday-AM buys, fill-verified, single path
```

The LLM's entire output is now **5 layer tilts + a regime flag + a thesis paragraph** —
no per-name weights. Everything below the layer line is deterministic and backtestable.

### The five-layer cake (physical → value capture)

| # | Layer | Role | Example names (from current universe) |
|---|-------|------|----------------------------------------|
| 1 | Power & Energy | Electrons: grid, generation, electrical gear | VST, CEG, NRG, NEE, ETN, PWR, EIX, AES, GEV, GE |
| 2 | Fabrication & Materials | Making silicon: foundry, semicap, EDA, materials | TSM, ASML, AMAT, LRCX, KLAC, SNPS, CDNS, ON, ALB, LIN, FCX |
| 3 | Compute & Silicon | Accelerators & memory | NVDA, AMD, AVGO, MU, MRVL, QCOM, TXN, ADI, ASX |
| 4 | Infrastructure & Networking | Where compute lives: DC, REIT, cooling, interconnect | VRT, EQIX, DLR, IRM, AMT, CSCO |
| 5 | Platform & Application | Value capture: hyperscalers & software (QQQ's core) | MSFT, GOOGL, AMZN, META, ORCL, PLTR, NOW, CRWD, DDOG, NFLX |

Every ticker in `config.TICKER_UNIVERSE` is assigned to exactly one layer (a new
`LAYER_MAP`). The structural thesis tilt overweights layers 1–3 (which QQQ underweights and
where MU/VST-type winners live) while still *holding* layer-5 mega-caps — just not at QQQ's
cap-weighted concentration. QQQ mega-caps enter naturally whenever they win their layer's
factor rank; they are not hardcoded in or out.

## Component design

### 1. Baseline layer budgets

Fixed weights encoding the structural tilt; the skeleton the LLM perturbs. Starting point
(to be calibrated in backtest — these are dials, not final):

| Layer | Baseline | QQQ-ish | Tilt |
|-------|----------|---------|------|
| 1 Power & Energy | 20% | ~1% | heavy OW |
| 2 Fabrication & Materials | 20% | ~3% | heavy OW |
| 3 Compute & Silicon | 25% | ~30% | ~neutral |
| 4 Infrastructure & Networking | 15% | ~2% | OW |
| 5 Platform & Application | 20% | ~64% | heavy UW |

### 2. LLM layer tilt (its only allocation power)

- The LLM nudges each layer's budget off baseline. **Magnitude is the agent's decision**,
  justified in the thesis — there is no fixed band cap.
- **Hard guardrails (backstop only):** each layer clamped to **[8%, 35%]**; tilts **sum to
  zero** (reallocation, not leverage). These exist solely to prevent the degenerate
  all-in-one-layer bet (the MU-as-luck failure mode); within them the agent has full
  discretion.
- **Persistence:** a tilt holds until the LLM explicitly declares a **regime shift**. No
  regime change → budgets unchanged → no trading from thesis noise.

### 3. Factor engine (within-layer name selection) — mechanical

- Rank names within each layer by a **factor composite read in layer context** (e.g. Power
  weights trend + capacity growth; Compute weights momentum + revisions).
- **Phase 1 (now): price/momentum factors** — 3/6-month momentum, trend, low-vol —
  derivable entirely from Alpaca data we already pull. Directly addresses the RCA finding
  that churn destroyed trend capture.
- **Phase 2: fundamental factors** — capex growth, EPS revisions, FCF margin, revenue
  growth — behind the same `FactorProvider` interface. Requires a point-in-time
  fundamentals source (e.g. yfinance free tier or a paid API) to backtest without
  look-ahead bias. Does not block Phase 1.
- Take **top 2–3 names per layer** (~10–13 total), weight by factor score, **cap any single
  name at 12%** of book.

### 4. Risk & exposure

- **Fully invested by default.** Replaces the old regime-scaled net-exposure model that
  caused chronic cash drag.
- **Narrow extreme-only risk-off switch:** raise a fixed cash buffer ONLY when an extreme
  objective trigger fires (credit-stress flag AND VIX above threshold). Otherwise 100%
  invested. This is a crash circuit-breaker, not a daily exposure dial.
- **Position cap 12%/name; layer floor 8% / ceiling 35%.**
- **Rebalance bands:** on the weekly run, only trade when drift breaches a threshold (layer
  >3 pts off target, or a name materially off). If nothing breached, do nothing.

### 5. Cadence & execution

- **Daily (passive, no trades):** ingest research, run the thesis pass, detect and log
  regime shifts, accumulate signal. Touches no orders.
- **Weekly trading window (the only one):**
  - **Friday AM** — lock thesis + layer budgets + factor-ranked target basket; execute
    **SELLS** (trims/exits).
  - **Monday AM** — execute **BUYS** into freed-up cash.
  - The rotated slice sits in cash over the weekend (accepted; small for a low-turnover
    book, exposed to the Monday open gap).
- **Single decision path:** one weekly target computed Friday; the launchd-main +
  execution-cron double-fire is **removed**.
- **Fill verification + alerting:** poll each leg until filled/rejected; surface
  unfilled/rejected orders (fixes the silent "stuck accepted" failure).
- **Logging:** every run leaves a complete trace (fixes the lost-log failure).

## Validation & testing

The backtest harness is a first-class deliverable. It must run **ablations** to attribute
edge to each design layer, on point-in-time data, scored on **return, Sharpe, max
drawdown** (the RCA scorecard):

| Variant | Question |
|---------|----------|
| Baseline budgets, no tilt, equal-weight within layer | Does the **thesis tilt alone** beat QQQ? |
| + momentum factor within layer | Does **factor selection** add over the tilt? |
| + LLM layer tilt | Does the **LLM earn its keep**? (the test impossible before) |
| vs QQQ / vs current engine | Net verdict |

**Ship gate:** the LLM-tilt variant ships **only if it beats baseline-no-tilt on Sharpe**.
If it does not, keep the mechanical baseline and the LLM stays advisory (thesis narrative
only).

**Unit tests** for the mechanical core: layer assignment, budget clamp/floor/ceiling/
sum-to-zero, factor ranking, rebalance-band logic, fill verification.

**Forward test:** run on paper through the new weekly cadence before trusting live results.

## What is removed / changed

- Gemini scorer's per-name conviction & weight output → replaced by layer-level thesis +
  tilt only (`scoring/gemini_scorer.py` reworked).
- Regime-scaled net-exposure model → narrow extreme-only risk-off.
- Daily execution + execution cron → single weekly Friday/Monday window.
- New: `LAYER_MAP`, baseline budgets, `FactorProvider` interface (momentum impl),
  portfolio assembler with bands/caps, fill verification, backtest ablation harness.

## Open items

- Calibrate baseline budgets, momentum lookbacks, rebalance-band thresholds, risk-off
  triggers during backtest.
- Choose the Phase-2 point-in-time fundamentals source.
- Confirm per-layer factor-composite weightings.
