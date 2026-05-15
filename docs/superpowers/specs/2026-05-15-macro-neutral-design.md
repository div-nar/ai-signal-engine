# AI Signal Engine: Macro Module + Market-Neutral Execution

**Date:** 2026-05-15
**Status:** Approved

## Problem

The engine's portfolio carries a beta of ~1.45 vs NASDAQ (QQQ) and ~1.85 vs S&P500, with 0.905 correlation to QQQ. This is structural: the long-only mandate, single AI-infra thesis, and 95-ticker universe dominated by large-cap tech means the portfolio largely replicates QQQ with leverage. The engine has no mechanism to reduce market sensitivity when macro conditions deteriorate, and no supply-chain or cross-sector signals to distinguish idiosyncratic from market-wide moves.

## Goal

Add three capabilities:
1. **Supply-chain lead indicators** — quantitative signals from FRED, yfinance proxies, and the Baltic shipping index
2. **Cross-sector spillover detection** — power→compute, copper→infrastructure, credit→risk-regime signals
3. **Market-neutral execution** — Gemini outputs a long book and a short book; Alpaca paper account executes both

## Architecture

```
Ingestion (unchanged)          macro/ (new)
  RSS / arXiv / EDGAR    +    supply_chain.py   ← FRED + yfinance
        ↓                     cross_sector.py   ← Power/Copper/Credit proxies
   documents table             regime.py        ← MacroSignal dict
        ↓                           ↓
        └──────────────────────────→ gemini_scorer.py (modified)
                                         ↓
                              long_weights + short_weights
                                         ↓
                              execution/alpaca.py (new)
                                   longs + shorts → Alpaca paper
```

**What doesn't change:** ingestion pipeline, document/scores DB tables, export.py, the 95-ticker universe.

**New files:** `macro/supply_chain.py`, `macro/cross_sector.py`, `macro/regime.py`, `execution/alpaca.py`

**Modified files:** `gemini_scorer.py`, `main.py`, `db.py`

**DB migration:** two new columns on `signals` — `short_weights TEXT`, `macro_signal TEXT`

## Section 1: Macro Module

### `macro/supply_chain.py`

Fetches and normalises two data streams:

- **FRED** (via `fredapi`): ISM Manufacturing PMI (`NAPM`), semiconductor industrial production (`IPG3344S`), durable goods orders (`DGORDER`). Computes 4-week trend direction for each.
- **yfinance proxies**: `BDRY` (dry bulk shipping ETF) as freight proxy, `FCX` as copper proxy. Returns z-scored 30-day momentum.

Output schema:
```python
{
  "shipping_pressure": float,          # 0.0–1.0, higher = more bottleneck
  "semis_inventory_trend": str,        # "drawing_down" | "building" | "neutral"
  "pmi": float,                        # latest ISM Manufacturing PMI
  "pmi_trend": str,                    # "expanding" | "contracting" | "stable"
}
```

### `macro/cross_sector.py`

Computes three spillover scores from yfinance prices:

- **Power → Compute**: 30-day momentum of `NEE + ETN + PWR` basket vs. `NVDA + TSM`. When utilities outperform semis by >1 std, power buildout is ahead of compute demand — bullish.
- **Copper → Infrastructure**: FCX 30-day return z-score. Rising copper signals physical buildout accelerating.
- **Credit → Risk regime**: `HYG` price trend + `^VIX` level. HYG falling + VIX > 25 triggers `credit_stress = True`.

Output schema:
```python
{
  "power_compute_lead": float,   # z-score, positive = power leading semis
  "copper_infra_lead": float,    # z-score, positive = copper rising
  "credit_stress": bool,         # True when HYG falling and VIX > 25
  "vix_level": float,
}
```

### `macro/regime.py`

Combines supply_chain and cross_sector outputs into a single `MacroSignal`:

```python
{
  "regime": str,                  # "compute_constrained" | "shipping_bottleneck"
                                  # | "credit_stress" | "balanced"
  "regime_confidence": float,     # 0–1
  "net_exposure_target": float,   # 0.2 (credit stress) → 0.8 (all green)
  "supply_chain": { ... },        # supply_chain.py output
  "cross_sector": { ... },        # cross_sector.py output
  "notes": str,                   # 1-line human-readable summary
}
```

**Regime rules (priority order):**
1. `credit_stress` → regime = `"credit_stress"`, `net_exposure_target = 0.20`
2. `shipping_pressure > 0.65` → regime = `"shipping_bottleneck"`, `net_exposure_target = 0.55`
3. `pmi_trend == "contracting"` and `copper_infra_lead < -0.5` → regime = `"balanced"`, `net_exposure_target = 0.65`
4. Default → regime = `"compute_constrained"`, `net_exposure_target = 0.80`

## Section 2: Scorer Changes

### Prompt injection

`build_signal_context()` receives a `macro_signal` parameter. A structured block is prepended before document sections:

```
### MACRO REGIME SIGNAL [computed by quant module — treat as ground truth]
Regime: shipping_bottleneck (confidence: 0.82)
Net exposure target: 0.55 (55% long / 45% short notional)
Supply chain: PMI 49.2 (contracting), shipping pressure 0.74 (elevated), semis inventory drawing_down
Cross-sector: power→compute lead +1.3σ (bullish), copper→infra -0.2σ (neutral), credit stress: FALSE, VIX 18.4
Notes: Freight pressure elevated but credit clean — reduce net exposure moderately, rotate toward supply bottleneck names

[Your portfolio must reflect the net_exposure_target above]
```

### New Gemini output schema

```json
{
  "p_score": 0.91,
  "market_regime": "shipping_bottleneck",
  "portfolio": [
    {"ticker": "NVDA", "weight": 0.12, "conviction": 0.95, "reasoning": "..."}
  ],
  "short_portfolio": [
    {"ticker": "AMD", "weight": 0.08, "conviction": 0.45, "reasoning": "..."}
  ],
  "net_exposure": 0.54,
  "signal_confidence": 0.88,
  "thesis_stress": false,
  "thesis_update": "..."
}
```

Short candidates are drawn from the 95-ticker universe. Gemini picks the lowest-conviction names with the same factor exposure as the longs — genuine pairs, not random hedges.

### Guardrails

- Max long weight per stock: 10% of gross long notional
- Max short weight per stock: 8% of gross short notional
- Gross exposure cap: `Σw_L + Σw_S ≤ 1.80`
- Net exposure tolerance: `|actual_net - η_target| ≤ 0.05`

## Section 3: Portfolio Equations

**Portfolio return:**

$$r_p = \sum_i w_i^L \cdot r_i - \sum_j w_j^S \cdot r_j$$

**Constraints:**

$$\eta_{target} = \sum_i w_i^L - \sum_j w_j^S \quad \text{(net exposure from macro signal)}$$

$$\Gamma = \sum_i w_i^L + \sum_j w_j^S \leq 1.80 \quad \text{(gross cap)}$$

**Gross allocation from net target:**

$$w_{gross,L} = \frac{1.80 + \eta_{target}}{2}, \quad w_{gross,S} = w_{gross,L} - \eta_{target}$$

**Weight derivation:**

$$w_i^L = \frac{conviction_i}{\sum_k conviction_k} \cdot w_{gross,L}$$

$$w_j^S = \frac{1 - conviction_j}{\sum_k (1-conviction_k)} \cdot w_{gross,S}$$

**Resulting beta:**

$$\beta_p = \sum_i w_i^L \cdot \beta_i - \sum_j w_j^S \cdot \beta_j \approx \eta_{target} \cdot \bar{\beta}_{book}$$

At `η_target = 0.55`: `β_p ≈ 0.55 × 1.45 ≈ 0.80` vs. the current 1.45.

## Section 4: Alpaca Execution

### `execution/alpaca.py`

Absorbs `get_alpaca_positions()` from `main.py` and adds a two-sided rebalance function.

**Position read — richer output:**
```python
{
  "longs": {"NVDA": 0.12, ...},
  "shorts": {"AMD": 0.07, ...},
  "net_exposure": 0.54,
  "gross_exposure": 1.40,
  "portfolio_value": 94230.0,
}
```

**Rebalance logic (order matters — closes before opens):**
1. Close any positions not in either target book
2. For longs: if currently short → close short, open long; else adjust size
3. For shorts: if currently long → close long, open short; else adjust size

**Order type:** market orders, 0.3s delay between legs.

**Pre-flight guardrails:**
- Gross > 1.80 → reject rebalance, log warning
- Order size < 0.5% of portfolio → skip (avoid churn)
- Net drift > 0.05 from `η_target` → log warning, execute anyway

### `main.py` orchestration

```
1. Ingest              (unchanged)
2. Compute MacroSignal (new)
3. Score via Gemini    (modified — receives MacroSignal)
4. Persist signal      (unchanged, writes short_weights + macro_signal columns)
5. Execute rebalance   (new)
6. Export              (unchanged)
```

## Out of Scope (v2)

- VectorDB / RAG retrieval for semantic document search
- Per-document `p_delta` scoring (scores table currently empty)
- Live account execution (paper only for now)
- Freightos direct API (using BDRY ETF proxy instead)
