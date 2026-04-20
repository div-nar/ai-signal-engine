# AI Signal Engine — Design Spec
**Date:** 2026-04-20  
**Layers:** 1 (Data) + 2 (Signals) of a 5-layer AI Portfolio Management System  
**Status:** Approved for implementation

---

## Overview

A standalone service that ingests forward-looking AI compute signals, scores them using Gemini 2.5 Pro grounded in Aschenbrenner's thesis, and exports structured outputs consumed by the existing `ai-portfolio-backtest` execution layer and future portfolio construction, risk, and monitoring layers.

**Core thesis:** Track whether the AI compute exponential is accelerating, steady, or stalling by reading signals that are causally upstream of stock prices — not contemporaneous news sentiment.

---

## The 5-Layer System (Full Picture)

| Layer | Name | Status |
|---|---|---|
| 1 | Data Ingestion | This spec |
| 2 | Signal Scoring (Gemini) | This spec |
| 3 | Portfolio Construction | Future |
| 4 | Risk Management | Future |
| 5 | Execution + Monitoring | Exists (`execute.py`) |

Layers 1–2 must output a contract that layers 3–5 can consume without modification.

---

## Service Structure

New standalone directory: `ai-signal-engine/` (sibling to `ai-portfolio-backtest/`)

```
ai-signal-engine/
├── ingestion/
│   ├── rss.py             # SemiAnalysis RSS parser (full content via feed)
│   ├── arxiv.py           # arXiv cs.AI / cs.LG / cs.AR paper velocity
│   └── transcripts.py     # SEC EDGAR 8-K earnings transcripts
├── scoring/
│   └── gemini_scorer.py   # Gemini 2.5 Pro → structured portfolio output
├── db.py                  # SQLite schema + read/write helpers
├── config.py              # Sources, ticker universe, weights, schedules
├── export.py              # Writes JSON output files for strategy.py + future layers
└── main.py                # Orchestrator — single run ingests, scores, exports
```

---

## Data Pipeline (Layer 1)

### Signal Philosophy

Signals are organised across the **full AI value chain** — not just the chip supply chain. This ensures Gemini has visibility into all six layers of the stack and can surface opportunities across the entire 90-stock universe, not just semiconductors.

```
Compute → Power → Infrastructure → Platform → Application → Domain
```

Each layer has its own leading indicators and its own set of stocks that benefit when that layer expands.

### Data Sources by Value Chain Layer

**Layer: Compute** — chips being designed and manufactured (lead: 2–4 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| ASML quarterly order backlog | EUV machines → future fab capacity | NVDA, AMD, AVGO, AMAT, LRCX, TSM, ASML |
| TSMC monthly revenue by node | Advanced node utilization | TSM, NVDA, AMD, AVGO |
| Micron/SK Hynix capacity outlook | HBM supply for training runs | MU, LRCX, AMAT |

**Layer: Power** — energy being committed to AI infrastructure (lead: 2–3 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| Power purchase agreements (PPAs) | Multi-year clean energy commitments by hyperscalers | VST, CEG, NEE, AES, ETN, PWR |
| Utility interconnection queue | Grid capacity reserved for datacenters | EIX, NRG, VST, CEG |
| Vertiv / cooling equipment orders | Datacenter thermal infra buildout | VRT, GE |

**Layer: Infrastructure** — physical compute capacity being built (lead: 1–2 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| Datacenter REIT leasing announcements | Space committed to hyperscalers | DLR, EQIX, IRM, AMT |
| Hyperscaler CapEx guidance | GPU cluster purchase commitments | MSFT, AMZN, GOOGL, META |
| Networking equipment orders | InfiniBand/Ethernet switch demand | ANET, AVGO, CSCO |

**Layer: Platform** — cloud and AI services being built (lead: 1–2 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| Cloud CapEx guidance (EDGAR 8-K) | MSFT/AMZN/GOOGL/META investment signals | MSFT, AMZN, GOOGL, META, BABA |
| GPU spot pricing index | Real-time supply/demand tension | NVDA, AMD, cloud providers |
| arXiv paper compute intensity | Scale of next training runs | NVDA, AMD, TSM, MU |

**Layer: Application** — AI being deployed in enterprise software (lead: 1–2 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| SemiAnalysis articles (RSS) | Curated AI/semiconductor analysis | Broad — Gemini maps mentions to tickers |
| Enterprise AI revenue disclosures | Copilot/AI feature adoption rates | CRM, NOW, PLTR, INTU, ADBE, SAP |
| arXiv application papers | AI adoption in specific domains | INFY, ACN, IBM |

**Layer: Domain** — AI transforming specific industries (lead: 2–4 quarters)

| Source | What it measures | Benefits |
|---|---|---|
| AI drug discovery announcements | Clinical trials using AI | LLY, ISRG, AMGN, GILD |
| Financial AI disclosures | Algorithmic trading, fraud detection | GS, JPM, V, MA, IBKR |
| Industrial automation orders | AI-driven robotics, defence AI | HON, GE, AXON, RTX, LMT |

### Ingestion Rules
- Deduplicate by URL / arXiv ID / EDGAR accession number
- Only process documents not yet in `documents` table
- Store full content — Gemini needs it for grounded reasoning
- Tag each document with its value chain layer on ingestion

---

## SQLite Schema (Layer 1 → Layer 2)

```sql
-- Raw ingested documents
CREATE TABLE documents (
  id           INTEGER PRIMARY KEY,
  source       TEXT NOT NULL,  -- 'rss' | 'arxiv' | 'edgar'
  title        TEXT NOT NULL,
  url          TEXT UNIQUE NOT NULL,
  published_at TIMESTAMP,
  content      TEXT,
  ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  scored       BOOLEAN DEFAULT FALSE,
  value_chain_layer TEXT   -- 'compute'|'power'|'infrastructure'|'platform'|'application'|'domain'
);

-- Per-document scores (intermediate)
CREATE TABLE scores (
  id         INTEGER PRIMARY KEY,
  doc_id     INTEGER REFERENCES documents(id),
  p_delta    REAL,            -- contribution to p (-1.0 → +1.0)
  stock_scores TEXT,          -- JSON: { ticker: conviction_float }
  thesis_tags  TEXT,          -- JSON: ["compute_scaling", "power_infra", ...]
  scored_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Final aggregated signal snapshot (one row per run)
CREATE TABLE signals (
  id                    INTEGER PRIMARY KEY,
  computed_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  -- Layer 3: Portfolio Construction
  p_final               REAL,
  stock_conviction      TEXT,   -- JSON: { ticker: float }
  sector_tilt           TEXT,   -- JSON: { sector: float }
  supply_demand_balance REAL,   -- +ve = demand constrained, -ve = supply constrained
  market_regime         TEXT,   -- 'compute_constrained'|'demand_constrained'|'balanced'|'stalling'

  -- Layer 4: Risk Management
  signal_confidence     REAL,   -- 0→1, how many sources agreed
  thesis_stress         BOOLEAN,-- supply AND demand signals diverging badly
  signal_age_days       INTEGER,-- staleness of freshest signal

  -- Layer 5: Monitoring
  sources_ingested      INTEGER,
  signal_breakdown      TEXT,   -- JSON: { source: p_delta }
  thesis_update         TEXT    -- Gemini's narrative on what changed
);
```

---

## Signal Scoring Layer (Layer 2) — Gemini

### Architecture

All ingested documents are assembled into a single structured context and passed to Gemini 2.5 Pro in one call. Gemini reasons across all signals simultaneously and outputs structured portfolio weights.

```
Documents (RSS + arXiv + EDGAR)
  → assemble signal context
  → single Gemini 2.5 Pro call
  → structured JSON output
  → write to signals table
  → export JSON files
```

### Gemini Prompt Structure

```
[SYSTEM]
You are a portfolio manager for an AI-focused long-only equity fund.
Your thesis is Aschenbrenner's: AI is on an exponential compute trajectory
toward AGI. Your job is to identify which stocks in the universe will
benefit most from the NEXT 1-4 quarters of AI compute expansion.

Universe: {90 tickers with sectors}
Constraints: max 10% per stock, weights sum to 1.0, long-only

[SIGNAL CONTEXT — injected fresh each run, organised by value chain layer]

Compute layer:
  - ASML order backlog: {value vs trend}
  - TSMC advanced node utilization: {value vs trend}
  - Memory (HBM) capacity outlook: {summary}

Power layer:
  - Recent PPA announcements: {hyperscaler clean energy commitments}
  - Utility interconnection queue changes: {summary}
  - Cooling/infrastructure orders: {Vertiv, GE summaries}

Infrastructure layer:
  - Datacenter REIT leasing activity: {DLR, EQIX, IRM announcements}
  - Hyperscaler CapEx guidance: {MSFT/AMZN/GOOGL/META excerpts}
  - Networking equipment demand: {Arista, Broadcom signals}

Platform layer:
  - GPU spot pricing: {current vs 30d avg}
  - arXiv compute intensity trend: {paper velocity stats}
  - Cloud availability zone announcements: {summary}

Application layer:
  - Enterprise AI revenue disclosures: {Copilot, AI feature adoption}
  - SemiAnalysis articles: {title + full content}

Domain layer:
  - AI drug discovery / clinical trial announcements
  - Financial AI disclosures
  - Industrial automation / defence AI orders

[TASK]
Given these forward-looking signals and Aschenbrenner's thesis, output
portfolio weights for next week. Weight stocks that will benefit from
what is being *committed to* today, not what has already happened.
```

### Structured Output Schema

```python
{
  "p_score":       float,    # 0→1 Aschenbrenner probability this week
  "market_regime": str,      # "compute_constrained" | "demand_constrained"
                             # | "balanced" | "stalling"
  "supply_demand_balance": float,  # +ve = demand > supply
  "portfolio": [
    {
      "ticker":     str,
      "weight":     float,   # all sum to 1.0
      "conviction": float,   # 0→1
      "reasoning":  str      # 1-2 sentences, forward-looking
    }
  ],
  "signal_confidence": float,  # 0→1, how consistent were signals
  "thesis_stress":     bool,   # true if signals strongly diverge
  "thesis_update":     str     # what changed vs last run (narrative)
}
```

### Guardrails (applied after Gemini output)
- Enforce max 10% per stock (hard cap)
- Enforce min 2% for hedge sectors (Health Care, Consumer Staples)
- Enforce max turnover vs previous run (configurable, default 20%)
- Reject output if weights don't sum to 1.0 ± 0.01 (retry once)

---

## Output Contract (Layer 2 → Layers 3–5)

Three JSON files written to `ai-portfolio-backtest/data/`:

**`p_estimate.json`** — already consumed by strategy.py
```json
{ "p": 0.82, "generated_at": "2026-04-20T06:00:00" }
```

**`stock_signals.json`** — new, consumed by strategy.py (blended with momentum)
```json
{
  "generated_at": "2026-04-20T06:00:00",
  "conviction": { "NVDA": 0.85, "ASML": 0.78, "MU": 0.71, ... },
  "reasoning":  { "NVDA": "ASML backlog surge signals...", ... }
}
```

**`market_regime.json`** — new, consumed by layers 3–5
```json
{
  "generated_at":          "2026-04-20T06:00:00",
  "market_regime":         "compute_constrained",
  "supply_demand_balance": 0.34,
  "sector_tilt":           { "Information Technology": 0.05, "Materials": 0.08 },
  "signal_confidence":     0.76,
  "thesis_stress":         false,
  "thesis_update":         "ASML order backlog up 18% QoQ...",
  "signal_breakdown":      {
    "compute": 0.25, "power": 0.15, "infrastructure": 0.20,
    "platform": 0.20, "application": 0.10, "domain": 0.10
  }
}
```

### strategy.py Integration (minimal change required)
Blend Gemini conviction with existing momentum scores:
```python
final_score[ticker] = 0.7 * momentum_score[ticker] + 0.3 * conviction[ticker]
```
If `stock_signals.json` absent → pure momentum fallback (backwards compatible).

---

## Scheduling

| Trigger | When | Why |
|---|---|---|
| Daily run | 6am ET weekdays | Fresh signals before market open |
| Earnings trigger | After NVDA/MSFT/AMZN/GOOGL/META reports | CapEx language is highest-signal event |
| TSMC monthly | Mid-month | Node utilization update |
| On-demand | `python main.py --force` | Manual override |

---

## What This Is NOT

- Not a backtesting system (that's `backtest_v2.py`)
- Not an execution system (that's `execute.py`)
- Not a real-time tick-by-tick system — daily cadence is intentional
- Not predicting short-term price moves — forecasting 1–4 quarter compute trends
