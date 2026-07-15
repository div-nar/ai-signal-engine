# How the AI Signal Engine Works — the "Layer Cake"

## What it does

The engine turns a view of the AI-infrastructure buildout into a systematic, fully-invested
equity portfolio, executed on Alpaca paper trading. The thesis is Aschenbrenner's: AI runs on
an exponential compute trajectory driven by a **physical buildout supercycle** — power, silicon,
chips, datacenters, and the software that captures the value on top.

The design principle, learned the hard way from the retired v1 engine (see the
[retirement report](reports/2026-06-20-ai-signal-engine-v1-retirement-report.md)), is:

> **Let the LLM do only the one job it might do well — read the value-chain thesis at the
> layer level — and make everything below that mechanical, low-churn, and reliably executed.**

v1 let a language model pick individual stocks and conviction-weight them. A full post-mortem
found that carried **zero** selection signal (Spearman IC ≈ −0.008), closet-indexed QQQ with
extra churn, and bled ~20 points to cash drag and execution failures. Layer cake keeps the LLM
strictly above the "layer line" and makes name selection deterministic and backtestable.

---

## The five-layer cake

The portfolio is organised as a value chain, physical → value-capture. Every eligible ticker
is assigned to exactly one layer (`strategy/layers.py::LAYER_MAP`). Names outside the AI-infra
thesis (healthcare, financials, staples, energy majors, defense) are intentionally unmapped and
therefore ineligible.

| # | Layer | Role | Example names |
|---|-------|------|---------------|
| 1 | **Power & Energy** | the electrons: grid, generation, electrical gear | VST, CEG, NRG, NEE, ETN, PWR, GEV, GE |
| 2 | **Fabrication & Materials** | making the silicon: foundry, semicap, EDA, materials | TSM, ASML, AMAT, LRCX, KLAC, SNPS, CDNS, LIN |
| 3 | **Compute & Silicon** | accelerators & memory | NVDA, AMD, AVGO, MU, MRVL, QCOM, ARM |
| 4 | **Infrastructure & Networking** | where compute lives: datacenters, REITs, cooling, interconnect | VRT, EQIX, DLR, IRM, AMT, CSCO |
| 5 | **Platform & Application** | value capture: hyperscalers & software (QQQ's core) | MSFT, GOOGL, AMZN, META, ORCL, PLTR, NOW |

The structural thesis **overweights layers 1–4** (which QQQ barely holds and where the
MU/VST-type winners live) and **underweights layer 5** (QQQ's concentrated mega-cap core). QQQ
mega-caps still enter — but only when they win their own layer's factor rank, never hardcoded in.

---

## Pipeline overview

```
                          ┌─── LLM (layer-level ONLY) ───┐
Ingest ─► Thesis pass ────►  5 layer tilts + regime flag  │
(RSS/HF/EDGAR)               (sum-to-zero reallocation)   │
                          └──────────────┬────────────────┘
                                         ▼
              Layer budgets = baseline ± tilt   (clamp [8%,35%], renormalize to 1.0)
                                         ▼
              Factor engine (MECHANICAL): momentum-rank names WITHIN each layer, take top 3
                                         ▼
              Assembler: split each layer's budget by factor score, cap 12%/name, fully invested
                                         ▼
              Execution: weekly — Friday-AM sells, Monday-AM buys, fill-verified
```

The LLM's *entire* output is **5 layer tilts + a regime flag + a thesis paragraph** — no
per-name weights. Everything below the layer line is deterministic.

---

## 1. Ingestion (`ingestion/`)

Three sources, run on every passive/sell cycle, deduplicated by URL in SQLite (`signals.db`)
and embedded into ChromaDB for semantic retrieval:

| Source | What |
|---|---|
| RSS feeds | SemiAnalysis, Fabricated Knowledge, Stratechery, Platformer, Latent Space, Import AI |
| HuggingFace Daily Papers | top arXiv papers from the HF API |
| SEC EDGAR 8-Ks | hyperscaler disclosures (MSFT, AMZN, GOOGL, META, NVDA) |

## 2. LLM thesis pass (`scoring/thesis_scorer.py`)

Gemini acts as the fund's macro strategist. Its **only** job is to decide how to tilt capital
across the five layers relative to a neutral baseline, based on where the binding bottleneck of
the AI buildout is right now. It returns strict JSON:

```json
{
  "layer_tilt": {"power": …, "fabrication": …, "compute": …,
                 "infrastructure": …, "platform": …},
  "market_regime": "compute_constrained | demand_constrained | balanced | stalling | shipping_bottleneck | credit_stress",
  "regime_shift": false,
  "signal_confidence": 0.0-1.0,
  "thesis_update": "1-3 sentences on the current bottleneck and what changed"
}
```

The tilt is recentered to sum to exactly zero (a pure reallocation, never leverage) before it
touches the budgets.

## 3. Layer budgets (`strategy/budgets.py`)

```
budgets = baseline + tilt,  clamped to [8%, 35%] per layer,  renormalized to sum to 1.0
```

The baseline encodes the structural tilt (power 20% · fabrication 20% · compute 25% ·
infrastructure 15% · platform 20%). The `[8%, 35%]` clamp is a **backstop only** — it exists to
prevent the degenerate all-in-one-layer bet (v1's MU-as-luck failure mode); within it the LLM
has full discretion to lean hard when the thesis is strong.

## 4. Factor engine — mechanical name selection (`strategy/factors.py`)

Within each layer, names are ranked by a price-momentum factor and the **top 3** are taken:

- **Momentum:** classic 12-1 construction — total return over a 126-trading-day lookback,
  skipping the most recent 21 days to avoid short-term reversal.
- Uses only the Alpaca price panel already pulled — no new data dependency.
- Phase 2 (separate plan) adds fundamental factors (capex growth, EPS revisions, FCF margin)
  behind the same dict-returning contract, without look-ahead bias.

## 5. Portfolio assembly (`strategy/assemble.py`)

Each layer's budget is split across its top-3 names in proportion to their factor score, then
every name is **capped at 12%** of the book and the whole thing is renormalized. The result is
a **fully-invested** target of ~10–13 names (replacing v1's chronic ~73% net-exposure cash drag).

## 6. Risk controls (`strategy/risk.py`)

- **Extreme-only risk-off switch:** raise a fixed cash buffer *only* when an objective crash
  trigger fires (credit-stress flag **and** VIX above threshold). This is a circuit-breaker, not
  a daily exposure dial — the default is 100% invested.
- **Rebalance bands:** only trade when drift breaches a threshold (a layer >3 pts off target, or
  a name materially off). If nothing breached, do nothing — this throttles turnover.

## 7. Execution & cadence (`execution/alpaca.py`, `ops/launchd/`)

A single weekly decision path, driven by three launchd jobs:

| Job | When | Mode | Action |
|---|---|---|---|
| passive | Tue–Fri, PM | `passive` | ingest research, run thesis pass, compute + persist the weekly target — **no trades** |
| sell | Friday, PM | `sell` | lock the target, execute the **sell** leg (trims/exits) |
| buy | Monday, PM | `buy` | execute the **buy** leg into freed-up cash, record a portfolio snapshot |

The rotated slice sits in cash over the weekend (accepted — small for a low-turnover book).
Each leg polls Alpaca until every order is filled or rejected, and surfaces anything stuck —
fixing v1's silent "accepted-but-unfilled" failures.

---

## Validation

The backtest harness (`backtest/`) runs **ablations** on point-in-time price data to attribute
edge to each design layer, scored on return / Sharpe / max-drawdown:

| Variant | Question it answers |
|---|---|
| baseline budgets, equal weight within layer | Does the **thesis tilt alone** beat QQQ? |
| + momentum factor within layer | Does **factor selection** add over the tilt? |
| + LLM layer tilt (forward/paper phase) | Does the **LLM earn its keep**? |

**Ship gate:** the LLM-tilt variant is trusted only if it beats the mechanical
baseline-no-tilt on Sharpe. If it does not, the mechanical baseline stands and the LLM stays
advisory (thesis narrative only).

---

## Key files

| File | Role |
|---|---|
| `main.py` | Entrypoint — routes `--mode passive/sell/buy` |
| `orchestrate.py` | Weekly target: thesis → budgets → momentum → persisted target |
| `config.py` | RSS feeds, EDGAR tickers, ticker universe, Gemini config |
| `strategy/layers.py` | Five-layer taxonomy, `LAYER_MAP`, baseline budgets |
| `strategy/budgets.py` | Apply LLM tilt under the floor/ceiling guardrails |
| `strategy/factors.py` | Within-layer momentum ranking |
| `strategy/assemble.py` | Budget × factor → capped, fully-invested weights |
| `strategy/pipeline.py` | Compose the mechanical target from budgets + prices |
| `strategy/risk.py` | Extreme-only risk-off switch + rebalance bands |
| `scoring/thesis_scorer.py` | Gemini layer-thesis pass (tilt + regime only) |
| `pricing/history.py` | Alpaca daily price panel |
| `execution/alpaca.py` | Fill-verified Friday-sell / Monday-buy legs |
| `backtest/` | Ablation harness + evaluation metrics |

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini thesis pass + embeddings |
| `ALPACA_API_KEY` | Yes | Paper trading account + market data |
| `ALPACA_SECRET_KEY` | Yes | Paper trading account |

Secrets live in a gitignored `.env` at the repo root; `run.sh` loads it automatically.

*Paper-trading research project; not investment advice.*
