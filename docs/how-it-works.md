# How the AI Signal Engine Works — the "Layer Cake"

## What it does

The engine turns a view of the AI-infrastructure buildout into a systematic, fully-invested
equity portfolio, executed on Alpaca paper trading. The thesis is Aschenbrenner's: AI runs on
an exponential compute trajectory driven by a **physical buildout supercycle** — power, silicon,
chips, datacenters, and the software that captures the value on top.

The engine has two autonomy modes (`config.LLM_AUTONOMY`, currently **full**):

- **`full`** — the LLM is the portfolio manager. It sees the current book, the momentum
  ranks, the research, and its own past theses; searches the vector archive with its own
  queries; and may author the target weights directly. No numeric clamps — *trade sensibly*
  is enforced by prompt (long-only, no leverage, turnover-aware, catalyst-required
  concentration, universe-restricted), and every decision is persisted for ablation.
- **`guardrailed`** — the bounded decision surface described below, with every dial clamped
  in code. One config line switches back.

The guardrailed design principle, learned the hard way from the retired v1 engine (see the
[retirement report](reports/2026-06-20-ai-signal-engine-v1-retirement-report.md)), was:

> **The LLM directs the portfolio only through a bounded, clamped, fully-logged decision
> surface — and the spine of name selection stays mechanical, low-churn, and reliably
> executed.**

v1 let a language model pick individual stocks and conviction-weight them, free-form. A full
post-mortem found that carried **zero** selection signal (Spearman IC ≈ −0.008), closet-indexed
QQQ with extra churn, and bled ~20 points to cash drag and execution failures. Layer cake
inverts the relationship: momentum ranks the names, and the LLM's judgments — where the
bottleneck is, how concentrated to be, which names deserve emphasis, when to hold cash, whether
this week's trade is worth its churn — enter only through guardrailed dials that are persisted
on every target, so each one can be ablated against the pure-mechanical baseline.

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
Ingest (RSS/HF/EDGAR) ──► SQLite + ChromaDB (embedded, semantic archive)
                                         │
                          ┌──────────────▼───────────────────┐
                          │  LLM thesis pass (AGENTIC)       │
                          │  • seed docs + own thesis memory │
                          │  • may search the archive with   │
                          │    its own queries (≤2 rounds)   │
                          │  • outputs a bounded decision    │
                          │    surface (all clamped, logged) │
                          └──────────────┬───────────────────┘
                                         ▼
              Layer budgets = baseline ± tilt   (clamp [8%,35%], renormalize to 1.0)
                                         ▼
              Factor engine (MECHANICAL): momentum-rank names WITHIN each layer
              (LLM may emphasize/veto names within [0.5×,1.5×]; top 2–4 per layer)
                                         ▼
              Assembler: split each layer's budget by factor score, cap 12%/name
                                         ▼
              Trade gate: LLM urgency × drift bands — trade, or skip the week
                                         ▼
              Execution: weekly — Friday-AM sells, Monday-AM buys, fill-verified
```

The LLM's output is a **bounded decision surface** — layer tilts, per-layer concentration,
name emphasis, cash buffer, rebalance urgency — every field clamped by mechanical guardrails
and logged in the persisted target so each power can be ablated against the pure-mechanical
baseline. Name *ranking* stays mechanical momentum.

---

## 1. Ingestion (`ingestion/`, `chroma_store.py`)

Three sources, run on every passive/sell cycle, deduplicated by URL in SQLite (`signals.db`)
and **written through to ChromaDB** (`data/chroma`, Gemini `gemini-embedding-001` embeddings)
so the archive is semantically searchable:

| Source | What |
|---|---|
| RSS feeds | SemiAnalysis, Fabricated Knowledge, Stratechery, Platformer, Latent Space, Import AI |
| HuggingFace Daily Papers | top arXiv papers from the HF API |
| SEC EDGAR 8-Ks | hyperscaler disclosures (MSFT, AMZN, GOOGL, META, NVDA) |

Two collections: `research_docs` (every ingested document) and `macro_signals` (the LLM's own
past theses — its memory). A one-time backfill indexes anything already in SQLite; if the
vector store is unavailable the run degrades gracefully to recency-only docs, never blocking a
trading window.

## 2. LLM thesis pass — agentic (`scoring/thesis_scorer.py`)

Gemini acts as the fund's macro strategist. Each run it is shown **seed documents** (recent
research), its **own recent thesis history** (semantic query over `macro_signals`), and the
prior layer budgets. If the evidence is insufficient it can **search the research archive with
its own queries** before deciding:

```json
{"action": "search", "queries": ["HBM supply constraints", "datacenter power PPA"]}
```

…up to 3 queries per round, 2 rounds max; retrieved docs are deduplicated against what it has
already seen and fed back. Every query and hit count is recorded in a `retrieval_log` on the
persisted target. Its final answer is strict JSON over the whole decision surface:

```json
{
  "layer_tilt": {"power": …, "fabrication": …, "compute": …,
                 "infrastructure": …, "platform": …},
  "layer_top_n": {"compute": 2, "power": 4},
  "name_adjustments": {"NVDA": 1.3, "MU": 0},
  "cash_buffer": 0.0,
  "rebalance_urgency": "urgent | normal | hold",
  "market_regime": "compute_constrained | demand_constrained | balanced | stalling | shipping_bottleneck | credit_stress",
  "regime_shift": false,
  "signal_confidence": 0.0-1.0,
  "thesis_update": "1-3 sentences on the current bottleneck and what changed"
}
```

Guardrails in `guardrailed` mode: tilts recentered to sum to exactly zero; budgets clamped to
[8%, 35%]; `layer_top_n` clamped to [2, 4]; `name_adjustments` restricted to the thesis
universe and clamped to [0.5×, 1.5×] (0 = veto); `cash_buffer` clamped to [0, 30%].

In **full autonomy** the same JSON gains an optional `"target_weights"` field — the complete
desired book, used directly instead of the dial pipeline. Validation is correctness-only
(tickers must be in `TICKER_UNIVERSE` so hallucinated symbols never reach the broker; weights
positive; normalized to 1 with any shortfall held as implied cash) — no clamps. If the LLM
stays with dials, the pipeline runs unclamped (all-in-one-layer allowed, any top-n, any cash
level). `weights_source` records which path produced each target.

After the target persists, the thesis is embedded back into `macro_signals` so the next pass
remembers it.

## 3. Layer budgets (`strategy/budgets.py`)

```
budgets = baseline + tilt,  clamped to [8%, 35%] per layer,  renormalized to sum to 1.0
```

The baseline encodes the structural tilt (power 20% · fabrication 20% · compute 25% ·
infrastructure 15% · platform 20%). The `[8%, 35%]` clamp is a **backstop only** — it exists to
prevent the degenerate all-in-one-layer bet (v1's MU-as-luck failure mode); within it the LLM
has full discretion to lean hard when the thesis is strong.

## 4. Factor engine — mechanical name selection (`strategy/factors.py`)

Within each layer, names are ranked by a price-momentum factor and the **top 2–4** (LLM
concentration dial, default 3) are taken:

- **Momentum:** classic 12-1 construction — total return over a 126-trading-day lookback,
  skipping the most recent 21 days to avoid short-term reversal.
- **LLM name emphasis:** the bounded `name_adjustments` multiply a name's momentum score
  ([0.5×, 1.5×]; a boost always *improves* the score, even when momentum is negative) or veto
  it for the week. Unadjusted names pass through untouched — momentum stays the ranking spine.
- Uses only the Alpaca price panel already pulled — no new data dependency.
- Phase 2 (separate plan) adds fundamental factors (capex growth, EPS revisions, FCF margin)
  behind the same dict-returning contract, without look-ahead bias.

## 5. Portfolio assembly (`strategy/assemble.py`)

Each layer's budget is split across its top names in proportion to their factor score, then
every name is **capped at 12%** of the book and the whole thing is renormalized. The result is
a **fully-invested** target of ~10–16 names (replacing v1's chronic ~73% net-exposure cash drag).

## 6. Risk controls & the trade gate (`strategy/risk.py`, `orchestrate.py`)

- **LLM cash buffer:** normally 0 (fully invested). The thesis pass can raise up to a 30%
  buffer when the research shows systemic stress; both trading legs scale their targets by
  (1 − buffer).
- **Trade gate — LLM urgency × drift bands:** on the Friday leg, `"hold"` skips the week
  outright; `"normal"` trades only if drift breaches the mechanical bands (a name or a layer
  >3 pts off target — `strategy/risk.py::needs_rebalance`); `"urgent"` trades regardless. The
  decision is recorded on the persisted target and **Monday's buy leg honours a skip**, so a
  no-trade week is truly zero-churn.

## 7. Execution & cadence (`execution/alpaca.py`, `ops/launchd/`)

**Daily cadence (current):** one launchd job runs `--mode trade` every market morning
(Mon–Fri ~09:30 ET). Each run ingests research, runs the agentic thesis pass, persists the
target, and — if the LLM says the move is worth its churn — executes sells (fill-verified)
then buys in the same session, no weekend gap. The LLM's `rebalance_urgency` is the daily
throttle: on an unchanged thesis it answers `"hold"` and the day is zero-churn. On-demand runs
any time with `./run.sh --mode trade` (`--force` bypasses the gate).

The weekly split (passive Tue–Fri / sells Friday / buys Monday) remains available as a legacy
alternative — see `ops/launchd/README.md`; only one path is ever loaded at a time.

| Mode | Action |
|---|---|
| `passive` | ingest research, thesis pass, compute + persist target — **no trades** |
| `sell` | compute + persist target, execute the **sell** leg |
| `buy` | execute the **buy** leg from the latest persisted target, record a snapshot |
| `trade` | full same-day rebalance: sells then buys in one session (daily driver) |

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
| `main.py` | Entrypoint — routes `--mode passive/sell/buy/trade`, inits ChromaDB |
| `orchestrate.py` | Weekly target: agentic thesis → budgets → momentum → trade gate → persisted target |
| `config.py` | RSS feeds, EDGAR tickers, ticker universe, Gemini config |
| `chroma_store.py` | Vector store: embed/upsert docs + theses, semantic queries, backfill |
| `strategy/layers.py` | Five-layer taxonomy, `LAYER_MAP`, baseline budgets |
| `strategy/budgets.py` | Apply LLM tilt under the floor/ceiling guardrails |
| `strategy/factors.py` | Within-layer momentum ranking + bounded LLM name emphasis |
| `strategy/assemble.py` | Budget × factor → capped, fully-invested weights (per-layer top-n) |
| `strategy/pipeline.py` | Compose the mechanical target from budgets + prices |
| `strategy/risk.py` | Drift bands behind the trade gate |
| `scoring/thesis_scorer.py` | Agentic Gemini thesis pass — bounded decision surface |
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
