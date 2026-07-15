# AI Signal Engine

An AI-infrastructure **long-only** equity signal engine. It ingests research (RSS feeds,
HuggingFace daily papers, SEC EDGAR filings) and turns a view of the AI buildout into a
systematic equity portfolio, executed on Alpaca paper trading.

**Status:** 🎂 Running the **"Layer Cake"** engine (v2) on Alpaca paper. The first-generation
engine has been **retired** — post-mortem below.

---

## First-generation engine — retired

The v1 engine used a large-language-model scorer to produce a conviction-weighted portfolio,
scaled by a macro regime model, rebalanced daily. It ran on Alpaca paper from **25 Mar 2026**
and finished at **+16.66%** ($100,000 → $116,659).

A full post-mortem — architecture, returns vs the S&P 500 and NASDAQ-100, attribution,
correlation/overlap with the benchmarks, the selection-edge ablations, operational issues, and
the rationale for retirement — is in:

➡️ **[`docs/reports/2026-06-20-ai-signal-engine-v1-retirement-report.md`](docs/reports/2026-06-20-ai-signal-engine-v1-retirement-report.md)**

The v1 launchd job has been unloaded; nothing from v1 trades anymore.

---

## The Layer Cake engine (v2)

v1's post-mortem was blunt: the LLM stock-picker carried **no** selection signal (Spearman
IC ≈ −0.008), the book closet-indexed QQQ with extra churn, the headline +16% was one lucky
name (MU, 58% of P&L, never conviction-sized), and ~20 points were lost to cash drag and
execution failures.

The Layer Cake redesign keeps the AI-infrastructure thesis but changes **who decides what**.
The LLM reads the research **agentically** — it searches the engine's vector archive (ChromaDB)
with its own queries, recalls its own past theses, and sees the current book plus momentum
ranks — then directs the portfolio. Two autonomy modes (`config.LLM_AUTONOMY`):

- **`full` (current):** the LLM is the portfolio manager — it may author the target weights
  outright (long-only, universe-restricted; no other clamps). *Trade sensibly* is enforced by
  prompt: turnover-aware, catalyst-required concentration, cash as a position. Every decision
  — weights, queries, urgency — is persisted per-target for ablation.
- **`guardrailed`:** the bounded dial surface — layer tilts (budgets clamped [8%, 35%]),
  per-layer concentration (2–4 names), name emphasis/veto (0.5×–1.5× on momentum), cash
  buffer ≤30% — with momentum assembling the book mechanically.

Cadence is **daily at the open** (`--mode trade`: sells then buys in one session), throttled by
the LLM's own rebalance urgency — an unchanged thesis means a zero-churn day. `--force`
overrides the gate; a legacy weekly split (Friday sells → Monday buys) remains available.

| # | Layer | Role |
|---|-------|------|
| 1 | **Power & Energy** | the electrons: grid, generation, electrical gear |
| 2 | **Fabrication & Materials** | making the silicon: foundry, semicap, EDA, materials |
| 3 | **Compute & Silicon** | accelerators & memory |
| 4 | **Infrastructure & Networking** | datacenters, REITs, cooling, interconnect |
| 5 | **Platform & Application** | hyperscalers & software (QQQ's core) |

```
Ingest ─► ChromaDB ─► agentic LLM ─► layer budgets ─► momentum rank ─► trade gate
(RSS/HF/EDGAR) (semantic  thesis pass    (baseline ±       (top 2–4/layer,   (urgency ×
                archive)  (searches +     tilt, clamped     cap 12%/name)     drift bands)
                          remembers)      [8%,35%])                                │
                                          weekly: Friday sells ─► Monday buys (fill-verified)
```

The structural tilt overweights layers 1–4 (which QQQ barely holds) and underweights layer 5
(QQQ's concentrated core), while still *holding* mega-caps whenever they win their layer's rank.

➡️ **Full walkthrough: [`docs/how-it-works.md`](docs/how-it-works.md)** ·
design spec: [`docs/superpowers/specs/2026-06-20-layer-cake-redesign-design.md`](docs/superpowers/specs/2026-06-20-layer-cake-redesign-design.md)

---

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/        # test suite
```

Secrets (`GEMINI_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) live in a `.env` file at the
repo root (gitignored); `run.sh` loads it automatically.

## Repository layout

| Path | Purpose |
|---|---|
| `ingestion/` | Research ingestion (RSS, HuggingFace papers, SEC EDGAR) → SQLite |
| `scoring/` | LLM thesis pass — layer tilts + regime flag only |
| `strategy/` | Layer taxonomy, budgets, momentum factor, assembler, risk bands |
| `pricing/` | Live daily price history (Alpaca market data) |
| `execution/` | Fill-verified Alpaca paper-trading legs |
| `orchestrate.py` | Weekly target: thesis → budgets → momentum → persisted target |
| `main.py` | Entrypoint — routes `--mode passive/sell/buy` |
| `backtest/` | Ablation harness + evaluation metrics |
| `db.py` | SQLite schema and accessors |
| `ops/launchd/` | The three weekly scheduler jobs (passive / sell / buy) |
| `docs/` | Design specs, plans, and reports |

*Paper-trading research project; not investment advice.*
