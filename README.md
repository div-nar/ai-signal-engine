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

The Layer Cake redesign keeps the AI-infrastructure thesis but changes **who decides what**:

> The LLM does only the one job it might do well — reading the value-chain thesis at the
> **layer** level. Everything below that is mechanical, fully invested, low-churn, and
> reliably executed.

The portfolio is a five-layer value chain, physical → value-capture. The LLM's *entire* output
is a set of **layer tilts** (how much to over/underweight each layer) plus a regime flag — never
per-name weights. Below the layer line, a mechanical momentum model picks the names.

| # | Layer | Role |
|---|-------|------|
| 1 | **Power & Energy** | the electrons: grid, generation, electrical gear |
| 2 | **Fabrication & Materials** | making the silicon: foundry, semicap, EDA, materials |
| 3 | **Compute & Silicon** | accelerators & memory |
| 4 | **Infrastructure & Networking** | datacenters, REITs, cooling, interconnect |
| 5 | **Platform & Application** | hyperscalers & software (QQQ's core) |

```
Ingest ─► LLM thesis pass ─► layer budgets ─► momentum-rank within each layer
(RSS/HF/EDGAR)  (5 tilts +      (baseline ±      (top 3, cap 12%/name,
                regime flag)    tilt, clamped)    fully invested)
                                                        │
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
