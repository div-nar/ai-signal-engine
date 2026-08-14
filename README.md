# AI Signal Engine

An AI-infrastructure **long-only** equity signal engine. It ingests research (RSS feeds,
HuggingFace daily papers, SEC EDGAR filings) plus market/macro news, and turns a view of the AI
buildout into a systematic equity portfolio, executed on Alpaca paper trading. The thesis is
written by a local **opencode** LLM; retrieval embeds locally — no third-party AI API keys.

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
The LLM reads the research **agentically** — it searches the engine's vector archive with its
own queries, recalls its own past theses, and sees the current book, momentum ranks, live
macro (rates, VIX, dollar, oil, gold) and upcoming earnings — then directs the portfolio.
Two autonomy modes (`config.LLM_AUTONOMY`):

- **`full` (current):** the LLM is the portfolio manager — it may author the target weights
  outright. The investable universe is the **whole market**: any liquid, fractionable US
  equity, validated against the broker's tradable list so no hallucinated ticker is sent
  (`WHOLE_MARKET`). It is **anchored** to the AI layers by the prompt — that's its edge — and
  must give a reason for any non-AI name. Cash is a single lever: whatever the weights leave
  unallocated (there is no separate cash dial). *Trade sensibly* is prompt-enforced:
  turnover-aware, catalyst-required concentration. Every decision — weights, queries, urgency
  — is persisted per-target for ablation.
- **`guardrailed`:** the bounded dial surface — layer tilts (budgets clamped [8%, 35%]),
  per-layer concentration (2–4 names), name emphasis/veto (0.5×–1.5× on momentum), cash
  buffer ≤30% — with momentum assembling the book mechanically.

The **thesis LLM is [opencode](https://opencode.ai)** (`opencode-go/qwen3.7-max`), invoked as a
local CLI — no third-party API key in the process. Momentum is scored across the **whole
market** (S&P 500 ∪ the AI universe, ~500 names; `WHOLE_MARKET_MOMENTUM`) so the model sees
cross-market strength, not just AI. Semantic retrieval embeds **locally** (fastembed /
`nomic-embed-text-v1.5`, 768-dim, offline) — no embeddings API.

Cadence is **daily at the open** (`--mode trade`: sells then buys in one session), throttled by
the LLM's own rebalance urgency — an unchanged thesis means a zero-churn day. `--force`
overrides the gate. One launchd agent, one decision path.

| # | Layer | Role |
|---|-------|------|
| 1 | **Power & Energy** | the electrons: grid, generation, electrical gear |
| 2 | **Fabrication & Materials** | making the silicon: foundry, semicap, EDA, materials |
| 3 | **Compute & Silicon** | accelerators & memory |
| 4 | **Infrastructure & Networking** | datacenters, REITs, cooling, interconnect |
| 5 | **Platform & Application** | hyperscalers & software (QQQ's core) |

```
Ingest ──────► vector archive ──► agentic LLM ──► target book ──► trade gate ──► Alpaca
RSS (AI blogs, (local nomic       (opencode:      whole-market,    (urgency ×      (daily,
market & macro  embeddings,        searches +      AI-anchored;     drift bands)    same-session
news), HF        offline)          remembers)      cash = 1−Σw)                     sell → buy)
papers, EDGAR;                     + macro, earnings,
structured macro                   momentum ranks
& earnings
```

The structural tilt overweights layers 1–4 (which QQQ barely holds) and underweights layer 5
(QQQ's concentrated core), while still *holding* mega-caps whenever they win their layer's rank.
A live dashboard reads the account from Alpaca on every load — equity vs QQQ/SPY, the layer-cake
allocation, holdings vs target, and the engine's last decision:

➡️ **[layercake-dashboard.vercel.app](https://layercake-dashboard.vercel.app)** (source in `ops/web/`) ·
full walkthrough: [`docs/how-it-works.md`](docs/how-it-works.md)

---

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/                 # test suite
./run.sh --mode passive                 # safe dry run: ingest + compute a target, NO trades
./run.sh --mode trade                   # live: same-session sell then buy (real paper orders)
python scripts/rebuild_chroma.py        # rebuild the vector index (after changing embed model)
```

Only broker secrets live in a `.env` at the repo root (gitignored; `run.sh` loads it):
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. The thesis LLM authenticates through the **opencode**
CLI itself (its own `opencode auth`), and embeddings run locally — so there are **no AI API
keys** in this project. There is **no dry-run flag**: `--mode passive` is the no-trade rehearsal.

## Repository layout

| Path | Purpose |
|---|---|
| `ingestion/` | Research ingestion (RSS, HF papers, SEC EDGAR) + `macro.py` (rates/VIX/FX/commodities, earnings) → SQLite |
| `scoring/` | Agentic thesis pass via the opencode CLI — may author the full target book |
| `strategy/` | Layer taxonomy, budgets, momentum factor, assembler, risk bands |
| `pricing/` | Live daily price history (Alpaca market data), batched for the whole-market universe |
| `execution/` | Fill-verified Alpaca legs + tradable-universe lookup |
| `chroma_store.py` | Local (fastembed/nomic) vector store — embed, upsert, query |
| `orchestrate.py` | Target pipeline: thesis → weights (LLM-direct or dial+momentum) → persisted target |
| `export_targets.py` | Writes the latest target + decision to `ops/web/targets.json` for the dashboard |
| `main.py` | Entrypoint — routes `--mode passive/sell/buy/trade` |
| `backtest/` | Mechanical ablation harness + evaluation metrics |
| `db.py` | SQLite schema and accessors |
| `ops/web/` | Live Vercel dashboard (`index.html` + `api/data.py`) |
| `ops/launchd/` | Scheduler: the daily `trade` agent (legacy weekly passive/sell/buy retained) |
| `docs/` | Design specs, plans, and reports |

*Paper-trading research project; not investment advice.*
