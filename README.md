# AI Signal Engine

An AI-infrastructure **long-only** equity signal engine. Ingests research from RSS feeds, HuggingFace daily papers, and SEC EDGAR filings, computes a macro regime signal from live economic data, then calls Gemini to produce a conviction-weighted portfolio — executed automatically via Alpaca paper trading.

**Status:** 🟢 Working · runs Mon–Fri, fully automated · **+16.0%** since inception (paper) · 85 tests green

---

# 1. Working

Operational status: what's running, how it's scheduled, and how it behaves when things break.

### Schedule

Runs Mon–Fri at **9:00am ET** via a **launchd** agent
(`~/Library/LaunchAgents/com.divnar.ai-signal-engine.plist`), which re-runs jobs missed while
the machine was asleep. The host is on **IST**, so the agent fires at **18:30 IST = 9:00am ET (EDT)**;
downstream portfolio execution runs at 19:00/19:01 IST (9:30/9:31am ET market open) via cron.

> **Note:** launchd/cron use local (IST) time. These IST times are EDT-correct; during EST
> (Nov–Mar) they land ~1h earlier — still pre-market, harmless.

### Run it

```bash
pip install -r requirements.txt

export GEMINI_API_KEY=...        # required (scoring + embeddings)
export FRED_API_KEY=...          # optional — live macro data (free at fred.stlouisfed.org)
export ALPACA_API_KEY=...        # optional — paper execution
export ALPACA_SECRET_KEY=...

python main.py              # full run: ingest → macro → score → execute → export → snapshot
python main.py --force      # re-score even if no new documents (uses last 30 days)
python main.py --dry-run    # ingest + macro + score only, no trades or file writes
```

Secrets live in a `.env` file at the repo root (gitignored); `run.sh` loads it automatically.

### Resilience

Each stage fails soft and never blocks the pipeline:
- ChromaDB backfill / PCA refit / semantic retrieval errors → warn and fall back (SQLite recency), continue.
- Gemini scoring failure (429 / rate limit / timeout) → **hold the current book**, skip rebalance,
  leave documents unscored for retry next run, exit cleanly so the scheduler stays healthy.
- Portfolio snapshot failure → logged, non-fatal.

### Vector store (ChromaDB)

Two persistent collections at `data/chroma/` (gitignored, rebuilt locally):

- **`research_docs`** — every ingested document, embedded with Gemini `gemini-embedding-001`. **At parity with SQLite (1,383 docs).**
- **`macro_signals`** — every signal record produced, embedded from its thesis text.

The scorer queries `research_docs` semantically for the 30 most *relevant* documents (not the 30
most recent), and falls back to SQLite recency if a query embedding fails.

> **Op note:** a Gemini credit outage (Jun 9–12 2026) left `research_docs` ~187 docs behind; this
> was gap-filled and the scorer now degrades gracefully instead of aborting the run. Changing
> `EMBEDDING_MODEL` requires deleting `data/chroma` and rebuilding (vector dimensionality differs).

### Tests

```bash
python -m pytest tests/ -v
```

85 tests covering macro signal computation (incl. PCA composite modifier), scorer guardrails +
ChromaDB fallback, DB migrations, portfolio snapshots, ingestion dedup, and long-only execution.

---

# 2. Model Design

How the engine forms a view and turns it into a book.

### Thesis

Aschenbrenner's exponential compute trajectory: AI is in a physical buildout supercycle driven by chips, memory, power, cooling, datacenters, and networking. The engine identifies which stocks will benefit from the *next* 1–4 quarters of AI infrastructure capex commitments and supply-chain bottlenecks — on both the supply side (semiconductors, power, cooling, interconnects) and demand side (hyperscalers, AI platforms, inference-scaling beneficiaries).

### Pipeline

```
Ingestion (RSS / HuggingFace papers / EDGAR)
        ↓  (SQLite + ChromaDB vector store)
Macro Signal (FRED + yfinance)
   PMI · Semis IP · BDRY shipping · Power→Compute · Copper→Infra · Credit stress
   + PCA composite stress modifier (weekly fit, daily apply)
        ↓
Gemini Scorer
   Macro block + semantically-retrieved, value-chain-layered documents → long book
        ↓
Alpaca Execution
   Long-only rebalance sized by net_exposure_target
        ↓
Export (stock_signals.json · market_regime.json · p_estimate.json) + portfolio snapshot
```

### Macro regimes

Base regime sets a starting net-exposure target; a PCA composite stress modifier then
adjusts it down continuously (replacing the old hard `shipping_bottleneck` threshold).

| Regime | Base Exposure | Trigger |
|---|---|---|
| `credit_stress` | 0.20 | VIX > 25 and HYG 30d return < −2% |
| `balanced` | 0.65 | PMI contracting and copper z-score < −0.5σ |
| `compute_constrained` | 0.80 | Default — thesis intact |

**PCA composite stress modifier** — a PCA fit weekly over five correlated stress signals
(shipping pressure, copper, power/compute lead, VIX, PMI). Today's values project onto PC1
to a stress score; `modifier = −0.25 × stress_percentile`, and
`net_exposure_target = max(0.15, base_exposure + modifier)`. `credit_stress` is exempt
(it already hard-floors at 0.20). This replaced the old `shipping_bottleneck` regime, whose
hard threshold was the main driver of prior underperformance vs. QQQ.

### Scoring & portfolio

- **Gemini scorer** receives the macro block plus the 30 most semantically-relevant documents,
  grouped by value-chain layer, and returns a conviction-weighted long book.
- **Long book** — highest-conviction AI buildout beneficiaries. Weights sum to 1.0, max 10% per position.
- **Net exposure** — set by `net_exposure_target`, scaling long notional down during stress.
- **Execution is long-only**: `long_notional = portfolio_value × net_exposure_target`; per-position
  orders below $500 are skipped; positions dropped from the book are closed first.
- **Universe** — ~95 globally-listed stocks across compute, power & utilities, datacenter REITs,
  hyperscalers, and AI software. Gemini picks freely from this list.

---

# 3. Results

Alpaca paper account, inception **24 Mar 2026** at $100,000. Every run appends a mark-to-market
snapshot to the `portfolio_history` table in `signals.db` (a self-owned equity curve, independent
of Alpaca's retention).

| Metric | Value (as of 16 Jun 2026) |
|---|---|
| Equity | **$116,014** |
| Total return since inception | **+16.0%** |
| Realized (booked to cash) | $15,410 |
| Open / unrealized | $604 |
| Return through prior close (Jun 13, 58 trading days) | +11.5% |
| Max drawdown | −10.3% |

Note: ~97% of profit is already **realized** — the engine rebalances frequently, banking gains
into cash and redeploying ~70% (the `net_exposure_target`), so the open book shows only a small
unrealized figure at any moment.

*Early-stage paper-trading results; not investment advice. Holdings and positioning omitted.*
