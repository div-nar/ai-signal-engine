# AI Signal Engine

An AI-infrastructure **long-only** equity signal engine. Ingests research from RSS feeds, HuggingFace daily papers, and SEC EDGAR filings, computes a macro regime signal from live economic data, then calls Gemini to produce a conviction-weighted portfolio — executed automatically via Alpaca paper trading.

## Thesis

Aschenbrenner's exponential compute trajectory: AI is in a physical buildout supercycle driven by chips, memory, power, cooling, datacenters, and networking. The engine identifies which stocks will benefit from the *next* 1–4 quarters of AI infrastructure capex commitments and supply-chain bottlenecks — on both the supply side (semiconductors, power, cooling, interconnects) and demand side (hyperscalers, AI platforms, inference-scaling beneficiaries).

## Architecture

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
Export (stock_signals.json · market_regime.json · p_estimate.json)
```

## Macro Regimes

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

## Schedule

Runs Mon–Fri at **9:00am ET** via a **launchd** agent
(`~/Library/LaunchAgents/com.divnar.ai-signal-engine.plist`), which re-runs jobs missed while
the machine was asleep. The host is on **IST**, so the agent fires at **18:30 IST = 9:00am ET (EDT)**;
downstream portfolio execution runs at 19:00/19:01 IST (9:30/9:31am ET market open) via cron.

> **Note:** launchd/cron use local (IST) time. These IST times are EDT-correct; during EST
> (Nov–Mar) they land ~1h earlier — still pre-market, harmless.

## Setup

```bash
pip install -r requirements.txt

# Required
export GEMINI_API_KEY=...

# Optional — enables live macro signals (free at fred.stlouisfed.org)
export FRED_API_KEY=...

# Optional — enables Alpaca paper execution
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

Store secrets in a `.env` file at the repo root (gitignored). `run.sh` loads it automatically.

## Usage

```bash
python main.py              # full run: ingest → macro → score → execute → export
python main.py --force      # re-score even if no new documents (uses last 30 days)
python main.py --dry-run    # ingest + macro + score only, no trades or file writes
```

## Portfolio Structure

- **Long book** — highest-conviction AI buildout beneficiaries. Weights sum to 1.0, max 10% per position.
- **Net exposure** — set by the macro regime signal (`net_exposure_target`), reducing long notional during stress.
- Execution is **long-only**: `long_notional = portfolio_value × net_exposure_target`; per-position
  orders below $500 are skipped; positions dropped from the book are closed first.

## Ticker Universe

~95 globally-listed stocks across compute, power & utilities, datacenter REITs, hyperscalers,
and AI software. Gemini picks freely from this list.

## Vector store (ChromaDB)

Two persistent collections at `data/chroma/` (gitignored, rebuilt locally):

- **`research_docs`** — every ingested document, embedded with Gemini `gemini-embedding-001`.
- **`macro_signals`** — every signal record produced, embedded from its thesis text.

The scorer queries `research_docs` semantically for the 30 most *relevant* documents (not the
30 most recent). Embedding/retrieval is an enhancement: on any failure the pipeline falls back
to SQLite recency ordering and continues.

> **Operational note:** ChromaDB upserts depend on live Gemini embedding calls. During a Gemini
> prepayment-credit outage (Jun 9–12 2026) embedding upserts returned 429 and were not retried
> (ingestion dedups by URL), leaving `research_docs` ~187 documents behind SQLite. Re-running the
> one-time backfill (delete `data/chroma_backfill_done`) re-indexes the gap. Changing
> `EMBEDDING_MODEL` requires deleting `data/chroma` and rebuilding (vector dimensionality differs).

## Resilience

Each stage fails soft and never blocks the pipeline:
- ChromaDB backfill / PCA refit errors → warn and continue with fallbacks.
- Gemini scoring failure (429 / rate limit / timeout) → **hold the current book**, skip rebalance,
  leave documents unscored for retry next run, exit cleanly so the scheduler stays healthy.

## Preliminary Results

Alpaca paper account, inception **24 Mar 2026** at $100,000:

| Metric | Value |
|---|---|
| Equity (16 Jun 2026, intraday) | ~$115.6k |
| Return since inception | **+15.6%** |
| Return through prior close (Jun 13, 58 trading days) | +11.5% |
| Max drawdown | −10.3% |

*Early-stage paper-trading results; not investment advice. Holdings and positioning omitted.*

## Tests

```bash
python -m pytest tests/ -v
```

80 tests covering macro signal computation (incl. PCA composite modifier), scorer guardrails,
DB migrations, ingestion deduplication, ChromaDB store, and long-only execution.
