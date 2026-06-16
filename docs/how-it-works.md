# How the AI Signal Engine Works

## What it does

A daily pipeline that reads AI infrastructure news, processes market data, asks Gemini to build a portfolio, and executes trades through Alpaca. The thesis is Aschenbrenner's: AI is on an exponential compute trajectory, driven by a physical buildout supercycle — chips, memory, power, cooling, datacenters, networking. The engine tries to hold the stocks that benefit most from the *next* 1–4 quarters of that buildout, not the stocks that already ran.

---

## Pipeline overview

```
main.py runs once daily (or on-demand)
│
├── 1. Ingest          RSS feeds, HuggingFace papers, SEC EDGAR 8-Ks
│                      → stored in SQLite (signals.db)
│                      → embedded + upserted into ChromaDB
│
├── 2. Macro signal    Fetch supply chain + cross-sector market data
│                      → determine base regime
│                      → apply PCA composite stress modifier
│                      → output: net_exposure_target (e.g. 0.72)
│
├── 3. Score           ChromaDB semantic query → top-30 relevant docs
│                      → assemble prompt context by value chain layer
│                      → call Gemini → structured portfolio JSON
│                      → apply guardrails (weight cap, normalization)
│
├── 4. Execute         Alpaca paper account rebalance
│                      long_notional = portfolio_value × net_exposure_target
│
└── 5. Export          Write signal JSON to disk
```

---

## 1. Ingestion

Three sources, run every cycle:

| Source | What | Layer tag |
|---|---|---|
| RSS feeds | SemiAnalysis, Stratechery, Platformer, Latent Space, Import AI, etc. | compute / platform / application |
| HuggingFace Daily Papers | Top arxiv papers from HF API | platform |
| SEC EDGAR 8-Ks | Hyperscaler quarterly disclosures (MSFT, AMZN, GOOGL, META, NVDA) | platform |

Each document is deduplicated by URL in SQLite. If it's new, it also gets embedded with Gemini `text-embedding-004` and upserted into ChromaDB (`research_docs` collection). Duplicates on re-run are skipped in both stores.

---

## 2. Macro signal

This is the risk-management layer. It decides how much of the portfolio to actually deploy.

### Step 1 — Supply chain signals (`macro/supply_chain.py`)

Pulled from FRED and yfinance:

- **PMI proxy** — DGORDER (Durable Goods Orders, FRED), converted to a PMI-scale number centred at 50. Tells us whether manufacturing demand is expanding or contracting.
- **Semis inventory trend** — FRED series IPG3344S (semiconductor industrial production). Rising = inventory building; falling = drawing down (bullish for chip names).
- **Shipping pressure** — BDRY ETF 30-day return, mapped to [0, 1]. High = global shipping under stress.

### Step 2 — Cross-sector signals (`macro/cross_sector.py`)

Pulled from yfinance over 90 days:

- **Power→compute lead** — z-score of (power basket momentum − compute basket momentum) over 30 days. Positive means power stocks are outrunning compute — a leading indicator that infrastructure buildout is being prioritised.
- **Copper→infra lead** — FCX 30-day return z-scored against its own history. Rising copper historically precedes infrastructure capex.
- **Credit stress** — VIX > 25 AND HYG 30-day return < −2%. If true, something systemic is breaking and the engine goes defensive.
- **VIX level** — raw number, fed into the composite modifier below.

### Step 3 — Base regime

Three regimes, evaluated in priority order:

| Regime | Trigger | Base exposure |
|---|---|---|
| `credit_stress` | VIX > 25 AND HYG 30d return < −2% | 0.20 |
| `balanced` | PMI contracting AND copper weak (< −0.5σ) | 0.65 |
| `compute_constrained` | everything else | 0.80 |

### Step 4 — PCA composite stress modifier (`macro/composite.py`)

This is where shipping pressure (and other correlated stress signals) now lives.

**The problem it solves:** Shipping pressure, weak copper, high VIX, weak PMI, and compute underperforming power are all correlated — they all tend to spike together during risk-off periods. Treating each one as a separate rule double-counts the same underlying stress. The old approach had a hard `shipping_bottleneck` regime that would chop 25 points off net exposure the moment BDRY crossed a threshold — regardless of whether anything else was actually stressed.

**How the PCA modifier works:**

Every Monday (or when the cache is stale), the engine fits a PCA over the last 90 days of five stress signals:

1. `shipping_pressure` — BDRY 30d return → [0, 1]
2. `copper_infra_lead_neg` — FCX 30d return z-score, negated (falling copper = stress)
3. `power_compute_lead_neg` — power/compute spread z-score, negated (compute outpacing power = stress)
4. `vix_level` — raw VIX
5. `pmi_neg` — PMI proxy, negated (contraction = stress)

PC1 is oriented so the VIX loading is always positive (stress = higher score). The fit is cached to `data/composite_modifier_cache.json`.

Each day, the current signal values are projected onto the cached PC1 to get a scalar stress score. That score is converted to its historical percentile over the 90-day window:

```
stress_score = percentile rank of today's PC1 score in the 90-day history
modifier = −0.25 × stress_score
net_exposure_target = max(0.15, base_exposure + modifier)
```

So modifier ranges from 0 (no stress) to −0.25 (extreme stress). If today is in the 60th percentile of historical stress, modifier = −0.15. The net exposure is floored at 0.15 so the engine always has some skin in the game.

`credit_stress` is exempt from the modifier — it already hard-floors at 0.20 and adding more drag on top would be double-counting a different signal.

**Before vs. after:**

Before: shipping pressure above 0.65 → hard switch to `shipping_bottleneck` → exposure cuts from 0.80 to 0.55 instantly. This was the main reason the portfolio underperformed QQQ (+13.4% vs +24.2%) — the threshold fired during a period when shipping was elevated but everything else was fine, and exposure stayed capped for weeks.

After: shipping pressure contributes proportionally to the composite modifier. If VIX is 16 and PMI is expanding, even moderately elevated shipping pressure barely moves the modifier. Only when multiple signals elevate together does the modifier meaningfully reduce exposure.

---

## 3. Scoring (Gemini)

### Document retrieval

When `chroma_client` is available, the scorer discards the SQLite recency-ordered docs and instead queries ChromaDB semantically:

```python
query = f"AI infrastructure buildout regime:{regime_label} semiconductor GPU power datacenter capex supply chain"
docs = query_research_docs(chroma_client, query, n_results=30)
past_signals = query_signal_records(chroma_client, query, n_results=3)
```

This means the scorer sees the 30 *most relevant* documents to what's happening right now, not the 30 most recent ones. Past signal records from the `macro_signals` collection give Gemini a running memory of recent regime changes.

### Prompt structure

The prompt is assembled in value chain layer order:

```
[MACRO REGIME SIGNAL — net exposure target, regime, supply chain, cross-sector]
[CURRENT PORTFOLIO POSITIONS]
[RECENT SIGNAL HISTORY — last 3 semantic matches from ChromaDB]
[COMPUTE LAYER SIGNALS — SemiAnalysis, chip papers, etc.]
[PLATFORM LAYER SIGNALS — hyperscaler 8-Ks, Stratechery, etc.]
[APPLICATION LAYER SIGNALS — Latent Space, Import AI, etc.]
```

Gemini is told to act as a portfolio manager for an AI infrastructure long-only fund. It outputs a JSON with `p_score` (Aschenbrenner probability this week), `market_regime`, `portfolio` (ticker + weight + conviction + reasoning per position), `signal_confidence`, `thesis_stress`, and `thesis_update`.

### Guardrails

After Gemini responds:
- Per-stock weight capped at 10%
- Weights renormalized to sum to 1.0
- If weights still don't sum within tolerance after 50 cap-and-normalize iterations, a last-resort division is applied

### Ticker universe

95 stocks across the full AI buildout stack — chips (NVDA, AMD, AVGO, ASML, TSM), power (VST, CEG, NEE, ETN, PWR), datacenters (EQIX, DLR), hyperscalers (MSFT, GOOGL, AMZN, META), plus copper (FCX), industrials (VRT, GE), and a broad set of software/cloud names. Gemini picks freely from this list.

---

## 4. Execution

Alpaca paper trading account. The rebalance logic:

```
long_notional = portfolio_value × net_exposure_target
target_dollar_value_per_stock = weight × long_notional
```

For each position: if the difference between target and current exceeds $500, submit a market order (buy or sell). Positions no longer in the portfolio are closed first. All open orders are cancelled before re-reading positions to avoid held-quantity artifacts.

---

## 5. ChromaDB

Two persistent collections at `data/chroma/`:

- **`research_docs`** — every ingested document, embedded with Gemini `text-embedding-004`, metadata includes source, value_chain_layer, ingested_at, ticker_mentions
- **`macro_signals`** — every signal record the engine produces, embedded from the thesis_update text, metadata includes regime, p_final, computed_at

On first run, a one-time backfill indexes all existing SQLite documents and signals. A sentinel file (`data/chroma_backfill_done`) prevents it from re-running.

---

## Data flow diagram

```
yfinance / FRED
      │
      ▼
macro/supply_chain.py ──┐
macro/cross_sector.py ──┤──► macro/regime.py ──► net_exposure_target
macro/composite.py ─────┘         │
  (PCA cache, weekly)             │
                                  │
RSS / HuggingFace / EDGAR         │
      │                           │
      ▼                           │
   SQLite                         │
   ChromaDB ◄──────────────────── │
      │                           │
      ▼                           ▼
scoring/gemini_scorer.py ──► Gemini API ──► portfolio weights
      │
      ▼
execution/alpaca.py ──► Alpaca paper account
      │
      ▼
export.py ──► signal JSON
```

---

## Key files

| File | Role |
|---|---|
| `main.py` | Orchestrator — runs the full pipeline |
| `config.py` | RSS feeds, tickers, guardrail constants |
| `db.py` | SQLite read/write for documents and signals |
| `chroma_store.py` | ChromaDB init, Gemini embed, upsert/query/backfill |
| `macro/supply_chain.py` | PMI proxy, semis trend, shipping pressure (FRED + yfinance) |
| `macro/cross_sector.py` | Power/compute lead, copper lead, credit stress, VIX (yfinance) |
| `macro/composite.py` | PCA stress modifier — weekly fit, daily apply |
| `macro/regime.py` | Combines the above into a single macro signal dict |
| `scoring/gemini_scorer.py` | Assembles prompt, calls Gemini, applies guardrails |
| `execution/alpaca.py` | Paper account rebalance via Alpaca SDK |
| `ingestion/rss.py` | RSS feed ingestion |
| `ingestion/huggingface_papers.py` | HF daily papers ingestion |
| `ingestion/transcripts.py` | SEC EDGAR 8-K ingestion |

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini scoring + text-embedding-004 |
| `ALPACA_API_KEY` | Yes | Paper trading account |
| `ALPACA_SECRET_KEY` | Yes | Paper trading account |
| `FRED_API_KEY` | No | Real PMI/DGORDER data (falls back to neutral defaults if absent) |
