# Design: Continuous Macro Modifiers + ChromaDB Integration

**Date:** 2026-05-27  
**Status:** Approved

---

## Overview

Two independent workstreams:

- **A — Continuous PCA Modifier**: Remove `shipping_bottleneck` as a hard regime. Replace all discrete scalar signal thresholds with a single PCA-derived composite stress score that continuously modulates net exposure on top of the base regime.
- **B — ChromaDB Integration**: Embed the research doc corpus and macro signal history into ChromaDB. Replace "last N docs" context window in the scorer with semantic retrieval.

---

## Workstream A — PCA Composite Modifier

### Motivation

The current regime logic uses hard thresholds on individual signals (e.g. `shipping_pressure > 0.65` → cut exposure 25 points). Signals are correlated — high VIX, weak copper, and soft PMI co-move — so additive independent modifiers would double-count shared stress. A single PCA-derived composite score captures the dominant stress factor without that bias.

### Signals

Five normalized daily signals form the input matrix:

| Signal | Source | Direction |
|---|---|---|
| `shipping_pressure` | BDRY ETF 30d momentum → [0,1] | higher = more stress |
| `copper_infra_lead` | FCX z-score | negative = more stress |
| `power_compute_lead` | Power vs compute momentum z-score | negative = more stress |
| `vix_level` | VIX closing price | higher = more stress |
| `pmi` | DGORDER-derived PMI proxy | lower = more stress |

Before PCA, each series is standardized (zero mean, unit variance) over the lookback window. Signals with inverse stress direction (`copper_infra_lead`, `power_compute_lead`, `pmi`) are negated before input so that PC1 always points in the stress direction.

### Cadence

The PCA fit (PC1 loadings + historical percentile distribution) is computed **once per week** on Monday before the 9am ET run, using a rolling 90-day daily window. The result is cached to `data/composite_modifier_cache.json`. Daily runs load the cache and apply the stored modifier — they do not refit.

Cache schema:
```json
{
  "computed_at": "2026-05-27T09:00:00+00:00",
  "pc1_loadings": [0.48, 0.42, 0.31, 0.51, 0.44],
  "signal_names": ["shipping_pressure", "copper_infra_lead_neg", "power_compute_lead_neg", "vix_level", "pmi_neg"],
  "history_mean": [...],
  "history_std": [...],
  "pc1_history_percentiles": [0.12, 0.45, ...]
}
```

If no cache exists (cold start), the modifier defaults to `0.0` (no adjustment) and the weekly fit runs immediately.

### Computation

**Weekly (fit):**
1. Pull 90-day daily time series for all 5 signals from the same yfinance/FRED sources already used in `supply_chain.py` and `cross_sector.py`.
2. Negate stress-inverse signals.
3. Standardize each column.
4. Fit PCA; extract PC1 loadings. Flip sign if VIX loading is negative (ensures stress orientation).
5. Project all 90 days onto PC1 → scalar history.
6. Store loadings, normalization params, and scalar history in cache.

**Daily (apply):**
1. Load cache.
2. Compute current normalized signal vector (same 5 signals, using end-of-day values from the most recent fetch).
3. Project onto cached PC1 loadings → current stress scalar.
4. Convert to historical percentile against cached PC1 history → `stress_score ∈ [0, 1]`.
5. `modifier = -0.25 × stress_score`.

### Regime Integration

`regime.py` changes:
- Remove `shipping_bottleneck` branch entirely.
- Remaining regimes: `credit_stress` (0.20), `balanced` (0.65), `compute_constrained` (0.80).
- After computing `base_exposure`, load cached modifier and apply:
  ```python
  net_exposure_target = max(0.15, base_exposure + modifier)
  ```
- `credit_stress` is **exempt** — modifier is not applied when regime is `credit_stress`.
- `market_regime` field in DB and exports reflects only the base regime name.
- `macro_signal` output gains a `composite_modifier` field: `{"stress_score": 0.71, "modifier": -0.178, "cache_age_days": 3}`.

### New Module

`macro/composite.py` — three public functions:

- `fit_and_cache_composite(cache_path) → None` — weekly fit, writes cache
- `load_composite_modifier(cache_path) → float` — daily apply, returns modifier value
- `is_cache_stale(cache_path, max_age_days=8) → bool` — triggers re-fit if cache is old

### Tests

- PCA orientation: if VIX spikes, stress score should increase.
- Modifier floor: `compute_constrained` (0.80) with max stress score stays ≥ 0.15.
- Cache staleness: stale cache (>8 days) triggers immediate refit.
- `credit_stress` exemption: modifier is 0.0 when regime is `credit_stress`.
- Cold start: missing cache defaults modifier to 0.0.

---

## Workstream B — ChromaDB Integration

### Motivation

The scorer currently fetches the last N documents from SQLite and stuffs them into Gemini's context window. This is positionally biased (recency only) and ignores relevance. Semantic retrieval surfaces the most pertinent research regardless of ingestion date.

### Collections

**`research_docs`**  
One document per ingested record (RSS, arXiv, EDGAR). Embedded at ingest time.

| Field | Type | Notes |
|---|---|---|
| `id` | string | SQLite `doc_id` — dedup key |
| `embedding` | vector | Gemini `text-embedding-004` on `title + summary` |
| `source` | metadata | `rss`, `arxiv`, `edgar` |
| `ticker_mentions` | metadata | comma-separated tickers |
| `ingested_at` | metadata | ISO timestamp |

**`macro_signals`**  
One record per signal run. Embedded on write.

| Field | Type | Notes |
|---|---|---|
| `id` | string | `signal_{id}` |
| `embedding` | vector | Gemini embedding on `thesis_update + macro_signal.notes` |
| `regime` | metadata | base regime name |
| `p_final` | metadata | float |
| `computed_at` | metadata | ISO timestamp |

### Embedding Model

`models/text-embedding-004` via `google-generativeai` SDK (already available via `GEMINI_API_KEY`). Task type `RETRIEVAL_DOCUMENT` for upserts, `RETRIEVAL_QUERY` for queries.

### Storage

ChromaDB in persistent embedded mode. DB path: `data/chroma/`. Gitignored.

### Code Changes

**`db.py`**  
- Add `init_chroma(path) → chromadb.Client` — initialises persistent client and ensures both collections exist.
- Add `upsert_research_doc(client, doc_id, text, metadata)`.
- Add `upsert_signal_record(client, signal_id, text, metadata)`.

**`ingestion/`**  
- After writing each doc to SQLite, call `upsert_research_doc`. Skip if `doc_id` already present (ChromaDB upsert is idempotent).

**`scoring/gemini_scorer.py`**  
- Replace `fetch_recent_docs(db, limit=N)` with:
  1. Build query string from current macro regime + thesis keywords.
  2. `research_docs.query(query_texts=[query], n_results=30)` — top-30 semantically relevant docs.
  3. `macro_signals.query(query_texts=[query], n_results=3)` — top-3 relevant past signal runs as few-shot context.
- Remaining scorer logic unchanged.

**`main.py`**  
- On startup, call `init_chroma()`.
- Weekly (Monday check): call `fit_and_cache_composite()` if cache is stale.

### Migration

On first run after deployment:
1. `init_chroma()` creates empty collections.
2. A one-time backfill loop embeds all existing SQLite documents into `research_docs`.
3. A one-time backfill loop embeds all existing signal runs into `macro_signals`.
4. Backfill is gated by a `data/chroma_backfill_done` sentinel file so it only runs once.

### Tests

- Upsert idempotency: upserting the same `doc_id` twice does not create duplicates.
- Query returns results: after upserting 5 docs, a query returns ≤ 5 results.
- Backfill sentinel: second run skips backfill.
- Scorer uses ChromaDB path: mock ChromaDB client confirms `query()` is called, not the old SQLite fetch.

---

## Execution Order

1. Workstream B (ChromaDB) first — no dependency on A, unblocks semantic retrieval immediately.
2. Workstream A (PCA modifier) second — self-contained macro change.

---

## Files Touched

| File | Change |
|---|---|
| `macro/composite.py` | **new** — PCA fit, cache, apply |
| `macro/regime.py` | remove `shipping_bottleneck`, apply composite modifier |
| `db.py` | add ChromaDB init + upsert helpers |
| `ingestion/*.py` | call upsert after SQLite write |
| `scoring/gemini_scorer.py` | semantic retrieval instead of recency fetch |
| `main.py` | init ChromaDB; weekly cache-staleness check |
| `requirements.txt` | add `chromadb`, `scikit-learn` |
| `tests/` | new tests per section above |
