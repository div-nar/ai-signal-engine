# Design: Swap LLM backend to opencode + local Chroma embeddings

**Date:** 2026-08-13
**Status:** Approved (design), pending spec review
**Branch:** `swap-llm-backend-opencode-local-embed`

## Problem

The Gemini API project behind `GEMINI_API_KEY` has been suspended by Google:

```
403 PERMISSION_DENIED — "Your project has been denied access. Please contact support."
```

This single failure disables **both** LLM-dependent subsystems, because both route
through that one key:

1. **Thesis scoring** (`scoring/thesis_scorer.py`) — the model's reasoning brain.
   Daily cron runs have been failing or no-op since ~Aug 7; last good target was
   `id 9` on Aug 9. The portfolio (+17% paper) is unaffected — Alpaca is independent —
   but it has been coasting on a stale thesis.
2. **Chroma embeddings** (`chroma_store.py`) — the agentic retrieval vector DB.
   343 vectors are intact on disk but the store can no longer ingest or query new
   text (embedding calls 403).

## Goals

- Replace Gemini thesis generation with the local `opencode` CLI (already installed,
  authed via the user's OpenCode Go subscription — independent of the dead key).
- Replace Gemini embeddings with a **local, offline** embedding model — no API key,
  nothing that can be remotely revoked again.
- Sever every Gemini/`google.genai` dependency from the daily run path.
- Preserve all existing interfaces and tests; this is a backend swap, not a redesign.

## Non-goals

- No general provider-abstraction layer. There are exactly two call sites, each
  already behind a clean seam. A formal `LLMProvider` interface + registry would be
  indirection without payoff (YAGNI). Revisit only if multi-provider failover is
  wanted later.
- No change to strategy, execution, scheduling, or the dashboard.

## Verified facts (feasibility checks already run)

- `opencode` v1.17.12 at `/opt/homebrew/bin/opencode`. `opencode run --format json`
  returns a JSON event stream; the `type:"text"` parts carry the assistant's answer.
  A live test returned clean structured JSON (`{"regime":...,"picks":[...]}`), no tool
  use, ~11s, ~$0.003, via the authed `opencode-go` gateway.
- Available strong models include `opencode-go/qwen3.7-max`, `opencode-go/kimi-k2.7`,
  `opencode-go/minimax-m3`, `opencode-go/glm-5.2`, `opencode-go/deepseek-v4-pro`.
- `scoring/thesis_scorer.py` already uses dependency injection: the agentic loop and
  `_generate_parsed()` only require an object with `.generate(prompt) -> str`.
  `_GeminiClient` is constructed at exactly one place: `score_layer_thesis(...)`,
  `client = _GeminiClient()` (thesis_scorer.py:399).
- `chroma_store.py` touches Gemini in exactly one function, `_embed()`
  (lines 18–31). Every upsert/query/backfill routes through it.
- `run_chroma_backfill()` already re-embeds all docs + signals from `signals.db`,
  gated by the `data/chroma_backfill_done` sentinel — the rebuild path exists.
- `fastembed` supports `nomic-ai/nomic-embed-text-v1.5` as ONNX (no PyTorch).

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Thesis backend | `opencode run` CLI wrapper | Authed Go subscription, no API key to revoke |
| Thesis model | `opencode-go/qwen3.7-max` (config constant) | Strong reasoner; trivially swappable |
| Embedding backend | `fastembed` (local ONNX) | Offline, free, unrevokable |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5` (768-dim) | 8192-token context — no truncation of long arXiv/EDGAR docs (all-MiniLM truncates at 256) |
| Existing vectors | Rebuild from SQLite | 768-dim ≠ old dims; incompatible, must re-embed |

## Architecture

Two surgical swaps behind existing interfaces.

### Component 1 — `_OpencodeClient` (thesis brain)

New class in `scoring/thesis_scorer.py`, same shape as `_GeminiClient`:

- `.generate(prompt: str) -> str`:
  - `subprocess.run(["opencode", "run", "--pure", "--format", "json",
    "-m", OPENCODE_MODEL, prompt], capture_output=True, text=True, timeout=OPENCODE_TIMEOUT_S)`
  - Parse stdout line-by-line as JSON events; concatenate the `text` field of every
    `{"type":"text",...}` event → return the assistant's raw text.
  - Non-zero exit, timeout, or empty text → raise (caught by existing retry).
- `--pure` (no external plugins) and no `-f`/file attachments keep it a pure-generation
  call, suppressing agentic tool use.
- Config: new constants in `config.py`:
  - `OPENCODE_MODEL = "opencode-go/qwen3.7-max"`
  - `OPENCODE_TIMEOUT_S = 120`
- Switch point: `score_layer_thesis(...)` default becomes `client = _OpencodeClient()`.
  `_GeminiClient` is retained (unused) so rollback is a one-line revert.

The scorer's `parse_thesis_response()`, the 3× exponential-backoff retry in
`_generate_parsed()`, and the whole agentic search loop are **unchanged** — they only
ever see `.generate()`.

### Component 2 — local embeddings + Chroma rebuild

- Add `fastembed` to `requirements.txt`.
- `chroma_store.py`:
  - Remove `from google import genai` and the `EMBEDDING_MODEL` import.
  - Module-level singleton: `_embedder = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")`
    (lazy-initialised on first use so imports stay cheap and tests can inject).
  - `_embed(text, task_type=...)`:
    - Nomic prefix convention keyed off the existing `task_type`:
      `search_query:` when `task_type == "RETRIEVAL_QUERY"`, else `search_document:`.
    - Return `list(next(_embedder.embed([prefixed_text])))` as `list[float]` (768-dim).
  - `upsert_*` and `query_*` are otherwise untouched; they still never raise on
    embedding failure (ingestion must not halt).
- One-time rebuild via a new script `scripts/rebuild_chroma.py`: delete the
  `research_docs` + `macro_signals` collections and the `data/chroma_backfill_done`
  sentinel, then call `run_chroma_backfill()` with all docs and signals from
  `signals.db`. Re-embeds all 343 records locally at 768-dim. Idempotent — safe to
  re-run.
- `config.py`: `EMBEDDING_MODEL` and `GEMINI_MODEL` are both kept as commented dead
  constants (not deleted) to ease rollback, but neither is imported or invoked on the
  run path after this change.

## Data flow (daily `--mode passive` / `--mode trade`, after change)

```
ingest docs → SQLite
   │
   ├─ upsert_research_doc ──► _embed (fastembed/nomic, local) ──► Chroma
   │
thesis pass:
   score_layer_thesis(client=_OpencodeClient)
      └─ _generate_parsed ─► opencode run (opencode-go/qwen3.7-max) ─► JSON text
             └─ agentic search ─► query_research_docs ─► _embed (local) ─► Chroma
   → target_weights persisted to `targets` table  (no Gemini anywhere)
```

## Error handling

- **opencode**: non-zero exit / timeout / empty assistant text → `RuntimeError`, caught
  by the scorer's existing 3-attempt exponential backoff. After 3 failures the run
  raises (same behaviour as today), so a total opencode outage fails loudly rather than
  trading on a blank thesis.
- **embeddings**: per-doc failures still swallowed inside `upsert_*` (return `False`,
  print WARNING) so ingestion never halts. A hard fastembed import/init failure surfaces
  at first use.
- **backfill**: unchanged — on any failure it does not write the sentinel, so the
  rebuild retries next run rather than locking in a partial index.

## Testing

- New `tests/test_opencode_client.py`: mock `subprocess.run` with a captured real
  opencode event stream; assert `.generate()` extracts concatenated assistant text, and
  that non-zero exit / timeout / empty output raise. No network in tests.
- New `tests/test_local_embeddings.py`: assert `_embed()` returns a 768-length vector
  and that `RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT` apply the correct nomic prefix
  (inject a fake embedder to avoid model download in CI).
- Existing 158 tests must stay green — interfaces (`client.generate`, `_embed`
  signature) are preserved. The scorer's DI tests already pass fake clients.

## Verification (before declaring done)

1. `pytest tests -q` → all green.
2. Live `./run.sh --mode passive` completes end-to-end: thesis computed via opencode,
   Chroma rebuilt and queried, a new target persisted.
3. `grep -rniE "genai|gemini|GEMINI_API_KEY" <run path>` proves no Gemini call on the
   daily path (config constants may remain as dead code but must not be invoked).
4. Confirm `data/chroma` collections report 768-dim and full record counts.

## Risks

- **fastembed/nomic download** (~260MB, one-time). First build must fetch it; verified
  offline afterward. Fallback if unavailable: `BAAI/bge-base-en-v1.5` (fastembed,
  768-dim, same seam) — chosen to match dims so no other change is needed.
- **opencode latency/variance**: ~11s/call observed; the once-daily cadence absorbs it.
  Model is a single config constant if quality/latency needs tuning.
- **opencode output format drift** across versions: mitigated by parsing defensively
  (collect all `type:"text"` parts; ignore unknown event types).
