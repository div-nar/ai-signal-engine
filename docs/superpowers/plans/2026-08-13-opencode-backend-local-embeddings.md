# opencode Thesis Backend + Local Chroma Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 403-suspended Gemini backend with the local `opencode` CLI for thesis reasoning and local `fastembed`/nomic for Chroma embeddings, behind the two existing seams.

**Architecture:** Two surgical swaps. `scoring/thesis_scorer.py` gains an `_OpencodeClient` (same `.generate(prompt)->str` interface as `_GeminiClient`) shelling out to `opencode run`. `chroma_store.py`'s single `_embed()` function switches to a local fastembed model. Existing interfaces, retry logic, agentic loop, and upsert/query functions are unchanged. The 343 existing vectors are rebuilt at 768-dim from SQLite.

**Tech Stack:** Python 3.14, `opencode` CLI v1.17.12 (`opencode-go` gateway), `fastembed` (ONNX), ChromaDB, pytest.

## Global Constraints

- Python interpreter: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` (hardcoded in `run.sh`).
- Run tests with: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest`.
- Thesis model: `opencode-go/qwen3.7-max` (config constant `OPENCODE_MODEL`).
- Embedding model: `nomic-ai/nomic-embed-text-v1.5`, 768-dim, via `fastembed`. Fallback `BAAI/bge-base-en-v1.5` (also 768-dim) if nomic won't download.
- Nomic prefix convention: `search_query:` for `RETRIEVAL_QUERY`, `search_document:` otherwise.
- No Gemini/`google.genai` call on the daily run path when done (verification gate).
- No provider-abstraction layer (YAGNI — two call sites only).
- Package installs via `bun` are N/A (Python); use the 3.14 pip: `python3 -m pip install`.
- Branch: `swap-llm-backend-opencode-local-embed`. Commit after each task.

---

### Task 1: `_OpencodeClient` for thesis generation

**Files:**
- Modify: `config.py` (add constants after line 74, the `# ── Gemini` block)
- Modify: `scoring/thesis_scorer.py` (add class after `_GeminiClient` at line 321; flip factory at line 399)
- Test: `tests/test_opencode_client.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `scoring.thesis_scorer._OpencodeClient` with `.generate(prompt: str) -> str`; `config.OPENCODE_MODEL: str`, `config.OPENCODE_TIMEOUT_S: int`.

- [ ] **Step 1: Add config constants**

In `config.py`, immediately after the `GEMINI_MAX_OUTPUT_TOKENS = 8192` line, add:

```python

# ── opencode (thesis backend; replaces Gemini API, which was 403-suspended) ────
# Invoked as a CLI via scoring.thesis_scorer._OpencodeClient. Uses the opencode-go
# subscription gateway — no API key in this process, nothing remotely revocable.
OPENCODE_MODEL = "opencode-go/qwen3.7-max"
OPENCODE_TIMEOUT_S = 120
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_opencode_client.py`:

```python
"""_OpencodeClient: parse `opencode run --format json` event stream -> text."""
import subprocess
import pytest
from scoring.thesis_scorer import _OpencodeClient

# A real captured opencode --format json stream (one JSON object per line).
_STREAM = (
    '{"type":"step_start","part":{"type":"step-start"}}\n'
    '{"type":"text","part":{"type":"text",'
    '"text":"{\\"market_regime\\":\\"compute_constrained\\"}"}}\n'
    '{"type":"step_finish","part":{"type":"step-finish","tokens":{"total":10}}}\n'
)


def test_generate_extracts_assistant_text(monkeypatch):
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=_STREAM, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = _OpencodeClient().generate("hello")
    assert out == '{"market_regime":"compute_constrained"}'
    # invoked the right binary + model, non-interactive, pure
    assert calls["cmd"][:3] == ["opencode", "run", "--pure"]
    assert "opencode-go/qwen3.7-max" in calls["cmd"]
    assert "hello" in calls["cmd"]


def test_generate_concatenates_multiple_text_parts(monkeypatch):
    stream = (
        '{"type":"text","part":{"type":"text","text":"foo"}}\n'
        '{"type":"text","part":{"type":"text","text":"bar"}}\n'
    )
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stream, stderr=""))
    assert _OpencodeClient().generate("x") == "foobar"


def test_generate_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")


def test_generate_raises_on_empty_text(monkeypatch):
    stream = '{"type":"step_finish","part":{"type":"step-finish"}}\n'
    monkeypatch.setattr(subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stream, stderr=""))
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")


def test_generate_raises_on_timeout(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 120)
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="opencode"):
        _OpencodeClient().generate("x")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_opencode_client.py -q`
Expected: FAIL — `ImportError: cannot import name '_OpencodeClient'`.

- [ ] **Step 4: Implement `_OpencodeClient`**

In `scoring/thesis_scorer.py`, add `import subprocess` and `import json` (json is already imported at line 14 — verify; add `subprocess` to the import block). Then, immediately after the `_GeminiClient` class (ends line 321), add:

```python
class _OpencodeClient:
    """Thesis client backed by the local `opencode` CLI (opencode-go gateway).

    Same interface as _GeminiClient: .generate(prompt) -> str. Runs opencode
    non-interactively, parses the --format json event stream, and returns the
    concatenated assistant text. Raises RuntimeError on any failure so the
    caller's existing retry/backoff handles it.
    """

    def __init__(self, model: str | None = None, timeout_s: int | None = None):
        self._model = model or config.OPENCODE_MODEL
        self._timeout = timeout_s or config.OPENCODE_TIMEOUT_S

    def generate(self, prompt: str) -> str:
        cmd = ["opencode", "run", "--pure", "--format", "json",
               "-m", self._model, prompt]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self._timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"opencode run timed out after {self._timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"opencode run exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        parts = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                parts.append(event.get("part", {}).get("text", ""))
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("opencode run produced no assistant text")
        return text
```

- [ ] **Step 5: Flip the default client**

In `scoring/thesis_scorer.py` at line 398-399, change:

```python
    if client is None:
        client = _GeminiClient()
```
to:
```python
    if client is None:
        client = _OpencodeClient()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_opencode_client.py tests/test_agentic_thesis.py -q`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add config.py scoring/thesis_scorer.py tests/test_opencode_client.py
git commit -m "feat: opencode CLI thesis backend, replacing 403'd Gemini"
```

---

### Task 2: Local embeddings in `chroma_store.py`

**Files:**
- Modify: `requirements.txt` (add `fastembed`)
- Modify: `chroma_store.py:1-31` (imports + `_embed`)
- Modify: `config.py` (comment out Gemini embedding constant)
- Test: `tests/test_local_embeddings.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `chroma_store._embed(text, task_type="RETRIEVAL_DOCUMENT") -> list[float]` (768-dim, local); `chroma_store._get_embedder() -> fastembed.TextEmbedding` (lazy singleton, monkeypatchable).

- [ ] **Step 1: Install fastembed**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pip install fastembed`
Then add to `requirements.txt` (new line): `fastembed>=0.3`

- [ ] **Step 2: Write the failing test**

Create `tests/test_local_embeddings.py`:

```python
"""_embed uses a local fastembed model with nomic prefix convention."""
import chroma_store


class _FakeEmbedder:
    """Records prefixed inputs; returns a fixed 768-dim vector per text."""
    def __init__(self):
        self.seen = []

    def embed(self, texts):
        for t in texts:
            self.seen.append(t)
            yield [0.01] * 768


def test_embed_returns_768_dim(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    v = chroma_store._embed("HBM supply constraints")
    assert isinstance(v, list) and len(v) == 768


def test_embed_applies_document_prefix_by_default(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    chroma_store._embed("grid power")
    assert fake.seen == ["search_document: grid power"]


def test_embed_applies_query_prefix_for_retrieval_query(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    chroma_store._embed("grid power", task_type="RETRIEVAL_QUERY")
    assert fake.seen == ["search_query: grid power"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_local_embeddings.py -q`
Expected: FAIL — `AttributeError: module 'chroma_store' has no attribute '_get_embedder'`.

- [ ] **Step 4: Rewrite the imports and `_embed`**

In `chroma_store.py`, replace lines 1-31 (the imports, `init_chroma` stays, and the whole `_embed`) so the file head reads:

```python
import os
from pathlib import Path
from typing import Optional

import chromadb

# Local embedding model — replaces the 403-suspended Gemini embeddings API.
# nomic-embed-text-v1.5: 768-dim, 8192-token context (no truncation of long
# arXiv/EDGAR docs), runs offline via fastembed's ONNX runtime. Changing this
# requires rebuilding the Chroma collections (scripts/rebuild_chroma.py) since
# vectors of different dimensionality cannot coexist in one collection.
_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_EMBEDDER = None


def init_chroma(path: str) -> chromadb.ClientAPI:
    client = chromadb.PersistentClient(path=path)
    client.get_or_create_collection("research_docs")
    client.get_or_create_collection("macro_signals")
    return client


def _get_embedder():
    """Lazy singleton fastembed model (first call downloads the ONNX weights)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        _EMBEDDER = TextEmbedding(_EMBED_MODEL)
    return _EMBEDDER


def _embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    prefix = "search_query: " if task_type == "RETRIEVAL_QUERY" else "search_document: "
    vec = next(_get_embedder().embed([prefix + text]))
    return [float(x) for x in vec]
```

(The `from config import EMBEDDING_MODEL` and `from google import genai` lines are removed by this replacement.)

- [ ] **Step 5: Retire the Gemini embedding constant**

In `config.py`, replace the `EMBEDDING_MODEL = "gemini-embedding-001"` line (and its comment block above it) with:

```python
# Embeddings moved to a LOCAL model (fastembed / nomic-embed-text-v1.5, 768-dim)
# in chroma_store.py after the Gemini project was 403-suspended. No API embedding
# constant remains. See docs/superpowers/specs/2026-08-13-*.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests/test_local_embeddings.py -q`
Expected: PASS.

- [ ] **Step 7: Verify nomic actually loads offline (real download, one-time)**

Run:
```bash
cd ~/sideproj/ai-signal-engine
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import chroma_store as c; v=c._embed('AI compute demand'); print('dim', len(v))"
```
Expected: prints `dim 768`. If this fails to download nomic, change `_EMBED_MODEL` to `"BAAI/bge-base-en-v1.5"` (also 768-dim) and re-run.

- [ ] **Step 8: Commit**

```bash
git add chroma_store.py config.py requirements.txt tests/test_local_embeddings.py
git commit -m "feat: local fastembed/nomic embeddings, replacing Gemini embeddings"
```

---

### Task 3: Chroma rebuild script

**Files:**
- Create: `scripts/rebuild_chroma.py`
- Test: none (operational script; verified by running it)

**Interfaces:**
- Consumes: `chroma_store.init_chroma`, `chroma_store.run_chroma_backfill`, `db.get_all_documents`, `db.get_all_signals`, `config.CHROMA_PATH` / `config.DB_PATH`.
- Produces: rebuilt `data/chroma` collections at 768-dim.

- [ ] **Step 1: Confirm the config/db symbol names**

Run:
```bash
cd ~/sideproj/ai-signal-engine
grep -nE "CHROMA|DB_PATH|BACKFILL|SENTINEL|DEFAULT_DB" config.py | head
grep -nE "def get_all_documents|def get_all_signals" db.py
```
Note the exact constant names (e.g. `CHROMA_PATH`, `DB_PATH`, backfill sentinel path). Use them verbatim in Step 2. If a name differs from the guesses below, substitute it.

- [ ] **Step 2: Write the rebuild script**

Create `scripts/rebuild_chroma.py`:

```python
"""Rebuild the Chroma collections from SQLite at the current embedding dim.

The old vectors were Gemini-embedded (different dimensionality) and cannot
coexist with the new local nomic 768-dim vectors, so this drops both
collections + the backfill sentinel, then re-embeds every doc and signal.
Idempotent — safe to re-run.

    python scripts/rebuild_chroma.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import db
import chroma_store

CHROMA_PATH = getattr(config, "CHROMA_PATH", "data/chroma")
SENTINEL = getattr(config, "CHROMA_BACKFILL_SENTINEL", "data/chroma_backfill_done")


def main() -> None:
    client = chroma_store.init_chroma(CHROMA_PATH)
    for name in ("research_docs", "macro_signals"):
        try:
            client.delete_collection(name)
            print(f"  dropped collection {name}")
        except Exception as e:
            print(f"  (collection {name} not dropped: {e})")
    # recreate empty collections
    client = chroma_store.init_chroma(CHROMA_PATH)
    sentinel = Path(SENTINEL)
    if sentinel.exists():
        sentinel.unlink()
        print("  removed backfill sentinel")

    docs = db.get_all_documents()
    signals = db.get_all_signals()
    print(f"  re-embedding {len(docs)} docs + {len(signals)} signals (local nomic)...")
    chroma_store.run_chroma_backfill(client, docs, signals, SENTINEL)

    counts = {c.name: client.get_collection(c.name).count()
              for c in client.list_collections()}
    print(f"  done. collection counts: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the rebuild**

Run:
```bash
cd ~/sideproj/ai-signal-engine
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 scripts/rebuild_chroma.py
```
Expected: drops both collections, re-embeds, prints non-zero counts for `research_docs` and `macro_signals` (≈334 and ≈9).

- [ ] **Step 4: Verify dim + query works**

Run:
```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "
import chroma_store as c
cli=c.init_chroma('data/chroma')
hits=c.query_research_docs(cli,'HBM memory supply',n_results=3)
print('query returned', len(hits), 'hits; first:', hits[0]['title'][:60] if hits else None)
"
```
Expected: returns ≥1 hit (proves 768-dim query round-trips against the rebuilt store).

- [ ] **Step 5: Commit**

```bash
git add scripts/rebuild_chroma.py
git commit -m "feat: scripts/rebuild_chroma.py — rebuild vectors at local 768-dim"
```

---

### Task 4: End-to-end verification (no Gemini on run path)

**Files:** none (verification only).

**Interfaces:** Consumes the whole pipeline via `run.sh --mode passive`.

- [ ] **Step 1: Full unit suite green**

Run: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m pytest tests -q`
Expected: all pass (158 prior + new opencode/embedding tests).

- [ ] **Step 2: Live passive run (no trades)**

Run:
```bash
cd ~/sideproj/ai-signal-engine
./run.sh --mode passive > /tmp/passive-verify.log 2>&1; echo "exit=$?"
tail -20 /tmp/passive-verify.log
```
Expected: exit 0, ends with `Done.`, no `PERMISSION_DENIED`, no traceback.

- [ ] **Step 3: Prove a fresh target was persisted via opencode**

Run:
```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "
import sqlite3, datetime
c=sqlite3.connect('signals.db')
r=c.execute('select id,computed_at,market_regime from targets order by id desc limit 1').fetchone()
print('latest target:', r)
assert r and str(datetime.date.today()) in r[1], 'no target persisted today'
print('OK — target computed today via opencode')
"
```
Expected: prints a target row dated today.

- [ ] **Step 4: Grep-prove no Gemini call on the run path**

Run:
```bash
grep -rniE "genai|PERMISSION_DENIED" /tmp/passive-verify.log && echo "FOUND GEMINI — investigate" || echo "OK — no Gemini/genai on run path"
```
Expected: `OK — no Gemini/genai on run path`.

- [ ] **Step 5: Commit any log/notes + open PR**

```bash
git add -A
git commit -m "chore: verify opencode + local embeddings end-to-end (passive run clean)" --allow-empty
```

---

## Post-plan operational steps (not code — done by the operator after merge)

- Reload the trade cron: `launchctl load ~/Library/LaunchAgents/com.divnar.layercake.trade.plist`.
- The `GEMINI_API_KEY` line in `.env` is now unused; leave or remove (no effect).
- Merge PR #2 (dashboard) is independent of this work.

## Self-Review

- **Spec coverage:** Component 1 (opencode client) → Task 1. Component 2 (local embeddings + rebuild) → Tasks 2-3. Error handling (retry, swallow, backfill sentinel) → preserved, asserted in Task 1/2 tests. Testing section → Tasks 1-2 unit tests. Verification gate (pytest, live passive, no-Gemini grep, dim check) → Task 4. Risk (nomic download fallback to bge-base) → Task 2 Step 7. All spec sections covered.
- **Placeholder scan:** none — every code step has full code; Task 3 Step 1 explicitly resolves the one unknown (config symbol names) before use.
- **Type consistency:** `.generate(prompt)->str` matches `_GeminiClient` and the scorer's usage. `_embed(text, task_type)->list[float]` matches all call sites in `chroma_store.py` (upsert/query). `_get_embedder()` is defined in Task 2 and monkeypatched in its tests.
