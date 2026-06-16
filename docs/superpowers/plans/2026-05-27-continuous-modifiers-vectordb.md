# Continuous Modifiers + ChromaDB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `shipping_bottleneck` hard regime with a PCA-derived composite stress modifier on net exposure, and add ChromaDB-backed semantic retrieval to replace the recency-biased document fetch in the scorer.

**Architecture:** ChromaDB (workstream B) runs first and is independent; PCA modifier (workstream A) is self-contained in `macro/`. A new `chroma_store.py` owns all ChromaDB + Gemini embedding logic. A new `macro/composite.py` owns PCA fit and cache; `macro/regime.py` imports from it to apply the modifier after computing base regime exposure.

**Tech Stack:** `chromadb`, `scikit-learn` (PCA), `google-genai` (already installed — Gemini text-embedding-004), `yfinance`, `fredapi` (already installed).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | modify | add `chromadb`, `scikit-learn` |
| `chroma_store.py` | **create** | ChromaDB init, Gemini embed helper, upsert/query/backfill |
| `db.py` | modify | expose `get_all_documents()` and `get_all_signals()` for backfill |
| `ingestion/rss.py` | modify | accept optional `chroma_client`, upsert new docs |
| `ingestion/huggingface_papers.py` | modify | same |
| `ingestion/transcripts.py` | modify | same |
| `scoring/gemini_scorer.py` | modify | accept `chroma_client`, semantic retrieval replaces recency fetch |
| `macro/composite.py` | **create** | PCA signal history build, weekly fit+cache, daily apply |
| `macro/regime.py` | modify | remove `shipping_bottleneck`, apply composite modifier |
| `main.py` | modify | init chroma, backfill gate, pass client, weekly staleness check |
| `tests/test_chroma_store.py` | **create** | upsert idempotency, query, backfill sentinel |
| `tests/test_composite.py` | **create** | PCA orientation, floor, cold start, staleness, credit_stress exempt |
| `tests/test_regime.py` | modify | remove shipping_bottleneck test, add composite modifier tests |

---

## Workstream B — ChromaDB Integration

### Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Add chromadb and scikit-learn to requirements.txt**

Open `requirements.txt` and append:
```
chromadb==0.6.3
scikit-learn==1.6.1
```

- [ ] **Install**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pip install chromadb==0.6.3 scikit-learn==1.6.1
```

Expected: both packages install without error.

- [ ] **Commit**

```bash
git add requirements.txt
git commit -m "chore: add chromadb and scikit-learn dependencies"
```

---

### Task 2: Create chroma_store.py

**Files:**
- Create: `chroma_store.py`

- [ ] **Write the failing test first**

Create `tests/test_chroma_store.py`:

```python
import os
import pytest
import chromadb
from unittest.mock import patch, MagicMock

# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_embed(text, task_type="RETRIEVAL_DOCUMENT"):
    """Return a deterministic 768-dim embedding without hitting the API."""
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(768)]
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def chroma_client(tmp_path):
    from chroma_store import init_chroma
    return init_chroma(str(tmp_path / "chroma"))


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_both_collections(chroma_client):
    names = {c.name for c in chroma_client.list_collections()}
    assert "research_docs" in names
    assert "macro_signals" in names


# ── upsert + idempotency ──────────────────────────────────────────────────────

def test_upsert_research_doc_idempotent(chroma_client):
    from chroma_store import upsert_research_doc
    with patch("chroma_store._embed", side_effect=_fake_embed):
        upsert_research_doc(chroma_client, "doc_1", "NVDA earnings beat", {"source": "rss", "ticker_mentions": "NVDA", "ingested_at": "2026-05-01"})
        upsert_research_doc(chroma_client, "doc_1", "NVDA earnings beat", {"source": "rss", "ticker_mentions": "NVDA", "ingested_at": "2026-05-01"})
    col = chroma_client.get_collection("research_docs")
    assert col.count() == 1


def test_upsert_signal_record_idempotent(chroma_client):
    from chroma_store import upsert_signal_record
    with patch("chroma_store._embed", side_effect=_fake_embed):
        upsert_signal_record(chroma_client, "signal_1", "Compute thesis intact.", {"regime": "compute_constrained", "p_final": 0.88, "computed_at": "2026-05-22"})
        upsert_signal_record(chroma_client, "signal_1", "Compute thesis intact.", {"regime": "compute_constrained", "p_final": 0.88, "computed_at": "2026-05-22"})
    col = chroma_client.get_collection("macro_signals")
    assert col.count() == 1


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_research_docs_returns_results(chroma_client):
    from chroma_store import upsert_research_doc, query_research_docs
    with patch("chroma_store._embed", side_effect=_fake_embed):
        for i in range(5):
            upsert_research_doc(chroma_client, f"doc_{i}", f"AI infra paper {i}", {"source": "arxiv", "ticker_mentions": "", "ingested_at": "2026-05-01"})
        results = query_research_docs(chroma_client, "AI GPU compute infrastructure", n_results=3)
    assert len(results) <= 3
    assert all("title" in r for r in results)


def test_query_signal_records_returns_results(chroma_client):
    from chroma_store import upsert_signal_record, query_signal_records
    with patch("chroma_store._embed", side_effect=_fake_embed):
        for i in range(4):
            upsert_signal_record(chroma_client, f"signal_{i}", f"Regime update {i}", {"regime": "compute_constrained", "p_final": 0.9, "computed_at": "2026-05-01"})
        results = query_signal_records(chroma_client, "compute constrained shipping", n_results=2)
    assert len(results) <= 2


# ── backfill sentinel ─────────────────────────────────────────────────────────

def test_backfill_runs_once(tmp_path, chroma_client):
    from chroma_store import run_chroma_backfill
    sentinel = tmp_path / "chroma_backfill_done"
    docs = [{"id": 1, "source": "rss", "title": "Test", "content": "content", "ingested_at": "2026-05-01", "value_chain_layer": "compute"}]
    signals = [{"id": 1, "thesis_update": "Thesis ok", "macro_signal": None, "market_regime": "compute_constrained", "p_final": 0.9, "computed_at": "2026-05-01"}]

    with patch("chroma_store._embed", side_effect=_fake_embed):
        run_chroma_backfill(chroma_client, docs, signals, str(sentinel))
        count_after_first = chroma_client.get_collection("research_docs").count()
        run_chroma_backfill(chroma_client, docs, signals, str(sentinel))
        count_after_second = chroma_client.get_collection("research_docs").count()

    assert count_after_first == 1
    assert count_after_second == 1  # sentinel prevented second run
```

- [ ] **Run test — verify it fails**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
python -m pytest tests/test_chroma_store.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'chroma_store'`

- [ ] **Implement chroma_store.py**

Create `chroma_store.py` at the repo root:

```python
import os
from pathlib import Path
from typing import Optional

import chromadb
from google import genai


def init_chroma(path: str) -> chromadb.ClientAPI:
    client = chromadb.PersistentClient(path=path)
    client.get_or_create_collection("research_docs")
    client.get_or_create_collection("macro_signals")
    return client


def _embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    api_key = os.environ.get("GEMINI_API_KEY")
    gc = genai.Client(api_key=api_key)
    result = gc.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config={"task_type": task_type},
    )
    return list(result.embeddings[0].values)


def upsert_research_doc(
    client: chromadb.ClientAPI,
    doc_id: str,
    text: str,
    metadata: dict,
) -> None:
    embedding = _embed(text, task_type="RETRIEVAL_DOCUMENT")
    col = client.get_collection("research_docs")
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )


def upsert_signal_record(
    client: chromadb.ClientAPI,
    signal_id: str,
    text: str,
    metadata: dict,
) -> None:
    embedding = _embed(text, task_type="RETRIEVAL_DOCUMENT")
    col = client.get_collection("macro_signals")
    col.upsert(
        ids=[signal_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata],
    )


def query_research_docs(
    client: chromadb.ClientAPI,
    query_text: str,
    n_results: int = 30,
) -> list[dict]:
    embedding = _embed(query_text, task_type="RETRIEVAL_QUERY")
    col = client.get_collection("research_docs")
    actual_n = min(n_results, col.count())
    if actual_n == 0:
        return []
    results = col.query(query_embeddings=[embedding], n_results=actual_n)
    docs = []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        docs.append({
            "id": results["ids"][0][i],
            "title": doc_text[:120],
            "content": doc_text,
            "source": meta.get("source", ""),
            "ticker_mentions": meta.get("ticker_mentions", ""),
            "ingested_at": meta.get("ingested_at", ""),
            "value_chain_layer": meta.get("value_chain_layer", "application"),
        })
    return docs


def query_signal_records(
    client: chromadb.ClientAPI,
    query_text: str,
    n_results: int = 3,
) -> list[dict]:
    embedding = _embed(query_text, task_type="RETRIEVAL_QUERY")
    col = client.get_collection("macro_signals")
    actual_n = min(n_results, col.count())
    if actual_n == 0:
        return []
    results = col.query(query_embeddings=[embedding], n_results=actual_n)
    records = []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        records.append({
            "id": results["ids"][0][i],
            "text": doc_text,
            "regime": meta.get("regime", ""),
            "p_final": meta.get("p_final", 0.0),
            "computed_at": meta.get("computed_at", ""),
        })
    return records


def run_chroma_backfill(
    client: chromadb.ClientAPI,
    all_docs: list[dict],
    all_signals: list[dict],
    sentinel_path: str,
) -> None:
    if Path(sentinel_path).exists():
        return
    print("  ChromaDB: running one-time backfill...")
    for doc in all_docs:
        text = f"{doc.get('title', '')} {doc.get('content', '')}".strip()
        metadata = {
            "source": doc.get("source", ""),
            "ticker_mentions": "",
            "ingested_at": str(doc.get("ingested_at", "")),
            "value_chain_layer": doc.get("value_chain_layer", "application"),
        }
        upsert_research_doc(client, str(doc["id"]), text, metadata)
    for sig in all_signals:
        thesis = sig.get("thesis_update") or ""
        notes = ""
        if sig.get("macro_signal"):
            import json as _json
            try:
                notes = _json.loads(sig["macro_signal"]).get("notes", "")
            except Exception:
                pass
        text = f"{thesis} {notes}".strip()
        metadata = {
            "regime": sig.get("market_regime", ""),
            "p_final": float(sig.get("p_final") or 0.0),
            "computed_at": str(sig.get("computed_at", "")),
        }
        upsert_signal_record(client, f"signal_{sig['id']}", text, metadata)
    Path(sentinel_path).write_text("done")
    print(f"  ChromaDB: backfill complete — {len(all_docs)} docs, {len(all_signals)} signals")
```

- [ ] **Run tests — verify they pass**

```bash
python -m pytest tests/test_chroma_store.py -v
```

Expected: all 6 tests PASS.

- [ ] **Commit**

```bash
git add chroma_store.py tests/test_chroma_store.py
git commit -m "feat: add chroma_store module with Gemini embeddings and backfill"
```

---

### Task 3: Add get_all_documents and get_all_signals to db.py

**Files:**
- Modify: `db.py`

These are needed by `main.py` to pass the full corpus to `run_chroma_backfill`.

- [ ] **Write the failing test**

Add to `tests/test_db.py` (open the file and append):

```python
def test_get_all_documents_returns_all_rows(tmp_path):
    from db import init_db, insert_document, get_all_documents
    db = str(tmp_path / "test.db")
    init_db(db)
    insert_document(db, "rss", "Title A", "http://a.com", None, "body a", "compute")
    insert_document(db, "rss", "Title B", "http://b.com", None, "body b", "platform")
    rows = get_all_documents(db)
    assert len(rows) == 2
    assert rows[0]["title"] == "Title A"


def test_get_all_signals_returns_all_rows(tmp_path):
    from db import init_db, insert_signal, get_all_signals
    db = str(tmp_path / "test.db")
    init_db(db)
    insert_signal(db, {
        "p_final": 0.88, "stock_conviction": "{}", "stock_weights": "{}",
        "stock_reasoning": "{}", "sector_tilt": "{}", "supply_demand_balance": 0.5,
        "market_regime": "compute_constrained", "signal_confidence": 0.9,
        "thesis_stress": False, "signal_age_days": 0, "sources_ingested": 10,
        "signal_breakdown": "{}", "thesis_update": "ok", "raw_response": None,
        "prompt_context_doc_ids": "[]", "short_weights": None, "macro_signal": None,
    })
    rows = get_all_signals(db)
    assert len(rows) == 1
    assert rows[0]["market_regime"] == "compute_constrained"
```

- [ ] **Run test — verify it fails**

```bash
python -m pytest tests/test_db.py::test_get_all_documents_returns_all_rows tests/test_db.py::test_get_all_signals_returns_all_rows -v
```

Expected: `ImportError: cannot import name 'get_all_documents'`

- [ ] **Add the two functions to db.py**

Open `db.py`. After the `get_recent_documents` function, insert:

```python
def get_all_documents(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_signals(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

- [ ] **Run tests — verify they pass**

```bash
python -m pytest tests/test_db.py -v
```

Expected: all db tests PASS.

- [ ] **Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add get_all_documents and get_all_signals to db"
```

---

### Task 4: Wire ingestion modules to upsert into ChromaDB

**Files:**
- Modify: `ingestion/rss.py`, `ingestion/huggingface_papers.py`, `ingestion/transcripts.py`

Each ingestion function gets an optional `chroma_client=None` parameter. When provided and a new doc is inserted (non-None return from `insert_document`), it calls `upsert_research_doc`.

- [ ] **Write the failing test**

Create `tests/test_ingestion_chroma.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def _make_chroma_client():
    client = MagicMock()
    col = MagicMock()
    col.count.return_value = 0
    client.get_collection.return_value = col
    return client


def test_rss_ingest_upserts_new_doc(tmp_path):
    from ingestion.rss import ingest_rss
    from db import init_db
    db = str(tmp_path / "test.db")
    init_db(db)
    chroma = _make_chroma_client()

    fake_entries = [{"title": "ASML Q1", "link": "http://asml.com/1", "published": "2026-05-01", "summary": "backlog surge"}]
    with patch("ingestion.rss.feedparser.parse", return_value=MagicMock(entries=fake_entries)), \
         patch("chroma_store.upsert_research_doc") as mock_upsert, \
         patch("chroma_store._embed", return_value=[0.1] * 768):
        ingest_rss("http://fake.com/feed", "compute", db, chroma_client=chroma)

    mock_upsert.assert_called_once()
    call_kwargs = mock_upsert.call_args
    assert call_kwargs[0][2] == "ASML Q1 backlog surge"  # text = title + content


def test_rss_ingest_no_upsert_on_duplicate(tmp_path):
    from ingestion.rss import ingest_rss
    from db import init_db
    db = str(tmp_path / "test.db")
    init_db(db)
    chroma = _make_chroma_client()

    fake_entries = [{"title": "ASML Q1", "link": "http://asml.com/1", "published": "2026-05-01", "summary": "backlog surge"}]
    with patch("ingestion.rss.feedparser.parse", return_value=MagicMock(entries=fake_entries)), \
         patch("chroma_store.upsert_research_doc") as mock_upsert, \
         patch("chroma_store._embed", return_value=[0.1] * 768):
        ingest_rss("http://fake.com/feed", "compute", db, chroma_client=chroma)
        ingest_rss("http://fake.com/feed", "compute", db, chroma_client=chroma)

    assert mock_upsert.call_count == 1  # second run is a duplicate, no upsert


def test_rss_ingest_skips_upsert_when_no_chroma_client(tmp_path):
    from ingestion.rss import ingest_rss
    from db import init_db
    db = str(tmp_path / "test.db")
    init_db(db)

    fake_entries = [{"title": "T", "link": "http://x.com/1", "published": "2026-05-01", "summary": "s"}]
    with patch("ingestion.rss.feedparser.parse", return_value=MagicMock(entries=fake_entries)), \
         patch("chroma_store.upsert_research_doc") as mock_upsert:
        ingest_rss("http://fake.com/feed", "compute", db)  # no chroma_client

    mock_upsert.assert_not_called()
```

- [ ] **Run — verify it fails**

```bash
python -m pytest tests/test_ingestion_chroma.py -v 2>&1 | head -20
```

Expected: `TypeError: ingest_rss() got an unexpected keyword argument 'chroma_client'`

- [ ] **Modify ingestion/rss.py**

Open `ingestion/rss.py`. Find the `ingest_rss` function signature and update it, then add the upsert call after a successful insert. The diff is:

```python
# OLD signature:
def ingest_rss(url: str, value_chain_layer: str, db_path: str = str(DEFAULT_DB)) -> int:

# NEW signature:
def ingest_rss(url: str, value_chain_layer: str, db_path: str = str(DEFAULT_DB), chroma_client=None) -> int:
```

Inside `ingest_rss`, after the `insert_document` call, add:

```python
        if result is not None and chroma_client is not None:
            from chroma_store import upsert_research_doc
            text = f"{entry.get('title', '')} {entry.get('summary', '')}".strip()
            metadata = {
                "source": "rss",
                "ticker_mentions": "",
                "ingested_at": published or "",
                "value_chain_layer": value_chain_layer,
            }
            upsert_research_doc(chroma_client, str(result), text, metadata)
```

- [ ] **Modify ingestion/huggingface_papers.py**

Find `ingest_hf_papers`. Update signature:

```python
# OLD:
def ingest_hf_papers(max_results: int = 50, db_path: str = str(DEFAULT_DB)) -> int:

# NEW:
def ingest_hf_papers(max_results: int = 50, db_path: str = str(DEFAULT_DB), chroma_client=None) -> int:
```

After each `insert_document` call:

```python
        if result is not None and chroma_client is not None:
            from chroma_store import upsert_research_doc
            text = f"{p['title']} {p.get('summary', '')}".strip()
            metadata = {
                "source": "arxiv",
                "ticker_mentions": "",
                "ingested_at": p.get("published", ""),
                "value_chain_layer": p.get("value_chain_layer", "platform"),
            }
            upsert_research_doc(chroma_client, str(result), text, metadata)
```

- [ ] **Modify ingestion/transcripts.py**

Find `ingest_edgar`. Update signature:

```python
# OLD:
def ingest_edgar(tickers: dict, max_per_ticker: int = 3, db_path: str = str(DEFAULT_DB)) -> int:

# NEW:
def ingest_edgar(tickers: dict, max_per_ticker: int = 3, db_path: str = str(DEFAULT_DB), chroma_client=None) -> int:
```

After each `insert_document` call:

```python
            if result is not None and chroma_client is not None:
                from chroma_store import upsert_research_doc
                text = f"{title} {content[:1000]}".strip()
                metadata = {
                    "source": "edgar",
                    "ticker_mentions": ticker,
                    "ingested_at": filed_at or "",
                    "value_chain_layer": "platform",
                }
                upsert_research_doc(chroma_client, str(result), text, metadata)
```

- [ ] **Run tests — verify they pass**

```bash
python -m pytest tests/test_ingestion_chroma.py tests/test_rss.py tests/test_transcripts.py -v
```

Expected: all PASS.

- [ ] **Commit**

```bash
git add ingestion/rss.py ingestion/huggingface_papers.py ingestion/transcripts.py tests/test_ingestion_chroma.py
git commit -m "feat: wire ingestion modules to upsert new docs into ChromaDB"
```

---

### Task 5: Semantic retrieval in scorer + wire main.py

**Files:**
- Modify: `scoring/gemini_scorer.py`
- Modify: `main.py`

- [ ] **Write the failing test**

Add to `tests/test_gemini_scorer.py`:

```python
def test_score_documents_queries_chroma_not_sqlite(tmp_path):
    """When chroma_client is provided, scorer must call chroma query, not get_recent_documents."""
    import json
    from unittest.mock import patch, MagicMock
    from scoring.gemini_scorer import score_documents

    chroma_client = MagicMock()

    mock_research = [
        {"id": "1", "title": "NVDA Q1", "content": "NVDA beats", "source": "rss",
         "ticker_mentions": "NVDA", "ingested_at": "2026-05-01", "value_chain_layer": "compute"},
    ]
    mock_signals = [
        {"id": "signal_1", "text": "Thesis intact", "regime": "compute_constrained", "p_final": 0.88, "computed_at": "2026-05-01"},
    ]

    valid_output = {
        "p_score": 0.88, "market_regime": "compute_constrained",
        "supply_demand_balance": 0.3,
        "portfolio": [{"ticker": "NVDA", "weight": 0.10, "conviction": 0.9, "reasoning": "GPU demand"}],
        "signal_confidence": 0.8, "thesis_stress": False, "thesis_update": "stable",
    }

    with patch("chroma_store.query_research_docs", return_value=mock_research) as mock_q_docs, \
         patch("chroma_store.query_signal_records", return_value=mock_signals) as mock_q_sigs, \
         patch("chroma_store._embed", return_value=[0.1]*768), \
         patch("scoring.gemini_scorer.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content.return_value.text = json.dumps(valid_output)
        result = score_documents(docs=[], chroma_client=chroma_client)

    mock_q_docs.assert_called_once()
    mock_q_sigs.assert_called_once()
    assert result["p_final"] == pytest.approx(0.88)
```

- [ ] **Run — verify it fails**

```bash
python -m pytest tests/test_gemini_scorer.py::test_score_documents_queries_chroma_not_sqlite -v 2>&1 | head -20
```

Expected: `TypeError: score_documents() got an unexpected keyword argument 'chroma_client'`

- [ ] **Modify scoring/gemini_scorer.py**

Update the `score_documents` signature and add semantic retrieval path:

```python
def score_documents(
    docs: list[dict],
    db_path: str = str(DEFAULT_DB),
    prev_weights: Optional[dict] = None,
    current_portfolio: Optional[dict] = None,
    macro_signal: Optional[dict] = None,
    chroma_client=None,
) -> dict:
    """Call Gemini with assembled signal context. Returns structured signal dict."""
    if prev_weights is None:
        prev_weights = {}

    # ── Document retrieval ────────────────────────────────────────────────────
    if chroma_client is not None:
        from chroma_store import query_research_docs, query_signal_records
        regime_label = (macro_signal or {}).get("regime", "compute_constrained")
        query = (
            f"AI infrastructure buildout regime:{regime_label} "
            "semiconductor GPU power datacenter capex supply chain"
        )
        docs = query_research_docs(chroma_client, query, n_results=30)
        past_signals = query_signal_records(chroma_client, query, n_results=3)
    else:
        past_signals = []

    guardrail_baseline = current_portfolio if current_portfolio else prev_weights
    context = build_signal_context(docs, current_portfolio=current_portfolio, macro_signal=macro_signal)

    # Prepend past signal context if available
    if past_signals:
        past_block = "### RECENT SIGNAL HISTORY\n" + "\n".join(
            f"[{s['computed_at']}] regime={s['regime']} p={s['p_final']:.2f}: {s['text']}"
            for s in past_signals
        ) + "\n\n"
        context = past_block + context

    # ... rest of the function unchanged from here (universe_str, system, user_prompt, genai call, etc.)
```

Keep the rest of the function body identical — only the top doc-fetching block and `context` assembly change.

- [ ] **Modify main.py — init chroma, backfill, pass client, upsert signal**

At the top of `main.py`, add imports after existing imports:

```python
from chroma_store import init_chroma, run_chroma_backfill, upsert_signal_record
from db import init_db, get_unscored_documents, get_recent_documents, mark_scored, insert_signal, get_all_documents, get_all_signals
```

In `main()`, after `init_db(DB_PATH)`:

```python
    # Init ChromaDB and run one-time backfill if needed
    chroma_path = str(Path(__file__).parent / "data" / "chroma")
    sentinel_path = str(Path(__file__).parent / "data" / "chroma_backfill_done")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = init_chroma(chroma_path)
    all_docs = get_all_documents(DB_PATH)
    all_sigs = get_all_signals(DB_PATH)
    run_chroma_backfill(chroma_client, all_docs, all_sigs, sentinel_path)
```

Pass `chroma_client` to ingestion calls:

```python
    n = ingest_rss(feed["url"], feed["value_chain_layer"], DB_PATH, chroma_client=chroma_client)
    # ...
    n = ingest_hf_papers(HF_PAPERS_MAX_RESULTS, DB_PATH, chroma_client=chroma_client)
    n = ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=DB_PATH, chroma_client=chroma_client)
```

Pass `chroma_client` to `score_documents`:

```python
    signal = score_documents(
        docs=unscored,
        db_path=DB_PATH,
        prev_weights=prev_weights,
        current_portfolio=current_portfolio,
        macro_signal=macro_signal,
        chroma_client=chroma_client,
    )
```

After `insert_signal`, upsert the new signal record:

```python
    signal_id = insert_signal(DB_PATH, signal)
    thesis = signal.get("thesis_update", "")
    notes = ""
    if signal.get("macro_signal"):
        try:
            notes = json.loads(signal["macro_signal"]).get("notes", "")
        except Exception:
            pass
    upsert_signal_record(
        chroma_client,
        f"signal_{signal_id}",
        f"{thesis} {notes}".strip(),
        {
            "regime": signal["market_regime"],
            "p_final": float(signal["p_final"]),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
```

Also add `from pathlib import Path` if not already imported in main.py (check — it currently isn't).

- [ ] **Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests PASS; new tests PASS.

- [ ] **Commit**

```bash
git add scoring/gemini_scorer.py main.py
git commit -m "feat: semantic retrieval in scorer via ChromaDB; wire main.py"
```

---

## Workstream A — PCA Composite Modifier

### Task 6: Create macro/composite.py

**Files:**
- Create: `macro/composite.py`
- Create: `tests/test_composite.py`

- [ ] **Write the failing tests**

Create `tests/test_composite.py`:

```python
import json
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta


def _make_fake_history(n=90, vix_spike_on_last=False) -> pd.DataFrame:
    """Return a (n, 5) DataFrame of stress-oriented signals."""
    rng = np.random.default_rng(42)
    data = {
        "shipping_pressure": rng.uniform(0.3, 0.7, n),
        "copper_infra_lead_neg": rng.standard_normal(n),
        "power_compute_lead_neg": rng.standard_normal(n),
        "vix_level": rng.uniform(14, 22, n),
        "pmi_neg": rng.uniform(-2, 2, n),
    }
    df = pd.DataFrame(data)
    if vix_spike_on_last:
        df.loc[n - 1, "vix_level"] = 40.0  # extreme spike on the last day
    return df


def _write_fake_cache(path: Path, history_df: pd.DataFrame):
    from macro.composite import fit_and_cache_composite
    with patch("macro.composite._build_signal_history", return_value=history_df):
        fit_and_cache_composite(str(path))


# ── is_cache_stale ─────────────────────────────────────────────────────────────

def test_missing_cache_is_stale(tmp_path):
    from macro.composite import is_cache_stale
    assert is_cache_stale(str(tmp_path / "nonexistent.json")) is True


def test_fresh_cache_is_not_stale(tmp_path):
    from macro.composite import is_cache_stale
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"computed_at": datetime.now(timezone.utc).isoformat()}))
    assert is_cache_stale(str(cache), max_age_days=8) is False


def test_old_cache_is_stale(tmp_path):
    from macro.composite import is_cache_stale
    cache = tmp_path / "cache.json"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
    cache.write_text(json.dumps({"computed_at": old_ts}))
    assert is_cache_stale(str(cache), max_age_days=8) is True


# ── fit_and_cache_composite ────────────────────────────────────────────────────

def test_fit_writes_cache(tmp_path):
    from macro.composite import fit_and_cache_composite
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    assert cache.exists()
    data = json.loads(cache.read_text())
    assert "pc1_loadings" in data
    assert "history_mean" in data
    assert "history_std" in data
    assert "pc1_history_scores" in data
    assert len(data["pc1_loadings"]) == 5


def test_vix_loading_is_positive_after_orientation(tmp_path):
    """PC1 must always have a positive VIX loading (stress-oriented)."""
    from macro.composite import fit_and_cache_composite
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    data = json.loads(cache.read_text())
    signal_names = data["signal_names"]
    vix_idx = signal_names.index("vix_level")
    assert data["pc1_loadings"][vix_idx] > 0


# ── load_composite_modifier ────────────────────────────────────────────────────

def test_cold_start_returns_zero_modifier(tmp_path):
    from macro.composite import load_composite_modifier
    supply = {"shipping_pressure": 0.5, "pmi": 52.0}
    cross = {"copper_infra_lead": 0.2, "power_compute_lead": 0.3, "vix_level": 16.0}
    modifier, info = load_composite_modifier(str(tmp_path / "nonexistent.json"), supply, cross)
    assert modifier == pytest.approx(0.0)
    assert info["stress_score"] == pytest.approx(0.0)


def test_high_vix_increases_stress_score(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    normal_df = _make_fake_history(n=90)
    with patch("macro.composite._build_signal_history", return_value=normal_df):
        fit_and_cache_composite(str(cache))

    low_vix_supply = {"shipping_pressure": 0.3, "pmi": 54.0}
    low_vix_cross = {"copper_infra_lead": 0.5, "power_compute_lead": 0.5, "vix_level": 14.0}
    _, low_info = load_composite_modifier(str(cache), low_vix_supply, low_vix_cross)

    high_vix_cross = {"copper_infra_lead": 0.5, "power_compute_lead": 0.5, "vix_level": 35.0}
    _, high_info = load_composite_modifier(str(cache), low_vix_supply, high_vix_cross)

    assert high_info["stress_score"] > low_info["stress_score"]


def test_modifier_is_non_positive(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    supply = {"shipping_pressure": 0.9, "pmi": 44.0}
    cross = {"copper_infra_lead": -2.0, "power_compute_lead": -2.0, "vix_level": 30.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    assert modifier <= 0.0


def test_modifier_ceiling_is_minus_0_25(tmp_path):
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    # Extreme stress — modifier should not exceed -0.25
    supply = {"shipping_pressure": 1.0, "pmi": 30.0}
    cross = {"copper_infra_lead": -5.0, "power_compute_lead": -5.0, "vix_level": 80.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    assert modifier >= -0.25


# ── regime integration (modifier floor) ───────────────────────────────────────

def test_net_exposure_never_falls_below_015(tmp_path):
    """Even max stress cannot push net_exposure below 0.15."""
    from macro.composite import fit_and_cache_composite, load_composite_modifier
    cache = tmp_path / "cache.json"
    df = _make_fake_history()
    with patch("macro.composite._build_signal_history", return_value=df):
        fit_and_cache_composite(str(cache))
    supply = {"shipping_pressure": 1.0, "pmi": 30.0}
    cross = {"copper_infra_lead": -5.0, "power_compute_lead": -5.0, "vix_level": 80.0}
    modifier, _ = load_composite_modifier(str(cache), supply, cross)
    base_exposure = 0.80  # compute_constrained
    net = max(0.15, base_exposure + modifier)
    assert net >= 0.15
```

- [ ] **Run — verify they fail**

```bash
python -m pytest tests/test_composite.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'macro.composite'`

- [ ] **Implement macro/composite.py**

Create `macro/composite.py`:

```python
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SIGNAL_NAMES = [
    "shipping_pressure",
    "copper_infra_lead_neg",
    "power_compute_lead_neg",
    "vix_level",
    "pmi_neg",
]

_POWER_TICKERS = ["NEE", "ETN", "PWR"]
_COMPUTE_TICKERS = ["NVDA", "TSM"]


def _build_signal_history(lookback: int = 90, fred_api_key: Optional[str] = None) -> pd.DataFrame:
    """Pull daily time series for all 5 signals. Returns (lookback, 5) DataFrame."""
    period = f"{lookback + 40}d"  # extra buffer for 30d rolling calcs

    # ── Download price data ────────────────────────────────────────────────────
    tickers = ["BDRY", "FCX", "^VIX"] + _POWER_TICKERS + _COMPUTE_TICKERS
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    try:
        prices = raw.xs("Close", axis=1, level=0)
    except (KeyError, TypeError):
        prices = raw

    # ── shipping_pressure: BDRY 30d rolling return → [0,1] ───────────────────
    bdry = prices["BDRY"].dropna()
    sp = ((bdry - bdry.shift(30)) / bdry.shift(30) + 0.10) / 0.30
    shipping = sp.clip(0.0, 1.0)

    # ── copper_infra_lead: FCX rolling 30d return z-score ────────────────────
    fcx_ret = prices["FCX"].pct_change(30)
    copper_z = (fcx_ret - fcx_ret.rolling(60).mean()) / (fcx_ret.rolling(60).std() + 1e-9)
    copper_neg = -copper_z  # negate: negative copper → positive stress

    # ── power_compute_lead: (power basket - compute basket) 30d spread z-score
    power_ser = prices[_POWER_TICKERS].mean(axis=1)
    compute_ser = prices[_COMPUTE_TICKERS].mean(axis=1)
    spread_ret = (power_ser - compute_ser).diff(30)
    pcl_z = (spread_ret - spread_ret.rolling(60).mean()) / (spread_ret.rolling(60).std() + 1e-9)
    pcl_neg = -pcl_z  # negate: compute outpacing power → positive stress

    # ── vix_level: raw VIX ───────────────────────────────────────────────────
    vix = prices["^VIX"]

    # ── pmi_neg: from FRED DGORDER, monthly forward-filled to daily ──────────
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if api_key:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        dgorder = fred.get_series("DGORDER", observation_start="2024-01-01")
        pmi_raw = (50.0 + (dgorder.pct_change() * 500)).clip(30.0, 70.0)
        pmi_daily = pmi_raw.resample("D").interpolate("linear")
    else:
        pmi_daily = pd.Series(50.0, index=vix.index)

    # ── Align all series on common trading days ───────────────────────────────
    df = pd.DataFrame({
        "shipping_pressure": shipping,
        "copper_infra_lead_neg": copper_neg,
        "power_compute_lead_neg": pcl_neg,
        "vix_level": vix,
        "pmi_neg": -pmi_daily,  # negate: low PMI → positive stress
    }).dropna()

    return df.iloc[-lookback:]


def fit_and_cache_composite(
    cache_path: str,
    lookback: int = 90,
    fred_api_key: Optional[str] = None,
) -> None:
    """Weekly fit: pull 90-day signal history, run PCA, write cache."""
    history = _build_signal_history(lookback=lookback, fred_api_key=fred_api_key)

    scaler = StandardScaler()
    X = scaler.fit_transform(history.values)

    pca = PCA(n_components=1)
    pca.fit(X)
    loadings = pca.components_[0]

    # Orient so VIX loading is always positive (stress direction)
    vix_idx = _SIGNAL_NAMES.index("vix_level")
    if loadings[vix_idx] < 0:
        loadings = -loadings

    # Project all 90 days → scalar history for percentile lookup
    pc1_scores = X @ loadings

    cache = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "signal_names": _SIGNAL_NAMES,
        "pc1_loadings": loadings.tolist(),
        "history_mean": scaler.mean_.tolist(),
        "history_std": scaler.scale_.tolist(),
        "pc1_history_scores": pc1_scores.tolist(),
    }
    Path(cache_path).write_text(json.dumps(cache))


def load_composite_modifier(
    cache_path: str,
    supply: dict,
    cross: dict,
) -> tuple[float, dict]:
    """Daily apply: project current signals onto cached PC1 → modifier ∈ [-0.25, 0].
    Returns (modifier, info_dict).
    """
    if not Path(cache_path).exists():
        return 0.0, {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": None}

    cache = json.loads(Path(cache_path).read_text())
    loadings = np.array(cache["pc1_loadings"])
    mean = np.array(cache["history_mean"])
    std = np.array(cache["history_std"])
    history_scores = np.array(cache["pc1_history_scores"])

    computed_at = datetime.fromisoformat(cache["computed_at"])
    cache_age_days = (datetime.now(timezone.utc) - computed_at).days

    # Build current signal vector (same order as _SIGNAL_NAMES)
    sp = supply.get("shipping_pressure", 0.5)
    copper_neg = -cross.get("copper_infra_lead", 0.0)
    pcl_neg = -cross.get("power_compute_lead", 0.0)
    vix = cross.get("vix_level", 16.0)
    pmi_neg = -(supply.get("pmi", 50.0) - 50.0) / 10.0  # normalised around 50

    current = np.array([sp, copper_neg, pcl_neg, vix, pmi_neg])

    # Standardize with cached params, project onto PC1
    current_std = (current - mean) / (std + 1e-9)
    score = float(current_std @ loadings)

    # Convert to percentile ∈ [0, 1]
    stress_score = float(np.mean(history_scores <= score))

    modifier = round(-0.25 * stress_score, 4)

    return modifier, {
        "stress_score": round(stress_score, 4),
        "modifier": modifier,
        "cache_age_days": cache_age_days,
    }


def is_cache_stale(cache_path: str, max_age_days: int = 8) -> bool:
    if not Path(cache_path).exists():
        return True
    try:
        data = json.loads(Path(cache_path).read_text())
        computed_at = datetime.fromisoformat(data["computed_at"])
        age = (datetime.now(timezone.utc) - computed_at).days
        return age > max_age_days
    except Exception:
        return True
```

- [ ] **Run tests — verify they pass**

```bash
python -m pytest tests/test_composite.py -v
```

Expected: all 11 tests PASS.

- [ ] **Commit**

```bash
git add macro/composite.py tests/test_composite.py
git commit -m "feat: PCA composite stress modifier (weekly fit, daily apply)"
```

---

### Task 7: Update macro/regime.py

**Files:**
- Modify: `macro/regime.py`
- Modify: `tests/test_regime.py`

- [ ] **Update test_regime.py**

Replace the entire content of `tests/test_regime.py` with:

```python
import pytest
from unittest.mock import patch

_SUPPLY_CLEAN = {
    "shipping_pressure": 0.3,
    "semis_inventory_trend": "neutral",
    "pmi": 52.0,
    "pmi_trend": "expanding",
}
_CROSS_CLEAN = {
    "power_compute_lead": 0.5,
    "copper_infra_lead": 0.2,
    "credit_stress": False,
    "vix_level": 16.0,
}
_ZERO_MODIFIER = (0.0, {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": 0})
_MAX_MODIFIER = (-0.25, {"stress_score": 1.0, "modifier": -0.25, "cache_age_days": 0})


def test_clean_macro_gives_compute_constrained():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "compute_constrained"
    assert result["net_exposure_target"] == pytest.approx(0.80)


def test_credit_stress_overrides_all():
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "credit_stress"
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_credit_stress_exempt_from_modifier():
    """Composite modifier must not be applied when regime is credit_stress."""
    from macro.regime import compute_macro_signal
    cross_stressed = {**_CROSS_CLEAN, "credit_stress": True, "vix_level": 30.0}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_stressed), \
         patch("macro.regime.load_composite_modifier", return_value=_MAX_MODIFIER):
        result = compute_macro_signal()
    assert result["net_exposure_target"] == pytest.approx(0.20)


def test_contracting_pmi_and_weak_copper_gives_balanced():
    from macro.regime import compute_macro_signal
    supply_weak = {**_SUPPLY_CLEAN, "pmi": 48.0, "pmi_trend": "contracting"}
    cross_weak = {**_CROSS_CLEAN, "copper_infra_lead": -0.8}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_weak), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=cross_weak), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] == "balanced"
    assert result["net_exposure_target"] == pytest.approx(0.65)


def test_shipping_pressure_no_longer_triggers_own_regime():
    """High shipping pressure alone must NOT produce a shipping_bottleneck regime."""
    from macro.regime import compute_macro_signal
    supply_ship = {**_SUPPLY_CLEAN, "shipping_pressure": 0.90}
    with patch("macro.regime.fetch_supply_chain_signal", return_value=supply_ship), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    assert result["regime"] != "shipping_bottleneck"


def test_composite_modifier_reduces_net_exposure():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=(-0.15, {"stress_score": 0.6, "modifier": -0.15, "cache_age_days": 2})):
        result = compute_macro_signal()
    assert result["net_exposure_target"] == pytest.approx(0.80 - 0.15)
    assert result["composite_modifier"]["stress_score"] == pytest.approx(0.6)


def test_net_exposure_floored_at_015():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=(-0.25, {"stress_score": 1.0, "modifier": -0.25, "cache_age_days": 0})):
        result = compute_macro_signal()
    assert result["net_exposure_target"] >= 0.15


def test_macro_signal_has_all_required_keys():
    from macro.regime import compute_macro_signal
    with patch("macro.regime.fetch_supply_chain_signal", return_value=_SUPPLY_CLEAN), \
         patch("macro.regime.fetch_cross_sector_signal", return_value=_CROSS_CLEAN), \
         patch("macro.regime.load_composite_modifier", return_value=_ZERO_MODIFIER):
        result = compute_macro_signal()
    required = {
        "computed_at", "regime", "regime_confidence",
        "net_exposure_target", "supply_chain", "cross_sector",
        "notes", "composite_modifier",
    }
    assert required.issubset(result.keys())
```

- [ ] **Run — verify the new tests fail**

```bash
python -m pytest tests/test_regime.py -v 2>&1 | tail -20
```

Expected: `test_shipping_pressure_no_longer_triggers_own_regime` FAIL, `test_composite_modifier_reduces_net_exposure` FAIL, `test_net_exposure_floored_at_015` FAIL, `test_credit_stress_exempt_from_modifier` FAIL.

- [ ] **Rewrite macro/regime.py**

Replace the full content of `macro/regime.py`:

```python
from datetime import datetime, timezone

from macro.supply_chain import fetch_supply_chain_signal
from macro.cross_sector import fetch_cross_sector_signal
from macro.composite import load_composite_modifier

_COMPOSITE_CACHE_PATH = str(__import__("pathlib").Path(__file__).parent.parent / "data" / "composite_modifier_cache.json")


def compute_macro_signal(cache_path: str = _COMPOSITE_CACHE_PATH) -> dict:
    """Combine supply chain, cross-sector, and composite stress modifier into a MacroSignal."""
    supply = fetch_supply_chain_signal()
    cross = fetch_cross_sector_signal()

    # ── Base regime (priority-ordered) ────────────────────────────────────────
    if cross["credit_stress"]:
        regime = "credit_stress"
        base_exposure = 0.20
        confidence = 0.90
        notes = (
            f"Credit stress: VIX {cross['vix_level']:.1f} > 25, HYG falling. "
            "Defensive positioning — minimal net long exposure."
        )
    elif supply["pmi_trend"] == "contracting" and cross["copper_infra_lead"] < -0.5:
        regime = "balanced"
        base_exposure = 0.65
        confidence = 0.70
        notes = (
            f"PMI contracting ({supply['pmi']:.1f}) and copper weak "
            f"({cross['copper_infra_lead']:.2f}σ). Cautious positioning."
        )
    else:
        regime = "compute_constrained"
        base_exposure = 0.80
        confidence = 0.85
        notes = (
            f"PMI {supply['pmi']:.1f} ({supply['pmi_trend']}), credit clean, "
            "compute thesis intact. Full net long exposure."
        )

    # ── Composite stress modifier (exempt for credit_stress) ──────────────────
    if regime == "credit_stress":
        modifier = 0.0
        modifier_info = {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": None}
    else:
        modifier, modifier_info = load_composite_modifier(cache_path, supply, cross)
        if modifier_info.get("cache_age_days") is not None:
            notes += (
                f" Composite stress score {modifier_info['stress_score']:.2f} "
                f"adding {modifier:.3f} drag to net exposure."
            )

    net_exposure_target = round(max(0.15, base_exposure + modifier), 4)

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_confidence": confidence,
        "net_exposure_target": net_exposure_target,
        "supply_chain": supply,
        "cross_sector": cross,
        "notes": notes,
        "composite_modifier": modifier_info,
    }
```

- [ ] **Run tests — verify they all pass**

```bash
python -m pytest tests/test_regime.py -v
```

Expected: all 8 tests PASS.

- [ ] **Commit**

```bash
git add macro/regime.py tests/test_regime.py
git commit -m "feat: replace shipping_bottleneck regime with PCA composite stress modifier"
```

---

### Task 8: Wire weekly PCA fit in main.py

**Files:**
- Modify: `main.py`

- [ ] **Add weekly staleness check to main()**

In `main.py`, after the ChromaDB init block (end of Task 5), add:

```python
    # Weekly PCA refit (Monday, or if cache is stale)
    from macro.composite import is_cache_stale, fit_and_cache_composite
    from pathlib import Path as _Path
    composite_cache = str(_Path(__file__).parent / "data" / "composite_modifier_cache.json")
    _Path(__file__).parent.joinpath("data").mkdir(parents=True, exist_ok=True)
    today_weekday = datetime.now(timezone.utc).weekday()  # 0 = Monday
    if today_weekday == 0 or is_cache_stale(composite_cache):
        print("\n--- Fitting PCA composite modifier (weekly) ---")
        fred_key = os.environ.get("FRED_API_KEY")
        fit_and_cache_composite(composite_cache, fred_api_key=fred_key)
        print("  PCA modifier cache written.")
```

Add `import os` to `main.py` imports if not already present (it is not — check top of file).

Also update the `compute_macro_signal` call to pass the cache path:

```python
    macro_signal = compute_macro_signal(cache_path=composite_cache)
```

And update the `compute_macro_signal` import to accept the kwarg (already does from Task 7).

- [ ] **Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all tests PASS. Note: tests that call `compute_macro_signal` already mock `load_composite_modifier`, so no network calls are made.

- [ ] **Commit**

```bash
git add main.py
git commit -m "feat: weekly PCA refit in main.py; pass cache path to compute_macro_signal"
```

---

### Task 9: Final validation

- [ ] **Run full test suite and confirm count**

```bash
python -m pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests PASS. Count should be ≥ 67 (57 existing + new tests).

- [ ] **Dry-run the engine end-to-end**

```bash
python main.py --dry-run 2>&1
```

Expected output includes:
- `ChromaDB: running one-time backfill...` (first run only)
- `Fitting PCA composite modifier (weekly)...` (if Monday or cache missing)
- `Regime: compute_constrained` (or `balanced`)
- `Composite stress score ... adding ... drag`
- No `shipping_bottleneck` anywhere in output

- [ ] **Verify shipping_bottleneck is gone from outputs**

```bash
grep -r "shipping_bottleneck" /Users/div-nar/sideproj/ai-signal-engine --include="*.py" --include="*.json"
```

Expected: zero matches in `.py` files. Any `.json` in `data/` is historical and expected.

- [ ] **Final commit**

```bash
git add -A
git commit -m "chore: final validation — continuous modifiers + ChromaDB complete"
```
