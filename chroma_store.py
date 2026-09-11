import os
import threading
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

# Local ONNX embedding is slow/pathological on very long sequences (a 50k-char
# EDGAR 8-K can take >100s on CPU), so cap the input. ~8000 chars ≈ 2000 tokens —
# well within nomic's context and far more than all-MiniLM's ~1000-char limit,
# while keeping per-doc embed sub-second. Research docs lead with their signal.
MAX_EMBED_CHARS = 8000

# TextEmbedding's first call re-validates/downloads the HF cache with no
# enforced timeout anywhere in that stack (huggingface_hub / xet-core) — this
# exact call hung a scheduled run for a full week (2026-08-26 to 2026-09-02).
# Bounding it in a daemon thread turns a silent multi-day hang into a fast,
# catchable failure instead.
_EMBEDDER_INIT_TIMEOUT_S = 60


def init_chroma(path: str) -> chromadb.ClientAPI:
    client = chromadb.PersistentClient(path=path)
    client.get_or_create_collection("research_docs")
    client.get_or_create_collection("macro_signals")
    return client


_EMBEDDER_INIT_FAILED = False


def _get_embedder():
    """Lazy singleton fastembed model (first call downloads the ONNX weights).

    Init runs in a daemon thread with a hard timeout — see
    _EMBEDDER_INIT_TIMEOUT_S. A timeout/failure is cached in
    _EMBEDDER_INIT_FAILED so every subsequent call this process fails FAST
    instead of re-attempting and re-waiting the full timeout — upsert_research_doc/
    upsert_signal_record catch this per-document and treat it as best-effort/
    non-fatal, so without caching the failure, one stuck download (e.g. a hung
    HF/xet transfer) turns a single 60s timeout into a 60s stall PER DOCUMENT
    across thousands of docs. This exact amplification hung a scheduled run
    for 4 days (2026-09-07 to 2026-09-11)."""
    global _EMBEDDER, _EMBEDDER_INIT_FAILED
    if _EMBEDDER_INIT_FAILED:
        raise RuntimeError("TextEmbedding init previously failed this process — not retrying")
    if _EMBEDDER is None:
        from fastembed import TextEmbedding
        result = {}

        def _init():
            try:
                result["model"] = TextEmbedding(_EMBED_MODEL)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        t.join(timeout=_EMBEDDER_INIT_TIMEOUT_S)
        if t.is_alive():
            _EMBEDDER_INIT_FAILED = True
            raise RuntimeError(
                f"TextEmbedding init exceeded {_EMBEDDER_INIT_TIMEOUT_S}s "
                "(stuck HF download?) — abandoning, not waiting further")
        if "error" in result:
            _EMBEDDER_INIT_FAILED = True
            raise result["error"]
        _EMBEDDER = result["model"]
    return _EMBEDDER


def _embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    prefix = "search_query: " if task_type == "RETRIEVAL_QUERY" else "search_document: "
    vec = next(_get_embedder().embed([prefix + text[:MAX_EMBED_CHARS]]))
    return [float(x) for x in vec]


def upsert_research_doc(
    client: chromadb.ClientAPI,
    doc_id: str,
    text: str,
    metadata: dict,
) -> bool:
    """Embed and upsert a research doc. Returns True on success. Never raises —
    an embedding/storage failure must not halt ingestion or the daily run."""
    try:
        embedding = _embed(text, task_type="RETRIEVAL_DOCUMENT")
        col = client.get_collection("research_docs")
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        print(f"  WARNING: ChromaDB upsert_research_doc({doc_id}) failed: {e}")
        return False


def upsert_signal_record(
    client: chromadb.ClientAPI,
    signal_id: str,
    text: str,
    metadata: dict,
) -> bool:
    """Embed and upsert a signal record. Returns True on success. Never raises."""
    try:
        embedding = _embed(text, task_type="RETRIEVAL_DOCUMENT")
        col = client.get_collection("macro_signals")
        col.upsert(
            ids=[signal_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return True
    except Exception as e:
        print(f"  WARNING: ChromaDB upsert_signal_record({signal_id}) failed: {e}")
        return False


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
    failures = 0
    docs_ok = 0
    for doc in all_docs:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        text = f"{doc.get('title', '')} {doc.get('content', '')}".strip()
        if not text:
            continue
        metadata = {
            "source": doc.get("source", ""),
            "ticker_mentions": "",
            "ingested_at": str(doc.get("ingested_at", "")),
            "value_chain_layer": doc.get("value_chain_layer", "application"),
        }
        if upsert_research_doc(client, str(doc_id), text, metadata):
            docs_ok += 1
        else:
            failures += 1
    sigs_ok = 0
    for sig in all_signals:
        sig_id = sig.get("id")
        if not sig_id:
            continue
        thesis = sig.get("thesis_update") or ""
        notes = ""
        if sig.get("macro_signal"):
            import json as _json
            try:
                notes = _json.loads(sig["macro_signal"]).get("notes", "")
            except Exception as e:
                print(f"  WARNING: could not parse macro_signal JSON: {e}")
        text = f"{thesis} {notes}".strip()
        if not text:
            continue
        metadata = {
            "regime": sig.get("market_regime", ""),
            "p_final": float(sig.get("p_final") or 0.0),
            "computed_at": str(sig.get("computed_at", "")),
        }
        if upsert_signal_record(client, f"signal_{sig_id}", text, metadata):
            sigs_ok += 1
        else:
            failures += 1
    if failures:
        # Don't write the sentinel — leave the backfill to retry on the next run
        # rather than locking in a partial/empty index.
        print(f"  ChromaDB: backfill incomplete — {docs_ok} docs, {sigs_ok} signals "
              f"embedded, {failures} failures. Will retry next run.")
        return
    Path(sentinel_path).write_text("done")
    print(f"  ChromaDB: backfill complete — {docs_ok} docs, {sigs_ok} signals")
