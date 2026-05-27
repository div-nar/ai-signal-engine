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
