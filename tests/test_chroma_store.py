import os
import pytest
import chromadb
from unittest.mock import patch, MagicMock


def _fake_embed(text, task_type="RETRIEVAL_DOCUMENT"):
    """Return a deterministic 768-dim embedding without hitting the API."""
    import hashlib
    import random
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(768)]
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def chroma_client(tmp_path):
    from chroma_store import init_chroma
    return init_chroma(str(tmp_path / "chroma"))


def test_init_creates_both_collections(chroma_client):
    names = {c.name for c in chroma_client.list_collections()}
    assert "research_docs" in names
    assert "macro_signals" in names


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
    assert count_after_second == 1
