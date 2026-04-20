# ai-signal-engine/tests/test_db.py
import sqlite3
import tempfile
import os
import pytest
from db import init_db, insert_document, get_unscored_documents, mark_scored, insert_signal


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def test_init_db_creates_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert {"documents", "scores", "signals"} == tables


def test_insert_document_returns_id(db_path):
    doc_id = insert_document(
        db_path,
        source="rss",
        title="Test Article",
        url="https://example.com/test",
        published_at="2026-04-20T06:00:00",
        content="Full article text here",
        value_chain_layer="platform",
    )
    assert isinstance(doc_id, int)
    assert doc_id > 0


def test_insert_document_deduplicates_by_url(db_path):
    id1 = insert_document(db_path, source="rss", title="A", url="https://example.com/same",
                          published_at=None, content="x", value_chain_layer="compute")
    id2 = insert_document(db_path, source="rss", title="B", url="https://example.com/same",
                          published_at=None, content="y", value_chain_layer="compute")
    assert id2 is None  # duplicate → skip


def test_get_unscored_documents_returns_new_docs(db_path):
    insert_document(db_path, source="rss", title="A", url="https://a.com",
                    published_at=None, content="text", value_chain_layer="platform")
    insert_document(db_path, source="arxiv", title="B", url="https://b.com",
                    published_at=None, content="text2", value_chain_layer="compute")
    docs = get_unscored_documents(db_path)
    assert len(docs) == 2
    assert all(d["scored"] == 0 for d in docs)


def test_mark_scored_updates_flag(db_path):
    doc_id = insert_document(db_path, source="rss", title="C", url="https://c.com",
                              published_at=None, content="x", value_chain_layer="domain")
    mark_scored(db_path, [doc_id])
    docs = get_unscored_documents(db_path)
    assert len(docs) == 0


def test_insert_signal_roundtrip(db_path):
    signal_id = insert_signal(db_path, {
        "p_final": 0.82,
        "stock_conviction": '{"NVDA": 0.85}',
        "sector_tilt": '{"Information Technology": 0.05}',
        "supply_demand_balance": 0.34,
        "market_regime": "compute_constrained",
        "signal_confidence": 0.76,
        "thesis_stress": False,
        "signal_age_days": 1,
        "sources_ingested": 5,
        "signal_breakdown": '{"compute": 0.25}',
        "thesis_update": "ASML backlog up 18% QoQ",
    })
    assert signal_id > 0
