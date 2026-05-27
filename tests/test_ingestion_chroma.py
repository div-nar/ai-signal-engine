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
    call_args = mock_upsert.call_args[0]
    assert call_args[2] == "ASML Q1 backlog surge"  # text = title + summary


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

    assert mock_upsert.call_count == 1


def test_rss_ingest_skips_upsert_when_no_chroma_client(tmp_path):
    from ingestion.rss import ingest_rss
    from db import init_db
    db = str(tmp_path / "test.db")
    init_db(db)

    fake_entries = [{"title": "T", "link": "http://x.com/1", "published": "2026-05-01", "summary": "s"}]
    with patch("ingestion.rss.feedparser.parse", return_value=MagicMock(entries=fake_entries)), \
         patch("chroma_store.upsert_research_doc") as mock_upsert:
        ingest_rss("http://fake.com/feed", "compute", db)

    mock_upsert.assert_not_called()
