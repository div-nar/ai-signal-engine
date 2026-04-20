# ai-signal-engine/tests/test_rss.py
import pytest
from unittest.mock import patch, MagicMock
from ingestion.rss import fetch_rss_entries, ingest_rss


FAKE_FEED = {
    "entries": [
        {
            "title": "ASML Order Surge",
            "link": "https://www.semianalysis.com/asml-order-surge",
            "published": "Mon, 20 Apr 2026 06:00:00 +0000",
            "content": [{"value": "<p>ASML booked 12B EUR in Q1 orders...</p>"}],
            "summary": "ASML booked 12B EUR in Q1 orders...",
        },
        {
            "title": "GPU Market Update",
            "link": "https://www.semianalysis.com/gpu-market",
            "published": "Sun, 19 Apr 2026 06:00:00 +0000",
            "content": [{"value": "<p>H100 spot prices tightening...</p>"}],
            "summary": "H100 spot prices tightening...",
        },
    ]
}


def test_fetch_rss_entries_returns_list():
    with patch("ingestion.rss.feedparser.parse", return_value=FAKE_FEED):
        entries = fetch_rss_entries("https://www.semianalysis.com/feed")
    assert len(entries) == 2
    assert entries[0]["title"] == "ASML Order Surge"
    assert entries[0]["url"] == "https://www.semianalysis.com/asml-order-surge"
    assert "12B EUR" in entries[0]["content"]


def test_fetch_rss_entries_strips_html():
    with patch("ingestion.rss.feedparser.parse", return_value=FAKE_FEED):
        entries = fetch_rss_entries("https://www.semianalysis.com/feed")
    assert "<p>" not in entries[0]["content"]


def test_ingest_rss_inserts_new_documents(tmp_path):
    from db import init_db, get_unscored_documents
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.rss.feedparser.parse", return_value=FAKE_FEED):
        count = ingest_rss(
            feed_url="https://www.semianalysis.com/feed",
            value_chain_layer="platform",
            db_path=db_path,
        )

    assert count == 2
    docs = get_unscored_documents(db_path)
    assert len(docs) == 2
    assert docs[0]["source"] == "rss"
    assert docs[0]["value_chain_layer"] == "platform"


def test_ingest_rss_skips_duplicates(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.rss.feedparser.parse", return_value=FAKE_FEED):
        first = ingest_rss("https://www.semianalysis.com/feed", "platform", db_path)
        second = ingest_rss("https://www.semianalysis.com/feed", "platform", db_path)

    assert first == 2
    assert second == 0
