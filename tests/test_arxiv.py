# ai-signal-engine/tests/test_arxiv.py
import pytest
from unittest.mock import patch, MagicMock
from ingestion.arxiv import fetch_arxiv_papers, ingest_arxiv


def _make_fake_result(arxiv_id, title, summary, categories):
    r = MagicMock()
    r.entry_id = f"http://arxiv.org/abs/{arxiv_id}v1"
    r.title = title
    r.summary = summary
    r.categories = categories
    r.published = MagicMock()
    r.published.isoformat.return_value = "2026-04-20T00:00:00+00:00"
    return r


FAKE_RESULTS = [
    _make_fake_result("2404.00001", "Scaling Laws for LLMs", "We study compute scaling...", ["cs.LG", "cs.AI"]),
    _make_fake_result("2404.00002", "Hardware Efficient Attention", "New CUDA kernels for H100...", ["cs.AR"]),
]


def test_fetch_arxiv_papers_returns_list():
    with patch("ingestion.arxiv.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter(FAKE_RESULTS)
        mock_client_cls.return_value = mock_client

        papers = fetch_arxiv_papers(["cs.AI", "cs.LG"], max_results=10)

    assert len(papers) == 2
    assert papers[0]["title"] == "Scaling Laws for LLMs"
    assert papers[0]["url"] == "http://arxiv.org/abs/2404.00001v1"
    assert "compute scaling" in papers[0]["content"]


def test_fetch_arxiv_papers_maps_layer_by_category():
    with patch("ingestion.arxiv.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter(FAKE_RESULTS)
        mock_client_cls.return_value = mock_client

        papers = fetch_arxiv_papers(["cs.AR"], max_results=10)

    ar_paper = next(p for p in papers if "CUDA" in p["content"])
    assert ar_paper["value_chain_layer"] == "compute"


def test_ingest_arxiv_inserts_new_documents(tmp_path):
    from db import init_db, get_unscored_documents
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.arxiv.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter(FAKE_RESULTS)
        mock_client_cls.return_value = mock_client

        count = ingest_arxiv(categories=["cs.AI", "cs.LG"], max_results=10, db_path=db_path)

    assert count == 2
    docs = get_unscored_documents(db_path)
    assert docs[0]["source"] == "arxiv"


def test_ingest_arxiv_skips_duplicates(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.arxiv.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter(FAKE_RESULTS)
        mock_client_cls.return_value = mock_client

        first = ingest_arxiv(["cs.AI"], 10, db_path)

    with patch("ingestion.arxiv.arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = iter(FAKE_RESULTS)
        mock_client_cls.return_value = mock_client

        second = ingest_arxiv(["cs.AI"], 10, db_path)

    assert first == 2
    assert second == 0
