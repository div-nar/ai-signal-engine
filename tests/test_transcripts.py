# ai-signal-engine/tests/test_transcripts.py
import pytest
from unittest.mock import patch, MagicMock
from ingestion.transcripts import fetch_edgar_filings, ingest_edgar


FAKE_SUBMISSION = {
    "filings": {
        "recent": {
            "accessionNumber": ["0000789019-26-000001", "0000789019-26-000002", "0000789019-26-000003"],
            "form": ["8-K", "10-Q", "8-K"],
            "filingDate": ["2026-04-15", "2026-04-01", "2026-03-10"],
            "primaryDocument": ["msft-8k-2026.htm", "msft-10q.htm", "msft-8k-march.htm"],
        }
    }
}

FAKE_FILING_TEXT = "Microsoft CapEx guidance: We expect to invest $80B in AI infrastructure in FY2026."


def test_fetch_edgar_filings_filters_8k_only():
    with patch("ingestion.transcripts.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_SUBMISSION
        mock_resp.text = FAKE_FILING_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        filings = fetch_edgar_filings(cik="0000789019", ticker="MSFT", max_filings=3)

    assert all(f["form"] == "8-K" for f in filings)
    assert len(filings) == 2


def test_fetch_edgar_filings_returns_content():
    with patch("ingestion.transcripts.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_SUBMISSION
        mock_resp.text = FAKE_FILING_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        filings = fetch_edgar_filings(cik="0000789019", ticker="MSFT", max_filings=3)

    assert "CapEx" in filings[0]["content"]
    assert filings[0]["url"].startswith("https://www.sec.gov")


def test_ingest_edgar_inserts_documents(tmp_path):
    from db import init_db, get_unscored_documents
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.transcripts.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_SUBMISSION
        mock_resp.text = FAKE_FILING_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        count = ingest_edgar(
            edgar_tickers={"MSFT": "0000789019"},
            max_per_ticker=3,
            db_path=db_path,
        )

    assert count == 2
    docs = get_unscored_documents(db_path)
    assert docs[0]["source"] == "edgar"
    assert docs[0]["value_chain_layer"] == "platform"


def test_ingest_edgar_skips_duplicates(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with patch("ingestion.transcripts.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_SUBMISSION
        mock_resp.text = FAKE_FILING_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        first = ingest_edgar({"MSFT": "0000789019"}, 3, db_path)
        second = ingest_edgar({"MSFT": "0000789019"}, 3, db_path)

    assert first == 2
    assert second == 0
