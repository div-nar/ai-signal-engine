# ai-signal-engine/ingestion/transcripts.py
import httpx
from db import insert_document, DEFAULT_DB

EDGAR_BASE = "https://data.sec.gov"
_HEADERS = {"User-Agent": "ai-signal-engine divith@dognosis.tech"}


def _cik_padded(cik: str) -> str:
    """Zero-pad CIK to 10 digits as required by EDGAR API."""
    return cik.lstrip("0").zfill(10)


def fetch_edgar_filings(cik: str, ticker: str, max_filings: int) -> list[dict]:
    """Fetch recent 8-K filings for a given company CIK."""
    padded = _cik_padded(cik)          # 10-digit zero-padded for submissions API
    cik_int = cik.lstrip("0") or "0"   # bare integer for Archives path
    url = f"{EDGAR_BASE}/submissions/CIK{padded}.json"
    resp = httpx.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    accession_numbers = recent["accessionNumber"]
    forms = recent["form"]
    dates = recent["filingDate"]
    docs = recent["primaryDocument"]

    filings = []
    for acc, form, date, doc in zip(accession_numbers, forms, dates, docs):
        if form != "8-K":
            continue
        if len(filings) >= max_filings:
            break

        acc_fmt = acc.replace("-", "")
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_fmt}/{doc}"

        try:
            filing_resp = httpx.get(filing_url, headers=_HEADERS, timeout=30)
            filing_resp.raise_for_status()
            content = filing_resp.text[:50_000]
        except Exception as exc:
            content = f"[fetch_error: {exc}]"

        filings.append({
            "url": filing_url,
            "title": f"{ticker} 8-K {date}",
            "published": date,
            "content": content,
            "form": form,
        })

    return filings


def ingest_edgar(
    edgar_tickers: dict[str, str],
    max_per_ticker: int,
    db_path: str = str(DEFAULT_DB),
) -> int:
    """Ingest 8-K filings for all configured tickers. Returns count of new docs."""
    count = 0
    for ticker, cik in edgar_tickers.items():
        filings = fetch_edgar_filings(cik=cik, ticker=ticker, max_filings=max_per_ticker)
        for f in filings:
            result = insert_document(
                db_path=db_path,
                source="edgar",
                title=f["title"],
                url=f["url"],
                published_at=f["published"],
                content=f["content"],
                value_chain_layer="infrastructure",
            )
            if result is not None:
                count += 1
    return count
