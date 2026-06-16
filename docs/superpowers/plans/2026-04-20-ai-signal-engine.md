# AI Signal Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone service that ingests forward-looking AI compute signals from SemiAnalysis RSS, arXiv, and SEC EDGAR, scores them via a single Gemini 2.5 Pro call, and exports structured JSON outputs consumed by `ai-portfolio-backtest/strategy.py`.

**Architecture:** SQLite stores raw documents and intermediate scores. A single Gemini 2.5 Pro call reads all unscored documents assembled by value chain layer and outputs structured portfolio weights + narrative. Three JSON files are exported to `ai-portfolio-backtest/data/` on each run.

**Tech Stack:** Python 3.11+, feedparser, arxiv, httpx, google-genai, SQLite (stdlib), pytest

---

## File Structure

| File | Responsibility |
|---|---|
| `ai-signal-engine/db.py` | SQLite schema creation + read/write helpers |
| `ai-signal-engine/config.py` | Sources, ticker universe, value chain layer tags |
| `ai-signal-engine/ingestion/__init__.py` | Empty package marker |
| `ai-signal-engine/ingestion/rss.py` | SemiAnalysis RSS ingestion |
| `ai-signal-engine/ingestion/arxiv.py` | arXiv cs.AI/cs.LG/cs.AR paper ingestion |
| `ai-signal-engine/ingestion/transcripts.py` | SEC EDGAR 8-K earnings transcript ingestion |
| `ai-signal-engine/scoring/__init__.py` | Empty package marker |
| `ai-signal-engine/scoring/gemini_scorer.py` | Gemini 2.5 Pro → structured portfolio output |
| `ai-signal-engine/export.py` | Writes 3 JSON files to `ai-portfolio-backtest/data/` |
| `ai-signal-engine/main.py` | Orchestrator — ingest, score, export |
| `ai-signal-engine/requirements.txt` | Python dependencies |
| `ai-portfolio-backtest/strategy.py` | Modify `compute_weights()` to blend stock_signals.json |
| `ai-signal-engine/tests/test_db.py` | Tests for schema + helpers |
| `ai-signal-engine/tests/test_rss.py` | Tests for RSS ingestion |
| `ai-signal-engine/tests/test_arxiv.py` | Tests for arXiv ingestion |
| `ai-signal-engine/tests/test_transcripts.py` | Tests for EDGAR ingestion |
| `ai-signal-engine/tests/test_gemini_scorer.py` | Tests for Gemini scorer |
| `ai-signal-engine/tests/test_export.py` | Tests for JSON export |

---

## Task 1: Project scaffold + requirements

**Files:**
- Create: `ai-signal-engine/requirements.txt`
- Create: `ai-signal-engine/ingestion/__init__.py`
- Create: `ai-signal-engine/scoring/__init__.py`
- Create: `ai-signal-engine/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
feedparser==6.0.11
arxiv==2.1.3
httpx==0.27.0
google-genai==1.9.0
pytest==8.2.0
pytest-mock==3.14.0
```

- [ ] **Step 2: Create package markers**

```bash
touch /Users/div-nar/sideproj/ai-signal-engine/ingestion/__init__.py
touch /Users/div-nar/sideproj/ai-signal-engine/scoring/__init__.py
touch /Users/div-nar/sideproj/ai-signal-engine/tests/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 4: Commit**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
git add requirements.txt ingestion/__init__.py scoring/__init__.py tests/__init__.py
git commit -m "chore: scaffold ai-signal-engine package structure"
```

---

## Task 2: SQLite schema + DB helpers (`db.py`)

**Files:**
- Create: `ai-signal-engine/db.py`
- Create: `ai-signal-engine/tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Implement `db.py`**

```python
# ai-signal-engine/db.py
import json
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).parent / "signals.db"


def init_db(db_path: str = str(DEFAULT_DB)) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id                INTEGER PRIMARY KEY,
            source            TEXT NOT NULL,
            title             TEXT NOT NULL,
            url               TEXT UNIQUE NOT NULL,
            published_at      TIMESTAMP,
            content           TEXT,
            ingested_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scored            BOOLEAN DEFAULT FALSE,
            value_chain_layer TEXT
        );

        CREATE TABLE IF NOT EXISTS scores (
            id           INTEGER PRIMARY KEY,
            doc_id       INTEGER REFERENCES documents(id),
            p_delta      REAL,
            stock_scores TEXT,
            thesis_tags  TEXT,
            scored_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS signals (
            id                    INTEGER PRIMARY KEY,
            computed_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            p_final               REAL,
            stock_conviction      TEXT,
            sector_tilt           TEXT,
            supply_demand_balance REAL,
            market_regime         TEXT,
            signal_confidence     REAL,
            thesis_stress         BOOLEAN,
            signal_age_days       INTEGER,
            sources_ingested      INTEGER,
            signal_breakdown      TEXT,
            thesis_update         TEXT
        );
    """)
    conn.commit()
    conn.close()


def insert_document(
    db_path: str,
    source: str,
    title: str,
    url: str,
    published_at: Optional[str],
    content: str,
    value_chain_layer: str,
) -> Optional[int]:
    """Insert a document. Returns row id, or None if URL already exists."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO documents (source, title, url, published_at, content, value_chain_layer)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source, title, url, published_at, content, value_chain_layer),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_unscored_documents(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM documents WHERE scored = 0 ORDER BY ingested_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_scored(db_path: str, doc_ids: list[int]) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE documents SET scored = 1 WHERE id IN ({','.join('?' * len(doc_ids))})",
        doc_ids,
    )
    conn.commit()
    conn.close()


def insert_signal(db_path: str, data: dict) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO signals
           (p_final, stock_conviction, sector_tilt, supply_demand_balance,
            market_regime, signal_confidence, thesis_stress, signal_age_days,
            sources_ingested, signal_breakdown, thesis_update)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["p_final"],
            data["stock_conviction"],
            data["sector_tilt"],
            data["supply_demand_balance"],
            data["market_regime"],
            data["signal_confidence"],
            int(data["thesis_stress"]),
            data["signal_age_days"],
            data["sources_ingested"],
            data["signal_breakdown"],
            data["thesis_update"],
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_db.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: SQLite schema and db helpers"
```

---

## Task 3: Config (`config.py`)

**Files:**
- Create: `ai-signal-engine/config.py`

No tests needed — config is pure data. It's validated implicitly by downstream tests.

- [ ] **Step 1: Create `config.py`**

```python
# ai-signal-engine/config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DB_PATH = str(ROOT / "signals.db")
BACKTEST_DATA_DIR = ROOT.parent / "ai-portfolio-backtest" / "data"

# ── SemiAnalysis RSS ───────────────────────────────────────────────────────────
RSS_FEEDS = [
    {
        "url": "https://www.semianalysis.com/feed",
        "value_chain_layer": "platform",  # Gemini maps specifics; default layer
    }
]

# ── arXiv ─────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.AR"]
ARXIV_MAX_RESULTS = 50  # per category per run

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
# Hyperscalers whose 8-K CapEx guidance is highest signal
EDGAR_TICKERS = {
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
}
EDGAR_FORM_TYPE = "8-K"
EDGAR_MAX_PER_TICKER = 3  # most recent filings

# ── Value chain layer tags ─────────────────────────────────────────────────────
VALUE_CHAIN_LAYERS = ["compute", "power", "infrastructure", "platform", "application", "domain"]

# ── Ticker universe (90 stocks) ────────────────────────────────────────────────
TICKER_UNIVERSE = [
    # Information Technology — compute stack (chips, EDA, foundry, memory, packaging)
    "NVDA", "AMD", "AVGO", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "ARM", "TOELY",
    "SNPS", "CDNS", "MRVL", "ASX", "SHECY", "ON",
    # Note: SK Hynix (HXSCL) and Samsung (SSNNF) are Korean-listed; no reliable US price data.
    # Information Technology — software and cloud
    "MSFT", "ORCL", "IBM", "SAP", "INFY", "CSCO", "PLTR", "CRM", "NOW", "DDOG",
    "CRWD", "ADBE", "INTU", "MU", "QCOM", "TXN", "ADI", "AAPL",
    # Communication Services
    "GOOGL", "META", "NFLX", "TMUS", "BIDU", "BABA", "TCEHY",
    # Consumer Discretionary
    "AMZN", "TSLA", "BKNG",
    # Consumer Staples
    "COST", "WMT",
    # Energy
    "XOM", "CVX",
    # Financials
    "V", "MA", "JPM", "GS", "COIN", "BLK", "SCHW", "ICE", "SPGI", "IBKR",
    # Health Care
    "LLY", "UNH", "ABBV", "ISRG", "TMO", "AMGN", "GILD", "VRTX", "REGN", "MRK", "JNJ",
    # Industrials
    "GE", "GEV", "HON", "CAT", "DE", "RTX", "AXON", "VRT", "LMT", "NOC", "SMTOY",
    # Materials
    "ALB", "LIN", "FCX",
    # Real Estate
    "EQIX", "DLR", "IRM", "AMT",
    # Utilities
    "VST", "CEG", "NRG", "NEE", "ETN", "PWR", "EIX", "AES",
]

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
GEMINI_MAX_OUTPUT_TOKENS = 8192

# ── Guardrails ─────────────────────────────────────────────────────────────────
MAX_STOCK_WEIGHT = 0.10          # hard cap per stock
MIN_HEDGE_SECTOR_WEIGHT = 0.02   # Health Care + Consumer Staples floor
MAX_TURNOVER_VS_PREV = 0.20      # max weight change vs last run
WEIGHT_SUM_TOLERANCE = 0.01      # reject if |sum - 1.0| > this
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "feat: config with sources, universe, guardrails"
```

---

## Task 4: RSS ingestion (`ingestion/rss.py`)

**Files:**
- Create: `ai-signal-engine/ingestion/rss.py`
- Create: `ai-signal-engine/tests/test_rss.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    # Should not contain HTML tags
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
    assert second == 0  # all duplicates
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_rss.py -v
```

Expected: `ModuleNotFoundError: No module named 'ingestion.rss'`

- [ ] **Step 3: Implement `ingestion/rss.py`**

```python
# ai-signal-engine/ingestion/rss.py
import re
from html.parser import HTMLParser
from typing import Optional
import feedparser
from db import init_db, insert_document, DEFAULT_DB


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return " ".join(self._parts).strip()


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return re.sub(r"\s+", " ", stripper.get_text())


def fetch_rss_entries(feed_url: str) -> list[dict]:
    """Parse RSS feed and return list of {title, url, published, content} dicts."""
    feed = feedparser.parse(feed_url)
    entries = []
    for entry in feed.get("entries", []):
        # Prefer full content over summary
        if entry.get("content"):
            raw_content = entry["content"][0]["value"]
        else:
            raw_content = entry.get("summary", "")

        entries.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published": entry.get("published", None),
            "content": _strip_html(raw_content),
        })
    return entries


def ingest_rss(
    feed_url: str,
    value_chain_layer: str,
    db_path: str = str(DEFAULT_DB),
) -> int:
    """Fetch RSS entries and insert new ones into DB. Returns count of new docs."""
    entries = fetch_rss_entries(feed_url)
    count = 0
    for e in entries:
        result = insert_document(
            db_path=db_path,
            source="rss",
            title=e["title"],
            url=e["url"],
            published_at=e["published"],
            content=e["content"],
            value_chain_layer=value_chain_layer,
        )
        if result is not None:
            count += 1
    return count
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_rss.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion/rss.py tests/test_rss.py
git commit -m "feat: SemiAnalysis RSS ingestion with HTML stripping"
```

---

## Task 5: arXiv ingestion (`ingestion/arxiv.py`)

**Files:**
- Create: `ai-signal-engine/ingestion/arxiv.py`
- Create: `ai-signal-engine/tests/test_arxiv.py`

- [ ] **Step 1: Write the failing tests**

```python
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

    # cs.AR (hardware) → compute layer
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
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_arxiv.py -v
```

Expected: `ModuleNotFoundError: No module named 'ingestion.arxiv'`

- [ ] **Step 3: Implement `ingestion/arxiv.py`**

```python
# ai-signal-engine/ingestion/arxiv.py
import arxiv
from db import insert_document, DEFAULT_DB

# Map arXiv categories to value chain layers
_CATEGORY_LAYER = {
    "cs.AR": "compute",       # Hardware architecture
    "cs.AI": "platform",      # AI methods → platform layer
    "cs.LG": "platform",      # ML → platform layer
}
_DEFAULT_LAYER = "application"


def fetch_arxiv_papers(categories: list[str], max_results: int) -> list[dict]:
    """Fetch recent papers from arXiv for given categories."""
    import arxiv as _arxiv
    client = _arxiv.Client()

    # Build query: one search covering all categories
    category_query = " OR ".join(f"cat:{c}" for c in categories)
    search = _arxiv.Search(
        query=category_query,
        max_results=max_results,
        sort_by=_arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        # Determine layer from primary category
        primary_cat = result.categories[0] if result.categories else ""
        layer = _CATEGORY_LAYER.get(primary_cat, _DEFAULT_LAYER)

        papers.append({
            "title": result.title,
            "url": result.entry_id,
            "published": result.published.isoformat(),
            "content": result.summary,
            "value_chain_layer": layer,
        })
    return papers


def ingest_arxiv(
    categories: list[str],
    max_results: int,
    db_path: str = str(DEFAULT_DB),
) -> int:
    """Fetch arXiv papers and insert new ones. Returns count of new docs."""
    papers = fetch_arxiv_papers(categories, max_results)
    count = 0
    for p in papers:
        result = insert_document(
            db_path=db_path,
            source="arxiv",
            title=p["title"],
            url=p["url"],
            published_at=p["published"],
            content=p["content"],
            value_chain_layer=p["value_chain_layer"],
        )
        if result is not None:
            count += 1
    return count
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_arxiv.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion/arxiv.py tests/test_arxiv.py
git commit -m "feat: arXiv paper ingestion with value chain layer mapping"
```

---

## Task 6: EDGAR transcript ingestion (`ingestion/transcripts.py`)

**Files:**
- Create: `ai-signal-engine/ingestion/transcripts.py`
- Create: `ai-signal-engine/tests/test_transcripts.py`

- [ ] **Step 1: Write the failing tests**

```python
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

    # Only 8-K forms returned
    assert all(f["form"] == "8-K" for f in filings)
    assert len(filings) == 2  # two 8-K out of three total


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

    assert count == 2  # two 8-K filings
    docs = get_unscored_documents(db_path)
    assert docs[0]["source"] == "edgar"
    assert docs[0]["value_chain_layer"] == "infrastructure"


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
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_transcripts.py -v
```

Expected: `ModuleNotFoundError: No module named 'ingestion.transcripts'`

- [ ] **Step 3: Implement `ingestion/transcripts.py`**

```python
# ai-signal-engine/ingestion/transcripts.py
import httpx
from db import insert_document, DEFAULT_DB

EDGAR_BASE = "https://data.sec.gov"
EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/full-index"
_HEADERS = {"User-Agent": "ai-signal-engine divith@dognosis.tech"}


def _cik_padded(cik: str) -> str:
    """Zero-pad CIK to 10 digits as required by EDGAR API."""
    return cik.lstrip("0").zfill(10)


def fetch_edgar_filings(cik: str, ticker: str, max_filings: int) -> list[dict]:
    """Fetch recent 8-K filings for a given company CIK. Returns list of {url, content, form, title}."""
    padded = _cik_padded(cik)
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
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{padded.lstrip('0')}/{acc_fmt}/{doc}"

        try:
            filing_resp = httpx.get(filing_url, headers=_HEADERS, timeout=30)
            filing_resp.raise_for_status()
            content = filing_resp.text[:50_000]  # cap at 50K chars
        except Exception:
            content = ""

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
                value_chain_layer="infrastructure",  # CapEx guidance = infrastructure layer
            )
            if result is not None:
                count += 1
    return count
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_transcripts.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ingestion/transcripts.py tests/test_transcripts.py
git commit -m "feat: SEC EDGAR 8-K ingestion"
```

---

## Task 7: Gemini scorer (`scoring/gemini_scorer.py`)

**Files:**
- Create: `ai-signal-engine/scoring/gemini_scorer.py`
- Create: `ai-signal-engine/tests/test_gemini_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# ai-signal-engine/tests/test_gemini_scorer.py
import json
import pytest
from unittest.mock import patch, MagicMock
from scoring.gemini_scorer import (
    build_signal_context,
    parse_gemini_response,
    apply_guardrails,
    score_documents,
)

SAMPLE_DOCS = [
    {
        "id": 1, "source": "rss", "title": "ASML Q1 Backlog Surge",
        "content": "ASML reported record order backlog of 12B EUR in Q1 2026.",
        "published_at": "2026-04-15", "value_chain_layer": "compute",
    },
    {
        "id": 2, "source": "edgar", "title": "MSFT 8-K 2026-04-10",
        "content": "Microsoft expects to invest $80B in AI infrastructure in FY2026.",
        "published_at": "2026-04-10", "value_chain_layer": "infrastructure",
    },
    {
        "id": 3, "source": "arxiv", "title": "Scaling Laws Updated",
        "content": "Compute optimal training now requires 10^26 FLOPs for frontier models.",
        "published_at": "2026-04-12", "value_chain_layer": "platform",
    },
]

VALID_GEMINI_OUTPUT = {
    "p_score": 0.82,
    "market_regime": "compute_constrained",
    "supply_demand_balance": 0.34,
    "portfolio": [
        {"ticker": "NVDA", "weight": 0.10, "conviction": 0.90, "reasoning": "ASML backlog signals GPU supply tightening in 2-3Q."},
        {"ticker": "ASML", "weight": 0.09, "conviction": 0.85, "reasoning": "Record backlog directly measures EUV capacity expansion."},
        {"ticker": "TSM", "weight": 0.08, "conviction": 0.80, "reasoning": "Advanced node utilization rising as hyperscaler orders accelerate."},
        {"ticker": "MU", "weight": 0.07, "conviction": 0.75, "reasoning": "HBM3E demand from AI training runs increasing QoQ."},
        {"ticker": "MSFT", "weight": 0.07, "conviction": 0.78, "reasoning": "$80B CapEx signals multi-quarter GPU cluster buildout."},
        {"ticker": "AMZN", "weight": 0.06, "conviction": 0.72, "reasoning": "AWS Trainium2 ramp indicates sustained AI workload growth."},
        {"ticker": "GOOGL", "weight": 0.06, "conviction": 0.70, "reasoning": "TPU v5 deployment scaling; Gemini inference at scale."},
        {"ticker": "AMD", "weight": 0.05, "conviction": 0.68, "reasoning": "MI300X gaining share in inferencing segment."},
        {"ticker": "AVGO", "weight": 0.05, "conviction": 0.65, "reasoning": "Custom ASIC revenue from hyperscalers accelerating."},
        {"ticker": "ANET", "weight": 0.04, "conviction": 0.62, "reasoning": "400G/800G switch demand tied to GPU cluster networking."},
        {"ticker": "LLY", "weight": 0.03, "conviction": 0.45, "reasoning": "AI drug discovery pipeline expanding."},
        {"ticker": "COST", "weight": 0.02, "conviction": 0.20, "reasoning": "Hedge: consumer staples floor."},
    ],
    "signal_confidence": 0.76,
    "thesis_stress": False,
    "thesis_update": "ASML order backlog up 18% QoQ signals compute expansion continues.",
}


def test_build_signal_context_groups_by_layer():
    context = build_signal_context(SAMPLE_DOCS)
    assert "compute" in context.lower()
    assert "infrastructure" in context.lower()
    assert "platform" in context.lower()
    assert "ASML Q1 Backlog Surge" in context
    assert "MSFT 8-K" in context


def test_parse_gemini_response_valid():
    result = parse_gemini_response(json.dumps(VALID_GEMINI_OUTPUT))
    assert result["p_score"] == 0.82
    assert result["market_regime"] == "compute_constrained"
    assert len(result["portfolio"]) == 12


def test_parse_gemini_response_strips_markdown():
    wrapped = f"```json\n{json.dumps(VALID_GEMINI_OUTPUT)}\n```"
    result = parse_gemini_response(wrapped)
    assert result["p_score"] == 0.82


def test_apply_guardrails_caps_max_weight():
    output = dict(VALID_GEMINI_OUTPUT)
    # Inject a stock over 10%
    output["portfolio"] = [{"ticker": "NVDA", "weight": 0.25, "conviction": 0.9, "reasoning": "x"}]
    guarded = apply_guardrails(output, prev_weights={})
    nvda = next(p for p in guarded["portfolio"] if p["ticker"] == "NVDA")
    assert nvda["weight"] <= 0.10


def test_apply_guardrails_weights_sum_to_one():
    output = dict(VALID_GEMINI_OUTPUT)
    guarded = apply_guardrails(output, prev_weights={})
    total = sum(p["weight"] for p in guarded["portfolio"])
    assert abs(total - 1.0) < 0.01


def test_score_documents_calls_gemini_and_returns_structured(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_GEMINI_OUTPUT)

    with patch("scoring.gemini_scorer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = score_documents(docs=SAMPLE_DOCS, db_path=db_path, prev_weights={})

    assert result["p_final"] == 0.82
    assert result["market_regime"] == "compute_constrained"
    assert "NVDA" in result["stock_conviction"]
    assert result["sources_ingested"] == 3
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_gemini_scorer.py -v
```

Expected: `ModuleNotFoundError: No module named 'scoring.gemini_scorer'`

- [ ] **Step 3: Implement `scoring/gemini_scorer.py`**

```python
# ai-signal-engine/scoring/gemini_scorer.py
import json
import os
import re
from collections import defaultdict
from typing import Optional

from google import genai

from config import (
    GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS,
    TICKER_UNIVERSE, MAX_STOCK_WEIGHT, MAX_TURNOVER_VS_PREV,
    WEIGHT_SUM_TOLERANCE, VALUE_CHAIN_LAYERS,
)
from db import insert_signal, DEFAULT_DB


_SYSTEM_PROMPT = """You are a portfolio manager for an AI-focused long-only equity fund.
Your thesis is Aschenbrenner's: AI is on an exponential compute trajectory toward AGI.
Your job is to identify which stocks in the universe will benefit most from the NEXT 1-4
quarters of AI compute expansion.

Universe: {universe}

Constraints:
- Max 10% weight per stock
- All weights must sum to exactly 1.0
- Long-only (no shorts)
- Include at least one Health Care and one Consumer Staples stock as hedges (min 2% each)

Output ONLY valid JSON matching this schema exactly:
{{
  "p_score": <float 0-1, Aschenbrenner probability this week>,
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling">,
  "supply_demand_balance": <float, positive=demand>supply>,
  "portfolio": [
    {{"ticker": <str>, "weight": <float>, "conviction": <float 0-1>, "reasoning": <str 1-2 sentences>}}
  ],
  "signal_confidence": <float 0-1>,
  "thesis_stress": <bool>,
  "thesis_update": <str, what changed vs last run>
}}"""


def build_signal_context(docs: list[dict]) -> str:
    """Assemble documents into a structured prompt context organised by value chain layer."""
    by_layer = defaultdict(list)
    for doc in docs:
        layer = doc.get("value_chain_layer", "application")
        by_layer[layer].append(doc)

    sections = []
    for layer in VALUE_CHAIN_LAYERS:
        layer_docs = by_layer.get(layer, [])
        if not layer_docs:
            continue
        header = f"\n### {layer.upper()} LAYER SIGNALS\n"
        entries = []
        for d in layer_docs:
            entries.append(
                f"Source: {d['source'].upper()} | Date: {d.get('published_at', 'unknown')}\n"
                f"Title: {d['title']}\n"
                f"Content: {d['content'][:2000]}\n"  # cap per-doc at 2K chars
            )
        sections.append(header + "\n---\n".join(entries))

    return "\n".join(sections)


def parse_gemini_response(text: str) -> dict:
    """Parse Gemini response text into a dict, stripping markdown code fences if present."""
    # Strip ```json ... ``` wrapper if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def apply_guardrails(output: dict, prev_weights: dict) -> dict:
    """Apply hard constraints to Gemini portfolio output."""
    portfolio = output["portfolio"]

    # 1. Cap max weight
    for p in portfolio:
        p["weight"] = min(p["weight"], MAX_STOCK_WEIGHT)

    # 2. Normalize to sum to 1.0
    total = sum(p["weight"] for p in portfolio)
    if total > 0:
        for p in portfolio:
            p["weight"] = p["weight"] / total

    # 3. Apply turnover cap vs previous weights
    if prev_weights:
        for p in portfolio:
            prev = prev_weights.get(p["ticker"], 0.0)
            delta = p["weight"] - prev
            if abs(delta) > MAX_TURNOVER_VS_PREV:
                p["weight"] = prev + MAX_TURNOVER_VS_PREV * (1 if delta > 0 else -1)
        # Re-normalize after turnover cap
        total = sum(p["weight"] for p in portfolio)
        if total > 0:
            for p in portfolio:
                p["weight"] = p["weight"] / total

    output["portfolio"] = portfolio
    return output


def score_documents(
    docs: list[dict],
    db_path: str = str(DEFAULT_DB),
    prev_weights: Optional[dict] = None,
) -> dict:
    """Call Gemini with assembled signal context. Returns structured signal dict."""
    if prev_weights is None:
        prev_weights = {}

    context = build_signal_context(docs)
    universe_str = ", ".join(TICKER_UNIVERSE)
    system = _SYSTEM_PROMPT.format(universe=universe_str)

    user_prompt = f"""Given these forward-looking signals, output portfolio weights for next week.
Weight stocks that will benefit from what is being *committed to* today, not what has already happened.

{context}

[TASK]
Output your portfolio JSON now. Remember: weights must sum to 1.0, max 10% per stock."""

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system}\n\n{user_prompt}",
        config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
    )

    raw = parse_gemini_response(response.text)

    # Validate weight sum
    weight_sum = sum(p["weight"] for p in raw.get("portfolio", []))
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Gemini output weights sum to {weight_sum:.4f}, not 1.0")

    guarded = apply_guardrails(raw, prev_weights)

    # Build structured signal dict for DB + export
    conviction_map = {p["ticker"]: p["conviction"] for p in guarded["portfolio"]}
    reasoning_map = {p["ticker"]: p["reasoning"] for p in guarded["portfolio"]}
    weight_map = {p["ticker"]: p["weight"] for p in guarded["portfolio"]}

    # Compute sector tilt (deviation from equal weight across sectors)
    # Simple pass-through for now — Gemini informs this implicitly via weights
    signal = {
        "p_final": guarded["p_score"],
        "stock_conviction": json.dumps(conviction_map),
        "stock_weights": json.dumps(weight_map),
        "stock_reasoning": json.dumps(reasoning_map),
        "sector_tilt": json.dumps({}),  # populated by export.py
        "supply_demand_balance": guarded.get("supply_demand_balance", 0.0),
        "market_regime": guarded["market_regime"],
        "signal_confidence": guarded.get("signal_confidence", 0.5),
        "thesis_stress": guarded.get("thesis_stress", False),
        "signal_age_days": 0,
        "sources_ingested": len(docs),
        "signal_breakdown": json.dumps({}),  # populated by export.py
        "thesis_update": guarded.get("thesis_update", ""),
    }

    return signal
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_gemini_scorer.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring/gemini_scorer.py tests/test_gemini_scorer.py
git commit -m "feat: Gemini 2.5 Pro scorer with guardrails"
```

---

## Task 8: Export (`export.py`)

**Files:**
- Create: `ai-signal-engine/export.py`
- Create: `ai-signal-engine/tests/test_export.py`

- [ ] **Step 1: Write the failing tests**

```python
# ai-signal-engine/tests/test_export.py
import json
import pytest
from pathlib import Path
from export import export_signal


SAMPLE_SIGNAL = {
    "p_final": 0.82,
    "stock_conviction": json.dumps({"NVDA": 0.90, "ASML": 0.85, "TSM": 0.80}),
    "stock_weights": json.dumps({"NVDA": 0.10, "ASML": 0.09, "TSM": 0.08}),
    "stock_reasoning": json.dumps({"NVDA": "ASML backlog signals GPU tightening.", "ASML": "Record backlog.", "TSM": "Node utilization rising."}),
    "sector_tilt": json.dumps({}),
    "supply_demand_balance": 0.34,
    "market_regime": "compute_constrained",
    "signal_confidence": 0.76,
    "thesis_stress": False,
    "signal_age_days": 0,
    "sources_ingested": 3,
    "signal_breakdown": json.dumps({"compute": 0.25, "infrastructure": 0.20}),
    "thesis_update": "ASML backlog up 18% QoQ.",
}


def test_export_signal_writes_three_files(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))

    assert (tmp_path / "p_estimate.json").exists()
    assert (tmp_path / "stock_signals.json").exists()
    assert (tmp_path / "market_regime.json").exists()


def test_export_signal_p_estimate_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "p_estimate.json").read_text())
    assert data["p"] == 0.82
    assert "generated_at" in data


def test_export_signal_stock_signals_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "stock_signals.json").read_text())
    assert data["conviction"]["NVDA"] == 0.90
    assert "ASML backlog" in data["reasoning"]["NVDA"]
    assert "generated_at" in data


def test_export_signal_market_regime_content(tmp_path):
    export_signal(SAMPLE_SIGNAL, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "market_regime.json").read_text())
    assert data["market_regime"] == "compute_constrained"
    assert data["supply_demand_balance"] == 0.34
    assert data["signal_confidence"] == 0.76
    assert data["thesis_stress"] is False
    assert "generated_at" in data
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_export.py -v
```

Expected: `ModuleNotFoundError: No module named 'export'`

- [ ] **Step 3: Implement `export.py`**

```python
# ai-signal-engine/export.py
import json
from datetime import datetime, timezone
from pathlib import Path

from config import BACKTEST_DATA_DIR


def export_signal(signal: dict, output_dir: str = str(BACKTEST_DATA_DIR)) -> None:
    """Write p_estimate.json, stock_signals.json, and market_regime.json."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    conviction = json.loads(signal["stock_conviction"])
    reasoning = json.loads(signal["stock_reasoning"])
    signal_breakdown = json.loads(signal["signal_breakdown"])

    # 1. p_estimate.json — consumed by strategy.py
    p_file = output / "p_estimate.json"
    p_file.write_text(json.dumps({
        "p": signal["p_final"],
        "generated_at": now,
    }, indent=2))

    # 2. stock_signals.json — consumed by strategy.py for conviction blending
    stock_signals_file = output / "stock_signals.json"
    stock_signals_file.write_text(json.dumps({
        "generated_at": now,
        "conviction": conviction,
        "reasoning": reasoning,
    }, indent=2))

    # 3. market_regime.json — consumed by layers 3–5
    market_regime_file = output / "market_regime.json"
    market_regime_file.write_text(json.dumps({
        "generated_at": now,
        "market_regime": signal["market_regime"],
        "supply_demand_balance": signal["supply_demand_balance"],
        "sector_tilt": json.loads(signal["sector_tilt"]),
        "signal_confidence": signal["signal_confidence"],
        "thesis_stress": bool(signal["thesis_stress"]),
        "thesis_update": signal["thesis_update"],
        "signal_breakdown": signal_breakdown,
    }, indent=2))

    print(f"Exported signals to {output}/")
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | sources={signal['sources_ingested']}")
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/test_export.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add export.py tests/test_export.py
git commit -m "feat: signal export to p_estimate.json, stock_signals.json, market_regime.json"
```

---

## Task 9: Orchestrator (`main.py`)

**Files:**
- Create: `ai-signal-engine/main.py`

No separate unit test — the orchestrator is integration-tested by running it with `--dry-run`.

- [ ] **Step 1: Create `main.py`**

```python
# ai-signal-engine/main.py
"""
AI Signal Engine — Main orchestrator.

Usage:
    python main.py            # full run: ingest, score, export
    python main.py --force    # force re-score even if no new documents
    python main.py --dry-run  # ingest + score but don't write JSON files

Prerequisites:
    export GEMINI_API_KEY=...
    pip install -r requirements.txt
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DB_PATH, RSS_FEEDS, ARXIV_CATEGORIES, ARXIV_MAX_RESULTS,
    EDGAR_TICKERS,
)
from db import init_db, get_unscored_documents, mark_scored, insert_signal
from ingestion.rss import ingest_rss
from ingestion.arxiv import ingest_arxiv
from ingestion.transcripts import ingest_edgar
from scoring.gemini_scorer import score_documents
from export import export_signal


def get_prev_weights(db_path: str) -> dict:
    """Load stock weights from most recent signals row."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT stock_conviction FROM signals ORDER BY computed_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row and row[0]:
        return json.loads(row[0])
    return {}


def main():
    parser = argparse.ArgumentParser(description="AI Signal Engine")
    parser.add_argument("--force", action="store_true",
                        help="Score even if no new documents were ingested")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest and score but don't write output JSON files")
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine starting...")

    # 1. Init DB
    init_db(DB_PATH)

    # 2. Ingest
    print("\n--- Ingestion ---")
    total_new = 0

    for feed in RSS_FEEDS:
        n = ingest_rss(feed["url"], feed["value_chain_layer"], DB_PATH)
        print(f"  RSS [{feed['value_chain_layer']}]: {n} new documents")
        total_new += n

    n = ingest_arxiv(ARXIV_CATEGORIES, ARXIV_MAX_RESULTS, DB_PATH)
    print(f"  arXiv: {n} new documents")
    total_new += n

    n = ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=DB_PATH)
    print(f"  EDGAR: {n} new documents")
    total_new += n

    print(f"\nTotal new documents: {total_new}")

    # 3. Decide whether to score
    unscored = get_unscored_documents(DB_PATH)
    if not unscored and not args.force:
        print("No unscored documents — nothing to do. Use --force to override.")
        return

    if not unscored and args.force:
        # Re-score all documents from last 30 days
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM documents WHERE ingested_at > datetime('now', '-30 days') ORDER BY ingested_at"
        ).fetchall()
        conn.close()
        unscored = [dict(r) for r in rows]
        print(f"Force mode: re-scoring {len(unscored)} documents from last 30 days")

    # 4. Score
    print(f"\n--- Scoring {len(unscored)} documents via Gemini ---")
    prev_weights = get_prev_weights(DB_PATH)
    signal = score_documents(docs=unscored, db_path=DB_PATH, prev_weights=prev_weights)

    # 5. Persist
    doc_ids = [d["id"] for d in unscored]
    mark_scored(DB_PATH, doc_ids)
    insert_signal(DB_PATH, signal)
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | confidence={signal['signal_confidence']:.2f}")
    print(f"  {signal['thesis_update']}")

    # 6. Export
    if args.dry_run:
        print("\n[DRY-RUN] Skipping JSON export")
    else:
        print("\n--- Exporting ---")
        export_signal(signal)

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify dry-run works (integration test)**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
python main.py --dry-run
```

Expected output structure:
```
[2026-04-20T...] AI Signal Engine starting...

--- Ingestion ---
  RSS [platform]: N new documents
  arXiv: N new documents
  EDGAR: N new documents

Total new documents: N
--- Scoring N documents via Gemini ---
  p=0.XX | regime=... | confidence=0.XX
  ...

[DRY-RUN] Skipping JSON export

Done.
```

Note: Requires `GEMINI_API_KEY` in env. If scoring fails due to missing key, ingestion still succeeds — check DB with:
```bash
sqlite3 signals.db "SELECT count(*) FROM documents;"
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: orchestrator — ingest, score, export pipeline"
```

---

## Task 10: Wire `strategy.py` to `stock_signals.json`

**Files:**
- Modify: `ai-portfolio-backtest/strategy.py`

The spec requires `compute_weights()` to blend Gemini conviction (30%) with momentum scores (70%) when `stock_signals.json` is present. Fallback to pure momentum if absent.

- [ ] **Step 1: Read the current `compute_weights` signature**

```bash
grep -n "def compute_weights\|stock_signals\|conviction" /Users/div-nar/sideproj/ai-portfolio-backtest/strategy.py | head -30
```

- [ ] **Step 2: Add `_load_stock_signals` helper to `strategy.py`**

Find the `_load_p_from_file` function (it loads `p_estimate.json`) and add immediately after it:

```python
def _load_stock_signals() -> dict:
    """Load Gemini conviction scores from stock_signals.json. Returns {} if absent."""
    import json as _json
    sig_file = Path(__file__).parent / "data" / "stock_signals.json"
    if not sig_file.exists():
        return {}
    try:
        with open(sig_file) as f:
            data = _json.load(f)
        return data.get("conviction", {})
    except Exception:
        return {}
```

- [ ] **Step 3: Blend conviction into `compute_weights`**

Inside `compute_weights`, find the section that returns the final `scores` dict (after momentum calculation). Before returning, add:

```python
# Blend with Gemini conviction if available (30% conviction, 70% momentum)
conviction = _load_stock_signals()
if conviction:
    for ticker in scores:
        gemini_score = conviction.get(ticker, 0.0)
        scores[ticker] = 0.7 * scores[ticker] + 0.3 * gemini_score
```

- [ ] **Step 4: Verify strategy still runs**

```bash
cd /Users/div-nar/sideproj/ai-portfolio-backtest
python -c "from strategy import get_params; p = get_params(); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/div-nar/sideproj/ai-portfolio-backtest
git add strategy.py
git commit -m "feat: blend Gemini conviction into momentum scores (30/70 split)"
```

---

## Task 11: Full run + end-to-end verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
pytest tests/ -v
```

Expected: All tests PASS (no failures, no errors).

- [ ] **Step 2: Run full pipeline**

```bash
cd /Users/div-nar/sideproj/ai-signal-engine
export GEMINI_API_KEY=<your-key>
python main.py
```

Expected: Ingestion + scoring + export completes. Three JSON files written to `ai-portfolio-backtest/data/`.

- [ ] **Step 3: Verify output files**

```bash
cat /Users/div-nar/sideproj/ai-portfolio-backtest/data/p_estimate.json
cat /Users/div-nar/sideproj/ai-portfolio-backtest/data/stock_signals.json | python -m json.tool | head -30
cat /Users/div-nar/sideproj/ai-portfolio-backtest/data/market_regime.json
```

Expected: Valid JSON with `generated_at` timestamp, `p` between 0 and 1, non-empty `conviction` dict.

- [ ] **Step 4: Verify strategy.py picks up signals**

```bash
cd /Users/div-nar/sideproj/ai-portfolio-backtest
python execute.py --paper --dry-run
```

Expected: Weights show influence of Gemini conviction (NVDA, ASML, TSM likely near top).

- [ ] **Step 5: Check SQLite DB**

```bash
sqlite3 /Users/div-nar/sideproj/ai-signal-engine/signals.db \
  "SELECT id, source, title, scored FROM documents ORDER BY ingested_at DESC LIMIT 10;"
sqlite3 /Users/div-nar/sideproj/ai-signal-engine/signals.db \
  "SELECT id, p_final, market_regime, signal_confidence, computed_at FROM signals ORDER BY computed_at DESC LIMIT 3;"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Covered By |
|---|---|
| SQLite `documents` table with all columns | Task 2 |
| SQLite `scores` table | Task 2 (schema), not separately populated (Gemini scores whole batch) |
| SQLite `signals` table with all columns | Task 2 |
| SemiAnalysis RSS ingestion | Task 4 |
| arXiv cs.AI/cs.LG/cs.AR ingestion | Task 5 |
| EDGAR 8-K ingestion for MSFT/AMZN/GOOGL/META/NVDA | Task 6 |
| Deduplicate by URL | Tasks 4, 5, 6 (via UNIQUE constraint) |
| Tag documents with value_chain_layer | Tasks 4, 5, 6 |
| Single Gemini 2.5 Pro call | Task 7 |
| Structured JSON output schema | Task 7 |
| Max 10% per stock guardrail | Task 7 |
| Min 2% hedge sector guardrail | System prompt |
| Max turnover vs prev run | Task 7 (`apply_guardrails`) |
| Reject if weights don't sum to 1.0 ± 0.01 | Task 7 |
| `p_estimate.json` export | Task 8 |
| `stock_signals.json` export | Task 8 |
| `market_regime.json` export | Task 8 |
| strategy.py blends conviction (30%) + momentum (70%) | Task 10 |
| Backwards-compatible fallback | Task 10 (returns {} if absent) |
| `--force` flag for on-demand runs | Task 9 |
| `--dry-run` flag | Task 9 |
| 6-layer value chain context in Gemini prompt | Task 7 (`build_signal_context`) |

### Notes
- The `scores` table is defined in schema but not populated per-document (Gemini scores all docs in one call). The table exists for future per-document scoring if needed. The `signals` table captures the aggregated run output.
- EDGAR email in `User-Agent` header is hardcoded to `divith@dognosis.tech` per SEC requirements.
- `GEMINI_MODEL` is set to `gemini-2.5-pro-preview-05-06` in config — update if model ID changes.
