# ai-signal-engine/ingestion/rss.py
import re
from html.parser import HTMLParser
import feedparser
import httpx
from db import insert_document, DEFAULT_DB

_FETCH_TIMEOUT_S = 15


def _fetch_feed(feed_url: str):
    """Fetch feed bytes with a timeout, then hand off to feedparser.

    feedparser.parse(url) fetches the URL itself with NO timeout — a single
    unresponsive host (seen hanging a scheduled trade run for 45+ minutes)
    blocks forever. Fetching with httpx first bounds that."""
    resp = httpx.get(feed_url, timeout=_FETCH_TIMEOUT_S, follow_redirects=True)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


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
    feed = _fetch_feed(feed_url)
    entries = []
    for entry in feed.get("entries", []):
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
    chroma_client=None,
) -> int:
    """Fetch RSS entries and insert new ones into DB. Returns count of new docs.

    A single unresponsive/erroring feed is non-fatal — logs a warning and
    contributes 0 new docs rather than blocking the rest of ingestion (and the
    trading run behind it)."""
    try:
        feed = _fetch_feed(feed_url)
    except Exception as e:
        print(f"  WARNING: RSS fetch failed for {feed_url} — skipping ({e})")
        return 0
    raw_entries = feed.entries if hasattr(feed, "entries") else feed.get("entries", [])
    count = 0
    for entry in raw_entries:
        if entry.get("content"):
            raw_content = entry["content"][0]["value"]
        else:
            raw_content = entry.get("summary", "")
        content = _strip_html(raw_content)

        title = entry.get("title", "")
        url = entry.get("link", "") or entry.get("url", "")
        published = entry.get("published", None)
        summary = entry.get("summary", "")

        result = insert_document(
            db_path=db_path,
            source="rss",
            title=title,
            url=url,
            published_at=published,
            content=content,
            value_chain_layer=value_chain_layer,
        )
        if result is not None:
            count += 1
            if chroma_client is not None:
                from chroma_store import upsert_research_doc
                text = f"{title} {summary}".strip()
                metadata = {
                    "source": "rss",
                    "ticker_mentions": "",
                    "ingested_at": published or "",
                    "value_chain_layer": value_chain_layer,
                }
                upsert_research_doc(chroma_client, str(result), text, metadata)
    return count
