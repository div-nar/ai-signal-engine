# ai-signal-engine/ingestion/rss.py
import re
from html.parser import HTMLParser
import feedparser
from db import insert_document, DEFAULT_DB


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
