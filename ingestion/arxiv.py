# ai-signal-engine/ingestion/arxiv.py
import arxiv
from db import insert_document, DEFAULT_DB

# Map arXiv categories to value chain layers
_CATEGORY_LAYER = {
    "cs.AR": "compute",    # Hardware architecture
    "cs.AI": "platform",   # AI methods → platform layer
    "cs.LG": "platform",   # ML → platform layer
}
_DEFAULT_LAYER = "application"


def fetch_arxiv_papers(categories: list[str], max_results: int) -> list[dict]:
    """Fetch recent papers from arXiv for given categories."""
    client = arxiv.Client()

    category_query = " OR ".join(f"cat:{c}" for c in categories)
    search = arxiv.Search(
        query=category_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
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
