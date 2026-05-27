import httpx
from db import insert_document, DEFAULT_DB

_API_URL = "https://huggingface.co/api/daily_papers"
_PAPER_BASE = "https://huggingface.co/papers"

# Map AI keywords → value chain layer (checked against paper.ai_keywords)
_KEYWORD_LAYER: dict[str, str] = {
    "hardware": "compute",
    "gpu": "compute",
    "chip": "compute",
    "semiconductor": "compute",
    "asic": "compute",
    "fpga": "compute",
    "memory": "compute",
    "processor": "compute",
    "inference": "compute",
    "training": "compute",
    "efficiency": "compute",
    "quantization": "compute",
    "datacenter": "infrastructure",
    "data center": "infrastructure",
    "networking": "infrastructure",
    "distributed": "infrastructure",
    "cluster": "infrastructure",
    "cloud": "platform",
    "foundation model": "platform",
    "large language model": "platform",
    "llm": "platform",
    "multimodal": "platform",
    "transformer": "platform",
    "diffusion": "platform",
    "reinforcement learning": "platform",
    "agent": "application",
    "autonomous": "application",
    "robotics": "application",
    "reasoning": "application",
}
_DEFAULT_LAYER = "platform"


def _classify_layer(keywords: list[str]) -> str:
    lowered = [k.lower() for k in (keywords or [])]
    for keyword, layer in _KEYWORD_LAYER.items():
        if any(keyword in k for k in lowered):
            return layer
    return _DEFAULT_LAYER


def fetch_hf_papers(max_results: int = 50) -> list[dict]:
    resp = httpx.get(_API_URL, timeout=30)
    resp.raise_for_status()
    items = resp.json()[:max_results]

    papers = []
    for item in items:
        p = item.get("paper", {})
        paper_id = p.get("id", "")
        title = p.get("title") or item.get("title", "")
        summary = p.get("summary") or item.get("summary", "")
        published_at = p.get("publishedAt", "")
        ai_keywords = p.get("ai_keywords", [])

        if not paper_id or not title:
            continue

        papers.append({
            "title": title,
            "url": f"{_PAPER_BASE}/{paper_id}",
            "published": published_at,
            "content": summary,
            "value_chain_layer": _classify_layer(ai_keywords),
        })
    return papers


def ingest_hf_papers(
    max_results: int = 50,
    db_path: str = str(DEFAULT_DB),
    chroma_client=None,
) -> int:
    try:
        papers = fetch_hf_papers(max_results)
    except Exception as e:
        print(f"  WARNING: HuggingFace Papers ingestion failed ({e}) — skipping")
        return 0

    count = 0
    for p in papers:
        result = insert_document(
            db_path=db_path,
            source="huggingface",
            title=p["title"],
            url=p["url"],
            published_at=p["published"],
            content=p["content"],
            value_chain_layer=p["value_chain_layer"],
        )
        if result is not None:
            count += 1
            if chroma_client is not None:
                from chroma_store import upsert_research_doc
                text = f"{p['title']} {p.get('content', '')}".strip()
                metadata = {
                    "source": "arxiv",
                    "ticker_mentions": "",
                    "ingested_at": p.get("published", ""),
                    "value_chain_layer": p.get("value_chain_layer", "platform"),
                }
                upsert_research_doc(chroma_client, str(result), text, metadata)
    return count
