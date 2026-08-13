"""_embed uses a local fastembed model with nomic prefix convention."""
import chroma_store


class _FakeEmbedder:
    """Records prefixed inputs; returns a fixed 768-dim vector per text."""
    def __init__(self):
        self.seen = []

    def embed(self, texts):
        for t in texts:
            self.seen.append(t)
            yield [0.01] * 768


def test_embed_returns_768_dim(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    v = chroma_store._embed("HBM supply constraints")
    assert isinstance(v, list) and len(v) == 768


def test_embed_applies_document_prefix_by_default(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    chroma_store._embed("grid power")
    assert fake.seen == ["search_document: grid power"]


def test_embed_applies_query_prefix_for_retrieval_query(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    chroma_store._embed("grid power", task_type="RETRIEVAL_QUERY")
    assert fake.seen == ["search_query: grid power"]


def test_embed_truncates_long_text_to_char_cap(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr(chroma_store, "_get_embedder", lambda: fake)
    long_text = "A" * 50_000  # e.g. a full EDGAR 8-K body
    chroma_store._embed(long_text)
    # prefix + at most MAX_EMBED_CHARS of the text is sent to the model,
    # bounding CPU embed latency (local ONNX chokes on very long sequences).
    sent = fake.seen[0]
    assert sent.startswith("search_document: ")
    assert len(sent) <= len("search_document: ") + chroma_store.MAX_EMBED_CHARS
