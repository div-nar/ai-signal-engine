"""Chroma wiring in the orchestrator: agentic retriever + thesis memory."""
import datetime as dt
import json
from unittest.mock import MagicMock, patch
from db import init_targets_table
from orchestrate import compute_weekly_target


class SearchingThesis:
    """Asks for one search round, then answers."""
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"action": "search", "queries": ["HBM supply"]})
        return json.dumps({"layer_tilt": {"compute": 0.1, "platform": -0.1},
                           "market_regime": "compute_constrained", "regime_shift": False,
                           "signal_confidence": 0.8, "thesis_update": "compute bound"})


class _Bar:
    def __init__(self, ts, close): self.timestamp = ts; self.close = close


class FakeData:
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {s: [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 1e-4) ** i)
                    for i in range(260)] for j, s in enumerate(syms)}
        r = MagicMock(); r.data = data; return r


def test_retriever_and_memory_wired_through_chroma(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    chroma = MagicMock()
    doc = {"id": "d1", "title": "HBM shortage deepens", "content": "hbm", "source": "rss"}
    with patch("chroma_store.query_research_docs", return_value=[doc]) as q_docs, \
         patch("chroma_store.query_signal_records", return_value=[]) as q_sigs, \
         patch("chroma_store.upsert_signal_record", return_value=True) as up_sig:
        out = compute_weekly_target(
            docs=[], db_path=db, thesis_client=SearchingThesis(),
            data_client=FakeData(), chroma_client=chroma,
            now=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
        )
    q_docs.assert_called_once()
    assert q_docs.call_args[0][1] == "HBM supply"
    q_sigs.assert_called_once()                      # memory queried for the prompt
    up_sig.assert_called_once()                      # thesis remembered after persist
    assert up_sig.call_args[0][1] == f"thesis_{out['id']}"
    assert out["retrieval_log"][0]["hits"] == 1


def test_no_chroma_client_means_no_chroma_calls(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    with patch("chroma_store.query_research_docs") as q_docs, \
         patch("chroma_store.upsert_signal_record") as up_sig:
        out = compute_weekly_target(
            docs=[], db_path=db, thesis_client=SearchingThesis(),
            data_client=FakeData(), chroma_client=None,
            now=dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc),
        )
    q_docs.assert_not_called()
    up_sig.assert_not_called()
    assert out["target_weights"]  # engine still produced a book
