"""Agentic retrieval loop + expanded LLM decision surface (clamps, logging)."""
import json
import pytest
from scoring.thesis_scorer import (
    score_layer_thesis, sanitize_top_n, sanitize_name_adjustments,
    sanitize_cash_buffer, sanitize_urgency, MAX_SEARCH_ROUNDS,
)


def _final(**overrides):
    body = {
        "layer_tilt": {"compute": 0.10, "platform": -0.10},
        "layer_top_n": {"compute": 2, "power": 4},
        "name_adjustments": {"NVDA": 1.4, "MU": 0.0},
        "cash_buffer": 0.1,
        "rebalance_urgency": "normal",
        "market_regime": "compute_constrained",
        "regime_shift": False,
        "signal_confidence": 0.8,
        "thesis_update": "compute bound",
    }
    body.update(overrides)
    return json.dumps(body)


_SEARCH = json.dumps({"action": "search", "queries": ["HBM supply", "datacenter power"]})


class ScriptedClient:
    """Returns queued responses in order; repeats the last one when exhausted."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def test_search_round_feeds_docs_back_and_logs():
    client = ScriptedClient([_SEARCH, _final()])
    retrieved = [{"id": "r1", "title": "HBM shortage", "content": "hbm", "source": "rss"}]
    queries_seen = []

    def retriever(q):
        queries_seen.append(q)
        return retrieved

    out = score_layer_thesis([{"id": "seed", "title": "s", "content": "c", "source": "rss"}],
                             client=client, retriever=retriever)
    assert queries_seen == ["HBM supply", "datacenter power"]
    # second prompt contains the retrieved doc (deduped per query round)
    assert "HBM shortage" in client.prompts[1]
    assert len(out["retrieval_log"]) == 2
    assert out["retrieval_log"][0]["query"] == "HBM supply"
    assert out["market_regime"] == "compute_constrained"


def test_search_without_retriever_still_terminates():
    client = ScriptedClient([_SEARCH, _final()])
    out = score_layer_thesis([], client=client, retriever=None)
    assert out["layer_budgets"]
    assert out["retrieval_log"][0]["note"] == "no retriever available"


def test_search_budget_exhaustion_forces_final():
    # Model keeps asking to search; after MAX_SEARCH_ROUNDS it must be forced.
    client = ScriptedClient([_SEARCH] * (MAX_SEARCH_ROUNDS + 1) + [_final(), _final()])
    out = score_layer_thesis([], client=client, retriever=lambda q: [])
    assert out["layer_budgets"]
    assert "Search budget exhausted" in client.prompts[-1]


def test_retriever_dedupes_already_seen_docs():
    client = ScriptedClient([_SEARCH, _final()])
    seed = {"id": "dup", "title": "seen", "content": "x", "source": "rss"}
    out = score_layer_thesis([seed], client=client,
                             retriever=lambda q: [dict(seed)])
    assert all(entry["hits"] == 0 for entry in out["retrieval_log"])


def test_expanded_fields_clamped_and_returned():
    client = ScriptedClient([_final(cash_buffer=0.9, layer_top_n={"compute": 9, "bogus": 3},
                                    name_adjustments={"NVDA": 5.0, "FAKE": 1.2},
                                    rebalance_urgency="panic")])
    out = score_layer_thesis([], client=client)
    assert out["cash_buffer"] == pytest.approx(0.30)          # clamped
    assert out["layer_top_n"] == {"compute": 9}               # unknown layer dropped (clamp at use)
    assert "FAKE" not in out["name_adjustments"]              # outside universe
    assert out["name_adjustments"]["NVDA"] == 5.0             # clamped at application time
    assert out["rebalance_urgency"] == "normal"               # unknown -> normal


def test_sanitizers_direct():
    assert sanitize_top_n({"power": "2", "nope": 3, "compute": None}) == {"power": 2}
    adj = sanitize_name_adjustments({"nvda": "1.3", "ZZZ": 1.0, "MU": "bad"})
    assert adj == {"NVDA": 1.3}
    assert sanitize_cash_buffer(-1) == 0.0
    assert sanitize_cash_buffer("nope") == 0.0
    assert sanitize_urgency("hold") == "hold"
    assert sanitize_urgency(None) == "normal"


def test_plain_final_answer_has_empty_retrieval_log():
    client = ScriptedClient([_final()])
    out = score_layer_thesis([], client=client, retriever=lambda q: [])
    assert out["retrieval_log"] == []
