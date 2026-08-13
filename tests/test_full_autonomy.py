"""Full-autonomy mode: direct LLM weights, unclamped dials, context in prompt."""
import datetime as dt
import json
import pytest
from unittest.mock import MagicMock
from db import init_targets_table, get_latest_target
from scoring.thesis_scorer import score_layer_thesis, sanitize_target_weights
from strategy.budgets import apply_layer_tilt
from orchestrate import compute_weekly_target


class OneShot:
    def __init__(self, body):
        self.body = json.dumps(body)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.body


def _base(**overrides):
    body = {"layer_tilt": {}, "market_regime": "balanced", "regime_shift": False,
            "signal_confidence": 0.5, "thesis_update": "steady"}
    body.update(overrides)
    return body


def test_direct_weights_validated_normalized_cash_implied():
    # ZZZZ is a valid ticker *format* but not in the supplied tradable set,
    # so it is dropped; "bad" weight and CASH are dropped too.
    client = OneShot(_base(target_weights={"NVDA": 0.4, "VST": 0.3, "ZZZZ": 0.2,
                                           "MU": "bad", "CASH": 0.1}))
    out = score_layer_thesis([], client=client, autonomy="full",
                             valid_symbols={"NVDA", "VST", "MU"})
    w = out["target_weights_direct"]
    assert "ZZZZ" not in w and "MU" not in w and "CASH" not in w
    assert sum(w.values()) == pytest.approx(1.0)
    # raw valid sum was 0.7 -> 0.3 implied cash (no separate cash dial)
    assert out["cash_buffer"] == pytest.approx(0.3)


def test_direct_weights_whole_market_accepts_any_valid_ticker():
    # With no tradable set injected, any well-formed ticker is allowed —
    # whole-market autonomy. Garbage-format keys are still dropped.
    client = OneShot(_base(target_weights={"COST": 0.5, "JPM": 0.3, "not a ticker": 0.2}))
    out = score_layer_thesis([], client=client, autonomy="full")
    w = out["target_weights_direct"]
    assert set(w) == {"COST", "JPM"}          # non-AI names now allowed
    assert sum(w.values()) == pytest.approx(1.0)


def test_guardrailed_mode_ignores_direct_weights():
    client = OneShot(_base(target_weights={"NVDA": 1.0}))
    out = score_layer_thesis([], client=client, autonomy="guardrailed")
    assert out["target_weights_direct"] == {}


def test_unclamped_budgets_can_go_all_in():
    # +0.60 compute breaches the 35% ceiling — allowed in full autonomy.
    tilt = {"compute": 0.60, "power": -0.15, "fabrication": -0.15,
            "infrastructure": -0.15, "platform": -0.15}
    budgets = apply_layer_tilt({"power": 0.20, "fabrication": 0.20, "compute": 0.25,
                                "infrastructure": 0.15, "platform": 0.20},
                               tilt, clamp=False)
    assert budgets["compute"] == pytest.approx(0.85)
    assert budgets["power"] == pytest.approx(0.05)
    assert sum(budgets.values()) == pytest.approx(1.0)


def test_unclamped_cash_buffer():
    client = OneShot(_base(cash_buffer=0.8))
    out = score_layer_thesis([], client=client, autonomy="full")
    assert out["cash_buffer"] == pytest.approx(0.8)


def test_sanitize_target_weights_rejects_garbage():
    assert sanitize_target_weights(None) == ({}, 0.0)
    assert sanitize_target_weights({"lowercase words": 1.0}) == ({}, 0.0)  # bad format
    assert sanitize_target_weights({"NVDA": -0.5}) == ({}, 0.0)            # non-positive
    # a valid-format ticker is kept when no tradable set restricts it
    assert sanitize_target_weights({"JPM": 1.0}) == ({"JPM": 1.0}, 0.0)
    # but dropped when a tradable set is supplied and excludes it
    assert sanitize_target_weights({"JPM": 1.0}, valid_symbols={"NVDA"}) == ({}, 0.0)


class _Bar:
    def __init__(self, ts, close): self.timestamp = ts; self.close = close


class FakeData:
    def get_stock_bars(self, request):
        d0 = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        syms = ["NVDA", "MU", "VST", "CEG", "TSM", "ASML", "VRT", "EQIX", "MSFT", "GOOGL"]
        data = {s: [_Bar(d0 + dt.timedelta(days=i), 100 * (1.002 + j * 1e-4) ** i)
                    for i in range(260)] for j, s in enumerate(syms)}
        r = MagicMock(); r.data = data; return r


_NOW = dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc)


def test_orchestrate_uses_direct_weights_and_persists_source(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    client = OneShot(_base(target_weights={"NVDA": 0.6, "VST": 0.4},
                           rebalance_urgency="urgent"))
    out = compute_weekly_target(docs=[], db_path=db, thesis_client=client,
                                data_client=FakeData(), now=_NOW, autonomy="full")
    assert out["weights_source"] == "llm_direct"
    assert out["target_weights"] == pytest.approx({"NVDA": 0.6, "VST": 0.4})
    got = get_latest_target(db)
    assert got["weights_source"] == "llm_direct"
    assert got["autonomy"] == "full"


def test_orchestrate_full_prompt_includes_book_and_momentum(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    client = OneShot(_base(target_weights={"NVDA": 1.0}))
    compute_weekly_target(docs=[], db_path=db, thesis_client=client,
                          data_client=FakeData(), now=_NOW, autonomy="full")
    prompt = client.prompts[0]
    assert "MOMENTUM RANKS" in prompt
    assert "CURRENT BOOK" in prompt
    # whole-market mode: no fixed executable-universe list is injected
    assert "EXECUTABLE UNIVERSE" not in prompt


def test_orchestrate_dials_fallback_when_no_direct_weights(tmp_path):
    db = str(tmp_path / "t.db")
    init_targets_table(db)
    client = OneShot(_base(layer_tilt={"compute": 0.1, "platform": -0.1}))
    out = compute_weekly_target(docs=[], db_path=db, thesis_client=client,
                                data_client=FakeData(), now=_NOW, autonomy="full")
    assert out["weights_source"] == "dial_pipeline"
    assert out["target_weights"]
    assert sum(out["target_weights"].values()) == pytest.approx(1.0)
