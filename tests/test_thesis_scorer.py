import json
import pytest
from strategy.budgets import LAYER_FLOOR, LAYER_CEILING
from scoring.thesis_scorer import score_layer_thesis


class FakeClient:
    def __init__(self, response, fail_times=0):
        self.response = response
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return self.response


def _resp(tilt, regime="compute_constrained", shift=True):
    return json.dumps({
        "layer_tilt": tilt,
        "market_regime": regime,
        "regime_shift": shift,
        "signal_confidence": 0.8,
        "thesis_update": "memory + power are the binding constraints",
    })


def test_budgets_within_bounds_and_sum_one():
    client = FakeClient(_resp({"compute": 0.30, "platform": -0.30}))
    out = score_layer_thesis([{"id": 1, "content": "x", "title": "t", "source": "rss"}],
                             client=client, autonomy="guardrailed")
    b = out["layer_budgets"]
    assert sum(b.values()) == pytest.approx(1.0)
    assert all(LAYER_FLOOR - 1e-9 <= v <= LAYER_CEILING + 1e-9 for v in b.values())
    assert out["market_regime"] == "compute_constrained"
    assert out["regime_shift"] is True


def test_retries_then_succeeds():
    client = FakeClient(_resp({"power": 0.05, "platform": -0.05}), fail_times=2)
    out = score_layer_thesis([], client=client)
    assert client.calls == 3
    assert "layer_budgets" in out


def test_raises_after_exhausting_retries():
    class AlwaysFails:
        def generate(self, prompt):
            raise RuntimeError("down")
    with pytest.raises(RuntimeError):
        score_layer_thesis([], client=AlwaysFails())
