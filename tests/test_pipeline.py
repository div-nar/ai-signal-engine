import pandas as pd
import pytest
from strategy.pipeline import build_target_portfolio

LAYER_MAP = {"A": "power", "B": "power", "C": "compute", "D": "compute"}
BUDGETS = {"power": 0.5, "compute": 0.5,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def _panel(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "A": [100 * 1.004 ** i for i in range(n)],
        "B": [100 * 1.001 ** i for i in range(n)],
        "C": [100 * 1.003 ** i for i in range(n)],
        "D": [100 * 0.999 ** i for i in range(n)],
    }, index=idx)


def test_target_is_fully_invested():
    w = build_target_portfolio(BUDGETS, _panel(), LAYER_MAP,
                               top_n=3, name_cap=0.5, lookback=20, skip=5)
    assert sum(w.values()) == pytest.approx(1.0)
    assert max(w.values()) <= 0.5 + 1e-9


def test_target_favours_momentum_winners():
    w = build_target_portfolio(BUDGETS, _panel(), LAYER_MAP,
                               top_n=1, name_cap=1.0, lookback=20, skip=5)
    # strongest in each layer: A (power), C (compute)
    assert set(w) == {"A", "C"}


def test_empty_when_insufficient_history():
    w = build_target_portfolio(BUDGETS, _panel(n=10), LAYER_MAP,
                               lookback=126, skip=21)
    assert w == {}
