import numpy as np
import pandas as pd
import pytest
from backtest.engine import equal_weight_scores, run_variant, metrics

LAYER_MAP = {"A": "power", "B": "power", "C": "compute", "D": "compute"}
BUDGETS = {"power": 0.5, "compute": 0.5,
           "fabrication": 0.0, "infrastructure": 0.0, "platform": 0.0}


def _panel(n=200):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "A": [100 * 1.004 ** i for i in range(n)],
        "B": [100 * 1.001 ** i for i in range(n)],
        "C": [100 * 1.003 ** i for i in range(n)],
        "D": [100 * 0.999 ** i for i in range(n)],
    }, index=idx)


def test_equal_weight_scores_all_zero():
    s = equal_weight_scores(["A", "B"])
    assert s == {"A": 0.0, "B": 0.0}


def test_run_variant_returns_growing_equity_in_uptrend():
    eq = run_variant(_panel(), LAYER_MAP, BUDGETS, variant="baseline")
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] > 1.0


def test_momentum_beats_baseline_when_winners_persist():
    p = _panel()
    # name_cap=0.5 keeps both variants fully invested on this 4-name test universe
    # (the 0.12 production cap is infeasible with so few names); short lookback so
    # momentum is active across the 200-row panel.
    base = run_variant(p, LAYER_MAP, BUDGETS, variant="baseline", name_cap=0.5)
    mom = run_variant(p, LAYER_MAP, BUDGETS, variant="momentum",
                      lookback=20, skip=5, name_cap=0.5)
    # momentum concentrates in the persistent winners (A, C) -> ends >= baseline
    assert mom.iloc[-1] >= base.iloc[-1]


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        run_variant(_panel(), LAYER_MAP, BUDGETS, variant="bogus")


def test_metrics_shape_and_drawdown_sign():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    m = metrics(eq)
    assert set(m) == {"total_return_pct", "sharpe", "max_drawdown_pct"}
    assert m["max_drawdown_pct"] <= 0
    assert m["total_return_pct"] == pytest.approx(20.0)
