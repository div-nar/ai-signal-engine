import pandas as pd
import pytest
from backtest.run_ablation import build_scorecard

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


def test_scorecard_has_all_variants():
    p = _panel()
    bench = p["A"] / p["A"].iloc[0]  # any benchmark series
    sc = build_scorecard(p, bench, LAYER_MAP, BUDGETS, lookback=20, skip=5)
    assert set(sc.index) == {"baseline", "momentum", "QQQ"}
    assert "sharpe" in sc.columns
    assert "total_return_pct" in sc.columns
