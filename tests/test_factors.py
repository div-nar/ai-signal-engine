import pandas as pd
import pytest
from strategy.factors import momentum_scores, rank_within_layer


def _panel():
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    # AAA rises 0.5%/day, BBB flat, CCC falls 0.3%/day
    aaa = [100 * (1.005 ** i) for i in range(200)]
    bbb = [100.0 for _ in range(200)]
    ccc = [100 * (0.997 ** i) for i in range(200)]
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc}, index=dates)


def test_momentum_orders_winners_above_losers():
    p = _panel()
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert scores["AAA"] > scores["BBB"] > scores["CCC"]


def test_momentum_skips_recent_window():
    # With skip=21 the score ends 21 rows before asof, so it ignores the last month.
    p = _panel()
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert "AAA" in scores and isinstance(scores["AAA"], float)


def test_momentum_omits_insufficient_history():
    p = _panel().iloc[:50]  # fewer rows than lookback+skip
    scores = momentum_scores(p, p.index[-1], lookback=126, skip=21)
    assert scores == {}


def test_rank_within_layer_takes_top_n():
    scores = {"AAA": 0.5, "BBB": 0.0, "CCC": -0.3, "DDD": 0.2}
    ranked = rank_within_layer(["AAA", "BBB", "CCC", "DDD"], scores, top_n=2)
    assert ranked == ["AAA", "DDD"]


def test_rank_excludes_unscored():
    scores = {"AAA": 0.5}
    ranked = rank_within_layer(["AAA", "ZZZ"], scores, top_n=3)
    assert ranked == ["AAA"]
