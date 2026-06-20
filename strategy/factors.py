"""Mechanical within-layer ranking factors.

Phase 1: price momentum (no new data dependency — uses the price panel we
already pull). Phase 2 (separate plan) adds fundamental factors behind the
same dict-returning contract.
"""
import pandas as pd


def momentum_scores(prices: pd.DataFrame, asof, lookback: int = 126, skip: int = 21) -> dict[str, float]:
    """Total return from (lookback+skip) rows ago to (skip) rows ago, as of `asof`.

    Skipping the most recent `skip` rows (~1 month) avoids short-term reversal,
    the classic 12-1 momentum construction. Columns without enough history are
    omitted from the result.
    """
    window = prices.loc[:asof]
    needed = lookback + skip + 1
    if len(window) < needed:
        return {}
    start = window.iloc[-(lookback + skip + 1)]
    end = window.iloc[-(skip + 1)]
    ratio = end / start - 1.0
    return {t: float(v) for t, v in ratio.items() if pd.notna(v)}


def rank_within_layer(layer_tickers: list[str], factor_scores: dict[str, float], top_n: int) -> list[str]:
    """Return the top_n tickers in `layer_tickers` by score (unscored excluded)."""
    scored = [t for t in layer_tickers if t in factor_scores]
    scored.sort(key=lambda t: factor_scores[t], reverse=True)
    return scored[:top_n]
