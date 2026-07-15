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


ADJUSTMENT_MIN = 0.5
ADJUSTMENT_MAX = 1.5


def apply_name_adjustments(scores: dict[str, float], adjustments: dict[str, float],
                           clamp: bool = True) -> dict[str, float]:
    """Apply LLM name emphasis to factor scores.

    Semantics: an adjustment of exactly 0 vetoes the name; anything else is a
    conviction multiplier — clamped to [0.5, 1.5] in guardrailed mode, taken
    as-is (any positive value) in full-autonomy mode — where >1 always
    *improves* the score and <1 always worsens it: for negative momentum that
    means dividing, so a boost can never push a name further down the rank.
    Tickers absent from `adjustments` pass through unchanged; adjustments for
    unscored tickers are ignored.
    """
    out = {}
    for t, score in scores.items():
        raw = adjustments.get(t)
        if raw is None:
            out[t] = score
            continue
        if raw == 0:
            continue  # veto
        mult = float(raw)
        if clamp:
            mult = min(max(mult, ADJUSTMENT_MIN), ADJUSTMENT_MAX)
        elif mult < 0:
            continue  # negative multiplier is nonsense; treat as veto
        out[t] = score * mult if score >= 0 else score / mult
    return out
