"""Compose the mechanical target portfolio from layer budgets + live prices."""
from strategy.factors import momentum_scores, apply_name_adjustments
from strategy.assemble import assemble_portfolio


def build_target_portfolio(layer_budgets: dict, price_history, layer_map: dict,
                           asof=None, top_n: int | dict = 3, name_cap: float = 0.12,
                           lookback: int = 126, skip: int = 21,
                           name_adjustments: dict | None = None) -> dict[str, float]:
    """Momentum-rank within each layer and assemble fully-invested target weights.

    `top_n` may be a per-layer dict (the LLM concentration dial, clamped in the
    assembler) and `name_adjustments` a bounded emphasis/veto map (clamped in
    apply_name_adjustments) — both optional; omitting them gives the pure
    mechanical portfolio.
    """
    if price_history is None or price_history.empty:
        return {}
    if asof is None:
        asof = price_history.index[-1]
    scores = momentum_scores(price_history, asof, lookback=lookback, skip=skip)
    if not scores:
        return {}
    if name_adjustments:
        scores = apply_name_adjustments(scores, name_adjustments)
        if not scores:
            return {}
    return assemble_portfolio(layer_budgets, scores, layer_map,
                              top_n=top_n, name_cap=name_cap)
