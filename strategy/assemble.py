"""Assemble final portfolio weights from layer budgets + within-layer factor ranks."""
from strategy.factors import rank_within_layer


def _cap_and_normalize(weights: dict[str, float], name_cap: float) -> dict[str, float]:
    """Cap each name at name_cap and renormalize to sum 1.0 (iterative)."""
    w = dict(weights)
    for _ in range(50):
        total = sum(w.values())
        if total <= 0:
            return w
        w = {k: v / total for k, v in w.items()}
        if all(v <= name_cap + 1e-9 for v in w.values()):
            break
        w = {k: min(v, name_cap) for k, v in w.items()}
    return w


def assemble_portfolio(
    budgets: dict[str, float],
    factor_scores: dict[str, float],
    layer_map: dict[str, str],
    top_n: int = 3,
    name_cap: float = 0.12,
) -> dict[str, float]:
    """Fully-invested weights: top_n per layer, budget split by factor score, name-capped."""
    weights: dict[str, float] = {}
    for layer, budget in budgets.items():
        if budget <= 0:
            continue
        layer_tickers = [t for t, lyr in layer_map.items() if lyr == layer]
        ranked = rank_within_layer(layer_tickers, factor_scores, top_n)
        if not ranked:
            continue
        shifted = {t: max(factor_scores[t], 0.0) for t in ranked}
        s = sum(shifted.values())
        if s <= 0:
            alloc = {t: budget / len(ranked) for t in ranked}
        else:
            alloc = {t: budget * shifted[t] / s for t in ranked}
        for t, a in alloc.items():
            weights[t] = weights.get(t, 0.0) + a

    if not weights:
        return {}
    return _cap_and_normalize(weights, name_cap)
