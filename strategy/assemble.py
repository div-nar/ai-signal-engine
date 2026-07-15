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


TOP_N_MIN = 2
TOP_N_MAX = 4
TOP_N_DEFAULT = 3


def _layer_top_n(top_n, layer: str, clamp: bool = True) -> int:
    """Resolve top_n for a layer. A plain int applies everywhere, unclamped
    (programmatic callers know what they want). A dict is the LLM's per-layer
    concentration dial, clamped to [2, 4] in guardrailed mode; full-autonomy
    (`clamp=False`) allows any n >= 1."""
    if not isinstance(top_n, dict):
        return int(top_n)
    try:
        n = int(top_n.get(layer, TOP_N_DEFAULT))
    except (TypeError, ValueError):
        return TOP_N_DEFAULT
    if not clamp:
        return max(n, 1)
    return min(max(n, TOP_N_MIN), TOP_N_MAX)


def assemble_portfolio(
    budgets: dict[str, float],
    factor_scores: dict[str, float],
    layer_map: dict[str, str],
    top_n: int | dict = TOP_N_DEFAULT,
    name_cap: float = 0.12,
    clamp_dial: bool = True,
) -> dict[str, float]:
    """Fully-invested weights: top_n per layer, budget split by factor score, name-capped."""
    weights: dict[str, float] = {}
    for layer, budget in budgets.items():
        if budget <= 0:
            continue
        layer_tickers = [t for t, lyr in layer_map.items() if lyr == layer]
        ranked = rank_within_layer(layer_tickers, factor_scores,
                                   _layer_top_n(top_n, layer, clamp=clamp_dial))
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
