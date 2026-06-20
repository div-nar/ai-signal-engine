"""Narrow extreme-only risk-off switch and turnover-throttling rebalance bands."""
from collections import defaultdict


def risk_off_cash(credit_stress: bool, vix: float, vix_threshold: float = 30.0,
                  buffer: float = 0.30) -> float:
    """Raise a fixed cash buffer only on an extreme objective trigger; else stay invested."""
    if credit_stress and vix > vix_threshold:
        return buffer
    return 0.0


def _layer_totals(weights: dict[str, float], layer_map: dict[str, str]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for t, w in weights.items():
        totals[layer_map.get(t, "_unmapped")] += w
    return totals


def needs_rebalance(current: dict[str, float], target: dict[str, float],
                    layer_map: dict[str, str], layer_band: float = 0.03,
                    name_band: float = 0.03) -> bool:
    """True if any name or any layer aggregate drifts beyond its band."""
    names = set(current) | set(target)
    for t in names:
        if abs(current.get(t, 0.0) - target.get(t, 0.0)) > name_band:
            return True
    cur_l = _layer_totals(current, layer_map)
    tgt_l = _layer_totals(target, layer_map)
    for layer in set(cur_l) | set(tgt_l):
        if abs(cur_l.get(layer, 0.0) - tgt_l.get(layer, 0.0)) > layer_band:
            return True
    return False
