"""Apply the LLM's layer tilt to the baseline budgets, under hard guardrails."""

LAYER_FLOOR = 0.08
LAYER_CEILING = 0.35
_TILT_SUM_TOL = 1e-6


def apply_layer_tilt(baseline: dict[str, float], tilt: dict[str, float]) -> dict[str, float]:
    """Return baseline + tilt, clamped to [floor, ceiling] and renormalized to 1.0.

    `tilt` must sum to ~0 (reallocation, not leverage). After clamping, the
    result is renormalized so the layer budgets sum to exactly 1.0; a second
    clamp pass keeps the ceiling honest after renormalization.
    """
    if abs(sum(tilt.values())) > _TILT_SUM_TOL:
        raise ValueError(f"tilt must sum to ~0, got {sum(tilt.values()):.6f}")

    budgets = {k: baseline[k] + tilt.get(k, 0.0) for k in baseline}

    # Clamp, then renormalize, repeating a few times so both bounds hold while
    # the total stays 1.0.
    for _ in range(50):
        budgets = {k: min(max(v, LAYER_FLOOR), LAYER_CEILING) for k, v in budgets.items()}
        total = sum(budgets.values())
        budgets = {k: v / total for k, v in budgets.items()}
        if all(LAYER_FLOOR - 1e-9 <= v <= LAYER_CEILING + 1e-9 for v in budgets.values()):
            break
    return budgets
