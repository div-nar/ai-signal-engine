from datetime import datetime, timezone

from macro.supply_chain import fetch_supply_chain_signal
from macro.cross_sector import fetch_cross_sector_signal
from macro.composite import load_composite_modifier

_COMPOSITE_CACHE_PATH = str(__import__("pathlib").Path(__file__).parent.parent / "data" / "composite_modifier_cache.json")


def compute_macro_signal(cache_path: str = _COMPOSITE_CACHE_PATH) -> dict:
    """Combine supply chain, cross-sector, and composite stress modifier into a MacroSignal."""
    supply = fetch_supply_chain_signal()
    cross = fetch_cross_sector_signal()

    # ── Base regime (priority-ordered) ────────────────────────────────────────
    if cross["credit_stress"]:
        regime = "credit_stress"
        base_exposure = 0.20
        confidence = 0.90
        notes = (
            f"Credit stress: VIX {cross['vix_level']:.1f} > 25, HYG falling. "
            "Defensive positioning — minimal net long exposure."
        )
    elif supply["pmi_trend"] == "contracting" and cross["copper_infra_lead"] < -0.5:
        regime = "balanced"
        base_exposure = 0.65
        confidence = 0.70
        notes = (
            f"PMI contracting ({supply['pmi']:.1f}) and copper weak "
            f"({cross['copper_infra_lead']:.2f}σ). Cautious positioning."
        )
    else:
        regime = "compute_constrained"
        base_exposure = 0.80
        confidence = 0.85
        notes = (
            f"PMI {supply['pmi']:.1f} ({supply['pmi_trend']}), credit clean, "
            "compute thesis intact. Full net long exposure."
        )

    # ── Composite stress modifier (exempt for credit_stress) ──────────────────
    if regime == "credit_stress":
        modifier = 0.0
        modifier_info = {"stress_score": 0.0, "modifier": 0.0, "cache_age_days": None}
    else:
        modifier, modifier_info = load_composite_modifier(cache_path, supply, cross)
        if modifier_info.get("cache_age_days") is not None:
            notes += (
                f" Composite stress score {modifier_info['stress_score']:.2f} "
                f"adding {modifier:.3f} drag to net exposure."
            )

    net_exposure_target = round(max(0.15, base_exposure + modifier), 4)

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_confidence": confidence,
        "net_exposure_target": net_exposure_target,
        "supply_chain": supply,
        "cross_sector": cross,
        "notes": notes,
        "composite_modifier": modifier_info,
    }
