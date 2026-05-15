from datetime import datetime, timezone

from macro.supply_chain import fetch_supply_chain_signal
from macro.cross_sector import fetch_cross_sector_signal


def compute_macro_signal() -> dict:
    """Combine supply chain and cross-sector signals into a single MacroSignal dict."""
    supply = fetch_supply_chain_signal()
    cross = fetch_cross_sector_signal()

    # Priority-ordered regime rules
    if cross["credit_stress"]:
        regime = "credit_stress"
        net_exposure_target = 0.20
        confidence = 0.90
        notes = (
            f"Credit stress: VIX {cross['vix_level']:.1f} > 25, HYG falling. "
            "Defensive positioning — minimal net long exposure."
        )
    elif supply["shipping_pressure"] > 0.65:
        regime = "shipping_bottleneck"
        net_exposure_target = 0.55
        confidence = 0.75
        notes = (
            f"Shipping pressure {supply['shipping_pressure']:.2f} elevated. "
            "Reduce net exposure, rotate toward supply bottleneck names."
        )
    elif supply["pmi_trend"] == "contracting" and cross["copper_infra_lead"] < -0.5:
        regime = "balanced"
        net_exposure_target = 0.65
        confidence = 0.70
        notes = (
            f"PMI contracting ({supply['pmi']:.1f}) and copper weak "
            f"({cross['copper_infra_lead']:.2f}σ). Cautious positioning."
        )
    else:
        regime = "compute_constrained"
        net_exposure_target = 0.80
        confidence = 0.85
        notes = (
            f"PMI {supply['pmi']:.1f} ({supply['pmi_trend']}), credit clean, "
            "compute thesis intact. Full net long exposure."
        )

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "regime_confidence": confidence,
        "net_exposure_target": net_exposure_target,
        "supply_chain": supply,
        "cross_sector": cross,
        "notes": notes,
    }
