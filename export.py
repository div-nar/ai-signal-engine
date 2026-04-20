# ai-signal-engine/export.py
import json
from datetime import datetime, timezone
from pathlib import Path

from config import BACKTEST_DATA_DIR


def export_signal(signal: dict, output_dir: str = str(BACKTEST_DATA_DIR)) -> None:
    """Write p_estimate.json, stock_signals.json, and market_regime.json."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    conviction = json.loads(signal["stock_conviction"])
    weights = json.loads(signal.get("stock_weights", "{}"))
    reasoning = json.loads(signal["stock_reasoning"])
    signal_breakdown = json.loads(signal["signal_breakdown"])

    # 1. p_estimate.json — consumed by strategy.py
    p_file = output / "p_estimate.json"
    p_file.write_text(json.dumps({
        "p": signal["p_final"],
        "generated_at": now,
    }, indent=2))

    # 2. stock_signals.json — consumed by strategy.py for conviction blending
    stock_signals_file = output / "stock_signals.json"
    stock_signals_file.write_text(json.dumps({
        "generated_at": now,
        "conviction": conviction,
        "weights": weights,
        "reasoning": reasoning,
    }, indent=2))

    # 3. market_regime.json — consumed by layers 3–5
    market_regime_file = output / "market_regime.json"
    market_regime_file.write_text(json.dumps({
        "generated_at": now,
        "market_regime": signal["market_regime"],
        "supply_demand_balance": signal["supply_demand_balance"],
        "sector_tilt": json.loads(signal["sector_tilt"]),
        "signal_confidence": signal["signal_confidence"],
        "thesis_stress": bool(signal["thesis_stress"]),
        "thesis_update": signal["thesis_update"],
        "signal_breakdown": signal_breakdown,
    }, indent=2))

    print(f"Exported signals to {output}/")
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | sources={signal['sources_ingested']}")
