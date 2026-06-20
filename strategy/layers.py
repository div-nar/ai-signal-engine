"""Five-layer AI value-chain taxonomy and baseline allocation.

The "cake", physical -> value-capture:
  1 power           electrons: grid, generation, electrical gear
  2 fabrication     making silicon: foundry, semicap, EDA, materials
  3 compute         accelerators & memory
  4 infrastructure  datacenters, REITs, cooling, interconnect
  5 platform        hyperscalers & software (QQQ's core)

Tickers outside the AI-infra thesis (healthcare, financials, staples, energy
majors, defense) are intentionally unmapped -> ineligible for the portfolio.
"""

LAYERS = ["power", "fabrication", "compute", "infrastructure", "platform"]

LAYER_MAP = {
    # power & energy
    "VST": "power", "CEG": "power", "NRG": "power", "NEE": "power",
    "ETN": "power", "PWR": "power", "EIX": "power", "AES": "power",
    "GEV": "power", "GE": "power",
    # fabrication & materials
    "TSM": "fabrication", "ASML": "fabrication", "AMAT": "fabrication",
    "LRCX": "fabrication", "KLAC": "fabrication", "SNPS": "fabrication",
    "CDNS": "fabrication", "ON": "fabrication", "TOELY": "fabrication",
    "SHECY": "fabrication", "LIN": "fabrication",
    # compute & silicon
    "NVDA": "compute", "AMD": "compute", "AVGO": "compute", "MU": "compute",
    "MRVL": "compute", "QCOM": "compute", "TXN": "compute", "ADI": "compute",
    "ASX": "compute", "ARM": "compute",
    # infrastructure & networking
    "VRT": "infrastructure", "EQIX": "infrastructure", "DLR": "infrastructure",
    "IRM": "infrastructure", "AMT": "infrastructure", "CSCO": "infrastructure",
    "FCX": "infrastructure",
    # platform & application
    "MSFT": "platform", "GOOGL": "platform", "AMZN": "platform",
    "META": "platform", "ORCL": "platform", "PLTR": "platform",
    "NOW": "platform", "CRWD": "platform", "DDOG": "platform",
    "NFLX": "platform", "CRM": "platform", "ADBE": "platform",
    "INTU": "platform", "IBM": "platform", "SAP": "platform",
    "AAPL": "platform", "BIDU": "platform", "BABA": "platform",
    "TCEHY": "platform",
}

# Structural thesis tilt: overweight the layers QQQ underweights (1-3 + infra),
# underweight platform (QQQ's concentrated core). Calibration dials.
BASELINE_BUDGETS = {
    "power": 0.20,
    "fabrication": 0.20,
    "compute": 0.25,
    "infrastructure": 0.15,
    "platform": 0.20,
}


def layer_of(ticker: str) -> str | None:
    """Return the layer for a ticker, or None if it is outside the thesis universe."""
    return LAYER_MAP.get(ticker)


def tickers_in_layer(layer: str) -> list[str]:
    """Return all tickers assigned to a layer."""
    return [t for t, lyr in LAYER_MAP.items() if lyr == layer]
