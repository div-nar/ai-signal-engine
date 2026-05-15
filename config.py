# ai-signal-engine/config.py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DB_PATH = str(ROOT / "signals.db")
BACKTEST_DATA_DIR = ROOT.parent / "ai-portfolio-backtest" / "data"

# ── RSS feeds ─────────────────────────────────────────────────────────────────
# Mix of supply-side (semis/infra deep-dives) and demand-side (hyperscaler &
# AI-platform strategy) sources. Layer tags drive how the scorer organises the
# context it shows the LLM (see VALUE_CHAIN_LAYERS).
RSS_FEEDS = [
    # Supply-side: semis / infra deep-dives
    {"url": "https://www.semianalysis.com/feed",        "value_chain_layer": "compute"},
    {"url": "https://www.fabricatedknowledge.com/feed", "value_chain_layer": "compute"},
    # Demand-side: hyperscaler & AI-platform strategy
    {"url": "https://stratechery.com/feed/",            "value_chain_layer": "platform"},
    {"url": "https://www.platformer.news/feed",         "value_chain_layer": "platform"},
    {"url": "https://tanay.substack.com/feed",          "value_chain_layer": "application"},
    {"url": "https://www.latent.space/feed",            "value_chain_layer": "application"},
    {"url": "https://importai.substack.com/feed",       "value_chain_layer": "application"},
]

# ── arXiv ─────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.AR"]
ARXIV_MAX_RESULTS = 50  # per category per run

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
EDGAR_TICKERS = {
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "NVDA": "0001045810",
}
EDGAR_FORM_TYPE = "8-K"
EDGAR_MAX_PER_TICKER = 3

# ── Value chain layer tags ─────────────────────────────────────────────────────
VALUE_CHAIN_LAYERS = ["compute", "power", "infrastructure", "platform", "application", "domain"]

# ── Ticker universe (95 stocks) ────────────────────────────────────────────────
TICKER_UNIVERSE = [
    # Information Technology — compute stack (chips, EDA, foundry, memory, packaging)
    "NVDA", "AMD", "AVGO", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "ARM", "TOELY",
    "SNPS", "CDNS", "MRVL", "ASX", "SHECY", "ON",
    # Note: SK Hynix (HXSCL) and Samsung (SSNNF) are Korean-listed; no reliable US price data.
    # Information Technology — software and cloud
    "MSFT", "ORCL", "IBM", "SAP", "INFY", "CSCO", "PLTR", "CRM", "NOW", "DDOG",
    "CRWD", "ADBE", "INTU", "MU", "QCOM", "TXN", "ADI", "AAPL",
    # Communication Services
    "GOOGL", "META", "NFLX", "TMUS", "BIDU", "BABA", "TCEHY",
    # Consumer Discretionary
    "AMZN", "TSLA", "BKNG",
    # Consumer Staples
    "COST", "WMT",
    # Energy
    "XOM", "CVX",
    # Financials
    "V", "MA", "JPM", "GS", "COIN", "BLK", "SCHW", "ICE", "SPGI", "IBKR",
    # Health Care
    "LLY", "UNH", "ABBV", "ISRG", "TMO", "AMGN", "GILD", "VRTX", "REGN", "MRK", "JNJ",
    # Industrials
    "GE", "GEV", "HON", "CAT", "DE", "RTX", "AXON", "VRT", "LMT", "NOC", "SMTOY",
    # Materials
    "ALB", "LIN", "FCX",
    # Real Estate
    "EQIX", "DLR", "IRM", "AMT",
    # Utilities
    "VST", "CEG", "NRG", "NEE", "ETN", "PWR", "EIX", "AES",
]

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MAX_OUTPUT_TOKENS = 8192

# ── Guardrails ─────────────────────────────────────────────────────────────────
MAX_STOCK_WEIGHT = 0.10
MIN_HEDGE_SECTOR_WEIGHT = 0.02
MAX_TURNOVER_VS_PREV = 0.20
WEIGHT_SUM_TOLERANCE = 0.01
MAX_SHORT_WEIGHT = 0.08
MAX_GROSS_EXPOSURE = 1.80
