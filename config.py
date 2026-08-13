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
    # Broad daily market/business media — day-to-day whole-market context now
    # that the LLM invests across the entire market, not just the AI complex.
    # (Tagged "market"; value_chain_layer is a free-form doc tag.)
    {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html", "value_chain_layer": "market"},  # CNBC Markets
    {"url": "http://feeds.marketwatch.com/marketwatch/topstories/", "value_chain_layer": "market"},  # MarketWatch
    {"url": "https://finance.yahoo.com/news/rssindex",             "value_chain_layer": "market"},  # Yahoo Finance
    {"url": "https://seekingalpha.com/feed.xml",                   "value_chain_layer": "market"},  # Seeking Alpha
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "value_chain_layer": "market"},  # NYT Business
    {"url": "https://www.investing.com/rss/news_25.rss",           "value_chain_layer": "market"},  # Investing.com
]

# ── HuggingFace Daily Papers ───────────────────────────────────────────────────
HF_PAPERS_MAX_RESULTS = 50

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

# ── opencode (thesis backend; replaces Gemini API, which was 403-suspended) ────
# Invoked as a CLI via scoring.thesis_scorer._OpencodeClient. Uses the opencode-go
# subscription gateway — no API key in this process, nothing remotely revocable.
OPENCODE_MODEL = "opencode-go/qwen3.7-max"
OPENCODE_TIMEOUT_S = 120

# Embeddings moved to a LOCAL model (fastembed / nomic-embed-text-v1.5, 768-dim)
# in chroma_store.py after the Gemini project was 403-suspended. No API embedding
# constant remains. See docs/superpowers/specs/2026-08-13-*.

# ── Guardrails ─────────────────────────────────────────────────────────────────
MAX_STOCK_WEIGHT = 0.10
MIN_HEDGE_SECTOR_WEIGHT = 0.02
MAX_TURNOVER_VS_PREV = 0.20
WEIGHT_SUM_TOLERANCE = 0.01

# ── LLM autonomy ───────────────────────────────────────────────────────────────
# "full":        the LLM is the portfolio manager. It may output target weights
#                directly (long-only, within TICKER_UNIVERSE), all numeric
#                clamps are off, and "trade sensibly" is enforced by prompt,
#                not code. Every decision is still persisted for ablation.
# "guardrailed": the bounded decision surface (tilt clamps [8%,35%], top-n
#                [2,4], name emphasis [0.5x,1.5x], cash <=30%).
LLM_AUTONOMY = "full"

# Whole-market autonomy: in full mode the LLM may hold ANY liquid, fractionable
# US equity (validated against Alpaca's tradable-asset list), not just the
# curated ~58-name layer universe. Set False to confine it to TICKER_UNIVERSE.
# The layer map below still drives the dashboard's layer view and the
# guardrailed dial pipeline; whole-market names outside it show as "other".
WHOLE_MARKET = True
# Agentic retrieval budget (rounds of follow-up queries against ChromaDB).
LLM_SEARCH_MAX_ROUNDS = 5
LLM_SEARCH_MAX_QUERIES = 5
LLM_DOCS_PER_QUERY = 10

# ── Performance accounting ─────────────────────────────────────────────────────
# Net capital deposited into the (paper) account, used as the cost basis for
# total-return math in portfolio snapshots. No top-ups/withdrawals to date.
STARTING_CAPITAL = 100_000.0
