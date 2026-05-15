# AI Signal Engine

An AI infrastructure long/short equity signal engine. Ingests research from RSS feeds, arXiv, and SEC EDGAR filings, computes a macro regime signal from live economic data, then calls Gemini to produce a long and short portfolio book — executed automatically via Alpaca paper trading.

## Thesis

Aschenbrenner's exponential compute trajectory: AI is in a physical buildout supercycle driven by chips, memory, power, cooling, datacenters, and networking. The engine identifies which stocks will benefit from the *next* 1–4 quarters of AI infrastructure capex commitments and supply-chain bottlenecks — on both the supply side (semiconductors, power, cooling, interconnects) and demand side (hyperscalers, AI platforms, inference-scaling beneficiaries).

## Architecture

```
Ingestion (RSS / arXiv / EDGAR)
        ↓
Macro Signal (FRED + yfinance)
   PMI · Semis IP · BDRY shipping · Power→Compute · Copper→Infra · Credit stress
        ↓
Gemini Scorer
   Macro block + value-chain-layered documents → long book + short book
        ↓
Alpaca Execution
   Two-sided rebalance sized by net_exposure_target
        ↓
Export (stock_signals.json · market_regime.json · p_estimate.json)
```

## Macro Regimes

| Regime | Net Exposure | Trigger |
|---|---|---|
| `credit_stress` | 0.20 | VIX > 25 and HYG 30d return < −2% |
| `shipping_bottleneck` | 0.55 | BDRY 30d momentum normalised > 0.65 |
| `balanced` | 0.65 | PMI contracting and copper z-score < −0.5σ |
| `compute_constrained` | 0.80 | Default — thesis intact |

## Schedule

Runs Mon–Fri at **9:00am ET** via cron. Signal engine output is consumed by the portfolio backtest/execution layer at 9:30am ET.

## Setup

```bash
pip install -r requirements.txt

# Required
export GEMINI_API_KEY=...

# Optional — enables live macro signals (free at fred.stlouisfed.org)
export FRED_API_KEY=...

# Optional — enables Alpaca paper execution
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

Store secrets in a `.env` file at the repo root (gitignored). `run.sh` loads it automatically.

## Usage

```bash
python main.py              # full run: ingest → macro → score → execute → export
python main.py --force      # re-score even if no new documents (uses last 30 days)
python main.py --dry-run    # ingest + macro + score only, no trades or file writes
```

## Portfolio Structure

- **Long book** — highest-conviction AI buildout beneficiaries. Weights sum to 1.0, max 10% per position.
- **Short book** — lowest-conviction names in the same factor space. Weights sum to 1.0, max 8% per position.
- **Net exposure** — set by macro regime signal (`net_exposure_target`), reducing gross long/short notional during stress.
- **Gross exposure cap** — Σ long + Σ short ≤ 1.80×.

## Ticker Universe

95 globally-listed stocks across compute (NVDA, ASML, TSM, AMAT…), power & utilities (VST, CEG, ETN, PWR…), datacenter REITs (EQIX, DLR…), hyperscalers (MSFT, GOOGL, META, AMZN…), and AI software (PLTR, DDOG, CRM, NOW…).

## Tests

```bash
python -m pytest tests/ -v
```

57 tests covering macro signal computation, scorer guardrails, DB migrations, ingestion deduplication, and export formatting.
