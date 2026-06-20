# AI Signal Engine

An AI-infrastructure **long-only** equity signal engine. It ingests research (RSS feeds,
HuggingFace daily papers, SEC EDGAR filings) and turns a view of the AI buildout into a
systematic equity portfolio, executed on Alpaca paper trading.

**Status:** 🔧 Next-generation engine **in the works**. The first-generation engine has been
**retired** — see the writeup below.

---

## First-generation engine — retired

The v1 engine used a large-language-model scorer to produce a conviction-weighted portfolio,
scaled by a macro regime model, rebalanced daily. It ran on Alpaca paper from **25 Mar 2026**
and finished at **+16.66%** ($100,000 → $116,659).

A full post-mortem — architecture, returns vs the S&P 500 and NASDAQ-100, attribution,
correlation/overlap with the benchmarks, the selection-edge ablations, operational issues, and
the rationale for retirement — is in:

➡️ **[`docs/reports/2026-06-20-ai-signal-engine-v1-retirement-report.md`](docs/reports/2026-06-20-ai-signal-engine-v1-retirement-report.md)**

The v1 launchd job has been unloaded; nothing from v1 trades anymore.

---

## Next-generation engine

A successor that keeps the AI-infrastructure thesis but reworks how the portfolio is formed is
**currently under development**. Details will be documented here once it is live.

---

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/        # test suite
```

Secrets (`GEMINI_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) live in a `.env` file at the
repo root (gitignored); `run.sh` loads it automatically.

## Repository layout

| Path | Purpose |
|---|---|
| `ingestion/` | Research ingestion (RSS, HuggingFace papers, SEC EDGAR) → SQLite |
| `pricing/` | Live daily price history (Alpaca market data) |
| `strategy/` | Portfolio construction primitives |
| `scoring/` | LLM thesis pass |
| `execution/` | Alpaca paper-trading execution |
| `backtest/` | Historical backtest + evaluation harness |
| `db.py` | SQLite schema and accessors |
| `docs/` | Design specs, plans, and reports |

*Paper-trading research project; not investment advice.*
