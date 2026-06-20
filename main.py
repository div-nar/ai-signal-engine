# ai-signal-engine/main.py
"""AI Signal Engine — entrypoint.

Usage:
    python main.py --mode passive   # ingest research + compute/persist weekly target (no trades)
    python main.py --mode sell      # compute/persist target, then execute the Friday sell leg
    python main.py --mode buy       # execute the Monday buy leg from the latest target + snapshot

Prerequisites:
    export GEMINI_API_KEY=...   ALPACA_API_KEY=...   ALPACA_SECRET_KEY=...   (or .env)
"""
import argparse
from datetime import datetime, timezone

from config import (
    DB_PATH, RSS_FEEDS, HF_PAPERS_MAX_RESULTS, EDGAR_TICKERS, STARTING_CAPITAL,
)
from db import init_db, get_recent_documents, insert_portfolio_snapshot
from ingestion.rss import ingest_rss
from ingestion.huggingface_papers import ingest_hf_papers
from ingestion.transcripts import ingest_edgar
from execution.alpaca import get_account_snapshot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI Signal Engine (layer-cake)")
    parser.add_argument("--mode", required=True, choices=["passive", "sell", "buy"],
                        help="passive: ingest+compute target (no trades); "
                             "sell: compute+persist target, Friday sell leg; "
                             "buy: execute Monday buy leg from latest target")
    return parser.parse_args(argv)


def _gather_docs(db_path=DB_PATH, chroma_client=None):
    """Run ingestion and return recent documents for the thesis pass."""
    init_db(db_path)
    total = 0
    for feed in RSS_FEEDS:
        total += ingest_rss(feed["url"], feed["value_chain_layer"], db_path,
                            chroma_client=chroma_client)
    total += ingest_hf_papers(HF_PAPERS_MAX_RESULTS, db_path, chroma_client=chroma_client)
    total += ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=db_path,
                          chroma_client=chroma_client)
    print(f"  Ingested {total} new documents")
    return get_recent_documents(db_path, days=30)


def _record_snapshot(db_path=DB_PATH):
    """Best-effort mark-to-market snapshot for performance history (never fatal)."""
    try:
        snap = get_account_snapshot(net_deposits=STARTING_CAPITAL)
        if snap:
            insert_portfolio_snapshot(db_path, snap)
            print(f"  Portfolio: equity ${snap['equity']:,.0f} | "
                  f"total return {snap['total_return_pct']:+.2f}%")
    except Exception as e:
        print(f"  WARNING: portfolio snapshot failed (non-fatal): {e}")


def dispatch(mode, run_passive=None, run_sell_fn=None, run_buy_fn=None,
             gather_docs=None, record_snapshot=None):
    """Route a mode to its handler. Collaborators are injected for testability.

    passive/sell ingest fresh research and feed it to the thesis pass; buy executes
    the persisted target and records a portfolio snapshot for performance history.
    """
    if gather_docs is None:
        gather_docs = _gather_docs
    if record_snapshot is None:
        record_snapshot = _record_snapshot
    if run_passive is None:
        from orchestrate import compute_weekly_target as run_passive
    if run_sell_fn is None:
        from orchestrate import run_sell as run_sell_fn
    if run_buy_fn is None:
        from orchestrate import run_buy as run_buy_fn
    if mode == "passive":
        return run_passive(gather_docs())
    if mode == "sell":
        return run_sell_fn(gather_docs())
    if mode == "buy":
        result = run_buy_fn()
        record_snapshot()
        return result
    raise ValueError(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    _args = parse_args()
    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine "
          f"({_args.mode}) starting...")
    dispatch(_args.mode)
    print("Done.")
