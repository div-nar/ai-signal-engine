# ai-signal-engine/main.py
"""
AI Signal Engine — Main orchestrator.

Usage:
    python main.py            # full run: ingest, score, export
    python main.py --force    # force re-score even if no new documents
    python main.py --dry-run  # ingest + score but don't write JSON files

Prerequisites:
    export GEMINI_API_KEY=...   (or set in .env file)
    pip install -r requirements.txt
"""
import argparse
import json
from datetime import datetime, timezone

from config import (
    DB_PATH, RSS_FEEDS, ARXIV_CATEGORIES, ARXIV_MAX_RESULTS,
    EDGAR_TICKERS,
)
from db import init_db, get_unscored_documents, get_recent_documents, mark_scored, insert_signal
from ingestion.rss import ingest_rss
from ingestion.arxiv import ingest_arxiv
from ingestion.transcripts import ingest_edgar
from scoring.gemini_scorer import score_documents
from export import export_signal


def get_alpaca_positions() -> dict:
    """Fetch current paper portfolio weights from Alpaca. Returns {} if unavailable."""
    import os
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return {}
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            return {}
        positions = client.get_all_positions()
        return {
            p.symbol: float(p.market_value) / portfolio_value
            for p in positions
        }
    except Exception as e:
        print(f"  WARNING: Could not fetch Alpaca positions: {e}")
        return {}


def get_prev_weights(db_path: str) -> dict:
    """Load stock conviction scores from most recent signals row."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT stock_conviction FROM signals ORDER BY computed_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row and row[0]:
        return json.loads(row[0])
    return {}


def main():
    parser = argparse.ArgumentParser(description="AI Signal Engine")
    parser.add_argument("--force", action="store_true",
                        help="Score even if no new documents were ingested")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ingest and score but don't write output JSON files")
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine starting...")

    # 1. Init DB
    init_db(DB_PATH)

    # 2. Ingest
    print("\n--- Ingestion ---")
    total_new = 0

    for feed in RSS_FEEDS:
        n = ingest_rss(feed["url"], feed["value_chain_layer"], DB_PATH)
        print(f"  RSS [{feed['value_chain_layer']}]: {n} new documents")
        total_new += n

    n = ingest_arxiv(ARXIV_CATEGORIES, ARXIV_MAX_RESULTS, DB_PATH)
    print(f"  arXiv: {n} new documents")
    total_new += n

    n = ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=DB_PATH)
    print(f"  EDGAR: {n} new documents")
    total_new += n

    print(f"\nTotal new documents: {total_new}")

    # 3. Decide whether to score
    unscored = get_unscored_documents(DB_PATH)
    if not unscored and not args.force:
        print("No unscored documents — nothing to do. Use --force to override.")
        return

    if not unscored and args.force:
        unscored = get_recent_documents(DB_PATH, days=30)
        print(f"Force mode: re-scoring {len(unscored)} documents from last 30 days")

    if not unscored:
        print("No documents available to score (even in last 30 days). Run ingestion first.")
        return

    # 4. Score
    print(f"\n--- Scoring {len(unscored)} documents via Gemini ---")
    prev_weights = get_prev_weights(DB_PATH)

    # Fetch live Alpaca positions for Gemini context + guardrail baseline
    print("\nFetching current Alpaca portfolio...")
    current_portfolio = get_alpaca_positions()
    if current_portfolio:
        top = sorted(current_portfolio.items(), key=lambda x: -x[1])[:5]
        print(f"  {len(current_portfolio)} positions | top: " + ", ".join(f"{t} {w:.1%}" for t, w in top))
    else:
        print("  No Alpaca positions (credentials not set or empty portfolio)")

    signal = score_documents(docs=unscored, db_path=DB_PATH, prev_weights=prev_weights, current_portfolio=current_portfolio)

    # 5. Persist
    doc_ids = [d["id"] for d in unscored]
    mark_scored(DB_PATH, doc_ids)
    insert_signal(DB_PATH, signal)
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | confidence={signal['signal_confidence']:.2f}")
    print(f"  {signal['thesis_update']}")

    # 6. Export
    if args.dry_run:
        print("\n[DRY-RUN] Skipping JSON export")
    else:
        print("\n--- Exporting ---")
        export_signal(signal)

    print("\nDone.")


if __name__ == "__main__":
    main()
