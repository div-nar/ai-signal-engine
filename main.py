# ai-signal-engine/main.py
"""
AI Signal Engine — Main orchestrator.

Usage:
    python main.py            # full run: ingest, score, export
    python main.py --force    # force re-score even if no new documents
    python main.py --dry-run  # ingest + score but don't write JSON files or execute trades

Prerequisites:
    export GEMINI_API_KEY=...   (or set in .env file)
    pip install -r requirements.txt
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DB_PATH, RSS_FEEDS, HF_PAPERS_MAX_RESULTS,
    EDGAR_TICKERS,
)
from db import init_db, get_unscored_documents, get_recent_documents, mark_scored, insert_signal, get_all_documents, get_all_signals
from ingestion.rss import ingest_rss
from ingestion.huggingface_papers import ingest_hf_papers
from ingestion.transcripts import ingest_edgar
from macro.regime import compute_macro_signal
from scoring.gemini_scorer import score_documents
from execution.alpaca import get_alpaca_positions, rebalance
from export import export_signal
from chroma_store import init_chroma, run_chroma_backfill, upsert_signal_record


def get_prev_weights(db_path: str) -> dict:
    """Load stock weights from most recent signal row."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT stock_weights FROM signals ORDER BY computed_at DESC LIMIT 1"
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
                        help="Ingest and score but don't write output JSON files or execute trades")
    args = parser.parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine starting...")

    # 1. Init DB
    init_db(DB_PATH)

    # Init ChromaDB and run one-time backfill if needed
    chroma_path = str(Path(__file__).parent / "data" / "chroma")
    sentinel_path = str(Path(__file__).parent / "data" / "chroma_backfill_done")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = init_chroma(chroma_path)
    all_docs = get_all_documents(DB_PATH)
    all_sigs = get_all_signals(DB_PATH)
    run_chroma_backfill(chroma_client, all_docs, all_sigs, sentinel_path)

    # 2. Ingest
    print("\n--- Ingestion ---")
    total_new = 0
    for feed in RSS_FEEDS:
        n = ingest_rss(feed["url"], feed["value_chain_layer"], DB_PATH, chroma_client=chroma_client)
        print(f"  RSS [{feed['value_chain_layer']}]: {n} new documents")
        total_new += n
    n = ingest_hf_papers(HF_PAPERS_MAX_RESULTS, DB_PATH, chroma_client=chroma_client)
    print(f"  HuggingFace Papers: {n} new documents")
    total_new += n
    n = ingest_edgar(EDGAR_TICKERS, max_per_ticker=3, db_path=DB_PATH, chroma_client=chroma_client)
    print(f"  EDGAR: {n} new documents")
    total_new += n
    print(f"\nTotal new documents: {total_new}")

    unscored = get_unscored_documents(DB_PATH)
    if not unscored and not args.force:
        print("No unscored documents — nothing to do. Use --force to override.")
        return
    if not unscored and args.force:
        unscored = get_recent_documents(DB_PATH, days=30)
        print(f"Force mode: re-scoring {len(unscored)} documents from last 30 days")
    if not unscored:
        print("No documents available to score. Run ingestion first.")
        return

    # 3. Compute macro signal
    print("\n--- Macro Signal ---")
    macro_signal = compute_macro_signal()
    print(f"  Regime: {macro_signal['regime']} (confidence: {macro_signal['regime_confidence']:.2f})")
    print(f"  Net exposure target: {macro_signal['net_exposure_target']:.2f}")
    print(f"  {macro_signal['notes']}")

    # 4. Fetch current Alpaca positions
    print("\nFetching current Alpaca portfolio...")
    positions = get_alpaca_positions()
    current_portfolio = positions["longs"]
    if current_portfolio:
        top = sorted(current_portfolio.items(), key=lambda x: -x[1])[:5]
        print(f"  {len(current_portfolio)} long positions | top: " + ", ".join(f"{t} {w:.1%}" for t, w in top))

    # 5. Score
    print(f"\n--- Scoring {len(unscored)} documents via Gemini ---")
    prev_weights = get_prev_weights(DB_PATH)
    signal = score_documents(
        docs=unscored,
        db_path=DB_PATH,
        prev_weights=prev_weights,
        current_portfolio=current_portfolio,
        macro_signal=macro_signal,
        chroma_client=chroma_client,
    )

    # 6. Persist
    doc_ids = [d["id"] for d in unscored]
    mark_scored(DB_PATH, doc_ids)
    signal_id = insert_signal(DB_PATH, signal)
    thesis = signal.get("thesis_update", "")
    notes = ""
    if signal.get("macro_signal"):
        try:
            notes = json.loads(signal["macro_signal"]).get("notes", "")
        except Exception:
            pass
    upsert_signal_record(
        chroma_client,
        f"signal_{signal_id}",
        f"{thesis} {notes}".strip(),
        {
            "regime": signal["market_regime"],
            "p_final": float(signal["p_final"]),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"  p={signal['p_final']:.3f} | regime={signal['market_regime']} | confidence={signal['signal_confidence']:.2f}")
    print(f"  {signal['thesis_update']}")

    # 7. Execute rebalance
    if args.dry_run:
        print("\n[DRY-RUN] Skipping rebalance and export")
        return

    print("\n--- Executing Rebalance ---")
    long_weights = json.loads(signal.get("stock_weights") or "{}")
    rebalance(
        long_weights=long_weights,
        net_exposure_target=macro_signal["net_exposure_target"],
    )

    # 8. Export
    print("\n--- Exporting ---")
    export_signal(signal)

    print("\nDone.")


if __name__ == "__main__":
    main()
