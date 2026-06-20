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
    EDGAR_TICKERS, STARTING_CAPITAL,
)
from db import init_db, get_unscored_documents, get_recent_documents, mark_scored, insert_signal, get_all_documents, get_all_signals, insert_portfolio_snapshot
from ingestion.rss import ingest_rss
from ingestion.huggingface_papers import ingest_hf_papers
from ingestion.transcripts import ingest_edgar
from macro.regime import compute_macro_signal
from macro.composite import is_cache_stale, fit_and_cache_composite
from scoring.gemini_scorer import score_documents
from execution.alpaca import get_alpaca_positions, rebalance, get_account_snapshot
from export import export_signal
from chroma_store import init_chroma, run_chroma_backfill, upsert_signal_record


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
    try:
        run_chroma_backfill(chroma_client, all_docs, all_sigs, sentinel_path)
    except Exception as e:
        # Semantic retrieval is an enhancement, not a hard dependency — never let
        # it block ingestion → scoring → execution → export.
        print(f"  WARNING: ChromaDB backfill errored, continuing without it: {e}")

    # Weekly PCA refit (Monday, or if cache is stale)
    composite_cache = str(Path(__file__).parent / "data" / "composite_modifier_cache.json")
    Path(__file__).parent.joinpath("data").mkdir(parents=True, exist_ok=True)
    today_weekday = datetime.now(timezone.utc).weekday()  # 0 = Monday
    if today_weekday == 0 or is_cache_stale(composite_cache):
        print("\n--- Fitting PCA composite modifier (weekly) ---")
        fred_key = os.environ.get("FRED_API_KEY")
        try:
            fit_and_cache_composite(composite_cache, fred_api_key=fred_key)
            print("  PCA modifier cache written.")
        except Exception as e:
            # The composite modifier is an enhancement — daily apply falls back
            # to a 0.0 modifier when the cache is missing/stale. Never let a macro
            # data hiccup block ingestion → scoring → execution → export.
            print(f"  WARNING: PCA composite refit failed, using prior/zero modifier: {e}")

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
    macro_signal = compute_macro_signal(cache_path=composite_cache)
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
    try:
        signal = score_documents(
            docs=unscored,
            db_path=DB_PATH,
            prev_weights=prev_weights,
            current_portfolio=current_portfolio,
            macro_signal=macro_signal,
            chroma_client=chroma_client,
        )
    except Exception as e:
        # Scoring needs a live Gemini call (LLM + embeddings). On any failure —
        # depleted credits (429), rate limits, timeouts, 5xx — do NOT crash and do
        # NOT rebalance on stale/partial data: hold the current book, leave the
        # documents unscored so they retry next run, and exit cleanly so the cron
        # stays healthy and resumes automatically once Gemini is reachable again.
        print(f"  ERROR: Gemini scoring failed — holding current book, skipping "
              f"rebalance and export. Documents left unscored for retry next run.")
        print(f"  Reason: {e}")
        return

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

    # 9. Record a mark-to-market portfolio snapshot for our own performance
    # history (independent of Alpaca's limited retention). Best-effort: a
    # snapshot failure must never fail an otherwise-successful run.
    try:
        snapshot = get_account_snapshot(net_deposits=STARTING_CAPITAL)
        if snapshot:
            insert_portfolio_snapshot(DB_PATH, snapshot)
            print(
                f"  Portfolio: equity ${snapshot['equity']:,.0f} | "
                f"total return {snapshot['total_return_pct']:+.2f}% | "
                f"realized ${snapshot['realized_to_date']:,.0f} | "
                f"unrealized ${snapshot['unrealized_pl']:,.0f}"
            )
    except Exception as e:
        print(f"  WARNING: portfolio snapshot failed (non-fatal): {e}")

    print("\nDone.")


if __name__ == "__main__":
    _args = parse_args()
    print(f"[{datetime.now(timezone.utc).isoformat()}] AI Signal Engine "
          f"({_args.mode}) starting...")
    dispatch(_args.mode)
    print("Done.")
