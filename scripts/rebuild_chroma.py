"""Rebuild the Chroma collections from SQLite at the current embedding dim.

The old vectors were Gemini-embedded (different dimensionality) and cannot
coexist with the new local nomic 768-dim vectors, so this drops both
collections + the backfill sentinel, then re-embeds every doc and signal.
Idempotent — safe to re-run.

    python scripts/rebuild_chroma.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chroma_store
import db
from config import DB_PATH

DATA_DIR = ROOT / "data"
CHROMA_PATH = str(DATA_DIR / "chroma")
SENTINEL = DATA_DIR / "chroma_backfill_done"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chroma_store.init_chroma(CHROMA_PATH)
    for name in ("research_docs", "macro_signals"):
        try:
            client.delete_collection(name)
            print(f"  dropped collection {name}")
        except Exception as e:
            print(f"  (collection {name} not dropped: {e})")
    # recreate empty collections
    client = chroma_store.init_chroma(CHROMA_PATH)
    if SENTINEL.exists():
        SENTINEL.unlink()
        print("  removed backfill sentinel")

    docs = db.get_all_documents(DB_PATH)
    signals = db.get_all_signals(DB_PATH)
    print(f"  re-embedding {len(docs)} docs + {len(signals)} signals (local nomic)...")
    chroma_store.run_chroma_backfill(client, docs, signals, str(SENTINEL))

    counts = {c.name: client.get_collection(c.name).count()
              for c in client.list_collections()}
    print(f"  done. collection counts: {counts}")


if __name__ == "__main__":
    main()
