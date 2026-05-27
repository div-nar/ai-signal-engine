# ai-signal-engine/db.py
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).parent / "signals.db"


def init_db(db_path: str = str(DEFAULT_DB)) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id                INTEGER PRIMARY KEY,
            source            TEXT NOT NULL,
            title             TEXT NOT NULL,
            url               TEXT UNIQUE NOT NULL,
            published_at      TIMESTAMP,
            content           TEXT,
            ingested_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scored            BOOLEAN DEFAULT FALSE,
            value_chain_layer TEXT
        );

        CREATE TABLE IF NOT EXISTS scores (
            id           INTEGER PRIMARY KEY,
            doc_id       INTEGER REFERENCES documents(id),
            p_delta      REAL,
            stock_scores TEXT,
            thesis_tags  TEXT,
            scored_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS signals (
            id                       INTEGER PRIMARY KEY,
            computed_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            p_final                  REAL,
            stock_conviction         TEXT,
            stock_weights            TEXT,
            stock_reasoning          TEXT,
            sector_tilt              TEXT,
            supply_demand_balance    REAL,
            market_regime            TEXT,
            signal_confidence        REAL,
            thesis_stress            BOOLEAN,
            signal_age_days          INTEGER,
            sources_ingested         INTEGER,
            signal_breakdown         TEXT,
            thesis_update            TEXT,
            raw_response             TEXT,
            prompt_context_doc_ids   TEXT
        );
    """)
    # Migration: add reasoning-trace columns to existing signals tables.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
    for col in ("stock_weights", "stock_reasoning", "raw_response", "prompt_context_doc_ids",
                "short_weights", "macro_signal"):
        if col not in existing:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} TEXT")
    conn.commit()
    conn.close()


def insert_document(
    db_path: str,
    source: str,
    title: str,
    url: str,
    published_at: Optional[str],
    content: str,
    value_chain_layer: str,
) -> Optional[int]:
    """Insert a document. Returns row id, or None if URL already exists."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO documents (source, title, url, published_at, content, value_chain_layer)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source, title, url, published_at, content, value_chain_layer),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_unscored_documents(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM documents WHERE scored = 0 ORDER BY ingested_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_documents(db_path: str = str(DEFAULT_DB), days: int = 30) -> list[dict]:
    """Return all documents ingested within the last N days."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM documents WHERE ingested_at > datetime('now', ?) ORDER BY ingested_at",
        (f"-{days} days",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_documents(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_signals(db_path: str = str(DEFAULT_DB)) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_scored(db_path: str, doc_ids: list[int]) -> None:
    if not doc_ids:
        return
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"UPDATE documents SET scored = 1 WHERE id IN ({','.join('?' * len(doc_ids))})",
        doc_ids,
    )
    conn.commit()
    conn.close()


def insert_signal(db_path: str, data: dict) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO signals
           (p_final, stock_conviction, stock_weights, stock_reasoning,
            sector_tilt, supply_demand_balance, market_regime, signal_confidence,
            thesis_stress, signal_age_days, sources_ingested, signal_breakdown,
            thesis_update, raw_response, prompt_context_doc_ids,
            short_weights, macro_signal)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["p_final"],
            data["stock_conviction"],
            data.get("stock_weights"),
            data.get("stock_reasoning"),
            data["sector_tilt"],
            data["supply_demand_balance"],
            data["market_regime"],
            data["signal_confidence"],
            int(data["thesis_stress"]),
            data["signal_age_days"],
            data["sources_ingested"],
            data["signal_breakdown"],
            data["thesis_update"],
            data.get("raw_response"),
            data.get("prompt_context_doc_ids"),
            data.get("short_weights"),
            data.get("macro_signal"),
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid
