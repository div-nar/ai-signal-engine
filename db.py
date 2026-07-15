# ai-signal-engine/db.py
import json
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

        CREATE TABLE IF NOT EXISTS portfolio_history (
            id                INTEGER PRIMARY KEY,
            snapshot_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            equity            REAL,
            cash              REAL,
            long_market_value REAL,
            unrealized_pl     REAL,
            realized_to_date  REAL,
            net_deposits      REAL,
            total_return_pct  REAL
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


def insert_portfolio_snapshot(db_path: str, data: dict) -> int:
    """Persist one mark-to-market account snapshot. Returns the new row id."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO portfolio_history
           (equity, cash, long_market_value, unrealized_pl,
            realized_to_date, net_deposits, total_return_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            data["equity"],
            data["cash"],
            data["long_market_value"],
            data["unrealized_pl"],
            data["realized_to_date"],
            data["net_deposits"],
            data["total_return_pct"],
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


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


# Columns added after the original targets schema shipped; init_targets_table
# migrates existing DBs in place (ALTER TABLE is cheap and idempotent-guarded).
_TARGETS_EXTRA_COLUMNS = {
    "layer_top_n": "TEXT",          # JSON {layer: 2..4} — LLM concentration dial
    "name_adjustments": "TEXT",     # JSON {ticker: 0 | 0.5..1.5} — LLM name emphasis/veto
    "cash_buffer": "REAL DEFAULT 0",     # LLM risk stance, clamped [0, 0.30]
    "rebalance_urgency": "TEXT",    # urgent | normal | hold
    "retrieval_log": "TEXT",        # JSON — agentic retrieval trace (for ablation)
    "trade_gate": "TEXT",           # traded | skipped_hold | skipped_within_bands
    "autonomy": "TEXT",             # full | guardrailed (mode that produced this target)
    "weights_source": "TEXT",       # llm_direct | dial_pipeline
}


def init_targets_table(db_path: str = str(DEFAULT_DB)) -> None:
    """Create the weekly-target table (idempotent) and migrate older schemas."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS targets (
            id             INTEGER PRIMARY KEY,
            computed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            layer_tilt     TEXT,
            layer_budgets  TEXT,
            target_weights TEXT,
            market_regime  TEXT,
            thesis_update  TEXT,
            regime_shift   INTEGER DEFAULT 0
        )
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(targets)")}
    for col, decl in _TARGETS_EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE targets ADD COLUMN {col} {decl}")
    conn.commit()
    conn.close()


def insert_target(db_path: str, data: dict) -> int:
    """Persist one weekly target. Dict fields are JSON-encoded."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """INSERT INTO targets
           (layer_tilt, layer_budgets, target_weights, market_regime,
            thesis_update, regime_shift, layer_top_n, name_adjustments,
            cash_buffer, rebalance_urgency, retrieval_log, autonomy,
            weights_source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            json.dumps(data.get("layer_tilt", {})),
            json.dumps(data.get("layer_budgets", {})),
            json.dumps(data.get("target_weights", {})),
            data.get("market_regime", ""),
            data.get("thesis_update", ""),
            int(bool(data.get("regime_shift", False))),
            json.dumps(data.get("layer_top_n", {})),
            json.dumps(data.get("name_adjustments", {})),
            float(data.get("cash_buffer", 0.0)),
            data.get("rebalance_urgency", "normal"),
            json.dumps(data.get("retrieval_log", [])),
            data.get("autonomy", ""),
            data.get("weights_source", ""),
        ),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


def get_latest_target(db_path: str = str(DEFAULT_DB)) -> dict | None:
    """Most recent target row with JSON fields parsed, or None."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM targets ORDER BY computed_at DESC, id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    keys = row.keys()
    return {
        "id": row["id"],
        "computed_at": row["computed_at"],
        "layer_tilt": json.loads(row["layer_tilt"] or "{}"),
        "layer_budgets": json.loads(row["layer_budgets"] or "{}"),
        "target_weights": json.loads(row["target_weights"] or "{}"),
        "market_regime": row["market_regime"],
        "thesis_update": row["thesis_update"],
        "regime_shift": bool(row["regime_shift"]),
        "layer_top_n": json.loads((row["layer_top_n"] if "layer_top_n" in keys else None) or "{}"),
        "name_adjustments": json.loads((row["name_adjustments"] if "name_adjustments" in keys else None) or "{}"),
        "cash_buffer": float(row["cash_buffer"] or 0.0) if "cash_buffer" in keys else 0.0,
        "rebalance_urgency": (row["rebalance_urgency"] if "rebalance_urgency" in keys else None) or "normal",
        "retrieval_log": json.loads((row["retrieval_log"] if "retrieval_log" in keys else None) or "[]"),
        "trade_gate": (row["trade_gate"] if "trade_gate" in keys else None) or "",
        "autonomy": (row["autonomy"] if "autonomy" in keys else None) or "",
        "weights_source": (row["weights_source"] if "weights_source" in keys else None) or "",
    }


def update_target_gate(db_path: str, target_id: int, gate: str) -> None:
    """Record the Friday trade-gate decision on a persisted target, so the
    Monday buy leg can honour a skip."""
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE targets SET trade_gate = ? WHERE id = ?", (gate, target_id))
    conn.commit()
    conn.close()
