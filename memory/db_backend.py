"""SQLite-Backed Memory Database Backend for AI Agent Architecture.

This module provides a SQLite persistence backend for the memory subsystem,
writing episodic events, semantic facts, and router decisions into the same
copperleaf.db used by the MCP server. This ensures the memory system is a
genuine extension of the existing database, not a parallel flat-file store.

All functions are thin wrappers around raw SQL. The higher-level memory
classes (EpisodicMemory, SemanticMemory, PromoteOrDropRouter) call these
functions as side-effects so that every in-memory mutation is also durable.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "copperleaf.db"


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    """Read-only SQLite connection with Row factory."""
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def _write_conn() -> Generator[sqlite3.Connection, None, None]:
    """Write SQLite connection wrapped in an explicit transaction."""
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.execute("BEGIN")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Events
# ─────────────────────────────────────────────────────────────────────────────

def ep_insert(
    event_id: str,
    event_type: str,
    summary: str,
    details: dict[str, Any],
    importance_score: float,
    tags: List[str],
    source: str,
    metadata: dict[str, Any],
) -> None:
    """Insert a new episodic event record into copperleaf.db."""
    with _write_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO episodic_events
                (event_id, event_type, summary, details_json, importance_score,
                 tags_json, source, is_consolidated, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                event_id,
                event_type,
                summary,
                json.dumps(details),
                importance_score,
                json.dumps(tags),
                source,
                json.dumps(metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def ep_mark_consolidated(event_ids: List[str]) -> int:
    """Mark episodic events as processed by the consolidation engine."""
    if not event_ids:
        return 0
    now_str = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" * len(event_ids))
    with _write_conn() as conn:
        cursor = conn.execute(
            f"UPDATE episodic_events SET is_consolidated = 1, consolidated_at = ? "
            f"WHERE event_id IN ({placeholders})",
            [now_str, *event_ids],
        )
    return cursor.rowcount


def ep_delete(event_id: str) -> bool:
    """Delete an episodic event by ID."""
    with _write_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM episodic_events WHERE event_id = ?", (event_id,)
        )
    return cursor.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Facts
# ─────────────────────────────────────────────────────────────────────────────

def sf_upsert(
    fact_id: str,
    subject: str,
    predicate: str,
    value: Any,
    version: int,
    status: str,
    confidence: float,
    source_event_ids: List[str],
    history: List[dict[str, Any]],
    valid_until: Optional[str],
    metadata: dict[str, Any],
) -> None:
    """Insert or update a semantic fact in copperleaf.db."""
    now_str = datetime.now(timezone.utc).isoformat()
    with _write_conn() as conn:
        existing = conn.execute(
            "SELECT fact_id FROM semantic_facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE semantic_facts SET
                    value_json = ?, version = ?, status = ?, confidence = ?,
                    source_event_ids = ?, history_json = ?, valid_until = ?,
                    metadata_json = ?, updated_at = ?
                WHERE fact_id = ?
                """,
                (
                    json.dumps(value),
                    version,
                    status,
                    confidence,
                    json.dumps(source_event_ids),
                    json.dumps(history),
                    valid_until,
                    json.dumps(metadata),
                    now_str,
                    fact_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO semantic_facts
                    (fact_id, subject, predicate, value_json, version, status,
                     confidence, source_event_ids, history_json, valid_until,
                     metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    subject,
                    predicate,
                    json.dumps(value),
                    version,
                    status,
                    confidence,
                    json.dumps(source_event_ids),
                    json.dumps(history),
                    valid_until,
                    json.dumps(metadata),
                    now_str,
                    now_str,
                ),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Router Decisions
# ─────────────────────────────────────────────────────────────────────────────

def rd_insert(
    decision_id: str,
    item_id: str,
    action: str,
    reason: str,
    confidence: float,
    item_role: str,
    item_summary: str,
    promoted_event_id: Optional[str],
) -> None:
    """Persist a router decision to the database for grader inspection."""
    with _write_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO router_decisions
                (decision_id, item_id, action, reason, confidence,
                 item_role, item_summary, promoted_event_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                item_id,
                action,
                reason,
                confidence,
                item_role,
                item_summary,
                promoted_event_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def rd_fetch_all() -> List[dict[str, Any]]:
    """Fetch all routing decisions from the database."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM router_decisions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
