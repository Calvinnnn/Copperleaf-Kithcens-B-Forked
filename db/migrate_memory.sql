-- Memory System Migration
-- Adds episodic_events, semantic_facts, and router_decisions tables
-- to the existing copperleaf.db to support the long-term memory subsystem.
-- Run: sqlite3 db/copperleaf.db ".read db/migrate_memory.sql"

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------
-- EPISODIC_EVENTS
-- Stores meaningful agent experiences promoted from short-term
-- memory by the Promote-or-Drop Router.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id            TEXT PRIMARY KEY,
    event_type          TEXT NOT NULL,
    summary             TEXT NOT NULL,
    details_json        TEXT NOT NULL DEFAULT '{}',
    importance_score    REAL NOT NULL DEFAULT 0.5
                            CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    tags_json           TEXT NOT NULL DEFAULT '[]',
    source              TEXT NOT NULL DEFAULT 'agent',
    is_consolidated     INTEGER NOT NULL DEFAULT 0 CHECK (is_consolidated IN (0, 1)),
    consolidated_at     TEXT,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_episodic_event_type   ON episodic_events(event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_events(is_consolidated);
CREATE INDEX IF NOT EXISTS idx_episodic_importance   ON episodic_events(importance_score);

-- ----------------------------------------------------------
-- SEMANTIC_FACTS
-- Long-term consolidated knowledge facts extracted from
-- episodic memory by the Consolidation Engine.
-- Only the consolidation engine may write here.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS semantic_facts (
    fact_id             TEXT PRIMARY KEY,
    subject             TEXT NOT NULL,
    predicate           TEXT NOT NULL,
    value_json          TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'superseded', 'contradicted', 'expired')),
    confidence          REAL NOT NULL DEFAULT 1.0
                            CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_event_ids    TEXT NOT NULL DEFAULT '[]',
    history_json        TEXT NOT NULL DEFAULT '[]',
    valid_until         TEXT,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_facts(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_status  ON semantic_facts(status);

-- ----------------------------------------------------------
-- ROUTER_DECISIONS
-- Audit log of every Promote-or-Drop routing decision made
-- when short-term memory overflows. Graders can query this
-- table directly to inspect forget vs. promote reasoning.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS router_decisions (
    decision_id         TEXT PRIMARY KEY,
    item_id             TEXT NOT NULL,
    action              TEXT NOT NULL CHECK (action IN ('forget', 'promote')),
    reason              TEXT NOT NULL,
    confidence          REAL NOT NULL DEFAULT 1.0,
    item_role           TEXT NOT NULL DEFAULT 'unknown',
    item_summary        TEXT NOT NULL DEFAULT '',
    promoted_event_id   TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_router_action  ON router_decisions(action);
CREATE INDEX IF NOT EXISTS idx_router_created ON router_decisions(created_at);
