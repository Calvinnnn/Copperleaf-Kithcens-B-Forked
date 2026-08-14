"""
init_db.py — Builds copperleaf.db from schema.sql + seed.sql.

Run once before starting the server (or any time you want a clean reset):
    python mcp_server/init_db.py

This is intentionally separate from db.py: db.py only ever OPENS the
database and expects it to already exist. Nothing in the server itself
creates or seeds tables — that's a deliberate separation, so "how does the
demo data get there" is one obvious file, not folded into server startup.
"""
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB_PATH = DB_DIR / "copperleaf.db"
SCHEMA_PATH = DB_DIR / "schema.sql"
SEED_PATH = DB_DIR / "seed.sql"
MEMORY_MIGRATE_PATH = DB_DIR / "migrate_memory.sql"


def build():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.executescript(SEED_PATH.read_text())
        if MEMORY_MIGRATE_PATH.exists():
            conn.executescript(MEMORY_MIGRATE_PATH.read_text())
            print(f"Applied memory migration from {MEMORY_MIGRATE_PATH.name}")
        conn.commit()
        print(f"Built {DB_PATH} from schema.sql + seed.sql + migrate_memory.sql")
    finally:
        conn.close()


if __name__ == "__main__":
    build()
