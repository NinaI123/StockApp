"""
db.py — Dual-mode database module for Fantasy Finance.
- Production (Render/Supabase): uses PostgreSQL via DATABASE_URL env var.
- Local dev / fallback:         uses SQLite (local.db) automatically.

Switch behaviour:
  Set USE_SQLITE=1  in .env   → force SQLite always
  Set DATABASE_URL  in .env   → try PostgreSQL; fall back to SQLite on failure
  Neither set                 → SQLite
"""
import os
import sqlite3
import logging
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Detect which backend to use ───────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "")
_FORCE_SQLITE  = os.getenv("USE_SQLITE", "0").strip() == "1"
_SQLITE_PATH   = os.getenv("SQLITE_PATH", "local.db")

_use_postgres = False

if _FORCE_SQLITE or not _DATABASE_URL:
    logger.info("db: Using SQLite (%s)", _SQLITE_PATH)
else:
    # Try importing psycopg2; fall back if missing
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor as _RealDictCursor
        _use_postgres = True
        logger.info("db: PostgreSQL mode — %s", _DATABASE_URL.split("@")[-1])
    except ImportError:
        logger.warning("db: psycopg2 not installed — falling back to SQLite")


# ═══════════════════════════════════════════════════════════════════════════
#  POSTGRES HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _pg_connection():
    return psycopg2.connect(_DATABASE_URL, cursor_factory=_RealDictCursor)


# ═══════════════════════════════════════════════════════════════════════════
#  SQLITE ADAPTER  (wraps sqlite3.Row so callers can do row["col"])
# ═══════════════════════════════════════════════════════════════════════════
class _SQLiteConn:
    """
    Thin wrapper around sqlite3.Connection that:
      - enables row_factory = sqlite3.Row
      - exposes .execute() / .commit() / .rollback() / .close()
      - makes psycopg2 %s placeholders work by converting them to ?
    """
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params=()):
        # Convert psycopg2-style %s to sqlite3 ?
        sql = sql.replace("%s", "?")
        return self._conn.execute(sql, params)

    def cursor(self):
        return _SQLiteCursor(self._conn.cursor())

    def commit(self):   self._conn.commit()
    def rollback(self): self._conn.rollback()
    def close(self):    self._conn.close()


class _SQLiteCursor:
    """Wraps sqlite3 cursor, converts %s → ? and exposes .lastrowid."""
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql: str, params=()):
        sql = sql.replace("%s", "?")
        self._cur.execute(sql, params)
        return self

    def fetchone(self):  return self._cur.fetchone()
    def fetchall(self):  return self._cur.fetchall()

    @property
    def lastrowid(self): return self._cur.lastrowid


def _sqlite_connection():
    return _SQLiteConn(_SQLITE_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════
def get_connection():
    """Return a DB connection (postgres or sqlite)."""
    if _use_postgres:
        try:
            return _pg_connection()
        except Exception as e:
            logger.warning("db: PostgreSQL unreachable (%s) — falling back to SQLite", e)
    return _sqlite_connection()


def get_db():
    """FastAPI dependency: yields a connection, commits on success, closes always."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables. Safe to run multiple times. Works on both backends."""
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Helper: use SERIAL on Postgres, INTEGER PRIMARY KEY on SQLite
        pk = "SERIAL PRIMARY KEY" if _use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        ts = "TIMESTAMP DEFAULT NOW()" if _use_postgres else "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        bl = "BOOLEAN DEFAULT TRUE"  if _use_postgres else "INTEGER DEFAULT 1"
        bf = "BOOLEAN DEFAULT FALSE" if _use_postgres else "INTEGER DEFAULT 0"

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS users (
            id            {pk},
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_pro        {bf},
            created_at    {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS trades (
            id         {pk},
            user_id    INTEGER,
            symbol     TEXT NOT NULL,
            qty        REAL NOT NULL,
            price      REAL NOT NULL,
            trade_date {ts},
            notes      TEXT DEFAULT ''
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS watchlist (
            id         {pk},
            user_id    INTEGER,
            symbol     TEXT NOT NULL,
            added_date {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pw_leagues (
            id           {pk},
            code         TEXT UNIQUE,
            name         TEXT NOT NULL,
            is_public    {bl},
            max_teams    INTEGER DEFAULT 8,
            season_weeks INTEGER DEFAULT 2,
            week_start   TEXT,
            week_end     TEXT,
            created_at   {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pw_teams (
            id          {pk},
            code        TEXT UNIQUE,
            league_id   INTEGER NOT NULL,
            user_id     INTEGER,
            player_name TEXT NOT NULL,
            team_name   TEXT,
            stocks      TEXT NOT NULL,
            created_at  {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pw_matchups (
            id        {pk},
            league_id INTEGER NOT NULL,
            team1_id  INTEGER NOT NULL,
            team2_id  INTEGER NOT NULL,
            week_start TEXT NOT NULL
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pw_results (
            id           {pk},
            matchup_id   INTEGER NOT NULL UNIQUE,
            team1_return REAL,
            team2_return REAL,
            winner_id    INTEGER,
            scored_at    {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pw_messages (
            id          {pk},
            league_id   INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            message     TEXT NOT NULL,
            likes       INTEGER DEFAULT 0,
            created_at  {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id              {pk},
            user_id         INTEGER,
            symbol          TEXT NOT NULL,
            target_price    REAL NOT NULL,
            resolution_date TEXT NOT NULL,
            resolved        {bf},
            outcome         TEXT,
            likes           INTEGER DEFAULT 0,
            created_at      {ts}
        )""")

        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS badges (
            id         {pk},
            user_id    INTEGER NOT NULL,
            badge_type TEXT NOT NULL,
            earned_at  {ts}
        )""")

        conn.commit()
        backend = "PostgreSQL" if _use_postgres else f"SQLite ({_SQLITE_PATH})"
        print(f"✅ Database initialized ({backend})")
    except Exception as e:
        conn.rollback()
        print(f"❌ Database init failed: {e}")
        raise
    finally:
        conn.close()
