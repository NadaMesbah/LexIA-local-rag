"""
Persistence layer (SQLite).

Stores everything that must survive a backend restart:
  - chat sessions          (one row per conversation)
  - chat messages          (full conversation history, multi-turn memory)
  - document registry      (which files are indexed + chunk counts)

Why SQLite (and not Postgres)?
  On a single local machine the LLM is the bottleneck (it can only serve
  ~1 generation at a time), so the database never sees heavy concurrent
  writes. SQLite in WAL mode handles this comfortably, with zero extra
  containers / RAM. Everything lives in one file on a Docker volume.

Connections are opened per-call (SQLite is happy with this) so the module
is safe to use from FastAPI's threadpool / run_in_executor workers.
"""

import os
import sqlite3
import uuid
import time
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.getenv("SQLITE_PATH", "/app/data/rag.db")


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def _connect():
    """Open a short-lived connection with sane pragmas."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- 'user' | 'assistant'
                content     TEXT NOT NULL,
                sources     TEXT,                   -- JSON string (assistant only)
                created_at  REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS documents (
                name        TEXT PRIMARY KEY,
                chunks      INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            """
        )


# ─── Sessions ──────────────────────────────────────────────────────────────

def create_session(title: Optional[str] = None) -> dict:
    sid = _new_id()
    ts = _now()
    title = (title or "Nouvelle conversation").strip() or "Nouvelle conversation"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, title, ts, ts),
        )
    return {"id": sid, "title": title, "created_at": ts, "updated_at": ts, "message_count": 0}


def list_sessions() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_session(session_id: str, title: str) -> bool:
    title = (title or "").strip()
    if not title:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title, _now(), session_id),
        )
        return cur.rowcount > 0


def touch_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id)
        )


def delete_session(session_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return cur.rowcount > 0


# ─── Messages ──────────────────────────────────────────────────────────────

def add_message(
    session_id: str, role: str, content: str, sources: Optional[str] = None
) -> dict:
    mid = _new_id()
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, sources, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (mid, session_id, role, content, sources, ts),
        )
        conn.execute(
            "UPDATE sessions SET updated_at=? WHERE id=?", (ts, session_id)
        )
    return {"id": mid, "role": role, "content": content, "created_at": ts}


def get_messages(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, role, content, sources, created_at FROM messages "
            "WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_messages(session_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return int(row["c"]) if row else 0


# ─── Document registry ───────────────────────────────────────────────────────

def upsert_document(name: str, chunks: int) -> None:
    """Insert a document or add chunks to an existing entry."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT chunks FROM documents WHERE name=?", (name,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE documents SET chunks=chunks+? WHERE name=?", (chunks, name)
            )
        else:
            conn.execute(
                "INSERT INTO documents (name, chunks, created_at) VALUES (?,?,?)",
                (name, chunks, _now()),
            )


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, chunks, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def document_exists(name: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE name=?", (name,)
        ).fetchone()
    return row is not None


def delete_document(name: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE name=?", (name,))
        return cur.rowcount > 0


def total_chunks() -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(chunks),0) AS total FROM documents"
        ).fetchone()
    return int(row["total"]) if row else 0


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
