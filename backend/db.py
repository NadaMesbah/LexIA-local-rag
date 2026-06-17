"""
Persistence layer (SQLite).

Stores everything that must survive a backend restart:
  - chat sessions
  - chat messages          (a TREE: each message has a parent; editing a user
                            message creates a sibling branch, ChatGPT-style)
  - document registry

Conversation tree
-----------------
Messages form a tree. The "active path" (what is displayed) is found by walking
from the active root, following each node's `active_child_id` (falling back to
the most recent child). Editing a user message adds a sibling under the same
parent and switches the active pointer to it; switching versions just moves the
active pointer, which restores that branch's whole downstream.

Why SQLite: on a single local machine the LLM is the bottleneck, so the DB never
sees heavy concurrent writes. WAL mode handles it with zero extra containers.
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


def _columns(conn, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db() -> None:
    """Create tables if needed and migrate older schemas in place."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                active_root_id  TEXT,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                parent_id       TEXT,                   -- NULL for root messages
                active_child_id TEXT,                   -- selected branch child
                role            TEXT NOT NULL,          -- 'user' | 'assistant'
                content         TEXT NOT NULL,
                sources         TEXT,                   -- JSON (assistant only)
                created_at      REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS documents (
                name        TEXT PRIMARY KEY,
                chunks      INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_parent  ON messages(parent_id);
            """
        )
        # Migrate older databases that predate the tree columns.
        scols = _columns(conn, "sessions")
        if "active_root_id" not in scols:
            conn.execute("ALTER TABLE sessions ADD COLUMN active_root_id TEXT")
        mcols = _columns(conn, "messages")
        if "parent_id" not in mcols:
            conn.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        if "active_child_id" not in mcols:
            conn.execute("ALTER TABLE messages ADD COLUMN active_child_id TEXT")


# ─── Sessions ──────────────────────────────────────────────────────────────

def create_session(title: Optional[str] = None) -> dict:
    sid = _new_id()
    ts = _now()
    title = (title or "Nouvelle conversation").strip() or "Nouvelle conversation"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, active_root_id, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (sid, title, None, ts, ts),
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
            "SELECT id, title, active_root_id, created_at, updated_at "
            "FROM sessions WHERE id=?",
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


def delete_session(session_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return cur.rowcount > 0


# ─── Messages (tree) ─────────────────────────────────────────────────────────

def add_message(
    session_id: str,
    role: str,
    content: str,
    parent_id: Optional[str] = None,
    sources: Optional[str] = None,
) -> dict:
    """Insert a message as a child of `parent_id` and make it the active branch."""
    mid = _new_id()
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, parent_id, active_child_id, role, "
            "content, sources, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (mid, session_id, parent_id, None, role, content, sources, ts),
        )
        # Make this new message the selected branch.
        if parent_id:
            conn.execute(
                "UPDATE messages SET active_child_id=? WHERE id=?", (mid, parent_id)
            )
        else:
            conn.execute(
                "UPDATE sessions SET active_root_id=? WHERE id=?", (mid, session_id)
            )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, session_id))
    return {"id": mid, "role": role, "content": content, "created_at": ts}


def get_message(message_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
    return dict(row) if row else None


def set_active_version(session_id: str, message_id: str) -> bool:
    """Select `message_id` among its siblings (switches the displayed branch)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT parent_id FROM messages WHERE id=? AND session_id=?",
            (message_id, session_id),
        ).fetchone()
        if not row:
            return False
        if row["parent_id"]:
            conn.execute(
                "UPDATE messages SET active_child_id=? WHERE id=?",
                (message_id, row["parent_id"]),
            )
        else:
            conn.execute(
                "UPDATE sessions SET active_root_id=? WHERE id=?",
                (message_id, session_id),
            )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))
    return True


def get_active_path(session_id: str) -> list[dict]:
    """Return the messages on the active branch, with version metadata."""
    sess = get_session(session_id)
    if not sess:
        return []
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM messages WHERE session_id=?", (session_id,)
        ).fetchall()]

    by_id = {m["id"]: m for m in rows}
    children: dict[Optional[str], list[dict]] = {}
    roots: list[dict] = []
    for m in rows:
        if m["parent_id"]:
            children.setdefault(m["parent_id"], []).append(m)
        else:
            roots.append(m)

    def ordered(lst):
        return sorted(lst, key=lambda x: x["created_at"])

    roots = ordered(roots)
    path: list[dict] = []

    # Pick the active root.
    cur = None
    ar = sess.get("active_root_id")
    if ar and ar in by_id:
        cur = by_id[ar]
    elif roots:
        cur = roots[-1]

    while cur:
        sibs = ordered(children[cur["parent_id"]]) if cur["parent_id"] else roots
        sib_ids = [s["id"] for s in sibs]
        path.append({
            "id": cur["id"],
            "role": cur["role"],
            "content": cur["content"],
            "sources": cur["sources"],
            "version_index": sib_ids.index(cur["id"]) + 1,
            "version_count": len(sib_ids),
            "sibling_ids": sib_ids,
        })
        kids = ordered(children.get(cur["id"], []))
        if not kids:
            break
        nxt = cur.get("active_child_id")
        cur = by_id[nxt] if (nxt and nxt in by_id) else kids[-1]

    return path


def get_history(session_id: str) -> list[dict]:
    """Active path as a flat [{role, content}] list (for the LLM memory)."""
    return [{"role": m["role"], "content": m["content"]} for m in get_active_path(session_id)]


def count_messages(session_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
    return int(row["c"]) if row else 0


# ─── Document registry ───────────────────────────────────────────────────────

def upsert_document(name: str, chunks: int) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT chunks FROM documents WHERE name=?", (name,)).fetchone()
        if row:
            conn.execute("UPDATE documents SET chunks=chunks+? WHERE name=?", (chunks, name))
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
        return conn.execute("SELECT 1 FROM documents WHERE name=?", (name,)).fetchone() is not None


def delete_document(name: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE name=?", (name,))
        return cur.rowcount > 0


def total_chunks() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COALESCE(SUM(chunks),0) AS total FROM documents").fetchone()
    return int(row["total"]) if row else 0


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")