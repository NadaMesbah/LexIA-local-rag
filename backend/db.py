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
                attachments     TEXT,                   -- JSON list of {id, filename} (user only)
                created_at      REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """
        )
        # Migrate older databases that predate the tree columns. This MUST run
        # before creating any index that references the new columns.
        scols = _columns(conn, "sessions")
        if "active_root_id" not in scols:
            conn.execute("ALTER TABLE sessions ADD COLUMN active_root_id TEXT")
        mcols = _columns(conn, "messages")
        added_parent = "parent_id" not in mcols
        if added_parent:
            conn.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        if "active_child_id" not in mcols:
            conn.execute("ALTER TABLE messages ADD COLUMN active_child_id TEXT")
        # Files attached to a user message in the composer (JSON list of
        # {id, filename}); used to display the attachment in the conversation and
        # to ground that turn's answer on those documents.
        if "attachments" not in mcols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")

        # Legacy flat conversations: link each session's messages into a single
        # chain (preserve order) so they display correctly under the tree model.
        if added_parent:
            for srow in conn.execute("SELECT id FROM sessions").fetchall():
                sid = srow["id"]
                ids = [r["id"] for r in conn.execute(
                    "SELECT id FROM messages WHERE session_id=? ORDER BY created_at ASC", (sid,)
                ).fetchall()]
                prev = None
                for mid in ids:
                    conn.execute("UPDATE messages SET parent_id=? WHERE id=?", (prev, mid))
                    if prev is not None:
                        conn.execute("UPDATE messages SET active_child_id=? WHERE id=?", (mid, prev))
                    prev = mid
                if ids:
                    conn.execute("UPDATE sessions SET active_root_id=? WHERE id=?", (ids[0], sid))

        # Indexes (safe now that the columns are guaranteed to exist).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)")

        # ── Documents: versioned model (migrate legacy flat table if present) ──
        doc_cols = _columns(conn, "documents")
        legacy_docs = bool(doc_cols) and ("id" not in doc_cols)
        legacy_rows: list[dict] = []
        if legacy_docs:
            legacy_rows = [dict(r) for r in conn.execute(
                "SELECT name, chunks, created_at FROM documents"
            ).fetchall()]
            conn.execute("ALTER TABLE documents RENAME TO documents_legacy")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                owner_id        TEXT,                             -- NULL for org-wide docs
                scope           TEXT NOT NULL DEFAULT 'org',      -- org | personal
                current_version INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'active',   -- active | deleted
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_versions (
                id              TEXT PRIMARY KEY,
                document_id     TEXT NOT NULL,
                version         INTEGER NOT NULL,
                filename        TEXT NOT NULL,
                indexing_status TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|indexed|failed
                is_latest       INTEGER NOT NULL DEFAULT 0,
                chunks          INTEGER NOT NULL DEFAULT 0,
                error           TEXT,
                uploaded_at     REAL NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_versions_doc ON document_versions(document_id);
            """
        )

        if legacy_docs:
            for r in legacy_rows:
                did = _new_id()
                ts = r.get("created_at") or _now()
                conn.execute(
                    "INSERT INTO documents (id,name,current_version,status,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (did, r["name"], 1, "active", ts, ts),
                )
                conn.execute(
                    "INSERT INTO document_versions (id,document_id,version,filename,indexing_status,"
                    "is_latest,chunks,error,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (_new_id(), did, 1, r["name"], "indexed", 1, r.get("chunks") or 0, None, ts),
                )
            conn.execute("DROP TABLE documents_legacy")

        # ── Auth / ownership (users, audit, and owner columns) ─────────────────
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT NOT NULL DEFAULT '',       -- login identifier
                password_hash TEXT NOT NULL,
                first_name    TEXT NOT NULL DEFAULT '',
                last_name     TEXT NOT NULL DEFAULT '',
                role          TEXT NOT NULL DEFAULT 'user',   -- admin | user
                created_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id          TEXT PRIMARY KEY,
                user_id     TEXT,
                username    TEXT,
                action      TEXT NOT NULL,
                document_id TEXT,
                version     INTEGER,
                detail      TEXT,
                created_at  REAL NOT NULL
            );
            """
        )
        if "user_id" not in _columns(conn, "sessions"):
            conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
        dcols = _columns(conn, "documents")
        if "owner_id" not in dcols:
            conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")
        if "scope" not in dcols:
            conn.execute("ALTER TABLE documents ADD COLUMN scope TEXT NOT NULL DEFAULT 'org'")
        # Who uploaded the document (user id). Org docs have a NULL owner_id, so
        # this is the only record of the uploader for the canonical KB.
        if "uploaded_by" not in dcols:
            conn.execute("ALTER TABLE documents ADD COLUMN uploaded_by TEXT")
        # Names on users (migrate older auth schema that only had username).
        ucols = _columns(conn, "users")
        if "first_name" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN first_name TEXT NOT NULL DEFAULT ''")
        if "last_name" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN last_name TEXT NOT NULL DEFAULT ''")
        # Email login identifier (migrate older auth schema that only had username).
        if "email" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
        # Case-insensitive uniqueness for non-empty emails (partial unique index
        # so legacy rows with an empty email don't collide).
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email "
            "ON users(email COLLATE NOCASE) WHERE email <> ''"
        )
        # Persisted original file path (so versions can be downloaded later).
        if "stored_path" not in _columns(conn, "document_versions"):
            conn.execute("ALTER TABLE document_versions ADD COLUMN stored_path TEXT")


# ─── Users & auth ────────────────────────────────────────────────────────────

def create_user(username: str, password_hash: str, role: str = "user",
                first_name: str = "", last_name: str = "", email: str = "") -> dict:
    uid = _new_id()
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash, first_name, last_name, role, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (uid, username, email, password_hash, first_name, last_name, role, ts),
        )
    return {"id": uid, "username": username, "email": email, "first_name": first_name,
            "last_name": last_name, "role": role, "created_at": ts}


def get_user(user_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email, case-insensitively (empty email never matches)."""
    email = (email or "").strip()
    if not email:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email <> '' AND email = ? COLLATE NOCASE", (email,)
        ).fetchone()
    return dict(row) if row else None


def set_user_email(user_id: str, email: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET email=? WHERE id=?", (email, user_id))


def update_user(user_id: str, first_name: Optional[str] = None, last_name: Optional[str] = None,
                username: Optional[str] = None, email: Optional[str] = None,
                role: Optional[str] = None, password_hash: Optional[str] = None) -> Optional[dict]:
    """Update the provided fields of a user (only non-None values are changed)."""
    fields: list[str] = []
    params: list = []
    if first_name is not None:
        fields.append("first_name=?"); params.append(first_name)
    if last_name is not None:
        fields.append("last_name=?"); params.append(last_name)
    if username is not None:
        fields.append("username=?"); params.append(username)
    if email is not None:
        fields.append("email=?"); params.append(email)
    if role is not None:
        fields.append("role=?"); params.append(role)
    if password_hash is not None:
        fields.append("password_hash=?"); params.append(password_hash)
    if fields:
        params.append(user_id)
        with _connect() as conn:
            conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id=?", params)
    return get_user(user_id)


def delete_user(user_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        return cur.rowcount > 0


def count_admins() -> int:
    """Number of admin accounts — used to protect the last administrator."""
    with _connect() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role='admin'"
        ).fetchone()["c"])


def delete_user_sessions(user_id: str) -> int:
    """Delete all chat sessions owned by a user (messages cascade)."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        return cur.rowcount


def user_personal_document_ids(owner_id: str) -> list[str]:
    """All personal document ids owned by a user (for cleanup on account deletion)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id FROM documents WHERE scope='personal' AND owner_id=?", (owner_id,)
        ).fetchall()
    return [r["id"] for r in rows]


def list_users() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, username, email, first_name, last_name, role, created_at "
            "FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def count_users() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"])


def log_audit(user_id: Optional[str], username: Optional[str], action: str,
              document_id: Optional[str] = None, version: Optional[int] = None,
              detail: Optional[str] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (id,user_id,username,action,document_id,version,detail,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_new_id(), user_id, username, action, document_id, version, detail, _now()),
        )


def list_audit(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def assign_legacy_owner(admin_id: str) -> None:
    """Give pre-auth sessions/documents an owner so they remain accessible."""
    with _connect() as conn:
        conn.execute("UPDATE sessions SET user_id=? WHERE user_id IS NULL", (admin_id,))
        conn.execute(
            "UPDATE documents SET owner_id=?, scope='org' WHERE owner_id IS NULL", (admin_id,)
        )
        # Attribute documents with no recorded uploader to the admin (they were
        # imported before this column existed).
        conn.execute(
            "UPDATE documents SET uploaded_by=? WHERE uploaded_by IS NULL", (admin_id,)
        )


# ─── Sessions ──────────────────────────────────────────────────────────────

def create_session(user_id: str, title: Optional[str] = None) -> dict:
    sid = _new_id()
    ts = _now()
    title = (title or "Nouvelle conversation").strip() or "Nouvelle conversation"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, active_root_id, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (sid, user_id, title, None, ts, ts),
        )
    return {"id": sid, "user_id": user_id, "title": title, "created_at": ts, "updated_at": ts, "message_count": 0}


def list_sessions(user_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, user_id, title, active_root_id, created_at, updated_at "
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
    attachments: Optional[str] = None,
) -> dict:
    """Insert a message as a child of `parent_id` and make it the active branch."""
    mid = _new_id()
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, parent_id, active_child_id, role, "
            "content, sources, attachments, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, session_id, parent_id, None, role, content, sources, attachments, ts),
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
            "parent_id": cur["parent_id"],
            "role": cur["role"],
            "content": cur["content"],
            "sources": cur["sources"],
            "attachments": cur["attachments"] if "attachments" in cur.keys() else None,
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


# ─── Documents (versioned) ───────────────────────────────────────────────────

def get_or_create_document(name: str, scope: str = "org", owner_id: Optional[str] = None,
                           uploaded_by: Optional[str] = None) -> dict:
    """
    Return the active document identified by (scope, owner, name), creating it
    (version 0) if needed. Org docs are keyed by name globally; personal docs are
    keyed by (owner_id, name) so two users' same-named files never collide.
    """
    ts = _now()
    with _connect() as conn:
        if scope == "personal":
            row = conn.execute(
                "SELECT * FROM documents WHERE name=? AND scope='personal' AND owner_id=? "
                "AND status!='deleted'",
                (name, owner_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM documents WHERE name=? AND scope='org' AND status!='deleted'",
                (name,),
            ).fetchone()
        if row:
            return dict(row)
        did = _new_id()
        conn.execute(
            "INSERT INTO documents (id,name,owner_id,scope,uploaded_by,current_version,status,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (did, name, owner_id, scope, uploaded_by, 0, "active", ts, ts),
        )
        return {"id": did, "name": name, "owner_id": owner_id, "scope": scope,
                "uploaded_by": uploaded_by, "current_version": 0, "status": "active",
                "created_at": ts, "updated_at": ts}


def add_version(document_id: str, filename: str, stored_path: Optional[str] = None) -> dict:
    """Create a new pending version for a document. Returns {id, version}."""
    ts = _now()
    with _connect() as conn:
        mx = conn.execute(
            "SELECT COALESCE(MAX(version),0) AS mx FROM document_versions WHERE document_id=?",
            (document_id,),
        ).fetchone()["mx"]
        version = int(mx) + 1
        vid = _new_id()
        conn.execute(
            "INSERT INTO document_versions (id,document_id,version,filename,indexing_status,"
            "is_latest,chunks,error,uploaded_at,stored_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (vid, document_id, version, filename, "pending", 0, 0, None, ts, stored_path),
        )
        conn.execute("UPDATE documents SET updated_at=?, status='active' WHERE id=?", (ts, document_id))
    return {"id": vid, "document_id": document_id, "version": version}


def set_version_file(version_id: str, stored_path: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE document_versions SET stored_path=? WHERE id=?", (stored_path, version_id)
        )


def document_stored_paths(document_id: str) -> list[str]:
    """All on-disk file paths for a document's versions (for cleanup on purge)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT stored_path FROM document_versions WHERE document_id=? AND stored_path IS NOT NULL",
            (document_id,),
        ).fetchall()
    return [r["stored_path"] for r in rows]


def set_version_status(version_id: str, status: str, chunks: Optional[int] = None,
                       error: Optional[str] = None) -> None:
    with _connect() as conn:
        if chunks is not None:
            conn.execute(
                "UPDATE document_versions SET indexing_status=?, chunks=?, error=? WHERE id=?",
                (status, chunks, error, version_id),
            )
        else:
            conn.execute(
                "UPDATE document_versions SET indexing_status=?, error=? WHERE id=?",
                (status, error, version_id),
            )


def mark_version_latest(document_id: str, version: int) -> None:
    """Make `version` the only is_latest version and the document's current_version."""
    with _connect() as conn:
        conn.execute("UPDATE document_versions SET is_latest=0 WHERE document_id=?", (document_id,))
        conn.execute(
            "UPDATE document_versions SET is_latest=1 WHERE document_id=? AND version=?",
            (document_id, version),
        )
        conn.execute(
            "UPDATE documents SET current_version=?, status='active', updated_at=? WHERE id=?",
            (version, _now(), document_id),
        )


def get_document(document_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return dict(row) if row else None


def find_document_id(name: str, scope: str, owner_id: Optional[str]) -> Optional[str]:
    """Resolve a document id from (name, scope, owner) — used to repair Chroma
    chunks whose document key was lost. Prefers an active, most-recent match."""
    if not name:
        return None
    with _connect() as conn:
        if scope == "personal":
            row = conn.execute(
                "SELECT id FROM documents WHERE name=? AND scope='personal' AND owner_id=? "
                "ORDER BY (status!='deleted') DESC, updated_at DESC LIMIT 1",
                (name, owner_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM documents WHERE name=? AND scope='org' "
                "ORDER BY (status!='deleted') DESC, updated_at DESC LIMIT 1",
                (name,),
            ).fetchone()
    return row["id"] if row else None


def get_version(version_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM document_versions WHERE id=?", (version_id,)).fetchone()
    return dict(row) if row else None


def get_version_by_number(document_id: str, version: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_versions WHERE document_id=? AND version=?",
            (document_id, version),
        ).fetchone()
    return dict(row) if row else None


def latest_version(document_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM document_versions WHERE document_id=? "
            "ORDER BY is_latest DESC, version DESC LIMIT 1",
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def list_versions(document_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM document_versions WHERE document_id=? ORDER BY version DESC",
            (document_id,),
        ).fetchall()
    return [_public_version(dict(r)) for r in rows]


def _public_version(v: Optional[dict]) -> Optional[dict]:
    """Annotate a version row with a `downloadable` flag and drop the internal
    server file path so it is never exposed to clients. A version is downloadable
    only when its original file was stored (versions that predate file storage
    have a NULL `stored_path`)."""
    if v is None:
        return None
    v = dict(v)
    v["downloadable"] = bool(v.get("stored_path"))
    v.pop("stored_path", None)
    return v


def list_documents(include_deleted: bool = False, owner_id: Optional[str] = None,
                   role: Optional[str] = None) -> list[dict]:
    """
    One row per document (latest version + count). Visibility: admins see all;
    everyone else sees org docs + their own personal docs.
    """
    where = ["1=1"]
    params: list = []
    if not include_deleted:
        where.append("status!='deleted'")
    if role != "admin":
        where.append("(scope='org' OR owner_id=?)")
        params.append(owner_id)
    with _connect() as conn:
        docs = [dict(r) for r in conn.execute(
            f"SELECT * FROM documents WHERE {' AND '.join(where)} ORDER BY updated_at DESC", params
        ).fetchall()]
        for d in docs:
            v = conn.execute(
                "SELECT * FROM document_versions WHERE document_id=? "
                "ORDER BY is_latest DESC, version DESC LIMIT 1",
                (d["id"],),
            ).fetchone()
            c = conn.execute(
                "SELECT COUNT(*) AS c FROM document_versions WHERE document_id=?", (d["id"],)
            ).fetchone()["c"]
            d["latest"] = _public_version(dict(v)) if v else None
            d["version_count"] = int(c)
            # Resolve the uploader's display name (full name, else username).
            d["uploaded_by_name"] = None
            uid = d.get("uploaded_by")
            if uid:
                u = conn.execute(
                    "SELECT first_name, last_name, username FROM users WHERE id=?", (uid,)
                ).fetchone()
                if u:
                    full = f"{u['first_name']} {u['last_name']}".strip()
                    d["uploaded_by_name"] = full or u["username"]
    return docs


def active_documents(owner_id: Optional[str] = None, role: Optional[str] = None) -> list[dict]:
    """Lightweight visible-documents list for the chat scope selector."""
    where = ["status='active'"]
    params: list = []
    if role != "admin":
        where.append("(scope='org' OR owner_id=?)")
        params.append(owner_id)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id, name, scope, owner_id, current_version FROM documents "
            f"WHERE {' AND '.join(where)} ORDER BY name", params
        ).fetchall()
    return [dict(r) for r in rows]


def set_document_status(document_id: str, status: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE documents SET status=?, updated_at=? WHERE id=?", (status, _now(), document_id)
        )
        return cur.rowcount > 0


def hard_delete_document(document_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM document_versions WHERE document_id=?", (document_id,))
        cur = conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return cur.rowcount > 0


def total_active_documents() -> int:
    with _connect() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE status='active'"
        ).fetchone()["c"])


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM document_versions")
        conn.execute("DELETE FROM documents")


def ensure_legacy_document(name: str, chunks: int) -> dict:
    """For startup backfill: return {document_id, version, owner_id, scope} for a
    file, creating org-scoped rows if missing."""
    ts = _now()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE name=?", (name,)).fetchone()
        if row:
            did = row["id"]
            owner_id = row["owner_id"] if "owner_id" in row.keys() else None
            scope = row["scope"] if "scope" in row.keys() else "org"
            v = conn.execute(
                "SELECT version FROM document_versions WHERE document_id=? AND is_latest=1",
                (did,),
            ).fetchone()
            if v:
                return {"document_id": did, "version": int(v["version"]), "owner_id": owner_id, "scope": scope}
            conn.execute(
                "INSERT INTO document_versions (id,document_id,version,filename,indexing_status,"
                "is_latest,chunks,error,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (_new_id(), did, 1, name, "indexed", 1, chunks, None, ts),
            )
            conn.execute("UPDATE documents SET current_version=1 WHERE id=?", (did,))
            return {"document_id": did, "version": 1, "owner_id": owner_id, "scope": scope}
        did = _new_id()
        conn.execute(
            "INSERT INTO documents (id,name,owner_id,scope,current_version,status,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (did, name, None, "org", 1, "active", ts, ts),
        )
        conn.execute(
            "INSERT INTO document_versions (id,document_id,version,filename,indexing_status,"
            "is_latest,chunks,error,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (_new_id(), did, 1, name, "indexed", 1, chunks, None, ts),
        )
        return {"document_id": did, "version": 1, "owner_id": None, "scope": "org"}