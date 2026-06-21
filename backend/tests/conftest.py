"""Shared pytest setup.

Isolated, throwaway storage is configured via env vars BEFORE any app module is
imported, so tests never touch your real SQLite/Chroma/files. The app modules
(`db`, `auth`, `main`, `rag_engine`) live in the parent `backend/` directory.
"""

import os
import sys
import tempfile

# Make `import db`, `import main`, ... resolve to backend/*.py
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Isolated storage + a deterministic admin for integration tests.
_TMP = tempfile.mkdtemp(prefix="ragtest_")
os.environ.setdefault("SQLITE_PATH", os.path.join(_TMP, "rag.db"))
os.environ.setdefault("CHROMA_PATH", os.path.join(_TMP, "chroma"))
os.environ.setdefault("FILES_DIR", os.path.join(_TMP, "files"))
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_EMAIL", "admin@test.local")
os.environ.setdefault("ADMIN_PASSWORD", "testadminpw")
os.environ.setdefault("JWT_SECRET", "test-secret")

import pytest  # noqa: E402
import db as _db  # noqa: E402


@pytest.fixture
def fresh_db():
    """A clean database for a single test (wipes file + recreates schema)."""
    for suffix in ("", "-wal", "-shm"):
        p = _db.DB_PATH + suffix
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    _db.init_db()
    return _db
