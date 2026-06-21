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

# Isolated, throwaway storage + a deterministic admin for tests.
# IMPORTANT: these are FORCE-overridden (not setdefault). When tests run INSIDE
# the backend container, the real SQLITE_PATH/CHROMA_PATH are already set to the
# live data volume — and the test fixtures wipe the DB file. Overriding guarantees
# tests can never touch your real sessions/documents/vectors.
_TMP = tempfile.mkdtemp(prefix="ragtest_")
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "rag.db")
os.environ["CHROMA_PATH"] = os.path.join(_TMP, "chroma")
os.environ["FILES_DIR"] = os.path.join(_TMP, "files")
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_EMAIL"] = "admin@test.local"
os.environ["ADMIN_PASSWORD"] = "testadminpw"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["HYBRID_ENABLED"] = "false"   # unit/integration tests don't need the lexical index

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