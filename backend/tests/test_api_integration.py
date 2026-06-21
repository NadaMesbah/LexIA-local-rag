"""API integration tests using FastAPI's TestClient.

The heavy RAG engine is replaced with a lightweight fake (no Ollama/Chroma), so
these run fast and offline while still exercising auth, documents, permissions,
chat streaming and the debug endpoint end-to-end. Requires `httpx` (TestClient).
"""

import json
import os

import pytest

# Replace the engine BEFORE importing main (main instantiates `rag = RAGEngine()`).
import rag_engine


class FakeEngine:
    def __init__(self):
        pass

    def get_status(self):
        return {"status": "ready", "doc_count": 0, "collection_name": "test",
                "llm_model": "x", "embed_model": "y", "reranker": None}

    def index_version(self, *a, **k):
        return 1

    def set_document_hidden(self, *a, **k):
        pass

    def restore_version(self, *a, **k):
        pass

    def purge_document(self, *a, **k):
        pass

    def reset(self):
        pass

    def generate_title(self, message):
        return "Titre test"

    def _relevant_nodes(self, message, scope, user):
        return []

    async def stream_chat(self, message, history, scope=None, user=None, strict=None):
        yield ("token", "Réponse de test.")
        yield ("sources", [])


# Patch the engine class ONLY for main's `rag = RAGEngine()` at import, then RESTORE
# it — otherwise the fake leaks into other test modules (e.g. test_retrieval_logic,
# which imports the real RAGEngine for its static methods).
_REAL_ENGINE = rag_engine.RAGEngine
rag_engine.RAGEngine = FakeEngine
import main  # noqa: E402
rag_engine.RAGEngine = _REAL_ENGINE
import db  # noqa: E402
import auth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Fresh DB + seeded admin per test, independent of other test files."""
    for suffix in ("", "-wal", "-shm"):
        p = db.DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)
    db.init_db()
    auth.ensure_default_admin()
    yield


@pytest.fixture
def client():
    return TestClient(main.app)


def _register(client, **over):
    body = {"first_name": "A", "last_name": "B", "username": "alice",
            "email": "alice@x.com", "password": "password123"}
    body.update(over)
    return client.post("/auth/register", json=body)


def _login(client, email, password):
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_register_requires_email_and_login_is_email_based(client):
    assert _register(client, email="not-an-email").status_code == 400
    assert _register(client).status_code == 201
    assert _register(client, username="alice2", email="ALICE@x.com").status_code == 409  # dup
    h = _login(client, "alice@x.com", "password123")
    me = client.get("/auth/me", headers=h)
    assert me.status_code == 200 and me.json()["email"] == "alice@x.com"


def test_debug_endpoint_is_admin_only(client):
    admin = _login(client, "admin@test.local", "testadminpw")
    r = client.post("/debug/retrieve", headers=admin, json={"message": "hello"})
    assert r.status_code == 200 and "items" in r.json()

    _register(client, username="user1", email="u@x.com")
    user = _login(client, "u@x.com", "password123")
    assert client.post("/debug/retrieve", headers=user, json={"message": "hi"}).status_code in (401, 403)


def test_upload_lists_and_isolates_documents(client):
    _register(client, username="alice", email="alice@x.com")
    _register(client, username="bob", email="bob@x.com")
    alice = _login(client, "alice@x.com", "password123")
    bob = _login(client, "bob@x.com", "password123")

    r = client.post("/documents", headers=alice,
                    files={"file": ("mine.txt", b"hello world content", "text/plain")})
    assert r.status_code == 200, r.text

    alice_docs = client.get("/documents", headers=alice).json()["documents"]
    bob_docs = client.get("/documents", headers=bob).json()["documents"]
    assert any(d["name"] == "mine.txt" for d in alice_docs)
    assert not any(d["name"] == "mine.txt" for d in bob_docs)   # isolation


def test_chat_round_trip_persists_answer(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = client.post("/sessions", headers=h, json={}).json()["id"]
    r = client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Bonjour"})
    assert r.status_code == 200
    # The body is NDJSON with JSON-escaped unicode; reassemble the streamed tokens.
    tokens = ""
    for line in r.text.splitlines():
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "token":
            tokens += ev.get("data", "")
    assert "Réponse de test." in tokens

    detail = client.get(f"/sessions/{sid}", headers=h).json()
    roles = [m["role"] for m in detail["messages"]]
    assert "user" in roles and "assistant" in roles
