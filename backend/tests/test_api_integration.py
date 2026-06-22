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
        # Records the args of the most recent stream_chat call so tests can assert
        # what the chat / edit / regenerate endpoints forwarded (scope + strict).
        self.last_call = None

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

    def _relevant_nodes(self, message, scope=None, user=None, debug=False):
        # Mirrors the real engine's contract: a plain list of nodes normally, but a
        # diagnostics dict (summary + items) when debug=True (used by /debug/retrieve).
        if debug:
            return {"summary": {"retrieved": 0, "kept": 0, "strict_would_answer": False},
                    "items": []}
        return []

    async def stream_chat(self, message, history, scope=None, user=None, strict=None):
        self.last_call = {"message": message, "history": history, "scope": scope,
                          "user": user, "strict": strict}
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
    assert r.status_code == 200
    body = r.json()
    # Diagnostics shape returned by /debug/retrieve (built from the engine's
    # debug=True diagnostics object).
    assert {"query", "strict_default", "summary", "count", "items"}.issubset(body)
    assert body["count"] == len(body["items"]) and isinstance(body["summary"], dict)

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



# ─── Edit / regenerate preserve strict + scope (vs a normal send) ─────────────
# These assert what the endpoints FORWARD to the engine (FakeEngine.last_call),
# which is exactly what the production fix guarantees: edit & regenerate use the
# same effective strict value and active document scope as a normal send.

def _new_session(client, h):
    return client.post("/sessions", headers=h, json={}).json()["id"]


def _first_user_message_id(client, h, sid):
    msgs = client.get(f"/sessions/{sid}", headers=h).json()["messages"]
    return next(m["id"] for m in msgs if m["role"] == "user")


def test_edit_preserves_strict_and_scope(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Question 1"})
    uid = _first_user_message_id(client, h, sid)

    r = client.post(f"/sessions/{sid}/messages/{uid}/edit", headers=h,
                    json={"message": "Question éditée", "strict": True,
                          "scope": {"type": "document", "ids": ["docA"]}})
    assert r.status_code == 200
    call = main.rag.last_call
    assert call["message"] == "Question éditée"
    assert call["strict"] is True                                   # strict preserved
    assert call["scope"] == {"type": "document", "ids": ["docA"]}    # active scope preserved


def test_edit_global_strict_when_no_scope_selected(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Q"})
    uid = _first_user_message_id(client, h, sid)

    # Documents uniquement ON but no document selected -> global strict RAG (no scope).
    r = client.post(f"/sessions/{sid}/messages/{uid}/edit", headers=h,
                    json={"message": "Q2", "strict": True})
    assert r.status_code == 200
    assert main.rag.last_call["strict"] is True
    assert main.rag.last_call["scope"] is None      # global RAG, not a forced doc scope


def test_edit_general_mode_when_strict_off(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Q"})
    uid = _first_user_message_id(client, h, sid)

    # Documents uniquement OFF -> the client omits strict; the endpoint must not
    # force documents-only mode (strict stays None -> server default applies).
    r = client.post(f"/sessions/{sid}/messages/{uid}/edit", headers=h, json={"message": "Q2"})
    assert r.status_code == 200
    assert main.rag.last_call["strict"] is None
    assert main.rag.last_call["scope"] is None


def test_regenerate_preserves_explicit_strict_and_scope(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Q"})
    uid = _first_user_message_id(client, h, sid)

    r = client.post(f"/sessions/{sid}/messages/{uid}/regenerate", headers=h,
                    json={"strict": True, "scope": {"type": "document", "ids": ["docB"]}})
    assert r.status_code == 200
    call = main.rag.last_call
    assert call["message"] == "Q"                                   # same question reused
    assert call["strict"] is True
    assert call["scope"] == {"type": "document", "ids": ["docB"]}


def test_regenerate_reconstructs_scope_from_attachments_when_body_empty(client):
    """Backward-compatible fallback: with no scope in the body, regenerate rebuilds
    the document scope from the original message's attachments."""
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h,
                json={"message": "Q", "attachments": [{"id": "docX", "filename": "x.pdf"}]})
    uid = _first_user_message_id(client, h, sid)

    r = client.post(f"/sessions/{sid}/messages/{uid}/regenerate", headers=h, json={})
    assert r.status_code == 200
    assert main.rag.last_call["scope"] == {"type": "document", "ids": ["docX"]}


def test_regenerate_empty_body_no_attachments_is_global(client):
    h = _login(client, "admin@test.local", "testadminpw")
    sid = _new_session(client, h)
    client.post(f"/sessions/{sid}/chat", headers=h, json={"message": "Q"})
    uid = _first_user_message_id(client, h, sid)

    r = client.post(f"/sessions/{sid}/messages/{uid}/regenerate", headers=h, json={})
    assert r.status_code == 200
    assert main.rag.last_call["scope"] is None       # global RAG, no fake doc scope