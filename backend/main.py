"""
RAG Local — Backend API (FastAPI).

Auth & isolation
----------------
All session/document/chat endpoints require a Bearer JWT (see auth.py). Each
resource is owned by a user:
  - sessions are private to their owner (admins may read any).
  - documents have a scope: `org` (admin-managed, visible to everyone) or
    `personal` (visible only to their owner). Identity is (scope, owner, name),
    so two users' same-named files never collide.
Retrieval is filtered, in Chroma, to "org docs + my own docs" per request.
A lightweight audit log records uploads / version / restore / delete actions.

Endpoints (auth required unless noted):
  GET    /                       status + KB stats              (public)
  GET    /health                 liveness probe                 (public)

  POST   /auth/login             obtain a JWT                   (public)
  POST   /auth/register          create a regular user          (public)
  GET    /auth/me                current user
  GET    /users                  list users                     (admin)
  GET    /audit                  recent audit entries           (admin)

  POST   /sessions               create a chat session
  GET    /sessions               list my sessions
  GET    /sessions/{id}          session + full message history
  PATCH  /sessions/{id}          rename a session
  DELETE /sessions/{id}          delete a session (+ its messages)
  POST   /sessions/{id}/chat     ask a question (streaming NDJSON, with memory)

  POST   /documents              add a document (scope by role)
  GET    /documents              list visible documents
  DELETE /documents/{id}         soft-delete a document
  DELETE /reset                  wipe the entire knowledge base (admin)

The chat endpoint streams newline-delimited JSON (NDJSON). Each line is one of:
  {"type":"token",   "data":"<text>"}
  {"type":"sources", "data":[ ... ]}
  {"type":"done",    "session_id":"...", "title":"..."}
  {"type":"error",   "data":"<message>"}
"""

import os
import re
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import db
import auth
from rag_engine import RAGEngine, SUPPORTED_EXTS

app = FastAPI(title="RAG Local API", version="4.0.0")

# CORS: lock down to the configured frontend origin(s) instead of "*".
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload size cap (bytes). 0 / unset disables the check.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

# Original uploaded files are kept here so any version can be downloaded later.
FILES_DIR = os.getenv("FILES_DIR", "/app/data/files")
os.makedirs(FILES_DIR, exist_ok=True)

# Initialise persistence, seed the admin + adopt legacy data, THEN build the
# engine (so its backfill sees the owners assigned by ensure_default_admin).
db.init_db()
auth.ensure_default_admin()
rag = RAGEngine()

# Single-worker pool: document indexing runs in the background, serialized so it
# never thrashes the GPU or races on the vector store.
_indexer = ThreadPoolExecutor(max_workers=1)

# Safety net: strip a trailing "Sources: …" / "Références : …" block the model may
# still append despite the system prompt (the UI shows sources separately).
_INLINE_SOURCES_RE = re.compile(
    r"\n\s*(?:\*\*|__|#+\s*)?\s*(?:sources?|r[ée]f[ée]rences?|citations?)\s*(?:\*\*|__)?\s*[:：][\s\S]*$",
    re.IGNORECASE,
)

def _strip_inline_sources(text: str) -> str:
    return _INLINE_SOURCES_RE.sub("", text).rstrip()


# Pragmatic email validation (used at registration). Good enough for a local app.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ─── Models ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    username: str
    email: str
    password: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    message: str
    scope: Optional[dict] = None   # {"type": "all"} | {"type": "document", "id"/"ids": ...}
    attachments: Optional[list] = None  # [{"id": "...", "filename": "..."}] shown on the user message


# ─── Ownership helpers ───────────────────────────────────────────────────────

def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"


def _owned_session(session_id: str, user: dict) -> dict:
    """Return the session if the user owns it (or is admin), else 404/403."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.get("user_id") != user["id"] and not _is_admin(user):
        raise HTTPException(status_code=403, detail="You do not own this session.")
    return session


def _visible_document(document_id: str, user: dict) -> dict:
    """Return the document if the user may see it (org, own, or admin), else 404."""
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    visible = _is_admin(user) or doc.get("scope") == "org" or doc.get("owner_id") == user["id"]
    if not visible:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


def _manageable_document(document_id: str, user: dict) -> dict:
    """
    Return the document if the user may modify it. Personal docs: only the owner
    (or admin). Org docs: admin only — regular users must not touch the canonical
    org knowledge base.
    """
    doc = _visible_document(document_id, user)
    if _is_admin(user):
        return doc
    if doc.get("scope") == "org":
        raise HTTPException(status_code=403, detail="Org documents are managed by administrators.")
    if doc.get("owner_id") != user["id"]:
        raise HTTPException(status_code=403, detail="You do not own this document.")
    return doc


# ─── Auth ────────────────────────────────────────────────────────────────────

@app.post("/auth/login")
async def login(req: LoginRequest):
    email = (req.email or "").strip()
    user = db.get_user_by_email(email)
    if not user or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    pub = {
        "id": user["id"],
        "username": user["username"],
        "email": user.get("email", ""),
        "first_name": user.get("first_name", ""),
        "last_name": user.get("last_name", ""),
        "role": user["role"],
    }
    return {"access_token": auth.create_access_token(pub), "token_type": "bearer", "user": pub}


@app.post("/auth/register", status_code=201)
async def register(req: RegisterRequest):
    """Create a regular user account. Does NOT log in — the client redirects to
    the sign-in screen afterwards. There is exactly one admin (seeded from env);
    self-registration always creates the `user` role. Login is by email."""
    first_name = req.first_name.strip()
    last_name = req.last_name.strip()
    username = req.username.strip()
    email = (req.email or "").strip()
    password = req.password
    if not first_name or not last_name:
        raise HTTPException(status_code=400, detail="First and last name are required.")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if db.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    db.create_user(username, auth.hash_password(password), role="user",
                   first_name=first_name, last_name=last_name, email=email)
    return {"ok": True, "username": username, "email": email}


@app.get("/auth/me")
async def me(user: dict = Depends(auth.current_user)):
    return user


@app.get("/users")
async def users(user: dict = Depends(auth.admin_user)):
    return {"users": db.list_users()}


@app.get("/audit")
async def audit(user: dict = Depends(auth.admin_user)):
    return {"entries": db.list_audit()}


# ─── Status ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    info = rag.get_status()
    info["total_documents"] = db.total_active_documents()
    return info


@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Sessions ──────────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session(req: CreateSessionRequest, user: dict = Depends(auth.current_user)):
    return db.create_session(user["id"], req.title)


@app.get("/sessions")
async def list_sessions(user: dict = Depends(auth.current_user)):
    return {"sessions": db.list_sessions(user["id"])}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(auth.current_user)):
    session = _owned_session(session_id, user)
    messages = db.get_active_path(session_id)
    for m in messages:
        if m.get("sources"):
            try:
                m["sources"] = json.loads(m["sources"])
            except Exception:
                m["sources"] = []
        if m.get("attachments"):
            try:
                atts = json.loads(m["attachments"])
            except Exception:
                atts = []
            # Enrich each attachment with the attached document's CURRENT indexing
            # status (source of truth), so a re-opened conversation shows the real
            # state instead of whatever was guessed when it was first sent.
            for a in atts:
                if isinstance(a, dict) and a.get("id"):
                    lv = db.latest_version(a["id"])
                    a["indexing_status"] = lv.get("indexing_status") if lv else None
            m["attachments"] = atts
    session["messages"] = messages
    return session


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest,
                         user: dict = Depends(auth.current_user)):
    _owned_session(session_id, user)
    if not db.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="Session not found or empty title.")
    return {"message": "Session renamed.", "title": req.title}


@app.post("/sessions/{session_id}/autotitle")
async def autotitle_session(session_id: str, user: dict = Depends(auth.current_user)):
    """Generate a short, LLM-written title from the conversation's first message."""
    session = _owned_session(session_id, user)
    messages = db.get_active_path(session_id)
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if not first_user:
        return {"title": session["title"]}

    import asyncio
    loop = asyncio.get_event_loop()
    title = await loop.run_in_executor(None, rag.generate_title, first_user["content"])
    db.rename_session(session_id, title)
    return {"title": title}


@app.post("/sessions/{session_id}/messages/{message_id}/select")
async def select_version(session_id: str, message_id: str,
                         user: dict = Depends(auth.current_user)):
    """Switch the active branch to a sibling version (ChatGPT-style edit history)."""
    _owned_session(session_id, user)
    if not db.set_active_version(session_id, message_id):
        raise HTTPException(status_code=404, detail="Message not found.")
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(auth.current_user)):
    _owned_session(session_id, user)
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


def _reply_generator(session_id: str, fallback_title: str, parent_user_id: str,
                     user_content: str, history: list[dict], scope: Optional[dict] = None,
                     user: Optional[dict] = None):
    """
    Stream an assistant reply as a child of an EXISTING user message and persist
    it (the answer survives a client disconnect via the `finally` block). Shared
    by chat, edit and regenerate.
    """
    async def gen():
        answer_parts: list[str] = []
        sources: list[dict] = []
        saved = {"done": False}

        def persist():
            if saved["done"]:
                return
            ans = _strip_inline_sources("".join(answer_parts).strip())
            if ans:
                db.add_message(session_id, "assistant", ans, parent_id=parent_user_id,
                               sources=json.dumps(sources))
            saved["done"] = True

        try:
            async for kind, payload in rag.stream_chat(user_content, history, scope, user):
                if kind == "token":
                    answer_parts.append(payload)
                    yield json.dumps({"type": "token", "data": payload}) + "\n"
                elif kind == "sources":
                    sources = payload
                    yield json.dumps({"type": "sources", "data": payload}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        finally:
            persist()

        current = db.get_session(session_id)
        yield json.dumps({
            "type": "done",
            "session_id": session_id,
            "title": current["title"] if current else fallback_title,
        }) + "\n"

    return gen


def _history_before(session_id: str, message_id: str) -> list[dict]:
    history: list[dict] = []
    for m in db.get_active_path(session_id):
        if m["id"] == message_id:
            break
        history.append({"role": m["role"], "content": m["content"]})
    return history


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest, user: dict = Depends(auth.current_user)):
    """Session-aware, streaming RAG chat with conversation memory."""
    session = _owned_session(session_id, user)
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    path = db.get_active_path(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in path]
    parent_id = path[-1]["id"] if path else None
    att = json.dumps(req.attachments) if req.attachments else None
    uid = db.add_message(session_id, "user", message, parent_id=parent_id, attachments=att)["id"]

    gen = _reply_generator(session_id, session["title"], uid, message, history, req.scope, user)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/sessions/{session_id}/messages/{message_id}/edit")
async def edit_message(session_id: str, message_id: str, req: ChatRequest,
                       user: dict = Depends(auth.current_user)):
    """
    Edit a user message: create a new sibling branch with the new text and
    regenerate the assistant reply (the old version is preserved as a sibling).
    """
    session = _owned_session(session_id, user)
    target = db.get_message(message_id)
    if not target or target["session_id"] != session_id or target["role"] != "user":
        raise HTTPException(status_code=404, detail="Editable user message not found.")
    content = (req.message or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Empty message.")

    history = _history_before(session_id, message_id)
    att = json.dumps(req.attachments) if req.attachments else None
    uid = db.add_message(session_id, "user", content, parent_id=target["parent_id"], attachments=att)["id"]
    gen = _reply_generator(session_id, session["title"], uid, content, history, req.scope, user)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/sessions/{session_id}/messages/{message_id}/regenerate")
async def regenerate_message(session_id: str, message_id: str,
                             user: dict = Depends(auth.current_user)):
    """
    (Re)generate the assistant reply to a user message. Recovers an interrupted
    answer, or creates a fresh variant (new assistant branch).
    """
    session = _owned_session(session_id, user)
    target = db.get_message(message_id)
    if not target or target["session_id"] != session_id or target["role"] != "user":
        raise HTTPException(status_code=404, detail="User message not found.")

    history = _history_before(session_id, message_id)
    gen = _reply_generator(session_id, session["title"], message_id, target["content"],
                           history, None, user)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ─── Documents (versioned) ───────────────────────────────────────────────────

def _save_temp(file: UploadFile, suffix: str) -> str:
    """Stream the upload to a temp file, enforcing the size cap as we go."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if MAX_UPLOAD_BYTES and size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (limit {MAX_UPLOAD_BYTES // (1024*1024)} MB).",
                    )
                out.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


def _index_job(version_id: str, document_id: str, version: int, file_path: str,
               filename: str, owner_id: Optional[str], scope: str):
    """Background: chunk → embed → index a version, then flip it to latest.

    `file_path` is the PERSISTENT stored copy (kept so the version can be
    downloaded later), so it is NOT deleted here."""
    db.set_version_status(version_id, "processing")
    try:
        n = rag.index_version(file_path, filename, document_id, version,
                              owner_id=owner_id, scope=scope)
        db.set_version_status(version_id, "indexed", chunks=n)
        db.mark_version_latest(document_id, version)
    except Exception as e:
        db.set_version_status(version_id, "failed", error=str(e))
        print(f"[API] indexing failed for {filename} v{version}: {e}")


def _store_file(tmp_path: str, version_id: str, suffix: str) -> str:
    """Move a freshly-uploaded temp file to its permanent per-version location."""
    dest = os.path.join(FILES_DIR, f"{version_id}{suffix}")
    shutil.move(tmp_path, dest)
    return dest


def _check_ext(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTS))}",
        )
    return suffix


def _resolve_scope(user: dict, requested: Optional[str]) -> str:
    """
    Admin uploads default to `org` (the canonical KB) but may pick `personal`;
    regular users may only create `personal` documents.
    """
    if _is_admin(user):
        return "personal" if (requested or "").lower() == "personal" else "org"
    return "personal"


@app.post("/documents")
async def create_document(file: UploadFile = File(...), scope: Optional[str] = Form(None),
                          user: dict = Depends(auth.current_user)):
    """Upload a file. Identity is (scope, owner, name) → same key = a new version."""
    suffix = _check_ext(file.filename)
    doc_scope = _resolve_scope(user, scope)
    owner_id = None if doc_scope == "org" else user["id"]
    # Stream to a temp file first so the size cap is enforced before any DB rows
    # or vectors are created.
    tmp = _save_temp(file, suffix)
    doc = db.get_or_create_document(file.filename, scope=doc_scope, owner_id=owner_id)
    ver = db.add_version(doc["id"], file.filename)
    stored = _store_file(tmp, ver["id"], suffix)
    db.set_version_file(ver["id"], stored)
    _indexer.submit(_index_job, ver["id"], doc["id"], ver["version"], stored,
                    file.filename, doc.get("owner_id"), doc.get("scope", doc_scope))
    db.log_audit(user["id"], user["username"], "upload", doc["id"], ver["version"],
                 detail=f"{file.filename} ({doc_scope})")
    return {
        "document_id": doc["id"],
        "version_id": ver["id"],
        "version": ver["version"],
        "scope": doc.get("scope", doc_scope),
        "indexing_status": "pending",
        "filename": file.filename,
    }


@app.post("/documents/{document_id}/versions")
async def add_document_version(document_id: str, file: UploadFile = File(...),
                               user: dict = Depends(auth.current_user)):
    """Explicitly upload a new version for an existing document."""
    doc = _manageable_document(document_id, user)
    if doc["status"] == "deleted":
        raise HTTPException(status_code=404, detail="Document not found.")
    suffix = _check_ext(file.filename)
    tmp = _save_temp(file, suffix)
    ver = db.add_version(document_id, file.filename)
    stored = _store_file(tmp, ver["id"], suffix)
    db.set_version_file(ver["id"], stored)
    _indexer.submit(_index_job, ver["id"], document_id, ver["version"], stored,
                    file.filename, doc.get("owner_id"), doc.get("scope", "org"))
    db.log_audit(user["id"], user["username"], "new_version", document_id, ver["version"],
                 detail=file.filename)
    return {
        "document_id": document_id,
        "version_id": ver["id"],
        "version": ver["version"],
        "indexing_status": "pending",
        "filename": file.filename,
    }


@app.get("/documents")
async def list_documents(user: dict = Depends(auth.current_user)):
    return {"documents": db.list_documents(owner_id=user["id"], role=user["role"])}


@app.get("/documents/trash")
async def trashed_documents(user: dict = Depends(auth.current_user)):
    docs = db.list_documents(include_deleted=True, owner_id=user["id"], role=user["role"])
    return {"documents": [d for d in docs if d["status"] == "deleted"]}


@app.post("/documents/{document_id}/undelete")
async def undelete_document(document_id: str, user: dict = Depends(auth.current_user)):
    """Restore a soft-deleted document from the trash (re-enables its latest version)."""
    doc = _manageable_document(document_id, user)
    db.set_document_status(document_id, "active")
    if doc["current_version"]:
        rag.restore_version(document_id, doc["current_version"])
    db.log_audit(user["id"], user["username"], "undelete", document_id, doc["current_version"])
    return {"ok": True}


@app.get("/documents/{document_id}/versions")
async def document_versions(document_id: str, user: dict = Depends(auth.current_user)):
    _visible_document(document_id, user)
    return {"versions": db.list_versions(document_id)}


def _file_response(ver: Optional[dict]):
    """Return the stored original file for a version, or a helpful 404."""
    if not ver or not ver.get("stored_path") or not os.path.exists(ver["stored_path"]):
        raise HTTPException(
            status_code=404,
            detail="Original file unavailable (this version predates file storage — "
                   "re-upload it to enable download).",
        )
    return FileResponse(ver["stored_path"], filename=ver["filename"],
                        media_type="application/octet-stream")


@app.get("/documents/{document_id}/download")
async def download_document(document_id: str, user: dict = Depends(auth.current_user)):
    """Download the latest version's original file (any visible document)."""
    _visible_document(document_id, user)
    return _file_response(db.latest_version(document_id))


@app.get("/documents/{document_id}/versions/{version}/download")
async def download_version(document_id: str, version: int,
                           user: dict = Depends(auth.current_user)):
    """Download a specific version's original file."""
    _visible_document(document_id, user)
    return _file_response(db.get_version_by_number(document_id, version))


@app.post("/documents/{document_id}/versions/{version}/restore")
async def restore_version(document_id: str, version: int,
                          user: dict = Depends(auth.current_user)):
    """Rollback: make an older version the latest one used for retrieval."""
    _manageable_document(document_id, user)
    if not any(v["version"] == version for v in db.list_versions(document_id)):
        raise HTTPException(status_code=404, detail="Version not found.")
    rag.restore_version(document_id, version)
    db.mark_version_latest(document_id, version)
    db.log_audit(user["id"], user["username"], "restore", document_id, version)
    return {"ok": True, "current_version": version}


@app.delete("/documents/{document_id}")
async def soft_delete_document(document_id: str, user: dict = Depends(auth.current_user)):
    """Soft-delete: hide from retrieval and the active list (vectors kept)."""
    doc = _manageable_document(document_id, user)
    rag.set_document_hidden(document_id)
    db.set_document_status(document_id, "deleted")
    db.log_audit(user["id"], user["username"], "soft_delete", document_id, doc["current_version"])
    return {"ok": True}


@app.delete("/documents/{document_id}/purge")
async def purge_document(document_id: str, user: dict = Depends(auth.current_user)):
    """Hard-delete: permanently remove vectors + records + stored files."""
    doc = _manageable_document(document_id, user)
    stored = db.document_stored_paths(document_id)
    rag.purge_document(document_id)
    db.hard_delete_document(document_id)
    for p in stored:
        try:
            os.unlink(p)
        except Exception:
            pass
    db.log_audit(user["id"], user["username"], "purge", document_id, doc["current_version"])
    return {"ok": True}


@app.delete("/reset")
async def reset_knowledge_base(user: dict = Depends(auth.admin_user)):
    rag.reset()
    db.clear_documents()
    # Remove all stored original files.
    for name in os.listdir(FILES_DIR):
        try:
            os.unlink(os.path.join(FILES_DIR, name))
        except Exception:
            pass
    db.log_audit(user["id"], user["username"], "reset_kb")
    return {"message": "Knowledge base has been reset."}