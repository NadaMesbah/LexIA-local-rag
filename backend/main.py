"""
RAG Local — Backend API (FastAPI).

Endpoints:
  GET    /                       status + KB stats
  GET    /health                 liveness probe

  POST   /sessions               create a chat session
  GET    /sessions               list sessions (most recent first)
  GET    /sessions/{id}          session + full message history
  PATCH  /sessions/{id}          rename a session
  DELETE /sessions/{id}          delete a session (+ its messages)
  POST   /sessions/{id}/chat     ask a question (streaming NDJSON, with memory)

  POST   /upload                 add a document to the knowledge base
  GET    /documents              list indexed documents
  DELETE /documents/{name}       remove a document (+ its vectors)
  DELETE /reset                  wipe the entire knowledge base

The chat endpoint streams newline-delimited JSON (NDJSON). Each line is one of:
  {"type":"token",   "data":"<text>"}
  {"type":"sources", "data":[ ... ]}
  {"type":"done",    "session_id":"...", "title":"..."}
  {"type":"error",   "data":"<message>"}
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from rag_engine import RAGEngine, SUPPORTED_EXTS

app = FastAPI(title="RAG Local API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise persistence + the (single) RAG engine on startup.
db.init_db()
rag = RAGEngine()


# ─── Models ──────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    message: str


# ─── Status ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    info = rag.get_status()
    info["total_documents"] = len(db.list_documents())
    return info


@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Sessions ──────────────────────────────────────────────────────────────

@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    return db.create_session(req.title)


@app.get("/sessions")
async def list_sessions():
    return {"sessions": db.list_sessions()}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = db.get_active_path(session_id)
    for m in messages:
        if m.get("sources"):
            try:
                m["sources"] = json.loads(m["sources"])
            except Exception:
                m["sources"] = []
    session["messages"] = messages
    return session


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    if not db.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="Session not found or empty title.")
    return {"message": "Session renamed.", "title": req.title}


@app.post("/sessions/{session_id}/autotitle")
async def autotitle_session(session_id: str):
    """Generate a short, LLM-written title from the conversation's first message."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
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
async def select_version(session_id: str, message_id: str):
    """Switch the active branch to a sibling version (ChatGPT-style edit history)."""
    if not db.set_active_version(session_id, message_id):
        raise HTTPException(status_code=404, detail="Message not found.")
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


def _reply_generator(session_id: str, fallback_title: str, parent_user_id: str,
                     user_content: str, history: list[dict]):
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
            ans = "".join(answer_parts).strip()
            if ans:
                db.add_message(session_id, "assistant", ans, parent_id=parent_user_id,
                               sources=json.dumps(sources))
            saved["done"] = True

        try:
            async for kind, payload in rag.stream_chat(user_content, history):
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
async def chat(session_id: str, req: ChatRequest):
    """Session-aware, streaming RAG chat with conversation memory."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    path = db.get_active_path(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in path]
    parent_id = path[-1]["id"] if path else None
    uid = db.add_message(session_id, "user", message, parent_id=parent_id)["id"]

    gen = _reply_generator(session_id, session["title"], uid, message, history)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/sessions/{session_id}/messages/{message_id}/edit")
async def edit_message(session_id: str, message_id: str, req: ChatRequest):
    """
    Edit a user message: create a new sibling branch with the new text and
    regenerate the assistant reply (the old version is preserved as a sibling).
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    target = db.get_message(message_id)
    if not target or target["session_id"] != session_id or target["role"] != "user":
        raise HTTPException(status_code=404, detail="Editable user message not found.")
    content = (req.message or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Empty message.")

    history = _history_before(session_id, message_id)
    uid = db.add_message(session_id, "user", content, parent_id=target["parent_id"])["id"]
    gen = _reply_generator(session_id, session["title"], uid, content, history)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/sessions/{session_id}/messages/{message_id}/regenerate")
async def regenerate_message(session_id: str, message_id: str):
    """
    (Re)generate the assistant reply to a user message. Recovers an interrupted
    answer, or creates a fresh variant (new assistant branch).
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    target = db.get_message(message_id)
    if not target or target["session_id"] != session_id or target["role"] != "user":
        raise HTTPException(status_code=404, detail="User message not found.")

    history = _history_before(session_id, message_id)
    gen = _reply_generator(session_id, session["title"], message_id, target["content"], history)
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ─── Documents ─────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = await rag.ingest_document(tmp_path, original_name=file.filename)
        db.upsert_document(file.filename, result["chunks"])
        return {
            "message": f"Document '{file.filename}' indexed successfully.",
            "chunks_added": result["chunks"],
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/documents")
async def list_documents():
    return {"documents": db.list_documents()}


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    if not db.document_exists(doc_name):
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    rag.delete_document(doc_name)
    db.delete_document(doc_name)
    return {"message": f"Document '{doc_name}' removed from knowledge base."}


@app.delete("/reset")
async def reset_knowledge_base():
    rag.reset()
    db.clear_documents()
    return {"message": "Knowledge base has been reset."}
