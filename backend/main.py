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
    messages = db.get_messages(session_id)
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
    messages = db.get_messages(session_id)
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if not first_user:
        return {"title": session["title"]}

    import asyncio
    loop = asyncio.get_event_loop()
    title = await loop.run_in_executor(None, rag.generate_title, first_user["content"])
    db.rename_session(session_id, title)
    return {"title": title}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted."}


def _derive_title(text: str) -> str:
    text = " ".join(text.strip().split())
    return (text[:48] + "…") if len(text) > 48 else (text or "Nouvelle conversation")


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    """Session-aware, streaming RAG chat with conversation memory."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")

    # History BEFORE this turn (multi-turn memory).
    prior = db.get_messages(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in prior]

    # Auto-name a brand-new conversation from its first message.
    is_first = len(prior) == 0

    # Persist the user message immediately.
    db.add_message(session_id, "user", message)
    if is_first:
        db.rename_session(session_id, _derive_title(message))

    async def ndjson_stream():
        answer_parts: list[str] = []
        sources: list[dict] = []
        saved = {"done": False}

        def persist():
            # Save the assistant message exactly once. Called both on normal
            # completion AND in `finally`, so the answer is preserved even if
            # the client disconnects / switches conversation mid-stream.
            if saved["done"]:
                return
            ans = "".join(answer_parts).strip()
            if ans:
                db.add_message(session_id, "assistant", ans, json.dumps(sources))
            saved["done"] = True

        try:
            async for kind, payload in rag.stream_chat(message, history):
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
        yield json.dumps(
            {
                "type": "done",
                "session_id": session_id,
                "title": current["title"] if current else session["title"],
            }
        ) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")


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
