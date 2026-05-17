"""
RAG Local - Backend API
FastAPI server for the persistent knowledge base RAG system.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import RAGEngine

app = FastAPI(title="RAG Local API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global RAG engine instance — knowledge base persists in Milvus
rag = RAGEngine()


# ─── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

class StatusResponse(BaseModel):
    status: str
    doc_count: int
    collection_name: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
async def root():
    """Health check + knowledge base status."""
    info = rag.get_status()
    return info


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a question against the persistent knowledge base.
    No document upload needed — uses pre-indexed docs.
    """
    if not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Add documents first via /upload."
        )
    result = await rag.query(req.question, top_k=req.top_k)
    return result


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming version of /query — tokens arrive progressively."""
    if not rag.is_ready():
        raise HTTPException(status_code=503, detail="Knowledge base is empty.")

    async def token_generator():
        async for token in rag.query_stream(req.question, top_k=req.top_k):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new document to permanently add to the knowledge base.
    Supported: PDF, TXT, DOCX, MD, CSV
    """
    allowed = {".pdf", ".txt", ".docx", ".md", ".csv"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed)}"
        )

    # Save to a temp file then ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await rag.ingest_document(tmp_path, original_name=file.filename)
        return {
            "message": f"Document '{file.filename}' indexed successfully.",
            "chunks_added": result["chunks"],
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ingest-folder")
async def ingest_folder(folder_path: str = "./docs"):
    """
    Bulk-ingest an entire folder into the knowledge base.
    Use this once to seed the KB from your existing documents.
    """
    result = await rag.ingest_folder(folder_path)
    return result


@app.get("/documents")
async def list_documents():
    """List all documents currently in the knowledge base."""
    return rag.list_documents()


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Remove a document and all its chunks from the knowledge base."""
    success = rag.delete_document(doc_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")
    return {"message": f"Document '{doc_name}' removed from knowledge base."}


@app.delete("/reset")
async def reset_knowledge_base():
    """⚠️ Wipe the entire knowledge base. Irreversible."""
    rag.reset()
    return {"message": "Knowledge base has been reset."}

@app.get("/health")
def health():
    return {"status": "ok"}