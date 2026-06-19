# Local RAG — private, multilingual knowledge base

A fully local Retrieval-Augmented Generation app. Your documents never leave the
machine. Built for privacy-sensitive use (legal, finance, internal docs).

**Stack:** Next.js + MUI (frontend) · LlamaIndex · Ollama (Qwen 2.5 7B + bge-m3) · ChromaDB · SQLite · FastAPI

## Features
- **Multilingual** Q&A (bge-m3 embeddings + Qwen 2.5) — answers in the language you ask in.
- **Conversation memory** — multi-turn chat that keeps context within a session.
- **Persistent sessions** — chats survive restarts (stored in SQLite).
- **Persistent knowledge base** — vectors stored on disk in ChromaDB; documents are remembered.
- **Multi-user** — independent conversation sessions.
- **Quality retrieval** — paragraph/sentence-aware chunking, vector search + cross-encoder reranking (bge-reranker-v2-m3).
- **Streaming** token-by-token responses.
- **Upload** PDF, TXT, DOCX, MD, CSV, XLSX.

## Architecture
```
Browser ──► Next.js UI (:3000) ──► FastAPI backend (:8000)
   (Chat + Documents)                 ├── ChromaDB (embedded, on-disk)   vectors
                                      ├── SQLite (on-disk)               sessions/messages/docs
                                      └── Ollama (host :11434)           Qwen 2.5 7B + bge-m3
```
The frontend is a **Next.js 14 + MUI** app with two sections: **Chat** (default
landing) and **Documents** (manage the knowledge base). It calls the backend
directly (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`). Ollama runs
**natively on the host** (best GPU support on Windows/WSL2); the backend reaches
it via `host.docker.internal`. All persistent data lives in the `rag_data` Docker volume.

## Prerequisites
1. **NVIDIA driver + CUDA 12.8+** (the RTX 5050 is Blackwell — needs a recent driver).
2. **Ollama** installed on the host: https://ollama.com/download
3. **Docker Desktop** (with WSL2 backend on Windows).

## Setup
```bash
# 1. Pull the models (on the host, once)
ollama pull qwen2.5:7b
ollama pull bge-m3

# 2. Configure
cp .env.example .env

# 3. Build & run
docker compose up --build
```
Open **http://localhost:3000**.

> First chat with reranking enabled downloads the reranker (~2.2 GB) into the
> `rag_data` volume. (Reranking is off by default.)

## Usage
1. After launch you land in **Chat**. Ask a question; the assistant answers from your
   documents (with citations in the Sources panel) or from general knowledge, and keeps
   context across turns. Edit a message, regenerate, or browse versions like ChatGPT.
2. Go to **Documents** to upload (drag & drop), search, and delete files in the knowledge base.
3. Conversations and documents persist across restarts.

## Frontend development (without Docker)
```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```
(`docker compose up --build` builds the frontend image automatically — no manual npm step needed.)

## Tuning for your hardware (8 GB VRAM)
- Qwen 2.5 7B (Q4) + bge-m3 fit in 8 GB.
- The cross-encoder **reranker is OFF by default** for fast, normal-assistant
  responsiveness (it runs on CPU and adds latency to every message). Turn it on
  with `RERANK_ENABLED=true` (and raise `RETRIEVE_TOP_K=10`) for higher retrieval
  quality at the cost of slower replies + a one-time ~2.2 GB model download.
- If you hit VRAM pressure: set `LLM_MODEL=qwen2.5:3b` in `.env`.

## Verify it's working
```bash
curl http://localhost:8000/            # status + models
curl -X POST http://localhost:8000/sessions -H 'Content-Type: application/json' -d '{}'
```

## Notes
- Changing `EMBED_MODEL` changes vector dimensions — you must re-index (DELETE /reset) afterwards.
- The knowledge base lives in the `rag_data` volume; `docker compose down` keeps it,
  `docker compose down -v` deletes it.