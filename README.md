# Local RAG — private, multilingual knowledge base

A fully local Retrieval-Augmented Generation app. Your documents never leave the
machine. Built for privacy-sensitive use (legal, finance, internal docs).

**Stack:** LlamaIndex · Ollama (Qwen 2.5 7B + bge-m3) · ChromaDB · SQLite · FastAPI · nginx

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
Browser ──► nginx (:3000) ──► FastAPI backend (:8000)
                                  ├── ChromaDB (embedded, on-disk)   vectors
                                  ├── SQLite (on-disk)               sessions/messages/docs
                                  └── Ollama (host :11434)           Qwen 2.5 7B + bge-m3
```
Ollama runs **natively on the host** (best GPU support on Windows/WSL2). The backend
reaches it via `host.docker.internal`. Everything else runs in Docker. All persistent
data lives in the `rag_data` Docker volume.

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
1. Upload documents (right panel). They are chunked, embedded and stored permanently.
2. Click **＋ Nouvelle conversation**, ask a question. The assistant answers from your
   docs and keeps context across turns.
3. Conversations and documents persist across restarts.

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