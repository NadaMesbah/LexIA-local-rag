# Local RAG Advanced

**Local RAG Advanced** is a private, multilingual Retrieval-Augmented Generation application that lets users upload documents, build a local knowledge base, and ask questions with grounded answers and sources.

The system runs locally with **Ollama**, **ChromaDB**, **SQLite**, **FastAPI**, and a **Next.js** web interface. Documents, conversations, embeddings, and metadata stay on the local machine.

---

## Overview

This project provides a local assistant capable of answering from indexed documents or, when allowed, from general knowledge.

It supports two main modes:

* **Documents only**: answers strictly from the indexed knowledge base and refuses when the information is not found.
* **General assistant**: can answer general questions while clearly marking responses that are outside the documents.

The app is designed for private document search, academic demos, internal knowledge bases, and multilingual document Q&A.

---

## Key Features

* **Local RAG pipeline** with ChromaDB and Ollama.
* **Multilingual Q&A**, including French and Arabic.
* **Document-grounded answers** with displayed sources.
* **Documents only mode** for strict RAG behavior.
* **General answer mode** with clear “outside documents” labeling.
* **Hybrid retrieval** using dense vector search and BM25 lexical search.
* **Optional reranker** for higher retrieval precision.
* **Persistent conversations** stored in SQLite.
* **Persistent knowledge base** stored in ChromaDB.
* **Streaming responses** token by token.
* **Multi-user authentication** with JWT.
* **Document upload and management** from the web interface.
* **Supported formats**: PDF, TXT, Markdown, DOCX, CSV, XLSX.

---

## Architecture

```text
Browser
  |
  v
Next.js Frontend (:3000)
  |
  v
FastAPI Backend (:8000)
  |
  |-- SQLite
  |     - users
  |     - sessions
  |     - messages
  |     - document metadata
  |
  |-- ChromaDB
  |     - chunks
  |     - embeddings
  |     - metadata
  |
  |-- Ollama on host machine (:11434)
        - LLM: qwen2.5:7b
        - Embeddings: bge-m3
        - Optional reranker
```

Ollama runs natively on the host machine.
The backend Docker container reaches it through:

```text
http://host.docker.internal:11434
```

Persistent data is stored in the Docker volume:

```text
rag_data
```

---

## Tech Stack

**Frontend**

* Next.js
* React
* TypeScript
* Material UI

**Backend**

* FastAPI
* LlamaIndex
* ChromaDB
* SQLite
* Ollama
* BM25 hybrid retrieval
* Optional cross-encoder reranker

**Default models**

```text
LLM        : qwen2.5:7b
Embeddings : bge-m3
Reranker   : BAAI/bge-reranker-v2-m3, optional
```

---

## Quick Start

### 1. Install Ollama models

Run on the host machine:

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` if needed.

### 3. Start the application

```bash
docker compose up -d --build
```

Open:

```text
http://localhost:3000
```

Backend API:

```text
http://localhost:8000
```

---

## Default Admin Account

```text
Email    : admin@local.host
Password : admin
```

These values can be changed in `.env`.

---

## Usage

### Document-based RAG

1. Go to **Documents**.
2. Upload documents.
3. Go to **Chat**.
4. Enable **Documents uniquement**.
5. Ask questions about the indexed documents.

In this mode, the assistant answers only from the knowledge base and displays the source document.

### General assistant mode

Disable **Documents uniquement** to allow general answers.

If the answer is not grounded in documents, the UI marks it as:

```text
Réponse générale (hors documents)
```

and no document source is displayed.

### Scoped document mode

A user can select or attach a document in a conversation.
The assistant then searches only inside that document until the scope is removed.

---

## Important Environment Variables

| Variable                | Description                       | Default      |
| ----------------------- | --------------------------------- | ------------ |
| `LLM_MODEL`             | Ollama generation model           | `qwen2.5:7b` |
| `EMBED_MODEL`           | Ollama embedding model            | `bge-m3`     |
| `HYBRID_ENABLED`        | Enables dense + BM25 retrieval    | `true`       |
| `RETRIEVE_CANDIDATES`   | Number of candidate chunks        | `20`         |
| `RETRIEVE_TOP_K`        | Number of final context chunks    | `6`          |
| `SOURCE_DISPLAY_TOP_K`  | Number of displayed sources       | `1`          |
| `STRICT_RAG`            | Backend strict mode by default    | `false`      |
| `QUERY_REWRITE_ENABLED` | Enables follow-up query rewriting | `true`       |
| `GROUNDING_MIN_OVERLAP` | Grounding check threshold         | `0.18`       |
| `RERANK_ENABLED`        | Enables optional reranker         | `false`      |

---

## Useful Commands

Check backend health:

```bash
curl http://localhost:8000/health
```

View logs:

```bash
docker compose logs backend -f
```

Stop containers:

```bash
docker compose down
```

Stop and delete all stored data:

```bash
docker compose down -v
```

---

## Development

Run frontend locally:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Run backend tests:

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

---

## Notes

* `.env` contains local secrets and should not be committed.
* `.env.example` is the safe template to share.
* Changing the embedding model requires re-indexing the knowledge base.
* Enabling the reranker improves retrieval quality but increases latency.
* `docker compose down` keeps documents and conversations.
* `docker compose down -v` deletes the entire local knowledge base.

---

## Project Goal

The goal of this project is to demonstrate a private, local, multilingual RAG system capable of managing documents, answering questions with reliable sources, and clearly separating document-grounded responses from general assistant responses.
