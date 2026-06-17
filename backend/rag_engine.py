"""
RAG Engine — core logic.

Stack:
  - LlamaIndex            orchestration
  - Ollama                local LLM (Qwen 2.5 7B) + embeddings (bge-m3)
  - ChromaDB (embedded)   persistent vector store (survives restarts)
  - bge-reranker-v2-m3    CPU cross-encoder reranker (optional, big quality gain)

Design notes:
  - The vector store is a Chroma PersistentClient on disk, so the knowledge
    base is never lost on restart. We never overwrite it on startup.
  - Embeddings use bge-m3 (1024-dim, strongly multilingual).
  - Chat is SESSION-AWARE: each request rebuilds a CondensePlusContext chat
    engine seeded with the conversation history (multi-turn memory). The
    history itself is persisted in SQLite by the caller (see db.py / main.py).
  - Retrieval is two-stage: fetch a wide set of candidates by vector
    similarity, then rerank with a cross-encoder and keep the best few before
    handing them to the LLM. This is the main quality lever.
"""

import os
import asyncio
from typing import Optional

import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")

CHROMA_PATH     = os.getenv("CHROMA_PATH", "/app/data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

# Context / generation
CONTEXT_WINDOW  = int(os.getenv("CONTEXT_WINDOW", "8192"))
NUM_OUTPUT      = int(os.getenv("NUM_OUTPUT", "1024"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Chunking — paragraph/sentence-aware splitting
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))

# Retrieval + reranking
RETRIEVE_TOP_K  = int(os.getenv("RETRIEVE_TOP_K", "10"))   # wide candidate set
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "4"))      # kept after rerank
RERANK_ENABLED  = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE   = os.getenv("RERANK_DEVICE", "cpu")        # keep VRAM free for the LLM

# Memory
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3000"))

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant for a private knowledge base. "
    "Answer using ONLY the information in the provided context when the question "
    "concerns the documents. If the answer is not in the context, say so clearly "
    "instead of inventing facts. "
    "Always reply in the SAME LANGUAGE the user used in their question. "
    "Be concise, structured, and cite document names when relevant."
)


class RAGEngine:
    """Persistent, session-aware RAG engine."""

    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._chroma_collection = None
        self._reranker = None
        self._setup()

    # ─── Setup ────────────────────────────────────────────────────────────────

    def _setup(self):
        print(f"[RAG] LLM={LLM_MODEL}  embed={EMBED_MODEL}  ollama={OLLAMA_BASE_URL}")

        Settings.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=REQUEST_TIMEOUT,
            context_window=CONTEXT_WINDOW,
            # keep the model warm so we don't pay reload latency between turns
            additional_kwargs={"keep_alive": "30m"},
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            paragraph_separator="\n\n\n",
        )
        Settings.num_output = NUM_OUTPUT
        Settings.context_window = CONTEXT_WINDOW

        print(f"[RAG] Opening ChromaDB at {CHROMA_PATH} (collection='{COLLECTION_NAME}')")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # get_or_create => never wipes existing vectors on startup
        self._chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)

        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            storage_context=storage_context,
        )

        self._reranker = self._build_reranker()
        print("[RAG] Engine ready.")

    def _build_reranker(self):
        if not RERANK_ENABLED:
            print("[RAG] Reranker disabled.")
            return None
        try:
            from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(
                model=RERANK_MODEL,
                top_n=RERANK_TOP_N,
                device=RERANK_DEVICE,
            )
            print(f"[RAG] Reranker '{RERANK_MODEL}' on {RERANK_DEVICE} (top_n={RERANK_TOP_N})")
            return reranker
        except Exception as e:
            # Never let a reranker problem take down the whole engine.
            print(f"[RAG] WARNING: reranker unavailable ({e}). Falling back to vector-only.")
            return None

    # ─── Chat (session-aware, streaming) ───────────────────────────────────────

    def _build_chat_engine(self, history: list[dict]):
        """
        Build a fresh chat engine seeded with prior conversation turns.

        `history` is a list of {"role": "user"|"assistant", "content": str}
        representing the messages BEFORE the new user message.
        """
        chat_history = [
            ChatMessage(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in history
            if m.get("content")
        ]

        memory = ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            token_limit=MEMORY_TOKEN_LIMIT,
        )

        postprocessors = [self._reranker] if self._reranker else []

        return self._index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            similarity_top_k=RETRIEVE_TOP_K,
            node_postprocessors=postprocessors,
            system_prompt=SYSTEM_PROMPT,
            streaming=True,
        )

    async def stream_chat(self, message: str, history: list[dict]):
        """
        Async generator over (token, sources).

        Yields ("token", str) for each streamed token, then a final
        ("sources", list[dict]) event once generation completes.
        """
        loop = asyncio.get_event_loop()

        chat_engine = self._build_chat_engine(history)

        # The retrieval + condensation step is blocking -> run in a worker thread.
        streaming_response = await loop.run_in_executor(
            None, lambda: chat_engine.stream_chat(message)
        )

        # Pull tokens one at a time without blocking the event loop.
        gen = streaming_response.response_gen
        sentinel = object()
        while True:
            token = await loop.run_in_executor(None, next, gen, sentinel)
            if token is sentinel:
                break
            yield ("token", token)

        sources = self._format_sources(getattr(streaming_response, "source_nodes", []))
        yield ("sources", sources)

    @staticmethod
    def _format_sources(source_nodes) -> list[dict]:
        sources = []
        for node in source_nodes or []:
            content = node.node.get_content() or ""
            sources.append(
                {
                    "text": content[:320] + ("…" if len(content) > 320 else ""),
                    "score": round(float(node.score or 0), 4),
                    "filename": node.node.metadata.get("file_name", "Unknown"),
                    "page": node.node.metadata.get("page_label", "?"),
                }
            )
        return sources

    # ─── Ingestion ────────────────────────────────────────────────────────────

    async def ingest_document(self, file_path: str, original_name: str) -> dict:
        """Parse, chunk, embed and persist a single document. Returns chunk count."""
        loop = asyncio.get_event_loop()

        def _run():
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            for doc in documents:
                doc.metadata["file_name"] = original_name
                doc.metadata["source"] = original_name

            # Split into chunks explicitly so we can report an accurate count.
            nodes = Settings.node_parser.get_nodes_from_documents(documents)
            for node in nodes:
                node.metadata["file_name"] = original_name
                node.metadata["source"] = original_name

            self._index.insert_nodes(nodes)
            return {"chunks": len(nodes), "filename": original_name}

        return await loop.run_in_executor(None, _run)

    # ─── Management ──────────────────────────────────────────────────────────

    def doc_count(self) -> int:
        """Total number of vector chunks currently stored."""
        try:
            return int(self._chroma_collection.count())
        except Exception:
            return -1

    def is_ready(self) -> bool:
        return self.doc_count() > 0

    def get_status(self) -> dict:
        return {
            "status": "ready",
            "doc_count": self.doc_count(),
            "collection_name": COLLECTION_NAME,
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "reranker": RERANK_MODEL if self._reranker else None,
        }

    def delete_document(self, doc_name: str) -> bool:
        """Remove all chunks belonging to a document from Chroma (by metadata)."""
        try:
            self._chroma_collection.delete(where={"file_name": doc_name})
            return True
        except Exception as e:
            print(f"[RAG] Delete error for '{doc_name}': {e}")
            return False

    def reset(self):
        """Drop and recreate the collection (wipes all vectors)."""
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._setup()
        print("[RAG] Knowledge base reset.")