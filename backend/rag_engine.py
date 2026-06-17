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
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ── Compatibility shim ───────────────────────────────────────────────────────
# Some llama-index-llms-ollama versions set additional_kwargs["tool_calls"]=None.
# llama-index-core's is_function() then calls len(None) and crashes during
# streaming ("object of type 'NoneType' has no len()"). Make it None-safe.
import llama_index.core.chat_engine.types as _chat_types  # noqa: E402

def _safe_is_function(message) -> bool:
    return bool(getattr(message, "additional_kwargs", {}).get("tool_calls"))

_chat_types.is_function = _safe_is_function


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
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.3"))   # low = focused, less drift

# Chunking — paragraph/sentence-aware splitting
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))

# Retrieval + reranking
RETRIEVE_TOP_K  = int(os.getenv("RETRIEVE_TOP_K", "5"))    # candidate set
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "4"))      # kept after rerank
# Reranker OFF by default: it runs on CPU and adds noticeable latency to EVERY
# message (incl. greetings) + a one-time ~2.2 GB download. Turn it on only if
# you want maximum retrieval quality and can accept slower replies.
RERANK_ENABLED  = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE   = os.getenv("RERANK_DEVICE", "cpu")        # keep VRAM free for the LLM

# Memory
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3000"))

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant for a private knowledge base. "
    "Answer the user's question using the context provided below, and mention the "
    "document name(s) you relied on. If the context does NOT contain the answer, say "
    "clearly that you couldn't find it in the documents — do NOT invent or guess. "
    "CRITICAL: reply ENTIRELY in the SAME LANGUAGE the user used, and never switch to "
    "another language in the middle of your answer. Be clear, concise and structured."
)

# Context block injected into the system message on each turn (may be empty
# when the knowledge base has no relevant content — that's fine, the assistant
# then just chats normally).
CONTEXT_TEMPLATE = (
    "Voici des extraits potentiellement pertinents de la base de connaissance "
    "(ils peuvent être vides ou hors-sujet — utilise-les seulement s'ils aident) :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------"
)

# Prompt for plain conversation (no document retrieval).
CONV_SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant. Chat naturally and answer clearly. "
    "CRITICAL: reply ENTIRELY in the SAME LANGUAGE the user used, and never switch to "
    "another language in the middle of your answer. Be concise."
)

# Router prompt: decides if a message needs the private documents.
# Biased toward YES so real questions are never answered without the documents.
ROUTER_PROMPT = (
    "Decide whether a user message should trigger a search in the user's private "
    "uploaded documents.\n"
    "Reply NO ONLY for pure greetings, thanks or social small talk "
    "(e.g. 'bonjour', 'merci', 'how are you', 'ok').\n"
    "Reply YES for ANY real question, request or task — especially anything that "
    "mentions a project, file, topic, or asks to explain / summarize / list / analyze / "
    "describe something. When unsure, reply YES.\n"
    "Reply with exactly one word: YES or NO.\n\n"
    "{conversation}"
    "Message: {message}\n"
    "Answer:"
)

# Prompt to generate a short conversation title.
TITLE_PROMPT = (
    "Generate a very short, descriptive title for a conversation that starts with "
    "the user message below. Rules: 3 to 6 words, no surrounding quotes, no trailing "
    "punctuation, and write it in the SAME LANGUAGE as the message.\n\n"
    "Message: {message}\n\n"
    "Title:"
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
            temperature=TEMPERATURE,
            keep_alive="30m",   # keep the model warm between turns
            # lower top_p + a repeat penalty reduce rambling and language drift
            additional_kwargs={"top_p": 0.9, "repeat_penalty": 1.1},
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

    def _memory(self, history: list[dict]):
        chat_history = [
            ChatMessage(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in history if m.get("content")
        ]
        return ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=MEMORY_TOKEN_LIMIT
        )

    def _build_doc_engine(self, memory):
        """Conversational RAG: retrieves doc chunks and answers grounded in them."""
        postprocessors = [self._reranker] if self._reranker else []
        return self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=RETRIEVE_TOP_K,
            node_postprocessors=postprocessors,
            system_prompt=SYSTEM_PROMPT,
            context_template=CONTEXT_TEMPLATE,
            streaming=True,
        )

    def _build_chat_only_engine(self, memory):
        """Plain conversation: NO retrieval (greetings, small talk, general questions)."""
        return SimpleChatEngine.from_defaults(
            memory=memory,
            system_prompt=CONV_SYSTEM_PROMPT,
        )

    def generate_title(self, message: str) -> str:
        """Make a short, human-friendly conversation title from the first message."""
        fallback = " ".join((message or "").split())[:48] or "Conversation"
        try:
            resp = Settings.llm.complete(TITLE_PROMPT.format(message=(message or "")[:500]))
            title = (resp.text or "").strip().strip('"').strip()
            title = title.splitlines()[0][:60] if title else ""
            return title or fallback
        except Exception as e:
            print(f"[RAG] title generation error ({e})")
            return fallback

    def _should_use_docs(self, message: str, history: list[dict]) -> bool:
        """Fast YES/NO: does this message actually need the private documents?"""
        try:
            convo = ""
            for m in history[-4:]:
                who = "User" if m["role"] == "user" else "Assistant"
                convo += f"{who}: {m['content'][:300]}\n"
            prompt = ROUTER_PROMPT.format(conversation=convo, message=message)
            resp = Settings.llm.complete(prompt)
            ans = (resp.text or "").strip().lower()
            return ans.startswith("y") or "yes" in ans[:8]
        except Exception as e:
            print(f"[RAG] router error ({e}); defaulting to document search")
            return True

    async def stream_chat(self, message: str, history: list[dict]):
        """
        Async generator yielding ("token", str) events, then a final
        ("sources", list[dict]) event.

        Smart routing: the documents are searched ONLY when the question needs
        them. Greetings / small talk / general questions are answered instantly
        with no retrieval and no sources.
        """
        loop = asyncio.get_event_loop()

        use_docs = False
        if self.doc_count() > 0:
            use_docs = await loop.run_in_executor(
                None, self._should_use_docs, message, history
            )

        memory = self._memory(history)
        engine = self._build_doc_engine(memory) if use_docs else self._build_chat_only_engine(memory)

        streaming_response = await loop.run_in_executor(
            None, lambda: engine.stream_chat(message)
        )

        gen = streaming_response.response_gen
        sentinel = object()
        while True:
            token = await loop.run_in_executor(None, next, gen, sentinel)
            if token is sentinel:
                break
            yield ("token", token)

        sources = (
            self._format_sources(getattr(streaming_response, "source_nodes", []))
            if use_docs else []
        )
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
