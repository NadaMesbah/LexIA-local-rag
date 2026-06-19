"""
RAG Engine — core logic (Phase 2: document versioning + freshness).

Stack: LlamaIndex · Ollama (Qwen 2.5 + bge-m3) · ChromaDB (embedded) · optional
bge-reranker.

Versioning / freshness
----------------------
Every chunk in Chroma carries metadata: document_id, version, is_latest.
Retrieval always filters on is_latest == true, so the chat answers only from the
newest indexed version of each document. Uploading a new version inserts fresh
is_latest chunks and flips the previous version's chunks to is_latest=false
(kept for rollback). Soft-delete flips all of a document's chunks to false;
hard-delete purges them.
"""

import os
import re
import asyncio
import threading
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
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import db

# ── Compatibility shim ───────────────────────────────────────────────────────
# Some llama-index-llms-ollama versions set additional_kwargs["tool_calls"]=None,
# which makes llama-index-core's is_function() call len(None) and crash during
# streaming. Make it None-safe.
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

CONTEXT_WINDOW  = int(os.getenv("CONTEXT_WINDOW", "8192"))
NUM_OUTPUT      = int(os.getenv("NUM_OUTPUT", "1024"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.2"))

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))

RETRIEVE_TOP_K  = int(os.getenv("RETRIEVE_TOP_K", "5"))
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "4"))
RERANK_ENABLED  = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
RELEVANCE_CUTOFF = float(os.getenv("RELEVANCE_CUTOFF", "0.6"))
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE   = os.getenv("RERANK_DEVICE", "cpu")

MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3000"))

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant for a private knowledge base. "
    "Answer using the context below. If the context does NOT contain the answer, say "
    "clearly that you couldn't find it in the documents — do NOT invent or guess. "
    "Do NOT add a 'Sources', 'Source' or citations list to your answer, and do NOT "
    "mention file names or version numbers — the application displays the sources "
    "separately. Be clear, concise and well structured."
)

CONTEXT_TEMPLATE = (
    "Voici des extraits potentiellement pertinents de la base de connaissance "
    "(ils peuvent être vides ou hors-sujet — utilise-les seulement s'ils aident) :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------"
)

CONV_SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant. Chat naturally and answer clearly. Be concise."
)

ROUTER_PROMPT = (
    "Decide whether a user message should trigger a search in the user's private "
    "uploaded documents.\n"
    "Reply NO ONLY for pure greetings, thanks or social small talk.\n"
    "Reply YES for ANY real question, request or task. When unsure, reply YES.\n"
    "Reply with exactly one word: YES or NO.\n\n"
    "{conversation}"
    "Message: {message}\n"
    "Answer:"
)

TITLE_PROMPT = (
    "Generate a very short, descriptive title (3 to 6 words, no quotes, no trailing "
    "punctuation, same language as the message) for a conversation that starts with:\n\n"
    "Message: {message}\n\nTitle:"
)


# ─── Language lock (prevents Qwen drifting into Chinese) ──────────────────────
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_AR = re.compile(r"[\u0600-\u06ff]")
_CYR = re.compile(r"[\u0400-\u04ff]")
_FR_ACC = re.compile(r"[àâçéèêëîïôûùüœ]")


def _detect_language(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if _CJK.search(t):
        return "Chinese"
    if _AR.search(t):
        return "Arabic"
    if _CYR.search(t):
        return "Russian"
    low = " " + t.lower() + " "
    fr = sum(m in low for m in (
        " le ", " la ", " les ", " un ", " une ", " des ", " du ", " et ", " est ", " que ",
        " qui ", " pour ", " avec ", " vous ", " je ", " ne ", " pas ", " bonjour ", " salut ",
        " merci ", " quoi ", " pourquoi ", " comment ", " quel ", " ça ", " c'est ",
        " explique ", " résume ",
    )) + len(_FR_ACC.findall(low)) * 2
    en = sum(m in low for m in (
        " the ", " is ", " are ", " you ", " what ", " why ", " how ", " please ", " hello ",
        " hi ", " thanks ", " and ", " to ", " of ", " in ", " for ", " with ", " this ",
        " explain ", " summarize ",
    ))
    if fr == 0 and en == 0:
        return None
    return "French" if fr >= en else "English"


def _language_directive(text: str) -> str:
    lang = _detect_language(text)
    if lang == "Chinese":
        return ""
    if lang:
        return (
            f"\n\nABSOLUTE RULE — LANGUAGE: The user is writing in {lang}. Write your ENTIRE "
            f"answer in {lang} only. Never output Chinese characters or switch languages. "
            f"This overrides everything else."
        )
    return (
        "\n\nABSOLUTE RULE — LANGUAGE: Reply only in the exact same language as the user's "
        "message. Never output Chinese characters or switch languages."
    )


def _scope_filters(scope: Optional[dict]) -> MetadataFilters:
    """Always restrict to is_latest=true; optionally to one document (scope)."""
    filters = [MetadataFilter(key="is_latest", value=True, operator=FilterOperator.EQ)]
    if scope and scope.get("type") == "document" and scope.get("id"):
        filters.append(
            MetadataFilter(key="document_id", value=scope["id"], operator=FilterOperator.EQ)
        )
    return MetadataFilters(filters=filters)


class RAGEngine:
    """Persistent, session-aware, version-aware RAG engine."""

    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._chroma_collection = None
        self._reranker = None
        self._write_lock = threading.Lock()
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
            keep_alive="30m",
            additional_kwargs={"top_p": 0.85, "top_k": 40, "repeat_penalty": 1.15},
        )
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        Settings.node_parser = SentenceSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, paragraph_separator="\n\n\n"
        )
        Settings.num_output = NUM_OUTPUT
        Settings.context_window = CONTEXT_WINDOW

        print(f"[RAG] Opening ChromaDB at {CHROMA_PATH} (collection='{COLLECTION_NAME}')")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store, storage_context=storage_context
        )

        self._reranker = self._build_reranker()
        self._backfill_legacy()
        print("[RAG] Engine ready.")

    def _build_reranker(self):
        if not RERANK_ENABLED:
            print("[RAG] Reranker disabled.")
            return None
        try:
            from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
            r = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N, device=RERANK_DEVICE)
            print(f"[RAG] Reranker '{RERANK_MODEL}' on {RERANK_DEVICE}")
            return r
        except Exception as e:
            print(f"[RAG] WARNING: reranker unavailable ({e}). Vector-only.")
            return None

    def _backfill_legacy(self):
        """Give pre-Phase-2 chunks the version metadata so they stay retrievable."""
        try:
            got = self._chroma_collection.get(include=["metadatas"])
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            up_ids, up_mds = [], []
            for cid, md in zip(ids, mds):
                md = dict(md or {})
                if "is_latest" not in md or "document_id" not in md:
                    name = md.get("file_name") or md.get("source") or "Unknown"
                    info = db.ensure_legacy_document(name, 0)
                    md["file_name"] = name
                    md["document_id"] = info["document_id"]
                    md["version"] = info["version"]
                    md["is_latest"] = True
                    up_ids.append(cid)
                    up_mds.append(md)
            if up_ids:
                self._chroma_collection.update(ids=up_ids, metadatas=up_mds)
                print(f"[RAG] Backfilled {len(up_ids)} legacy chunks with version metadata.")
        except Exception as e:
            print(f"[RAG] backfill skipped ({e})")

    # ─── Chat ──────────────────────────────────────────────────────────────────

    def _memory(self, history: list[dict]):
        chat_history = [
            ChatMessage(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in history if m.get("content")
        ]
        return ChatMemoryBuffer.from_defaults(chat_history=chat_history, token_limit=MEMORY_TOKEN_LIMIT)

    def _rerank(self, nodes, query: str):
        if self._reranker and nodes:
            try:
                from llama_index.core.schema import QueryBundle
                return self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
            except Exception as e:
                print(f"[RAG] rerank error ({e})")
        return nodes

    def _relevant_nodes(self, message: str, scope: Optional[dict]):
        """
        Retrieve candidates, then keep only LATEST-version chunks (optionally
        scoped to one document) above the relevance cutoff. Pure-Python filtering
        so correctness never depends on vector-store filter translation.
        """
        try:
            retriever = self._index.as_retriever(similarity_top_k=max(RETRIEVE_TOP_K * 3, 12))
            nodes = retriever.retrieve(message)
        except Exception as e:
            print(f"[RAG] retrieval error ({e})")
            return []
        scoped_id = scope.get("id") if (scope and scope.get("type") == "document") else None
        kept = []
        for n in nodes:
            md = n.node.metadata or {}
            if md.get("is_latest") is False:   # missing -> treat as latest (legacy safety)
                continue
            if scoped_id and md.get("document_id") != scoped_id:
                continue
            kept.append(n)
        kept = [n for n in kept if (n.score or 0) >= RELEVANCE_CUTOFF]
        kept = self._rerank(kept, message)
        return kept[:RETRIEVE_TOP_K]

    @staticmethod
    def _context_block(nodes) -> str:
        parts = []
        for n in nodes:
            md = n.node.metadata or {}
            head = md.get("file_name", "document")
            if md.get("version"):
                head += f" v{md['version']}"
            page = md.get("page_label")
            if page and page != "?":
                head += f" p.{page}"
            parts.append(f"[{head}]\n{n.node.get_content()}")
        return "\n\n".join(parts)

    def generate_title(self, message: str) -> str:
        fallback = " ".join((message or "").split())[:48] or "Conversation"
        try:
            prompt = _language_directive(message) + "\n\n" + TITLE_PROMPT.format(
                message=(message or "")[:500]
            )
            resp = Settings.llm.complete(prompt.strip())
            title = (resp.text or "").strip().strip('"').strip()
            return (title.splitlines()[0][:60] if title else "") or fallback
        except Exception as e:
            print(f"[RAG] title generation error ({e})")
            return fallback

    def _should_use_docs(self, message: str, history: list[dict]) -> bool:
        try:
            convo = ""
            for m in history[-4:]:
                who = "User" if m["role"] == "user" else "Assistant"
                convo += f"{who}: {m['content'][:300]}\n"
            resp = Settings.llm.complete(ROUTER_PROMPT.format(conversation=convo, message=message))
            ans = (resp.text or "").strip().lower()
            return ans.startswith("y") or "yes" in ans[:8]
        except Exception as e:
            print(f"[RAG] router error ({e}); defaulting to document search")
            return True

    async def stream_chat(self, message: str, history: list[dict], scope: Optional[dict] = None):
        """
        Yields ("token", str)… then ("sources", list).

        Retrieves over latest-version chunks (optionally scoped to one document)
        and streams the answer directly from the LLM with the retrieved context
        injected — no chat-engine wrapper, so the stream can't silently come back
        empty. Grounds + cites when relevant chunks exist; otherwise answers
        generally with no sources.
        """
        loop = asyncio.get_event_loop()
        directive = _language_directive(message)
        scoped = bool(scope and scope.get("type") == "document")

        nodes = []
        if self.doc_count() > 0:
            wants = scoped or await loop.run_in_executor(
                None, self._should_use_docs, message, history
            )
            if wants:
                nodes = await loop.run_in_executor(None, self._relevant_nodes, message, scope)

        use_docs = len(nodes) > 0
        if use_docs:
            system_text = (
                SYSTEM_PROMPT + directive + "\n\n"
                + CONTEXT_TEMPLATE.format(context_str=self._context_block(nodes))
            )
        else:
            system_text = CONV_SYSTEM_PROMPT + directive

        msgs = [ChatMessage(role=MessageRole.SYSTEM, content=system_text)]
        for m in history[-12:]:
            if not m.get("content"):
                continue
            msgs.append(ChatMessage(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            ))
        msgs.append(ChatMessage(role=MessageRole.USER, content=message))

        gen = await loop.run_in_executor(None, lambda: Settings.llm.stream_chat(msgs))
        sentinel = object()
        while True:
            resp = await loop.run_in_executor(None, next, gen, sentinel)
            if resp is sentinel:
                break
            delta = getattr(resp, "delta", None)
            if delta:
                yield ("token", delta)

        yield ("sources", self._format_sources(nodes) if use_docs else [])

    @staticmethod
    def _format_sources(source_nodes) -> list[dict]:
        sources = []
        for node in source_nodes or []:
            content = node.node.get_content() or ""
            md = node.node.metadata or {}
            sources.append({
                "text": content[:320] + ("…" if len(content) > 320 else ""),
                "score": round(float(node.score or 0), 4),
                "filename": md.get("file_name", "Unknown"),
                "page": md.get("page_label", "?"),
                "document_id": md.get("document_id"),
                "version": md.get("version"),
            })
        return sources

    # ─── Ingestion & version management ────────────────────────────────────────

    def index_version(self, file_path: str, filename: str, document_id: str, version: int) -> int:
        """Chunk/embed a file as a new version, mark it is_latest, demote older ones."""
        with self._write_lock:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            nodes = Settings.node_parser.get_nodes_from_documents(documents)
            for node in nodes:
                node.metadata.update({
                    "file_name": filename,
                    "source": filename,
                    "document_id": document_id,
                    "version": version,
                    "is_latest": True,
                })
            self._index.insert_nodes(nodes)
            self._demote_other_versions(document_id, keep_version=version)
            return len(nodes)

    def _demote_other_versions(self, document_id: str, keep_version: int):
        try:
            got = self._chroma_collection.get(
                where={"$and": [{"document_id": document_id}, {"version": {"$ne": keep_version}}]},
                include=["metadatas"],
            )
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            cids, cmds = [], []
            for cid, md in zip(ids, mds):
                if md and md.get("is_latest"):
                    md = dict(md); md["is_latest"] = False
                    cids.append(cid); cmds.append(md)
            if cids:
                self._chroma_collection.update(ids=cids, metadatas=cmds)
        except Exception as e:
            print(f"[RAG] demote error ({e})")

    def set_document_hidden(self, document_id: str):
        """Soft-delete: remove all of a document's chunks from retrieval (is_latest=false)."""
        with self._write_lock:
            self._set_is_latest_where({"document_id": document_id}, False)

    def restore_version(self, document_id: str, version: int):
        """Rollback: make `version`'s chunks is_latest, all other versions not."""
        with self._write_lock:
            try:
                got = self._chroma_collection.get(
                    where={"document_id": document_id}, include=["metadatas"]
                )
                ids = got.get("ids", []) or []
                mds = got.get("metadatas", []) or []
                cids, cmds = [], []
                for cid, md in zip(ids, mds):
                    want = md.get("version") == version
                    if bool(md.get("is_latest")) != want:
                        md = dict(md); md["is_latest"] = want
                        cids.append(cid); cmds.append(md)
                if cids:
                    self._chroma_collection.update(ids=cids, metadatas=cmds)
            except Exception as e:
                print(f"[RAG] restore error ({e})")

    def _set_is_latest_where(self, where: dict, value: bool):
        try:
            got = self._chroma_collection.get(where=where, include=["metadatas"])
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            cids, cmds = [], []
            for cid, md in zip(ids, mds):
                if bool(md.get("is_latest")) != value:
                    md = dict(md); md["is_latest"] = value
                    cids.append(cid); cmds.append(md)
            if cids:
                self._chroma_collection.update(ids=cids, metadatas=cmds)
        except Exception as e:
            print(f"[RAG] is_latest update error ({e})")

    def purge_document(self, document_id: str):
        """Hard-delete: permanently remove all of a document's vectors."""
        with self._write_lock:
            try:
                self._chroma_collection.delete(where={"document_id": document_id})
            except Exception as e:
                print(f"[RAG] purge error ({e})")

    # ─── Status ────────────────────────────────────────────────────────────────

    def doc_count(self) -> int:
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

    def reset(self):
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._setup()
        print("[RAG] Knowledge base reset.")
