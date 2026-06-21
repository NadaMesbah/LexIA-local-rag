"""
RAG Engine — core logic (Phase 2: document versioning + freshness).

Stack: LlamaIndex · Ollama (Qwen 2.5 + bge-m3) · ChromaDB (embedded) · optional
bge-reranker.

Versioning / freshness
----------------------
Every chunk in Chroma carries metadata: rag_doc_id, version, is_latest (plus
owner_id/scope for access control). NOTE: we deliberately do NOT use the key
"document_id" — LlamaIndex reserves it and overwrites it with its own ref_doc_id
on insert, which silently breaks any where-filter on it.
Retrieval always filters on is_latest == true, so the chat answers only from the
newest indexed version of each document. Uploading a new version inserts fresh
is_latest chunks and flips the previous version's chunks to is_latest=false
(kept for rollback). Soft-delete flips all of a document's chunks to false;
hard-delete purges them.
"""

import os
import re
import math
import asyncio
import threading
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage, MessageRole
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
# Optional lighter model used ONLY for short auto-titles. Defaults to the main
# model; set TITLE_MODEL (e.g. "qwen2.5:1.5b") to make titling faster/cheaper.
TITLE_MODEL     = os.getenv("TITLE_MODEL", "").strip() or LLM_MODEL
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")

CHROMA_PATH     = os.getenv("CHROMA_PATH", "/app/data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

# Fully local & private: disable ChromaDB's anonymized product telemetry so the
# system makes ZERO outbound network calls at runtime (everything stays on the
# host — demonstrable by pulling the network and watching it still answer).
CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)

CONTEXT_WINDOW  = int(os.getenv("CONTEXT_WINDOW", "8192"))
NUM_OUTPUT      = int(os.getenv("NUM_OUTPUT", "1024"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.2"))

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))

RETRIEVE_TOP_K  = int(os.getenv("RETRIEVE_TOP_K", "6"))
# Wider candidate pool pulled from each retriever before fusion/rerank/cutoff.
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_ENABLED  = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
# Hybrid lexical (BM25) retrieval on top of dense vectors — catches exact terms,
# names, codes, dates and short queries that pure vector search misses.
HYBRID_ENABLED  = os.getenv("HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
# Strict RAG: answer ONLY from documents (never fall back to general chat). Off by
# default so casual small talk still works; can be forced per-request or via env.
STRICT_RAG_DEFAULT = os.getenv("STRICT_RAG", "false").lower() in ("1", "true", "yes")
# Minimum cosine similarity (score = 1 - cosine_distance) for a chunk to count as
# relevant. With bge-m3 and the reranker OFF, genuinely relevant passages often
# score ~0.4-0.6, so a high cutoff silently discards them and the answer falls
# back to "general" with no sources. 0.35 favours recall; raise it (e.g. 0.5)
# only if the reranker is enabled or answers pull in off-topic chunks.
RELEVANCE_CUTOFF = float(os.getenv("RELEVANCE_CUTOFF", "0.35"))

# Source DISPLAY filtering (independent of retrieval/context). Retrieval keeps a
# low cutoff for recall, but we only SHOW the user sources that are clearly
# relevant: close to the best match (relative gap), above a modest floor, and
# capped in number — so a correct answer isn't accompanied by an unrelated long
# tail, and pure small talk (no real match) shows no source at all. Tunable via
# env without touching answer quality. For the sharpest source precision, enable
# the cross-encoder reranker (RERANK_ENABLED=true).
SOURCE_MIN_SCORE = float(os.getenv("SOURCE_MIN_SCORE", "0.40"))
SOURCE_REL_GAP   = float(os.getenv("SOURCE_REL_GAP", "0.10"))
SOURCE_MAX       = int(os.getenv("SOURCE_MAX", "4"))
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE   = os.getenv("RERANK_DEVICE", "cpu")

MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3000"))

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant for a private knowledge base. "
    "Answer using the context extracts below. Do NOT use general/prior knowledge to "
    "add or invent facts that are not in the extracts. If the extracts do NOT contain "
    "the answer, say clearly (in the user's language) that you couldn't find it in the "
    "documents — do NOT guess. "
    "Do NOT add a 'Sources', 'Source' or citations list to your answer, and do NOT "
    "mention file names or version numbers — the application displays the sources "
    "separately. Be clear, concise and well structured."
)

# Used when strict RAG is on: documents-only, no general knowledge, refuse if absent.
SYSTEM_PROMPT_STRICT = (
    "You are a STRICT retrieval assistant for a private knowledge base. Answer ONLY "
    "from the context extracts below. Never use general or prior knowledge to complete "
    "or guess. If the exact answer is not explicitly present in the extracts, reply (in "
    "the user's language) exactly: \"Je n'ai pas trouvé cette information dans les "
    "documents.\" Do NOT add a sources/citations list or mention file names — the app "
    "shows sources separately. Be concise and faithful to the extracts."
)

CONTEXT_TEMPLATE = (
    "Extraits de la base de connaissance (utilise-les comme seule source factuelle "
    "pour répondre) :\n"
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


# ─── Lexical search (BM25) for hybrid retrieval ───────────────────────────────
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    # \w with re.UNICODE keeps Arabic/accented letters and digits — works for the
    # French/Arabic/English mix in this app.
    return _WORD_RE.findall((text or "").lower())


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class _BM25:
    """Minimal BM25-Okapi over an in-memory chunk corpus. Pure Python, no extra
    dependency, fully offline — the lexical half of hybrid retrieval, catching
    exact terms/names/codes/dates that dense vectors miss."""

    def __init__(self, ids: list, texts: list, metas: list, k1: float = 1.5, b: float = 0.75):
        self.ids, self.texts, self.metas, self.k1, self.b = ids, texts, metas, k1, b
        docs = [_tokenize(t) for t in texts]
        self._len = [len(d) for d in docs]
        self.N = len(docs)
        self.avgdl = (sum(self._len) / self.N) if self.N else 0.0
        df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        for d in docs:
            tf: dict[str, int] = {}
            for term in d:
                tf[term] = tf.get(term, 0) + 1
            self._tf.append(tf)
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, n: int) -> list[tuple]:
        q = _tokenize(query)
        if not q or self.N == 0:
            return []
        scored: list[tuple] = []
        for i in range(self.N):
            tf, dl = self._tf[i], (self._len[i] or 1)
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self.ids[i], self.texts[i], self.metas[i], sc) for i, sc in scored[:n]]


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep the most recent turns within the memory budget (~4 chars/token)."""
    budget = MEMORY_TOKEN_LIMIT * 4
    out: list[dict] = []
    total = 0
    for m in reversed(history):
        content = m.get("content") or ""
        if not content:
            continue
        total += len(content)
        if total > budget and out:
            break
        out.append(m)
    out.reverse()
    return out


class RAGEngine:
    """Persistent, session-aware, version-aware RAG engine."""
    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._chroma_collection = None
        self._reranker = None
        self._title_llm = None
        # Lexical (BM25) index over latest chunks, lazily (re)built and cached.
        self._bm25 = None
        self._bm25_dirty = True
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

        # Dedicated (optionally lighter) LLM for short auto-titles. Reusing a small
        # model here keeps titling from competing with / waiting behind the main
        # generation model. Falls back to the main model when TITLE_MODEL is unset.
        if TITLE_MODEL == LLM_MODEL:
            self._title_llm = Settings.llm
        else:
            print(f"[RAG] Title model={TITLE_MODEL}")
            self._title_llm = Ollama(
                model=TITLE_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=REQUEST_TIMEOUT,
                temperature=0.2,
                keep_alive="30m",
            )

        print(f"[RAG] Opening ChromaDB at {CHROMA_PATH} (collection='{COLLECTION_NAME}')")
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=CHROMA_SETTINGS)
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
        self._reconcile_latest()
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
        """Repair chunk metadata at startup so existing chunks stay retrievable and
        correctly filtered: give pre-versioning chunks version/owner/scope, and
        backfill `rag_doc_id` for chunks indexed before that key existed (their
        reserved "document_id" was clobbered by LlamaIndex with its ref_doc_id)."""
        try:
            got = self._chroma_collection.get(include=["metadatas"])
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            up_ids, up_mds = [], []
            for cid, md in zip(ids, mds):
                md = dict(md or {})
                changed = False
                # Pre-versioning chunks: no version/is_latest at all.
                if "is_latest" not in md or "version" not in md:
                    name = md.get("file_name") or md.get("source") or "Unknown"
                    info = db.ensure_legacy_document(name, 0)
                    md["file_name"] = name
                    md.setdefault("rag_doc_id", info["document_id"])
                    md.setdefault("version", info["version"])
                    md["is_latest"] = True
                    md.setdefault("owner_id", info["owner_id"] or "")
                    md.setdefault("scope", info["scope"])
                    changed = True
                if "scope" not in md or "owner_id" not in md:
                    doc = db.get_document(md.get("rag_doc_id")) if md.get("rag_doc_id") else None
                    md["scope"] = (doc or {}).get("scope") or md.get("scope") or "org"
                    md["owner_id"] = (doc or {}).get("owner_id") or md.get("owner_id") or ""
                    changed = True
                # Backfill our document key for chunks indexed before "rag_doc_id"
                # existed. We can still resolve it from (file_name, scope, owner_id),
                # which LlamaIndex does NOT clobber.
                if not md.get("rag_doc_id"):
                    did = db.find_document_id(
                        md.get("file_name") or md.get("source") or "",
                        md.get("scope") or "org",
                        md.get("owner_id") or None,
                    )
                    if did:
                        md["rag_doc_id"] = did
                        changed = True
                if changed:
                    up_ids.append(cid); up_mds.append(md)
            if up_ids:
                self._chroma_collection.update(ids=up_ids, metadatas=up_mds)
                print(f"[RAG] Backfilled {len(up_ids)} chunks (rag_doc_id / version / owner / scope).")
            self._invalidate_bm25()
        except Exception as e:
            print(f"[RAG] backfill skipped ({e})")

    def _reconcile_latest(self):
        """
        Repair is_latest across all chunks using the DB as source of truth:
        only the active document's current_version stays is_latest=true. Fixes
        stale/duplicate-version chunks left by older code or failed flips.
        """
        try:
            latest = {}  # document_id -> current_version (None if deleted/inactive)
            # role="admin" is REQUIRED here: without it, list_documents applies the
            # per-user visibility filter "(scope='org' OR owner_id=?)" with owner_id
            # = None, which silently drops every PERSONAL document. Those documents
            # would then be absent from `latest`, so the loop below would flip their
            # chunks to is_latest=False on every startup and make a user's own
            # documents permanently unretrievable. We need the full, unfiltered set.
            for d in db.list_documents(include_deleted=True, role="admin"):
                latest[d["id"]] = d["current_version"] if d["status"] == "active" else None
            got = self._chroma_collection.get(include=["metadatas"])
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            cids, cmds = [], []
            for cid, md in zip(ids, mds):
                md = dict(md or {})
                did = md.get("rag_doc_id")
                want = did in latest and latest[did] is not None and md.get("version") == latest[did]
                if bool(md.get("is_latest")) != want:
                    md["is_latest"] = want
                    cids.append(cid); cmds.append(md)
            if cids:
                self._chroma_collection.update(ids=cids, metadatas=cmds)
                print(f"[RAG] Reconciled is_latest on {len(cids)} chunks.")
            self._invalidate_bm25()
        except Exception as e:
            print(f"[RAG] reconcile skipped ({e})")

    # ─── Chat ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _scoped_ids(scope: Optional[dict]) -> list[str]:
        """Document ids a query is explicitly restricted to (composer attachments)."""
        if not (scope and scope.get("type") == "document"):
            return []
        ids = scope.get("ids") or ([scope.get("id")] if scope.get("id") else [])
        return [i for i in ids if i]

    async def _await_attachments_indexed(self, doc_ids: list[str], timeout: float = 120.0):
        """
        Block until the explicitly-attached documents finish indexing (status
        leaves pending/processing). A question asked immediately after attaching
        a file would otherwise race ahead of indexing, retrieve nothing, and the
        model would hallucinate. Bounded by `timeout` so a stuck job can't hang
        the request forever.
        """
        if not doc_ids:
            return
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            pending = [d for d in doc_ids
                       if (db.latest_version(d) or {}).get("indexing_status") in ("pending", "processing")]
            if not pending:
                return
            if asyncio.get_event_loop().time() >= deadline:
                print(f"[RAG] attachments still indexing after {timeout:.0f}s, proceeding: {pending}")
                return
            await asyncio.sleep(0.7)

    @staticmethod
    def _visibility_where(user: Optional[dict], scope: Optional[dict]) -> dict:
        """
        Chroma `where` clause: only latest chunks, restricted to the documents the
        user may see (admins see all; others see org docs + their own), optionally
        narrowed to one or several documents. Pushed down into the vector store so
        a user's nearest-neighbour set is never polluted by chunks they can't access.

        `scope` narrows the search to specific documents (e.g. files attached in the
        composer): {"type": "document", "id": "..."} or {"type": "document",
        "ids": ["...", "..."]}.
        """
        conds: list[dict] = [{"is_latest": True}]
        ids = RAGEngine._scoped_ids(scope)
        if len(ids) == 1:
            conds.append({"rag_doc_id": ids[0]})
        elif len(ids) > 1:
            conds.append({"rag_doc_id": {"$in": ids}})
        if user and user.get("role") != "admin":
            conds.append({"$or": [{"scope": "org"}, {"owner_id": user["id"]}]})
        return {"$and": conds} if len(conds) > 1 else conds[0]

    def _rerank(self, items: list[dict], query: str) -> list[dict]:
        if not (self._reranker and items):
            return items
        try:
            from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
            nodes = [NodeWithScore(node=TextNode(text=it["text"], metadata=it["metadata"]),
                                   score=it["score"]) for it in items]
            ranked = self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
            return [{"text": n.node.get_content(), "metadata": n.node.metadata,
                     "score": float(n.score or 0)} for n in ranked]
        except Exception as e:
            print(f"[RAG] rerank error ({e})")
            return items

    def _relevant_nodes(self, message: str, scope: Optional[dict], user: Optional[dict]) -> list[dict]:
        """Hybrid retrieval: dense vector search + lexical BM25, fused with
        Reciprocal Rank Fusion, then optional rerank and a relevance filter.

        - Lexical BM25 catches exact terms/names/codes the vector search misses.
        - Scoped (attached-doc) queries skip the cosine cutoff (the user chose the doc).
        - Degrades gracefully to vector-only if the lexical index is unavailable."""
        where = self._visibility_where(user, scope)
        scoped_ids = self._scoped_ids(scope)
        scoped = bool(scoped_ids)

        try:
            qemb = Settings.embed_model.get_query_embedding(message)
        except Exception as e:
            print(f"[RAG] embed error ({e})")
            return []

        by_id: dict[str, dict] = {}
        vec_ids: list[str] = []
        # 1) Dense vector candidates (visibility pushed into Chroma).
        try:
            res = self._chroma_collection.query(
                query_embeddings=[qemb], n_results=RETRIEVE_CANDIDATES, where=where,
                include=["documents", "metadatas", "distances"],
            )
            for cid, t, m, d in zip((res.get("ids") or [[]])[0], (res.get("documents") or [[]])[0],
                                    (res.get("metadatas") or [[]])[0], (res.get("distances") or [[]])[0]):
                by_id[cid] = {"id": cid, "text": t or "", "metadata": m or {},
                              "score": 1.0 - float(d), "bm25": None}
                vec_ids.append(cid)
        except Exception as e:
            print(f"[RAG] vector retrieval error ({e}); where={where}")

        # 2) Lexical BM25 candidates (visibility filtered in Python).
        lex_ids: list[str] = []
        if HYBRID_ENABLED:
            bm = self._ensure_bm25()
            if bm is not None:
                try:
                    for cid, t, m, sc in bm.search(message, RETRIEVE_CANDIDATES * 3):
                        if not self._visible(m, user, scoped_ids):
                            continue
                        if cid in by_id:
                            by_id[cid]["bm25"] = sc
                        else:
                            by_id[cid] = {"id": cid, "text": t or "", "metadata": m or {},
                                          "score": None, "bm25": sc}
                        lex_ids.append(cid)
                        if len(lex_ids) >= RETRIEVE_CANDIDATES:
                            break
                except Exception as e:
                    print(f"[RAG] bm25 search error ({e})")

        if not by_id:
            print(f"[RAG] retrieve: where={where} scoped={scoped} raw=0 (no hits)")
            return []

        # 3) Reciprocal Rank Fusion of the two rankings.
        rv = {cid: i for i, cid in enumerate(vec_ids)}
        rl = {cid: i for i, cid in enumerate(lex_ids)}
        K = 60
        fused = sorted(
            by_id.values(),
            key=lambda it: (1.0 / (K + rv[it["id"]]) if it["id"] in rv else 0.0)
                           + (1.0 / (K + rl[it["id"]]) if it["id"] in rl else 0.0),
            reverse=True,
        )

        # 4) Cosine score for BM25-only hits (for the cutoff + source display).
        missing = [it for it in fused if it["score"] is None]
        if missing:
            try:
                got = self._chroma_collection.get(ids=[it["id"] for it in missing], include=["embeddings"])
                emap = dict(zip(got.get("ids", []) or [], got.get("embeddings", []) or []))
                for it in missing:
                    e = emap.get(it["id"])
                    it["score"] = _cosine(qemb, e) if e is not None else RELEVANCE_CUTOFF
            except Exception as e:
                print(f"[RAG] embedding fetch error ({e})")
                for it in missing:
                    it["score"] = RELEVANCE_CUTOFF

        raw = len(fused)
        top_raw = round(max((it["score"] for it in fused), default=0.0), 4)

        # 5) Relevance filter: keep lexical (exact-term) hits OR vector hits above
        # the cutoff. Scoped queries keep everything (the document choice is the signal).
        if scoped:
            kept = fused
        else:
            kept = [it for it in fused if it.get("bm25") is not None or (it["score"] or 0) >= RELEVANCE_CUTOFF]

        kept = self._rerank(kept, message)
        result = kept[:RETRIEVE_TOP_K]
        print(f"[RAG] retrieve: where={where} scoped={scoped} vec={len(vec_ids)} bm25={len(lex_ids)} "
              f"fused={raw} top_score={top_raw} kept={len(kept)} returned={len(result)} "
              f"cutoff={RELEVANCE_CUTOFF}")
        return result

    # ─── Lexical index (BM25) ───────────────────────────────────────────────────

    @staticmethod
    def _visible(md: dict, user: Optional[dict], scoped_ids: list[str]) -> bool:
        """Python mirror of _visibility_where, for filtering BM25 results."""
        if not md.get("is_latest"):
            return False
        if scoped_ids and md.get("rag_doc_id") not in scoped_ids:
            return False
        if user and user.get("role") != "admin":
            if md.get("scope") != "org" and md.get("owner_id") != user.get("id"):
                return False
        return True

    def _invalidate_bm25(self):
        self._bm25_dirty = True

    def _ensure_bm25(self):
        """(Re)build the lexical index over all latest chunks when stale."""
        if self._bm25 is not None and not self._bm25_dirty:
            return self._bm25
        try:
            got = self._chroma_collection.get(where={"is_latest": True},
                                              include=["documents", "metadatas"])
            ids = got.get("ids", []) or []
            self._bm25 = _BM25(ids, got.get("documents", []) or [], got.get("metadatas", []) or [])
            self._bm25_dirty = False
            print(f"[RAG] BM25 lexical index built over {len(ids)} latest chunks.")
        except Exception as e:
            print(f"[RAG] BM25 build failed ({e}); vector-only this query")
            self._bm25, self._bm25_dirty = None, False
        return self._bm25

    @staticmethod
    def _context_block(items: list[dict]) -> str:
        parts = []
        for it in items:
            md = it["metadata"]
            head = md.get("file_name", "document")
            if md.get("version"):
                head += f" v{md['version']}"
            page = md.get("page_label")
            if page and page != "?":
                head += f" p.{page}"
            parts.append(f"[{head}]\n{it['text']}")
        return "\n\n".join(parts)

    def generate_title(self, message: str) -> str:
        fallback = " ".join((message or "").split())[:48] or "Conversation"
        try:
            prompt = _language_directive(message) + "\n\n" + TITLE_PROMPT.format(
                message=(message or "")[:500]
            )
            resp = (self._title_llm or Settings.llm).complete(prompt.strip())
            title = (resp.text or "").strip().strip('"').strip()
            return (title.splitlines()[0][:60] if title else "") or fallback
        except Exception as e:
            print(f"[RAG] title generation error ({e})")
            return fallback

    def _should_use_docs(self, message: str, history: list[dict]) -> bool:
        # Deprecated: retrieval is now always attempted when the KB is non-empty
        # (the relevance cutoff decides grounding). Kept as a no-op for safety; the
        # old LLM yes/no gate misfired for non-English answers. Always returns True.
        return True

    async def stream_chat(self, message: str, history: list[dict],
                          scope: Optional[dict] = None, user: Optional[dict] = None,
                          strict: Optional[bool] = None):
        """
        Yields ("token", str)… then ("sources", list).

        Retrieves over latest-version chunks the user is allowed to see (visibility
        pushed into Chroma), optionally scoped to one or more attached documents,
        and streams the answer with the retrieved context injected.

        When the query is explicitly scoped to attached documents we (a) wait for
        their indexing to finish so a question asked right after attaching isn't
        answered from an empty index, and (b) ALWAYS use the grounded prompt — so
        if no extract is found the model says it couldn't find it instead of
        hallucinating an unrelated answer.
        """
        loop = asyncio.get_event_loop()
        directive = _language_directive(message)
        strict = STRICT_RAG_DEFAULT if strict is None else bool(strict)
        scoped_ids = self._scoped_ids(scope)
        scoped = bool(scoped_ids)

        items = []
        if scoped:
            # Explicit attachment(s): block until they are indexed, then ground on them.
            await self._await_attachments_indexed(scoped_ids)
            items = await loop.run_in_executor(None, self._relevant_nodes, message, scope, user)
            print(f"[RAG] chat: scoped_docs={scoped_ids} items={len(items)} "
                  f"role={(user or {}).get('role')}")
        elif self.doc_count() > 0:
            # Always retrieve when the KB is non-empty. A previous LLM yes/no
            # "router" gated this per message, but it misfired badly for non-English
            # replies — the model answering "Oui"/"نعم" was parsed as "no search" —
            # so French/Arabic questions were answered WITHOUT the documents. The
            # relevance cutoff already decides whether the hits are good enough to
            # ground on, so the gate was both redundant and harmful.
            items = await loop.run_in_executor(None, self._relevant_nodes, message, scope, user)
            print(f"[RAG] chat: doc_count={self.doc_count()} items={len(items)} "
                  f"role={(user or {}).get('role')}")
        else:
            print("[RAG] chat: knowledge base is empty (doc_count=0) — answering conversationally")

        use_docs = len(items) > 0
        base_prompt = SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT
        if use_docs:
            system_text = (
                base_prompt + directive + "\n\n"
                + CONTEXT_TEMPLATE.format(context_str=self._context_block(items))
            )
        elif scoped or strict:
            # Scoped to attachments (nothing retrievable) OR strict mode with no
            # match: stay grounded and instruct the model to say it couldn't find
            # the answer rather than inventing one or chatting from general knowledge.
            no_ctx = (
                "(Aucun extrait pertinent n'a été trouvé dans les documents pour cette "
                "question.)"
            )
            system_text = (
                base_prompt + directive + "\n\n"
                + CONTEXT_TEMPLATE.format(context_str=no_ctx)
            )
        else:
            system_text = CONV_SYSTEM_PROMPT + directive

        msgs = [ChatMessage(role=MessageRole.SYSTEM, content=system_text)]
        for m in _trim_history(history):
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

        yield ("sources", self._format_sources(items, scoped=scoped) if use_docs else [])

    @staticmethod
    def _format_sources(items: list[dict], scoped: bool = False) -> list[dict]:
        """Build the user-facing sources list. For normal questions, show only the
        clearly-relevant matches (near the best score, above a floor) so an unrelated
        long tail isn't displayed and small talk shows nothing. For scoped
        (attached-document) queries, always surface the chosen document(s).
        One entry per file (collapses multi-chunk hits and org/personal same-name)."""
        if not items:
            return []
        ranked = sorted(items, key=lambda it: float(it.get("score") or 0), reverse=True)
        top = float(ranked[0].get("score") or 0)
        sources: list[dict] = []
        seen = set()  # one source per file
        for it in ranked:
            score = float(it.get("score") or 0)
            if not scoped:
                # Sorted desc: once we hit a weak/out-of-gap chunk, all the rest are
                # worse, so stop. This drops the irrelevant tail and (when even the
                # best match is weak, e.g. greetings) shows no source.
                if score < SOURCE_MIN_SCORE:
                    break
                if sources and score < top - SOURCE_REL_GAP:
                    break
            md = it.get("metadata") or {}
            key = md.get("file_name") or md.get("rag_doc_id")
            if not key or key in seen:
                continue
            seen.add(key)
            text = it.get("text") or ""
            sources.append({
                "text": text[:320] + ("…" if len(text) > 320 else ""),
                "score": round(score, 4),
                "filename": md.get("file_name", "Unknown"),
                "page": md.get("page_label", "?"),
                "document_id": md.get("rag_doc_id"),
                "version": md.get("version"),
            })
            if len(sources) >= SOURCE_MAX:
                break
        print(f"[RAG] sources: scoped={scoped} top={round(top, 4)} "
              f"shown={len(sources)}/{len(items)} -> {[(s['filename'], s['score']) for s in sources]}")
        return sources

    # ─── Ingestion & version management ────────────────────────────────────────

    def index_version(self, file_path: str, filename: str, document_id: str, version: int,
                      owner_id: Optional[str] = None, scope: str = "org") -> int:
        """Chunk/embed a file as a new version, mark it is_latest, demote older ones."""
        with self._write_lock:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            nodes = Settings.node_parser.get_nodes_from_documents(documents)
            for node in nodes:
                node.metadata.update({
                    "file_name": filename,
                    "source": filename,
                    # LlamaIndex reserves and OVERWRITES "document_id" with its own
                    # ref_doc_id on insert, so a where={"document_id": ...} filter
                    # never matches our id. Store our identity under "rag_doc_id"
                    # and filter on that everywhere.
                    "rag_doc_id": document_id,
                    "version": version,
                    "is_latest": True,
                    "owner_id": owner_id or "",   # Chroma metadata can't be None
                    "scope": scope,
                })
            self._index.insert_nodes(nodes)
            self._demote_other_versions(document_id, keep_version=version)
            self._invalidate_bm25()
            return len(nodes)

    def _set_latest_for_document(self, document_id: str, latest_version):
        """Set is_latest=true only for chunks of `latest_version` (None => none)."""
        try:
            got = self._chroma_collection.get(where={"rag_doc_id": document_id}, include=["metadatas"])
            ids = got.get("ids", []) or []
            mds = got.get("metadatas", []) or []
            cids, cmds = [], []
            for cid, md in zip(ids, mds):
                want = latest_version is not None and md.get("version") == latest_version
                if bool(md.get("is_latest")) != want:
                    md = dict(md); md["is_latest"] = want
                    cids.append(cid); cmds.append(md)
            if cids:
                self._chroma_collection.update(ids=cids, metadatas=cmds)
            self._invalidate_bm25()
        except Exception as e:
            print(f"[RAG] set-latest error ({e})")

    def _demote_other_versions(self, document_id: str, keep_version: int):
        self._set_latest_for_document(document_id, keep_version)

    def set_document_hidden(self, document_id: str):
        """Soft-delete: remove all of a document's chunks from retrieval."""
        with self._write_lock:
            self._set_latest_for_document(document_id, None)

    def restore_version(self, document_id: str, version: int):
        """Rollback / undelete: make `version`'s chunks the only latest ones."""
        with self._write_lock:
            self._set_latest_for_document(document_id, version)

    def purge_document(self, document_id: str):
        """Hard-delete: permanently remove all of a document's vectors."""
        with self._write_lock:
            try:
                self._chroma_collection.delete(where={"rag_doc_id": document_id})
                self._invalidate_bm25()
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
        client = chromadb.PersistentClient(path=CHROMA_PATH, settings=CHROMA_SETTINGS)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._setup()
        print("[RAG] Knowledge base reset.")