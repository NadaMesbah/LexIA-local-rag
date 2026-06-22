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
import unicodedata
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
# Optional: drop reranked candidates whose cross-encoder score is below this
# (0 = disabled). Lets a serious test prune off-topic passages even after rerank,
# without affecting the cosine-based source display. Scale depends on the model.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))
# Hybrid lexical (BM25) retrieval on top of dense vectors — catches exact terms,
# names, codes, dates and short queries that pure vector search misses.
HYBRID_ENABLED  = os.getenv("HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
# Minimum BM25 score for a lexical-only hit to be trusted on its own. BM25 gives
# *some* score to any shared word, so common terms create weak, noisy matches
# (e.g. an ML paper surfacing for a news question). A lexical hit below this is
# kept ONLY if its dense cosine also passes RELEVANCE_CUTOFF. 0 = keep any hit.
MIN_BM25_SCORE  = float(os.getenv("MIN_BM25_SCORE", "0.0"))
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

# Source DISPLAY layering. The answer is grounded on the retrieved context, so the
# single best chunk is ALWAYS shown as a source (no display threshold may hide it).
# These knobs only affect ADDITIONAL cards / ordering:
#   - SOURCE_DISPLAY_TOP_K   max cards shown (default 1 = highest precision)
#   - SOURCE_DISPLAY_MIN_RERANK  min rerank score for a SECONDARY card (reranker ON)
SOURCE_DISPLAY_MIN_RERANK = float(os.getenv("SOURCE_DISPLAY_MIN_RERANK", "0.0"))
SOURCE_DISPLAY_TOP_K = int(os.getenv("SOURCE_DISPLAY_TOP_K", "1"))

# Rewrite follow-up questions into standalone retrieval queries using recent
# conversation, so anaphora ("this complex", "ce projet", "بهذا المركب") resolves
# before retrieval. Used ONLY for retrieval — the answer still uses the original
# message. Skipped on the first turn (no history).
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE   = os.getenv("RERANK_DEVICE", "cpu")

MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "3000"))

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}

SYSTEM_PROMPT = (
    "You are a precise, helpful assistant for a private knowledge base. "
    "Use the context extracts below when they are relevant, and prefer them over "
    "prior knowledge for facts about the documents. NEVER fabricate specific figures, "
    "amounts, numbers, dates or names: state such details ONLY if they appear in the "
    "extracts, copying them exactly. If the question is about the documents and the "
    "answer (or the requested figure) is not in the extracts, say clearly (in the "
    "user's language) that you couldn't find it — do NOT guess. For general questions "
    "or small talk unrelated to the documents, answer naturally. "
    "Do NOT add a 'Sources', 'Source' or citations list to your answer, and do NOT "
    "mention file names or version numbers — the application displays the sources "
    "separately. Be clear, concise and well structured."
)

# Used ONLY when strict RAG is on (opt-in): documents-only, refuse if absent.
SYSTEM_PROMPT_STRICT = (
    "You are a STRICT retrieval assistant for a private knowledge base. Answer ONLY "
    "from the context extracts below. Never use general or prior knowledge to complete "
    "or guess. If the exact answer is not explicitly present in the extracts, reply with "
    "ONLY this sentence, in the user's language and nothing else:\n"
    "- English: \"I could not find this information in the provided documents.\"\n"
    "- French: \"Je n’ai pas trouvé cette information dans les documents fournis.\"\n"
    "- Arabic: \"لم أجد هذه المعلومة في الوثائق المقدمة.\"\n"
    "Do NOT add a sources/citations list or mention file names — the app shows sources "
    "separately. Be concise and faithful to the extracts."
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

# Rewrites a follow-up into a standalone retrieval query (resolves pronouns/deixis
# using recent turns). The model must answer in the SAME language and output ONLY
# the query — it is never shown to the user and never used for the final answer.
QUERY_REWRITE_PROMPT = (
    "You rewrite a user's latest question into a SINGLE standalone search query.\n"
    "Resolve pronouns and references (e.g. \"this\", \"it\", \"ce projet\", \"بهذا\") using "
    "the conversation so the query makes sense on its own. Keep the SAME language as "
    "the question. Do not answer it. Output ONLY the rewritten query, no quotes, no "
    "explanation.\n\nConversation:\n{history}\n\nLatest question: {question}\n\nStandalone query:"
)


# ─── Language lock (prevents Qwen drifting into Chinese) ──────────────────────
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_AR = re.compile(r"[\u0600-\u06ff]")
_CYR = re.compile(r"[\u0400-\u04ff]")
_FR_ACC = re.compile(r"[àâçéèêëîïôûùüœ]")


def _scripts(text: str) -> set:
    """The set of writing systems present in `text` (latin/arabic/cyrillic/cjk).
    Used to reject a query rewrite that injects a foreign script (e.g. the Latin
    word 'Mercer' leaking from a corrupted assistant answer into an Arabic query)."""
    out = set()
    for ch in text or "":
        o = ord(ch)
        if "a" <= ch.lower() <= "z":
            out.add("latin")
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            out.add("arabic")
        elif 0x0400 <= o <= 0x04FF:
            out.add("cyrillic")
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            out.add("cjk")
    return out


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
    if lang == "Arabic":
        return (
            "\n\nABSOLUTE RULE — LANGUAGE: The user is writing in Arabic. Write your ENTIRE "
            "answer in Modern Standard Arabic only. Do NOT output any Chinese, Latin or other "
            "non-Arabic words (numbers and units copied from the extracts are fine). If an "
            "extract contains garbled or non-Arabic fragments, IGNORE them and never copy "
            "them. This overrides everything else."
        )
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


# ─── Strict-mode refusal (deterministic, language-aware) ──────────────────────
# When strict RAG is on and retrieval finds no RELIABLE grounding, we refuse with
# one of these exact messages instead of letting the model improvise (or
# hallucinate). The message language follows the user's question.
STRICT_REFUSALS = {
    "English": "I could not find this information in the provided documents.",
    "French": "Je n’ai pas trouvé cette information dans les documents fournis.",
    "Arabic": "لم أجد هذه المعلومة في الوثائق المقدمة.",
}
STRICT_REFUSAL_DEFAULT = STRICT_REFUSALS["English"]


def _strict_refusal(message: str) -> str:
    """The canonical refusal in the user's language (English fallback)."""
    return STRICT_REFUSALS.get(_detect_language(message), STRICT_REFUSAL_DEFAULT)


def _no_context_note(message: str) -> str:
    """Language-aware placeholder used as the 'context' when nothing was retrieved
    (scoped non-strict), so the model says it couldn't find it in the right language."""
    lang = _detect_language(message)
    if lang == "Arabic":
        return "(لم يتم العثور على أي مقطع ذي صلة في الوثائق لهذا السؤال.)"
    if lang == "English":
        return "(No relevant extract was found in the documents for this question.)"
    return "(Aucun extrait pertinent n'a été trouvé dans les documents pour cette question.)"


# ─── Lexical search (BM25) for hybrid retrieval ───────────────────────────────
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Arabic diacritics (tashkeel), the superscript alef and the decorative tatweel.
# They are optional in Arabic, are frequently mis-emitted by PDF extractors, read
# poorly for the LLM and break matching — so we drop them everywhere.
_AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670\u0640]"
)


def _normalize_text(text: str) -> str:
    # NFKC folds Arabic presentation-form glyphs (e.g. ﺍﻟﺤﺪ) back to standard
    # letters (الحد). PDF extractors often emit presentation forms, which (a) read
    # poorly for the LLM and (b) use different codepoints than a user's typed query
    # — breaking BM25/embedding matches. Normalizing fixes all three. We also strip
    # Arabic diacritics/tatweel (optional marks, often PDF noise).
    t = unicodedata.normalize("NFKC", text or "")
    return _AR_DIACRITICS.sub("", t)


# Aggressive Arabic letter folding for LEXICAL MATCHING ONLY (never applied to the
# stored/displayed text): unify the alef/ya/hamza/ta-marbuta variants a user might
# type differently than the document, so BM25 matches regardless of spelling.
_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})


def _fold_arabic(text: str) -> str:
    """Fold Arabic spelling variants for matching (no-op for non-Arabic text)."""
    return (text or "").translate(_AR_FOLD)


# Obvious corruption to remove from a chunk before it enters the LLM context: CJK
# (this KB is Arabic/French/English, so any Chinese is bad PDF extraction), the
# Unicode replacement char (decode errors) and zero-width / bidi control glyphs.
_CONTEXT_GARBAGE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\ufffd"
    r"\u200b-\u200f\u202a-\u202e\u2060\ufeff]"
)


def _clean_context_text(text: str) -> str:
    """Strip obviously-corrupted artifacts from a chunk before it goes into the
    LLM context, so the model can't echo garbage (e.g. Chinese fragments inside an
    Arabic answer). Conservative: only removes CJK/replacement/control characters,
    never legitimate Arabic/Latin content."""
    t = _CONTEXT_GARBAGE.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def _tokenize(text: str) -> list[str]:
    # Normalize first so Arabic written as presentation forms matches a query typed
    # in standard Arabic. \w with re.UNICODE keeps Arabic/accented letters + digits.
    return _WORD_RE.findall(_normalize_text(text).lower())


# ─── Stopwords (lexical noise reduction for BM25) ─────────────────────────────
# Removing stopwords from BM25 tokenization stops ultra-common words from creating
# spurious lexical matches across unrelated documents. Lists are deliberately
# focused on the three supported UI languages (English, French, Arabic).
_STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "from", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they", "what",
    "which", "who", "whom", "how", "why", "when", "where", "do", "does", "did", "can",
    "could", "will", "would", "should", "shall", "may", "might", "must", "have", "has",
    "had", "not", "no", "yes", "please", "hello", "hi", "hey", "thanks", "thank", "about",
    "into", "over", "than", "then", "there", "here", "your", "my", "our", "their", "his",
    "her", "me", "us", "them", "so", "such", "very", "more", "most", "some", "any", "all",
    "each", "both", "few", "much", "many", "give", "tell", "show", "explain", "summarize",
}
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "donc", "or",
    "ni", "car", "est", "sont", "était", "étaient", "être", "été", "à", "au", "aux", "ce",
    "cet", "cette", "ces", "il", "elle", "ils", "elles", "on", "je", "tu", "nous", "vous",
    "que", "qui", "quoi", "dont", "où", "quel", "quelle", "quels", "quelles", "comment",
    "pourquoi", "quand", "combien", "ne", "pas", "plus", "moins", "pour", "avec", "sans",
    "sur", "sous", "dans", "par", "en", "se", "sa", "son", "ses", "leur", "leurs", "mon",
    "ma", "mes", "ton", "ta", "tes", "notre", "nos", "votre", "vos", "qu", "aussi",
    "alors", "comme", "tout", "tous", "toute", "toutes", "bonjour", "salut", "merci",
    "oui", "non", "ça", "cela", "ceci", "explique", "résume", "donne", "dis",
}
_STOPWORDS_AR = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "الذي",
    "التي", "الذين", "ما", "ماذا", "متى", "أين", "اين", "كيف", "لماذا", "كم", "هل", "أن",
    "ان", "إن", "كان", "كانت", "يكون", "هو", "هي", "هم", "هن", "نحن", "أنا", "انا", "أنت",
    "انت", "أنتم", "لا", "نعم", "هنا", "هناك", "كل", "بعض", "قد", "ثم", "أو", "او", "أم",
    "لكن", "حتى", "إذا", "اذا", "عند", "عندما", "بين", "كذلك", "أيضا", "ايضا", "غير",
}
# Normalize stopwords the SAME way tokens are (NFKC + diacritic strip + Arabic
# folding + lowercase) so they match the corpus/query tokens exactly.
_STOPWORDS = {_fold_arabic(_normalize_text(w)).lower()
              for w in (_STOPWORDS_EN | _STOPWORDS_FR | _STOPWORDS_AR)}


def _content_tokens(text: str) -> list[str]:
    """Tokens used for lexical (BM25) matching: normalized + Arabic-folded +
    lowercased, with stopwords and single non-digit characters (e.g. French
    elisions l'/d'/qu') dropped, so common words don't manufacture noisy
    cross-document matches and Arabic spelling variants still match."""
    out: list[str] = []
    for t in _tokenize(text):
        t = _fold_arabic(t)
        if t in _STOPWORDS:
            continue
        if len(t) > 1 or t.isdigit():
            out.append(t)
    return out


def _to_list(x) -> list:
    """Coerce a Chroma/NumPy result to a plain Python list.

    Chroma returns `embeddings` and `distances` as NumPy ndarrays. Putting an
    ndarray in a boolean context — ``arr or default``, ``if arr:``, ``not arr`` —
    raises "The truth value of an array with more than one element is ambiguous".
    NEVER test those results with truthiness; pass them through here first, then
    test with explicit ``is None`` / ``len(...)``.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    tolist = getattr(x, "tolist", None)
    if callable(tolist):          # numpy ndarray (or scalar) -> nested py list
        return tolist()
    try:
        return list(x)            # any other iterable
    except TypeError:
        return [x]


def _first_row(x) -> list:
    """First row of a Chroma query result (shape ``[[...]]``), NumPy-safe."""
    rows = _to_list(x)
    return _to_list(rows[0]) if rows else []


def _cosine(a, b) -> float:
    # NumPy-safe: Chroma returns embeddings as numpy arrays, so avoid truthiness
    # checks like `not a` (ambiguous for arrays) — use explicit None/length tests.
    if a is None or b is None:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else 0.0


class _BM25:
    """Minimal BM25-Okapi over an in-memory chunk corpus. Pure Python, no extra
    dependency, fully offline — the lexical half of hybrid retrieval, catching
    exact terms/names/codes/dates that dense vectors miss."""

    def __init__(self, ids: list, texts: list, metas: list, k1: float = 1.5, b: float = 0.75):
        self.ids, self.texts, self.metas, self.k1, self.b = ids, texts, metas, k1, b
        docs = [_content_tokens(t) for t in texts]
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
        q = _content_tokens(query)
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

    def _rerank(self, items: list[dict], query: str):
        """Cross-encoder rerank of the candidate pool.

        Returns ``(items, applied)``. When the reranker is enabled it scores EVERY
        candidate (not a pre-filtered subset) and returns them ordered by true
        query relevance, with the cross-encoder score attached as ``rerank``. The
        original cosine ``score`` and ``bm25`` fields are PRESERVED — source display
        and the relevance cutoff rely on the cosine score, which is on a different
        scale than the reranker's. When disabled/unavailable, returns the input
        unchanged with ``applied=False``."""
        if not (self._reranker and items):
            return items, False
        try:
            from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
            # id_ = index so we can map the reranked nodes back to the rich source
            # dicts (preserving bm25/score/metadata) instead of rebuilding them.
            nodes = [
                NodeWithScore(
                    node=TextNode(text=it.get("text") or "",
                                  metadata=it.get("metadata") or {}, id_=str(i)),
                    score=float(it.get("score") or 0.0),
                )
                for i, it in enumerate(items)
            ]
            ranked = self._reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))
            out = []
            for n in ranked:
                try:
                    src = dict(items[int(n.node.id_)])
                except (ValueError, IndexError, TypeError):
                    src = {"text": n.node.get_content(), "metadata": n.node.metadata,
                           "score": float(n.score or 0), "bm25": None}
                src["rerank"] = float(n.score or 0.0)
                out.append(src)
            return out, True
        except Exception as e:
            print(f"[RAG] rerank error ({e})")
            return items, False

    @staticmethod
    def _final_select(reranked: list[dict], scoped: bool, reranked_applied: bool) -> list[dict]:
        """The post-rerank selection (aggressive filtering happens HERE, after the
        reranker has seen the wide pool):
          - scoped queries keep everything (the user chose the document);
          - reranker ON  → trust its order, optionally drop very low rerank scores
            (RERANK_MIN_SCORE), then keep the top RERANK_TOP_N;
          - reranker OFF → cosine relevance cutoff, still keeping exact BM25 hits.
        The caller applies the final RETRIEVE_TOP_K cap."""
        if scoped:
            return list(reranked)
        if reranked_applied:
            pool = reranked
            if RERANK_MIN_SCORE > 0:
                strong = [it for it in pool if (it.get("rerank") or 0.0) >= RERANK_MIN_SCORE]
                pool = strong or pool      # never filter everything away
            return pool[:RERANK_TOP_N]
        return [it for it in reranked
                if (it.get("bm25") is not None and it["bm25"] >= MIN_BM25_SCORE)
                or (it.get("score") or 0) >= RELEVANCE_CUTOFF]

    def _relevant_nodes(self, message: str, scope: Optional[dict], user: Optional[dict],
                        debug: bool = False):
        """Hybrid retrieval: dense vector search + lexical BM25, fused with
        Reciprocal Rank Fusion, then optional rerank and a relevance filter.

        - Lexical BM25 catches exact terms/names/codes the vector search misses.
        - Scoped (attached-doc) queries skip the cosine cutoff (the user chose the doc).
        - Degrades gracefully to vector-only if the lexical index is unavailable.

        Returns the final list of chunk dicts. When ``debug=True`` it instead
        returns a ``{"summary": {...}, "items": [...]}`` diagnostics dict describing
        every fused candidate and how it was scored/filtered (for /debug/retrieve)."""
        where = self._visibility_where(user, scope)
        scoped_ids = self._scoped_ids(scope)
        scoped = bool(scoped_ids)

        try:
            qemb = Settings.embed_model.get_query_embedding(_normalize_text(message))
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
            # NumPy-safe extraction: `distances` comes back as a NumPy array, so we
            # must not use `res.get("distances") or [[]]` (ambiguous truth value).
            r_ids = _first_row(res.get("ids"))
            r_docs = _first_row(res.get("documents"))
            r_metas = _first_row(res.get("metadatas"))
            r_dists = _first_row(res.get("distances"))
            for cid, t, m, d in zip(r_ids, r_docs, r_metas, r_dists):
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
            if debug:
                return {"summary": self._debug_summary(0, 0, 0, 0, False, 0, 0, 0, 0.0, True),
                        "items": []}
            return []

        # 3) Reciprocal Rank Fusion of the two rankings. The fused score and
        # provenance (dense / bm25 / both) are stored on each item for diagnostics.
        rv = {cid: i for i, cid in enumerate(vec_ids)}
        rl = {cid: i for i, cid in enumerate(lex_ids)}
        K = 60
        for it in by_id.values():
            cid = it["id"]
            in_v, in_l = cid in rv, cid in rl
            it["fused"] = ((1.0 / (K + rv[cid]) if in_v else 0.0)
                           + (1.0 / (K + rl[cid]) if in_l else 0.0))
            it["source"] = "both" if (in_v and in_l) else ("dense" if in_v else "bm25")
        fused = sorted(by_id.values(), key=lambda it: it["fused"], reverse=True)

        # 4) Cosine score for BM25-only hits (for the cutoff + source display).
        missing = [it for it in fused if it["score"] is None]
        if missing:
            try:
                got = self._chroma_collection.get(ids=[it["id"] for it in missing],
                                                  include=["embeddings"])
                # NumPy-safe: `embeddings` is a NumPy array — coerce to plain lists
                # and test membership with `is None`, never with truthiness.
                got_ids = _to_list(got.get("ids"))
                got_embs = _to_list(got.get("embeddings"))
                emap = {cid: emb for cid, emb in zip(got_ids, got_embs)}
                for it in missing:
                    e = emap.get(it["id"])
                    it["score"] = _cosine(qemb, e) if e is not None else RELEVANCE_CUTOFF
            except Exception as e:
                print(f"[RAG] embedding fetch error ({e})")
                for it in missing:
                    it["score"] = RELEVANCE_CUTOFF

        raw = len(fused)
        top_raw = round(max((it["score"] for it in fused), default=0.0), 4)

        # 5) LIGHT pre-filter only. We deliberately do NOT apply the aggressive
        # relevance cutoff here — that happens AFTER reranking, so the cross-encoder
        # gets a chance to rank borderline-but-relevant passages. We only drop empty
        # chunks and cap the pool the reranker has to score.
        candidates = [it for it in fused if (it.get("text") or "").strip()][:RETRIEVE_CANDIDATES]

        # 6) Rerank the FULL candidate pool (cross-encoder if enabled): it scores
        # every candidate and returns them ordered by true query relevance.
        reranked, reranked_applied = self._rerank(candidates, message)

        # 7) Final selection — the aggressive filtering, now AFTER rerank.
        selected = self._final_select(reranked, scoped, reranked_applied)
        result = selected[:RETRIEVE_TOP_K]
        print(f"[RAG] retrieve: where={where} scoped={scoped} vec={len(vec_ids)} bm25={len(lex_ids)} "
              f"fused={raw} cand={len(candidates)} rerank={'on' if reranked_applied else 'off'} "
              f"top_score={top_raw} selected={len(selected)} returned={len(result)} "
              f"cutoff={RELEVANCE_CUTOFF}")
        if debug:
            return self._build_debug(message, scoped, vec_ids, lex_ids, fused, candidates,
                                     reranked, reranked_applied, selected, result, top_raw)
        return result

    # ─── Retrieval diagnostics (/debug/retrieve) ────────────────────────────────

    @staticmethod
    def _debug_summary(dense, bm25, fused, candidates, rerank_on, after_filter,
                       returned, shown, top_cosine, strict_refuse):
        return {
            "dense_results": dense,
            "bm25_results": bm25,
            "fused_results": fused,
            "candidates": candidates,
            "reranker": "on" if rerank_on else "off",
            "after_filter": after_filter,
            "returned": returned,
            "shown_sources": shown,
            "top_cosine": round(float(top_cosine), 4),
            "strict_decision": "refuse" if strict_refuse else "answer",
            "relevance_cutoff": RELEVANCE_CUTOFF,
            "min_bm25_score": MIN_BM25_SCORE,
        }

    @staticmethod
    def _build_debug(query, scoped, vec_ids, lex_ids, fused, candidates,
                     reranked, reranked_applied, selected, result, top_raw):
        """Per-candidate + summary diagnostics describing how each fused candidate
        was scored and why it was kept or filtered — and whether it is shown as a
        source card."""
        rerank_map = {it["id"]: it.get("rerank") for it in reranked}
        cand_ids = {it["id"] for it in candidates}
        selected_ids = {it["id"] for it in selected}
        final_ids = {it["id"] for it in result}
        # Which final chunks become source cards (strict display layer).
        display_items = RAGEngine._select_source_items(result, scoped, reranked_applied)
        display_ids = {it["id"] for it in display_items}

        def reason(it):
            cid = it["id"]
            if cid in final_ids:
                return True, "returned in final top-k"
            if cid not in cand_ids:
                return False, "dropped at light pre-filter (empty/over candidate cap)"
            if cid in selected_ids:
                return False, "selected but beyond RETRIEVE_TOP_K"
            if scoped:
                return False, "beyond RETRIEVE_TOP_K (scoped keeps all)"
            if reranked_applied:
                rk = it.get("rerank") or 0.0
                if RERANK_MIN_SCORE > 0 and rk < RERANK_MIN_SCORE:
                    return False, f"rerank {round(rk, 4)} < RERANK_MIN_SCORE {RERANK_MIN_SCORE}"
                return False, "beyond RERANK_TOP_N"
            bm, sc = it.get("bm25"), (it.get("score") or 0)
            if (bm is not None and bm >= MIN_BM25_SCORE) or sc >= RELEVANCE_CUTOFF:
                return False, "passed relevance but beyond RETRIEVE_TOP_K"
            detail = f"cosine {round(sc, 4)} < {RELEVANCE_CUTOFF}"
            detail += (f" and bm25 {round(bm, 4)} < {MIN_BM25_SCORE}"
                       if bm is not None else " and no BM25 hit")
            return False, f"weak: {detail}"

        items = []
        for idx, it in enumerate(fused):
            md = it.get("metadata") or {}
            passed, why = reason(it)
            rk = rerank_map.get(it["id"])
            shown = it["id"] in display_ids
            if shown:
                why = (why + "; shown as source card") if passed else "shown as source card"
            elif passed and not RAGEngine._is_lexical_only(it):
                pass
            elif passed:
                why = why + "; hidden from sources (lexical-only / off-topic)"
            items.append({
                "chunk_id": it.get("id"),
                "chunk_index": idx,
                "score": round(float(it.get("score") or 0), 4),       # dense / cosine
                "bm25": (round(it["bm25"], 4) if it.get("bm25") is not None else None),
                "fused": round(float(it.get("fused") or 0), 6),
                "rerank": (round(rk, 4) if rk is not None else None),
                "source": it.get("source"),
                "filename": md.get("file_name"),
                "rag_doc_id": md.get("rag_doc_id"),
                "version": md.get("version"),
                "page": md.get("page_label"),
                "passed_filter": passed,
                "displayed_as_source": shown,
                "reason": why,
                "text": RAGEngine._snippet(it.get("text") or "", query),
            })

        strict_refuse = RAGEngine._strict_should_refuse(len(result) > 0)
        summary = RAGEngine._debug_summary(len(vec_ids), len(lex_ids), len(fused), len(candidates),
                                           reranked_applied, len(selected), len(result),
                                           len(display_ids), top_raw, strict_refuse)
        summary["original_query"] = query
        summary["retrieval_query"] = query   # /debug/retrieve has no history; rewrite is a no-op
        return {"summary": summary, "items": items}

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
            parts.append(f"[{head}]\n{_clean_context_text(it['text'])}")
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

    def _contextualize_query(self, message: str, history: list[dict]) -> str:
        """Rewrite a follow-up into a standalone retrieval query using recent turns,
        resolving anaphora (e.g. 'بهذا المركب' → 'المركب السياحي كيمادو'). Returns the
        ORIGINAL message on the first turn, when disabled, or whenever the rewrite
        looks unsafe. Used for retrieval only — never shown, never used to answer.

        Hardened against contamination: prior assistant answers can be wrong/corrupted
        (e.g. a hallucinated 'Mercer'), so the rewrite is REJECTED if it introduces a
        foreign script, contains corruption, drifts off-topic, or is malformed — in
        which case we fall back to the (unmodified) original question."""
        if not (QUERY_REWRITE_ENABLED and history):
            return message
        try:
            recent = _trim_history(history)[-6:]
            if not recent:
                return message
            # Clean each turn so corrupted glyphs (CJK/control) don't seed the rewrite.
            convo = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {_clean_context_text(m['content'])}"
                for m in recent
            )[:2000]
            prompt = QUERY_REWRITE_PROMPT.format(history=convo, question=message)
            resp = (self._title_llm or Settings.llm).complete(prompt.strip())
            rewritten = ((resp.text or "").strip().strip('"').strip())
            rewritten = _clean_context_text(rewritten.splitlines()[0]) if rewritten else ""

            def reject(reason):
                print(f"[RAG] query-rewrite REJECTED ({reason}): {message!r} -> {rewritten!r}")
                return message

            if not rewritten or rewritten.lower() == message.strip().lower():
                return message
            if len(rewritten) > len(message) * 4 + 120:
                return reject("too long")
            # Anti-contamination: the rewrite must not introduce a writing system
            # absent from the user's question (this is what catches 'Mercer').
            new_scripts = _scripts(rewritten) - _scripts(message)
            if new_scripts:
                return reject(f"introduced foreign script {sorted(new_scripts)}")
            lm, lr = _detect_language(message), _detect_language(rewritten)
            if lm and lr and lm != lr:
                return reject(f"language drift {lm}->{lr}")
            # Must stay on-topic: share at least one content token with the question.
            q_terms, r_terms = set(_content_tokens(message)), set(_content_tokens(rewritten))
            if q_terms and not (q_terms & r_terms):
                return reject("no shared content token with the question")
            print(f"[RAG] query-rewrite ACCEPTED: {message!r} -> {rewritten!r}")
            return rewritten
        except Exception as e:
            print(f"[RAG] query-rewrite error ({e})")
            return message

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

        # Resolve follow-up references against recent turns BEFORE retrieval, so an
        # ambiguous question ("…بهذا المركب", "ce projet") fetches the right chunk.
        # The rewrite is used ONLY for retrieval; the answer still uses `message`.
        retrieval_query = await loop.run_in_executor(None, self._contextualize_query, message, history)

        items = []
        if scoped:
            # Explicit attachment(s): block until they are indexed, then ground on them.
            await self._await_attachments_indexed(scoped_ids)
            items = await loop.run_in_executor(None, self._relevant_nodes, retrieval_query, scope, user)
            print(f"[RAG] chat: scoped_docs={scoped_ids} items={len(items)} "
                  f"role={(user or {}).get('role')}")
        elif self.doc_count() > 0:
            # Always retrieve when the KB is non-empty; the relevance cutoff decides
            # whether the hits are good enough to ground on.
            items = await loop.run_in_executor(None, self._relevant_nodes, retrieval_query, scope, user)
            print(f"[RAG] chat: doc_count={self.doc_count()} items={len(items)} "
                  f"role={(user or {}).get('role')}")
        else:
            print("[RAG] chat: knowledge base is empty (doc_count=0) — answering conversationally")

        use_docs = len(items) > 0
        reranked_applied = any(it.get("rerank") is not None for it in items)

        # GROUNDING is decided by the retrieved CONTEXT chunks (`items`), which have
        # already passed the relevance gate (cosine >= cutoff OR strong BM25, or a
        # scoped match). Strict refusal and the source display BOTH follow grounding
        # — never the other way round. A chunk good enough to answer from is shown.
        if strict and self._strict_should_refuse(use_docs):
            print(f"[RAG] strict: refusing — no grounding chunk retrieved "
                  f"(items=0 scoped={scoped})")
            yield ("token", _strict_refusal(message))
            yield ("sources", [])
            return

        # Source cards: always ≥1 when grounded (top chunk), precision-filtered tail.
        display_sources = (self._format_sources(items, scoped=scoped, query=retrieval_query,
                                                reranked_applied=reranked_applied)
                           if use_docs else [])
        grounded = use_docs

        base_prompt = SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT
        if grounded:
            system_text = (
                base_prompt + directive + "\n\n"
                + CONTEXT_TEMPLATE.format(context_str=self._context_block(items))
            )
        elif scoped or strict:
            # Scoped to attachments with nothing retrievable: stay grounded and
            # instruct the model to say it couldn't find the answer (in the user's
            # language) rather than inventing one or chatting from general knowledge.
            system_text = (
                base_prompt + directive + "\n\n"
                + CONTEXT_TEMPLATE.format(context_str=_no_context_note(message))
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

        yield ("sources", display_sources)

    @staticmethod
    def _strict_should_refuse(use_docs: bool) -> bool:
        """Strict refusal is GROUNDING-based: refuse only when retrieval produced no
        reliable chunk to answer from (``use_docs`` is False). It must NOT depend on
        the source-card display filter — a chunk good enough to answer from is good
        enough to cite, so a display bug must never cause a false refusal."""
        return not use_docs

    @staticmethod
    def _source_eff(it: dict, use_rerank: bool) -> float:
        """Effective relevance for DISPLAY ordering. With the reranker ON, use its
        score (best, language-robust). With it OFF, use the FUSED (dense+BM25) score
        — NOT raw cosine — so a strong lexical match (e.g. Arabic keywords) ranks as
        the top source instead of an off-topic chunk with higher cosine. Falls back
        to cosine when no fused score is present (e.g. in unit tests)."""
        if use_rerank and it.get("rerank") is not None:
            return float(it.get("rerank"))
        if it.get("fused") is not None:
            return float(it.get("fused"))
        return float(it.get("score") or 0)

    @staticmethod
    def _is_lexical_only(it: dict) -> bool:
        """A BM25 hit whose dense cosine is below the relevance cutoff: matched on
        word overlap only. Used to keep such chunks out of SECONDARY source cards."""
        return it.get("bm25") is not None and (it.get("score") or 0) < RELEVANCE_CUTOFF

    @staticmethod
    def _select_source_items(items: list[dict], scoped: bool = False,
                             reranked_applied: bool = False) -> list[dict]:
        """Pick the chunk(s) to display as source cards. The answer is grounded on
        `items`, so the SINGLE best chunk is ALWAYS shown when items exist — a chunk
        good enough to answer from is good enough to cite, and no display threshold
        may hide it (this is what previously caused 'hors documents'/false refusals).
        Additional cards (only when SOURCE_DISPLAY_TOP_K > 1) are filtered for
        precision: close to the top and not lexical-only/off-topic. Scoped queries
        surface the chosen document(s)."""
        if not items:
            return []
        use_rerank = reranked_applied and any(it.get("rerank") is not None for it in items)
        ranked = sorted(items, key=lambda it: RAGEngine._source_eff(it, use_rerank), reverse=True)
        top = RAGEngine._source_eff(ranked[0], use_rerank)
        # Cards favour PRECISION: cap to SOURCE_DISPLAY_TOP_K for global search
        # (default 1). Scoped queries may show up to SOURCE_MAX of the chosen docs.
        limit = SOURCE_MAX if scoped else min(SOURCE_MAX, max(1, SOURCE_DISPLAY_TOP_K))

        chosen, seen = [], set()
        for i, it in enumerate(ranked):
            if i > 0 and not scoped:
                # SECONDARY cards only: must be near the best AND genuinely relevant.
                # (The top card is always kept — it is what the answer is grounded on.)
                e = RAGEngine._source_eff(it, use_rerank)
                if use_rerank:
                    if SOURCE_DISPLAY_MIN_RERANK > 0 and e < SOURCE_DISPLAY_MIN_RERANK:
                        break
                else:
                    if RAGEngine._is_lexical_only(it):
                        continue                   # skip lexical-only noise tail
                    if top > 0 and e < top - SOURCE_REL_GAP:
                        break                      # too far below the best match
            md = it.get("metadata") or {}
            key = md.get("file_name") or md.get("rag_doc_id")
            if not key or key in seen:
                continue
            seen.add(key)
            chosen.append(it)
            if len(chosen) >= limit:
                break
        return chosen

    @staticmethod
    def _snippet(text: str, query: Optional[str], width: int = 320) -> str:
        """A clean preview that ACTUALLY shows the part of the chunk relevant to the
        query — not blindly the first N chars. News/roundup chunks often pack several
        topics; this windows around the first matching query term so the FortiBleed
        question shows the FortiBleed sentence, not an unrelated lead paragraph."""
        clean = _clean_context_text(text)
        toks = _content_tokens(query) if query else []
        if toks:
            hay = _fold_arabic(clean.lower())   # length-aligned with `clean`
            pos = min((hay.find(t) for t in toks if hay.find(t) != -1), default=-1)
            if pos > 0:
                start = max(0, pos - 60)
                end = min(len(clean), start + width)
                seg = clean[start:end].strip()
                return ("…" if start > 0 else "") + seg + ("…" if end < len(clean) else "")
        return clean[:width] + ("…" if len(clean) > width else "")

    @staticmethod
    def _format_sources(items: list[dict], scoped: bool = False, query: Optional[str] = None,
                        reranked_applied: bool = False) -> list[dict]:
        """Build the user-facing source cards from the selection, with a query-relevant
        snippet per card. One entry per file. When `items` is non-empty (the answer is
        grounded) this ALWAYS returns at least one card. See _select_source_items."""
        chosen = RAGEngine._select_source_items(items, scoped, reranked_applied)
        sources = []
        for it in chosen:
            md = it.get("metadata") or {}
            sources.append({
                "text": RAGEngine._snippet(it.get("text") or "", query),
                "score": round(float(it.get("score") or 0), 4),
                "filename": md.get("file_name", "Unknown"),
                "page": md.get("page_label", "?"),
                "document_id": md.get("rag_doc_id"),
                "version": md.get("version"),
            })
        print(f"[RAG] sources: scoped={scoped} "
              f"shown={len(sources)}/{len(items)} -> {[(s['filename'], s['score']) for s in sources]}")
        return sources

    # ─── Ingestion & version management ────────────────────────────────────────

    def _load_documents(self, file_path: str, filename: str) -> list:
        """Load a file into LlamaIndex Documents. PDFs use an Arabic/RTL-aware
        extractor (PyMuPDF, then pypdfium2) that preserves reading order and
        embedded digits — the default pypdf path drops Arabic numbers and emits
        presentation-form glyphs. All extracted text is NFKC-normalized."""
        suffix = os.path.splitext(file_path)[1].lower()
        docs = None
        if suffix == ".pdf":
            docs = self._pdf_via_pymupdf(file_path) or self._pdf_via_pdfium(file_path)
        if not docs:
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for d in docs:
            try:
                d.text = _normalize_text(d.text)
            except Exception:
                pass
        return docs

    @staticmethod
    def _pdf_via_pymupdf(file_path: str) -> Optional[list]:
        """Best Arabic/RTL extraction. PyMuPDF (`fitz`) — AGPL; install `pymupdf`."""
        try:
            import fitz  # PyMuPDF
            from llama_index.core import Document
        except Exception:
            return None
        try:
            out = []
            with fitz.open(file_path) as pdf:
                for i, page in enumerate(pdf):
                    text = page.get_text("text")
                    if text and text.strip():
                        out.append(Document(text=text, metadata={"page_label": str(i + 1)}))
            return out or None
        except Exception as e:
            print(f"[RAG] PyMuPDF extraction failed ({e}); trying next extractor")
            return None

    @staticmethod
    def _pdf_via_pdfium(file_path: str) -> Optional[list]:
        """Permissively-licensed fallback (pypdfium2, BSD/Apache) — good Arabic too."""
        try:
            import pypdfium2 as pdfium
            from llama_index.core import Document
        except Exception:
            return None
        try:
            out = []
            pdf = pdfium.PdfDocument(file_path)
            try:
                for i in range(len(pdf)):
                    text = pdf[i].get_textpage().get_text_range()
                    if text and text.strip():
                        out.append(Document(text=text, metadata={"page_label": str(i + 1)}))
            finally:
                pdf.close()
            return out or None
        except Exception as e:
            print(f"[RAG] pypdfium2 extraction failed ({e}); falling back to default")
            return None

    def index_version(self, file_path: str, filename: str, document_id: str, version: int,
                      owner_id: Optional[str] = None, scope: str = "org") -> int:
        """Chunk/embed a file as a new version, mark it is_latest, demote older ones."""
        with self._write_lock:
            documents = self._load_documents(file_path, filename)
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