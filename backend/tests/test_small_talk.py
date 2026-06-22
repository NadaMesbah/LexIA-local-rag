"""Unit tests for the small-talk / greeting fast path.

These exercise pure module helpers and the early-return fast path of
``RAGEngine.stream_chat`` WITHOUT a running Ollama/Chroma: the fast path returns
before any query rewrite, retrieval, embedding, BM25, rerank or strict-refusal
logic, so it never touches ``self``. We drive the async generator with a guard
"self" that explodes if the small-talk path ever calls a retrieval method.
"""

import asyncio

import rag_engine as rg
from rag_engine import RAGEngine


def _drive(agen):
    """Collect an async generator into a list (sync test, no pytest-asyncio)."""
    async def run():
        return [item async for item in agen]
    return asyncio.run(run())


class _GuardSelf:
    """Stand-in for RAGEngine: any retrieval-path call means the fast path leaked."""
    def _scoped_ids(self, scope):
        raise AssertionError("scope resolution called on the small-talk fast path")

    def _relevant_nodes(self, *a, **k):
        raise AssertionError("retrieval called on the small-talk fast path")

    def _contextualize_query(self, *a, **k):
        raise AssertionError("query rewrite called on the small-talk fast path")

    async def _await_attachments_indexed(self, *a, **k):
        raise AssertionError("attachment indexing awaited on the small-talk fast path")

    def doc_count(self):
        raise AssertionError("doc_count called on the small-talk fast path")


# ─── Detector ─────────────────────────────────────────────────────────────────

def test_is_small_talk_matches_greetings_and_thanks():
    for m in ["Bonjour", "Salut", "Salut cv ?", "Ça va ?", "Merci", "Thanks",
              "Hello", "Hi", "السلام عليكم", "مرحبا", "شكرا", "صباح الخير"]:
        assert rg._is_small_talk(m) is True, m


def test_is_small_talk_rejects_real_questions():
    # Real informational questions must fall through to normal RAG (NOT the fast path).
    for m in ["C'est quoi le machine learning ?",
              "Explique-moi le machine learning.",
              "ماذا ضبطت الشرطة القضائية في طنجة في قضية التهريب الدولي؟",
              "ما درجة الحرارة التي حذرت منها الأرصاد الجوية؟",
              "Quel taux directeur BMCE Capital anticipe-t-il pour Bank Al-Maghrib ?",
              "Résume le document"]:
        assert rg._is_small_talk(m) is False, m


# ─── Deterministic localized reply ────────────────────────────────────────────

def test_smalltalk_reply_is_localized_and_strict_aware():
    # French, ON vs OFF (the documented expected strings).
    assert rg._smalltalk_reply("Salut", strict=True) == \
        "Salut ! Comment puis-je vous aider avec vos documents ?"
    assert rg._smalltalk_reply("Salut", strict=False) == \
        "Salut ! Comment puis-je vous aider ?"
    # English + Arabic stay in-language.
    assert rg._smalltalk_reply("Hello", strict=False).startswith("Hello!")
    assert "وثائقك" in rg._smalltalk_reply("مرحبا", strict=True)
    # Thanks gets a distinct reply from a plain greeting.
    assert rg._smalltalk_reply("Merci").startswith("Je vous en prie")
    assert rg._smalltalk_reply("شكرا").startswith("على الرحب")


# ─── Fast path in stream_chat (no retrieval, no sources, no refusal) ──────────

def test_fastpath_greeting_yields_reply_then_empty_sources_no_retrieval():
    for msg in ["Salut", "Bonjour", "Hello", "Merci", "السلام عليكم"]:
        events = _drive(RAGEngine.stream_chat(_GuardSelf(), msg, [], strict=True))
        kinds = [k for k, _ in events]
        assert "token" in kinds and kinds[-1] == "sources"
        sources = [p for k, p in events if k == "sources"][0]
        assert sources == []                                    # no document sources
        token = "".join(p for k, p in events if k == "token")
        assert token.strip()                                    # a natural greeting
        assert not rg._is_refusal(token)                        # NOT a strict refusal


def test_fastpath_works_in_non_strict_mode_too():
    events = _drive(RAGEngine.stream_chat(_GuardSelf(), "Hello", [], strict=False))
    sources = [p for k, p in events if k == "sources"][0]
    assert sources == []
    token = "".join(p for k, p in events if k == "token")
    assert token.startswith("Hello!")