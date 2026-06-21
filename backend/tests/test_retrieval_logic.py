"""Unit tests for the pure retrieval/ranking logic.

These import `rag_engine`, which pulls in chromadb + llama-index, so run them in
the backend environment (where requirements are installed). They do NOT need a
running Ollama — only module-level helpers and static methods are exercised.
"""

import rag_engine as rg
from rag_engine import RAGEngine


# ─── Arabic normalization (the fix for dropped/garbled Arabic) ────────────────

def test_nfkc_folds_arabic_presentation_forms():
    # U+FE8D = ARABIC LETTER ALEF, ISOLATED FORM -> U+0627 = ARABIC LETTER ALEF
    assert rg._normalize_text("\uFE8D") == "\u0627"


def test_tokenize_is_nfkc_invariant():
    # Text stored as presentation forms tokenizes the same as a typed query, so
    # BM25 can match Arabic regardless of how the PDF extractor emitted it.
    assert rg._tokenize("\uFE8D") == rg._tokenize("\u0627")
    assert rg._tokenize("Bonjour LE Monde") == ["bonjour", "le", "monde"]


# ─── BM25 lexical search ──────────────────────────────────────────────────────

def _bm25(texts):
    ids = [f"c{i}" for i in range(len(texts))]
    metas = [{"file_name": f"{i}.pdf"} for i in range(len(texts))]
    return rg._BM25(ids, texts, metas)


def test_bm25_ranks_exact_term_match_first():
    bm = _bm25([
        "JADE platform agent messaging ACL container",
        "Switch Pay commission 2,9% transactions Banque Atlas PME",
        "machine learning model training cross validation",
    ])
    hits = bm.search("Quelle est la commission appliquée par Switch Pay ?", 3)
    assert hits and hits[0][0] == "c1"


def test_bm25_greeting_has_no_lexical_hit():
    bm = _bm25(["Switch Pay commission Banque Atlas", "machine learning model"])
    assert bm.search("Bonjour, comment vas-tu ?", 3) == []


# ─── Cosine ───────────────────────────────────────────────────────────────────

def test_cosine_basic():
    assert rg._cosine([1, 0], [1, 0]) == 1.0
    assert rg._cosine([1, 0], [0, 1]) == 0.0
    assert rg._cosine([], [1]) == 0.0


# ─── Scope helpers / visibility ───────────────────────────────────────────────

def test_scoped_ids():
    assert RAGEngine._scoped_ids(None) == []
    assert RAGEngine._scoped_ids({"type": "all"}) == []
    assert RAGEngine._scoped_ids({"type": "document", "id": "d1"}) == ["d1"]
    assert RAGEngine._scoped_ids({"type": "document", "ids": ["d1", "d2"]}) == ["d1", "d2"]


def test_visibility_where():
    assert RAGEngine._visibility_where({"role": "admin", "id": "a"}, None) == {"is_latest": True}
    w = RAGEngine._visibility_where({"role": "user", "id": "u1"}, None)
    assert w == {"$and": [{"is_latest": True}, {"$or": [{"scope": "org"}, {"owner_id": "u1"}]}]}
    w = RAGEngine._visibility_where({"role": "admin", "id": "a"}, {"type": "document", "ids": ["d1", "d2"]})
    assert w == {"$and": [{"is_latest": True}, {"rag_doc_id": {"$in": ["d1", "d2"]}}]}


def test_visible_predicate():
    admin = {"role": "admin", "id": "a"}
    user = {"role": "user", "id": "u1"}
    org = {"is_latest": True, "scope": "org", "owner_id": "", "rag_doc_id": "d1"}
    mine = {"is_latest": True, "scope": "personal", "owner_id": "u1", "rag_doc_id": "d2"}
    other = {"is_latest": True, "scope": "personal", "owner_id": "u2", "rag_doc_id": "d3"}
    old = {"is_latest": False, "scope": "org", "owner_id": "", "rag_doc_id": "d1"}
    assert RAGEngine._visible(org, user, []) is True
    assert RAGEngine._visible(mine, user, []) is True
    assert RAGEngine._visible(other, user, []) is False     # someone else's personal doc
    assert RAGEngine._visible(other, admin, []) is True      # admin sees all
    assert RAGEngine._visible(old, admin, []) is False       # not latest
    assert RAGEngine._visible(org, user, ["d2"]) is False    # scoped to a different doc
    assert RAGEngine._visible(mine, user, ["d2"]) is True


# ─── Source display filtering ─────────────────────────────────────────────────

def _item(name, score, bm25=None):
    return {"text": f"text {name}", "metadata": {"file_name": name, "rag_doc_id": name}, "score": score, "bm25": bm25}


def names(srcs):
    return [s["filename"] for s in srcs]


def test_sources_keep_only_top_relevant():
    src = RAGEngine._format_sources([
        _item("banque_atlas", 0.55), _item("ML", 0.41), _item("JADE", 0.40), _item("IA", 0.42),
    ])
    assert names(src) == ["banque_atlas"]


def test_sources_dedupe_same_filename():
    src = RAGEngine._format_sources([
        _item("ar_bank.pdf", 0.62), _item("ar_bank.pdf", 0.62), _item("noise.pdf", 0.38),
    ])
    assert names(src) == ["ar_bank.pdf"]


def test_sources_empty_on_small_talk():
    assert RAGEngine._format_sources([_item("x.pdf", 0.37), _item("y.pdf", 0.33)]) == []


def test_sources_scoped_bypasses_floor():
    # Attached-doc answer: surface the document even if its cosine is modest.
    src = RAGEngine._format_sources([_item("attached.pdf", 0.33), _item("attached.pdf", 0.30)], scoped=True)
    assert names(src) == ["attached.pdf"]


def test_sources_capped():
    items = [_item(f"{i}.pdf", 0.60) for i in range(8)]
    assert len(RAGEngine._format_sources(items)) == rg.SOURCE_MAX