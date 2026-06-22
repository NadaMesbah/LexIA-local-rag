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


def test_normalize_strips_arabic_diacritics_and_tatweel():
    assert rg._normalize_text("العَمُولَة") == "العمولة"     # tashkeel removed
    assert rg._normalize_text("العمـــولة") == "العمولة"     # tatweel removed


def test_fold_arabic_unifies_spelling_variants():
    assert rg._fold_arabic("إلى") == "الي"      # alef-hamza-below + alef-maqsura
    assert rg._fold_arabic("نسبة") == "نسبه"    # ta marbuta -> ha
    assert rg._fold_arabic("hello") == "hello"  # no-op for Latin


def test_bm25_matches_arabic_spelling_variants():
    bm = _bm25([
        "التقرير يذكر أن إجمالي العمولة مرتفع",
        "وثيقة عن تعلم الآلة والشبكات العصبية",
    ])
    # Query typed without the hamza / with ta-marbuta differences still matches.
    hits = bm.search("ما هو اجمالي العموله ؟", 2)
    assert hits and hits[0][0] == "c0"


def test_clean_context_strips_cjk_and_control_keeps_arabic():
    dirty = "العمولة 2,9% \u4f60\u597d\ufffd\u200b للمعاملات"
    clean = rg._clean_context_text(dirty)
    import re as _re
    assert not _re.search(r"[\u4e00-\u9fff]", clean)   # CJK gone
    assert "\ufffd" not in clean and "\u200b" not in clean
    assert "العمولة" in clean and "للمعاملات" in clean  # Arabic preserved


def test_arabic_language_directive_is_arabic_only():
    d = rg._language_directive("ما هي نسبة العمولة؟")
    assert "Arabic" in d and "non-Arabic" in d


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


# ─── Stopwords / content tokens (BM25 noise reduction) ────────────────────────

def test_content_tokens_drop_stopwords_keep_content():
    ct = rg._content_tokens("Quelle est la commission appliquée par Switch Pay ?")
    assert "la" not in ct and "est" not in ct and "par" not in ct      # FR stopwords gone
    assert {"commission", "switch", "pay"}.issubset(set(ct))            # content kept


def test_content_tokens_drop_french_elision_keep_digits():
    assert rg._content_tokens("l'application d'agent") == ["application", "agent"]
    assert "2024" in rg._content_tokens("rapport 2024")


def test_content_tokens_drop_arabic_stopwords():
    ar = rg._content_tokens("ما هي نسبة العمولة في البنك؟")
    assert "ما" not in ar and "هي" not in ar and "في" not in ar


def test_tokenize_still_keeps_stopwords():
    # _tokenize is unchanged (used for NFKC invariants); only BM25 strips stopwords.
    assert rg._tokenize("Bonjour LE Monde") == ["bonjour", "le", "monde"]


def test_min_bm25_score_filters_weak_lexical_noise(monkeypatch):
    monkeypatch.setattr(rg, "MIN_BM25_SCORE", 1.5)
    monkeypatch.setattr(rg, "RELEVANCE_CUTOFF", 0.25)
    pool = [
        {"id": "vec", "score": 0.55, "bm25": None, "text": "a"},        # strong cosine
        {"id": "bm_ok", "score": 0.10, "bm25": 3.0, "text": "b"},       # weak cos, strong BM25
        {"id": "bm_weak", "score": 0.10, "bm25": 0.4, "text": "c"},     # weak both -> noise
    ]
    out = [x["id"] for x in RAGEngine._final_select(pool, scoped=False, reranked_applied=False)]
    assert "vec" in out and "bm_ok" in out
    assert "bm_weak" not in out


# ─── NumPy-safety (the fix for "truth value of an array is ambiguous") ────────
# Chroma returns embeddings/distances as numpy arrays; the retrieval code must
# never put them in a boolean context. These tests use REAL numpy arrays.

def test_to_list_handles_numpy_and_none():
    import numpy as np
    assert rg._to_list(None) == []
    assert rg._to_list([1, 2]) == [1, 2]
    # 2D ndarray (a Chroma get(embeddings) result) -> nested python list.
    assert rg._to_list(np.array([[0.1, 0.2], [0.3, 0.4]])) == [[0.1, 0.2], [0.3, 0.4]]
    # An empty ndarray must not raise and must be falsy-safe.
    assert rg._to_list(np.array([])) == []


def test_first_row_handles_numpy_query_result():
    import numpy as np
    # Chroma query results are shaped [[...]] and distances come back as ndarrays.
    assert rg._first_row(np.array([[0.1, 0.25, 0.4]])) == [0.1, 0.25, 0.4]
    assert rg._first_row([["c1", "c2"]]) == ["c1", "c2"]
    assert rg._first_row(None) == []
    assert rg._first_row(np.array([])) == []


def test_cosine_accepts_numpy_arrays():
    import numpy as np
    # The BM25-only cosine backfill passes coerced numpy embeddings here.
    assert abs(rg._cosine([1.0, 0.0], np.array([1.0, 0.0])) - 1.0) < 1e-9
    assert abs(rg._cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9


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


def test_top_source_shown_when_grounded_even_below_old_floor():
    # Small talk no longer filtered HERE — that's the retrieval layer (empty items).
    # When chunks ARE retrieved (grounded), the single best is always shown, even
    # with a modest Arabic-style cosine, so a grounded answer always cites a source.
    assert RAGEngine._format_sources([]) == []
    items = [_item("AL-ALAM2026-AR.pdf", 0.3594), _item("AL-ALAM2026-AR.pdf", 0.31)]
    assert names(RAGEngine._format_sources(items)) == ["AL-ALAM2026-AR.pdf"]


def test_sources_scoped_bypasses_floor():
    # Attached-doc answer: surface the document even if its cosine is modest.
    src = RAGEngine._format_sources([_item("attached.pdf", 0.33), _item("attached.pdf", 0.30)], scoped=True)
    assert names(src) == ["attached.pdf"]


def test_source_display_top_k_default_one():
    # Precision-first: by default only the single best source card is shown.
    items = [_item("a.pdf", 0.60), _item("b.pdf", 0.58), _item("c.pdf", 0.56)]
    assert len(RAGEngine._format_sources(items)) == 1


def test_sources_cap_respects_top_k_and_source_max(monkeypatch):
    monkeypatch.setattr(rg, "SOURCE_DISPLAY_TOP_K", 10)
    items = [_item(f"{i}.pdf", 0.60) for i in range(8)]
    assert len(RAGEngine._format_sources(items)) == rg.SOURCE_MAX


# ─── Source display = strictest layer (grounding/snippet/precision) ───────────

def _lex(name, cosine, bm25):
    # A lexical-only candidate: BM25 hit whose dense cosine is below the cutoff.
    return {"id": name, "text": f"text {name}", "score": cosine, "bm25": bm25,
            "metadata": {"file_name": name, "rag_doc_id": name}}


def test_lexical_only_chunk_is_not_displayed_as_source():
    # Weather question: ML chunk matched only on lexical overlap (temperature/hot).
    pool = [_item("Weather.pdf", 0.52), _lex("Machine_Learning_Ch4.pdf", 0.20, 3.0)]
    assert rg.RAGEngine._is_lexical_only(pool[1]) is True
    assert names(RAGEngine._format_sources(pool)) == ["Weather.pdf"]


def test_lexical_only_excluded_from_secondary_cards(monkeypatch):
    # With more than one card allowed, a lexical-only ML chunk is NOT a secondary
    # source for a weather question — but the genuine top source is still shown.
    monkeypatch.setattr(rg, "SOURCE_DISPLAY_TOP_K", 3)
    pool = [_item("Weather.pdf", 0.52), _item("News.pdf", 0.50),
            _lex("Machine_Learning_Ch4.pdf", 0.20, 3.0)]
    out = names(RAGEngine._format_sources(pool))
    assert "Weather.pdf" in out and "Machine_Learning_Ch4.pdf" not in out


def test_display_orders_by_fused_score_not_raw_cosine():
    # Arabic-style: AL-ALAM has a STRONG BM25/fused signal but a lower cosine than an
    # off-topic ML chunk. Display must surface AL-ALAM (fused), not ML (cosine).
    items = [
        {"id": "ml", "score": 0.3919, "fused": 0.0164, "text": "ml",
         "metadata": {"file_name": "Machine_Learning_Ch1.pdf"}},
        {"id": "al", "score": 0.3300, "bm25": 6.0, "fused": 0.0320, "text": "al",
         "metadata": {"file_name": "AL-ALAM2026-AR.pdf"}},
    ]
    assert names(RAGEngine._format_sources(items)) == ["AL-ALAM2026-AR.pdf"]


def test_scripts_detects_foreign_script_injection():
    # Used to reject a query rewrite that injects Latin ('Mercer') into Arabic.
    assert "latin" not in rg._scripts("من وصف حالة الاحتقان بهذا المركب؟")
    assert "latin" in rg._scripts("من وصف بالم Mercer الكيمادو؟")
    assert "arabic" in rg._scripts("المركب السياحي كيمادو")


def test_display_is_rerank_aware(monkeypatch):
    monkeypatch.setattr(rg, "SOURCE_DISPLAY_MIN_RERANK", 0.3)
    monkeypatch.setattr(rg, "SOURCE_DISPLAY_TOP_K", 5)   # see more than the top-1 card
    rr = [
        {"id": "c", "score": 0.20, "rerank": 0.95, "text": "t", "metadata": {"file_name": "C.pdf"}},
        {"id": "a", "score": 0.55, "rerank": 0.80, "text": "t", "metadata": {"file_name": "A.pdf"}},
        {"id": "b", "score": 0.30, "bm25": 2.0, "rerank": 0.05, "text": "t", "metadata": {"file_name": "B.pdf"}},
    ]
    disp = names(RAGEngine._format_sources(rr, reranked_applied=True))
    assert disp[:2] == ["C.pdf", "A.pdf"]   # ordered by rerank, not cosine
    assert "B.pdf" not in disp              # below SOURCE_DISPLAY_MIN_RERANK


def test_snippet_windows_to_query_relevant_text():
    chunk = ("Eau dessalée de Safi vers Marrakech sur 185 km. " * 4 +
             "Par ailleurs, la fuite FortiBleed a exposé des identifiants VPN; la DGSSI alerte.")
    snip = RAGEngine._snippet(chunk, "Que révèle la fuite FortiBleed ?")
    assert "FortiBleed" in snip
    assert snip.startswith("…")             # a window, not the head of the chunk
    # No query → head of (cleaned) text.
    assert RAGEngine._snippet("ABCDEFGH" * 60, None).startswith("ABCDEFGH")


def test_contextualize_query_noop_without_history():
    # The guard returns the original message (no LLM call) on the first turn.
    assert RAGEngine._contextualize_query(object(), "ما هذا؟", []) == "ما هذا؟"


# ─── Strict mode: deterministic, language-aware refusal ───────────────────────

def test_strict_refusal_messages_are_language_aware():
    assert rg._strict_refusal("What is the commission rate?") == \
        "I could not find this information in the provided documents."
    assert rg._strict_refusal("Quelle est la commission appliquée ?") == \
        "Je n’ai pas trouvé cette information dans les documents fournis."
    assert rg._strict_refusal("ما هي نسبة العمولة المطبقة؟") == \
        "لم أجد هذه المعلومة في الوثائق المقدمة."
    # Unknown language falls back to English.
    assert rg._strict_refusal("?!?") == \
        "I could not find this information in the provided documents."


def test_no_context_note_follows_language():
    assert "العثور" in rg._no_context_note("ما هذا؟")                 # Arabic
    assert "No relevant extract" in rg._no_context_note("What is this?")
    assert "Aucun extrait" in rg._no_context_note("Quel est ce document ?")


def test_strict_grounding_gate():
    R = RAGEngine._strict_should_refuse
    # Grounding-based (NOT display-based): refuse only when nothing was retrieved.
    assert R(False) is True       # no grounding chunk -> refuse
    assert R(True) is False        # at least one reliable chunk retrieved -> answer


# ─── Pipeline order: rerank BEFORE the aggressive final filter ────────────────

def _cand(id_, score=None, bm25=None, rerank=None):
    return {"id": id_, "score": score, "bm25": bm25, "rerank": rerank,
            "text": id_, "metadata": {"file_name": id_}}


def test_final_select_rerank_off_applies_cosine_cutoff():
    pool = [_cand("a", score=0.55), _cand("b", score=0.30, bm25=2.0),
            _cand("c", score=0.20), _cand("d", score=0.42), _cand("e", score=0.10, bm25=1.0)]
    out = [x["id"] for x in RAGEngine._final_select(pool, scoped=False, reranked_applied=False)]
    assert "a" in out and "d" in out      # strong cosine kept
    assert "b" in out and "e" in out      # exact BM25 hits kept even below cutoff
    assert "c" not in out                 # weak vector-only noise dropped


def test_final_select_rerank_on_trusts_reranker_order_no_premature_cutoff():
    # The reranker ranked a low-cosine chunk #1 — it must survive (the old code
    # filtered by cosine BEFORE rerank and would have dropped it).
    reranked = [_cand("c", score=0.20, rerank=0.99), _cand("a", score=0.55, rerank=0.80),
                _cand("b", score=0.30, bm25=2.0, rerank=0.10)]
    out = RAGEngine._final_select(reranked, scoped=False, reranked_applied=True)
    assert [x["id"] for x in out] == ["c", "a", "b"]
    assert out[0]["id"] == "c"


def test_final_select_scoped_keeps_all():
    pool = [_cand("a", score=0.55), _cand("c", score=0.05)]
    assert len(RAGEngine._final_select(pool, scoped=True, reranked_applied=False)) == 2