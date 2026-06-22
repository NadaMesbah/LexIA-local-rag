"""Unit tests for the retrieval-only evaluator's pure helpers.

We load eval/retrieval_eval.py by path (it lives outside backend/) and exercise
only its label-matching helpers — these need no Ollama, Chroma or LLM.
"""

import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL = os.path.join(_REPO, "eval", "retrieval_eval.py")

_spec = importlib.util.spec_from_file_location("retrieval_eval", _EVAL)
re_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(re_eval)


def test_expected_parses_labels():
    exp = re_eval._expected({"expected_doc": "Reuters", "expected_page": 3})
    assert exp == {"doc": "reuters", "doc_id": "", "page": "3"}


def test_is_match_by_filename_substring_case_insensitive():
    exp = re_eval._expected({"expected_doc": "reuters"})
    assert re_eval._is_match({"file_name": "Reuters_Report_2026.pdf"}, exp) is True
    assert re_eval._is_match({"file_name": "bank_al_maghrib.pdf"}, exp) is False


def test_is_match_by_doc_id():
    exp = re_eval._expected({"expected_doc_id": "abc123"})
    assert re_eval._is_match({"rag_doc_id": "abc123"}, exp) is True
    assert re_eval._is_match({"rag_doc_id": "other"}, exp) is False


def test_page_match():
    exp = re_eval._expected({"expected_doc": "x", "expected_page": "3"})
    assert re_eval._page_match({"page_label": "3"}, exp) is True
    assert re_eval._page_match({"page_label": "4"}, exp) is False
    # No page label expected -> never a page match.
    assert re_eval._page_match({"page_label": "3"}, re_eval._expected({"expected_doc": "x"})) is False