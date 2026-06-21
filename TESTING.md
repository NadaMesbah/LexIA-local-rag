# Testing the RAG

Three layers: **unit** (fast, no Ollama), **integration** (FastAPI end-to-end, no
Ollama), and **RAGAS** (quality metrics, needs Ollama + an indexed corpus).

## Setup
```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
```

## 1. Unit + integration tests (fast, offline)
```bash
cd backend
pytest                      # runs everything in backend/tests/
pytest tests/test_db.py     # just the DB unit tests
pytest -k "sources or bm25" # filter by name
```
What they cover:
- `tests/test_db.py` — email signup/login, document/version model, `find_document_id`
  org-vs-personal disambiguation, the reconcile-input fix, per-user isolation,
  message attachments, the `downloadable` flag, the email migration. (Pure SQLite.)
- `tests/test_retrieval_logic.py` — Arabic NFKC normalization, BM25 ranking
  (exact-term first; greeting → no hit), cosine, scope/visibility helpers, and the
  source-display filtering (gap/floor/cap/dedup, scoped bypass).
- `tests/test_api_integration.py` — register(email)/login(email), `/auth/me`,
  document upload + listing + **isolation between users**, admin-only `/debug/retrieve`,
  and a chat round-trip that persists the answer. The heavy engine is stubbed, so
  **no Ollama/Chroma is needed**.

Tests use throwaway temp storage (set in `tests/conftest.py`) — they never touch
your real data.

## 2. RAGAS quality evaluation (needs Ollama + indexed corpus)
RAGAS measures *how good* the RAG is, using your **local Ollama as the judge** and
**bge-m3** for embeddings — fully offline, nothing leaves your machine.

```bash
# 1) Ollama running with your models (same as the app).
# 2) Point at the SAME data the app uses, so the corpus is already indexed:
export CHROMA_PATH=/abs/path/to/data/chroma
export SQLITE_PATH=/abs/path/to/data/rag.db
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_MODEL=qwen2.5:7b EMBED_MODEL=bge-m3

python eval/ragas_eval.py --dataset eval/dataset.example.json
```
It prints scores for **faithfulness, answer relevancy, context precision, context
recall** and writes per-question detail to `eval/ragas_results.csv`.

### How to use the metrics
- **Low faithfulness** → the model is adding things not in the context (hallucination).
- **Low context recall** → retrieval isn't bringing back the needed passage (chunking /
  cutoff / hybrid). This is the one to watch for the Arabic number cases.
- **Low context precision** → too much noise retrieved (lower `RETRIEVE_TOP_K`, raise cutoff).
- **Low answer relevancy** → the answer drifts from the question (prompt/model).

### Comparing configurations (the real value)
Run it, change one thing in `.env`, re-run, compare. Good A/B tests:
- `LLM_MODEL=qwen2.5:7b` vs `aya-expanse:8b` vs `qwen2.5:14b` (Arabic quality).
- `HYBRID_ENABLED=true` vs `false`.
- `RELEVANCE_CUTOFF=0.35` vs `0.20`, `RETRIEVE_TOP_K=6` vs `10`.
- `RERANK_ENABLED=true` vs `false`.

Build your own `eval/dataset.json` (same shape as the example) from your real demo
corpus — 15–30 question/ground-truth pairs is plenty for meaningful numbers.

> Notes: the local judge gives *indicative* scores (great for comparison, not an
> absolute grade). RAGAS APIs change between versions — if an import in
> `eval/ragas_eval.py` fails, align it to your installed `ragas`.