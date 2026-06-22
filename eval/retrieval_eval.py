"""
Retrieval-only evaluation — FAST, no LLM judge.

Measures how well retrieval finds the right document/page for each question, using
ONLY the real engine's retrieval (rag._relevant_nodes). This is the first thing to
run when tuning retrieval: it isolates retrieval quality from answer/judge quality,
finishes in seconds, and never times out on a slow local model.

Metrics (over questions that carry an expected-document label):
  - top-1 / top-3 / top-5 accuracy : expected document appears within the top-N
  - MRR                            : mean reciprocal rank of the first correct hit
  - page accuracy                  : expected page present among the hits (when labelled)

Dataset
-------
Same JSON as the RAGAS dataset; each row MAY add optional labels:
  {
    "question": "...",
    "ground_truth": "...",                  # (used by RAGAS, ignored here)
    "expected_doc": "reuters",              # filename substring (case-insensitive)
    "expected_doc_id": "abc123",            # OR exact rag_doc_id
    "expected_page": "3"                    # optional page label
  }
Rows without any expected_* label are skipped (reported as "unlabeled").

Run
---
  export CHROMA_PATH=/path/to/data/chroma SQLITE_PATH=/path/to/data/rag.db
  python eval/retrieval_eval.py --dataset eval/dataset.example.json --top-k 5
"""

import argparse
import json
import os
import sys

# Consider at least the top-5 so top-5 accuracy is meaningful regardless of the
# app's configured RETRIEVE_TOP_K. Set BEFORE importing rag_engine (env read once).
_DEFAULT_TOPK = 5


def _add_backend_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(os.path.dirname(here), "backend"), os.path.dirname(here), os.getcwd()):
        if os.path.exists(os.path.join(cand, "rag_engine.py")):
            sys.path.insert(0, cand)
            return
    raise SystemExit("Could not locate backend/rag_engine.py")


def _expected(row: dict) -> dict:
    return {
        "doc": (row.get("expected_doc") or "").strip().lower(),
        "doc_id": (row.get("expected_doc_id") or "").strip(),
        "page": str(row.get("expected_page") or "").strip(),
    }


def _is_match(meta: dict, exp: dict) -> bool:
    fn = (meta.get("file_name") or "").lower()
    if exp["doc_id"] and meta.get("rag_doc_id") == exp["doc_id"]:
        return True
    if exp["doc"] and exp["doc"] in fn:
        return True
    return False


def _page_match(meta: dict, exp: dict) -> bool:
    return bool(exp["page"]) and str(meta.get("page_label") or "") == exp["page"]


def evaluate(dataset, top_k: int):
    from rag_engine import RAGEngine
    rag = RAGEngine()
    user = {"id": "eval", "role": "admin"}

    rows, misses, unlabeled = [], [], []
    hits = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    page_hits = page_total = 0

    for row in dataset:
        q = row["question"]
        exp = _expected(row)
        if not (exp["doc"] or exp["doc_id"]):
            unlabeled.append(q)
            continue

        items = rag._relevant_nodes(q, None, user)[:top_k]
        rank = next((i + 1 for i, it in enumerate(items)
                     if _is_match(it.get("metadata") or {}, exp)), 0)
        for n in (1, 3, 5):
            if rank and rank <= n:
                hits[n] += 1
        rr_sum += (1.0 / rank) if rank else 0.0

        page_ok = None
        if exp["page"]:
            page_total += 1
            page_ok = any(_page_match(it.get("metadata") or {}, exp) for it in items)
            page_hits += 1 if page_ok else 0

        rec = {
            "question": q,
            "expected": {k: v for k, v in exp.items() if v},
            "rank": rank,
            "hit@1": bool(rank and rank <= 1),
            "hit@3": bool(rank and rank <= 3),
            "hit@5": bool(rank and rank <= 5),
            "page_ok": page_ok,
            "retrieved": [{"filename": (it.get("metadata") or {}).get("file_name"),
                           "page": (it.get("metadata") or {}).get("page_label"),
                           "score": round(float(it.get("score") or 0), 4),
                           "bm25": it.get("bm25")} for it in items],
        }
        rows.append(rec)
        if not rank:
            misses.append(rec)
        print(f"{'HIT@'+str(rank) if rank else 'MISS  '}  {q[:70]}")

    n = len(rows)
    summary = {
        "labeled_questions": n,
        "unlabeled_skipped": len(unlabeled),
        "top_1_accuracy": round(hits[1] / n, 4) if n else None,
        "top_3_accuracy": round(hits[3] / n, 4) if n else None,
        "top_5_accuracy": round(hits[5] / n, 4) if n else None,
        "mrr": round(rr_sum / n, 4) if n else None,
        "page_accuracy": round(page_hits / page_total, 4) if page_total else None,
        "misses": len(misses),
        "top_k": top_k,
    }
    return summary, rows, misses, unlabeled


def main():
    ap = argparse.ArgumentParser(description="Fast retrieval-only evaluation (no LLM).")
    ap.add_argument("--dataset", default="eval/dataset.example.json")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (0=all)")
    ap.add_argument("--top-k", type=int, default=_DEFAULT_TOPK,
                    help="how many retrieved chunks to consider (>=5 for top-5 accuracy)")
    ap.add_argument("--out", default="eval/retrieval_results.json")
    args = ap.parse_args()

    top_k = max(args.top_k, 5)
    # Ensure the engine returns enough chunks to score top-5 regardless of app config.
    os.environ["RETRIEVE_TOP_K"] = str(max(int(os.getenv("RETRIEVE_TOP_K", "6")), top_k))
    _add_backend_to_path()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]

    summary, rows, misses, unlabeled = evaluate(dataset, top_k)

    print("\n================ Retrieval-only results ================")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if unlabeled:
        print(f"\n  NOTE: {len(unlabeled)} question(s) had no expected_doc/expected_doc_id "
              f"label and were skipped. Add labels to score them.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, ensure_ascii=False, indent=2)
    if misses:
        miss_path = os.path.splitext(args.out)[0] + "_misses.json"
        with open(miss_path, "w", encoding="utf-8") as f:
            json.dump(misses, f, ensure_ascii=False, indent=2)
        print(f"  {len(misses)} miss(es) written to {miss_path}")
    print(f"  Detail written to {args.out}")
    return summary


if __name__ == "__main__":
    main()