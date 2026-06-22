"""
RAGAS evaluation of the local RAG — fully offline (local Ollama judge + bge-m3).

For each question (question + ground_truth) it runs the REAL engine to get the
retrieved contexts and the generated answer, then scores faithfulness, answer
relevancy, context precision and context recall.

Reliability (why this rewrite)
------------------------------
A local 7B judge is slow and Ollama serializes requests, so high concurrency just
makes everything time out → NaN metrics. This script therefore:
  - defaults to STRICT mode + low concurrency (--max-workers 1);
  - applies a per-question answer-generation timeout (--timeout) and keeps going;
  - SAVES PARTIAL results after every question (a crash/timeout never loses work);
  - writes failed/timed-out questions to a separate file;
  - reports the timeout count and the NaN count PER METRIC so "nan" is explained;
  - supports --limit to start small (run 5–10 questions first!);
  - supports --retrieval-only to skip the LLM judge entirely (delegates to
    retrieval_eval.py — fast sanity check before a full RAGAS run).

Prerequisites
-------------
  1. Ollama running with your models (LLM_MODEL + bge-m3).
  2. Your corpus already indexed (the engine reads the existing Chroma store).
  3. pip install -r backend/requirements.txt -r backend/requirements-dev.txt

Run
---
  export CHROMA_PATH=/path/to/data/chroma SQLITE_PATH=/path/to/data/rag.db
  # start SMALL on a local judge:
  python eval/ragas_eval.py --dataset eval/dataset.example.json --limit 5
  # fast retrieval sanity check (no LLM):
  python eval/ragas_eval.py --retrieval-only --dataset eval/dataset.example.json
"""

import argparse
import asyncio
import json
import math
import os
import sys

# Locate the backend code (rag_engine.py) whether run from the repo (<repo>/backend)
# or inside the container (flat /app layout).
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(_here), "backend"), os.path.dirname(_here), os.getcwd()):
    if os.path.exists(os.path.join(_cand, "rag_engine.py")):
        sys.path.insert(0, _cand)
        break

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EVAL_USER = {"id": "eval", "role": "admin"}   # admin → no visibility restrictions


def _dump(path, obj):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _collect_answer(rag, question: str, strict: bool, timeout: float) -> str:
    """Run the engine's streaming chat to completion (bounded by `timeout`) and
    return the full answer. Raises asyncio.TimeoutError if it overruns."""
    async def run():
        parts = []
        async for kind, payload in rag.stream_chat(question, [], None, EVAL_USER, strict):
            if kind == "token":
                parts.append(payload)
        return "".join(parts).strip()
    return asyncio.run(asyncio.wait_for(run(), timeout=timeout))


def build_samples(dataset, strict: bool, timeout: float, partial_path: str, failed_path: str):
    """Build RAGAS samples by running the real engine. Resilient: per-question
    errors/timeouts are recorded and skipped, and partial results are saved after
    every question so nothing is lost on a crash."""
    from rag_engine import RAGEngine
    rag = RAGEngine()
    samples, failed = [], []
    total = len(dataset)
    for i, row in enumerate(dataset, 1):
        q = row["question"]
        try:
            items = rag._relevant_nodes(q, None, EVAL_USER)
            contexts = [it.get("text", "") for it in items]
            answer = _collect_answer(rag, q, strict, timeout)
            samples.append({
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts or ["(no context retrieved)"],
                "reference": row["ground_truth"],
            })
            print(f"[{i}/{total}] OK       ctx={len(contexts):2d}  {q[:60]}")
        except Exception as e:                       # incl. asyncio.TimeoutError
            kind = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
            failed.append({"question": q, "reason": kind, "detail": str(e)})
            print(f"[{i}/{total}] {kind.upper():7s}  {q[:60]}  ({e})")
        # Persist partial progress after EVERY question.
        _dump(partial_path, samples)
        _dump(failed_path, failed)
    return samples, failed


def run_ragas(samples, metric_names: str, timeout: int, max_workers: int, out_csv: str):
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings      # preferred (newer)
    except Exception:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
    from ragas import evaluate, EvaluationDataset
    try:
        from ragas import RunConfig
    except Exception:
        from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    judge = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL))

    available = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": ResponseRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }
    names = [n.strip() for n in metric_names.split(",") if n.strip()] if metric_names else list(available)
    metrics = [available[n] for n in names if n in available] or list(available.values())

    dataset = EvaluationDataset.from_list(samples)
    # Serialize (low max_workers) with a generous timeout — a local judge can't
    # handle parallel load without timing out (→ NaN). Tunable via the CLI.
    run_config = RunConfig(timeout=timeout, max_workers=max_workers)
    print(f"\nEvaluating {len(samples)} sample(s) x {len(metrics)} metric(s) "
          f"[timeout={timeout}s, max_workers={max_workers}] — be patient on a local judge…")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings,
                      run_config=run_config)

    print("\n================ RAGAS scores ================")
    print(result)

    # NaN accounting per metric — explains "nan" averages instead of hiding them.
    try:
        df = result.to_pandas()
        df.to_csv(out_csv, index=False)
        print(f"\nPer-question detail written to {out_csv}")
        print("\n---- NaN count per metric (failed judge calls) ----")
        for col in df.columns:
            try:
                series = df[col]
                if series.map(lambda v: isinstance(v, float)).any():
                    nan_n = int(series.isna().sum())
                    valid = series.dropna()
                    mean = round(float(valid.mean()), 4) if len(valid) else None
                    print(f"  {col:42s} nan={nan_n:3d}/{len(series)}  mean(valid)={mean}")
            except Exception:
                continue
    except Exception as e:
        print(f"(could not build per-question dataframe: {e})")
    return result


def main():
    ap = argparse.ArgumentParser(description="Reliable, offline RAGAS evaluation of the local RAG.")
    ap.add_argument("--dataset", default="eval/dataset.example.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N questions (0 = all). Start small (5–10).")
    ap.add_argument("--metrics", default="",
                    help="comma list: faithfulness,answer_relevancy,context_precision,context_recall")
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-question answer-generation timeout AND RAGAS judge timeout (seconds)")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="RAGAS judge concurrency. Keep at 1 for a local Ollama judge.")
    ap.add_argument("--strict", dest="strict", action="store_true", default=True,
                    help="documents-only answers (default ON).")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="allow general answers (turn strict OFF).")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip the LLM judge; run the fast retrieval-only evaluation instead.")
    ap.add_argument("--out-prefix", default="eval/ragas",
                    help="prefix for output files (_samples.json, _failed.json, _results.csv).")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]

    if args.retrieval_only:
        import retrieval_eval
        summary, *_ = retrieval_eval.evaluate(dataset, top_k=5)
        print("\n================ Retrieval-only results ================")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    samples_path = f"{args.out_prefix}_samples.json"
    failed_path = f"{args.out_prefix}_failed.json"
    results_csv = f"{args.out_prefix}_results.csv"

    samples, failed = build_samples(dataset, args.strict, float(args.timeout),
                                    samples_path, failed_path)

    print(f"\n---- build summary ----")
    print(f"  strict mode      : {args.strict}")
    print(f"  succeeded        : {len(samples)}/{len(dataset)}")
    print(f"  failed           : {len(failed)}  "
          f"(timeouts={sum(1 for f in failed if f['reason'] == 'timeout')}, "
          f"errors={sum(1 for f in failed if f['reason'] == 'error')})")
    if failed:
        print(f"  failed questions written to {failed_path}")
    if not samples:
        print("\nNo successful samples — skipping RAGAS. Check Ollama is running and raise --timeout.")
        return

    run_ragas(samples, args.metrics, args.timeout, args.max_workers, results_csv)


if __name__ == "__main__":
    main()