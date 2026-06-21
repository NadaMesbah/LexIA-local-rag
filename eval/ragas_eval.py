"""
RAGAS evaluation of the local RAG — fully offline (local Ollama judge + bge-m3).

What it does
------------
For each question in a dataset (question + ground_truth), it runs the REAL engine
to get (a) the retrieved contexts and (b) the generated answer, then scores:
  - faithfulness        : is the answer supported by the retrieved contexts?
  - answer relevancy    : does the answer address the question?
  - context precision   : are the retrieved contexts relevant (not noise)?
  - context recall      : did retrieval bring back what's needed for the answer?

Prerequisites
-------------
  1. Ollama running with your models (LLM_MODEL + bge-m3) — same as the app.
  2. Your corpus already indexed (the engine reads the existing Chroma store).
  3. pip install -r backend/requirements.txt -r backend/requirements-dev.txt

Run
---
  # point at the SAME data the app uses (so the corpus is already indexed):
  export CHROMA_PATH=/path/to/data/chroma SQLITE_PATH=/path/to/data/rag.db
  python eval/ragas_eval.py --dataset eval/dataset.example.json

Notes
-----
- The judge is your local model, so scores are indicative, not absolute — use them
  to COMPARE configs (qwen2.5:7b vs aya-expanse:8b, hybrid on/off, cutoff values).
- RAGAS APIs move between versions; if an import fails, adjust it to your installed
  ragas (this targets the 0.2.x line).
"""

import argparse
import asyncio
import json
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


def _collect_answer(rag, question: str) -> str:
    """Run the engine's streaming chat to completion and return the full answer."""
    async def run():
        parts = []
        async for kind, payload in rag.stream_chat(question, [], None, EVAL_USER):
            if kind == "token":
                parts.append(payload)
        return "".join(parts).strip()
    return asyncio.run(run())


def build_samples(dataset):
    from rag_engine import RAGEngine
    rag = RAGEngine()
    samples = []
    for row in dataset:
        q = row["question"]
        items = rag._relevant_nodes(q, None, EVAL_USER)
        contexts = [it.get("text", "") for it in items]
        answer = _collect_answer(rag, q)
        print(f"\nQ: {q}\nA: {answer}\ncontexts: {len(contexts)}")
        samples.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": contexts or ["(no context retrieved)"],
            "reference": row["ground_truth"],
        })
    return samples


def run_ragas(samples):
    # Local, offline judge + embeddings via Ollama.
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings  # preferred (newer)
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

    dataset = EvaluationDataset.from_list(samples)
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]
    # A local 7B judge is SLOW and Ollama serializes requests, so hammering it with
    # many parallel jobs just makes them all time out. Serialize (low max_workers)
    # and give each job a generous timeout. Tune via RAGAS_WORKERS / RAGAS_TIMEOUT.
    run_config = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "1800")),
        max_workers=int(os.getenv("RAGAS_WORKERS", "1")),
    )
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings,
                      run_config=run_config)
    print("\n================ RAGAS scores ================")
    print(result)
    try:
        result.to_pandas().to_csv("eval/ragas_results.csv", index=False)
        print("Per-question detail written to eval/ragas_results.csv")
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.example.json")
    args = ap.parse_args()
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    samples = build_samples(dataset)
    run_ragas(samples)


if __name__ == "__main__":
    main()
