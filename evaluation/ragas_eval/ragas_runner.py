"""
RAGAS Evaluation Runner — Person 4

Runs end-to-end evaluation using the RAGAS framework.
Compares chunked RAG vs long-context approaches.

Metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Install: pip install ragas
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

from evaluation.common import (
    artifact_root,
    available_models,
    ensure_paper_artifacts,
    generate,
    load_gold_qa,
    load_paper_chunks,
    load_paper_markdown,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    gold_dataset_path: Path = Path("evaluation/gold_dataset/gold_qa.json")
    output_dir: Path = Path("evaluation/ragas_eval/results")
    models_to_test: tuple[str, ...] = ()
    index_basename: str = "bge_m3_coarse"
    chunk_type: str = "coarse"
    top_k: int = 6
    max_questions: int | None = 8
    max_context_chars: int = 12000
    max_tokens: int = 256
    long_context_model: str | None = None
    long_context_max_chars: int = 120000


def _ensure_shared_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    shared_path = repo_root / "packages" / "shared"
    if shared_path.exists() and str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))
    return repo_root


def _subset(rows: list[dict], max_questions: int | None) -> list[dict]:
    if max_questions is None or max_questions <= 0 or max_questions >= len(rows):
        return rows
    return rows[:max_questions]


def _paper_ids(item: dict) -> list[str]:
    ids: set[str] = set()
    for claim in item.get("claims", []):
        for span in claim.get("supporting_spans", []):
            pid = span.get("paper_id")
            if pid:
                ids.add(pid)
    return sorted(ids)


def _safe_name(model_spec: str) -> str:
    return model_spec.replace(":", "_").replace("/", "_")


def _as_text(result: Any) -> str:
    if isinstance(result, dict):
        return result.get("text", "")
    return getattr(result, "text", "")


def _build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
    return f"Question: {question}\n\nContext:\n{joined}\n\nAnswer:"


def _safe_generate(model_spec: str, prompt: str, *, max_tokens: int) -> str:
    try:
        return generate(
            model_spec,
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            system=(
                "You are a careful research assistant. Answer using only the provided context. "
                "If the context is insufficient, say you do not know."
            ),
        ).strip()
    except Exception as exc:  # pragma: no cover - best-effort in notebooks
        return f"[generation failed: {exc}]"


def _build_retriever(config: EvalConfig):
    _ensure_shared_on_path()
    from rag_pipeline.rag.embedder import SentenceTransformerEmbedder
    from rag_pipeline.rag.retriever import HybridRetriever

    indexes_dir = artifact_root() / "indexes"
    index_path = indexes_dir / f"{config.index_basename}.faiss"
    meta_path = indexes_dir / f"{config.index_basename}_metadata.json"

    embedder = None
    if config.index_basename.startswith("bge_m3"):
        embedder = SentenceTransformerEmbedder("BAAI/bge-m3", short_name="bge_m3")
    elif config.index_basename.startswith("e5_large"):
        embedder = SentenceTransformerEmbedder("intfloat/e5-large-v2", short_name="e5_large")

    if index_path.exists() and meta_path.exists():
        openai_key = os.environ.get("OPENAI_API_KEY")
        if embedder is None and not openai_key:
            raise RuntimeError(
                "OpenAI key missing for the default index. "
                "Set OPENAI_API_KEY or switch to a local index_basename."
            )
        return HybridRetriever.from_path(
            indexes_dir=indexes_dir,
            chunk_type=config.chunk_type,
            openai_api_key=openai_key,
            embedder=embedder,
            index_basename=config.index_basename,
            zeroentropy_api_key=None,
        )

    logger.warning(
        "Index %s missing; building a mini index from gold papers.", config.index_basename
    )
    import faiss

    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", short_name="bge_m3")
    gold = load_gold_qa()
    paper_ids = set()
    for item in _subset(gold, config.max_questions):
        paper_ids.update(_paper_ids(item))

    if not paper_ids:
        raise RuntimeError("No paper IDs found to build a mini index.")

    chunks = []
    for pid in sorted(paper_ids):
        chunks.extend(load_paper_chunks(pid, config.chunk_type))

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_passages(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    class MiniRetriever:
        def __init__(self, index, chunks, embedder):
            self.index = index
            self.chunks = chunks
            self.embedder = embedder

        def search(self, query: str, top_k: int = 5):
            vector = self.embedder.encode_queries([query])
            distances, indices = self.index.search(vector, top_k)
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx == -1:
                    break
                chunk = self.chunks[idx]
                results.append(
                    {
                        "chunk_id": chunk.get("chunk_id", f"mini_{idx}"),
                        "paper_id": chunk.get("paper_id", ""),
                        "text": chunk.get("text", ""),
                        "score": float(score),
                        "rank": rank,
                    }
                )
            return results

    return MiniRetriever(index, chunks, embedder)


def _run_rag(rows: list[dict], retriever, model_spec: str, config: EvalConfig) -> list[dict]:
    out: list[dict] = []
    for item in rows:
        question = item["question"]
        ground_truth = item.get("gold_answer", "")
        start = time.perf_counter()
        hits = retriever.search(question, top_k=config.top_k)
        contexts = [_as_text(h) for h in hits]
        prompt = _build_prompt(question, contexts)
        answer = _safe_generate(model_spec, prompt, max_tokens=config.max_tokens)
        latency = time.perf_counter() - start
        out.append(
            {
                "id": item.get("id"),
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "model_spec": model_spec,
                "variant": "rag",
                "latency_s": latency,
                "prompt_chars": len(prompt),
                "context_chars": sum(len(c) for c in contexts),
                "answer_chars": len(answer),
            }
        )
    return out


def _run_long_context(rows: list[dict], model_spec: str, config: EvalConfig) -> list[dict]:
    out: list[dict] = []
    for item in rows:
        question = item["question"]
        ground_truth = item.get("gold_answer", "")
        paper_ids = _paper_ids(item)
        for pid in paper_ids:
            ensure_paper_artifacts(pid)
        docs = [f"[Paper {pid}]\n{load_paper_markdown(pid)}" for pid in paper_ids]
        context = "\n\n".join(docs)
        if config.long_context_max_chars and len(context) > config.long_context_max_chars:
            context = context[: config.long_context_max_chars]
        prompt = f"Question: {question}\n\nFull papers:\n{context}\n\nAnswer:"
        start = time.perf_counter()
        answer = _safe_generate(model_spec, prompt, max_tokens=config.max_tokens)
        latency = time.perf_counter() - start
        out.append(
            {
                "id": item.get("id"),
                "question": question,
                "answer": answer,
                "contexts": [context],
                "ground_truth": ground_truth,
                "model_spec": model_spec,
                "variant": "long_context",
                "latency_s": latency,
                "prompt_chars": len(prompt),
                "context_chars": len(context),
                "answer_chars": len(answer),
            }
        )
    return out


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        from datasets import Dataset  # noqa: F401
    except Exception:
        return False
    return True


def _ragas_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("RAGAS_RUN"))


def _run_ragas(rows: list[dict], output_path: Path, tag: str) -> None:
    if not rows:
        return
    if not _ragas_available():
        logger.info("RAGAS not available; skipping %s.", tag)
        return
    if not _ragas_enabled():
        logger.info("RAGAS disabled; set OPENAI_API_KEY or RAGAS_RUN=1.")
        return

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "id": [row.get("id") for row in rows],
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = result.to_pandas()
    df["variant"] = tag
    df.to_csv(output_path, index=False)


def run_ragas_evaluation(config: EvalConfig = EvalConfig()) -> None:
    """
    Run RAGAS evaluation on the RAG pipeline.

    Load the gold dataset, generate RAG answers, and optionally score with RAGAS.
    """
    _ensure_shared_on_path()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold_qa()
    if not gold:
        raise RuntimeError("Gold dataset is empty; cannot run RAGAS evaluation.")

    models = config.models_to_test or tuple(
        available_models(include_api=bool(os.environ.get("OPENROUTER_API_KEY")))
    )
    if not models:
        raise RuntimeError("No model specs available; set OPENROUTER_API_KEY or use local models.")

    subset = _subset(gold, config.max_questions)
    retriever = _build_retriever(config)

    for model_spec in models:
        rows = _run_rag(subset, retriever, model_spec, config)
        suffix = _safe_name(model_spec)
        _write_json(output_dir / f"rag_inputs_{suffix}.json", rows)
        _run_ragas(rows, output_dir / f"ragas_scores_{suffix}.csv", "rag")


def run_long_context_baseline(config: EvalConfig = EvalConfig()) -> None:
    """
    Compare against long-context approach (128K context window).

    Instead of chunked retrieval, stuff full paper text into the context.

    Compare against a long-context approach (128K window).
    """
    _ensure_shared_on_path()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_spec = config.long_context_model or os.environ.get("LONG_CONTEXT_MODEL_SPEC")
    if not model_spec:
        logger.warning("No long-context model spec; set LONG_CONTEXT_MODEL_SPEC to run.")
        return

    gold = load_gold_qa()
    if not gold:
        raise RuntimeError("Gold dataset is empty; cannot run long-context baseline.")

    subset = _subset(gold, config.max_questions)
    rows = _run_long_context(subset, model_spec, config)
    suffix = _safe_name(model_spec)
    _write_json(output_dir / f"long_context_inputs_{suffix}.json", rows)
    _run_ragas(rows, output_dir / f"ragas_scores_long_context_{suffix}.csv", "long_context")


if __name__ == "__main__":
    print("RAGAS evaluation runner ready.")
    print("Install: pip install ragas datasets")
    print("See individual notebook for full experiments.")
