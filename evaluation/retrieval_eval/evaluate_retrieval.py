"""
Retrieval Evaluation Script — Person 1

Evaluates retrieval quality across:
- Embedding models (text-embedding-3-small, BGE-M3, E5-large)
- Chunk strategies (coarse, fine, hybrid)
- With/without reranking (ZeroEntropy)

Metrics:
- Precision@k (k=5, 10, 20)
- Recall@k
- MRR (Mean Reciprocal Rank)
- Hit Rate@k
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    retrieved_ids: list[str]
    gold_ids: list[str]
    scores: list[float]


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & gold) / len(top_k)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not gold:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    """Mean Reciprocal Rank — rank of the first relevant doc."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1 if any gold doc appears in top-k, else 0."""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def load_gold_dataset(path: Path) -> list[dict]:
    """Load gold Q&A pairs."""
    with open(path) as f:
        return json.load(f)


def evaluate_retriever(
    retriever_fn,
    gold_data: list[dict],
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """
    Run retriever on all gold queries and compute metrics.

    Args:
        retriever_fn: callable(query: str, k: int) -> list[{id, score}]
        gold_data: list of gold Q&A entries
        k_values: tuple of k values to evaluate at
    """
    results = {f"precision@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})
    results.update({f"hit_rate@{k}": [] for k in k_values})
    results["mrr"] = []

    max_k = max(k_values)

    for entry in gold_data:
        query = entry["question"]
        gold_ids = {p["paper_id"] for p in entry["gold_passages"]}

        retrieved = retriever_fn(query, max_k)
        retrieved_ids = [r["id"] for r in retrieved]

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(retrieved_ids, gold_ids, k))
            results[f"recall@{k}"].append(recall_at_k(retrieved_ids, gold_ids, k))
            results[f"hit_rate@{k}"].append(hit_rate_at_k(retrieved_ids, gold_ids, k))

        results["mrr"].append(mrr(retrieved_ids, gold_ids))

    # Average all metrics
    return {metric: sum(values) / len(values) for metric, values in results.items()}


if __name__ == "__main__":
    print("Retrieval evaluation script ready.")
    print("Usage: import and call evaluate_retriever() with your retriever function.")
    print("See individual notebook for full experiments.")
