"""Retrieval evaluation CLI.

Runs one named retriever configuration over the gold dataset, computes
per-query metrics at both **paper** and **chunk** granularity, and writes
both the per-query trace and the aggregate to a JSON file.

Two granularities of ground truth, two flavours of metric:

* **paper-level** (primary) — for each retrieved chunk we look at its
  ``paper_id``; the query is "scored" against the set of paper IDs cited
  by the gold question. Every evaluable query carries this signal.
* **chunk-level** (secondary) — for queries where the gold span actually
  overlaps a chunk at the given granularity (the ``has_chunk_coverage``
  subset), we also score retrieved ``chunk_id`` vs. gold ``chunk_id``.
  The existing chunker skips 40-80 % of document content between
  sections, so this metric is only well-defined on a subset of queries —
  see ``gold_resolver.py`` for the coverage-gap analysis.

Output schema::

    {
      "config": "coarse_faiss",
      "chunk_type": "coarse",
      "use_reranker": false,
      "k_values": [5, 10, 20],
      "n_queries_paper":  37,    # evaluable for paper-level
      "n_queries_chunk":  20,    # subset with chunk coverage
      "aggregate": {
        "paper":  {"precision@5": ..., "recall@5": ..., ...},
        "chunk":  {"precision@5": ..., "recall@5": ..., ...}
      },
      "per_query": [{"query_id": "q001", "paper": {...}, "chunk": {...}}, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Final, Iterable

from evaluation.gold_dataset._validator import DEFAULT_GOLD_QA, REPO_ROOT
from evaluation.retrieval_eval.gold_resolver import (
    DEFAULT_CHUNKS_DIR,
    ResolvedQuery,
    resolve_from_file,
)
from evaluation.retrieval_eval.retrievers import (
    CONFIGS_BY_NAME,
    DEFAULT_INDEXES_DIR,
    RetrieverAdapter,
    load_adapter,
)

DEFAULT_RESULTS_DIR: Final[Path] = REPO_ROOT / "evaluation" / "retrieval_eval" / "results"
DEFAULT_K_VALUES: Final[tuple[int, ...]] = (5, 10, 20)

logger = logging.getLogger(__name__)


# ---- metrics ---------------------------------------------------------------


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of the top-k retrieved IDs that are in the gold set."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in gold) / len(top_k)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold IDs found anywhere in the top-k."""
    if not gold:
        return 0.0
    top_k_set = set(retrieved[:k])
    return len(top_k_set & gold) / len(gold)


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold ID appears in the top-k, else 0.0."""
    return 1.0 if any(d in gold for d in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """1 / rank of the first gold hit (1-indexed). 0 if no hit."""
    for i, d in enumerate(retrieved):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


# ---- per-query metric record ----------------------------------------------


def _metrics_for(retrieved_ids: list[str], gold_ids: set[str], k_values: Iterable[int]) -> dict:
    out: dict[str, float] = {}
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(retrieved_ids, gold_ids, k)
        out[f"recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, gold_ids, k)
    out["mrr"] = reciprocal_rank(retrieved_ids, gold_ids)
    return out


def _empty_metrics(k_values: Iterable[int]) -> dict:
    """Placeholder used when a query has no chunk coverage at this granularity."""
    out: dict[str, float | None] = {}
    for k in k_values:
        out[f"precision@{k}"] = None
        out[f"recall@{k}"] = None
        out[f"hit_rate@{k}"] = None
    out["mrr"] = None
    return out


def _aggregate(per_query: list[dict], key: str, k_values: Iterable[int]) -> dict:
    """Mean each metric over queries where the key's metrics are populated."""
    metric_names = (
        [f"precision@{k}" for k in k_values]
        + [f"recall@{k}" for k in k_values]
        + [f"hit_rate@{k}" for k in k_values]
        + ["mrr"]
    )
    agg: dict[str, float | None] = {}
    for m in metric_names:
        values = [row[key][m] for row in per_query if row[key][m] is not None]
        agg[m] = sum(values) / len(values) if values else None
    agg["n"] = sum(1 for row in per_query if row[key]["mrr"] is not None)
    return agg


# ---- runner ---------------------------------------------------------------


def evaluate_config(
    adapter: RetrieverAdapter,
    queries: list[ResolvedQuery],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    progress: bool = True,
) -> dict:
    """Run one config over all queries, return the result dict ready to serialise."""
    max_k = max(k_values)
    per_query: list[dict] = []

    for i, q in enumerate(queries, 1):
        if progress:
            print(f"  [{i:>3}/{len(queries)}] {q.query_id} ...", end="", flush=True)
        t0 = time.perf_counter()
        results = adapter.search(q.query_text, k=max_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        retrieved_papers = [r["paper_id"] for r in results]
        retrieved_chunks = [r["chunk_id"] for r in results]

        # Paper-level metrics: always defined (gold_paper_ids is always non-empty)
        paper_metrics = _metrics_for(retrieved_papers, set(q.gold_paper_ids), k_values)

        # Chunk-level metrics: only defined when chunks at this granularity overlap gold spans
        chunk_gold = q.gold_chunk_ids.get(adapter.config.chunk_type, frozenset())
        if chunk_gold:
            chunk_metrics = _metrics_for(retrieved_chunks, set(chunk_gold), k_values)
        else:
            chunk_metrics = _empty_metrics(k_values)

        per_query.append(
            {
                "query_id": q.query_id,
                "category": q.category,
                "difficulty": q.difficulty,
                "gold_paper_count": len(q.gold_paper_ids),
                "gold_chunk_count": len(chunk_gold),
                "has_chunk_coverage": bool(chunk_gold),
                "latency_ms": round(elapsed_ms, 1),
                "paper": paper_metrics,
                "chunk": chunk_metrics,
            }
        )
        if progress:
            print(
                f" {elapsed_ms:6.0f}ms  paper-hit@10={paper_metrics['hit_rate@10']:.0f}", flush=True
            )

    return {
        "config": adapter.config.name,
        "chunk_type": adapter.config.chunk_type,
        "use_reranker": adapter.config.use_reranker,
        "k_values": list(k_values),
        "n_queries_total": len(queries),
        "n_queries_chunk_coverage": sum(1 for row in per_query if row["has_chunk_coverage"]),
        "aggregate": {
            "paper": _aggregate(per_query, "paper", k_values),
            "chunk": _aggregate(per_query, "chunk", k_values),
        },
        "per_query": per_query,
    }


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(CONFIGS_BY_NAME),
        help="Named retriever configuration to evaluate.",
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_QA)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to evaluation/retrieval_eval/results/<config>.json.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="k values to compute precision/recall/hit_rate at (default: 5 10 20).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query progress lines.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    output = args.output or (DEFAULT_RESULTS_DIR / f"{args.config}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Resolving gold dataset → chunks ...")
    queries = resolve_from_file(args.gold, chunks_dir=args.chunks_dir)
    print(f"  {len(queries)} evaluable queries")

    print(f"Loading retriever config: {args.config}")
    adapter = load_adapter(args.config, indexes_dir=args.indexes_dir)

    print("Running evaluation ...")
    result = evaluate_config(adapter, queries, k_values=tuple(args.ks), progress=not args.quiet)

    output.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {output}")
    print(
        f"  paper-level n={result['aggregate']['paper']['n']}: "
        f"hit@5={result['aggregate']['paper']['hit_rate@5']:.3f}  "
        f"hit@10={result['aggregate']['paper']['hit_rate@10']:.3f}  "
        f"mrr={result['aggregate']['paper']['mrr']:.3f}"
    )
    chunk_agg = result["aggregate"]["chunk"]
    if chunk_agg["n"]:
        print(
            f"  chunk-level n={chunk_agg['n']}: "
            f"hit@5={chunk_agg['hit_rate@5']:.3f}  "
            f"hit@10={chunk_agg['hit_rate@10']:.3f}  "
            f"mrr={chunk_agg['mrr']:.3f}"
        )
    else:
        print("  chunk-level: no queries with chunk coverage at this granularity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
