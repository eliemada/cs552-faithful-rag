"""End-to-end RAGAS experiment: RAG vs long-context, on a sample of gold questions.

For each selected gold question we generate two answers — one from the
chunked-RAG pipeline, one from the long-context pipeline — and score both
sets with the same four RAGAS metrics. The side-by-side comparison is
the M2 deliverable.

This is deliberately a *preliminary* run (default ``--sample 8``): RAGAS'
faithfulness metric alone fires ~3 judge calls per claim per sample, so a
50-question full run can cost a few dollars. M2 only needs an honest
trend, not Pareto-optimal coverage. Scale up for M3.

CLI::

    uv run python -m scripts.run_ragas_experiment \
        --sample 8 \
        --retriever-config coarse_rerank \
        --output evaluation/ragas_eval/results/01_rag_vs_long_context.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from evaluation.gold_dataset._validator import DEFAULT_GOLD_QA, REPO_ROOT
from evaluation.ragas_eval.pipelines import (
    DEFAULT_PROCESSED_DIR,
    RagasSample,
    build_chunk_lookup,
    run_long_context_pipeline,
    run_rag_pipeline,
)
from evaluation.ragas_eval.ragas_runner import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_JUDGE_MODEL,
    METRIC_NAMES,
    evaluate_samples,
)
from evaluation.retrieval_eval.gold_resolver import resolve_from_file
from evaluation.retrieval_eval.retrievers import DEFAULT_INDEXES_DIR, load_adapter

DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "ragas_eval" / "results"
DEFAULT_SAMPLE_SIZE = 8
DEFAULT_RETRIEVER_CONFIG = "coarse_rerank"
DEFAULT_ANSWER_MODEL_RAG = "api:openrouter/openai/gpt-4o-mini"
DEFAULT_ANSWER_MODEL_LC = "api:openrouter/google/gemini-2.5-flash"
DEFAULT_LC_CHARS_PER_PAPER = 60_000  # ~15k tokens — fits one paper comfortably in any LC model

logger = logging.getLogger(__name__)


def _select_sample(queries: list, sample_size: int, seed: int) -> list:
    """Stratify by category so the small sample still spans question types."""
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for q in queries:
        by_cat.setdefault(q.category, []).append(q)
    selected: list = []
    # round-robin across categories until we hit sample_size
    while sum(len(v) for v in by_cat.values()) > 0 and len(selected) < sample_size:
        for cat in list(by_cat):
            if not by_cat[cat]:
                continue
            idx = rng.randrange(len(by_cat[cat]))
            selected.append(by_cat[cat].pop(idx))
            if len(selected) >= sample_size:
                break
    return selected


def _to_chunks_metadata_path(chunk_type: str) -> Path:
    return REPO_ROOT / "data" / "s3_archive" / "indexes" / f"{chunk_type}_metadata.json"


def _markdown_summary(rag_agg: dict, lc_agg: dict, n: int) -> str:
    lines = [
        "# RAGAS evaluation — preliminary",
        "",
        f"n = {n} gold questions, stratified across categories.",
        "",
        "| metric | chunked RAG | long-context | Δ (LC − RAG) |",
        "|---|---|---|---|",
    ]
    for m in METRIC_NAMES:
        r = rag_agg.get(m, 0.0)
        lc = lc_agg.get(m, 0.0)
        delta = lc - r
        lines.append(f"| `{m}` | {r:.3f} | {lc:.3f} | {delta:+.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_QA)
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"How many gold questions to evaluate (default {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retriever-config",
        default=DEFAULT_RETRIEVER_CONFIG,
        help="Name of the retrieval config to use for the RAG pipeline.",
    )
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--answer-model-rag", default=DEFAULT_ANSWER_MODEL_RAG)
    parser.add_argument("--answer-model-lc", default=DEFAULT_ANSWER_MODEL_LC)
    parser.add_argument(
        "--lc-chars-per-paper",
        type=int,
        default=DEFAULT_LC_CHARS_PER_PAPER,
        help="Truncate each paper's full text to this many chars before stuffing into LC context.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-long-context",
        action="store_true",
        help="Only run the RAG pipeline (useful for re-running judge changes).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "01_rag_vs_long_context.json",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    load_dotenv()

    # Fail fast on missing keys rather than burning compute first.
    for required in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(required):
            print(f"ERROR: {required} is not set.", file=sys.stderr)
            return 2

    print("Loading gold dataset ...")
    queries = resolve_from_file(args.gold)
    sample = _select_sample(queries, args.sample, args.seed)
    print(f"  {len(queries)} evaluable; sampled {len(sample)} stratified")

    print(f"Loading retriever: {args.retriever_config}")
    retriever = load_adapter(args.retriever_config, indexes_dir=args.indexes_dir)

    print("Building chunk text lookup ...")
    chunk_lookup = build_chunk_lookup(_to_chunks_metadata_path(retriever.config.chunk_type))
    print(f"  {len(chunk_lookup)} chunks indexed")

    print(f"\nGenerating RAG answers (model={args.answer_model_rag}) ...")
    rag_samples: list[RagasSample] = []
    for i, q in enumerate(sample, 1):
        print(f"  [{i}/{len(sample)}] {q.query_id} ...", end="", flush=True)
        try:
            s = run_rag_pipeline(
                question=q.query_text,
                ground_truth=_load_gold_answer(args.gold, q.query_id),
                query_id=q.query_id,
                retriever=retriever,
                chunk_lookup=chunk_lookup,
                answer_model=args.answer_model_rag,
                top_k=args.top_k,
            )
            rag_samples.append(s)
            print(f" {len(s.contexts)} ctx, ans={len(s.answer)} chars")
        except Exception as exc:  # pragma: no cover — defensive against transient API errors
            print(f" FAILED: {exc}")

    lc_samples: list[RagasSample] = []
    if not args.skip_long_context:
        print(f"\nGenerating long-context answers (model={args.answer_model_lc}) ...")
        for i, q in enumerate(sample, 1):
            print(f"  [{i}/{len(sample)}] {q.query_id} ...", end="", flush=True)
            try:
                s = run_long_context_pipeline(
                    question=q.query_text,
                    ground_truth=_load_gold_answer(args.gold, q.query_id),
                    query_id=q.query_id,
                    paper_ids=sorted(q.gold_paper_ids),
                    answer_model=args.answer_model_lc,
                    processed_dir=args.processed_dir,
                    max_chars_per_paper=args.lc_chars_per_paper,
                )
                lc_samples.append(s)
                print(f" {len(s.contexts)} papers, ans={len(s.answer)} chars")
            except Exception as exc:  # pragma: no cover
                print(f" FAILED: {exc}")

    print(f"\nScoring RAG samples ({len(rag_samples)}) with RAGAS ...")
    rag_result = evaluate_samples(
        rag_samples, judge_model=args.judge_model, embed_model=args.embed_model
    )
    print("Scoring long-context samples ...")
    lc_result = evaluate_samples(
        lc_samples, judge_model=args.judge_model, embed_model=args.embed_model
    )

    payload = {
        "config": {
            "sample_size": args.sample,
            "seed": args.seed,
            "retriever_config": args.retriever_config,
            "answer_model_rag": args.answer_model_rag,
            "answer_model_lc": args.answer_model_lc,
            "judge_model": args.judge_model,
            "embed_model": args.embed_model,
            "top_k": args.top_k,
            "lc_chars_per_paper": args.lc_chars_per_paper,
        },
        "rag": {
            "n": rag_result.n,
            "aggregate": rag_result.aggregate,
            "per_sample": rag_result.per_sample,
        },
        "long_context": {
            "n": lc_result.n,
            "aggregate": lc_result.aggregate,
            "per_sample": lc_result.per_sample,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        _markdown_summary(rag_result.aggregate, lc_result.aggregate, n=len(rag_samples))
    )

    print(f"\nWrote {_pretty(args.output)}")
    print(f"Wrote {_pretty(md_path)}")
    print("\nAggregate:")
    for m in METRIC_NAMES:
        print(f"  {m:<22}  RAG={rag_result.aggregate[m]:.3f}  LC={lc_result.aggregate[m]:.3f}")
    return 0


def _load_gold_answer(gold_path: Path, query_id: str) -> str:
    """One-shot lookup of a single pair's gold_answer by id."""
    pairs = json.loads(gold_path.read_text() or "[]")
    for p in pairs:
        if p.get("id") == query_id:
            return p.get("gold_answer", "")
    return ""


def _pretty(path: Path) -> str:
    """Path display that prefers repo-relative but falls back to absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
