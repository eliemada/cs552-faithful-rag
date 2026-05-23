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


def _aggregate_usage(per_sample: list[dict]) -> dict:
    """Sum token counts and cost across answer-LLM calls for one pipeline.

    RAGAS judge calls are *not* included here — they're attributed to the
    judge model and tracked separately if at all. This block only reports
    the cost of generating answers (RAG vs long-context), which is the
    M2 trade-off story.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    n = 0
    for row in per_sample:
        u = row.get("usage")
        if not u:
            continue
        totals["prompt_tokens"] += int(u["prompt_tokens"])
        totals["completion_tokens"] += int(u["completion_tokens"])
        totals["total_tokens"] += int(u["total_tokens"])
        totals["cost_usd"] += float(u["cost_usd"])
        n += 1
    if n:
        totals["per_query_avg_cost_usd"] = totals["cost_usd"] / n
        totals["per_query_avg_total_tokens"] = totals["total_tokens"] / n
    totals["n_calls"] = n
    return totals


def _markdown_summary(
    rag_agg: dict, lc_agg: dict, n: int, rag_usage: dict | None = None, lc_usage: dict | None = None
) -> str:
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
    if rag_usage and lc_usage and rag_usage.get("n_calls") and lc_usage.get("n_calls"):
        lines.extend(
            [
                "",
                "## Answer-LLM cost & tokens (judge calls excluded)",
                "",
                "| measure | chunked RAG | long-context | ratio (LC / RAG) |",
                "|---|---|---|---|",
            ]
        )
        r_cost = rag_usage["cost_usd"]
        lc_cost = lc_usage["cost_usd"]
        r_tok = rag_usage["total_tokens"]
        lc_tok = lc_usage["total_tokens"]
        r_avg = rag_usage.get("per_query_avg_cost_usd", 0.0)
        lc_avg = lc_usage.get("per_query_avg_cost_usd", 0.0)
        ratio_cost = (lc_cost / r_cost) if r_cost else float("inf")
        ratio_tok = (lc_tok / r_tok) if r_tok else float("inf")
        lines.append(f"| total cost (USD) | ${r_cost:.5f} | ${lc_cost:.5f} | {ratio_cost:.1f}× |")
        lines.append(f"| total tokens     | {r_tok:,} | {lc_tok:,} | {ratio_tok:.1f}× |")
        lines.append(f"| avg cost / query | ${r_avg:.5f} | ${lc_avg:.5f} | — |")
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
            "usage": _aggregate_usage(rag_result.per_sample),
            "per_sample": rag_result.per_sample,
        },
        "long_context": {
            "n": lc_result.n,
            "aggregate": lc_result.aggregate,
            "usage": _aggregate_usage(lc_result.per_sample),
            "per_sample": lc_result.per_sample,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    md_path = args.output.with_suffix(".md")
    md_path.write_text(
        _markdown_summary(
            rag_result.aggregate,
            lc_result.aggregate,
            n=len(rag_samples),
            rag_usage=payload["rag"]["usage"],
            lc_usage=payload["long_context"]["usage"],
        )
    )

    print(f"\nWrote {_pretty(args.output)}")
    print(f"Wrote {_pretty(md_path)}")
    print("\nAggregate:")
    for m in METRIC_NAMES:
        print(f"  {m:<22}  RAG={rag_result.aggregate[m]:.3f}  LC={lc_result.aggregate[m]:.3f}")
    rag_u = payload["rag"]["usage"]
    lc_u = payload["long_context"]["usage"]
    if rag_u.get("n_calls") and lc_u.get("n_calls"):
        print(
            f"\nAnswer-LLM cost (judge calls excluded):"
            f"\n  RAG total ${rag_u['cost_usd']:.5f}  ({rag_u['total_tokens']:,} tokens, avg ${rag_u['per_query_avg_cost_usd']:.5f}/query)"
            f"\n  LC  total ${lc_u['cost_usd']:.5f}  ({lc_u['total_tokens']:,} tokens, avg ${lc_u['per_query_avg_cost_usd']:.5f}/query)"
        )
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
