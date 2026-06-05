"""Multi-config RAGAS sweep: scale past the M2 n=8 single-config preliminary run.

This is the M3 driver. It evaluates the chunked-RAG pipeline across *several*
retriever configurations on the *full* evaluable gold set, and compares all of
them against a single shared long-context baseline.

Two design choices keep the cost tractable on RCP:

1. **The long-context arm is retriever-independent.** It stuffs the full text of
   each question's cited papers (``gold_paper_ids``) into the context — the
   retriever never runs. So we generate + score it **once** and reuse it for
   every config's comparison, instead of re-paying for it per config.

2. **Resumable.** Each config writes its own ``ragas_<config>.json``. A config
   whose output already exists is skipped (unless ``--overwrite``). If a long
   RCP job dies partway, just resubmit — finished configs are not recomputed.

Presets (``--configs``):

* ``sota``   — just ``e5_large_coarse_rerank`` (M2 retrieval SOTA). Cheapest;
               lifts the n=8 ceiling on the single best retriever.
* ``key``    — SOTA + OpenAI baseline, both coarse and fine. A four-config
               comparison that exposes the coarse-vs-fine chunk-boundary effect
               without paying for the full grid. **Default.**
* ``base16`` — all 16 base configs (4 embedders × coarse/fine × ±reranker).
* ``all``    — every config in ``CONFIGS`` (includes the 9 chunker variants).

You may also pass an explicit comma-separated list of config names.

CLI::

    # Sanity-check the plan offline (no API calls), then queue the real run:
    uv run python -m scripts.run_ragas_sweep --configs key --sample 0 --list-only

    uv run python -m scripts.run_ragas_sweep \
        --configs key \
        --sample 0 \
        --out-dir evaluation/ragas_eval/results/m3_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
from evaluation.retrieval_eval.retrievers import (
    CONFIGS,
    CONFIGS_BY_NAME,
    DEFAULT_INDEXES_DIR,
    load_adapter,
)
from scripts.run_ragas_experiment import (
    DEFAULT_ANSWER_MODEL_LC,
    DEFAULT_ANSWER_MODEL_RAG,
    DEFAULT_LC_CHARS_PER_PAPER,
    _aggregate_usage,
    _load_gold_answer,
    _select_sample,
    _to_chunks_metadata_path,
)

logger = logging.getLogger(__name__)

SOTA_CONFIG = "e5_large_coarse_rerank"
DEFAULT_OUT_DIR = REPO_ROOT / "evaluation" / "ragas_eval" / "results" / "m3_sweep"

_BASE16 = [c.name for c in CONFIGS if c.chunk_type in ("coarse", "fine")]
PRESETS: dict[str, list[str]] = {
    "sota": [SOTA_CONFIG],
    "key": [SOTA_CONFIG, "e5_large_fine_rerank", "coarse_rerank", "fine_rerank"],
    "base16": _BASE16,
    "all": [c.name for c in CONFIGS],
}


def _resolve_configs(spec: str) -> list[str]:
    """Map a preset name or comma-separated list to concrete config names.

    De-duplicates while preserving order and validates every name against
    ``CONFIGS_BY_NAME`` so we fail before spending any API budget.
    """
    if spec in PRESETS:
        names = PRESETS[spec]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    unknown = [n for n in ordered if n not in CONFIGS_BY_NAME]
    if unknown:
        raise SystemExit(
            f"Unknown retriever config(s): {', '.join(unknown)}\n"
            f"Valid names: {', '.join(sorted(CONFIGS_BY_NAME))}"
        )
    return ordered


def _pretty(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _config_block(args: argparse.Namespace, n_sample: int) -> dict:
    return {
        "sample_size": n_sample,
        "seed": args.seed,
        "answer_model_rag": args.answer_model_rag,
        "answer_model_lc": args.answer_model_lc,
        "judge_model": args.judge_model,
        "embed_model": args.embed_model,
        "top_k": args.top_k,
        "lc_chars_per_paper": args.lc_chars_per_paper,
    }


def _run_long_context(
    args: argparse.Namespace,
    sample: list,
    out_path: Path,
) -> dict:
    """Generate + score the shared long-context baseline once, with resume."""
    if out_path.exists() and not args.overwrite:
        print(f"Long-context already done → reusing {_pretty(out_path)}")
        return json.loads(out_path.read_text())["long_context"]

    print(f"\nGenerating long-context answers (model={args.answer_model_lc}) ...")
    lc_samples: list[RagasSample] = []
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
        except Exception as exc:  # pragma: no cover — transient API/IO errors
            print(f" FAILED: {exc}")

    print(f"Scoring long-context samples ({len(lc_samples)}) with RAGAS ...")
    result = evaluate_samples(
        lc_samples, judge_model=args.judge_model, embed_model=args.embed_model
    )
    block = {
        "n": result.n,
        "aggregate": result.aggregate,
        "usage": _aggregate_usage(result.per_sample),
        "per_sample": result.per_sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"config": _config_block(args, len(sample)), "long_context": block}, indent=2)
    )
    print(f"Wrote {_pretty(out_path)}")
    return block


def _run_rag_config(
    args: argparse.Namespace,
    config_name: str,
    sample: list,
    out_path: Path,
) -> dict:
    """Generate + score the RAG pipeline for one retriever config, with resume."""
    if out_path.exists() and not args.overwrite:
        print(f"  {config_name}: already done → reusing {_pretty(out_path)}")
        return json.loads(out_path.read_text())["rag"]

    retriever = load_adapter(config_name, indexes_dir=args.indexes_dir)
    chunk_lookup = build_chunk_lookup(_to_chunks_metadata_path(retriever.config.chunk_type))

    rag_samples: list[RagasSample] = []
    for i, q in enumerate(sample, 1):
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
        except Exception as exc:  # pragma: no cover — transient API/IO errors
            print(f"    [{i}/{len(sample)}] {q.query_id} FAILED: {exc}")

    result = evaluate_samples(
        rag_samples, judge_model=args.judge_model, embed_model=args.embed_model
    )
    block = {
        "n": result.n,
        "aggregate": result.aggregate,
        "usage": _aggregate_usage(result.per_sample),
        "per_sample": result.per_sample,
    }
    payload = {
        "config": {**_config_block(args, len(sample)), "retriever_config": config_name},
        "chunk_type": retriever.config.chunk_type,
        "rag": block,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return block


def _write_summary(
    args: argparse.Namespace,
    sample: list,
    config_names: list[str],
    config_blocks: dict[str, dict],
    lc_block: dict | None,
    out_dir: Path,
) -> None:
    """Write the combined config×metric table (JSON + Markdown)."""
    summary = {
        "config": _config_block(args, len(sample)),
        "questions": [q.query_id for q in sample],
        "configs": {
            name: {
                "n": block["n"],
                "aggregate": block["aggregate"],
                "usage": block["usage"],
                "chunk_type": CONFIGS_BY_NAME[name].chunk_type,
            }
            for name, block in config_blocks.items()
        },
    }
    if lc_block is not None:
        summary["long_context"] = {
            "n": lc_block["n"],
            "aggregate": lc_block["aggregate"],
            "usage": lc_block["usage"],
        }
    json_path = out_dir / "ragas_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    header = "| config | chunk | n | " + " | ".join(f"`{m}`" for m in METRIC_NAMES) + " | $/query |"
    sep = "|" + "---|" * (4 + len(METRIC_NAMES))
    lines = [
        "# RAGAS sweep — M3",
        "",
        f"n = {len(sample)} evaluable gold questions (shared across configs).",
        "",
        header,
        sep,
    ]
    for name in config_names:
        block = config_blocks.get(name)
        if not block:
            continue
        agg = block["aggregate"]
        cells = " | ".join(f"{agg.get(m, 0.0):.3f}" for m in METRIC_NAMES)
        avg_cost = block["usage"].get("per_query_avg_cost_usd", 0.0)
        lines.append(
            f"| `{name}` | {CONFIGS_BY_NAME[name].chunk_type} | {block['n']} | {cells} | ${avg_cost:.5f} |"
        )
    if lc_block is not None:
        agg = lc_block["aggregate"]
        cells = " | ".join(f"{agg.get(m, 0.0):.3f}" for m in METRIC_NAMES)
        avg_cost = lc_block["usage"].get("per_query_avg_cost_usd", 0.0)
        lines.append(f"| **long-context** | — | {lc_block['n']} | {cells} | ${avg_cost:.5f} |")
    md_path = out_dir / "ragas_sweep_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {_pretty(json_path)}")
    print(f"Wrote {_pretty(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_QA)
    parser.add_argument(
        "--configs",
        default="key",
        help="Preset (sota|key|base16|all) or comma-separated config names. Default: key.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="How many gold questions to evaluate. 0 (default) = all evaluable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--answer-model-rag", default=DEFAULT_ANSWER_MODEL_RAG)
    parser.add_argument("--answer-model-lc", default=DEFAULT_ANSWER_MODEL_LC)
    parser.add_argument("--lc-chars-per-paper", type=int, default=DEFAULT_LC_CHARS_PER_PAPER)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-long-context",
        action="store_true",
        help="Only run the RAG configs (the shared LC baseline is left untouched).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute configs even if their result JSON already exists (disables resume).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the plan (configs, question count, rough call volume) and exit. No API calls.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    config_names = _resolve_configs(args.configs)

    print("Loading gold dataset ...")
    queries = resolve_from_file(args.gold)
    sample_size = args.sample if args.sample > 0 else len(queries)
    sample = _select_sample(queries, sample_size, args.seed)
    print(f"  {len(queries)} evaluable; using {len(sample)} questions")
    print(f"  configs ({len(config_names)}): {', '.join(config_names)}")

    if args.list_only:
        n_rag = len(config_names) * len(sample)
        n_lc = 0 if args.skip_long_context else len(sample)
        # RAGAS judge fires several calls per sample per metric; this is a
        # deliberately rough lower bound to gauge magnitude, not a billing quote.
        n_judge_samples = n_rag + n_lc
        print("\n--- plan (no API calls made) ---")
        print(f"  RAG answer generations:  {n_rag}")
        print(f"  LC  answer generations:  {n_lc}")
        print(
            f"  RAGAS-scored samples:    {n_judge_samples} (×~4 metrics, several judge calls each)"
        )
        print(f"  out-dir:                 {_pretty(args.out_dir)}")
        print(
            "  resume:                  on"
            if not args.overwrite
            else "  resume:  off (--overwrite)"
        )
        return 0

    load_dotenv()
    for required in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(required):
            print(f"ERROR: {required} is not set.", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lc_block: dict | None = None
    if not args.skip_long_context:
        lc_block = _run_long_context(args, sample, args.out_dir / "ragas_long_context.json")

    config_blocks: dict[str, dict] = {}
    for idx, name in enumerate(config_names, 1):
        print(f"\n[{idx}/{len(config_names)}] config: {name}")
        t0 = time.perf_counter()
        config_blocks[name] = _run_rag_config(
            args, name, sample, args.out_dir / f"ragas_{name}.json"
        )
        agg = config_blocks[name]["aggregate"]
        dt = time.perf_counter() - t0
        print(
            f"  done in {dt:.0f}s — "
            + "  ".join(f"{m}={agg.get(m, 0.0):.3f}" for m in METRIC_NAMES)
        )

    _write_summary(args, sample, config_names, config_blocks, lc_block, args.out_dir)

    print("\nAggregate (RAG by config):")
    for name in config_names:
        agg = config_blocks[name]["aggregate"]
        print(f"  {name:<28} " + "  ".join(f"{m}={agg.get(m, 0.0):.3f}" for m in METRIC_NAMES))
    if lc_block is not None:
        agg = lc_block["aggregate"]
        print(
            f"  {'long-context':<28} "
            + "  ".join(f"{m}={agg.get(m, 0.0):.3f}" for m in METRIC_NAMES)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
