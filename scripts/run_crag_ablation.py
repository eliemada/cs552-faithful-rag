"""CRAG threshold ablation on the full gold set (M3 driver, Faruk).

Concretises the INCORRECT-branch behaviour the TA asked about: with Option A
(abstain) wired in ``corrective_rag.py``, this script measures *what abstaining
buys and costs* across the confidence-threshold grid, on the full gold set
(M2 only ran a 10-question slice).

Efficiency: retrieval + cross-encoder scoring is the expensive step, and the
normalised confidence is **independent of the thresholds**. So we *probe* each
gold question exactly once (retrieve top-k, score, record whether retrieval hit
the gold paper), cache that, then sweep every ``(tau_high, tau_low)`` cell
**analytically** over the cached confidences. No LLM is required — query
refinement is the only LLM step and is left to the notebook / live pipeline; the
sweep brackets its effect with strict/loose abstain bounds (see below).

Abstain accounting (per threshold cell):

* **strict** — only INCORRECT (conf < tau_low) abstains; AMBIGUOUS is assumed to
  recover via refine. Lower bound on abstains.
* **loose**  — INCORRECT *and* AMBIGUOUS abstain (the no-refine / refine-fails
  case). Upper bound on abstains.

Reported for both bounds: abstain precision/recall against the **unanswerable**
gold stratum (a "good" abstain is one where the gold is genuinely unanswerable),
and the correct-retrieval rate among *answered* questions vs. the no-CRAG
baseline.

CLI::

    # Fast probe + analytical sweep on the M2 SOTA retriever:
    uv run python -m scripts.run_crag_ablation \
        --retriever-config e5_large_coarse_rerank \
        --output evaluation/crag/results/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from evaluation.crag.corrective_rag import CRAGConfig, evaluate_retrieval_quality
from evaluation.gold_dataset._validator import DEFAULT_GOLD_QA, REPO_ROOT
from evaluation.retrieval_eval.retrievers import DEFAULT_INDEXES_DIR, load_adapter

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVER_CONFIG = "e5_large_coarse_rerank"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "crag" / "results" / "threshold_sweep.json"
DEFAULT_TAU_HIGHS = (0.6, 0.7, 0.8)
DEFAULT_TAU_LOWS = (0.2, 0.3, 0.4)
DEFAULT_OPERATING_POINT = (0.7, 0.4)  # M2 latency-optimal cell


def _load_gold(gold_path: Path) -> list[dict]:
    """Load every gold pair as a flat probe record (includes unanswerable)."""
    pairs = json.loads(gold_path.read_text() or "[]")
    records: list[dict] = []
    for p in pairs:
        if p.get("annotator") == "adversarial":
            continue
        gold_papers: set[str] = set()
        for claim in p.get("claims", []):
            for span in claim.get("supporting_spans", []):
                pid = span.get("paper_id")
                if isinstance(pid, str):
                    gold_papers.add(pid)
        records.append(
            {
                "id": p["id"],
                "question": p["question"],
                "difficulty": p.get("difficulty", ""),
                "is_unanswerable": p.get("difficulty") == "unanswerable",
                "gold_paper_ids": sorted(gold_papers),
            }
        )
    return records


def _probe(records: list[dict], retriever, k: int) -> list[dict]:
    """Retrieve + score each question once. Threshold-independent."""
    cfg = CRAGConfig(retrieval_k=k)
    probe: list[dict] = []
    for i, rec in enumerate(records, 1):
        t0 = time.perf_counter()
        hits = retriever.search(rec["question"], k=k)
        # Confidence is the normalised best cross-encoder score; thresholds
        # are applied later, so any config with the default scorer works here.
        _, confidence = evaluate_retrieval_quality(rec["question"], hits, cfg)
        elapsed = time.perf_counter() - t0
        gold = set(rec["gold_paper_ids"])
        retrieved_papers = {h.get("paper_id") for h in hits}
        probe.append(
            {
                "id": rec["id"],
                "difficulty": rec["difficulty"],
                "is_unanswerable": rec["is_unanswerable"],
                "confidence": confidence,
                # Retrieval "hit" only meaningful for answerable questions.
                "retrieval_hit": bool(gold & retrieved_papers) if gold else None,
                "latency_s": round(elapsed, 4),
            }
        )
        print(f"  [{i}/{len(records)}] {rec['id']} conf={confidence:.4f}", flush=True)
    return probe


def _bucket(confidence: float, tau_high: float, tau_low: float) -> str:
    if confidence >= tau_high:
        return "correct"
    if confidence >= tau_low:
        return "ambiguous"
    return "incorrect"


def _sweep_cell(probe: list[dict], tau_high: float, tau_low: float, n_unanswerable: int) -> dict:
    buckets = {"correct": [], "ambiguous": [], "incorrect": []}
    for r in probe:
        buckets[_bucket(r["confidence"], tau_high, tau_low)].append(r)

    answerable = [r for r in probe if not r["is_unanswerable"] and r["retrieval_hit"] is not None]
    baseline_correct = (
        sum(1 for r in answerable if r["retrieval_hit"]) / len(answerable) if answerable else 0.0
    )

    def _abstain_stats(abstained: list[dict]) -> dict:
        good_unans = sum(1 for r in abstained if r["is_unanswerable"])
        n = len(abstained)
        precision = good_unans / n if n else 0.0
        recall = good_unans / n_unanswerable if n_unanswerable else 0.0
        answered = [r for r in probe if r not in abstained]
        answered_answerable = [
            r for r in answered if not r["is_unanswerable"] and r["retrieval_hit"] is not None
        ]
        correct_among_answered = (
            sum(1 for r in answered_answerable if r["retrieval_hit"]) / len(answered_answerable)
            if answered_answerable
            else 0.0
        )
        return {
            "n_abstained": n,
            "abstain_precision": round(precision, 4),
            "abstain_recall": round(recall, 4),
            "correct_retrieval_among_answered": round(correct_among_answered, 4),
        }

    strict_abstain = buckets["incorrect"]
    loose_abstain = buckets["incorrect"] + buckets["ambiguous"]

    return {
        "tau_high": tau_high,
        "tau_low": tau_low,
        "branch_distribution": {b: len(v) for b, v in buckets.items()},
        "baseline_correct_retrieval": round(baseline_correct, 4),
        "abstain_strict": _abstain_stats(strict_abstain),
        "abstain_loose": _abstain_stats(loose_abstain),
    }


def _markdown(payload: dict) -> str:
    cfg = payload["config"]
    lines = [
        "# CRAG threshold ablation — M3",
        "",
        f"Retriever: `{cfg['retriever_config']}` · k={cfg['retrieval_k']} · "
        f"{payload['n_questions']} gold questions ({payload['n_unanswerable']} unanswerable).",
        "",
        "Abstain bounds: **strict** = INCORRECT only; **loose** = INCORRECT + AMBIGUOUS.",
        "",
        "| τ_high | τ_low | correct | ambig | incorrect | abst(strict) | prec | rec | "
        "abst(loose) | prec | rec | base hit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in payload["sweep"]:
        bd = cell["branch_distribution"]
        s = cell["abstain_strict"]
        lo = cell["abstain_loose"]
        lines.append(
            f"| {cell['tau_high']} | {cell['tau_low']} | {bd['correct']} | {bd['ambiguous']} | "
            f"{bd['incorrect']} | {s['n_abstained']} | {s['abstain_precision']:.2f} | "
            f"{s['abstain_recall']:.2f} | {lo['n_abstained']} | {lo['abstain_precision']:.2f} | "
            f"{lo['abstain_recall']:.2f} | {cell['baseline_correct_retrieval']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_QA)
    parser.add_argument("--retriever-config", default=DEFAULT_RETRIEVER_CONFIG)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=None,
        help="Reuse a cached probe JSON if present; otherwise write one here after probing.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    print("Loading gold dataset ...")
    records = _load_gold(args.gold)
    print(
        f"  {len(records)} questions "
        f"({sum(1 for r in records if r['is_unanswerable'])} unanswerable)"
    )

    cache = args.probe_cache
    if cache and cache.exists():
        print(f"Reusing probe cache {cache}")
        probe = json.loads(cache.read_text())
    else:
        print(f"Loading retriever: {args.retriever_config}")
        retriever = load_adapter(args.retriever_config, indexes_dir=args.indexes_dir)
        print("Probing (retrieve + score) — once per question ...")
        probe = _probe(records, retriever, args.retrieval_k)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(probe, indent=2))
            print(f"Wrote probe cache {cache}")

    # Derive the recall denominator from the probe itself so cached runs stay
    # self-consistent (probe == full gold in a normal run).
    n_unanswerable = sum(1 for r in probe if r["is_unanswerable"])

    sweep = [
        _sweep_cell(probe, th, tl, n_unanswerable)
        for th in DEFAULT_TAU_HIGHS
        for tl in DEFAULT_TAU_LOWS
        if tl < th
    ]

    payload = {
        "config": {
            "retriever_config": args.retriever_config,
            "retrieval_k": args.retrieval_k,
            "tau_highs": list(DEFAULT_TAU_HIGHS),
            "tau_lows": list(DEFAULT_TAU_LOWS),
            "operating_point": list(DEFAULT_OPERATING_POINT),
        },
        "n_questions": len(probe),
        "n_unanswerable": n_unanswerable,
        "probe": probe,
        "sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    args.output.with_suffix(".md").write_text(_markdown(payload))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")

    op_high, op_low = DEFAULT_OPERATING_POINT
    op = next((c for c in sweep if c["tau_high"] == op_high and c["tau_low"] == op_low), None)
    if op:
        print(f"\nOperating point (τ_high={op_high}, τ_low={op_low}):")
        print(f"  branches: {op['branch_distribution']}")
        print(
            f"  abstain (strict): n={op['abstain_strict']['n_abstained']} "
            f"prec={op['abstain_strict']['abstain_precision']:.2f} "
            f"rec={op['abstain_strict']['abstain_recall']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
