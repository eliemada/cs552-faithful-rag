"""Stratified blinded sampler for the M3 faithfulness human-eval.

Reads claim-level results from ``evaluation/faithfulness/results/02_full_results.json``,
joins against ``evaluation/gold_dataset/gold_qa.json`` to pull the source
``supporting_spans[0].quote`` per question, and writes a blinded CSV that two
raters annotate independently.

Sampling strategy is defined in
``evaluation/faithfulness/HUMAN_EVAL_PLAN.md``::

    NLI=supported, judge=not_supported     → 30
    NLI=not_supported, judge=supported     → 20
    Both supported                          → 20  (control)
    Total                                   → 80

Within each stratum we balance across the three generator models when possible.

The output CSV does NOT include ``nli_label``, ``judge_label``, ``model``,
``judge_reason``, or ``judge_confidence`` — raters must be blinded for κ to
be valid. Those fields land in a sidecar ``ground_truth.json`` so
``compute_human_eval_kappa.py`` can join back after annotation.

Run::

    uv run python -m scripts.sample_human_eval \
        --seed 2026 \
        --out-dir evaluation/faithfulness/human_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from evaluation.gold_dataset._validator import REPO_ROOT

FAITHFULNESS_RESULTS = (
    REPO_ROOT / "evaluation" / "faithfulness" / "results" / "02_full_results.json"
)
GOLD_QA = REPO_ROOT / "evaluation" / "gold_dataset" / "gold_qa.json"


class ClaimRow(NamedTuple):
    """One claim-with-context row, before any stratum assignment."""

    claim_uid: str  # f"{question_id}_{model_short}_{claim_idx}"
    question_id: str
    difficulty: str
    category: str
    model: str
    claim_idx: int
    claim_text: str
    nli_label: str
    judge_label: str
    supporting_quote: str
    paper_id: str


# ---- loaders -------------------------------------------------------------------


def _model_short(model: str) -> str:
    """Compact form for filenames and uids: ``openai/gpt-4o-mini`` -> ``gpt4o``."""
    base = model.rsplit("/", 1)[-1]
    return base.replace("-", "").replace(".", "").replace("_", "")[:8]


def _supporting_quote_for(gold_qa: list[dict], question_id: str) -> tuple[str, str]:
    """Return ``(quote, paper_id)`` for the first claim's first supporting span."""
    for q in gold_qa:
        if q.get("id") != question_id:
            continue
        claims = q.get("claims") or []
        if not claims:
            return "", ""
        spans = claims[0].get("supporting_spans") or []
        if not spans:
            return "", ""
        first = spans[0]
        return first.get("quote", ""), first.get("paper_id", "")
    return "", ""


def _load_rows() -> list[ClaimRow]:
    """Flatten 02_full_results into one row per (question, model, claim_idx)."""
    results = json.loads(FAITHFULNESS_RESULTS.read_text())
    gold_qa = json.loads(GOLD_QA.read_text())
    rows: list[ClaimRow] = []
    for entry in results:
        qid = entry["question_id"]
        quote, paper_id = _supporting_quote_for(gold_qa, qid)
        model_short = _model_short(entry["model"])
        for i, claim in enumerate(entry.get("claims", [])):
            if claim.get("judge_label") == "error":
                continue
            rows.append(
                ClaimRow(
                    claim_uid=f"{qid}_{model_short}_{i:02d}",
                    question_id=qid,
                    difficulty=entry["difficulty"],
                    category=entry["category"],
                    model=entry["model"],
                    claim_idx=i,
                    claim_text=claim["claim"],
                    nli_label=claim["nli_label"],
                    judge_label=claim["judge_label"],
                    supporting_quote=quote,
                    paper_id=paper_id,
                )
            )
    return rows


# ---- stratified sampler --------------------------------------------------------


def _stratum_of(row: ClaimRow) -> str | None:
    """Map (nli, judge) → one of the strata in HUMAN_EVAL_PLAN.md."""
    pair = (row.nli_label, row.judge_label)
    if pair == ("supported", "not_supported"):
        return "nli_sup_judge_not"
    if pair == ("not_supported", "supported"):
        return "nli_not_judge_sup"
    if pair == ("supported", "supported"):
        return "both_supported"
    return None  # both_not_supported is empty by construction; drop


_STRATUM_TARGETS = {
    "nli_sup_judge_not": 30,
    "nli_not_judge_sup": 20,
    "both_supported": 20,
}


def _balanced_pick(
    pool: list[ClaimRow], target: int, *, by: str, rng: random.Random
) -> list[ClaimRow]:
    """Pick ``target`` rows from ``pool``, spreading across values of ``pool[i][by]``."""
    buckets: dict[str, list[ClaimRow]] = defaultdict(list)
    for r in pool:
        buckets[getattr(r, by)].append(r)
    for k in buckets:
        rng.shuffle(buckets[k])

    chosen: list[ClaimRow] = []
    while len(chosen) < target and any(buckets.values()):
        for k in list(buckets.keys()):
            if buckets[k]:
                chosen.append(buckets[k].pop())
                if len(chosen) >= target:
                    break
    return chosen


def stratified_sample(rows: list[ClaimRow], *, seed: int) -> list[ClaimRow]:
    """Apply stratum targets with model-balancing inside each stratum."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[ClaimRow]] = defaultdict(list)
    for r in rows:
        s = _stratum_of(r)
        if s is not None:
            by_stratum[s].append(r)

    chosen: list[ClaimRow] = []
    for stratum, target in _STRATUM_TARGETS.items():
        pool = by_stratum.get(stratum, [])
        actual_target = min(target, len(pool))
        chosen.extend(_balanced_pick(pool, actual_target, by="model", rng=rng))

    rng.shuffle(chosen)  # randomise presentation order so raters can't infer strata
    return chosen


# ---- writers --------------------------------------------------------------------


_BLINDED_COLUMNS = (
    "claim_uid",
    "question_id",
    "difficulty",
    "category",
    "claim_text",
    "supporting_quote",
    "paper_id",
    "human_label",  # rater fills: "supported" | "not_supported"
    "needs_context",  # rater optional: "y" if quote alone insufficient
    "notes",
)


def write_blinded_csv(rows: list[ClaimRow], path: Path) -> None:
    """Write the rater-facing CSV with hidden columns omitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_BLINDED_COLUMNS)
        for r in rows:
            w.writerow(
                [
                    r.claim_uid,
                    r.question_id,
                    r.difficulty,
                    r.category,
                    r.claim_text,
                    r.supporting_quote,
                    r.paper_id,
                    "",  # human_label
                    "",  # needs_context
                    "",  # notes
                ]
            )


def write_ground_truth(rows: list[ClaimRow], path: Path) -> None:
    """Write the unblinded sidecar so the κ script can join back later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        r.claim_uid: {
            "model": r.model,
            "nli_label": r.nli_label,
            "judge_label": r.judge_label,
            "question_id": r.question_id,
            "claim_idx": r.claim_idx,
            "stratum": _stratum_of(r),
        }
        for r in rows
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---- CLI ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "evaluation" / "faithfulness" / "human_eval",
    )
    args = parser.parse_args(argv)

    rows = _load_rows()
    sample = stratified_sample(rows, seed=args.seed)

    if not sample:
        print("Empty sample — check that 02_full_results.json contains claims.")
        return 1

    sample_path = args.out_dir / "sample.csv"
    gt_path = args.out_dir / "ground_truth.json"
    andrea_path = args.out_dir / "rater_andrea.csv"
    yusif_path = args.out_dir / "rater_yusif.csv"

    write_blinded_csv(sample, sample_path)
    write_blinded_csv(sample, andrea_path)
    write_blinded_csv(sample, yusif_path)
    write_ground_truth(sample, gt_path)

    by_stratum: dict[str, int] = defaultdict(int)
    by_model: dict[str, int] = defaultdict(int)
    for r in sample:
        s = _stratum_of(r)
        if s is not None:
            by_stratum[s] += 1
        by_model[r.model] += 1

    print(f"wrote {sample_path.relative_to(REPO_ROOT)} ({len(sample)} rows)")
    print(f"wrote {andrea_path.relative_to(REPO_ROOT)} (rater copy)")
    print(f"wrote {yusif_path.relative_to(REPO_ROOT)} (rater copy)")
    print(f"wrote {gt_path.relative_to(REPO_ROOT)} (ground truth — DO NOT show raters)")
    print()
    print("Stratum distribution:")
    for s, n in by_stratum.items():
        target = _STRATUM_TARGETS.get(s, "?")
        print(f"  {s}: {n}  (target {target})")
    print()
    print("Model distribution:")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
