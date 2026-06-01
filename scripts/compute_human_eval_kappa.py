"""Compute Cohen's kappa for the M3 faithfulness human-eval.

Joins two filled rater CSVs (``rater_andrea.csv``, ``rater_yusif.csv``)
against the blinded ground-truth sidecar produced by
``sample_human_eval.py``, then reports:

1. κ(rater A, rater B) — inter-rater agreement (sanity check).
2. κ(majority_human, NLI label) — does NLI track what humans see?
3. κ(majority_human, LLM judge label) — does the judge track humans?
4. 2×2 confusion matrices for the three pairs above.
5. Per-stratum breakdown so we can talk about the
   NLI-overcalls-supported vs. judge-overcalls-not_supported failure modes.

Output: a markdown report at
``evaluation/faithfulness/human_eval/results.md`` plus a
``results.json`` for downstream notebooks.

Run after both raters have filled in their CSVs::

    uv run python -m scripts.compute_human_eval_kappa
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

from evaluation.gold_dataset._validator import REPO_ROOT

HUMAN_EVAL_DIR = REPO_ROOT / "evaluation" / "faithfulness" / "human_eval"

LABELS = ("supported", "not_supported")


class JoinedRow(NamedTuple):
    """One claim with all four labels glued together."""

    claim_uid: str
    stratum: str
    model: str
    rater_a: str
    rater_b: str
    nli: str
    judge: str


# ---- loaders --------------------------------------------------------------------


def _load_rater_csv(path: Path) -> dict[str, str]:
    """Return ``{claim_uid: human_label}`` for rows where ``human_label`` is set."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("human_label") or "").strip().lower()
            if label in LABELS:
                out[row["claim_uid"]] = label
    return out


def _load_ground_truth(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())


def join_rows(
    rater_a: dict[str, str],
    rater_b: dict[str, str],
    ground_truth: dict[str, dict],
) -> list[JoinedRow]:
    """Inner-join by ``claim_uid``; drop claims either rater hasn't labelled."""
    shared = set(rater_a) & set(rater_b) & set(ground_truth)
    rows = []
    for uid in sorted(shared):
        gt = ground_truth[uid]
        rows.append(
            JoinedRow(
                claim_uid=uid,
                stratum=gt.get("stratum", "unknown"),
                model=gt.get("model", "unknown"),
                rater_a=rater_a[uid],
                rater_b=rater_b[uid],
                nli=gt["nli_label"],
                judge=gt["judge_label"],
            )
        )
    return rows


# ---- kappa ----------------------------------------------------------------------


def cohen_kappa(pairs: Iterable[tuple[str, str]]) -> tuple[float, dict]:
    """Standard Cohen's κ for binary labels.

    Returns ``(kappa, info)`` where info contains the 2×2 confusion counts.
    """
    pair_list = list(pairs)
    n = len(pair_list)
    if n == 0:
        return float("nan"), {"n": 0}

    matrix = {(a, b): 0 for a in LABELS for b in LABELS}
    a_marg: Counter[str] = Counter()
    b_marg: Counter[str] = Counter()
    for a, b in pair_list:
        if a in LABELS and b in LABELS:
            matrix[(a, b)] += 1
            a_marg[a] += 1
            b_marg[b] += 1

    valid = sum(matrix.values())
    if valid == 0:
        return float("nan"), {"n": 0, "matrix": matrix}

    po = sum(matrix[(L, L)] for L in LABELS) / valid
    pe = sum((a_marg[L] / valid) * (b_marg[L] / valid) for L in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    return kappa, {
        "n": valid,
        "po": po,
        "pe": pe,
        "matrix": {f"{a}|{b}": matrix[(a, b)] for a in LABELS for b in LABELS},
    }


def majority(a: str, b: str) -> str | None:
    """Pick the agreed label when raters agree; ``None`` when they don't."""
    return a if a == b and a in LABELS else None


# ---- report ---------------------------------------------------------------------


def _fmt_kappa(k: float) -> str:
    return f"{k:.3f}" if k == k else "nan"  # nan != nan


def _confusion_md(label_a: str, label_b: str, info: dict) -> str:
    m = info.get("matrix", {})
    return (
        f"|                 | {label_b}=supported | {label_b}=not_supported |\n"
        f"|---|---|---|\n"
        f"| {label_a}=supported    | {m.get('supported|supported', 0)} | {m.get('supported|not_supported', 0)} |\n"
        f"| {label_a}=not_supported| {m.get('not_supported|supported', 0)} | {m.get('not_supported|not_supported', 0)} |\n"
    )


def render_report(rows: list[JoinedRow]) -> tuple[str, dict]:
    """Produce the markdown report + a machine-readable summary."""
    # Inter-rater
    k_ab, info_ab = cohen_kappa((r.rater_a, r.rater_b) for r in rows)

    # Build majority-human label; drop disagreements for NLI/judge comparison
    majority_rows = [(r, majority(r.rater_a, r.rater_b)) for r in rows]
    paired_human_nli = [(maj, r.nli) for r, maj in majority_rows if maj is not None]
    paired_human_judge = [(maj, r.judge) for r, maj in majority_rows if maj is not None]
    k_hn, info_hn = cohen_kappa(paired_human_nli)
    k_hj, info_hj = cohen_kappa(paired_human_judge)

    # Per-stratum agreement: fraction of cases where majority_human matches NLI / judge
    stratum_breakdown: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "human_eq_nli": 0, "human_eq_judge": 0}
    )
    for r, maj in majority_rows:
        if maj is None:
            continue
        bucket = stratum_breakdown[r.stratum]
        bucket["n"] += 1
        if maj == r.nli:
            bucket["human_eq_nli"] += 1
        if maj == r.judge:
            bucket["human_eq_judge"] += 1

    stratum_lines = []
    for s, b in sorted(stratum_breakdown.items()):
        n = b["n"]
        if n == 0:
            continue
        stratum_lines.append(
            f"| {s} | {n} | {b['human_eq_nli']}/{n} ({b['human_eq_nli'] / n:.0%}) "
            f"| {b['human_eq_judge']}/{n} ({b['human_eq_judge'] / n:.0%}) |"
        )

    md = f"""# Faithfulness human-eval — results

## Sample

- Joined rows (both raters labelled, ground-truth available): **{len(rows)}**
- Inter-rater (A, B) agreement: po = {info_ab.get("po", float("nan")):.3f}

## Headline κ

| Pair                       | κ     | n  |
|---|---|---|
| rater_a vs rater_b         | {_fmt_kappa(k_ab)} | {info_ab["n"]} |
| majority_human vs NLI      | {_fmt_kappa(k_hn)} | {info_hn["n"]} |
| majority_human vs judge    | {_fmt_kappa(k_hj)} | {info_hj["n"]} |

## Per-stratum breakdown

| stratum | n (raters agree) | human = NLI | human = judge |
|---|---|---|---|
{chr(10).join(stratum_lines) if stratum_lines else "| — | 0 | — | — |"}

## Confusion matrices

### rater A vs rater B
{_confusion_md("rater_a", "rater_b", info_ab)}

### majority_human vs NLI
{_confusion_md("human", "NLI", info_hn)}

### majority_human vs judge
{_confusion_md("human", "judge", info_hj)}

## Notes for the report

- The headline number to cite in §Results/Faithfulness is **κ(human, judge) = {_fmt_kappa(k_hj)}**
  with **κ(human, NLI) = {_fmt_kappa(k_hn)}** as the contrast.
- Inter-rater κ = {_fmt_kappa(k_ab)} bounds how seriously the two κ values
  above can be read. If it's below ~0.6, the protocol needs to be revisited
  before drawing strong conclusions.
"""

    summary = {
        "n_joined": len(rows),
        "kappa_rater_a_vs_b": None if k_ab != k_ab else k_ab,
        "kappa_human_vs_nli": None if k_hn != k_hn else k_hn,
        "kappa_human_vs_judge": None if k_hj != k_hj else k_hj,
        "confusion_rater_ab": info_ab.get("matrix", {}),
        "confusion_human_nli": info_hn.get("matrix", {}),
        "confusion_human_judge": info_hj.get("matrix", {}),
        "stratum_breakdown": dict(stratum_breakdown),
    }
    return md, summary


# ---- CLI ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--rater-a", type=Path, default=HUMAN_EVAL_DIR / "rater_andrea.csv")
    parser.add_argument("--rater-b", type=Path, default=HUMAN_EVAL_DIR / "rater_yusif.csv")
    parser.add_argument("--ground-truth", type=Path, default=HUMAN_EVAL_DIR / "ground_truth.json")
    parser.add_argument("--out-dir", type=Path, default=HUMAN_EVAL_DIR)
    args = parser.parse_args(argv)

    rater_a = _load_rater_csv(args.rater_a)
    rater_b = _load_rater_csv(args.rater_b)
    ground_truth = _load_ground_truth(args.ground_truth)

    rows = join_rows(rater_a, rater_b, ground_truth)
    if not rows:
        print(
            "No joinable rows — make sure both raters have filled in 'human_label' "
            "with 'supported' or 'not_supported' for the same claim_uids.",
            flush=True,
        )
        return 1

    md, summary = render_report(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.md").write_text(md)
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    def _pretty(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"wrote {_pretty(args.out_dir / 'results.md')}")
    print(f"wrote {_pretty(args.out_dir / 'results.json')}")
    print(f"  joined rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
