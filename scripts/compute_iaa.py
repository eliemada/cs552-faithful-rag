"""Compute Cohen's κ on the IAA subset of the gold dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.gold_dataset._validator import (
    DEFAULT_GOLD_QA,
    IAAError,
    cohens_kappa,
    extract_iaa_labels,
)

KAPPA_TARGET = 0.6
LABEL_ORDER = ("supports", "contradicts", "unrelated")


def _label_distribution(labels: list[str]) -> str:
    counts = Counter(labels)
    return ", ".join(f"{l}={counts.get(l, 0)}" for l in LABEL_ORDER)


def _degeneracy_warning(annotator: list[str], reviewer: list[str]) -> str | None:
    """Flag when κ is mathematically defensible but methodologically empty.

    κ becomes degenerate when the chance-agreement term p_e ≈ 1.0 — i.e. one
    label dominates the marginals. The formula returns 1.0 by convention in
    the implementation, but it carries no signal until the label space has
    real diversity.
    """
    union = set(annotator) | set(reviewer)
    if len(union) <= 1:
        return (
            "WARNING: single-class IAA subset — every label is the same value. "
            "κ is conventional (=1.0), not informative. Add adversarial controls "
            "(contributions/adversarial.json) or unanswerable pairs to populate "
            "the contradicts/unrelated regions."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-qa", type=Path, default=DEFAULT_GOLD_QA)
    parser.add_argument(
        "--target",
        type=float,
        default=KAPPA_TARGET,
        help=f"Minimum acceptable κ (default {KAPPA_TARGET}).",
    )
    parser.add_argument(
        "--strict-target",
        action="store_true",
        help="Exit non-zero if κ < target.",
    )
    args = parser.parse_args(argv)

    pairs = json.loads(args.gold_qa.read_text() or "[]")
    annotator, reviewer = extract_iaa_labels(pairs)

    if not annotator:
        sys.stdout.write(
            "No reviewed claims in IAA subset yet — κ undefined. "
            "Mark pairs with iaa_subset=true and add reviewer_label.\n"
        )
        return 0

    try:
        kappa = cohens_kappa(annotator, reviewer)
    except IAAError as exc:
        sys.stderr.write(f"IAA error: {exc}\n")
        return 2

    sys.stdout.write(
        f"Cohen's κ = {kappa:.3f}  (n={len(annotator)} claims, target ≥ {args.target})\n"
    )
    sys.stdout.write(f"  annotator labels: {_label_distribution(annotator)}\n")
    sys.stdout.write(f"  reviewer  labels: {_label_distribution(reviewer)}\n")
    warning = _degeneracy_warning(annotator, reviewer)
    if warning:
        sys.stdout.write(f"  {warning}\n")

    if args.strict_target and kappa < args.target:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
