"""Compute Cohen's κ on the IAA subset of the gold dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.gold_dataset._validator import (
    DEFAULT_GOLD_QA,
    IAAError,
    cohens_kappa,
    extract_iaa_labels,
)

KAPPA_TARGET = 0.6


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
    if args.strict_target and kappa < args.target:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
