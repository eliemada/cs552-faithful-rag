"""CLI wrapper around ``evaluation.gold_dataset._validator``.

Default invocation runs the offline (fast) validator against every
``contributions/<name>.json`` plus the aggregated ``gold_qa.json``. CI
relies on this exit code; local users typically pass ``--strict`` to also
re-check span/quote correctness against the live corpus.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from evaluation.common.paths import LOCAL_ARCHIVE, SCRATCH_CACHE, is_rcp
from evaluation.gold_dataset._validator import (
    DEFAULT_ALLOWLIST,
    DEFAULT_CONTRIB,
    DEFAULT_GOLD_QA,
    ValidationError,
    aggregate_contributions,
    validate_fast,
    validate_strict,
)

logger = logging.getLogger("validate_gold_qa")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also verify spans against the live corpus (requires data access).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print human-readable summary on stdout (used by GitHub step summary).",
    )
    parser.add_argument(
        "--contrib-dir",
        type=Path,
        default=DEFAULT_CONTRIB,
        help="Directory containing per-member <name>.json files.",
    )
    parser.add_argument(
        "--gold-qa",
        type=Path,
        default=DEFAULT_GOLD_QA,
        help="Aggregated gold_qa.json artifact.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Path to paper_ids.txt allowlist.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Override corpus root for --strict mode (default: auto-detect).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    pairs = aggregate_contributions(args.contrib_dir)
    errors = validate_fast(pairs, allowlist=args.allowlist)

    if args.strict:
        corpus_root = args.corpus_root or _resolve_corpus_root()
        if corpus_root is None:
            logger.error("--strict requires a corpus root; pass --corpus-root or run on RCP.")
            return 2
        errors = list(errors) + validate_strict(
            pairs, allowlist=args.allowlist, corpus_root=corpus_root
        )

    if args.report:
        _print_report(pairs, errors)

    for err in errors:
        sys.stdout.write(f"{err.as_github_annotation(args.gold_qa)}\n")

    return 1 if errors else 0


def _resolve_corpus_root() -> Path | None:
    if is_rcp() and SCRATCH_CACHE.is_dir():
        return SCRATCH_CACHE
    if LOCAL_ARCHIVE.is_dir():
        return LOCAL_ARCHIVE
    return None


def _print_report(pairs: list[dict], errors: list[ValidationError]) -> None:
    """Emit a markdown summary suitable for `$GITHUB_STEP_SUMMARY`."""
    print("## Gold dataset validation\n")
    print(f"- Pairs: **{len(pairs)}**")
    by_member = Counter(p.get("annotator", "?") for p in pairs)
    print(f"- By annotator: {dict(by_member)}")
    by_diff = Counter(p.get("difficulty", "?") for p in pairs)
    print(f"- By difficulty: {dict(by_diff)}")
    by_cat = Counter(p.get("category", "?") for p in pairs)
    print(f"- By category: {dict(by_cat)}")
    iaa_pairs = sum(1 for p in pairs if p.get("iaa_subset"))
    print(f"- IAA subset: {iaa_pairs} pairs")
    n_claims = sum(len(p.get("claims") or []) for p in pairs)
    print(f"- Claims: {n_claims}")
    print(f"- Errors: **{len(errors)}**")


if __name__ == "__main__":
    raise SystemExit(main())
