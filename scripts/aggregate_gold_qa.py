"""Concatenate ``contributions/<name>.json`` into ``gold_qa.json``.

Two modes:

* default (write) — re-derive ``gold_qa.json`` from the contributions.
* ``--check`` — exit non-zero if the committed ``gold_qa.json`` is stale.
  CI uses this to enforce that whoever edits a contribution also commits
  the refreshed aggregate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.gold_dataset._validator import (
    DEFAULT_CONTRIB,
    DEFAULT_GOLD_QA,
    aggregate_contributions,
    check_aggregate_matches,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contrib-dir",
        type=Path,
        default=DEFAULT_CONTRIB,
    )
    parser.add_argument(
        "--gold-qa",
        type=Path,
        default=DEFAULT_GOLD_QA,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if gold_qa.json drifts from contributions/.",
    )
    args = parser.parse_args(argv)

    if args.check:
        if check_aggregate_matches(contrib_dir=args.contrib_dir, committed=args.gold_qa):
            return 0
        sys.stderr.write(
            f"::error file={args.gold_qa}::"
            f"gold_qa.json is out of sync with {args.contrib_dir.name}/. "
            f"Run: uv run python scripts/aggregate_gold_qa.py\n"
        )
        return 1

    aggregated = aggregate_contributions(args.contrib_dir)
    args.gold_qa.write_text(json.dumps(aggregated, indent=2) + "\n")
    sys.stdout.write(f"Wrote {len(aggregated)} pairs → {args.gold_qa}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
