"""Refresh ``evaluation/gold_dataset/paper_ids.txt`` from the live corpus.

Keeps CI offline: the validator reads this snapshot rather than touching the
3 GB HF dataset. Re-run whenever the corpus contents change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.common.data_loader import list_paper_ids
from evaluation.gold_dataset._validator import DEFAULT_ALLOWLIST


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_ALLOWLIST)
    args = parser.parse_args(argv)

    ids = list_paper_ids()
    args.out.write_text("\n".join(ids) + "\n")
    print(f"Wrote {len(ids)} paper ids → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
