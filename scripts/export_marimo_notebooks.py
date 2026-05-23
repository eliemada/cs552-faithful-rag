"""Export every marimo notebook to its submission-format ``.ipynb`` twin.

Each team member's marimo notebook at
``notebooks/marimo/<firstname>_<task>.py`` re-exports to
``notebooks/<firstname>_<lastname>_<sciper>.ipynb`` — the path the course
submission script expects. Outputs are baked in
(``--include-outputs``) so reviewers see the rendered tables without
re-running anything.

The mapping is explicit (not auto-derived from filenames) because the
submission-required SCIPER number isn't in the marimo path. To register
a new member, add a row to :data:`SUBMISSION_TARGETS`.

This script is normally invoked by the ``export-marimo-notebooks`` prek
hook in ``.pre-commit-config.yaml``; it's also runnable directly:

    uv run python -m scripts.export_marimo_notebooks
    uv run python -m scripts.export_marimo_notebooks --check    # CI-style
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation.gold_dataset._validator import REPO_ROOT

MARIMO_DIR = REPO_ROOT / "notebooks" / "marimo"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

# (marimo source filename, submission target filename)
# Add a row when a teammate's marimo notebook lands.
SUBMISSION_TARGETS: list[tuple[str, str]] = [
    ("elie_retrieval_ablation.py", "elie_bruno_355932.ipynb"),
]


def export_one(marimo_path: Path, ipynb_path: Path) -> int:
    """Run ``marimo export ipynb`` and return its exit code."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            str(marimo_path),
            "--sort",
            "top-down",
            "--include-outputs",
            "-f",
            "-o",
            str(ipynb_path),
        ],
        cwd=REPO_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-export to a tmp file and assert byte-equal — for CI / pre-commit.",
    )
    args = parser.parse_args(argv)

    failures: list[str] = []
    for src_name, dst_name in SUBMISSION_TARGETS:
        marimo_path = MARIMO_DIR / src_name
        ipynb_path = NOTEBOOK_DIR / dst_name

        if not marimo_path.is_file():
            failures.append(f"missing marimo source: {marimo_path}")
            continue

        if args.check:
            # Render to a sibling tmp path and diff against the tracked file.
            tmp_path = ipynb_path.with_suffix(".ipynb.check")
            rc = export_one(marimo_path, tmp_path)
            if rc != 0:
                failures.append(f"export failed for {marimo_path.name}")
                tmp_path.unlink(missing_ok=True)
                continue
            if not ipynb_path.is_file():
                failures.append(f"missing committed ipynb: {ipynb_path}")
                tmp_path.unlink(missing_ok=True)
                continue
            on_disk = ipynb_path.read_bytes()
            fresh = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            if on_disk != fresh:
                failures.append(
                    f"{ipynb_path.name} is out of sync with {marimo_path.name}. "
                    "Run `uv run python -m scripts.export_marimo_notebooks` "
                    "and commit the regenerated .ipynb."
                )
            else:
                print(f"  ✓ {ipynb_path.name}")
        else:
            rc = export_one(marimo_path, ipynb_path)
            if rc != 0:
                failures.append(f"export failed for {marimo_path.name}")
            else:
                print(f"  → {ipynb_path.name}")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
