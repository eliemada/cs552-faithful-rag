"""Build every alternative-embedder FAISS index in one pass.

Runs ``scripts.build_hf_index`` four times: ``{bge-m3, e5-large} ×
{coarse, fine}``. Designed to be fired once on a GPU cluster and left
overnight. On CUDA it finishes in ~30-60 minutes; on Apple MPS it takes
~7 hours, so prefer the cluster.

Skips any (model, chunk_type) whose output ``<embedder>_<chunk_type>.faiss``
already exists, so re-runs are cheap.

Examples::

    # cluster, NVIDIA GPU
    uv run python -m scripts.build_all_hf_indices --device cuda

    # local fallback
    uv run python -m scripts.build_all_hf_indices --device mps
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from evaluation.gold_dataset._validator import REPO_ROOT

DEFAULT_INDEXES_DIR = REPO_ROOT / "data" / "s3_archive" / "indexes"

logger = logging.getLogger(__name__)

PLAN: tuple[tuple[str, str], ...] = (
    ("bge-m3", "coarse"),
    ("bge-m3", "fine"),
    ("e5-large", "coarse"),
    ("e5-large", "fine"),
)


def already_built(indexes_dir: Path, model: str, chunk_type: str) -> bool:
    """The build_hf_index script writes ``<embedder>_<chunk_type>.faiss``."""
    short = model.replace("-", "_")
    faiss_path = indexes_dir / f"{short}_{chunk_type}.faiss"
    meta_path = indexes_dir / f"{short}_{chunk_type}_metadata.json"
    return faiss_path.is_file() and meta_path.is_file()


def run_one(model: str, chunk_type: str, *, device: str | None, batch_size: int) -> int:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.build_hf_index",
        "--model",
        model,
        "--chunk-type",
        chunk_type,
        "--batch-size",
        str(batch_size),
    ]
    if device:
        cmd.extend(["--device", device])
    logger.info("→ %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detect by default)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    started = time.perf_counter()
    failures: list[str] = []

    for model, chunk_type in PLAN:
        if not args.force and already_built(args.indexes_dir, model, chunk_type):
            logger.info("✓ already built: %s × %s", model, chunk_type)
            continue
        rc = run_one(model, chunk_type, device=args.device, batch_size=args.batch_size)
        if rc != 0:
            failures.append(f"{model}/{chunk_type}")
            logger.error("✗ failed: %s × %s (rc=%d)", model, chunk_type, rc)
        else:
            logger.info("✓ built: %s × %s", model, chunk_type)

    elapsed = time.perf_counter() - started
    logger.info("Total wall: %.1f min", elapsed / 60)
    if failures:
        logger.error("Failures: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
