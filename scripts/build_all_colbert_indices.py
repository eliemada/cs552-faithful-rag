"""Build both ColBERT PLAID indices (coarse + fine) in one pass.

Fires ``scripts.build_colbert_index`` twice. Idempotent: skips a
granularity whose ``colbert_<chunk_type>/`` index folder already exists.

Run on a GPU node — the per-token encoding is ~5–10× heavier than the
dense passage encoding the BGE-M3 / E5-large indices needed.

Examples::

    uv run python -m scripts.build_all_colbert_indices --device cuda --batch-size 64
    uv run python -m scripts.build_all_colbert_indices --device mps  # local fallback
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from evaluation.gold_dataset._validator import REPO_ROOT
from rag_pipeline.rag.colbert_retriever import DEFAULT_MODEL_ID

DEFAULT_INDEXES_DIR = REPO_ROOT / "data" / "s3_archive" / "indexes"

logger = logging.getLogger(__name__)

CHUNK_TYPES: tuple[str, ...] = ("coarse", "fine")


def already_built(indexes_dir: Path, chunk_type: str) -> bool:
    """The build script writes ``colbert_<chunk_type>/`` + a metadata sidecar."""
    folder = indexes_dir / f"colbert_{chunk_type}"
    meta = indexes_dir / f"colbert_{chunk_type}_metadata.json"
    return folder.is_dir() and meta.is_file()


def run_one(
    chunk_type: str,
    *,
    model: str,
    device: str | None,
    batch_size: int,
    nbits: int,
) -> int:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.build_colbert_index",
        "--chunk-type",
        chunk_type,
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--nbits",
        str(nbits),
    ]
    if device:
        cmd.extend(["--device", device])
    logger.info("→ %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    started = time.perf_counter()
    failures: list[str] = []
    for chunk_type in CHUNK_TYPES:
        if not args.force and already_built(args.indexes_dir, chunk_type):
            logger.info("✓ already built: colbert × %s", chunk_type)
            continue
        rc = run_one(
            chunk_type,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            nbits=args.nbits,
        )
        if rc != 0:
            failures.append(f"colbert/{chunk_type}")
            logger.error("✗ failed: colbert × %s (rc=%d)", chunk_type, rc)
        else:
            logger.info("✓ built: colbert × %s", chunk_type)

    elapsed = time.perf_counter() - started
    logger.info("Total wall: %.1f min", elapsed / 60)
    if failures:
        logger.error("Failures: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
