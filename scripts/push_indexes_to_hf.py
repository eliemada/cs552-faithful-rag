"""Upload alternative-embedder FAISS indices to the team's HF dataset.

Pushes ``bge_m3_*`` and ``e5_large_*`` files from
``data/s3_archive/indexes/`` to ``citeright/corpus`` (dataset repo) under
``indexes/`` so teammates can ``snapshot_download`` them instead of
rebuilding from scratch.

Requires ``HF_TOKEN`` (or ``HUGGINGFACEHUB_API_TOKEN``) in the environment.

Run inside a Run:AI pod that mounts the scratch PVC, or anywhere the four
``<embedder>_<chunk_type>.faiss`` files exist locally::

    HF_TOKEN=hf_... uv run python -m scripts.push_indexes_to_hf
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Final

from evaluation.gold_dataset._validator import REPO_ROOT

DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--repo-id", default="citeright/corpus")
    parser.add_argument("--path-in-repo", default="indexes")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["bge_m3_*", "e5_large_*"],
        help="Filename glob patterns (faiss + metadata both match by default).",
    )
    parser.add_argument(
        "--commit-message",
        default="add BGE-M3 and E5-large FAISS indices (coarse + fine)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) not set. Generate a write-scoped "
            "token at https://huggingface.co/settings/tokens and export it."
        )

    matched = sorted(p for pat in args.patterns for p in args.indexes_dir.glob(pat) if p.is_file())
    if not matched:
        logger.error("No files matched %s in %s", args.patterns, args.indexes_dir)
        return 1

    logger.info("Uploading %d files to %s/%s:", len(matched), args.repo_id, args.path_in_repo)
    for p in matched:
        logger.info("  %s  (%.1f MB)", p.name, p.stat().st_size / 1e6)

    from huggingface_hub import HfApi  # heavy import, defer

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(args.indexes_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        allow_patterns=args.patterns,
        commit_message=args.commit_message,
    )
    logger.info(
        "Done. Pull locally with snapshot_download(allow_patterns=['indexes/bge_m3_*', 'indexes/e5_large_*'])."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
