"""Path resolution for CiteRight evaluation artifacts.

Resolution order (first match wins):

1. ``CITERIGHT_DATA_DIR`` env var — explicit override (used for tests).
2. ``/scratch/citeright_artifacts`` if present — RCP cluster cache.
3. ``${REPO}/data/s3_archive`` — local dev archive (mirror of the dead S3 bucket).
4. HuggingFace download to a sensible cache directory.

Layout (all backends share this shape, mirrored from the original S3 bucket)::

    chunks/<paper_id>_<coarse|fine>.json
    indexes/<coarse|fine>.faiss
    indexes/<coarse|fine>_metadata.json
    processed/<paper_id>/document.md
    processed/<paper_id>/metadata.json
    raw_metadata/<paper_id>.json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
LOCAL_ARCHIVE: Final[Path] = REPO_ROOT / "data" / "s3_archive"
SCRATCH_CACHE: Final[Path] = Path("/scratch/citeright_artifacts")
HF_FALLBACK_CACHE: Final[Path] = REPO_ROOT / "data" / "hf_artifacts"

HF_CORPUS_REPO: Final[str] = os.environ.get("CITERIGHT_HF_REPO", "citeright/corpus")
GOLD_QA_PATH: Final[Path] = REPO_ROOT / "evaluation" / "gold_dataset" / "gold_qa.json"


def is_rcp() -> bool:
    """Heuristic: are we running inside the EPFL RCP pod?

    The course image always mounts ``/scratch`` as the group PVC.
    """
    return Path("/scratch").is_dir() and os.access("/scratch", os.W_OK)
