"""Loaders for CiteRight evaluation artifacts (chunks / FAISS / markdown / Q&A).

Designed so the same notebook code works in three environments:

* Grader on RCP (no S3, no AWS) — downloads from HuggingFace into ``/scratch``.
* Team on RCP — uses cached download.
* Team on laptop — uses ``data/s3_archive/`` mirror of the original bucket.

The user-facing functions never deal with paths or HF; they just return data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final, Iterator, Literal

from evaluation.common.paths import (
    GOLD_QA_PATH,
    HF_CORPUS_REPO,
    HF_FALLBACK_CACHE,
    LOCAL_ARCHIVE,
    SCRATCH_CACHE,
    is_rcp,
)

logger = logging.getLogger(__name__)

Granularity = Literal["coarse", "fine"]
_GRANULARITIES: Final[tuple[str, ...]] = ("coarse", "fine")


def artifact_root() -> Path:
    """Resolve where corpus artifacts live, downloading from HF if needed.

    Raises
    ------
    RuntimeError
        If no usable archive is reachable (e.g. offline laptop with no mirror).
    """
    if (override := _explicit_override()) is not None:
        return override

    if is_rcp():
        if _populated(SCRATCH_CACHE):
            return SCRATCH_CACHE
        return _hf_download(SCRATCH_CACHE)

    if _populated(LOCAL_ARCHIVE):
        return LOCAL_ARCHIVE

    if _populated(HF_FALLBACK_CACHE):
        return HF_FALLBACK_CACHE

    return _hf_download(HF_FALLBACK_CACHE)


# ---------- chunks & FAISS ----------


def load_chunk_metadata(granularity: Granularity) -> dict[int, dict]:
    """Return the FAISS-row → chunk-info mapping used to materialise hits.

    The JSON on disk uses string keys (``"0"``, ``"1"``, ...); we coerce to ``int``
    so callers can index directly with the row id returned by FAISS.
    """
    path = artifact_root() / "indexes" / f"{granularity}_metadata.json"
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def load_faiss_index(granularity: Granularity):  # type: ignore[no-untyped-def]
    """Load the pre-built FAISS index. Returns a ``faiss.IndexFlatIP``."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover — installed everywhere we run
        raise RuntimeError("faiss is required: pip install faiss-cpu") from exc
    path = artifact_root() / "indexes" / f"{granularity}.faiss"
    return faiss.read_index(str(path))


def load_paper_chunks(paper_id: str, granularity: Granularity) -> list[dict]:
    """Return the flat list of chunks for a single paper."""
    path = artifact_root() / "chunks" / f"{paper_id}_{granularity}.json"
    return json.loads(path.read_text())["chunks"]


def iter_all_chunks(granularity: Granularity) -> Iterator[dict]:
    """Yield every chunk in the corpus at the given granularity."""
    chunks_dir = artifact_root() / "chunks"
    for path in sorted(chunks_dir.glob(f"*_{granularity}.json")):
        yield from json.loads(path.read_text())["chunks"]


def list_paper_ids() -> list[str]:
    """Return the sorted list of paper IDs present in the corpus."""
    chunks_dir = artifact_root() / "chunks"
    seen: set[str] = set()
    for path in chunks_dir.glob("*.json"):
        # Filenames look like ``<paper_id>_<granularity>.json``.
        stem = path.stem
        for suffix in _GRANULARITIES:
            if stem.endswith(f"_{suffix}"):
                seen.add(stem.removesuffix(f"_{suffix}"))
                break
    return sorted(seen)


# ---------- documents & metadata ----------


def load_paper_markdown(paper_id: str) -> str:
    """Return the Dolphin-parsed markdown for one paper."""
    return (artifact_root() / "processed" / paper_id / "document.md").read_text()


def load_paper_openalex(paper_id: str) -> dict:
    """Return the OpenAlex metadata record for one paper (DOI, year, OA status, ...)."""
    return json.loads((artifact_root() / "raw_metadata" / f"{paper_id}.json").read_text())


# ---------- gold dataset (lives in repo, not in HF artifacts) ----------


def load_gold_qa() -> list[dict]:
    """Return the gold Q&A pairs annotated by the team."""
    if not GOLD_QA_PATH.exists():
        logger.warning("Gold Q&A file missing at %s", GOLD_QA_PATH)
        return []
    return json.loads(GOLD_QA_PATH.read_text())


# ---------- internals ----------


def _explicit_override() -> Path | None:
    import os

    if env_path := os.environ.get("CITERIGHT_DATA_DIR"):
        path = Path(env_path)
        if not _populated(path):
            raise RuntimeError(f"CITERIGHT_DATA_DIR={path!s} but no artifacts found")
        return path
    return None


def _populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _hf_download(target: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Cannot reach corpus: no local archive and huggingface_hub not installed. "
            "Install with: pip install huggingface_hub"
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading corpus %s -> %s (first run, ~3 GB)", HF_CORPUS_REPO, target)
    snapshot_download(HF_CORPUS_REPO, repo_type="dataset", local_dir=str(target))
    return target
