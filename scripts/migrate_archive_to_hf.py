"""Upload ``data/s3_archive/`` to a public HuggingFace dataset.

The original AWS S3 bucket dies in a few days; this script publishes the corpus
artefacts so graders and future readers can run the evaluation notebooks
without any private-bucket access.

What is uploaded
----------------
- ``chunks/`` — pre-computed semantic chunks, one JSON per paper × granularity.
- ``indexes/`` — FAISS ``IndexFlatIP`` plus the row → chunk metadata.
- ``processed/`` — Dolphin-1.5 parsed markdown + page-level layout JSON.
- ``raw_metadata/`` — OpenAlex bibliographic record per paper.

What is **not** uploaded
------------------------
- ``raw_pdfs/`` — original PDFs. We don't have explicit redistribution rights
  for every Bronze-OA paper in the corpus. Anyone who needs the PDFs can refetch
  them from OpenAlex using the ``id`` field in ``raw_metadata/<paper_id>.json``.

Usage
-----
::

    # 1. Auth (HF "huggingface-cli" is now "hf"; either works for SDK auth)
    hf auth login   # or export HF_TOKEN=...

    # 2. Dry run (verifies sizes, no upload)
    uv run python scripts/migrate_archive_to_hf.py --dry-run

    # 3. Upload (~15-30 min depending on link speed)
    uv run python scripts/migrate_archive_to_hf.py --repo-id citeright/corpus
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Final

logger = logging.getLogger("citeright.migrate")

# Subdirectories of ``data/s3_archive/`` to upload.
DEFAULT_SUBDIRS: Final[tuple[str, ...]] = ("chunks", "indexes", "processed", "raw_metadata")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    archive = args.archive.resolve()
    if not archive.is_dir():
        logger.error("Archive directory not found: %s", archive)
        return 1

    summary = _summarise(archive, args.subdirs)
    if not summary:
        logger.error("Nothing to upload (subdirs are missing or empty)")
        return 1

    if args.dry_run:
        logger.info("--dry-run: skipping upload")
        return 0

    # Build ignore patterns for subdirs we're skipping (everything in
    # ``DEFAULT_SUBDIRS`` minus the ones the caller actually selected). Always
    # ignore ``raw_pdfs/`` because we don't redistribute the source PDFs.
    selected = set(args.subdirs)
    skipped = [s for s in DEFAULT_SUBDIRS if s not in selected]
    ignore_patterns = ["raw_pdfs/**", *(f"{s}/**" for s in skipped)]

    try:
        return _upload(
            archive=archive,
            repo_id=args.repo_id,
            private=args.private,
            ignore_patterns=ignore_patterns,
        )
    except Exception:
        logger.exception("Upload failed")
        return 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/s3_archive"),
        help="Local mirror of the S3 bucket (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-id",
        default="citeright/corpus",
        help="HuggingFace dataset repo, ``org/name`` (default: %(default)s).",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=list(DEFAULT_SUBDIRS),
        help="Subdirectories of the archive to upload.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--dry-run", action="store_true", help="Compute sizes only.")
    return parser.parse_args(argv)


def _summarise(archive: Path, subdirs: list[str]) -> dict[str, tuple[int, int]]:
    """Print and return ``{subdir: (file_count, total_bytes)}``."""
    summary: dict[str, tuple[int, int]] = {}
    for sub in subdirs:
        path = archive / sub
        if not path.is_dir():
            logger.warning("Skipping missing subdir: %s", path)
            continue
        files = [p for p in path.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        summary[sub] = (len(files), total)
        logger.info("%-15s %7d files  %6.2f GB", sub, len(files), total / 1e9)

    grand = sum(b for _, b in summary.values())
    logger.info("%-15s %7s files  %6.2f GB", "TOTAL", "—", grand / 1e9)
    return summary


def _upload(
    *,
    archive: Path,
    repo_id: str,
    private: bool,
    ignore_patterns: list[str],
) -> int:
    from huggingface_hub import HfApi, create_repo  # type: ignore[import-not-found]

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info("Repo: https://huggingface.co/datasets/%s", repo_id)

    # 1. Dataset card first so the repo landing page is sensible even mid-upload.
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(_dataset_card())
        readme_path = fh.name
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: dataset card",
    )
    Path(readme_path).unlink(missing_ok=True)

    # 2. Bulk upload via ``upload_large_folder`` — resumable, multi-commit,
    # and the only sane choice for the ~64 K small files in ``processed/``.
    logger.info("Uploading %s (ignoring: %s)", archive, ", ".join(ignore_patterns))
    api.upload_large_folder(
        folder_path=str(archive),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=ignore_patterns,
        print_report=True,
    )

    logger.info("Done. https://huggingface.co/datasets/%s", repo_id)
    return 0


def _dataset_card() -> str:
    return textwrap.dedent(
        """\
        ---
        license: cc-by-nc-4.0
        language: en
        pretty_name: CiteRight RAG Corpus (IP and Innovation Policy)
        tags:
        - rag
        - retrieval
        - citation
        - faithfulness
        - scientific-papers
        size_categories:
        - 1K<n<10K
        ---

        # CiteRight Corpus

        Research dataset accompanying the CS-552 (EPFL Spring 2026) Open Project
        **Faithful RAG: Citation Accuracy and Retrieval Robustness in Domain-Specific
        Scientific Literature** by Team CiteRight (Elie Bruno, Andrea Trugenberger,
        Faruk Zahiragic, Yusif Askari).

        ## Contents

        | Path | Description |
        |---|---|
        | `chunks/<paper_id>_<coarse\\|fine>.json` | Pre-computed semantic chunks per paper (~2 000 chars coarse, ~300 chars fine). |
        | `indexes/<coarse\\|fine>.faiss` | FAISS `IndexFlatIP` built from OpenAI `text-embedding-3-small` (1 536-d). |
        | `indexes/<coarse\\|fine>_metadata.json` | Row → chunk mapping with paper id, section hierarchy, text. |
        | `processed/<paper_id>/document.md` | Dolphin-1.5 parsed markdown (preserves sections, equations, tables). |
        | `processed/<paper_id>/metadata.json` | Page-level layout metadata from the VLM. |
        | `raw_metadata/<paper_id>.json` | OpenAlex bibliographic record (DOI, year, OA status). |

        Original PDFs are **not** redistributed: anyone who needs them can refetch
        from OpenAlex using the `id` field in `raw_metadata/<paper_id>.json`.

        ## Provenance

        - **Source:** OpenAlex, filtered with `primary_topic.id=t10856` and `open_access.is_oa=true`.
        - **PDF parsing:** [Dolphin 1.5](https://github.com/ByteDance-Seed/Dolphin) on a CUDA 11.8 worker fleet.
        - **Embeddings:** OpenAI `text-embedding-3-small` (1 536-d), normalised for cosine similarity.

        ## Loading

        ```python
        from huggingface_hub import snapshot_download

        local = snapshot_download("citeright/corpus", repo_type="dataset",
                                  local_dir="/scratch/citeright_artifacts")
        ```

        Or use the helpers in [`evaluation/common/data_loader.py`](https://github.com/eliemada/cs552-faithful-rag/blob/main/evaluation/common/data_loader.py)
        which auto-resolves to the right cache.

        ## License

        Released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
        Each underlying paper retains its own license. For takedown requests, open
        a discussion on this repo.

        ## Citation

        ```bibtex
        @misc{citeright2026,
          title  = {CiteRight: Citation Accuracy and Retrieval Robustness in Scientific RAG},
          author = {Bruno, Elie and Trugenberger, Andrea and Zahiragic, Faruk and Askari, Yusif},
          year   = 2026,
          note   = {CS-552 Modern NLP Open Project, EPFL}
        }
        ```
        """
    )


if __name__ == "__main__":
    sys.exit(main())
