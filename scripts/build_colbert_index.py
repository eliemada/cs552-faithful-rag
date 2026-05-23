"""Build a PyLate PLAID index for one chunk granularity.

For each ``<paper>_<chunk_type>.json`` in ``data/s3_archive/chunks/``,
read the chunks, encode their text with the requested ColBERTv2 model,
and write a PLAID index plus a JSON metadata sidecar that mirrors the
shape produced by ``scripts.build_hf_index`` for the dense indices.

The output layout is::

    data/s3_archive/indexes/
        colbert_coarse/                   (PLAID index folder)
            index/                          (Voyager/PLAID artefacts)
            ...
        colbert_coarse_metadata.json      (chunk_id -> paper-level fields)

Run::

    uv run python -m scripts.build_colbert_index --chunk-type coarse
    uv run python -m scripts.build_colbert_index --chunk-type fine --device cuda

ColBERT emits one vector per token rather than per chunk, so the index
build is heavier than the dense one. Expect 1-3 h per granularity on
an A100 versus the ~50 min the dense indices took.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Final

from evaluation.gold_dataset._validator import REPO_ROOT
from rag_pipeline.rag.colbert_retriever import DEFAULT_MODEL_ID

DEFAULT_CHUNKS_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "chunks"
DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"

logger = logging.getLogger(__name__)


def collect_chunk_files(chunks_dir: Path, chunk_type: str) -> list[Path]:
    return sorted(chunks_dir.glob(f"*_{chunk_type}.json"))


def load_chunks(chunk_files: list[Path]) -> tuple[list[str], list[str], list[dict]]:
    """Return parallel lists ``(chunk_ids, texts, metadata)`` in stable order."""
    chunk_ids: list[str] = []
    texts: list[str] = []
    metadata: list[dict] = []
    for path in chunk_files:
        doc = json.loads(path.read_text())
        for chunk in doc["chunks"]:
            chunk_ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
            metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "paper_title": chunk.get("paper_title", ""),
                    "text": chunk["text"],
                    "section_hierarchy": chunk.get("section_hierarchy", []),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", 0),
                }
            )
    return chunk_ids, texts, metadata


def write_metadata(metadata: list[dict], path: Path) -> None:
    """Write a chunk_id -> metadata dict (matches the dense-index sidecars)."""
    payload = {entry["chunk_id"]: entry for entry in metadata}
    path.write_text(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--chunk-type", required=True, choices=["coarse", "fine"])
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="ColBERT HF id.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default=None, help="cuda | mps | cpu (auto-detected by PyLate)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N chunks. Smoke-test affordance.",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        help="PLAID residual-quantization bit width (paper default: 4).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    chunk_files = collect_chunk_files(args.chunks_dir, args.chunk_type)
    if not chunk_files:
        logger.error("No chunk files in %s for chunk_type=%s", args.chunks_dir, args.chunk_type)
        return 1
    chunk_ids, texts, metadata = load_chunks(chunk_files)
    if args.limit is not None:
        chunk_ids = chunk_ids[: args.limit]
        texts = texts[: args.limit]
        metadata = metadata[: args.limit]
    logger.info("Loaded %d chunks from %d papers", len(texts), len(chunk_files))

    # PyLate imports are heavy (transformers + fast-plaid); defer until we
    # know the chunks loaded cleanly.
    from pylate import indexes, models

    logger.info("Loading ColBERT model %s", args.model)
    model = models.ColBERT(model_name_or_path=args.model, device=args.device)

    args.indexes_dir.mkdir(parents=True, exist_ok=True)
    basename = f"colbert_{args.chunk_type}"
    index_folder = args.indexes_dir / basename

    logger.info("Encoding %d passages with %s", len(texts), args.model)
    started = time.perf_counter()
    documents_embeddings = model.encode(
        texts,
        is_query=False,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    encode_seconds = time.perf_counter() - started
    logger.info("Encoded in %.1f s (%.1f passages/s)", encode_seconds, len(texts) / encode_seconds)

    logger.info("Building PLAID index at %s (nbits=%d)", index_folder, args.nbits)
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name="index",
        override=True,
        nbits=args.nbits,
    )
    index.add_documents(documents_ids=chunk_ids, documents_embeddings=documents_embeddings)

    metadata_path = args.indexes_dir / f"{basename}_metadata.json"
    write_metadata(metadata, metadata_path)

    logger.info("Wrote PLAID index at %s", index_folder)
    logger.info("Wrote metadata at %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
