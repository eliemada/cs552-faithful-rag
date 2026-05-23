"""Build a FAISS index from existing chunk JSONs using a sentence-transformers model.

For each ``<paper_id>_<chunk_type>.json`` file in
``data/s3_archive/chunks/``, this script reads the chunk texts, encodes
them with a Hugging Face checkpoint (BGE-M3, E5-large, ...), and writes a
FAISS ``IndexFlatL2`` plus a JSON metadata sidecar in the same layout the
existing OpenAI indices use:

    data/s3_archive/indexes/<embedder_name>_<chunk_type>.faiss
    data/s3_archive/indexes/<embedder_name>_<chunk_type>_metadata.json

The metadata is a dict keyed by the FAISS row index (as a string) so
``FAISSRetriever`` can look it up directly. Vectors are L2-normalised
inside the embedder, so ``IndexFlatL2`` over unit vectors matches the
existing OpenAI-index convention exactly.

Run e.g.::

    uv run python -m scripts.build_hf_index --model bge-m3 --chunk-type coarse
    uv run python -m scripts.build_hf_index --model e5-large --chunk-type fine
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Final

import faiss
import numpy as np

from evaluation.gold_dataset._validator import REPO_ROOT
from rag_pipeline.rag.embedder import SentenceTransformerEmbedder

DEFAULT_CHUNKS_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "chunks"
DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"

# Short, file-safe aliases mapped to their Hugging Face IDs. The aliases
# double as the ``embedder.name`` prefix used in the output filenames.
MODEL_ALIASES: Final[dict[str, str]] = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/e5-large-v2",
    "e5-large-multi": "intfloat/multilingual-e5-large",
}

logger = logging.getLogger(__name__)


def collect_chunk_files(chunks_dir: Path, chunk_type: str) -> list[Path]:
    """All ``<paper>_<chunk_type>.json`` files in the chunks directory, sorted."""
    return sorted(chunks_dir.glob(f"*_{chunk_type}.json"))


def load_chunks(chunk_files: list[Path]) -> tuple[list[str], list[dict]]:
    """Return (texts, metadata) in stable order across paper files and chunks."""
    texts: list[str] = []
    metadata: list[dict] = []
    for path in chunk_files:
        doc = json.loads(path.read_text())
        for chunk in doc["chunks"]:
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
    return texts, metadata


def build_index(
    embedder: SentenceTransformerEmbedder,
    texts: list[str],
    *,
    batch_size: int,
) -> tuple[faiss.IndexFlatL2, np.ndarray]:
    """Encode ``texts`` in batches and pack them into a flat L2 index."""
    logger.info("Encoding %d passages with %s (batch=%d)", len(texts), embedder.name, batch_size)
    chunk_size = max(batch_size, 1024)
    vectors_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    for offset in range(0, len(texts), chunk_size):
        batch = texts[offset : offset + chunk_size]
        vectors_chunks.append(embedder.encode_passages(batch))
        done = offset + len(batch)
        elapsed = time.perf_counter() - start
        rate = done / max(elapsed, 1e-6)
        eta = (len(texts) - done) / max(rate, 1e-6)
        logger.info("  %d / %d  (%.1f passages/s, ETA %.0fs)", done, len(texts), rate, eta)
    vectors = np.vstack(vectors_chunks).astype(np.float32)
    index = faiss.IndexFlatL2(embedder.dim)
    index.add(vectors)
    return index, vectors


def write_outputs(
    index: faiss.IndexFlatL2,
    metadata: list[dict],
    indexes_dir: Path,
    embedder_name: str,
    chunk_type: str,
) -> tuple[Path, Path]:
    """Save FAISS + metadata to ``<embedder_name>_<chunk_type>.{faiss,_metadata.json}``."""
    indexes_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{embedder_name}_{chunk_type}"
    index_path = indexes_dir / f"{basename}.faiss"
    metadata_path = indexes_dir / f"{basename}_metadata.json"
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps({str(i): m for i, m in enumerate(metadata)}))
    return index_path, metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_ALIASES),
        help="Short alias resolved to a HF model id (see MODEL_ALIASES).",
    )
    parser.add_argument("--chunk-type", required=True, choices=["coarse", "fine"])
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detect by default)")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Cap on encoder context length. Default 512 matches the SBERT convention.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N chunks. Smoke-test affordance.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to the embedder name (e.g. '_smoke').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    chunk_files = collect_chunk_files(args.chunks_dir, args.chunk_type)
    if not chunk_files:
        logger.error("No chunk files in %s for chunk_type=%s", args.chunks_dir, args.chunk_type)
        return 1

    texts, metadata = load_chunks(chunk_files)
    if args.limit is not None:
        texts = texts[: args.limit]
        metadata = metadata[: args.limit]
    logger.info("Loaded %d chunks from %d papers", len(texts), len(chunk_files))

    embedder = SentenceTransformerEmbedder(
        MODEL_ALIASES[args.model],
        short_name=args.model.replace("-", "_"),
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    logger.info(
        "Embedder ready: model=%s dim=%d device=%s",
        embedder.name,
        embedder.dim,
        embedder.device,
    )

    index, _ = build_index(embedder, texts, batch_size=args.batch_size)
    name = embedder.name + args.output_suffix
    index_path, metadata_path = write_outputs(
        index, metadata, args.indexes_dir, name, args.chunk_type
    )
    logger.info("Wrote %s (%d vectors)", index_path, index.ntotal)
    logger.info("Wrote %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
