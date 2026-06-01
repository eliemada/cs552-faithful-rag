"""Generate chunk variants for the M3 chunker ablation.

Produces drop-in-compatible chunk JSONs under
``data/s3_archive/chunks/<paper_id>_<variant_key>.json`` that match the
existing ``MarkdownChunker`` output schema. Downstream tools (``build_hf_index``,
``gold_resolver``) read them by ``chunk_type == variant_key``.

Variants in the grid (anchored on ``e5_large + reranker``):

- ``s{SIZE}_o{OVERLAP_CHARS}`` for SIZE in {200, 400, 600, 800} and OVERLAP
  in {0, 50%-of-SIZE} — 8 fixed-size paragraph-aware configs.
- ``recursive_400`` — separator-cascade splitter (no heading-aware first
  pass), 400-char target, 80-char overlap.

Run::

    uv run python -m scripts.generate_chunk_variants \
        --processed-dir data/s3_archive/processed \
        --out-dir data/s3_archive/chunks \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, NamedTuple

from rag_pipeline.rag.markdown_chunker import Chunk, MarkdownChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_chunk_variants")


class VariantSpec(NamedTuple):
    """Specification of one chunking variant."""

    key: str
    strategy: str  # "fixed" | "recursive"
    chunk_size: int
    overlap: int


VARIANT_GRID: tuple[VariantSpec, ...] = (
    VariantSpec("s200_o0", "fixed", 200, 0),
    VariantSpec("s200_o100", "fixed", 200, 100),
    VariantSpec("s400_o0", "fixed", 400, 0),
    VariantSpec("s400_o200", "fixed", 400, 200),
    VariantSpec("s600_o0", "fixed", 600, 0),
    VariantSpec("s600_o300", "fixed", 600, 300),
    VariantSpec("s800_o0", "fixed", 800, 0),
    VariantSpec("s800_o400", "fixed", 800, 400),
    VariantSpec("recursive_400", "recursive", 400, 80),
)


# ---- chunk-id and schema helpers ------------------------------------------------


def _to_dict(chunk: Chunk, variant_key: str) -> dict:
    """Serialize a ``Chunk`` to the on-disk schema with the variant key swapped in."""
    d = asdict(chunk)
    d["chunk_id"] = f"{chunk.paper_id}_{variant_key}_{chunk.chunk_index:04d}"
    d["chunk_type"] = variant_key
    return d


def _wrap_payload(paper_id: str, paper_title: str, variant_key: str, chunks: list[Chunk]) -> dict:
    """Wrap a chunk list in the top-level dict format used by existing files."""
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "chunk_type": variant_key,
        "total_chunks": len(chunks),
        "chunks": [_to_dict(c, variant_key) for c in chunks],
    }


# ---- fixed-size paragraph-aware (MarkdownChunker with custom params) ------------


def _chunks_fixed(
    markdown_text: str,
    paper_id: str,
    paper_title: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Use MarkdownChunker with custom target/max/overlap to emit paragraph-aware chunks."""
    max_size = int(chunk_size * 1.25)
    overlap_pct = overlap / chunk_size if chunk_size > 0 else 0.0
    chunker = MarkdownChunker(
        coarse_target_size=chunk_size,
        coarse_max_size=max_size,
        coarse_overlap_pct=overlap_pct,
    )
    return chunker.create_coarse_chunks(markdown_text, paper_id, paper_title)


# ---- recursive separator-cascade splitter ---------------------------------------


_RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _recursive_split(text: str, target: int, separators: Iterable[str]) -> list[str]:
    """Split ``text`` recursively at the first separator that yields a useful split.

    Mirrors the LangChain ``RecursiveCharacterTextSplitter`` contract: try each
    separator in order; if a fragment is still oversize, recurse with the
    remaining separators.
    """
    seps = list(separators)
    if not seps:
        return [text]

    sep, *rest = seps
    if sep == "":
        # Last-resort: hard char-level split.
        return [text[i : i + target] for i in range(0, len(text), target)] or [text]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""

    for part in parts:
        piece = part + (sep if part is not parts[-1] else "")
        if len(piece) > target:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_recursive_split(piece, target, rest))
        elif len(buf) + len(piece) > target:
            if buf:
                out.append(buf)
            buf = piece
        else:
            buf += piece

    if buf:
        out.append(buf)

    return [p for p in out if p.strip()]


def _chunks_recursive(
    markdown_text: str,
    paper_id: str,
    paper_title: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Recursive separator-cascade splitter with sliding-window overlap."""
    cleaned = re.sub(r"\n---+\n", "\n\n", markdown_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    raw_parts = _recursive_split(cleaned, chunk_size, _RECURSIVE_SEPARATORS)

    # Sliding-window overlap: prepend the last `overlap` chars of the previous chunk.
    overlapped: list[str] = []
    prev_tail = ""
    for part in raw_parts:
        text = (prev_tail + part) if prev_tail else part
        overlapped.append(text)
        prev_tail = part[-overlap:] if overlap > 0 else ""

    # Locate each chunk's start offset in the original (cleaned) document.
    chunks: list[Chunk] = []
    cursor = 0
    for i, text in enumerate(overlapped):
        # Find the chunk's body (ignore the prepended overlap) in the source text.
        body = text[len(prev_tail) :] if i == 0 else text[overlap:] if overlap > 0 else text
        idx = cleaned.find(body.strip()[:64], cursor) if body.strip() else cursor
        char_start = idx if idx >= 0 else cursor
        cursor = max(cursor, char_start + len(text))
        chunks.append(
            Chunk(
                text=text,
                chunk_id=f"{paper_id}_recursive_{i:04d}",  # overwritten by _to_dict
                chunk_type="recursive",
                paper_id=paper_id,
                paper_title=paper_title,
                section_hierarchy=[],
                char_start=char_start,
                char_end=char_start + len(text),
                chunk_index=i,
                total_chunks=len(overlapped),
                overlap_with_previous=(i > 0 and overlap > 0),
            )
        )

    return chunks


# ---- per-paper driver -----------------------------------------------------------


def _process_paper(args: tuple[Path, Path, tuple[VariantSpec, ...], bool]) -> tuple[str, int, int]:
    paper_dir, out_dir, variants, skip_existing = args
    paper_id = paper_dir.name
    doc_path = paper_dir / "document.md"
    meta_path = paper_dir / "metadata.json"

    if not doc_path.is_file():
        return paper_id, 0, 0

    markdown_text = doc_path.read_text(encoding="utf-8")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        paper_title = meta.get("title") or meta.get("display_name") or paper_id
    except Exception:
        paper_title = paper_id

    written = 0
    skipped = 0
    for v in variants:
        out_path = out_dir / f"{paper_id}_{v.key}.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        if v.strategy == "fixed":
            chunks = _chunks_fixed(markdown_text, paper_id, paper_title, v.chunk_size, v.overlap)
        elif v.strategy == "recursive":
            chunks = _chunks_recursive(
                markdown_text, paper_id, paper_title, v.chunk_size, v.overlap
            )
        else:  # pragma: no cover — guarded by VARIANT_GRID
            raise ValueError(f"unknown strategy: {v.strategy}")
        out_path.write_text(json.dumps(_wrap_payload(paper_id, paper_title, v.key, chunks)))
        written += 1
    return paper_id, written, skipped


# ---- CLI ------------------------------------------------------------------------


def _select_variants(only: str | None) -> tuple[VariantSpec, ...]:
    if not only:
        return VARIANT_GRID
    keys = {k.strip() for k in only.split(",") if k.strip()}
    chosen = tuple(v for v in VARIANT_GRID if v.key in keys)
    missing = keys - {v.key for v in chosen}
    if missing:
        raise SystemExit(f"unknown variant key(s): {sorted(missing)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/s3_archive/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/s3_archive/chunks"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated variant keys; default: all in VARIANT_GRID.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N papers.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Don't regenerate variants that already have a chunk JSON.",
    )
    args = parser.parse_args()

    variants = _select_variants(args.only)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = sorted(p for p in args.processed_dir.iterdir() if p.is_dir())
    if args.limit:
        paper_dirs = paper_dirs[: args.limit]

    log.info(
        "processing %d papers × %d variants → %s",
        len(paper_dirs),
        len(variants),
        args.out_dir,
    )

    total_written = 0
    total_skipped = 0
    work = [(pd, args.out_dir, variants, args.skip_existing) for pd in paper_dirs]
    if args.workers <= 1:
        for w in work:
            _, written, skipped = _process_paper(w)
            total_written += written
            total_skipped += skipped
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_paper, w) for w in work]
            for i, fut in enumerate(as_completed(futs), 1):
                _, written, skipped = fut.result()
                total_written += written
                total_skipped += skipped
                if i % 50 == 0:
                    log.info("progress: %d / %d papers", i, len(paper_dirs))

    log.info("done: %d written, %d skipped", total_written, total_skipped)


if __name__ == "__main__":
    main()
