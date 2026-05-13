"""Map gold-dataset supporting spans to chunk IDs.

The gold dataset stores citations as ``(paper_id, char_start, char_end, quote)``
spans into the LF-normalised ``document.md``. Retrieval, however, returns
``chunk_id`` strings. Computing retrieval metrics requires bridging the two:
for each gold span, which chunk IDs in the corpus would count as a "correct
hit"?

This module answers that question by defining a chunk as gold iff its
``[chunk.char_start, chunk.char_end)`` interval overlaps the gold span's
``[span.char_start, span.char_end)`` interval by ≥ 1 character. Strict
containment was considered and rejected — fine-granularity chunks (~300 chars)
are often shorter than a multi-sentence claim, which would force the same
claim to have its gold-chunk set be empty under containment semantics.

Adversarial pairs (``annotator == "adversarial"``) and unanswerable pairs
(``difficulty == "unanswerable"``) are filtered out: the former have
deliberately wrong claims (used elsewhere for fooling-rate eval), and the
latter have no supporting spans by schema invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

from evaluation.gold_dataset._validator import DEFAULT_GOLD_QA, REPO_ROOT

DEFAULT_CHUNKS_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "chunks"
CHUNK_TYPES: Final[tuple[str, ...]] = ("coarse", "fine")


@dataclass(frozen=True)
class ResolvedQuery:
    """One gold question with its query text and gold targets at two granularities.

    Two levels of ground truth are exposed:

    * ``gold_paper_ids`` — the set of paper IDs the question's claims cite.
      This is the **primary** retrieval target. It is always non-empty for
      evaluable queries.
    * ``gold_chunk_ids[chunk_type]`` — for each chunk granularity, the set
      of chunk IDs whose ``[char_start, char_end)`` interval overlaps any
      of the question's supporting spans. **Often empty** for our current
      corpus: the existing chunker (`packages/.../rag/chunking.py`) skips
      ~40–80 % of document content between section boundaries, so gold
      spans frequently land in chunk-coverage gaps. The ``has_chunk_coverage``
      flag below records whether any chunks were found at the given
      granularity — chunk-level metrics should be reported on the
      ``has_chunk_coverage`` subset only, paper-level metrics on all
      evaluable queries.
    """

    query_id: str
    query_text: str
    category: str
    difficulty: str
    gold_paper_ids: frozenset[str]
    gold_chunk_ids: dict[str, frozenset[str]]

    def has_chunk_coverage(self, chunk_type: str) -> bool:
        """True iff at least one chunk at this granularity overlaps a gold span."""
        return bool(self.gold_chunk_ids.get(chunk_type))


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Return True iff half-open intervals [a_start, a_end) and [b_start, b_end) share ≥ 1 char."""
    return not (a_end <= b_start or a_start >= b_end)


def _load_chunks(paper_id: str, chunk_type: str, chunks_dir: Path) -> list[dict]:
    """Load the per-paper chunk file. Returns an empty list if the file is missing."""
    path = chunks_dir / f"{paper_id}_{chunk_type}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    chunks = payload.get("chunks", [])
    return chunks if isinstance(chunks, list) else []


def _gold_chunks_for_span(
    paper_id: str,
    span_start: int,
    span_end: int,
    chunk_type: str,
    chunks_dir: Path,
    cache: dict[tuple[str, str], list[dict]],
) -> set[str]:
    """All chunks of one paper at one granularity that overlap the given span."""
    key = (paper_id, chunk_type)
    if key not in cache:
        cache[key] = _load_chunks(paper_id, chunk_type, chunks_dir)
    hits: set[str] = set()
    for chunk in cache[key]:
        if _intervals_overlap(
            chunk.get("char_start", 0),
            chunk.get("char_end", 0),
            span_start,
            span_end,
        ):
            hits.add(chunk["chunk_id"])
    return hits


def _is_evaluable(pair: dict) -> bool:
    """Filter rule: only natural, answerable pairs are evaluated for retrieval."""
    if pair.get("annotator") == "adversarial":
        return False
    if pair.get("difficulty") == "unanswerable":
        return False
    return True


def resolve(
    pairs: Iterable[dict],
    *,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    chunk_types: tuple[str, ...] = CHUNK_TYPES,
) -> list[ResolvedQuery]:
    """Resolve gold spans to chunk IDs for every evaluable pair.

    Caches loaded chunk files within a single call so re-visiting the same
    (paper, chunk_type) pair across multiple spans/claims is O(1).
    """
    cache: dict[tuple[str, str], list[dict]] = {}
    out: list[ResolvedQuery] = []
    for pair in pairs:
        if not _is_evaluable(pair):
            continue

        paper_ids: set[str] = set()
        per_type: dict[str, set[str]] = {t: set() for t in chunk_types}

        for claim in pair.get("claims", []):
            for span in claim.get("supporting_spans", []):
                paper = span.get("paper_id")
                cs = span.get("char_start")
                ce = span.get("char_end")
                if not isinstance(paper, str) or not isinstance(cs, int) or not isinstance(ce, int):
                    continue
                paper_ids.add(paper)
                for chunk_type in chunk_types:
                    per_type[chunk_type].update(
                        _gold_chunks_for_span(paper, cs, ce, chunk_type, chunks_dir, cache)
                    )

        out.append(
            ResolvedQuery(
                query_id=pair["id"],
                query_text=pair["question"],
                category=pair.get("category", ""),
                difficulty=pair.get("difficulty", ""),
                gold_paper_ids=frozenset(paper_ids),
                gold_chunk_ids={t: frozenset(per_type[t]) for t in chunk_types},
            )
        )
    return out


def resolve_from_file(
    gold_qa_path: Path = DEFAULT_GOLD_QA,
    *,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    chunk_types: tuple[str, ...] = CHUNK_TYPES,
) -> list[ResolvedQuery]:
    """Convenience wrapper: read ``gold_qa.json`` and resolve."""
    pairs = json.loads(gold_qa_path.read_text() or "[]")
    return resolve(pairs, chunks_dir=chunks_dir, chunk_types=chunk_types)
