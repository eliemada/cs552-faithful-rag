"""Tests for retrieval-ablation building blocks.

Focused on the parts that touch real data:
  - gold_resolver: span-to-chunk overlap correctness, filter rules
  - evaluate_retrieval: metric correctness on hand-crafted inputs

The retriever adapter is integration-tested implicitly by the end-to-end
smoke run in ``scripts/run_retrieval_ablation.py``; we don't unit-test it
here because it would require mocking HybridRetriever + FAISS + the
embedding API, which costs more than the test buys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.retrieval_eval.evaluate_retrieval import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_eval.gold_resolver import (
    ResolvedQuery,
    _intervals_overlap,
    resolve,
)


# ---- _intervals_overlap ----------------------------------------------------


@pytest.mark.parametrize(
    "a_start,a_end,b_start,b_end,expected",
    [
        # disjoint, a before b
        (0, 10, 20, 30, False),
        # disjoint, b before a
        (20, 30, 0, 10, False),
        # touching at boundary — half-open semantics say no overlap
        (0, 10, 10, 20, False),
        (10, 20, 0, 10, False),
        # one inside the other
        (0, 100, 40, 50, True),
        (40, 50, 0, 100, True),
        # partial overlap, a starts before
        (0, 50, 30, 80, True),
        # partial overlap, b starts before
        (30, 80, 0, 50, True),
        # identical
        (10, 20, 10, 20, True),
        # single-character overlap
        (0, 11, 10, 20, True),
    ],
)
def test_intervals_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int, expected: bool
) -> None:
    assert _intervals_overlap(a_start, a_end, b_start, b_end) is expected


# ---- resolve() filtering --------------------------------------------------


def _pair(
    *,
    pair_id: str,
    annotator: str = "elie",
    difficulty: str = "single-hop",
    spans: list[tuple[str, int, int]] | None = None,
) -> dict:
    spans = spans or [("paper_X", 0, 10)]
    return {
        "id": pair_id,
        "question": f"Q for {pair_id}",
        "gold_answer": "A",
        "difficulty": difficulty,
        "category": "factual",
        "annotator": annotator,
        "reviewer": None,
        "review_status": "pending",
        "iaa_subset": False,
        "claims": [
            {
                "id": "c001",
                "text": "claim text",
                "supporting_spans": [
                    {"paper_id": p, "char_start": s, "char_end": e, "quote": "..."}
                    for (p, s, e) in spans
                ],
                "annotator_label": "supports",
                "reviewer_label": None,
            }
        ],
    }


def test_resolve_filters_adversarial_and_unanswerable(tmp_path: Path) -> None:
    pairs = [
        _pair(pair_id="q001"),
        _pair(pair_id="q002", annotator="adversarial"),
        _pair(pair_id="q003", difficulty="unanswerable", spans=[]),
        _pair(pair_id="q004"),
    ]
    # Make resolve happy by writing empty chunk files (no chunks → empty gold_chunk_ids,
    # but gold_paper_ids still populated from the spans).
    (tmp_path / "paper_X_coarse.json").write_text(json.dumps({"chunks": []}))
    (tmp_path / "paper_X_fine.json").write_text(json.dumps({"chunks": []}))

    out = resolve(pairs, chunks_dir=tmp_path)
    ids = [q.query_id for q in out]
    assert ids == ["q001", "q004"]


def test_resolve_picks_overlapping_chunks(tmp_path: Path) -> None:
    # Three chunks: one fully before, one overlapping, one fully after.
    chunks = {
        "chunks": [
            {"chunk_id": "p_coarse_0", "char_start": 0, "char_end": 100},
            {"chunk_id": "p_coarse_1", "char_start": 90, "char_end": 200},
            {"chunk_id": "p_coarse_2", "char_start": 300, "char_end": 400},
        ]
    }
    (tmp_path / "p_coarse.json").write_text(json.dumps(chunks))
    (tmp_path / "p_fine.json").write_text(json.dumps({"chunks": []}))

    pair = _pair(pair_id="q001", spans=[("p", 95, 150)])
    out = resolve([pair], chunks_dir=tmp_path)
    assert len(out) == 1
    coarse_gold = out[0].gold_chunk_ids["coarse"]
    # First chunk overlaps at [95, 100); second chunk overlaps at [95, 150)
    assert coarse_gold == frozenset({"p_coarse_0", "p_coarse_1"})


def test_resolve_collects_gold_paper_ids_across_claims(tmp_path: Path) -> None:
    (tmp_path / "paperA_coarse.json").write_text(json.dumps({"chunks": []}))
    (tmp_path / "paperA_fine.json").write_text(json.dumps({"chunks": []}))
    (tmp_path / "paperB_coarse.json").write_text(json.dumps({"chunks": []}))
    (tmp_path / "paperB_fine.json").write_text(json.dumps({"chunks": []}))

    pair = _pair(pair_id="q001", spans=[("paperA", 0, 10), ("paperB", 0, 10)])
    out = resolve([pair], chunks_dir=tmp_path)
    assert out[0].gold_paper_ids == frozenset({"paperA", "paperB"})


def test_resolved_query_chunk_coverage_helper() -> None:
    q = ResolvedQuery(
        query_id="q001",
        query_text="Q",
        category="factual",
        difficulty="single-hop",
        gold_paper_ids=frozenset({"p"}),
        gold_chunk_ids={"coarse": frozenset({"p_coarse_0"}), "fine": frozenset()},
    )
    assert q.has_chunk_coverage("coarse") is True
    assert q.has_chunk_coverage("fine") is False
    assert q.has_chunk_coverage("missing") is False


# ---- metric correctness ----------------------------------------------------


def test_precision_at_k_basic() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "z"}
    # top-3: a, b, c → 2 hits out of 3 → 2/3
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    # top-5: a, b, c, d, e → 2 hits / 5
    assert precision_at_k(retrieved, gold, 5) == pytest.approx(2 / 5)


def test_recall_at_k_basic() -> None:
    retrieved = ["a", "b", "c"]
    gold = {"a", "c", "z"}
    # top-3 contains 2 of 3 gold items
    assert recall_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_gold_is_zero() -> None:
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_hit_rate_binary() -> None:
    retrieved = ["a", "b", "c"]
    assert hit_rate_at_k(retrieved, {"c"}, 3) == 1.0
    assert hit_rate_at_k(retrieved, {"c"}, 2) == 0.0
    assert hit_rate_at_k(retrieved, {"z"}, 3) == 0.0


def test_reciprocal_rank() -> None:
    # First gold hit at index 0 (rank 1) → RR = 1.0
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)
    # First gold hit at index 2 (rank 3) → RR = 1/3
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    # No hit → RR = 0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
