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

import math

from evaluation.retrieval_eval.evaluate_retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_eval.gold_resolver import (
    ResolvedQuery,
    _intervals_overlap,
    resolve,
)
from evaluation.retrieval_eval.retrievers import (
    CONFIGS,
    CONFIGS_BY_NAME,
    RetrieverConfig,
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


def test_ndcg_single_gold_rank_one_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1.0)


def test_ndcg_single_gold_rank_three() -> None:
    # rank 3 → DCG = 1/log2(4) = 0.5; IDCG = 1 → nDCG = 0.5
    assert ndcg_at_k(["x", "y", "a"], {"a"}, 5) == pytest.approx(0.5)


def test_ndcg_no_hit_is_zero() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, 5) == 0.0


def test_ndcg_two_gold_both_in_topk() -> None:
    # gold ranks 1 and 3 → DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
    # IDCG (G=2 ideal at ranks 1, 2) = 1/log2(2) + 1/log2(3)
    retrieved = ["a", "x", "b"]
    gold = {"a", "b"}
    expected = (1.0 + 1 / math.log2(4)) / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(retrieved, gold, 3) == pytest.approx(expected)


def test_ndcg_caps_at_k() -> None:
    # gold at rank 4 → outside top-3 → nDCG@3 = 0
    assert ndcg_at_k(["x", "y", "z", "a"], {"a"}, 3) == 0.0
    # but inside top-5
    assert ndcg_at_k(["x", "y", "z", "a"], {"a"}, 5) == pytest.approx(1 / math.log2(5))


def test_ndcg_empty_gold_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_ndcg_dedupes_repeated_gold_paper() -> None:
    # Paper-level retrieval can repeat a paper across chunks. The gold paper
    # must count once, at rank 1; subsequent repeats add nothing.
    retrieved = ["paperA"] * 20
    assert ndcg_at_k(retrieved, {"paperA"}, 10) == pytest.approx(1.0)


def test_ndcg_bounded_by_one() -> None:
    # Random-ish ranking with duplicates — value must stay in [0, 1].
    retrieved = ["a", "a", "b", "a", "b", "c", "a"]
    gold = {"a", "b"}
    score = ndcg_at_k(retrieved, gold, 5)
    assert 0.0 <= score <= 1.0


# ---- retriever configs ----------------------------------------------------


def test_config_matrix_covers_four_embedders_x_two_chunks_x_pm_rerank() -> None:
    # 4 embedders × 2 granularities × 2 ±rerank
    assert len(CONFIGS) == 16
    embedders = {c.embedder for c in CONFIGS}
    assert embedders == {"openai", "bge_m3", "e5_large", "colbert"}
    granularities = {c.chunk_type for c in CONFIGS}
    assert granularities == {"coarse", "fine"}


def test_openai_configs_keep_legacy_index_basename() -> None:
    # Legacy OpenAI indices live at coarse.faiss / fine.faiss, unprefixed.
    assert CONFIGS_BY_NAME["coarse_faiss"].index_basename() == "coarse"
    assert CONFIGS_BY_NAME["fine_rerank"].index_basename() == "fine"


def test_alt_embedder_configs_prefix_their_basename() -> None:
    assert CONFIGS_BY_NAME["bge_m3_coarse_faiss"].index_basename() == "bge_m3_coarse"
    assert CONFIGS_BY_NAME["e5_large_fine_rerank"].index_basename() == "e5_large_fine"
    assert CONFIGS_BY_NAME["colbert_coarse_faiss"].index_basename() == "colbert_coarse"
    assert CONFIGS_BY_NAME["colbert_fine_rerank"].index_basename() == "colbert_fine"


def test_only_colbert_configs_report_is_colbert() -> None:
    colbert_configs = [c for c in CONFIGS if c.embedder == "colbert"]
    other_configs = [c for c in CONFIGS if c.embedder != "colbert"]
    assert len(colbert_configs) == 4  # 2 granularities × 2 ±rerank
    assert all(c.is_colbert() for c in colbert_configs)
    assert not any(c.is_colbert() for c in other_configs)


def test_requires_openai_only_for_openai_embedder_configs() -> None:
    openai_configs = [c for c in CONFIGS if c.embedder == "openai"]
    hf_configs = [c for c in CONFIGS if c.embedder != "openai"]
    assert all(c.requires_openai() for c in openai_configs)
    assert not any(c.requires_openai() for c in hf_configs)


def test_requires_zeroentropy_only_for_rerank_configs() -> None:
    rerank_configs = [c for c in CONFIGS if c.use_reranker]
    plain_configs = [c for c in CONFIGS if not c.use_reranker]
    assert all(c.requires_zeroentropy() for c in rerank_configs)
    assert not any(c.requires_zeroentropy() for c in plain_configs)


# ---- BaseRetriever protocol + HybridRetriever composition ----------------


class _FakeRetriever:
    """Minimal BaseRetriever stand-in for tests.

    Records the queries it sees and returns a synthetic ranked list. Lets us
    exercise HybridRetriever's composition logic without instantiating FAISS
    or any HF model.
    """

    def __init__(self, results: list) -> None:
        self._results = results
        self.seen_queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 50):  # noqa: ANN201 — duck-typed
        self.seen_queries.append((query, top_k))
        return list(self._results[:top_k])


def _make_search_result(rank: int, chunk_id: str = "c", paper_id: str = "p"):
    # Late import keeps the module test-importable without faiss at parse time.
    from rag_pipeline.rag.retriever import SearchResult

    return SearchResult(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title="",
        text="",
        section_hierarchy=[],
        score=1.0 - rank * 0.1,
        rank=rank,
    )


def test_hybrid_retriever_accepts_arbitrary_base_via_protocol() -> None:
    from rag_pipeline.rag.retriever import HybridRetriever
    from rag_pipeline.rag.retriever_base import BaseRetriever

    base = _FakeRetriever([_make_search_result(i) for i in range(20)])
    # Runtime-checkable Protocol — duck-typed conformance.
    assert isinstance(base, BaseRetriever)

    hybrid = HybridRetriever(base_retriever=base, reranker=None, faiss_candidates=10)
    results = hybrid.search("q", top_k=3, use_reranker=False)
    assert [r.rank for r in results] == [0, 1, 2]
    # Verify the base saw the candidate budget, not the top_k.
    assert base.seen_queries == [("q", 10)]


def test_hybrid_retriever_legacy_positional_still_works() -> None:
    """Old callers passed the retriever positionally as ``faiss_retriever``."""
    from rag_pipeline.rag.retriever import HybridRetriever

    base = _FakeRetriever([_make_search_result(i) for i in range(5)])
    hybrid = HybridRetriever(base, None, faiss_candidates=5)  # legacy positional form
    assert hybrid.base_retriever is base
    # The historical ``faiss_retriever`` attribute is preserved for callers
    # like ``api/main.py`` that introspect ``hybrid.faiss_retriever.index``.
    assert hybrid.faiss_retriever is base


def test_hybrid_retriever_refuses_construction_without_base() -> None:
    from rag_pipeline.rag.retriever import HybridRetriever

    with pytest.raises(TypeError, match="base retriever"):
        HybridRetriever()


def test_retriever_config_is_frozen() -> None:
    cfg = CONFIGS[0]
    with pytest.raises(Exception):
        setattr(cfg, "name", "modified")
    # Equality / hashing works for ``frozen=True``.
    assert (
        RetrieverConfig(
            name=cfg.name,
            chunk_type=cfg.chunk_type,
            use_reranker=cfg.use_reranker,
            embedder=cfg.embedder,
        )
        == cfg
    )
