"""Unit tests for the Corrective RAG pipeline (Faruk's component).

All tests run offline: the cross-encoder is replaced with a fake scorer and
query refinement is monkeypatched, so no model download, GPU, or LLM call
happens. The focus is the routing/threshold logic and the M3 abstain branch.
"""

from __future__ import annotations

import math

import pytest

from evaluation.crag import corrective_rag as crag
from evaluation.crag.corrective_rag import (
    CRAGConfig,
    CRAGResult,
    RetrievalQuality,
    _sigmoid_if_logit,
    corrective_rag,
    evaluate_retrieval_quality,
    finalize_answer,
)


class _FakeScorer:
    """Stand-in for a sentence-transformers CrossEncoder.

    ``score_fn(query, doc_text) -> float`` lets each test script the
    relevance landscape without loading a model.
    """

    def __init__(self, score_fn):
        self._score_fn = score_fn

    def predict(self, pairs, **kwargs):  # noqa: ANN001 — duck-typed
        return [self._score_fn(q, d) for (q, d) in pairs]


@pytest.fixture
def patch_scorer(monkeypatch):
    def _install(score_fn):
        monkeypatch.setattr(crag, "_get_cross_encoder", lambda model_name: _FakeScorer(score_fn))

    return _install


def _one_doc_retriever(query, k):  # noqa: ANN001 — matches Retriever protocol
    return [{"text": "some passage"}]


# ---- score normalisation -------------------------------------------------


def test_sigmoid_passes_through_probabilities():
    assert _sigmoid_if_logit(0.0) == 0.0
    assert _sigmoid_if_logit(0.42) == 0.42
    assert _sigmoid_if_logit(1.0) == 1.0


def test_sigmoid_squashes_raw_logits():
    assert _sigmoid_if_logit(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert 0.0 < _sigmoid_if_logit(-3.0) < 0.5


# ---- evaluate_retrieval_quality ------------------------------------------


def test_empty_documents_are_incorrect():
    quality, confidence = evaluate_retrieval_quality("q", [], CRAGConfig())
    assert quality is RetrievalQuality.INCORRECT
    assert confidence == 0.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.9, RetrievalQuality.CORRECT),
        (0.7, RetrievalQuality.CORRECT),  # boundary: >= high
        (0.5, RetrievalQuality.AMBIGUOUS),
        (0.3, RetrievalQuality.AMBIGUOUS),  # boundary: >= low
        (0.1, RetrievalQuality.INCORRECT),
    ],
)
def test_threshold_mapping(patch_scorer, score, expected):
    patch_scorer(lambda q, d: score)
    quality, confidence = evaluate_retrieval_quality("q", [{"text": "d"}], CRAGConfig())
    assert quality is expected
    assert confidence == pytest.approx(score)


def test_best_score_across_documents_is_used(patch_scorer):
    patch_scorer(lambda q, d: 0.95 if d == "good" else 0.05)
    quality, confidence = evaluate_retrieval_quality(
        "q", [{"text": "bad"}, {"text": "good"}], CRAGConfig()
    )
    assert quality is RetrievalQuality.CORRECT
    assert confidence == pytest.approx(0.95)


# ---- corrective_rag routing ----------------------------------------------


def test_correct_returns_documents_immediately(patch_scorer):
    patch_scorer(lambda q, d: 0.9)
    res = corrective_rag("q", _one_doc_retriever, CRAGConfig())
    assert res.quality is RetrievalQuality.CORRECT
    assert res.abstained is False
    assert res.final_documents
    assert res.retrieval_rounds == 1
    assert res.refined_query is None


def test_incorrect_abstains_by_default(patch_scorer):
    patch_scorer(lambda q, d: 0.1)
    res = corrective_rag("q", _one_doc_retriever, CRAGConfig())
    assert res.quality is RetrievalQuality.INCORRECT
    assert res.abstained is True
    assert res.final_documents == []
    # INCORRECT never refines, so exactly one retrieval round.
    assert res.retrieval_rounds == 1


def test_incorrect_keeps_documents_when_abstain_disabled(patch_scorer):
    patch_scorer(lambda q, d: 0.1)
    res = corrective_rag("q", _one_doc_retriever, CRAGConfig(use_abstain_fallback=False))
    assert res.quality is RetrievalQuality.INCORRECT
    assert res.abstained is False
    assert res.final_documents  # legacy behaviour: low-quality docs passed through


def test_ambiguous_refines_up_to_budget_then_abstains(patch_scorer, monkeypatch):
    patch_scorer(lambda q, d: 0.5)  # perpetually ambiguous
    refine_calls: list[str] = []

    def _fake_refine(query, docs, config):  # noqa: ANN001
        refine_calls.append(query)
        return query + " +"

    monkeypatch.setattr(crag, "refine_query", _fake_refine)
    res = corrective_rag("q", _one_doc_retriever, CRAGConfig(max_retries=2))

    # attempt 0 (ambiguous → refine), attempt 1 (ambiguous → refine),
    # attempt 2 (ambiguous, budget spent → abstain) = 3 retrieval rounds.
    assert res.retrieval_rounds == 3
    assert len(refine_calls) == 2
    assert res.quality is RetrievalQuality.AMBIGUOUS
    assert res.abstained is True
    assert res.final_documents == []
    assert res.refined_query is not None  # query was rewritten


def test_ambiguous_recovers_to_correct_after_refine(patch_scorer, monkeypatch):
    patch_scorer(lambda q, d: 0.9 if "REFINED" in q else 0.5)
    monkeypatch.setattr(crag, "refine_query", lambda query, docs, config: "REFINED " + query)
    res = corrective_rag("q", _one_doc_retriever, CRAGConfig(max_retries=2))
    assert res.quality is RetrievalQuality.CORRECT
    assert res.abstained is False
    assert res.retrieval_rounds == 2
    assert res.refined_query is not None
    assert res.final_documents


# ---- finalize_answer -----------------------------------------------------


def test_finalize_answer_abstain_skips_generator():
    res = CRAGResult("q", RetrievalQuality.INCORRECT, 0.1, None, 1, [], abstained=True)
    generator_calls: list[CRAGResult] = []

    def _gen(r):  # noqa: ANN001
        generator_calls.append(r)
        return "GENERATED"

    out = finalize_answer(res, _gen, CRAGConfig(abstain_message="NO EVIDENCE"))
    assert out == "NO EVIDENCE"
    assert generator_calls == []  # generator never invoked on abstain


def test_finalize_answer_delegates_when_not_abstained():
    res = CRAGResult("q", RetrievalQuality.CORRECT, 0.9, None, 1, [{"text": "d"}], abstained=False)
    out = finalize_answer(res, lambda r: f"ans:{len(r.final_documents)}")
    assert out == "ans:1"


# ---- regression: doc-text plumbing ---------------------------------------
# These guard the bug where `RetrieverAdapter.search` dropped the `text`
# field and collapsed every confidence to ~0 in the ablation sweep.


def test_empty_doc_text_collapses_to_incorrect(patch_scorer):
    """A doc dict missing `text` must be treated as empty by the scorer.

    This documents the actual failure mode of the adapter bug: the scorer
    sees `(query, "")` pairs and (with a realistic logit-returning model)
    confidence collapses far below any usable threshold.
    """
    # Simulate ms-marco's behaviour on empty docs: strongly-negative logit.
    patch_scorer(lambda q, d: -10.0 if not d else 5.0)
    quality, conf = evaluate_retrieval_quality(
        "q",
        [{"chunk_id": "c1", "paper_id": "p1", "score": 0.99, "rank": 0}],  # no `text`
        CRAGConfig(),
    )
    assert quality is RetrievalQuality.INCORRECT
    assert conf < 0.001  # sigmoid(-10) ~= 4.5e-5


def test_doc_text_present_recovers_correct(patch_scorer):
    """When `text` is present, the same scorer returns a CORRECT verdict."""
    patch_scorer(lambda q, d: -10.0 if not d else 5.0)
    quality, conf = evaluate_retrieval_quality(
        "q",
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "text": "the answer is 42",
                "score": 0.99,
                "rank": 0,
            }
        ],
        CRAGConfig(),
    )
    assert quality is RetrievalQuality.CORRECT
    assert conf > 0.99  # sigmoid(5) ~= 0.993


def test_batch_sigmoid_decision_uses_full_distribution(patch_scorer):
    """A best score in [0,1] must still be sigmoided when other batch scores
    are out of range — the model is returning logits and we should normalise.
    """
    # Best = 0.5 (looks like a probability), but min = -3.0 reveals it's a logit.
    scores = iter([0.5, -3.0, -2.0])
    patch_scorer(lambda q, d: next(scores))
    _, conf = evaluate_retrieval_quality(
        "q",
        [{"text": "a"}, {"text": "b"}, {"text": "c"}],
        CRAGConfig(),
    )
    # sigmoid(0.5) ~= 0.622, NOT 0.5 (which the old per-score heuristic would give).
    assert conf == pytest.approx(1.0 / (1.0 + math.exp(-0.5)), abs=1e-6)
