"""Tests for the RAGAS evaluation glue.

We don't unit-test RAGAS itself (the library is heavily tested upstream) and
we don't hit live LLM / embedding APIs in tests (cost + flakiness). What we
do cover:

* :func:`pipelines.run_rag_pipeline` returns the right shape with a stub
  retriever and stub LLM
* :func:`pipelines.run_long_context_pipeline` concatenates papers correctly
  and routes through the configured LLM
* :func:`pipelines.build_chunk_lookup` maps chunk IDs to text from the
  aggregated metadata file
* :func:`ragas_runner.evaluate_samples` short-circuits empty input without
  trying to allocate an LLM
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.ragas_eval import pipelines
from evaluation.ragas_eval.pipelines import (
    RagasSample,
    build_chunk_lookup,
    run_long_context_pipeline,
    run_rag_pipeline,
)
from evaluation.ragas_eval.ragas_runner import METRIC_NAMES, evaluate_samples


class _StubRetriever:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search(self, query: str, k: int) -> list[dict]:
        return self._hits[:k]


def _patch_generate(monkeypatch: pytest.MonkeyPatch, response: str) -> list[str]:
    """Replace pipelines.generate with a deterministic stub. Returns the call log."""
    calls: list[str] = []

    def fake_generate(model: str, prompt: str, **kw):
        calls.append(prompt)
        return response

    monkeypatch.setattr(pipelines, "generate", fake_generate)
    return calls


def test_rag_sample_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _patch_generate(monkeypatch, "stub answer")
    retriever = _StubRetriever(
        [
            {"chunk_id": "c1", "paper_id": "p", "score": 0.9, "rank": 0},
            {"chunk_id": "c2", "paper_id": "p", "score": 0.7, "rank": 1},
        ]
    )
    chunk_lookup = {"c1": "passage one", "c2": "passage two", "c3": "unrelated"}

    sample = run_rag_pipeline(
        question="What is X?",
        ground_truth="X is a thing.",
        query_id="q001",
        retriever=retriever,
        chunk_lookup=chunk_lookup,
        top_k=2,
    )

    assert sample.pipeline == "rag"
    assert sample.query_id == "q001"
    assert sample.contexts == ["passage one", "passage two"]
    assert sample.answer == "stub answer"
    assert sample.ground_truth == "X is a thing."
    assert sample.to_ragas_dict() == {
        "question": "What is X?",
        "answer": "stub answer",
        "contexts": ["passage one", "passage two"],
        "ground_truth": "X is a thing.",
    }
    # The LLM should see both retrieved passages in the prompt.
    assert "passage one" in prompts[0]
    assert "passage two" in prompts[0]
    assert "What is X?" in prompts[0]


def test_rag_skips_missing_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retriever can return a chunk_id that isn't in the lookup; we drop it."""
    _patch_generate(monkeypatch, "ok")
    retriever = _StubRetriever(
        [
            {"chunk_id": "missing", "paper_id": "p", "score": 0.9, "rank": 0},
            {"chunk_id": "c1", "paper_id": "p", "score": 0.7, "rank": 1},
        ]
    )
    chunk_lookup = {"c1": "the only known passage"}

    sample = run_rag_pipeline(
        question="Q",
        ground_truth="GT",
        query_id="qX",
        retriever=retriever,
        chunk_lookup=chunk_lookup,
        top_k=5,
    )
    assert sample.contexts == ["the only known passage"]


def test_long_context_reads_paper_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _patch_generate(monkeypatch, "lc answer")
    # Build a fake processed-papers tree
    paper_dir = tmp_path / "paperA"
    paper_dir.mkdir()
    (paper_dir / "document.md").write_text("Full text of paper A. Body body body.")
    paper_dir_b = tmp_path / "paperB"
    paper_dir_b.mkdir()
    (paper_dir_b / "document.md").write_text("Full text of paper B.")

    sample = run_long_context_pipeline(
        question="Compare A and B",
        ground_truth="A and B differ in body content.",
        query_id="q-lc",
        paper_ids=["paperA", "paperB"],
        processed_dir=tmp_path,
    )

    assert sample.pipeline == "long_context"
    assert len(sample.contexts) == 2
    assert sample.contexts[0].startswith("Full text of paper A.")
    assert sample.contexts[1] == "Full text of paper B."
    assert "Full text of paper A" in prompts[0]
    assert "Full text of paper B" in prompts[0]


def test_long_context_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_generate(monkeypatch, "ok")
    paper_dir = tmp_path / "paperX"
    paper_dir.mkdir()
    (paper_dir / "document.md").write_text("A" * 1000)

    sample = run_long_context_pipeline(
        question="Q",
        ground_truth="GT",
        query_id="qT",
        paper_ids=["paperX"],
        processed_dir=tmp_path,
        max_chars_per_paper=50,
    )
    assert len(sample.contexts[0]) == 50


def test_build_chunk_lookup(tmp_path: Path) -> None:
    # Aggregated metadata format: dict[str index → chunk record]
    raw = {
        "0": {
            "chunk_id": "p1_coarse_0000",
            "paper_id": "p1",
            "text": "alpha",
            "char_start": 0,
            "char_end": 5,
        },
        "1": {
            "chunk_id": "p1_coarse_0001",
            "paper_id": "p1",
            "text": "beta",
            "char_start": 5,
            "char_end": 9,
        },
    }
    path = tmp_path / "coarse_metadata.json"
    path.write_text(json.dumps(raw))
    lookup = build_chunk_lookup(path)
    assert lookup == {"p1_coarse_0000": "alpha", "p1_coarse_0001": "beta"}


def test_evaluate_samples_empty_short_circuits() -> None:
    """Empty input must not try to construct an LLM (so it runs without API keys)."""
    result = evaluate_samples([])
    assert result.n == 0
    assert set(result.aggregate) == set(METRIC_NAMES)
    assert all(v == 0.0 for v in result.aggregate.values())
    assert result.per_sample == []


def test_ragas_sample_to_dict_strips_provenance() -> None:
    """Provenance fields (pipeline, query_id) must not leak into RAGAS' dataset."""
    s = RagasSample(
        question="q",
        answer="a",
        contexts=["c1"],
        ground_truth="g",
        pipeline="rag",
        query_id="q001",
    )
    d = s.to_ragas_dict()
    assert set(d) == {"question", "answer", "contexts", "ground_truth"}
