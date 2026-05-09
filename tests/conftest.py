"""Shared fixtures for the gold-dataset validator tests.

Fixtures here build small in-memory QA pairs and a tiny mock corpus so
unit tests stay isolated from the real 999-paper HF corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def make_span() -> Callable[..., dict[str, Any]]:
    """Build a span dict with sane defaults; override any field via kwargs."""

    def _build(
        *,
        paper_id: str = "00002_W2122361802",
        char_start: int = 100,
        char_end: int = 150,
        quote: str = "verbatim slice from the document",
    ) -> dict[str, Any]:
        return {
            "paper_id": paper_id,
            "char_start": char_start,
            "char_end": char_end,
            "quote": quote,
        }

    return _build


@pytest.fixture
def make_claim(make_span: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    def _build(
        *,
        id: str = "c001",
        text: str = "An atomic verifiable claim.",
        supporting_spans: list[dict[str, Any]] | None = None,
        annotator_label: str | None = "supports",
        reviewer_label: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": id,
            "text": text,
            "supporting_spans": [make_span()] if supporting_spans is None else supporting_spans,
            "annotator_label": annotator_label,
            "reviewer_label": reviewer_label,
        }

    return _build


@pytest.fixture
def make_pair(make_claim: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    def _build(
        *,
        id: str = "q001",
        question: str = "What is X?",
        gold_answer: str = "X is Y.",
        difficulty: str = "single-hop",
        category: str = "factual",
        annotator: str = "elie",
        reviewer: str | None = None,
        review_status: str = "pending",
        iaa_subset: bool = False,
        claims: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": id,
            "question": question,
            "gold_answer": gold_answer,
            "difficulty": difficulty,
            "category": category,
            "annotator": annotator,
            "reviewer": reviewer,
            "review_status": review_status,
            "iaa_subset": iaa_subset,
            "claims": [make_claim()] if claims is None else claims,
        }

    return _build


@pytest.fixture
def paper_id_allowlist(tmp_path: Path) -> Path:
    """Tiny paper_ids.txt covering only the IDs used in the test corpus."""
    path = tmp_path / "paper_ids.txt"
    path.write_text("00002_W2122361802\n00004_W2114989862\n")
    return path


@pytest.fixture
def fake_corpus(tmp_path: Path) -> Path:
    """Mock corpus directory with two `processed/<id>/document.md` files.

    Returns the corpus root; tests can monkeypatch ``artifact_root`` to point
    here when running --strict-mode validation.
    """
    root = tmp_path / "corpus"
    (root / "processed" / "00002_W2122361802").mkdir(parents=True)
    (root / "processed" / "00004_W2114989862").mkdir(parents=True)
    md1 = "0123456789" * 20  # 200 chars
    md1 = md1[:100] + "verbatim slice from the document" + md1[100:]
    (root / "processed" / "00002_W2122361802" / "document.md").write_text(md1)
    (root / "processed" / "00004_W2114989862" / "document.md").write_text("short paper " * 50)
    return root


@pytest.fixture
def write_contributions(tmp_path: Path) -> Callable[..., Path]:
    """Write one or more `contributions/<name>.json` files; returns dir."""

    def _write(files: dict[str, list[dict[str, Any]]]) -> Path:
        contrib = tmp_path / "contributions"
        contrib.mkdir(exist_ok=True)
        for name, pairs in files.items():
            (contrib / f"{name}.json").write_text(json.dumps(pairs, indent=2))
        return contrib

    return _write
