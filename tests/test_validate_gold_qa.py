"""Tests for the gold-dataset validator (fast and strict modes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


from evaluation.gold_dataset._validator import (
    validate_fast,
    validate_strict,
)


def test_empty_dataset_is_valid(paper_id_allowlist: Path) -> None:
    errors = validate_fast([], allowlist=paper_id_allowlist)
    assert errors == []


def test_minimal_valid_pair(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    errors = validate_fast([make_pair()], allowlist=paper_id_allowlist)
    assert errors == []


def test_missing_required_question_field_fails(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    pair = make_pair()
    del pair["question"]
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("question" in e.message for e in errors)


def test_unknown_category_enum_fails(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    pair = make_pair(category="made_up")
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("category" in e.message or "made_up" in e.message for e in errors)


def test_duplicate_question_ids_across_dataset_fail(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    a = make_pair(id="q001", annotator="elie")
    b = make_pair(id="q001", annotator="andrea")
    errors = validate_fast([a, b], allowlist=paper_id_allowlist)
    assert any("duplicate" in e.message.lower() and "q001" in e.message for e in errors)


def test_duplicate_claim_ids_within_pair_fail(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
) -> None:
    pair = make_pair(claims=[make_claim(id="c001"), make_claim(id="c001")])
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("duplicate" in e.message.lower() and "c001" in e.message for e in errors)


def test_unanswerable_with_claims_fails(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    pair = make_pair(difficulty="unanswerable")
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("unanswerable" in e.message.lower() for e in errors)


def test_answerable_without_claims_fails(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    pair = make_pair(claims=[])
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("at least one claim" in e.message.lower() for e in errors)


def test_paper_id_not_in_allowlist_fails(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
) -> None:
    pair = make_pair(claims=[make_claim(supporting_spans=[make_span(paper_id="99999_WUNKNOWN")])])
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("99999_WUNKNOWN" in e.message for e in errors)


def test_char_start_after_char_end_fails(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
) -> None:
    pair = make_pair(
        claims=[make_claim(supporting_spans=[make_span(char_start=200, char_end=100)])]
    )
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any("char_end" in e.message and "char_start" in e.message for e in errors)


def test_empty_quote_fails(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
) -> None:
    pair = make_pair(claims=[make_claim(supporting_spans=[make_span(quote="")])])
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert errors  # at least one


def test_strict_mode_quote_matches_corpus(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
    fake_corpus: Path,
) -> None:
    span = make_span(
        paper_id="00002_W2122361802",
        char_start=100,
        char_end=132,
        quote="verbatim slice from the document",
    )
    pair = make_pair(claims=[make_claim(supporting_spans=[span])])
    errors = validate_strict([pair], allowlist=paper_id_allowlist, corpus_root=fake_corpus)
    assert errors == []


def test_strict_mode_quote_mismatch_fails(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
    fake_corpus: Path,
) -> None:
    span = make_span(
        paper_id="00002_W2122361802",
        char_start=100,
        char_end=132,
        quote="this is not what the document says",
    )
    pair = make_pair(claims=[make_claim(supporting_spans=[span])])
    errors = validate_strict([pair], allowlist=paper_id_allowlist, corpus_root=fake_corpus)
    assert any("quote" in e.message.lower() for e in errors)


def test_strict_mode_span_out_of_bounds_fails(
    make_pair: Callable[..., dict[str, Any]],
    make_claim: Callable[..., dict[str, Any]],
    make_span: Callable[..., dict[str, Any]],
    paper_id_allowlist: Path,
    fake_corpus: Path,
) -> None:
    span = make_span(
        paper_id="00002_W2122361802",
        char_start=100_000,
        char_end=100_050,
        quote="anything",
    )
    pair = make_pair(claims=[make_claim(supporting_spans=[span])])
    errors = validate_strict([pair], allowlist=paper_id_allowlist, corpus_root=fake_corpus)
    assert any("out of bounds" in e.message.lower() for e in errors)


def test_validation_error_includes_pair_id(
    make_pair: Callable[..., dict[str, Any]], paper_id_allowlist: Path
) -> None:
    pair = make_pair(id="q042", category="made_up")
    errors = validate_fast([pair], allowlist=paper_id_allowlist)
    assert any(e.pair_id == "q042" for e in errors)
