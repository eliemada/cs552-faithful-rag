"""Tests for Cohen's κ on inter-annotator-agreement labels."""

from __future__ import annotations

import math

import pytest

from evaluation.gold_dataset._validator import (
    IAAError,
    cohens_kappa,
    extract_iaa_labels,
)


def test_perfect_agreement_kappa_one() -> None:
    annotator = ["supports", "contradicts", "unrelated", "supports"]
    reviewer = ["supports", "contradicts", "unrelated", "supports"]
    assert cohens_kappa(annotator, reviewer) == pytest.approx(1.0)


def test_complete_disagreement_kappa_negative() -> None:
    annotator = ["supports", "supports", "contradicts", "contradicts"]
    reviewer = ["contradicts", "contradicts", "supports", "supports"]
    kappa = cohens_kappa(annotator, reviewer)
    assert kappa < 0


def test_chance_level_kappa_near_zero() -> None:
    # Two independent label streams with the same marginal distribution.
    annotator = ["supports", "contradicts"] * 50
    reviewer = ["supports", "supports", "contradicts", "contradicts"] * 25
    kappa = cohens_kappa(annotator, reviewer)
    assert math.isclose(kappa, 0.0, abs_tol=0.05)


def test_empty_labels_raises() -> None:
    with pytest.raises(IAAError, match="empty"):
        cohens_kappa([], [])


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(IAAError, match="length"):
        cohens_kappa(["supports"], ["supports", "contradicts"])


def test_extract_iaa_labels_only_includes_paired_labels() -> None:
    pairs = [
        {
            "id": "q001",
            "iaa_subset": True,
            "claims": [
                {"id": "c001", "annotator_label": "supports", "reviewer_label": "supports"},
                {"id": "c002", "annotator_label": "supports", "reviewer_label": None},
            ],
        },
        {
            "id": "q002",
            "iaa_subset": False,
            "claims": [
                {"id": "c001", "annotator_label": "supports", "reviewer_label": "supports"},
            ],
        },
    ]
    a, r = extract_iaa_labels(pairs)
    assert a == ["supports"]
    assert r == ["supports"]
