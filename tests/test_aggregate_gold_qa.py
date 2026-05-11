"""Tests for the contributions → gold_qa.json aggregator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from evaluation.gold_dataset._validator import (
    AggregationError,
    aggregate_contributions,
    check_aggregate_matches,
)


def test_aggregate_two_members_stable_order(
    write_contributions: Callable[..., Path],
    make_pair: Callable[..., dict[str, Any]],
) -> None:
    contrib = write_contributions(
        {
            "andrea": [make_pair(id="q100", annotator="andrea")],
            "elie": [make_pair(id="q001", annotator="elie")],
        }
    )
    aggregated = aggregate_contributions(contrib)
    # Sorted by id — q001 before q100 regardless of file iteration order.
    assert [p["id"] for p in aggregated] == ["q001", "q100"]


def test_aggregate_rejects_id_collision_across_members(
    write_contributions: Callable[..., Path],
    make_pair: Callable[..., dict[str, Any]],
) -> None:
    contrib = write_contributions(
        {
            "elie": [make_pair(id="q001", annotator="elie")],
            "andrea": [make_pair(id="q001", annotator="andrea")],
        }
    )
    with pytest.raises(AggregationError, match="q001"):
        aggregate_contributions(contrib)


def test_check_passes_when_aggregate_matches_committed(
    write_contributions: Callable[..., Path],
    make_pair: Callable[..., dict[str, Any]],
    tmp_path: Path,
) -> None:
    contrib = write_contributions({"elie": [make_pair(id="q001", annotator="elie")]})
    committed = tmp_path / "gold_qa.json"
    committed.write_text(json.dumps(aggregate_contributions(contrib), indent=2) + "\n")
    assert check_aggregate_matches(contrib_dir=contrib, committed=committed) is True


def test_check_fails_when_aggregate_drifts(
    write_contributions: Callable[..., Path],
    make_pair: Callable[..., dict[str, Any]],
    tmp_path: Path,
) -> None:
    contrib = write_contributions({"elie": [make_pair(id="q001", annotator="elie")]})
    committed = tmp_path / "gold_qa.json"
    committed.write_text("[]\n")  # stale
    assert check_aggregate_matches(contrib_dir=contrib, committed=committed) is False


def test_aggregate_empty_dir_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "contributions").mkdir()
    assert aggregate_contributions(tmp_path / "contributions") == []
