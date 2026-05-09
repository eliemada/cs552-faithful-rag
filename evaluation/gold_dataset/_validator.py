"""Pure functions for gold-dataset validation, aggregation, and IAA.

CLI wrappers in ``scripts/`` import from here. Keeping logic library-shaped
makes everything trivially unit-testable: no argparse, no I/O glue, no
``sys.exit`` in the call paths under test.

Two validation modes:

* ``validate_fast`` — JSON-Schema + structural checks, all offline. The CI
  workflow runs this on every PR.
* ``validate_strict`` — fast checks + verifies that every span actually
  resolves to its quoted text in the live corpus. Run locally / on RCP
  before opening a PR; cannot run in CI (corpus is on HF, ~3 GB).

Aggregation walks ``contributions/<name>.json``, asserts no cross-member ID
collisions, and emits a stable-sorted ``gold_qa.json``-shaped list.

IAA computes Cohen's κ on the three-class label space (supports /
contradicts / unrelated) over the subset of claims where both annotator
and reviewer have voted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from jsonschema import Draft202012Validator

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCHEMA_PATH: Final[Path] = REPO_ROOT / "evaluation" / "gold_dataset" / "SCHEMA.json"
DEFAULT_ALLOWLIST: Final[Path] = REPO_ROOT / "evaluation" / "gold_dataset" / "paper_ids.txt"
DEFAULT_GOLD_QA: Final[Path] = REPO_ROOT / "evaluation" / "gold_dataset" / "gold_qa.json"
DEFAULT_CONTRIB: Final[Path] = REPO_ROOT / "evaluation" / "gold_dataset" / "contributions"


# ---- result/error types ----------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """One human-readable validation failure, addressable to a pair."""

    message: str
    pair_id: str | None = None

    def as_github_annotation(self, file: Path) -> str:
        return f"::error file={file}::{self.message}"


class AggregationError(Exception):
    """Raised when aggregation cannot produce a deterministic merge."""


class IAAError(Exception):
    """Raised when κ cannot be computed (e.g. mismatched / empty inputs)."""


# ---- validation: fast (offline) -------------------------------------------


def validate_fast(
    pairs: list[dict[str, Any]],
    *,
    allowlist: Path,
    schema_path: Path = SCHEMA_PATH,
) -> list[ValidationError]:
    """Run all offline checks; return every failure (don't short-circuit).

    Returning a list rather than raising lets the CLI emit one annotation per
    error so reviewers see the full picture, not just the first issue.
    """
    errors: list[ValidationError] = []
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)

    # 1. JSON-Schema structural validation.
    for jerr in validator.iter_errors(pairs):
        pair_id = _pair_id_from_path(pairs, jerr.absolute_path)
        errors.append(ValidationError(message=jerr.message, pair_id=pair_id))

    # If schema is broken at the root level, structural invariants below
    # would mostly explode on KeyError — bail out with what we have.
    if errors and not isinstance(pairs, list):
        return errors

    # 2. Cross-pair invariants (only meaningful if the input is list-shaped).
    if isinstance(pairs, list):
        errors.extend(_check_unique_pair_ids(pairs))
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            errors.extend(_check_pair_invariants(pair))
            errors.extend(_check_spans_offline(pair, allowlist=_load_allowlist(allowlist)))

    return errors


def validate_strict(
    pairs: list[dict[str, Any]],
    *,
    allowlist: Path,
    corpus_root: Path,
    schema_path: Path = SCHEMA_PATH,
) -> list[ValidationError]:
    """Fast checks plus quote-against-corpus verification.

    ``corpus_root`` should contain ``processed/<paper_id>/document.md``
    files — same layout the live ``data_loader`` resolves.
    """
    errors = list(validate_fast(pairs, allowlist=allowlist, schema_path=schema_path))
    if not isinstance(pairs, list):
        return errors

    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        errors.extend(_check_spans_strict(pair, corpus_root=corpus_root))
    return errors


# ---- aggregation -----------------------------------------------------------


def aggregate_contributions(contrib_dir: Path) -> list[dict[str, Any]]:
    """Concatenate every ``contributions/<name>.json`` into one stable list.

    Stable sort by ``id`` so the aggregated artifact is deterministic
    regardless of file iteration order on disk.
    """
    seen_ids: dict[str, str] = {}
    all_pairs: list[dict[str, Any]] = []
    for path in sorted(contrib_dir.glob("*.json")):
        data = json.loads(path.read_text() or "[]")
        if not isinstance(data, list):
            raise AggregationError(f"{path.name} is not a JSON array")
        for pair in data:
            pid = pair.get("id") if isinstance(pair, dict) else None
            if pid is None:
                # Let validate_fast catch missing-id; aggregation just keeps it.
                all_pairs.append(pair)
                continue
            if pid in seen_ids:
                raise AggregationError(
                    f"Duplicate pair id {pid!r} in {path.name} and {seen_ids[pid]}"
                )
            seen_ids[pid] = path.name
            all_pairs.append(pair)
    all_pairs.sort(key=lambda p: p.get("id", "") if isinstance(p, dict) else "")
    return all_pairs


def check_aggregate_matches(*, contrib_dir: Path, committed: Path) -> bool:
    """Return ``True`` iff ``committed`` equals the freshly-aggregated artifact."""
    fresh = aggregate_contributions(contrib_dir)
    if not committed.exists():
        return fresh == []
    on_disk = json.loads(committed.read_text() or "[]")
    return on_disk == fresh


# ---- inter-annotator agreement --------------------------------------------


_LABELS: Final[tuple[str, ...]] = ("supports", "contradicts", "unrelated")


def extract_iaa_labels(
    pairs: Iterable[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return paired (annotator, reviewer) labels over the IAA subset.

    Only claims with both labels populated count — partial reviews don't
    contribute. Pairs without ``iaa_subset: true`` are skipped entirely.
    """
    annotator: list[str] = []
    reviewer: list[str] = []
    for pair in pairs:
        if not pair.get("iaa_subset"):
            continue
        for claim in pair.get("claims", []):
            a = claim.get("annotator_label")
            r = claim.get("reviewer_label")
            if a in _LABELS and r in _LABELS:
                annotator.append(a)
                reviewer.append(r)
    return annotator, reviewer


def cohens_kappa(annotator: list[str], reviewer: list[str]) -> float:
    """Cohen's κ on a categorical label space.

    Hand-rolled to avoid pulling scikit-learn into the dev deps for one
    formula. The label set is closed (``_LABELS``) but we accept any string
    for forward compatibility.
    """
    if len(annotator) != len(reviewer):
        raise IAAError(f"length mismatch: {len(annotator)} annotator vs {len(reviewer)} reviewer")
    n = len(annotator)
    if n == 0:
        raise IAAError("cannot compute κ on empty label set")

    labels = sorted(set(annotator) | set(reviewer))
    p_o = sum(1 for a, r in zip(annotator, reviewer) if a == r) / n
    p_e = sum((annotator.count(label) / n) * (reviewer.count(label) / n) for label in labels)
    if p_e == 1.0:
        # All raters always pick the same single label → agreement is trivial.
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ---- internals -------------------------------------------------------------


def _check_pair_invariants(pair: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pid = pair.get("id")

    # Claim ID uniqueness within pair.
    seen: set[str] = set()
    for claim in pair.get("claims", []):
        cid = claim.get("id")
        if cid is None:
            continue
        if cid in seen:
            errors.append(
                ValidationError(
                    message=f"duplicate claim id {cid!r} within pair",
                    pair_id=pid,
                )
            )
        seen.add(cid)

    # answerable ↔ claims invariant.
    difficulty = pair.get("difficulty")
    claims = pair.get("claims", [])
    if difficulty == "unanswerable" and claims:
        errors.append(
            ValidationError(
                message="unanswerable pair must have empty claims[]",
                pair_id=pid,
            )
        )
    elif difficulty in {"single-hop", "multi-hop"} and not claims:
        errors.append(
            ValidationError(
                message="answerable pair must have at least one claim",
                pair_id=pid,
            )
        )

    return errors


def _check_unique_pair_ids(pairs: list[dict[str, Any]]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        pid = pair.get("id")
        if not isinstance(pid, str):
            continue
        if pid in seen:
            errors.append(
                ValidationError(
                    message=f"duplicate pair id {pid!r} across dataset",
                    pair_id=pid,
                )
            )
        seen.add(pid)
    return errors


def _check_spans_offline(pair: dict[str, Any], *, allowlist: set[str]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pid = pair.get("id")
    for claim in pair.get("claims", []):
        for span in claim.get("supporting_spans", []):
            paper = span.get("paper_id")
            if paper not in allowlist:
                errors.append(
                    ValidationError(
                        message=f"paper_id {paper!r} not in allowlist",
                        pair_id=pid,
                    )
                )
            cs = span.get("char_start")
            ce = span.get("char_end")
            if isinstance(cs, int) and isinstance(ce, int) and ce <= cs:
                errors.append(
                    ValidationError(
                        message=f"char_end ({ce}) must be > char_start ({cs})",
                        pair_id=pid,
                    )
                )
    return errors


def _check_spans_strict(pair: dict[str, Any], *, corpus_root: Path) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pid = pair.get("id")
    cache: dict[str, str] = {}
    for claim in pair.get("claims", []):
        for span in claim.get("supporting_spans", []):
            paper = span.get("paper_id")
            cs = span.get("char_start")
            ce = span.get("char_end")
            quote = span.get("quote")
            if not isinstance(paper, str):
                continue
            if not isinstance(cs, int) or not isinstance(ce, int):
                continue
            if not isinstance(quote, str):
                continue

            md = cache.get(paper)
            if md is None:
                md_path = corpus_root / "processed" / paper / "document.md"
                if not md_path.is_file():
                    errors.append(
                        ValidationError(
                            message=f"paper {paper!r}: document.md not found at {md_path}",
                            pair_id=pid,
                        )
                    )
                    continue
                md = _normalise_lf(md_path.read_text())
                cache[paper] = md

            if ce > len(md):
                errors.append(
                    ValidationError(
                        message=(
                            f"span out of bounds in {paper!r}: "
                            f"char_end={ce} but document length is {len(md)}"
                        ),
                        pair_id=pid,
                    )
                )
                continue

            actual = md[cs:ce]
            if actual != quote:
                errors.append(
                    ValidationError(
                        message=(
                            f"quote mismatch in {paper!r} at [{cs}:{ce}]: "
                            f"expected {_truncate(quote)!r}, got {_truncate(actual)!r}"
                        ),
                        pair_id=pid,
                    )
                )
    return errors


def _normalise_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _truncate(s: str, limit: int = 60) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _pair_id_from_path(pairs: Any, path: Iterable[Any]) -> str | None:
    """Walk a jsonschema error path back to the enclosing pair's ``id``."""
    parts = list(path)
    if not parts or not isinstance(pairs, list):
        return None
    idx = parts[0]
    if not isinstance(idx, int) or idx >= len(pairs):
        return None
    item = pairs[idx]
    return item.get("id") if isinstance(item, dict) else None


def _load_allowlist(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}
