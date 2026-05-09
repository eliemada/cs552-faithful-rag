# Gold evaluation dataset (v0.1)

Small, expert-annotated benchmark for citation-faithful RAG. Target: 50
validated Q&A pairs by **2026-05-16** for M2.

## Files

| File                                | What it is                                     |
|-------------------------------------|------------------------------------------------|
| [`SCHEMA.json`](./SCHEMA.json)      | JSON Schema 2020-12, the canonical spec.       |
| [`SCHEMA.md`](./SCHEMA.md)          | Human walkthrough of the schema.               |
| [`RUBRIC.md`](./RUBRIC.md)          | Annotation guide. **Read first.**              |
| [`paper_ids.txt`](./paper_ids.txt)  | Allowlist snapshot for offline CI.             |
| `contributions/<name>.json`         | Per-member sources of truth.                   |
| `gold_qa.json`                      | Aggregated artifact (CI keeps in sync).        |
| `reviews/`                          | IAA review notes — see folder README.          |

## Workflow

```bash
# 1. edit your own contribution
$EDITOR evaluation/gold_dataset/contributions/<your-name>.json

# 2. validate locally with full corpus access
uv run python scripts/validate_gold_qa.py --strict

# 3. refresh the aggregated artifact
uv run python scripts/aggregate_gold_qa.py

# 4. open a PR — CI runs the fast (offline) validator
git add evaluation/gold_dataset/
git commit -m "feat(gold): add 3 pairs (q010-q012) on patent reform impact"
```

## What CI checks (fast / offline)

- JSON Schema compliance
- Question / claim ID uniqueness and namespacing
- `paper_id` ∈ `paper_ids.txt`
- `char_end > char_start ≥ 0`, non-empty `quote`
- `unanswerable` ↔ empty `claims[]` invariant
- `gold_qa.json == aggregate(contributions/*)`

## What you must check locally (--strict)

- `markdown[char_start:char_end] == quote` after LF normalisation

CI cannot do this — the corpus is 3 GB on HF and not on the runner.

## Targets (M2)

- Per person: 13 pairs (≥ 1 unanswerable, ≥ 2 multi-hop, ≥ 5 in IAA subset).
- Total: 52 pairs by 2026-05-16, frozen as `gold_qa_v1`.
- IAA: Cohen's κ ≥ 0.6 on the IAA subset before reporting headline numbers.

## Schema versioning

This is **v0.1**. Breaking changes get a new file (`SCHEMA.v0.2.json`) and a
fresh PR. Don't mutate v0.1 once the M2 freeze is in.
