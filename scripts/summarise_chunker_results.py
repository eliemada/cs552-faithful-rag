"""Summarise the M3 chunker-ablation results into a markdown table + CSV.

Reads:
  evaluation/retrieval_eval/results/e5_large_coarse_rerank.json  (M2 baseline anchor)
  evaluation/retrieval_eval/results/e5_rerank_<variant>.json     (9 M3 variants)

Writes:
  evaluation/retrieval_eval/results/chunker_summary.md
  evaluation/retrieval_eval/results/chunker_summary.csv

Run from repo root:
  uv run python -m scripts.summarise_chunker_results
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "evaluation" / "retrieval_eval" / "results"

VARIANTS: tuple[str, ...] = (
    "s200_o0",
    "s200_o100",
    "s400_o0",
    "s400_o200",
    "s600_o0",
    "s600_o300",
    "s800_o0",
    "s800_o400",
    "recursive_400",
)


@dataclass(frozen=True)
class Row:
    config: str
    n: int
    hit_at_5: float
    hit_at_10: float
    mrr: float
    ndcg_at_10: float

    def fmt(self) -> list[str]:
        return [
            self.config,
            str(self.n),
            f"{self.hit_at_5:.3f}",
            f"{self.hit_at_10:.3f}",
            f"{self.mrr:.3f}",
            f"{self.ndcg_at_10:.3f}",
        ]


def _row(path: Path, label: str) -> Row | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    paper = data["aggregate"]["paper"]
    return Row(
        config=label,
        n=paper["n"],
        hit_at_5=paper["hit_rate@5"],
        hit_at_10=paper["hit_rate@10"],
        mrr=paper["mrr"],
        ndcg_at_10=paper["ndcg@10"],
    )


def collect() -> list[Row]:
    rows: list[Row] = []
    baseline = _row(
        RESULTS_DIR / "e5_large_coarse_rerank.json", "e5_large_coarse_rerank (M2 baseline)"
    )
    if baseline is not None:
        rows.append(baseline)
    for v in VARIANTS:
        row = _row(RESULTS_DIR / f"e5_rerank_{v}.json", f"e5_rerank_{v}")
        if row is not None:
            rows.append(row)
    return rows


def to_markdown(rows: list[Row]) -> str:
    headers = ["config", "n", "hit@5", "hit@10", "MRR", "nDCG@10"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r.fmt()) + " |")
    return "\n".join(out) + "\n"


def to_csv(rows: list[Row], path: Path) -> None:
    headers = ["config", "n", "hit@5", "hit@10", "MRR", "nDCG@10"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r.fmt())


def highlights(rows: list[Row]) -> str:
    if not rows:
        return "_no results on disk._"
    baseline = next((r for r in rows if "M2 baseline" in r.config), None)
    variants = [r for r in rows if "M2 baseline" not in r.config]
    if not variants:
        return "_only baseline present._"
    best_mrr = max(variants, key=lambda r: r.mrr)
    best_hit10 = max(variants, key=lambda r: r.hit_at_10)
    best_ndcg = max(variants, key=lambda r: r.ndcg_at_10)
    parts = [
        f"Best MRR:      **{best_mrr.config}**  MRR={best_mrr.mrr:.3f}  hit@10={best_mrr.hit_at_10:.3f}",
        f"Best hit@10:   **{best_hit10.config}**  hit@10={best_hit10.hit_at_10:.3f}  MRR={best_hit10.mrr:.3f}",
        f"Best nDCG@10:  **{best_ndcg.config}**  nDCG@10={best_ndcg.ndcg_at_10:.3f}",
    ]
    if baseline is not None:
        delta_mrr = best_mrr.mrr - baseline.mrr
        delta_hit = best_hit10.hit_at_10 - baseline.hit_at_10
        parts.append(
            f"\nΔ vs baseline ({baseline.config}, n={baseline.n}):  "
            f"MRR {delta_mrr:+.3f},  hit@10 {delta_hit:+.3f}"
        )
    return "\n".join(parts)


def main() -> None:
    rows = collect()
    md_path = RESULTS_DIR / "chunker_summary.md"
    csv_path = RESULTS_DIR / "chunker_summary.csv"

    md_path.write_text(to_markdown(rows) + "\n" + highlights(rows) + "\n")
    to_csv(rows, csv_path)

    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print()
    print(to_markdown(rows))
    print(highlights(rows))


if __name__ == "__main__":
    main()
