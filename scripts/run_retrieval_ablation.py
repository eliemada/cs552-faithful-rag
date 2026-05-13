"""Run every retriever config in the ablation and emit a comparison table.

Reads per-config result JSONs from
``evaluation/retrieval_eval/results/<config>.json`` (produced by
``evaluation.retrieval_eval.evaluate_retrieval``), aggregates them into a
single side-by-side table at both granularities (paper-level + chunk-level),
and writes both ``comparison.json`` and a human-readable ``comparison.md``.

If a per-config JSON is missing, this script can also drive the runs itself
(``--run-missing``) — useful when re-doing an ablation from a clean state.

Run order is fixed (coarse_faiss → coarse_rerank → fine_faiss → fine_rerank)
so the resulting Markdown table is reproducible across re-runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.gold_dataset._validator import REPO_ROOT
from evaluation.retrieval_eval.retrievers import CONFIGS

RESULTS_DIR = REPO_ROOT / "evaluation" / "retrieval_eval" / "results"
DEFAULT_KS = (5, 10, 20)


def _run_config(name: str) -> None:
    """Invoke the evaluate_retrieval CLI for one config."""
    print(f"Running {name} ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "evaluation.retrieval_eval.evaluate_retrieval",
         "--config", name, "--quiet"],
        cwd=REPO_ROOT,
        check=True,
    )


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _row_for(config_name: str, agg: dict, granularity: str, ks: tuple[int, ...]) -> list[str]:
    g = agg[granularity]
    n = g.get("n", 0)
    cells = [config_name, f"{n}"]
    for k in ks:
        cells.append(_fmt(g.get(f"hit_rate@{k}")))
    cells.append(_fmt(g.get(f"precision@{k}")))  # noqa: B023 — last k value
    cells.append(_fmt(g.get(f"recall@{k}")))     # noqa: B023
    cells.append(_fmt(g.get("mrr")))
    return cells


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _per_category_table(per_config: dict[str, dict], granularity: str) -> str:
    """Hit-rate@10 per category × per config — secondary breakdown."""
    # Collect categories across all configs (should be identical, but be safe)
    by_cat_config: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cfg_name, result in per_config.items():
        for q in result["per_query"]:
            metric = q[granularity].get("hit_rate@10")
            if metric is not None:
                by_cat_config[q["category"]][cfg_name].append(metric)

    categories = sorted(by_cat_config)
    if not categories:
        return "_(no data at this granularity)_"

    headers = ["category", "n"] + [c.name for c in CONFIGS]
    rows: list[list[str]] = []
    for cat in categories:
        cells = [cat]
        # n is the same across configs at the paper level (paper coverage is universal);
        # at chunk level it can differ slightly — use the first config that has data.
        any_cfg = next((c.name for c in CONFIGS if by_cat_config[cat].get(c.name)), None)
        n = len(by_cat_config[cat][any_cfg]) if any_cfg else 0
        cells.append(str(n))
        for cfg in CONFIGS:
            vals = by_cat_config[cat].get(cfg.name, [])
            cells.append(_fmt(sum(vals) / len(vals)) if vals else "—")
        rows.append(cells)
    return _markdown_table(rows, headers)


def build_comparison(results_dir: Path, ks: tuple[int, ...]) -> dict:
    per_config: dict[str, dict] = {}
    for cfg in CONFIGS:
        path = results_dir / f"{cfg.name}.json"
        if path.is_file():
            per_config[cfg.name] = json.loads(path.read_text())

    return {
        "configs": [c.name for c in CONFIGS if c.name in per_config],
        "k_values": list(ks),
        "results": per_config,
    }


def render_markdown(comparison: dict, ks: tuple[int, ...]) -> str:
    per_config = comparison["results"]
    last_k = ks[-1]

    def section(granularity: str, title: str) -> str:
        headers = ["config", "n"] + [f"hit@{k}" for k in ks] + [f"P@{last_k}", f"R@{last_k}", "MRR"]
        rows = []
        for cfg in CONFIGS:
            if cfg.name not in per_config:
                continue
            rows.append(_row_for(cfg.name, per_config[cfg.name]["aggregate"], granularity, ks))
        return f"### {title}\n\n{_markdown_table(rows, headers)}"

    paper_table = section("paper", "Paper-level retrieval (primary)")
    chunk_table = section("chunk", "Chunk-level retrieval (secondary — coverage subset)")
    paper_per_cat = _per_category_table(per_config, "paper")

    return f"""# Retrieval ablation — M2

Pre-built FAISS indices over OpenAI `text-embedding-3-small` embeddings.
Four configurations spanning chunk granularity × ±ZeroEntropy reranker.

{paper_table}

`n` is the number of evaluable queries. At paper level every gold pair contributes; at
chunk level only queries whose gold span overlaps at least one chunk at the relevant
granularity contribute (~50 % of queries due to the existing chunker's coverage gaps —
see `evaluation/retrieval_eval/gold_resolver.py`).

{chunk_table}

### Paper-level hit@10 by question category

{paper_per_cat}

---

Numbers above are produced by `scripts/run_retrieval_ablation.py`; per-config
detail (per-query metrics, latency, gold-set sizes) lives in
`evaluation/retrieval_eval/results/<config>.json`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--run-missing", action="store_true",
                        help="Invoke evaluate_retrieval for any config that has no result JSON yet.")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    args = parser.parse_args(argv)
    ks = tuple(args.ks)

    if args.run_missing:
        for cfg in CONFIGS:
            if not (args.results_dir / f"{cfg.name}.json").is_file():
                _run_config(cfg.name)

    comparison = build_comparison(args.results_dir, ks)
    if not comparison["results"]:
        print("No per-config results found. Run with --run-missing or run "
              "`evaluate_retrieval` first.", file=sys.stderr)
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "comparison.json"
    md_path = args.results_dir / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2))
    md_path.write_text(render_markdown(comparison, ks))

    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"  configs included: {comparison['configs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
