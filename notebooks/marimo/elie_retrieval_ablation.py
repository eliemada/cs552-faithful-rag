"""Elie Bruno — Retrieval Ablation for Faithful RAG (CS-552 M2)

Run with:
    uv run marimo edit notebooks/marimo/elie_retrieval_ablation.py
    # or, headless:
    uv run marimo run  notebooks/marimo/elie_retrieval_ablation.py

This notebook is the individual-contribution write-up for my proposal-owned
component (retrieval ablation) plus two cross-cutting items I led on the
team (adversarial controls for the κ paradox, and the chunker-coverage
diagnosis that connects retrieval and end-to-end results).

Per TA Madhur's clarification on the project Ed thread: shared modules are
referenced and explained, not copy-pasted. The notebook focuses on *my*
design choices, results, and analysis.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
# Retrieval Ablation: Embedders, Chunk Granularity, and Reranking

**Elie Bruno** &middot; SCIPER 355932 &middot; CS-552 Spring 2026 &middot;
Team Faithful RAG &middot; M2 Progress

---

## Question

How much retrieval quality does chunk granularity cost us, and how much
does a cross-encoder reranker recover, on a domain-specific scientific
corpus? The proposal commits to four embedders (OpenAI
`text-embedding-3-small`, BGE-M3, E5-large, ColBERTv2) crossed with
coarse/fine chunks and ±reranker. For M2 I ship the slice that is
already indexed end-to-end: OpenAI embeddings on coarse and fine chunks
with and without the ZeroEntropy reranker, giving four configurations.
The three remaining embedders move to M3 and are listed at the bottom.

## What I built

* `evaluation/retrieval_eval/gold_resolver.py` bridges the gold dataset's
  char-span ground truth `(paper_id, char_start, char_end)` and the
  retriever's chunk-id output. I picked **any-character-overlap** over
  strict containment because fine chunks (~300 chars) are shorter than
  most multi-sentence gold claims; strict containment would empty the
  gold-chunk set for most queries.
* `evaluation/retrieval_eval/retrievers.py` wraps
  `HybridRetriever.from_path()` into 4 named configs and lazy-loads the
  FAISS indices so the 1.4 GB on disk is paid only when a config runs.
* `evaluation/retrieval_eval/evaluate_retrieval.py` is the CLI that loops
  one config over the resolved queries and writes per-query and
  aggregate JSON. It computes metrics at two granularities (paper and
  chunk) because the chunk-level signal is only well-defined on a
  subset.

The notebook below loads the per-config JSONs that those scripts produce.
One command to reproduce everything end-to-end:

```bash
uv run python -m scripts.run_retrieval_ablation --run-missing
```
"""
    )
    return


@app.cell
def __(mo):
    """Wire `sys.path` so the in-repo `evaluation.*` packages import cleanly."""
    import json
    import sys
    from pathlib import Path

    repo_root = Path.cwd().resolve()
    while not (repo_root / "pyproject.toml").exists():
        if repo_root.parent == repo_root:
            raise RuntimeError("Repo root not found above notebook.")
        repo_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    mo.md(f"Repo root resolved at `{repo_root}`.")
    return (json, repo_root)


@app.cell
def __(mo):
    mo.md(
        r"""
## Design choice: paper-level vs chunk-level evaluation

Building the gold-span to chunk-id resolver, I audited the existing
semantic chunker against the 15 papers the gold set cites. Coarse chunks
cover 18 to 59 % of document text (mean ≈ 43 %). Fine chunks cover 12 to
35 % (mean ≈ 24 %). The chunker drops everything between section
boundaries, and many gold spans land in those gaps. Concretely: 17 of
37 queries have at least one gold match at the coarse granularity, 16
at the fine granularity. The other ~20 queries have no chunk for the
retriever to hit even in principle.

Two methodology decisions follow:

1. **Paper-level retrieval is the primary metric.** A retrieved chunk
   counts as a hit when its `paper_id` belongs to the gold paper set
   for the question. Defined for all 37 evaluable queries.
2. **Chunk-level retrieval is a secondary metric**, reported only on
   the `has_chunk_coverage` subset (~50 % of queries). Mixing it into
   the headline would quietly degrade the numbers.

Rebuilding the chunker with a no-gap sliding-window scheme is the
highest-leverage M3 task. It is off the M2 critical path because it
triggers a 46k-chunk FAISS re-index.
"""
    )
    return


@app.cell
def __(json, repo_root):
    """Load the four per-config result JSONs."""
    results_dir = repo_root / "evaluation" / "retrieval_eval" / "results"
    per_config = {
        path.stem: json.loads(path.read_text())
        for path in sorted(results_dir.glob("*.json"))
        if path.stem != "comparison"
    }
    config_order = ["coarse_faiss", "coarse_rerank", "fine_faiss", "fine_rerank"]
    return (config_order, per_config)


@app.cell
def __(mo):
    mo.md(r"""## Headline numbers (paper-level)""")
    return


@app.cell
def __(config_order, per_config):
    """Build the headline-table rows. All loop-locals stay inside the function."""

    def build_headline_rows():
        rows = []
        for cfg in config_order:
            a = per_config[cfg]["aggregate"]["paper"]
            rows.append(
                {
                    "config": cfg,
                    "n": a["n"],
                    "hit@5": round(a["hit_rate@5"], 3),
                    "hit@10": round(a["hit_rate@10"], 3),
                    "hit@20": round(a["hit_rate@20"], 3),
                    "P@20": round(a["precision@20"], 3),
                    "R@20": round(a["recall@20"], 3),
                    "MRR": round(a["mrr"], 3),
                }
            )
        return rows

    headline_rows = build_headline_rows()
    return (headline_rows,)


@app.cell
def __(headline_rows, mo):
    import pandas as pd

    headline_df = pd.DataFrame(headline_rows).set_index("config")
    headline_table = mo.ui.table(headline_df, selection=None)
    headline_table
    return (pd,)


@app.cell
def __(mo):
    mo.md(
        r"""
**Reading the table.**

* Reranking helps every config. Coarse moves from hit@5 = 0.784 to
  0.946 (+16 pp). Fine moves from 0.784 to 0.865 (+8 pp).
* coarse_rerank wins MRR (0.842). The right paper sits at rank 1 in
  84 % of queries, the metric to optimise for a downstream that only
  consumes the top result.
* fine_rerank wins hit@10 (0.973). The right paper is somewhere in the
  top 10 for 36 of 37 queries. Different best depending on whether the
  downstream cares about rank 1 or set membership.
* The queries coarse_faiss misses at top-10 (q004, q011, q017, q026,
  q212) appear in the failure-mode cell below.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Bootstrap confidence intervals

Point estimates over 37 queries leave room for sampling noise: a
two-point gap can be inside the bootstrap band. Below I draw 1000
resamples per metric and report a 95 % CI on each cell, the same setup
ALCE and FActScore use.
"""
    )
    return


@app.cell
def __(config_order, per_config):
    """Bootstrap mean ± 95 % CI on hit@10 and MRR for each config."""
    import numpy as np

    def bootstrap_ci(values, n_boot=1000, ci=95, rng=None):
        rng = rng or np.random.default_rng(0)
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return float("nan"), float("nan"), float("nan")
        boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        return arr.mean(), lo, hi

    def build_ci_rows():
        rng = np.random.default_rng(42)
        out = []
        for cfg in config_order:
            per_q = per_config[cfg]["per_query"]
            for metric in ("hit_rate@10", "mrr"):
                values = [q["paper"][metric] for q in per_q if q["paper"][metric] is not None]
                mean, lo, hi = bootstrap_ci(values, rng=rng)
                out.append(
                    {
                        "config": cfg,
                        "metric": metric,
                        "mean": round(mean, 3),
                        "lo95": round(lo, 3),
                        "hi95": round(hi, 3),
                        "ci_width": round(hi - lo, 3),
                    }
                )
        return out

    ci_rows = build_ci_rows()
    return (ci_rows, np)


@app.cell
def __(ci_rows, mo, pd):
    ci_df = pd.DataFrame(ci_rows)
    mo.ui.table(ci_df, selection=None)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
**Reading the CIs.** The 95 % bands on `hit_rate@10` are roughly ±0.10,
so the coarse_rerank vs coarse_faiss gap on hit@10 (+0.08) is
borderline. The MRR gap on the same comparison (0.842 vs 0.661,
Δ = +0.18) sits well outside its CI band and is robust. The report will
say the reranker improves MRR confidently and hit@10 modestly; point
estimates alone would overstate the second claim.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## nDCG@k (paper-level)

Hit-rate and MRR ignore most of the ranked list. nDCG@k weights every
position by its inverse log-rank, so a near-miss at rank 3 scores higher
than the same hit at rank 10. The evaluator now persists the top-K
retrieved IDs in each per-query record, so nDCG@k is computed exactly
from the ranking and deduplicates gold papers (each gold paper counts
once at its earliest rank). Standard practice for BEIR and MTEB.
"""
    )
    return


@app.cell
def __(config_order, per_config):
    """Pull the exact nDCG@k aggregates from each config's result JSON."""

    def build_ndcg_rows():
        ks = (5, 10, 20)
        rows = []
        for cfg in config_order:
            agg = per_config[cfg]["aggregate"]["paper"]
            row = {"config": cfg}
            for k in ks:
                row[f"nDCG@{k}"] = round(agg[f"ndcg@{k}"], 3)
            rows.append(row)
        return rows

    ndcg_rows = build_ndcg_rows()
    return (ndcg_rows,)


@app.cell
def __(mo, ndcg_rows, pd):
    ndcg_df = pd.DataFrame(ndcg_rows).set_index("config")
    mo.ui.table(ndcg_df, selection=None)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
coarse_rerank tops the table at every k (0.848 / 0.854 / 0.862), which
agrees with its MRR lead: the reranker shoves correct papers up to rank
1 and 2 rather than just into the top-10 window. fine_rerank trails it
slightly even though it owns hit@10, because hit@10 cares only about set
membership; nDCG penalises hits at rank 6-10.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Per-category breakdown

Group paper-level hit@10 by question category to see where each config
struggles.
"""
    )
    return


@app.cell
def __(config_order, per_config):
    """Per-category × per-config hit@10 mean."""

    def build_category_rows():
        by_cat: dict[str, dict[str, list[float]]] = {}
        for cfg in config_order:
            for q in per_config[cfg]["per_query"]:
                by_cat.setdefault(q["category"], {}).setdefault(cfg, []).append(
                    q["paper"]["hit_rate@10"]
                )
        rows: list[dict[str, object]] = []
        for cat in sorted(by_cat):
            per_cfg = by_cat[cat]
            any_cfg = next(iter(per_cfg))
            row: dict[str, object] = {"category": cat, "n": len(per_cfg[any_cfg])}
            for cfg in config_order:
                vals = per_cfg.get(cfg, [])
                row[cfg] = round(sum(vals) / len(vals), 3) if vals else None
            rows.append(row)
        return rows

    category_rows = build_category_rows()
    return (category_rows,)


@app.cell
def __(category_rows, mo, pd):
    cat_df = pd.DataFrame(category_rows).set_index("category")
    mo.ui.table(cat_df, selection=None)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
* `comparison` and `methodology` saturate at 100 % hit@10 across every
  config. Answer-bearing papers stand out from the rest of the corpus
  and the retriever finds them easily.
* `policy_impact` is the hardest category and shows the widest spread
  (0.786 to 1.000). Faruk's Corrective RAG (proposal Component 3) targets
  exactly this case: a query-rewrite step lets the pipeline recover when
  the first retrieval misses.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Failure-mode inspection

Which queries fail at top-10 under every config, and which only fail
under some?
"""
    )
    return


@app.cell
def __(config_order, per_config):
    """Queries with hit_rate@10 = 0 across all 4 configs."""

    def build_failure_buckets():
        misses: dict[str, list[str]] = {}
        for cfg in config_order:
            for q in per_config[cfg]["per_query"]:
                misses.setdefault(q["query_id"], [])
                if q["paper"]["hit_rate@10"] == 0.0:
                    misses[q["query_id"]].append(cfg)
        always = sorted(qid for qid, miss in misses.items() if len(miss) == 4)
        sometimes = sorted(qid for qid, miss in misses.items() if 0 < len(miss) < 4)
        return always, sometimes

    always_miss, sometimes_miss = build_failure_buckets()
    return (always_miss, sometimes_miss)


@app.cell
def __(always_miss, mo, sometimes_miss):
    mo.md(
        f"""
* **Always miss** (top-10 hit=0 in every config): `{", ".join(always_miss) or "∅"}`
* **Sometimes miss** (depends on config): `{", ".join(sometimes_miss) or "∅"}`

A small always-miss set keeps the corpus inside the retriever's reach.
The sometimes-miss set is the population where the reranker earns its
keep.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Cross-cutting work: adversarial controls and the κ paradox

Supporting the gold-dataset effort, I caught Cohen's κ returning 1.000
on the IAA subset while delivering no information. Every claim carried
the same `annotator_label` and `reviewer_label`: `"supports"`. A
single-class label space collapses the chance-agreement term:

$$
p_e = \sum_{\ell} P(a = \ell) \, P(r = \ell) = 1^2 + 0^2 + 0^2 = 1
$$

and κ returns 1.0 by convention with zero real signal behind it.

I designed 6 adversarial control claims (q900 to q905) that fail their
cited spans on purpose, each targeting a distinct RAG failure mode:

| id | failure mode | label |
|----|--------------|-------|
| q900 | negation flip (Bessen & Maskin) | `contradicts` |
| q901 | quantitative drift (7-15 % vs 30-40 %) | `contradicts` |
| q902 | category swap (real-property zero-sum vs IP) | `contradicts` |
| q903 | entity swap (trademark vs patent) | `unrelated` |
| q904 | scope overreach (conditional vs universal) | `contradicts` |
| q905 | date drift (1996-2002 vs 1980-1995) | `contradicts` |

Once these land, the label-space union is `{supports=12, contradicts=5,
unrelated=1}`, $p_e \approx 0.52$, and κ = 1.000 means perfect agreement
well above chance.

Files: `evaluation/gold_dataset/contributions/adversarial.json`,
`scripts/compute_iaa.py` (added per-rater label distribution and a
degeneracy warning).
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Cross-component synthesis: the chunker is the bottleneck

The retrieval ablation above and Yusif's RAGAS run (PR #21, end-to-end
RAG vs long-context on 8 stratified questions) point at the same root
cause from two angles:

| observation | metric | implication |
|---|---|---|
| Chunker drops 40-80 % of doc text | coverage audit | gold spans land in the gaps often |
| Chunk-level retrieval is well-defined on ~50 % of queries | retrieval metric availability | the gaps are systematic, not edge cases |
| RAGAS `context_recall`: 0.792 RAG vs 1.000 LC | end-to-end | retrieved chunks omit content the full doc carries |
| RAGAS `faithfulness`: 0.806 RAG vs 0.975 LC | end-to-end | the missing context produces wrong answers downstream |

Two metrics on disjoint slices of the pipeline point at one bottleneck:
chunker coverage. Re-chunking with a no-gap sliding window is the
highest-leverage M3 task, and it unlocks chunk-level retrieval metrics
across the full 37-query set instead of the 50 % subset we evaluate
today.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Deferred to M3 (final 4-page report)

* New embedders (BGE-M3, E5-large, ColBERTv2). Each requires a fresh
  46k-chunk FAISS index.
* No-gap chunker rebuild. Replaces the section-aware chunker with a
  sliding-window scheme that has no internal gaps and unlocks chunk-level
  metrics on full coverage.
* Scale the RAGAS sweep from $n=8$ to the full $n=37$. One CLI
  invocation, roughly \$0.50 in API calls.
* Wire CRAG (Faruk's component) into the ablation. One extra column in
  the headline table.

## Reproducibility

```bash
# one config
uv run python -m evaluation.retrieval_eval.evaluate_retrieval \
    --config coarse_rerank

# full ablation + Markdown table
uv run python -m scripts.run_retrieval_ablation --run-missing

# this notebook
uv run marimo edit notebooks/marimo/elie_retrieval_ablation.py
```

All artifacts in the notebook come from
`evaluation/retrieval_eval/results/*.json`, themselves produced by the
scripts above. Each result JSON now embeds the top-K retrieved papers
and chunks per query, so nDCG and any future ranking-based metric can
be recomputed without re-running the retriever.
"""
    )
    return


if __name__ == "__main__":
    app.run()
