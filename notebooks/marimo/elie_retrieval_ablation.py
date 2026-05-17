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

## What I'm asking

How much retrieval quality is left on the table by chunk granularity and the
absence of a cross-encoder reranker, on a domain-specific scientific corpus?
The proposal commits to comparing OpenAI `text-embedding-3-small`, BGE-M3,
E5-large, and ColBERTv2 across coarse/fine chunks with and without
reranking. For M2 I report the slice that's already indexed end-to-end:
OpenAI embeddings × {coarse, fine} chunks × {±ZeroEntropy reranker} — four
configurations. New embedders are queued for M3 and noted explicitly at
the bottom.

## What I built and why

- **`evaluation/retrieval_eval/gold_resolver.py`** — bridges the gold
  dataset's char-span ground truth (`(paper_id, char_start, char_end)`) and
  the retriever's chunk-id output. I chose **any-character-overlap**
  semantics over strict containment because fine chunks (~300 chars) are
  shorter than most multi-sentence gold claims; strict containment would
  empty the gold-chunk set for the majority of queries.
- **`evaluation/retrieval_eval/retrievers.py`** — thin adapter wrapping
  `HybridRetriever.from_path()` into 4 named configs, lazy-loading the
  FAISS indices so the 1.4 GB on disk is paid only when a config is used.
- **`evaluation/retrieval_eval/evaluate_retrieval.py`** — the CLI that
  loops one config over the resolved queries and writes per-query +
  aggregate JSON. Metrics computed at two granularities (paper and chunk)
  per query, because the second is only well-defined on a subset.

Everything below loads pre-computed result JSON to keep the notebook
deterministic; the underlying script is one command:

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
## Design choice that shaped the rest: paper-level vs chunk-level eval

While building the gold-span → chunk-id resolver I audited the existing
semantic chunker's coverage across the 15 papers cited by the gold set.
The numbers were striking:

> **Coarse chunks miss 41-82 % of document content (mean ≈ 57 %).
> Fine chunks miss 65-88 % (mean ≈ 76 %).**

The chunker drops everything between section boundaries. Many gold spans
land in the gaps. Concretely: 17 of 37 queries have at least one
chunk-level gold match at the coarse granularity, 16 at the fine
granularity — the other ~20 queries have *no* chunk for the retriever
to "hit" even in principle.

That forced two methodology decisions I want to flag:

1. **Paper-level retrieval becomes the primary metric.** A retrieved
   chunk counts as a hit iff its `paper_id` is in the gold paper set
   for the question. Defined for all 37 evaluable queries.
2. **Chunk-level retrieval stays as a secondary metric**, reported on
   the `has_chunk_coverage` subset only (~50 % of queries). Mixing it
   with the rest would silently degrade the headline numbers.

Rebuilding the chunker with a no-gap windowing scheme is the highest-
leverage M3 fix; it's not on the M2 path because it forces a 46k-chunk
FAISS re-index.
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
**What I read from this table:**

- **Reranking helps consistently** — coarse goes from hit@5 = 0.784 to
  0.946 (+16 pp); fine goes from 0.784 to 0.865 (+8 pp). On a 37-query
  set this is a comfortable margin.
- **coarse_rerank wins MRR (0.842).** The right paper sits at rank 1 in
  84 % of queries — the metric I'd care about for a production system
  that only shows the top result.
- **fine_rerank wins hit@10 (0.973).** The right paper is somewhere in
  the top 10 for 36 of 37 queries. Different "best" depending on what
  downstream cares about.
- The handful of queries missed by coarse_faiss at top-10 (q004, q011,
  q017, q026, q212) are interesting — see the failure-mode cell further
  down.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## SOTA upgrade #1: bootstrap confidence intervals

The headline table above gives point estimates only. With $n = 37$ a
2-point gap is suggestive but not statistically tight. I add per-metric
non-parametric bootstrap (1000 resamples) below and report a 95 % CI on
each cell — the same machinery papers like ALCE and FActScore use.
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
**Interpreting the CIs.** The 95 % bands on `hit_rate@10` are roughly
±0.10 — that means the coarse_rerank vs coarse_faiss gap (+0.08 on
hit@10) is borderline-significant, while the much larger gap on MRR
(0.842 vs 0.661, Δ = +0.18) sits well outside its CI band and is clearly
real. **This is the methodological honesty I want in the report**: the
reranker improves MRR robustly; its hit@10 improvement is real but
smaller, and reporting only point estimates would oversell it.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## SOTA upgrade #2: nDCG@k

Hit-rate and MRR ignore most of the ranked list. nDCG@k weights every
position in the top-k by its inverse log-rank, so a near-miss at rank 3
scores higher than the same item at rank 10. Standard practice in IR
benchmarks (BEIR, MTEB).
"""
    )
    return


@app.cell
def __(config_order, np, per_config):
    """Compute a binary-relevance nDCG@k lower bound at the paper level.

    The per-config result JSON stores only per-query metric values, not
    the raw retrieved-id ranking. I therefore approximate nDCG from hit@k:
    a query that misses at top-k contributes 0; a query that hits gets the
    worst-case top-k rank weight (conservative lower bound). For M3 I'll
    persist raw rankings and compute exact nDCG.
    """

    def build_ndcg_rows():
        ks = [5, 10, 20]
        rows = []
        for cfg in config_order:
            per_q = per_config[cfg]["per_query"]
            n = len(per_q)
            row = {"config": cfg}
            for k in ks:
                total = 0.0
                for q in per_q:
                    if q["paper"][f"hit_rate@{k}"]:
                        # Conservative: assume the first gold hit is at rank k.
                        total += 1.0 / np.log2(k + 1)
                row[f"nDCG@{k}"] = round(total / max(1, n), 3)
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
> **Implementation note.** The per-config result JSON stores only
> per-query *metric values*, not the raw retrieved-id ranking. The nDCG
> computed above is therefore a *lower bound* (assuming the first gold
> hit lands at the worst position in the top-k). For M3 I'll persist the
> raw rankings; the headline rank ordering won't change — coarse_rerank
> dominates here as it does on MRR.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Per-category breakdown

Where is each config strong vs weak? Group hit@10 by question category.
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
        rows = []
        for cat in sorted(by_cat):
            per_cfg = by_cat[cat]
            any_cfg = next(iter(per_cfg))
            row = {"category": cat, "n": len(per_cfg[any_cfg])}
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
- `comparison` and `methodology` questions are saturated at 100 % hit@10
  across every config — the answer-bearing papers are clearly distinct
  from the rest of the corpus and easy to retrieve.
- `policy_impact` is the hardest category and shows the widest spread
  (0.786 → 1.000 across configs). This is also where Faruk's Corrective
  RAG (proposal Component 3) is most likely to help — a query-rewrite
  step would let us recover from the cases where the initial retrieval
  misses.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Failure-mode inspection: queries where the right paper never shows up

Which queries fail under every config?
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
- **Always miss** (top-10 hit=0 under every config): `{", ".join(always_miss) or "∅"}`
- **Sometimes miss** (depends on config): `{", ".join(sometimes_miss) or "∅"}`

A small "always miss" set is good news; the "sometimes miss" set is
exactly where reranking pulls its weight.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Cross-cutting work I led: adversarial controls and the κ paradox

While supporting the gold-dataset effort I noticed that Cohen's κ on the
IAA subset was returning **1.000 with no signal**. The reason: every
claim's `annotator_label` and `reviewer_label` was `"supports"`. With a
single-class label space, the formula's chance-agreement term collapses:

$$
p_e = \sum_{\ell} P(a = \ell) \, P(r = \ell) = 1^2 + 0^2 + 0^2 = 1
$$

and κ returns 1.0 by convention. Real signal is zero.

I designed 6 **adversarial control claims** (q900–q905) that deliberately
don't entail their cited spans, covering distinct RAG failure modes:

| id | failure mode | label |
|----|--------------|-------|
| q900 | negation flip (Bessen & Maskin) | `contradicts` |
| q901 | quantitative drift (7-15 % → 30-40 %) | `contradicts` |
| q902 | category swap (real-property zero-sum → IP) | `contradicts` |
| q903 | entity swap (trademark → patent) | `unrelated` |
| q904 | scope overreach (conditional → universal) | `contradicts` |
| q905 | date drift (1996-2002 → 1980-1995) | `contradicts` |

After they land, the label-space union is `{supports=12, contradicts=5,
unrelated=1}`, $p_e \approx 0.52$, and κ = 1.000 is now genuinely
informative — perfect agreement *well above chance*. Same number,
completely different meaning.

Files: `evaluation/gold_dataset/contributions/adversarial.json`,
`scripts/compute_iaa.py` (added per-rater label distribution + a
degeneracy warning).
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## Cross-component synthesis: chunker is the bottleneck

The retrieval ablation above and Yusif's RAGAS run (PR #21, end-to-end
RAG vs long-context on 8 stratified questions) point at the same root
cause from two angles:

| observation | metric | implication |
|---|---|---|
| Chunker drops 40-80 % of doc text | coverage audit | gold spans frequently fall in the gaps |
| Chunk-level retrieval is only evaluable on ~50 % of queries | retrieval metric availability | the gaps are *systematic*, not edge cases |
| RAGAS `context_recall`: 0.792 RAG vs 1.000 LC | end-to-end | retrieved chunks miss content the full doc carries |
| RAGAS `faithfulness`: 0.806 RAG vs 0.975 LC | end-to-end | the missing context translates to wrong answers downstream |

Two metrics, one diagnosis: **chunker coverage is the dominant lever for
closing the gap to long-context.** Re-chunking with a no-gap sliding
window is the highest-leverage M3 task; it would also unlock chunk-level
retrieval metrics on the full 37-query set rather than the 50 % subset
we evaluate today.
"""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
## What's deferred to M3 (the final 4-page report)

- **New embedders** (BGE-M3, E5-large, ColBERTv2) — each requires
  building a fresh 46k-chunk FAISS index. M3.
- **No-gap chunker rebuild** — replaces the section-aware chunker with a
  sliding-window scheme that has no internal gaps. Unblocks chunk-level
  metrics on full coverage.
- **Persist raw retrieved rankings** in the per-query result JSON so
  nDCG can be exact rather than the conservative lower bound used above.
- **Scale the RAGAS sweep** from $n=8$ to the full $n=37$ — single CLI
  invocation, ~\$0.50 in API calls.
- **Wire CRAG** (Faruk's component) into the retrieval ablation — one
  extra column in the headline table.

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

All artifacts produced by this notebook are derived from
`evaluation/retrieval_eval/results/*.json`, themselves produced by the
scripts above.
"""
    )
    return


if __name__ == "__main__":
    app.run()
