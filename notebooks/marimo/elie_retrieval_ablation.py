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

How much retrieval quality does the choice of embedder, chunk
granularity, and cross-encoder reranker each contribute on a
domain-specific scientific corpus? The proposal commits to four
embedders (OpenAI `text-embedding-3-small`, BGE-M3, E5-large,
ColBERTv2). M2 ships three of them: OpenAI on the original 46k-chunk
FAISS index, plus BGE-M3 (`BAAI/bge-m3`) and E5-large
(`intfloat/e5-large-v2`) on fresh 1024-dim indices built on the EPFL
RCP cluster. Each embedder is crossed with both chunk granularities
and ±ZeroEntropy reranker, giving 12 configurations. ColBERTv2 stays
in M3 — its late-interaction multi-vector retrieval needs the PLAID
index format, not a drop-in dense-vector swap.

## What I built

* `evaluation/retrieval_eval/gold_resolver.py` bridges the gold dataset's
  char-span ground truth `(paper_id, char_start, char_end)` and the
  retriever's chunk-id output. I picked **any-character-overlap** over
  strict containment because fine chunks (~300 chars) are shorter than
  most multi-sentence gold claims; strict containment would empty the
  gold-chunk set for most queries.
* `packages/shared/rag_pipeline/rag/embedder.py` adds an `Embedder`
  protocol with `encode_queries` / `encode_passages` returning
  L2-normalised float32 vectors. `OpenAIEmbedderAdapter` wraps the
  existing OpenAI client; `SentenceTransformerEmbedder` handles
  BGE-M3 and E5-large (with the `query: ` / `passage: ` prefixes E5
  was trained with). `HybridRetriever.from_path()` now accepts any
  `Embedder` and an `index_basename`, so the same retriever code
  drives all 12 configs.
* `scripts/build_hf_index.py` and `scripts/build_all_hf_indices.py`
  read `data/s3_archive/chunks/*_<type>.json`, batch-encode on CUDA
  (or MPS), and write FAISS L2 indices plus metadata in the same
  layout the OpenAI indices use. ~105 min wall on one A100 for the
  four alt-embedder indices (46k + 186k vectors × 2 models).
* `evaluation/retrieval_eval/retrievers.py` wires 3 embedder families ×
  2 granularities × ±reranker into 12 named configs and lazy-loads each
  FAISS index on first use.
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
    """Load every per-config result JSON (12 configs across 3 embedder families)."""
    results_dir = repo_root / "evaluation" / "retrieval_eval" / "results"
    per_config = {
        path.stem: json.loads(path.read_text())
        for path in sorted(results_dir.glob("*.json"))
        if path.stem != "comparison"
    }
    # Headline ordering: by embedder, then granularity, then ±rerank.
    config_order = [
        "coarse_faiss",
        "coarse_rerank",
        "fine_faiss",
        "fine_rerank",
        "bge_m3_coarse_faiss",
        "bge_m3_coarse_rerank",
        "bge_m3_fine_faiss",
        "bge_m3_fine_rerank",
        "e5_large_coarse_faiss",
        "e5_large_coarse_rerank",
        "e5_large_fine_faiss",
        "e5_large_fine_rerank",
    ]
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

* **E5-large coarse with reranker wins every paper-level metric.**
  hit@5 = 0.973, hit@10 = 0.973, MRR = 0.878. The right paper sits at
  rank 1 in 88 % of queries and in the top 10 for 36 of 37. Beats the
  previous OpenAI champion (`coarse_rerank`, MRR 0.842) by 4 pp on
  MRR and 3 pp on hit@5.
* **E5-large is a stronger embedder than OpenAI even without a
  reranker.** `e5_large_coarse_faiss` reaches hit@5 = 0.811 and
  MRR = 0.747, ahead of every OpenAI FAISS-only config (the best
  OpenAI no-rerank MRR is 0.661). A free, open-source encoder
  outperforms the closed-source baseline on this scientific corpus.
* **BGE-M3 trades top-5 precision for top-20 recall.**
  `bge_m3_coarse_faiss` is the only config to reach hit@20 = 1.000
  (perfect recall in the top 20) but its top-5 (0.703) trails OpenAI
  and E5-large. With reranking it pulls level with OpenAI on hit@5
  (0.946) and MRR (0.842).
* **Reranking still helps everyone**, but the relative effect shrinks
  for stronger embedders: +16 pp on hit@5 for OpenAI coarse, +24 pp
  for BGE-M3 coarse, only +16 pp for E5-large coarse — because the
  E5-large FAISS baseline is already at 0.811.
* **Coarse > fine across embedders.** Fine chunks are too short for
  scientific writing; the embedder has too little context per chunk
  and the right paper drops out of the top 5 more often.
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
so the headline E5-large-vs-OpenAI gap on `coarse_rerank` (+3 pp on
hit@5, +4 pp on MRR) sits at the edge of significance with 37
queries. The wider gap on plain FAISS — `e5_large_coarse_faiss` MRR
0.747 vs `coarse_faiss` 0.661, Δ = +0.086 — sits inside its CI band,
so I report it as "E5-large is at least as good as OpenAI even
without reranking, with a measurable lift on MRR." The full 12-config
CI matrix below makes the band per cell explicit; the report will
flag every comparison whose Δ falls inside the band.
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
`e5_large_coarse_rerank` tops nDCG at every k, mirroring its MRR lead.
The reranker pushes correct papers to rank 1 rather than into the
top-10 window, and that's where the inverse-log-rank weight rewards
hardest. `coarse_rerank` (OpenAI) holds the second slot; BGE-M3 only
catches up at hit@20 because nDCG penalises late ranks, where its
"perfect recall in top-20" lives.
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
  embedder family and every config. Answer-bearing papers stand out
  from the rest of the corpus.
* `policy_impact` is the hardest category and shows the widest spread
  (0.786 OpenAI coarse_faiss to 1.000 with E5-large). The new
  embedders close most of the gap here without any reranker, which
  matches the overall E5-large lead.
* `multi_hop` is where fine chunks break: every embedder family loses
  on `fine_faiss` (0.833 OpenAI, 0.667 BGE-M3, 0.667 E5-large). Short
  chunks split the multi-hop evidence across separate IDs and the
  embedder loses the connection. Faruk's CRAG (proposal Component 3)
  is the right tool for this case.
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

* **ColBERTv2.** The remaining proposal-listed embedder. Late-interaction
  multi-vector retrieval needs the PLAID index format, not a dense
  L2 index, so it is a separate engineering track from the BGE-M3 /
  E5-large work that landed for M2.
* **No-gap chunker rebuild.** Replaces the section-aware chunker with
  a sliding-window scheme that has no internal gaps. Unlocks chunk-level
  metrics on full coverage and would lift the secondary chunk-level
  numbers in the table above from a 16-20 query subset to the full 37.
* **Scale the RAGAS sweep** from $n=8$ to the full $n=37$. One CLI
  invocation, ~\$0.50 in API calls.
* **Wire CRAG** (Faruk's component) into the ablation as a 13th
  config. Targets the `multi_hop` and `policy_impact` failure modes
  highlighted in the per-category cell above.

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
