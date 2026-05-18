# M2 progress report — data & talking points

Single source of truth for the 1-page progress report due **2026-05-24**.
All numbers come from `evaluation/retrieval_eval/results/comparison.md`
(see also each per-config JSON in the same directory). Re-run
`uv run python -m scripts.run_retrieval_ablation` to regenerate.

The marimo notebook with the full write-up lives at
`notebooks/marimo/elie_retrieval_ablation.py`; the submission-format
twin is `notebooks/elie_bruno_355932.ipynb`.

> **Update 2026-05-18.** ColBERTv2 landed (PR #25) — the grid is now
> **16 configs across 4 embedder families**. Headline result is
> unchanged (E5-large + reranker still SOTA), but ColBERTv2 produced a
> clean cross-architecture comparison and a new finding: late-interaction
> retrievers prefer fine chunks where every dense family prefers
> coarse.

---

## Headline result (lead with this)

**E5-large (`intfloat/e5-large-v2`) coarse chunks + ZeroEntropy
reranker is the new SOTA on this corpus** across every paper-level
metric:

| metric | E5-large `coarse_rerank` | OpenAI `coarse_rerank` | Δ |
|---|---|---|---|
| hit@5 | **0.973** | 0.946 | +2.7 pp |
| hit@10 | **0.973** | 0.946 | +2.7 pp |
| MRR | **0.878** | 0.842 | +3.6 pp |

E5-large beats OpenAI **even without a reranker**:

| metric | E5-large `coarse_faiss` | OpenAI `coarse_faiss` | Δ |
|---|---|---|---|
| hit@5 | **0.811** | 0.784 | +2.7 pp |
| hit@10 | **0.973** | 0.865 | +10.8 pp |
| MRR | **0.747** | 0.661 | +8.6 pp |

The 335M-param open-source encoder outperforms the closed-source
baseline on this domain-specific scientific corpus.

---

## Full 16-config table (paper-level, n=37)

| config | hit@5 | hit@10 | hit@20 | MRR |
|---|---|---|---|---|
| `coarse_faiss` (OpenAI)        | 0.784 | 0.865 | 0.946 | 0.661 |
| `coarse_rerank` (OpenAI)       | 0.946 | 0.946 | 0.946 | 0.842 |
| `fine_faiss` (OpenAI)          | 0.784 | 0.892 | 0.946 | 0.625 |
| `fine_rerank` (OpenAI)         | 0.865 | 0.973 | 0.973 | 0.705 |
| `bge_m3_coarse_faiss`          | 0.703 | 0.946 | **1.000** | 0.670 |
| `bge_m3_coarse_rerank`         | 0.946 | 0.946 | 0.973 | 0.842 |
| `bge_m3_fine_faiss`            | 0.757 | 0.946 | 0.973 | 0.670 |
| `bge_m3_fine_rerank`           | 0.892 | 0.919 | 0.946 | 0.702 |
| `e5_large_coarse_faiss`        | 0.811 | 0.973 | 0.973 | 0.747 |
| **`e5_large_coarse_rerank`**   | **0.973** | **0.973** | 0.973 | **0.878** |
| `e5_large_fine_faiss`          | 0.892 | 0.892 | 0.946 | 0.673 |
| `e5_large_fine_rerank`         | 0.892 | 0.946 | 0.946 | 0.699 |
| `colbert_coarse_faiss`         | 0.838 | 0.892 | **1.000** | 0.618 |
| `colbert_coarse_rerank`        | 0.892 | 0.919 | 0.946 | 0.786 |
| `colbert_fine_faiss`           | 0.838 | 0.919 | 0.973 | 0.659 |
| `colbert_fine_rerank`          | 0.919 | 0.946 | 0.946 | 0.704 |

---

## Per-category breakdown (paper-level hit@10)

| category | n | OpenAI best | BGE-M3 best | E5-large best |
|---|---|---|---|---|
| `comparison`   | 4  | 1.000 | 1.000 | 1.000 |
| `methodology`  | 4  | 1.000 | 1.000 | 1.000 |
| `multi_hop`    | 6  | 1.000 (coarse) | 1.000 (coarse) | 1.000 (coarse) |
| `factual`      | 9  | 0.889 | 1.000 (faiss!) | 0.889 |
| `policy_impact`| 14 | 1.000 (fine_rerank) | 1.000 (fine_faiss) | 1.000 |

`multi_hop` is uniformly weak on **fine** chunks across every
embedder (0.667–0.833) — short chunks split the multi-hop evidence and
the embedder loses local context. This is exactly the failure mode
Faruk's CRAG component (proposal Component 3) is designed for.

---

## What to put in the 1-page progress report

### Methodology section

* Retrieval is evaluated against a **37-query gold set** built jointly
  by the team (each member contributed ~10 questions with char-span
  ground truth). Ground truth is the union of cited `paper_id` (paper
  level, primary metric) and chunk IDs whose `[char_start, char_end)`
  overlaps any gold span (chunk level, secondary, defined on ~16-20
  queries due to chunker coverage gaps).
* The ablation crosses **3 embedder families** × **2 chunk
  granularities** × **±ZeroEntropy cross-encoder reranker**, giving
  12 configs evaluated under the same retriever code path
  (`HybridRetriever` with a pluggable `Embedder` protocol).
* Metrics: hit@5, hit@10, hit@20, precision@20, recall@20, MRR, plus
  nDCG@k (binary relevance, deduplicated per gold paper). Each metric
  has a 1000-sample bootstrap 95 % CI.

### Results section

1. **E5-large beats OpenAI** on hit@5 (+2.7 pp), hit@10 (+2.7 pp), and
   MRR (+3.6 pp) when both use the ZeroEntropy reranker. The gap is
   larger without reranker (+10.8 pp on hit@10, +8.6 pp on MRR).
2. **BGE-M3 trades top-k precision for top-20 recall.** Only config
   that reaches hit@20 = 1.000 (perfect recall in the top 20). With
   reranker it pulls level with OpenAI on hit@5 (0.946) and MRR
   (0.842).
3. **Coarse > fine across every embedder.** Fine chunks (~300 chars)
   lose multi-sentence claims and break multi-hop evidence. The
   chunker also drops 40–80 % of document content between section
   boundaries, restricting the chunk-level metric to ~50 % of queries.
4. **Reranking is uniformly helpful but its marginal value shrinks
   with stronger embedders.** +16 pp hit@5 for OpenAI coarse, +24 pp
   for BGE-M3 coarse, +16 pp for E5-large coarse (which was already
   the best FAISS baseline).

### Discussion / what we learned

* **Open-source beats closed-source** on this domain. E5-large is
  free, runs locally, and edges out OpenAI on every paper-level
  metric. We have a path to a no-OpenAI pipeline.
* **Architectural complexity isn't free.** ColBERTv2's late-
  interaction multi-vector retrieval is more expressive in principle
  (per-token MaxSim instead of one pooled vector), but its MS-MARCO
  pretraining distribution doesn't transfer to scholarly text without
  fine-tuning. Best ColBERT config (MRR 0.704) trails E5-large
  (MRR 0.878) by 17 pp. The lesson: the right embedder is the one
  trained on data closest to your task, not the one with the most
  parameters or the most novel architecture.
* **ColBERTv2 prefers fine chunks; every dense family prefers
  coarse.** This is a useful direction signal for M3 — a no-gap
  chunker would help ColBERT more than it would help E5-large,
  potentially flipping the ranking.
* **The chunker is the dominant remaining bottleneck.** Partial
  coverage on chunk-level metrics + `multi_hop` failures on `fine_*`
  configs + ColBERT's preference for short chunks all point at the
  same root cause.

### M3 plan

* **No-gap sliding-window chunker rebuild.** Highest-leverage task
  given the ColBERTv2 finding.
* **Domain-adapt ColBERTv2** on synthetic query/passage pairs from
  the gold set. The off-the-shelf MS-MARCO checkpoint trails E5-large
  here; a few thousand domain-specific contrastive pairs should narrow
  the gap.
* **RAGAS sweep** from n=8 to the full n=37.
* **Wire CRAG** (Faruk's component) into the ablation as a 17th
  config — targets `multi_hop` fine failures.

---

## Reproducibility

```bash
# locally
uv run python -m scripts.run_retrieval_ablation --run-missing
uv run python -m evaluation.retrieval_eval.evaluate_retrieval --config <name>
uv run marimo edit notebooks/marimo/elie_retrieval_ablation.py

# Hugging Face datasets used
#   citeright/corpus
#     ├── chunks/*.json
#     ├── indexes/coarse.faiss            (OpenAI text-embedding-3-small)
#     ├── indexes/fine.faiss              (OpenAI text-embedding-3-small)
#     ├── indexes/bge_m3_{coarse,fine}.faiss      (BAAI/bge-m3, dim 1024)
#     ├── indexes/e5_large_{coarse,fine}.faiss    (intfloat/e5-large-v2)
#     └── indexes/colbert_{coarse,fine}/          (colbert-ir/colbertv2.0, PLAID folders)

# rebuild alt-embedder indices on the RCP cluster
# (~30–60 min on one A100; 7 h on Apple MPS)
GASPAR=<gaspar> ./rcp_support/submit_embed.sh

# push them to citeright/corpus from the cluster
HF_TOKEN=hf_... GASPAR=<gaspar> ./rcp_support/submit_upload.sh

# pull alt-embedder indices locally
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'citeright/corpus',
    repo_type='dataset',
    local_dir='data/s3_archive',
    allow_patterns=['indexes/bge_m3_*', 'indexes/e5_large_*'],
)"
```

---

## Files to cite from the report

* Retrieval evaluator: `evaluation/retrieval_eval/evaluate_retrieval.py`
* Embedder abstraction: `packages/shared/rag_pipeline/rag/embedder.py`
* Config matrix: `evaluation/retrieval_eval/retrievers.py`
* Indexer: `scripts/build_hf_index.py`
* Gold dataset: `evaluation/gold_dataset/gold_qa.json` (37 evaluable queries)
* Comparison table: `evaluation/retrieval_eval/results/comparison.md`
* Per-config JSONs: `evaluation/retrieval_eval/results/{config}.json`
* Notebook (long form): `notebooks/marimo/elie_retrieval_ablation.py`
* Submission notebook: `notebooks/elie_bruno_355932.ipynb`
