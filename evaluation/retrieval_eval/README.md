# Retrieval evaluation

Evaluates retrieval quality on the gold dataset across named configurations
of the existing FAISS + ZeroEntropy stack. Owns the **paper-level** and
**chunk-level** retrieval metrics that go into the M2 progress report.

## What it does

For each gold question, the retriever returns ranked chunks. We score those
against ground truth at two granularities:

| Granularity | Gold target | Always defined? |
|-------------|-------------|-----------------|
| **paper** (primary) | the set of paper IDs cited by the question's spans | yes |
| **chunk** (secondary) | the set of chunk IDs whose `[char_start, char_end)` overlaps any gold span | only when chunks exist at the span's location |

The chunk-level metric is reported on a subset because the existing semantic
chunker (`packages/shared/rag_pipeline/rag/chunking.py`) skips 40–80 % of
each document between section boundaries. Gold annotators frequently pick
passages in those gaps. The resolver flags this per query via
`ResolvedQuery.has_chunk_coverage(chunk_type)`; chunk-level metrics aggregate
only over the covered subset, so the denominators are reported separately.

Span-to-chunk overlap is defined as **any character overlap** (half-open
intervals): `not (chunk.char_end <= span.char_start or chunk.char_start >= span.char_end)`.
Strict containment was rejected because fine-granularity chunks (~300 chars)
are often shorter than a multi-sentence gold claim, which would force the
gold-chunk set to be empty.

Adversarial pairs (`annotator == "adversarial"`) and unanswerable pairs
(`difficulty == "unanswerable"`) are filtered out of the retrieval eval —
the former have deliberately wrong claims that test the faithfulness scorer
elsewhere, the latter have no supporting spans by schema invariant.

## Configurations

Three embedder families crossed with two granularities and ±reranker,
12 configs total. OpenAI is the original index; BGE-M3 and E5-large
indices are built locally by `scripts.build_hf_index` from the chunk
JSONs in `data/s3_archive/chunks/`.

| family | configs |
|--------|---------|
| `openai` (`text-embedding-3-small`, 1536-dim) | `coarse_faiss`, `coarse_rerank`, `fine_faiss`, `fine_rerank` |
| `bge_m3` (`BAAI/bge-m3`, 1024-dim) | `bge_m3_coarse_faiss`, `bge_m3_coarse_rerank`, `bge_m3_fine_faiss`, `bge_m3_fine_rerank` |
| `e5_large` (`intfloat/e5-large-v2`, 1024-dim) | `e5_large_coarse_faiss`, ..., `e5_large_fine_rerank` |

ColBERTv2 stays out: it is late-interaction multi-vector retrieval and
needs the PLAID index format, not a drop-in dense-vector swap.

### Building the alternative-embedder indices

The HF indices are not tracked (each is ~200 MB–1.4 GB). Build them once
on a GPU cluster:

```bash
# all four indices, ~30–60 min on CUDA
uv run python -m scripts.build_all_hf_indices --device cuda --batch-size 256

# or build just one
uv run python -m scripts.build_hf_index --model bge-m3 --chunk-type coarse --device cuda
```

The script reads `data/s3_archive/chunks/*_<type>.json`, batch-encodes
with the requested HF model (L2-normalised, ``max_seq_length=512``), and
writes `data/s3_archive/indexes/<embedder>_<chunk_type>.{faiss,_metadata.json}`
in the same layout the OpenAI indices use.

On Apple MPS expect ~7 h for the full set; on an A100 it drops to under
an hour.

## Run one config

```bash
uv run python -m evaluation.retrieval_eval.evaluate_retrieval --config coarse_rerank
```

Writes `evaluation/retrieval_eval/results/coarse_rerank.json` containing
`{config, k_values, n_queries_total, n_queries_chunk_coverage, aggregate, per_query}`.

Requires `OPENAI_API_KEY` (always) and `ZEROENTROPY_API_KEY` (only for the
`*_rerank` configs).

## Run the full ablation

```bash
uv run python -m scripts.run_retrieval_ablation --run-missing
```

Iterates the four configs (running any whose result JSON is missing), then
writes `comparison.json` and a human-readable `comparison.md` side-by-side
table.

## Current M2 numbers

Latest run (37 evaluable gold queries, paper-level metric primary):

| config | hit@5 | hit@10 | hit@20 | MRR |
|--------|-------|--------|--------|-----|
| `coarse_faiss` | 0.784 | 0.865 | 0.946 | 0.661 |
| `coarse_rerank` | 0.946 | 0.946 | 0.946 | **0.842** |
| `fine_faiss` | 0.784 | 0.892 | 0.946 | 0.625 |
| `fine_rerank` | 0.865 | **0.973** | 0.973 | 0.705 |

Headline observations:

- **Reranking helps consistently** — +6 to +16 pp on hit@5 over the
  unranked baseline.
- **coarse_rerank wins MRR** (0.842) — the right paper is at rank 1 in
  84 % of queries.
- **fine_rerank wins hit@10** (0.973) — the right paper is somewhere in
  the top-10 for 36 of 37 queries.
- Per-category, `policy_impact` is the hardest (widest spread across
  configs); `comparison` and `methodology` questions are saturated at
  100 % hit@10.

## Out of scope for M2 (queued for M3)

- New embedding models (BGE-M3 / E5-large / ColBERTv2 / their fresh
  indices)
- Rebuilt chunks with no-gap windowing — would unlock chunk-level metric
  on all queries instead of the ~50 % coverage subset
- Per-paper retrieval cost / latency reporting
- Query-rewrite ablation (CRAG-adjacent — Faruk's lane)

## Files

```
evaluation/retrieval_eval/
├── gold_resolver.py        # gold-span → chunk IDs (with overlap & filter rules)
├── retrievers.py           # adapter wrapping HybridRetriever, 4 named configs
├── evaluate_retrieval.py   # CLI: run one config, write results JSON
├── results/                # per-config + comparison artifacts
└── README.md               # this file

scripts/run_retrieval_ablation.py  # full-ablation driver + Markdown table
tests/test_retrieval_eval.py       # unit tests for resolver + metrics
```
