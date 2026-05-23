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

Four embedder families crossed with two granularities and ±reranker,
**16 configs total**. The four families:

| family | model | dim | retriever backend | configs |
|--------|-------|-----|---------|---------|
| `openai` | `text-embedding-3-small` | 1536 | FAISS L2 | `coarse_faiss`, `coarse_rerank`, `fine_faiss`, `fine_rerank` |
| `bge_m3` | `BAAI/bge-m3` | 1024 | FAISS L2 | `bge_m3_coarse_faiss`, ..., `bge_m3_fine_rerank` |
| `e5_large` | `intfloat/e5-large-v2` | 1024 | FAISS L2 | `e5_large_coarse_faiss`, ..., `e5_large_fine_rerank` |
| `colbert` | `colbert-ir/colbertv2.0` | per-token | PyLate PLAID (MaxSim) | `colbert_coarse_faiss`, ..., `colbert_fine_rerank` |

The first three share the same `FAISSRetriever` + `HybridRetriever`
code path. ColBERTv2 is a different paradigm — one vector per token,
scored by MaxSim — and uses a separate `ColBERTRetriever` that conforms
to a small shared `BaseRetriever` protocol. The reranker layer
(`HybridRetriever`) composes any base, so a single eval pipeline drives
both.

> **Naming caveat.** The historical `_faiss` suffix is preserved for
> ColBERT configs even though they use a PLAID index. The suffix now
> means "no reranker" rather than "FAISS backend". A future cleanup may
> rename everything to `_base` / `_rerank`.

### Building the alternative-embedder indices

The dense indices (~200 MB–1.4 GB each) and the PyLate PLAID folders
are not tracked. Build them once on a GPU cluster:

```bash
# dense (BGE-M3 + E5-large) — ~30–60 min on one A100
uv run python -m scripts.build_all_hf_indices --device cuda --batch-size 256

# ColBERTv2 (coarse + fine) — ~1–3 h on one A100 (per-token encoding is heavier)
uv run python -m scripts.build_all_colbert_indices --device cuda --batch-size 64

# or build just one
uv run python -m scripts.build_hf_index --model bge-m3 --chunk-type coarse --device cuda
uv run python -m scripts.build_colbert_index --chunk-type fine --device cuda
```

Dense indexers write `data/s3_archive/indexes/<embedder>_<chunk_type>.{faiss,_metadata.json}`.
The ColBERT indexer writes `data/s3_archive/indexes/colbert_<chunk_type>/`
(a PLAID folder) plus a `colbert_<chunk_type>_metadata.json` sidecar.

On Apple MPS expect ~7 h for the dense set + ~6 h for ColBERT; on an
A100 the dense set takes under an hour and ColBERT takes 1–3 h.

### Run:AI launchers

Use `rcp_support/submit_embed.sh` for the dense indices and
`rcp_support/submit_colbert.sh` for the ColBERT indices. Both default to
`GROUP=g68` (Team CiteRight) and read `GASPAR` from the environment.
After either job finishes, push the artefacts to the team's HF dataset
with `rcp_support/submit_upload.sh` (the upload patterns now cover
`colbert_*` paths in addition to `bge_m3_*` and `e5_large_*`).

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
