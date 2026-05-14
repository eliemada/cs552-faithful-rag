# Retrieval ablation — M2

Pre-built FAISS indices over OpenAI `text-embedding-3-small` embeddings.
Four configurations spanning chunk granularity × ±ZeroEntropy reranker.

### Paper-level retrieval (primary)

| config | n | hit@5 | hit@10 | hit@20 | P@20 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| coarse_faiss | 37 | 0.784 | 0.865 | 0.946 | 0.266 | 0.919 | 0.661 |
| coarse_rerank | 37 | 0.946 | 0.946 | 0.946 | 0.296 | 0.946 | 0.842 |
| fine_faiss | 37 | 0.784 | 0.892 | 0.946 | 0.262 | 0.932 | 0.625 |
| fine_rerank | 37 | 0.865 | 0.973 | 0.973 | 0.288 | 0.946 | 0.705 |

`n` is the number of evaluable queries. At paper level every gold pair contributes; at
chunk level only queries whose gold span overlaps at least one chunk at the relevant
granularity contribute (~50 % of queries due to the existing chunker's coverage gaps —
see `evaluation/retrieval_eval/gold_resolver.py`).

### Chunk-level retrieval (secondary — coverage subset)

| config | n | hit@5 | hit@10 | hit@20 | P@20 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| coarse_faiss | 20 | 0.500 | 0.600 | 0.800 | 0.055 | 0.557 | 0.478 |
| coarse_rerank | 20 | 0.800 | 0.850 | 0.850 | 0.068 | 0.614 | 0.670 |
| fine_faiss | 16 | 0.250 | 0.375 | 0.500 | 0.044 | 0.221 | 0.150 |
| fine_rerank | 16 | 0.500 | 0.625 | 0.625 | 0.053 | 0.332 | 0.326 |

### Paper-level hit@10 by question category

| category | n | coarse_faiss | coarse_rerank | fine_faiss | fine_rerank |
|---|---|---|---|---|---|
| comparison | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| factual | 9 | 0.778 | 0.889 | 0.889 | 0.889 |
| methodology | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 6 | 1.000 | 1.000 | 0.833 | 1.000 |
| policy_impact | 14 | 0.786 | 0.929 | 0.857 | 1.000 |

---

Numbers above are produced by `scripts/run_retrieval_ablation.py`; per-config
detail (per-query metrics, latency, gold-set sizes) lives in
`evaluation/retrieval_eval/results/<config>.json`.
