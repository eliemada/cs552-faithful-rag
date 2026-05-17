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
| bge_m3_coarse_faiss | 37 | 0.703 | 0.946 | 1.000 | 0.232 | 0.986 | 0.670 |
| bge_m3_coarse_rerank | 37 | 0.946 | 0.946 | 0.973 | 0.311 | 0.959 | 0.842 |
| bge_m3_fine_faiss | 37 | 0.757 | 0.946 | 0.973 | 0.249 | 0.959 | 0.670 |
| bge_m3_fine_rerank | 37 | 0.892 | 0.919 | 0.946 | 0.268 | 0.932 | 0.702 |
| e5_large_coarse_faiss | 37 | 0.811 | 0.973 | 0.973 | 0.277 | 0.959 | 0.747 |
| e5_large_coarse_rerank | 37 | 0.973 | 0.973 | 0.973 | 0.319 | 0.959 | 0.878 |
| e5_large_fine_faiss | 37 | 0.892 | 0.892 | 0.946 | 0.249 | 0.932 | 0.673 |
| e5_large_fine_rerank | 37 | 0.892 | 0.946 | 0.946 | 0.277 | 0.919 | 0.699 |

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
| bge_m3_coarse_faiss | 20 | 0.550 | 0.750 | 0.750 | 0.040 | 0.405 | 0.489 |
| bge_m3_coarse_rerank | 20 | 0.850 | 0.850 | 0.850 | 0.055 | 0.542 | 0.708 |
| bge_m3_fine_faiss | 16 | 0.312 | 0.500 | 0.562 | 0.031 | 0.264 | 0.294 |
| bge_m3_fine_rerank | 16 | 0.500 | 0.625 | 0.625 | 0.044 | 0.320 | 0.260 |
| e5_large_coarse_faiss | 20 | 0.700 | 0.800 | 0.850 | 0.055 | 0.512 | 0.516 |
| e5_large_coarse_rerank | 20 | 0.850 | 0.900 | 0.900 | 0.060 | 0.629 | 0.735 |
| e5_large_fine_faiss | 16 | 0.375 | 0.438 | 0.562 | 0.041 | 0.307 | 0.330 |
| e5_large_fine_rerank | 16 | 0.438 | 0.625 | 0.750 | 0.050 | 0.362 | 0.253 |

### Paper-level hit@10 by question category

| category | n | coarse_faiss | coarse_rerank | fine_faiss | fine_rerank | bge_m3_coarse_faiss | bge_m3_coarse_rerank | bge_m3_fine_faiss | bge_m3_fine_rerank | e5_large_coarse_faiss | e5_large_coarse_rerank | e5_large_fine_faiss | e5_large_fine_rerank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comparison | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| factual | 9 | 0.778 | 0.889 | 0.889 | 0.889 | 1.000 | 0.889 | 1.000 | 0.889 | 0.889 | 0.889 | 0.889 | 0.889 |
| methodology | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 6 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.667 | 0.833 | 1.000 | 1.000 | 0.667 | 0.833 |
| policy_impact | 14 | 0.786 | 0.929 | 0.857 | 1.000 | 0.857 | 0.929 | 1.000 | 0.929 | 1.000 | 1.000 | 0.929 | 1.000 |

---

Numbers above are produced by `scripts/run_retrieval_ablation.py`; per-config
detail (per-query metrics, latency, gold-set sizes) lives in
`evaluation/retrieval_eval/results/<config>.json`.
