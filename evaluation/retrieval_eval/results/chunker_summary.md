| config | n | hit@5 | hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| e5_large_coarse_rerank (M2 baseline) | 88 | 0.966 | 0.977 | 0.865 | 0.882 |
| e5_rerank_s200_o0 | 88 | 0.943 | 0.977 | 0.824 | 0.839 |
| e5_rerank_s200_o100 | 88 | 0.932 | 0.977 | 0.834 | 0.851 |
| e5_rerank_s400_o0 | 88 | 0.943 | 0.966 | 0.875 | 0.879 |
| e5_rerank_s400_o200 | 88 | 0.920 | 0.977 | 0.810 | 0.836 |
| e5_rerank_s600_o0 | 88 | 0.943 | 0.989 | 0.864 | 0.875 |
| e5_rerank_s600_o300 | 88 | 0.955 | 0.977 | 0.856 | 0.873 |
| e5_rerank_s800_o0 | 88 | 0.943 | 0.966 | 0.902 | 0.895 |
| e5_rerank_s800_o400 | 88 | 0.955 | 0.989 | 0.862 | 0.874 |
| e5_rerank_recursive_400 | 88 | 0.920 | 0.955 | 0.778 | 0.807 |

Best MRR:      **e5_rerank_s800_o0**  MRR=0.902  hit@10=0.966
Best hit@10:   **e5_rerank_s600_o0**  hit@10=0.989  MRR=0.864
Best nDCG@10:  **e5_rerank_s800_o0**  nDCG@10=0.895

Δ vs baseline (e5_large_coarse_rerank (M2 baseline), n=88):  MRR +0.036,  hit@10 +0.011
