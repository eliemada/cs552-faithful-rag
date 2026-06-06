# CRAG threshold ablation — M3 (SCITAS izar)

Full gold: **97** questions (88 answerable + 9 unanswerable) · retriever `e5_large_coarse_rerank` · k=10 · cross-encoder `ms-marco-MiniLM-L-6-v2`

Operating point (τ_high=0.7, τ_low=0.4):

| quantity | value |
|---|---|
| baseline direct-correct retrieval | **0.876** (85/97) |
| branch distribution | correct: **85**, ambiguous: **9**, incorrect: **3** |
| refinement upside cap | **+9.3 pp** (all ambiguous rescuable) |
| abstain strict | n=3, precision=0.33, recall=0.11 |
| abstain loose | n=12, precision=0.25, recall=0.33 |

Source: SCITAS izar job 2962739 (post adapter-text fix, PR #72). Full JSON:
`threshold_sweep_2962739.json` on Elie's scratch — not yet committed (large probe).
