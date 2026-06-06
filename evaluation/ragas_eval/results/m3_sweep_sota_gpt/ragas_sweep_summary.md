# RAGAS sweep — M3 (SCITAS izar, Haiku judge)

n = 88 evaluable gold questions · retriever `e5_rerank_s800_o0` · RAG answer LLM **gpt-4o-mini** · LC **gemini-2.5-flash**

| arm | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|
| `e5_rerank_s800_o0` | 88 | 0.836 | 0.835 | 0.807 | 0.864 | ~0.00032 |
| **long-context** | 88 | 0.920 | 0.731 | 0.864 | 0.875 | ~0.00370 |

Source: SCITAS job on `main` via `scitas_support/ragas_sweep_izar.sbatch` with
`RAGAS_CONFIGS=e5_rerank_s800_o0`, `RAGAS_OUT_DIR=evaluation/ragas_eval/results/m3_sweep_sota_gpt`.
Report integration: PR #76.
