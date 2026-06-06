# RAGAS sweep — M3 (SCITAS izar, Haiku judge)

n = 88 evaluable gold questions · retriever `e5_rerank_s800_o0` · RAG answer LLM **gemini-2.5-flash** · LC **gemini-2.5-flash**

| arm | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|
| `e5_rerank_s800_o0` | 88 | 0.806 | 0.825 | 0.797 | 0.847 | ~0.00029 |
| **long-context** | 88 | 0.920 | 0.731 | 0.864 | 0.875 | ~0.00370 |

Source: SCITAS job on `main` via `scitas_support/ragas_sweep_izar.sbatch` with
`RAGAS_CONFIGS=e5_rerank_s800_o0`,
`RAGAS_RAG_MODEL=api:openrouter/google/gemini-2.5-flash`,
`RAGAS_OUT_DIR=evaluation/ragas_eval/results/m3_sweep_sota_gemini`.
Report integration: PR #76.
