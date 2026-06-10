# RAGAS sweep — M3

n = 88 evaluable gold questions (shared across configs).

| config | chunk | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|---|
| `e5_large_coarse_rerank` | coarse | 88 | 0.870 | 0.839 | 0.709 | 0.824 | $0.00040 |
| **long-context** | — | 88 | 0.921 | 0.730 | 0.864 | 0.875 | $0.00378 |
