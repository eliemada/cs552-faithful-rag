# RAGAS sweep — M3

n = 88 evaluable gold questions (shared across configs).

| config | chunk | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|---|
| `e5_rerank_s800_o0` | s800_o0 | 88 | 0.806 | 0.825 | 0.797 | 0.847 | $0.00029 |
| **long-context** | — | 88 | 0.922 | 0.731 | 0.864 | 0.875 | $0.00360 |
