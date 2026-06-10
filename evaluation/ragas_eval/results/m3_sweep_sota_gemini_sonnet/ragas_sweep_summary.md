# RAGAS sweep — M3

n = 88 evaluable gold questions (shared across configs).

| config | chunk | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|---|
| `e5_rerank_s800_o0` | s800_o0 | 88 | 0.921 | 0.741 | 0.806 | 0.864 | $0.00075 |
| **long-context** | — | 88 | 0.955 | 0.753 | 0.852 | 0.869 | $0.00374 |
