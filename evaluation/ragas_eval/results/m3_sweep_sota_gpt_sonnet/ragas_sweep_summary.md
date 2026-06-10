# RAGAS sweep — M3

n = 88 evaluable gold questions (shared across configs).

| config | chunk | n | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|---|---|
| `e5_rerank_s800_o0` | s800_o0 | 88 | 0.861 | 0.819 | 0.795 | 0.852 | $0.00032 |
| **long-context** | — | 88 | 0.958 | 0.752 | 0.852 | 0.875 | $0.00385 |
