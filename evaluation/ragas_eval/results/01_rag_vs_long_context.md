# RAGAS evaluation — preliminary

n = 8 gold questions, stratified across categories.

| metric | chunked RAG | long-context | Δ (LC − RAG) |
|---|---|---|---|
| `faithfulness` | 0.824 | 1.000 | +0.176 |
| `answer_relevancy` | 0.739 | 0.796 | +0.056 |
| `context_precision` | 1.000 | 1.000 | -0.000 |
| `context_recall` | 0.917 | 1.000 | +0.083 |

## Answer-LLM cost & tokens (judge calls excluded)

| measure | chunked RAG | long-context | ratio (LC / RAG) |
|---|---|---|---|
| total cost (USD) | $0.00270 | $0.03560 | 13.2× |
| total tokens     | 18,403 | 114,466 | 6.2× |
| avg cost / query | $0.00034 | $0.00445 | — |
