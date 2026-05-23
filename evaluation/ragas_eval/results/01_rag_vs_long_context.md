# RAGAS evaluation — preliminary

n = 8 gold questions, stratified across categories.

| metric | chunked RAG | long-context | Δ (LC − RAG) |
|---|---|---|---|
| `faithfulness` | 0.899 | 0.933 | +0.035 |
| `answer_relevancy` | 0.713 | 0.789 | +0.076 |
| `context_precision` | 1.000 | 1.000 | -0.000 |
| `context_recall` | 0.917 | 1.000 | +0.083 |

## Answer-LLM cost & tokens (judge calls excluded)

| measure | chunked RAG | long-context | ratio (LC / RAG) |
|---|---|---|---|
| total cost (USD) | $0.00304 | $0.03063 | 10.1× |
| total tokens     | 18,476 | 114,466 | 6.2× |
| avg cost / query | $0.00038 | $0.00383 | — |
