# RAGAS evaluation — preliminary

n = 8 gold questions, stratified across categories.

| metric | chunked RAG | long-context | Δ (LC − RAG) |
|---|---|---|---|
| `faithfulness` | 0.806 | 0.975 | +0.169 |
| `answer_relevancy` | 0.662 | 0.799 | +0.138 |
| `context_precision` | 0.979 | 1.000 | +0.021 |
| `context_recall` | 0.792 | 1.000 | +0.208 |
