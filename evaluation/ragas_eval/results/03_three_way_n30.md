# RAGAS evaluation — preliminary

n = 30 gold questions, stratified across categories.

| metric | RAG · gemini-2.5-flash | RAG · gpt-4o-mini | LC · gemini-2.5-flash |
|---|---|---|---|
| `faithfulness` | 0.848 | 0.864 | 0.910 |
| `answer_relevancy` | 0.648 | 0.781 | 0.791 |
| `context_precision` | 0.775 | 0.774 | 0.833 |
| `context_recall` | 0.833 | 0.833 | 0.933 |

## Answer-LLM cost & tokens (judge calls excluded)

| measure | chunked RAG | long-context | ratio (LC / RAG) |
|---|---|---|---|
| total cost (USD) | $0.02240 | $0.13364 | 6.0× |
| total tokens     | 58,392 | 438,454 | 7.5× |
| avg cost / query | $0.00075 | $0.00445 | — |

### RAG-alt (RAG · gpt-4o-mini) cost

- total $0.00946, 56,704 tokens, avg $0.00032/query
