# RAGAS evaluation — preliminary

n = 8 gold questions, stratified across categories.

| metric | RAG · gpt-4o-mini | RAG · claude-haiku-4-5 | LC · gemini-2.5-flash |
|---|---|---|---|
| `faithfulness` | 0.756 | 0.926 | 0.881 |
| `answer_relevancy` | 0.748 | 0.738 | 0.743 |
| `context_precision` | 0.598 | 0.598 | 0.937 |
| `context_recall` | 0.750 | 0.750 | 0.875 |

## Answer-LLM cost & tokens (judge calls excluded)

| measure | chunked RAG | long-context | ratio (LC / RAG) |
|---|---|---|---|
| total cost (USD) | $0.00289 | $0.03201 | 11.1× |
| total tokens     | 18,514 | 114,466 | 6.2× |
| avg cost / query | $0.00036 | $0.00400 | — |

### RAG-alt (RAG · claude-haiku-4-5) cost

- total $0.02401, 20,445 tokens, avg $0.00300/query
