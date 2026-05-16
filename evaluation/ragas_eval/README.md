# RAGAS end-to-end evaluation

Runs the four canonical [RAGAS](https://docs.ragas.io/) metrics over two
answer-generation pipelines: the chunked-RAG system under evaluation and
a long-context baseline that puts the full paper(s) directly into the
model's context window. Owns the M2 "preliminary end-to-end numbers"
deliverable.

## What it measures

| Metric | Catches | Signal |
|---|---|---|
| **Faithfulness** | Answer-side hallucination | Are the generated claims actually supported by the retrieved context? |
| **Answer relevancy** | Off-topic answers | Does the answer actually address the question? |
| **Context precision** | Retriever noise | Are the retrieved passages relevant to the question's answer? |
| **Context recall** | Retriever gaps | Does the retrieved context contain all the information needed to answer? |

All four are LLM-as-judge metrics; the project routes the judge LLM
(`gpt-4o-mini` by default) through OpenRouter and the embeddings used by
answer-relevancy directly through OpenAI.

## What it compares

| Pipeline | How context is selected | Why it's in the comparison |
|---|---|---|
| **Chunked RAG** | `HybridRetriever` (FAISS + ZeroEntropy rerank) returns top-k chunks | The system the proposal committed to building |
| **Long-context** | Full `document.md` of every cited paper | The proposal-committed baseline (128k window) |

Both pipelines route LLM calls through `evaluation.common.models.generate()`
using the project's `api:` model-spec convention, so swapping
`gpt-4o-mini` for any OpenRouter-served model is a one-flag change.

## M2 scope

This module is sized for **preliminary** results (Rubi's brief: "2-3
clean comparisons that produce a conclusion"). The default
`--sample 8` runs RAG + long-context on 8 stratified gold questions and
costs a small handful of judge calls. The full 50-pair sweep is M3 work
once we've validated the metric pipeline.

## Run one experiment

```bash
uv run python -m scripts.run_ragas_experiment
```

Defaults: 8-question stratified sample, `coarse_rerank` retriever,
`gpt-4o-mini` for RAG answers + judge, `gemini-2.5-flash` for the
long-context answers (Gemini's 1M-token window handles even the 80k-char
papers without truncation; we still cap at 60k chars/paper by default
as a budget knob).

Writes:

- `evaluation/ragas_eval/results/01_rag_vs_long_context.json` — full
  per-sample trace
- `evaluation/ragas_eval/results/01_rag_vs_long_context.md` — Markdown
  comparison table for the M2 report

```bash
# Cheaper smoke test
uv run python -m scripts.run_ragas_experiment --sample 2

# Skip long-context (e.g. when iterating on the judge prompt)
uv run python -m scripts.run_ragas_experiment --skip-long-context

# Different retriever config (must be one defined in retrievers.CONFIGS)
uv run python -m scripts.run_ragas_experiment --retriever-config fine_rerank
```

## Required env vars

```bash
OPENROUTER_API_KEY   # answer LLM + judge LLM
OPENAI_API_KEY       # query embedding + RAGAS answer-relevancy embedding
ZEROENTROPY_API_KEY  # only if the retriever-config name ends with _rerank
```

The script fails fast if any required key is missing rather than burning
compute on the answer-generation step before discovering it can't score.

## Architecture

```
evaluation/ragas_eval/
├── pipelines.py        # RAG + long-context answer generators (→ RagasSample)
├── ragas_runner.py     # evaluate_samples(): the 4 RAGAS metrics, OpenRouter-backed judge
├── results/            # per-experiment JSON + Markdown summaries
└── README.md           # this file

scripts/run_ragas_experiment.py  # the CLI you actually run
tests/test_ragas_eval.py         # unit tests for pipeline + runner glue
```

`RagasSample` carries the four columns RAGAS needs (`question`, `answer`,
`contexts`, `ground_truth`) plus provenance (`pipeline`, `query_id`)
that `to_ragas_dict()` strips before handing to the library.

## Cost notes (M2 budget realism)

For one full run on the default 8-question sample × both pipelines:

- 8 × 2 = 16 answer generations (cheap, gpt-4o-mini)
- 4 metrics × 16 samples × judge calls — RAGAS' `Faithfulness` alone is
  ~3 calls per claim per sample. Empirically ≈ 100-150 judge calls per
  full experiment.

At `gpt-4o-mini` rates this is under \$0.10. Scaling to the full 39-pair
gold set ≈ \$0.50. Scaling to 50 pairs × 3 retriever configs × 3
generator models = \$5-10 — fine for M3 but disproportionate for M2.

## Out of scope for M2 (queued for M3)

- Full 39/50-pair sweep
- Generator-model ablation (gpt-4o-mini vs claude-3.5-haiku vs deepseek-chat)
- All four retriever configs (currently only one is used per run)
- Cost / latency reporting in the comparison table
- Wiring the existing NLI faithfulness scorer into the comparison
  alongside the LLM-judge faithfulness (would let us replicate the
  57 pp gap finding from PR #19 inside the RAGAS framing)

## Where this fits in the project

The proposal commits four evaluation components:

1. Retrieval ablation (Elie) — shipped (PR #20)
2. Citation verification (Andrea) — shipped (PR #14, PR #19)
3. Corrective RAG (Faruk)
4. **This module — RAGAS + long-context baseline**

RAGAS + the long-context baseline is the closest thing the project has
to a "headline accuracy number" for the full system. Final-report
comparisons feed off this.
