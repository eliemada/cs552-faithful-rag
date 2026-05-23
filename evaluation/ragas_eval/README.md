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

Every answer-LLM call is also instrumented for **token usage and cost**.
We prefer `resp.usage.cost` (the actual amount OpenRouter charged for the
call) over `litellm.completion_cost`'s static pricing table — the table
silently returned `$0` for `openrouter/openai/gpt-4o-mini` before this
fallback was added. Per-call usage is written into each per-sample
record under `usage.{prompt_tokens, completion_tokens, total_tokens,
cost_usd, model}`, and aggregated into the pipeline-level `usage` block
of the result JSON. *Judge* calls are not included in this cost; the
RAGAS judge runs inside the library and is not routed through
`generate_with_usage`.

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

Live numbers from the canonical `01_rag_vs_long_context` run (n=8, judge
calls *not* included — see below):

| pipeline | total cost | total tokens | per-query |
|---|---|---|---|
| Chunked RAG (gpt-4o-mini) | $0.003 | 18.4k | $0.00038 |
| Long-context (gemini-2.5-flash) | $0.031 | 114k | $0.00383 |

LC costs ≈ **10× more per query** for ≈ **6× more tokens** (the
extra factor is gemini-2.5-flash's higher per-token rate vs gpt-4o-mini).

Judge calls are RAGAS-internal and not in this table. Empirically the
judge adds another ~100-150 `gpt-4o-mini` calls per full experiment,
which is under $0.10 on the default setup. Scaling to the full
39/50-pair gold set ≈ $0.50 all-in.

## Known limitation: RAGAS judge variance (M2 caveat)

Two runs with identical seed, sample, and answer tokens produced metric
shifts of up to **+6.7 pp** on `faithfulness` (LC: 1.000 → 0.933). The
underlying answer tokens were byte-identical across runs, so the drift
is in the judge, not the answers.

Root cause is visible in the RAGAS warning during the run:

```
WARNING:ragas.prompt.pydantic_prompt: LLM returned 1 generations
instead of requested 3. Proceeding with 1 generations.
```

RAGAS uses self-consistency (n=3 generations per judge call) for
faithfulness; `openrouter/openai/gpt-4o-mini` does not honor `n>1` and
silently returns one. Self-consistency collapses, and a single noisy
judgment moves the metric by ~0.1 per sample.

Reading guidance for the M2 report:

- `context_precision` and `context_recall` are **stable across reruns**
  (they evaluate retrieval coverage, not generative faithfulness).
- `faithfulness` and `answer_relevancy` have ±5 pp judge-noise floor at
  n=8; deltas smaller than that are not real.
- The RAG-vs-LC **direction** is robust across reruns even where
  magnitudes shift; the cost-per-query ratio is robust too.

## Out of scope for M2 (queued for M3)

- Full 39/50-pair sweep
- Generator-model ablation (gpt-4o-mini vs claude-haiku-4-5 vs deepseek-chat)
- All four retriever configs (currently only one is used per run)
- Wiring the existing NLI faithfulness scorer into the comparison
  alongside the LLM-judge faithfulness (would let us replicate the
  57 pp gap finding from PR #19 inside the RAGAS framing)

## Model recommendations for M3

The biggest lever for tightening the headline numbers is the **judge
model**, not the answer model.

| role | M2 setting | M3 recommendation | rationale |
|---|---|---|---|
| RAGAS judge | `openai/gpt-4o-mini` | **`anthropic/claude-haiku-4-5`** (or `claude-sonnet-4-5`) | Honors `n>1` for self-consistency, tighter judgments. Haiku-4-5 ~5× the per-call cost (~$0.50/run) — still cheap. Sonnet-4-5 ~15× ($2-3/run) — defensible if cited in the report as a stronger judge. |
| RAG answer | `openai/gpt-4o-mini` | keep, *or* add `claude-haiku-4-5` as an ablation arm | gpt-4o-mini is fine for the role. Swapping confounds the retrieval comparison; better to *add* it as a row. |
| LC answer | `google/gemini-2.5-flash` | **keep** | 1M-token context window at $0.075/$0.30 per 1M is irreplaceable for the LC baseline. Gemini-2.5-pro is an option only if cost stops mattering. |
| Embeddings (answer-relevancy) | `text-embedding-3-small` | keep, optionally try `text-embedding-3-large` | ~6× cost for a marginal answer_relevancy gain; not worth it unless that metric is doing real lifting in the M3 conclusions. |

Switching the judge alone is a one-flag change at the CLI:

```bash
uv run python -m scripts.run_ragas_experiment \
  --judge-model anthropic/claude-haiku-4-5
```

## Where this fits in the project

The proposal commits four evaluation components:

1. Retrieval ablation (Elie) — shipped (PR #20)
2. Citation verification (Andrea) — shipped (PR #14, PR #19)
3. Corrective RAG (Faruk)
4. **This module — RAGAS + long-context baseline**

RAGAS + the long-context baseline is the closest thing the project has
to a "headline accuracy number" for the full system. Final-report
comparisons feed off this.
