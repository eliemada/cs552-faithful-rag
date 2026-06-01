# Faithfulness human-eval — M3 plan (TA-feedback-driven)

**Owners:** Andrea (lead), Yusif (co-rater) · **Target:** annotations done by 2026-06-04 · **For §Results/Faithfulness and §Limitations in the final report.**

## Why this exists

TA feedback on M2:

> Your faithfulness section flags the 55–61 pp NLI-vs-judge disagreement, but doesn't provide information about the faithfulness of the judge LLM. Human evaluation would be a strong addition to validate your claims here.

The point: M2 reports a huge NLI-vs-judge gap and treats the judge as the truth, but never validates that *either* head matches a human's reading. Without that, the gap is uninterpretable — it could mean NLI is wrong (the M2 read), the judge is wrong, or both.

## What `evaluation/faithfulness/results/02_full_results.json` actually contains

(verified 2026-06-01 via `python -c "import json; ..."`)

- 535 claim-level rows across 72 (question, model) pairs
- 3 generators: `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`, `deepseek/deepseek-chat`
- NLI label distribution: `supported` 405 / `not_supported` 130
- Judge label distribution: `supported` 76 / `not_supported` 458 / `error` 1
- **371 claim-level disagreements**, **164 claims where both say `supported`**, **0 claims where both say `not_supported`**

The "0 both-not_supported" is the smoking gun: NLI overcalls supported, judge overcalls not_supported — they're literally never co-pessimistic. Human eval breaks the tie.

## Sampling strategy

Target: **80 claims × 2 raters = 160 annotations.** Stratified to maximize signal-per-annotation.

| Stratum                                                  | n   | Why                                                                       |
|----------------------------------------------------------|-----|---------------------------------------------------------------------------|
| NLI=supported, judge=not_supported                       | 30  | The majority disagreement class — measures whether judge over-rejects     |
| NLI=not_supported, judge=supported                       | 20  | The minority disagreement class — measures whether NLI over-rejects       |
| Both supported (control 1)                               | 20  | Establishes that humans also call obviously-supported claims supported    |
| Spread across all 3 generator models                     | —   | 10 from each model per stratum where possible                             |
| At least 5 per difficulty bucket (single/multi/unanswer) | —   | Avoid difficulty confound                                                 |

Both-not_supported control is **n=0 by construction** (no claims fall in that bucket). Document as a finding: NLI never agrees with judge on rejection.

A reproducible sampler is in `scripts/sample_human_eval.py` (to be added by the lead — stub below).

## Rater protocol

- **Independence:** raters annotate the same 80 claims in parallel without consulting each other. Cohen's κ requires independent labels.
- **Blinding:** raters do NOT see the NLI label, judge label, judge reason, or which model produced the answer until after submitting. (CSV template hides those columns; reveal during disagreement reconciliation only.)
- **Decision criterion:** for each claim, decide whether the cited `supporting_span` (the `quote` from the source paper) **entails** the claim. Use exactly the 2 labels NLI/judge use:
  - `supported` — span makes the claim true; no extra info or inference beyond paraphrase needed.
  - `not_supported` — span doesn't license the claim (could be too weak, off-topic, or contradicted).
- **No "partial" label.** Forcing binary matches NLI/judge so κ is comparable.
- **Time budget:** ~1 min/claim. 80 claims = ~80 min of focused work per rater.

## Metrics to report

1. **Inter-rater κ (human vs human)** — sanity check; if <0.6, refine the protocol before computing the other two.
2. **κ(human vs NLI)** — per stratum and overall.
3. **κ(human vs judge)** — per stratum and overall.
4. **Confusion matrices**, 2×2 per pair (human vs NLI; human vs judge).
5. **Disagreement breakdown:** in the NLI=sup, judge=not stratum, what fraction of human labels side with NLI vs judge?

The headline number for the report: **"on the 56 strict NLI-vs-judge disagreement claims, humans side with [NLI/judge] X% of the time, yielding κ(human, NLI) = a, κ(human, judge) = b."** Plug into §Results/Faithfulness.

## Code stubs needed

1. **`scripts/sample_human_eval.py`** (~50 lines)
   - Loads `02_full_results.json`, applies the stratified sampler with a fixed seed (recommend `seed=2026`).
   - Emits `evaluation/faithfulness/human_eval/sample.csv` with columns:
     `claim_uid, question_id, difficulty, claim_text, supporting_quote, paper_id`
     (NO model name, NO NLI label, NO judge label, NO judge reason — blinded.)

2. **`evaluation/faithfulness/human_eval/sample.csv`** — generated artifact, committed for traceability.

3. **`evaluation/faithfulness/human_eval/template_andrea.csv`** and **`_yusif.csv`** — pre-populated copies for each rater, with an empty `human_label` column. Raters fill in `supported` / `not_supported`.

4. **`scripts/compute_human_eval_kappa.py`** (~80 lines)
   - Joins the two rater CSVs with `02_full_results.json` (by `claim_uid`).
   - Computes κ(rater1, rater2), κ(majority_human, NLI), κ(majority_human, judge).
   - Emits `evaluation/faithfulness/human_eval/results.md` with the table for the report.

## Risks

- **Rater fatigue (MEDIUM):** 80 claims is right at the upper limit. If quality drops, split into two sessions.
- **Quote-only context (MEDIUM):** raters see the `quote` but not the full document. Some claims may need broader context. Mitigation: a `needs_context` flag column; if >10% of claims trigger it, expand the sample to include 2 surrounding sentences.
- **Two raters, one team (LOW):** rater non-independence is a known limitation. Document it; report κ as a lower bound on judge–human agreement, not as the definitive number.

## Out of scope

- Hiring external annotators — no budget/time.
- Re-running M2 NLI/judge on the new sample — already done; the labels in `02_full_results.json` are authoritative.
- Comparing >2 raters — would need 4 team members on the same 80 claims, infeasible in the time budget.
