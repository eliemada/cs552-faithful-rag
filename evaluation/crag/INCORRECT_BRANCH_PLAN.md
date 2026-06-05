# CRAG INCORRECT-branch concretisation — M3 plan (TA-feedback-driven)

**Owner:** Faruk · **Target:** branch logic + 3 qualitative examples landed by 2026-06-04 · **For §Approach/CRAG-router and §Results/CRAG in the final report.**

> **Status (2026-06-05): Option A IMPLEMENTED and landed on `main` via PR #72.**
> `corrective_rag.py` abstains on INCORRECT / exhausted-AMBIGUOUS
> (`final_documents=[]`, `abstained=True`); `use_abstain_fallback` (default
> `True`) and `finalize_answer()` emit `abstain_message` instead of calling the
> generator. `tests/test_crag.py` covers abstain + adapter-text regressions (19
> tests). Full-gold ablation: `scripts/run_crag_ablation.py` (probe once +
> analytical sweep). **Headline n=97 result** (post adapter-text fix, τ_h=0.7,
> τ_l=0.4, `e5_large_coarse_rerank`): `{correct: 85, ambiguous: 9, incorrect: 3}`;
> baseline direct-correct retrieval **0.876**. Report numbers in PR #73.

## Why this exists

Intermediate (M2) feedback called out:

> a brief note on how the experiments are planned across the team, and what the **INCORRECT-branch fallback** will actually do, would turn the high-level plan into something more concrete.

Status as of 2026-06-01 (verified by reading `evaluation/crag/corrective_rag.py:195-207`):

```python
# INCORRECT, or retries exhausted.
if config.use_web_fallback:
    # TODO: implement web search fallback
    pass
return CRAGResult(... quality=INCORRECT, final_documents=documents)
```

So today, when the cross-encoder labels retrieval `INCORRECT`, CRAG silently returns the same (bad) documents. The "+20pp on a mixed slice" headline number is therefore driven entirely by the **AMBIGUOUS** branch — the INCORRECT branch is decorative. The report can't honestly call this "Corrective" RAG without making the INCORRECT path do something.

## What the INCORRECT branch will actually do (the decision)

Three candidate behaviors, ranked by implementation cost vs. defensibility:

| Option | What it does                                                                 | Cost  | Defensible in §Approach?                                                    |
|--------|------------------------------------------------------------------------------|-------|------------------------------------------------------------------------------|
| **A — abstain** | Return `final_documents=[]` and an abstain marker. Downstream generator emits "I don't have enough evidence to answer." | LOW   | YES — matches the unanswerable-question stratum in our gold; honest fallback. |
| B — broaden + dedup | Re-retrieve at top-50 with the reranker off, then dedup against the original top-k. | MED   | Partially — measures recall under loosened constraints; risk of polluting context. |
| C — web search | Off-corpus query via Tavily / Brave / Google CSE.                            | HIGH  | YES per CRAG paper, NO for our scope: contradicts the "domain-specific scientific literature" framing of the project (off-corpus = off-topic). |

**Decision: Option A (abstain).** Reasoning:

1. Our project framing is faithfulness *to a specific corpus*. Web fallback would conflict with that — the paper would have to argue why off-corpus retrieval is still "faithful".
2. Abstain is precisely the behavior the M2 gold already evaluates: the **unanswerable** stratum tests whether the system correctly says "no answer". Wiring the INCORRECT branch to abstain lets us measure abstain-precision/recall using the existing gold.
3. Adds a meaningful comparison point: chunked-RAG and long-context never abstain; CRAG-with-abstain trades coverage for precision.

Option B is logged in `corrective_rag.py` as a future extension; Option C is explicitly out of scope and noted in §Ethical considerations (off-corpus retrieval would change the threat model).

## Code change

In `evaluation/crag/corrective_rag.py` around line 195:

```python
# INCORRECT, or retries exhausted.
if config.use_abstain_fallback:  # rename from use_web_fallback
    return CRAGResult(
        original_query=query,
        quality=RetrievalQuality.INCORRECT,
        confidence=confidence,
        refined_query=refined_marker,
        retrieval_rounds=retrieval_rounds,
        final_documents=[],          # <- the abstain
        abstained=True,              # <- new flag (add to dataclass)
    )
```

Plus a `CRAGResult.abstained: bool = False` field. Generator wrapper checks this and emits an abstain string instead of running the LLM. ~30 lines total including the dataclass field.

## Evaluation on the 93-question gold

Run CRAG end-to-end on all 93 gold pairs (M2 only ran 10). Report:

1. **Branch distribution:** how many questions are routed to CORRECT / AMBIGUOUS / INCORRECT at threshold `(τ_h=0.7, τ_l=0.4)` (M2's latency-optimal cell).
2. **Abstain precision/recall** on the unanswerable subset: when CRAG abstains, is the gold also unanswerable?
3. **Δ correct-retrieval pre/post** on the full 93 (M2 only had 10-sample headline; refresh).
4. **Latency budget** vs. M2's 0.04–0.40 s/query range — abstain is free, so should narrow the band.

## Qualitative examples for the report (1 per branch)

Pick representative cases from the 93-question run and include their full trace:

- **CORRECT example:** a single-hop policy_impact question where the top-1 chunk is the literal `supporting_quote` from `gold_qa.json`. Show the retrieved chunk + the gold span overlap.
- **AMBIGUOUS example:** a multi-hop question where the original query retrieves one of the two needed chunks; show the LLM-refined query and the second retrieval round.
- **INCORRECT-and-abstain example:** an `unanswerable` gold pair where CRAG correctly abstains. Show the cross-encoder score (below `τ_l`) and the empty `final_documents`.

These three traces go into Faruk's individual notebook (`notebooks/faruk_zahiragic_415360.ipynb`) and one summarized table into §Results.

## Risks

- **Threshold drift (MEDIUM):** M2 thresholds were tuned on a 10-question slice. Re-tuning on the full 93 may shift `(τ_h, τ_l)`. Plan: keep M2's values and report both M2-tuned and a fresh 93-question sweep if time allows; if not, note the limitation.
- **Abstain failure mode (LOW):** if CRAG abstains on answerable questions due to a too-strict `τ_l`, abstain-precision will be high but recall will tank. Watch this in the metrics; if recall <0.7 on answerable items, raise `τ_l` to 0.3.
- **Reproducibility (LOW):** the cross-encoder + LLM-refinement step is stochastic. Pin model versions + temperature=0 in the run script.

## Out of scope

- Web search fallback (Option C). Documented as future work.
- Multi-step refinement chains (e.g. INCORRECT → broaden → re-route to AMBIGUOUS). Single-pass abstain only.
- CRAG on top of the long-context baseline — the long-context system doesn't have a retrieval-quality signal, so CRAG doesn't apply.
