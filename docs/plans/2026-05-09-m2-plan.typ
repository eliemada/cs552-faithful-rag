#set page(paper: "a4", margin: 1.6cm)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)

#align(center)[
  #text(size: 14pt, weight: "bold")[CiteRight — M2 Plan]
  #v(-4pt)
  #text(size: 9pt)[CS-552 · Spring 2026 · drafted 2026-05-09 · rev. 2026-05-09]
]

#block(
  fill: rgb("#e9f7ee"),
  inset: 6pt,
  radius: 3pt,
  width: 100%,
)[
  #text(size: 9pt)[
    *Status — 2026-05-09:* Phase 1 (foundation) merged in
    #link("https://github.com/eliemada/cs552-faithful-rag/pull/9")[PR #9].
    Schema, validator, aggregator, IAA pipeline, CI gate — *shipped*. 26
    unit tests green. Annotation can start now.
  ]
]

== What the feedback told us
Scope is broad but feasible. For *M2 (May 24)* prioritise:
+ a small annotated citation-faithfulness benchmark,
+ *≥ 2* retrieval/citation configurations compared on it,
+ qualitative citation errors.

Push to 75–100 QA pairs *only* if quality holds. Defer the full ablation grid, CRAG, RAGAS, long-context to the final report (June 7).

== M2 targets (hard)
- *50 validated QA pairs* in `gold_qa.json`, frozen *May 16*.
- *2 retrieval configs* end-to-end with Recall\@k, MRR, nDCG.
- *Citation faithfulness* (NLI entailment) on the same 2 configs.
- *Failure-mode taxonomy* + *≥ 15 qualitative citation errors* in the appendix.

== Timeline (15 days)
#table(
  columns: (auto, auto, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  [*Dates*], [*Owner*], [*Deliverable*],
  [#strike[May 9–10]], [Elie], [#strike[Lock schema, rubric.] *Done — PR #9.*],
  [#strike[May 9–11]], [Faruk], [#strike[`validate_gold_qa.py`, IAA pipeline, κ.] *Done — PR #9. Faruk now starts Phase 5 (failure-mode taxonomy) earlier; see below.*],
  [May 9–16], [All], [13 QA pairs each → freeze v1 ≥ 50 pairs. Daily 15-min sync. *Annotate `contributions/<your-name>.json`, run `validate_gold_qa.py --strict` locally before PR.*],
  [May 11–18], [Elie], [Two retrieval configs: \ A = `text-embedding-3-small`, 300-char, no reranker \ B = `BGE-M3`, 2000-char, ZeroEntropy reranker. Start re-indexing May 11.],
  [May 13–20], [Andrea], [Faithfulness: claim split → DeBERTa-v3 NLI, `CitationSupport@τ` (τ = 0.5, 0.7).],
  [May 16–22], [Faruk], [Failure-mode taxonomy across both configs: hallucination · mis-citation · partial support · retrieval miss. Curate 15 errors → seeds CRAG triggers post-M2.],
  [May 20–24], [Yusif + all], [Progress report: methodology, dataset stats + κ, retrieval table, faithfulness table, qualitative appendix.],
)

== Post-M2 (May 24 → June 7)
- Elie: full retrieval grid (3 embed × 3 chunk × reranker).
- Andrea: cross-LLM faithfulness + claim-extractor sensitivity.
- Faruk: CRAG + threshold ablation.
- Yusif: RAGAS suite + 128K long-context baseline + cost/latency.
- All: expand gold to 75–100 *only if* κ on v1 > 0.6.

== Honest realism check
- *Throughput:* 13 pairs / 7 days ≈ 2/day/person · 30–45 min each. Tight but doable.
- *Failure mode:* nobody annotates until May 15. Mitigation = daily Slack count.
- *Critical path:* schema locked May 10, BGE-M3 re-index started May 11. Slipping either kills the parallelism.
- *Quality > quantity:* 50 clean pairs beat 80 noisy ones. Feedback says so explicitly.

== Decisions needed from the team
+ Adopt `claims[]` + char-span schema (replaces `chunk_id`)? *recommend yes*
+ Two configs A/B above — confirm or swap?
+ Annotation freeze hard at 50 on May 16, or push to 60?
+ Pin *one* generator (`gpt-4o-mini`) for M2, cross-LLM for final?
+ Roles still: Elie = retrieval, Andrea = faithfulness, Faruk = CRAG, Yusif = RAGAS?
