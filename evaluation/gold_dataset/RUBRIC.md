# Annotation rubric

Read this before writing your first pair. ~10 minutes.

## What we're building

A small, **high-quality** benchmark to measure whether a RAG system's
citations actually support the claims in its answer. 50 pairs × 1–4 claims
each × 1+ supporting span per claim = ~120 entailment data points. Quality
beats quantity — feedback was explicit on this.

## Per-pair budget

Aim for ~30–45 minutes per pair. If a pair takes > 1 h, skip it and write a
different one — that's a sign the corpus doesn't really cover it well.

## What makes a good question

✅ Answerable from the corpus (or cleanly *un*answerable — see below).
✅ Specific enough that the gold answer is short (1–3 sentences).
✅ The kind of question someone doing a real lit review would ask.
✅ Verifiable against a passage you can copy-paste.

❌ Trick questions or rare phrasings.
❌ Yes/no questions ("Is X true?" — make it "What does paper X find about Y?").
❌ Questions whose answer is a single number lifted from a table (the corpus
   was OCR'd; tables are noisy — favour prose).
❌ Anything where you'd need to run new analysis to answer.

## Difficulty buckets — pick deliberately

- **single-hop** — answer in a single passage from a single paper. Default.
- **multi-hop** — requires combining ≥ 2 passages, possibly across papers.
  The `claims[]` array should reflect this: one claim per supporting passage.
- **unanswerable** — corpus does *not* cover this. We need ≥ 5 of these
  total to test rejection. Per person: aim for ≥ 1, ≤ 3.

## Categories — pick one

- `policy_impact` — effect of a policy / instrument (patent reforms,
  subsidies, tax credits, ...).
- `methodology` — what method/data did paper X use?
- `comparison` — A vs B (countries, sectors, instruments).
- `factual` — direct numerical/textual extraction.
- `multi_hop` — only when no other category fits.

## Splitting the gold answer into claims

A claim is **one verifiable proposition**. Rule of thumb: if you can imagine
a reviewer saying "I agree that bit, but not that bit", split there.

Example (single-hop, one claim):

> "Branstetter et al. (2006) find a positive effect of patent reforms on FDI."

→ one claim, one supporting span.

Example (multi-hop, two claims):

> "Park (2008) reports that patent strength rose globally between 1960 and
> 2005, while Branstetter (2006) shows this rise was followed by increased
> technology-transfer-intensive FDI."

→ two claims:
- c001: "Patent strength rose globally between 1960 and 2005" → cite Park.
- c002: "The rise was followed by increased technology-transfer-intensive
  FDI" → cite Branstetter.

## Choosing a supporting span

- The span must **literally entail** the claim. If you have to squint, it
  doesn't. Either rewrite the claim or pick a different span.
- Keep spans tight (1–3 sentences). Long spans dilute the entailment signal
  and make NLI noisier.
- Multiple short spans > one long span. The schema supports it.
- Quote must be **verbatim** — copy-paste, don't re-type.

## Self-checks before you commit

- [ ] Does `markdown[char_start:char_end] == quote`? Run
      `uv run python scripts/validate_gold_qa.py --strict` locally.
- [ ] Could a peer reviewer label every claim "supports" given only the span?
- [ ] Is the gold answer fully grounded in the spans (no extra info)?
- [ ] Is the question something a researcher would actually type?

## Per-person targets

| Member  | Pairs | Of which `unanswerable` | Of which `multi-hop` |
|---------|-------|--------------------------|-----------------------|
| Elie    | 13    | ≥ 1                      | ≥ 2                   |
| Andrea  | 13    | ≥ 1                      | ≥ 2                   |
| Faruk   | 13    | ≥ 1                      | ≥ 2                   |
| Yusif   | 13    | ≥ 1                      | ≥ 2                   |

Total target: 52 pairs, ≥ 4 unanswerable, ≥ 8 multi-hop.

## Inter-annotator agreement (IAA)

Each annotator marks `iaa_subset: true` on **5 of their 13 pairs**. After
the freeze, `compute_iaa.py` computes Cohen's κ on the
`(annotator_label, reviewer_label)` pairs across all claims in the IAA
subset. Target κ ≥ 0.6 before we trust the benchmark for headline numbers.

## Workflow

1. Branch off `main`, edit only `contributions/<your-name>.json`.
2. Run `uv run python scripts/validate_gold_qa.py --strict` locally.
3. Run `uv run python scripts/aggregate_gold_qa.py` to refresh `gold_qa.json`.
4. Open a PR — CI runs the fast validator. Get one teammate to review.
