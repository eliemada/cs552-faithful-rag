# Faithfulness human-eval — results

## Sample

- Joined rows (both raters labelled, ground-truth available): **70**
- Inter-rater (A, B) agreement: po = 0.800

## Headline κ

| Pair                       | κ     | n  |
|---|---|---|
| rater_a vs rater_b         | 0.583 | 70 |
| majority_human vs NLI      | -0.454 | 56 |
| majority_human vs judge    | 0.882 | 56 |

## Per-stratum breakdown

| stratum | n (raters agree) | human = NLI | human = judge |
|---|---|---|---|
| both_supported | 18 | 18/18 (100%) | 18/18 (100%) |
| nli_not_judge_sup | 17 | 0/17 (0%) | 17/17 (100%) |
| nli_sup_judge_not | 21 | 3/21 (14%) | 18/21 (86%) |

## Confusion matrices

### rater A vs rater B
|                 | rater_b=supported | rater_b=not_supported |
|---|---|---|
| rater_a=supported    | 38 | 0 |
| rater_a=not_supported| 14 | 18 |


### majority_human vs NLI
|                 | NLI=supported | NLI=not_supported |
|---|---|---|
| human=supported    | 21 | 17 |
| human=not_supported| 18 | 0 |


### majority_human vs judge
|                 | judge=supported | judge=not_supported |
|---|---|---|
| human=supported    | 35 | 3 |
| human=not_supported| 0 | 18 |


## Notes for the report

- The headline number to cite in §Results/Faithfulness is **κ(human, judge) = 0.882**
  with **κ(human, NLI) = -0.454** as the contrast.
- Inter-rater κ = 0.583 bounds how seriously the two κ values
  above can be read. If it's below ~0.6, the protocol needs to be revisited
  before drawing strong conclusions.
