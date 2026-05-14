# Per-member contributions

Edit **only your own file**:

| Member  | File                |
|---------|---------------------|
| Elie    | `elie.json`         |
| Andrea  | `andrea.json`       |
| Faruk   | `faruk.json`        |
| Yusif   | `yusif.json`        |
| —       | `adversarial.json`  |

The `adversarial.json` file is a shared synthetic set of *control* claims —
deliberately designed to not entail their cited spans (negation flip,
quantitative drift, entity swap, scope overreach, etc.). These exist to
populate the `contradicts` and `unrelated` regions of the κ label space,
which natural annotation rarely produces. Don't add positive ("supports")
cases here.

## Why per-member files

Four people committing to one `gold_qa.json` is a merge-conflict factory.
Per-member files mean conflicts only happen when two people touch the same
file — which they shouldn't.

`gold_qa.json` at the parent level is the **aggregated** artifact, produced
by `scripts/aggregate_gold_qa.py`. CI verifies it stays in sync.

## ID convention

To prevent ID collisions across members without coordination:

| Member       | Question id range |
|--------------|-------------------|
| Elie         | `q001` … `q099`   |
| Andrea       | `q100` … `q199`   |
| Faruk        | `q200` … `q299`   |
| Yusif        | `q300` … `q399`   |
| adversarial  | `q900` … `q999`   |

Claim IDs (`c001`, `c002`, ...) only need to be unique inside one QA pair.
