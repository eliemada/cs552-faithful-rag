# Reviews & inter-annotator agreement

When you review someone else's pairs marked `iaa_subset: true`, edit their
contribution file and fill `reviewer`, `review_status`, and per-claim
`reviewer_label`. Don't create a separate file — keeping reviews colocated
with the pair simplifies κ computation.

`scripts/compute_iaa.py` reads every claim with both `annotator_label` and
`reviewer_label` set and reports Cohen's κ across the three labels:

- `supports` — span entails the claim.
- `contradicts` — span contradicts the claim.
- `unrelated` — span is off-topic.
