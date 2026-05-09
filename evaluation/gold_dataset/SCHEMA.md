# Gold Q&A schema (v0.1)

The canonical machine-readable spec is [`SCHEMA.json`](./SCHEMA.json) — JSON
Schema 2020-12. This file is the human walkthrough.

## Why claims, not "one passage per question"

Faithfulness is scored **per atomic claim**, not per question. Splitting the
gold answer into claims at annotation time means:

- The NLI scorer (DeBERTa-v3) gets one `(claim, passage)` pair to entail.
- We can localise *which* claim was hallucinated when a citation is wrong.
- A single multi-hop QA pair contributes several entailment data points.

Single-hop questions usually have one claim. Multi-hop questions have 2–4.
Unanswerable questions have **zero** claims.

## Why `(paper_id, char_start, char_end, quote)` and not `chunk_id`

Chunks have already been re-derived once and will drift again when we sweep
chunk sizes. Char-spans on the LF-normalised `document.md` are stable across
re-chunking. The `quote` field is a self-check: the validator slices
`document.md[char_start:char_end]` and asserts it matches `quote` after
whitespace normalisation.

## One example

```json
[
  {
    "id": "q001",
    "question": "What does Branstetter (2006) find about the effect of TRIPS-style patent strengthening on inward FDI in developing countries?",
    "gold_answer": "Branstetter et al. (2006) find a positive and significant effect of patent reforms on technology transfer via FDI, with the effect concentrated in technology-intensive industries.",
    "difficulty": "single-hop",
    "category": "policy_impact",
    "annotator": "elie",
    "reviewer": null,
    "review_status": "pending",
    "iaa_subset": false,
    "claims": [
      {
        "id": "c001",
        "text": "Branstetter et al. (2006) find a positive and significant effect of patent reforms on technology transfer via FDI.",
        "supporting_spans": [
          {
            "paper_id": "02596_W1962380625",
            "char_start": 12345,
            "char_end": 12498,
            "quote": "We find that patent reforms in 16 countries between 1982 and 1999 are associated with a significant increase in technology-transfer-intensive FDI ..."
          }
        ],
        "annotator_label": "supports",
        "reviewer_label": null
      }
    ]
  }
]
```

## Unanswerable example

```json
{
  "id": "q042",
  "question": "What is the median patent litigation cost in EPFL spin-offs after 2024?",
  "gold_answer": "Not covered by the corpus.",
  "difficulty": "unanswerable",
  "category": "factual",
  "annotator": "faruk",
  "reviewer": null,
  "review_status": "pending",
  "iaa_subset": false,
  "claims": []
}
```

## Span coordinates — the only tricky part

Offsets are **Python `str` codepoint offsets** into the document after
LF-normalisation. The validator does this normalisation for you:

```python
markdown = path.read_text().replace("\r\n", "\n").replace("\r", "\n")
slice_   = markdown[char_start:char_end]
assert slice_ == quote
```

That is: load the file, normalise CRLF → LF, then slice. **Not** byte offsets.
**Not** UTF-16 code units. Just `len(s)` semantics on `str`.

To find a span while annotating, the easiest path is:

```python
from evaluation.common.data_loader import load_paper_markdown
md = load_paper_markdown("02596_W1962380625")
needle = "We find that patent reforms"
start  = md.find(needle)
end    = start + len(needle)  # extend to whatever boundary is faithful
```

## Versioning

Breaking changes get a new schema file (`SCHEMA.v0.2.json`) and a fresh PR.
Don't change `v0.1` once the M2 freeze is in.
