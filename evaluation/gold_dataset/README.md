# Gold Evaluation Dataset

## Purpose
50-100 expert-annotated Q&A pairs for evaluating RAG pipeline quality.

## Format
Each entry in `gold_qa.json` should have:
```json
{
  "id": "q001",
  "question": "What is the impact of patent protection on innovation in developing countries?",
  "gold_answer": "The expected answer based on the literature...",
  "gold_passages": [
    {
      "paper_id": "02596_W1962380625",
      "passage": "The exact passage from the paper that answers this...",
      "chunk_id": "coarse_1234"
    }
  ],
  "difficulty": "single-hop",
  "category": "policy_impact",
  "annotator": "first_last"
}
```

## Categories
- `policy_impact` — Questions about effects of policies
- `methodology` — Questions about research methods used
- `comparison` — Questions comparing approaches/countries/sectors
- `factual` — Direct factual extraction
- `multi_hop` — Requires combining info from multiple papers

## Difficulty Levels
- `single-hop` — Answer found in one passage
- `multi-hop` — Requires combining 2+ passages
- `unanswerable` — Not covered by the corpus (tests rejection)

## Assignment
Each team member writes 12-25 Q&A pairs:
- [ ] Person 1: 
- [ ] Person 2: 
- [ ] Person 3: 
- [ ] Person 4: 
