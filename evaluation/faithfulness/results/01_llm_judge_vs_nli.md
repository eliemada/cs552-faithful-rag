# Experiment 01 — LLM Judge vs NLI for citation faithfulness

## Question
Are NLI models (DeBERTa-v3) sufficient to detect hallucinated citations, or do they miss specific-but-wrong claims?

## Setup
- Gold pair: q004 (Shapiro 2001 on transaction costs from overlapping patents)
- Three LLMs answered the question without retrieval (cold, from training data)
- Each answer was decomposed into atomic claims using GPT-4o-mini
- Each claim was verified two ways:
  - NLI: zero-shot DeBERTa-v3-small (entailment / contradiction / neutral)
  - LLM judge: GPT-4o-mini, prompted to require specific entities/terms in the claim to appear in the passage

## Result

| Model                      | NLI faithfulness | LLM-judge faithfulness |
|----------------------------|-----------------:|-----------------------:|
| openai/gpt-4o-mini         |           100.0% |                   0.0% |
| anthropic/claude-3.5-haiku |           100.0% |                  12.5% |
| deepseek/deepseek-chat     |           100.0% |                   0.0% |

The gold answer (per Shapiro 2001) names two transaction costs: the **complements problem** and the **hold-up problem**. None of the three LLMs produced these terms. Instead they invented:

- GPT-4o-mini: "negotiation costs" and "enforcement costs"
- Claude: "search costs" and "licensing costs"
- DeepSeek: "search costs" and "bargaining costs"

NLI labelled every claim as supported, because the hallucinated terms are topically compatible with the passage's general statement that transaction costs burden innovation. The LLM judge labelled almost all of them as not-supported, because the specific terms in the claim do not appear in the passage.

## Why NLI fails here

NLI checks whether the passage logically entails the claim. A vague claim ("transaction costs hinder innovation") IS entailed by a passage that names specific transaction costs and says they burden innovation, even if the claim's named cost is different from the passage's. NLI is doing what it was trained to do — detect logical compatibility, not citation specificity.

## Why this matters for the project

RAG with citations exists precisely to stop this kind of hallucination. A faithfulness metric that gives 100% to answers full of fabricated specifics provides false reassurance. The dual-check approach catches both failure modes:

- NLI catches claims that directly contradict the passage
- LLM judge catches claims that are topically related but factually wrong

## Reproduce

    uv run python scripts/run_faithfulness_experiment.py

Requires OPENROUTER_API_KEY in .env. Cost per run: ~$0.02.
