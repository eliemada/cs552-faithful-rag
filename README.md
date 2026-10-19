# Faithful RAG

**CS-552 Modern NLP, EPFL, Spring 2026.** Team CiteRight.

When a RAG system cites a scientific paper, does the cited passage support the claim? We use 999 IP and innovation-policy papers to measure how retrieval, chunking, and corrective routing affect citation support.

The RAG pipeline comes from our earlier [CS-433 project](https://github.com/eliemada/cs433_ml_project_2). This repo adds evaluation. Pipeline docs live in [docs/PIPELINE.md](docs/PIPELINE.md).

Full write-up: [report/final_report/main.tex](report/final_report/main.tex).

## What we built

- **Gold benchmark.** 97 atomic claims across 88 answerable queries, each tied to a ground-truth span. ([evaluation/gold_dataset](evaluation/gold_dataset))
- **Retrieval ablation.** 16 configurations over embedding models, chunk sizes, and reranking, plus a 9-variant chunker sweep. ([evaluation/retrieval_eval](evaluation/retrieval_eval))
- **Faithfulness verifier.** We compare scores from DeBERTa-v3 NLI and an LLM judge with human ratings. ([evaluation/faithfulness](evaluation/faithfulness))
- **Corrective RAG.** A CRAG router ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) with an abstain fallback. ([evaluation/crag](evaluation/crag))
- **Long context vs. chunked RAG.** A RAGAS comparison against a 1M-token baseline, with cost in USD. ([evaluation/ragas_eval](evaluation/ragas_eval), [evaluation/long_context](evaluation/long_context))

## Results (n = 88)

- An 800-char chunker with no overlap lifts E5-large MRR from 0.865 to **0.902** and nDCG@10 from 0.882 to 0.895.
- NLI and the LLM judge disagree on 55 to 61 pp of claims. On a blinded 70-claim human eval, the judge reaches Cohen's κ = 0.882 against humans. NLI scores κ = −0.454, so we dropped it as a faithfulness gate.
- CRAG sorts the gold set into 85 correct, 9 ambiguous, and 3 incorrect. That caps refinement at +9.3 pp over an 87.6% baseline. Abstain catches 1 of 9 unanswerable queries.
- Long context beats chunked RAG on faithfulness (+3.7 pp with a Sonnet 4.6 judge, +11.4 pp with Haiku 4.5) and on context precision. It costs 13× more in answer-LLM spend. Answer relevancy depends on which judge you use.

## Layout

```
evaluation/   CS-552 work: gold set, retrieval, faithfulness, CRAG, RAGAS, long context
notebooks/    one notebook per team member (run on EPFL RCP)
report/       proposal, literature review, progress report, final report (LaTeX)
packages/     RAG pipeline from CS-433 (shared, api, worker)
frontend/     Next.js UI
scripts/      utilities and benchmarks
docs/         pipeline docs and team guide
```

## Running the notebooks

The notebooks run on EPFL's RCP cluster. See [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) and [rcp_support/README.md](rcp_support/README.md). The notebook bootstrap pulls the corpus from [citeright/corpus](https://huggingface.co/datasets/citeright/corpus) on first run.

## Team

| Member | Focus |
|--------|-------|
| Elie Bruno | Retrieval ablation: embeddings, chunking, reranking |
| Andrea Trugenberger | Citation verification: NLI pipeline, cross-LLM comparison |
| Faruk Zahiragic | Corrective RAG and threshold ablation |
| Yusif Askari | RAGAS and long-context baseline |

All four of us built the gold dataset.
