[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QDjEejvC)

# CS-552 — Faithful RAG: Citation Accuracy & Retrieval Robustness in Domain-Specific Scientific Literature

**Course**: CS-552 Modern Natural Language Processing, Spring 2026, EPFL  
**Project type**: Open Project  
**Team**: CiteRight (4 members)  
**Base**: Forked from [cs433_ml_project_2](https://github.com/eliemada/cs433_ml_project_2) (Evidence-Based Policy Support Through RAG)

> **Assignment info**: See the [open project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.8ww6fxjeedkw) for deliverables, [rubric](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit#heading=h.psuj40jgnavg), and [RCP setup guide](rcp_support/README.md).

---

## Research Question

> When RAG systems generate answers with citations from scientific papers, do those citations actually support the claims? How do retrieval strategy, chunk granularity, and corrective mechanisms affect faithfulness?

## What We Inherited (from CS-433 ML Project)

A production-grade RAG pipeline for IP/innovation policy research:
- 999 academic papers parsed via Dolphin 1.5 VLM (PDF -> markdown)
- Hybrid chunking (coarse ~2000 chars + fine ~300 chars) with FAISS indexes
- Neural reranking (ZeroEntropy) + multi-LLM generation (OpenRouter)
- FastAPI backend + Next.js frontend with citations

See the original [README.md](code/README.md) for full architecture details.

## What We Added (CS-552 Contribution)

### 1. Gold Evaluation Dataset
- 50-100 expert-annotated Q&A pairs with ground-truth passages
- Location: `code/evaluation/gold_dataset/`

### 2. Retrieval Ablation Study
- Comparison across embedding models (text-embedding-3-small, BGE-M3, E5-large)
- Chunk size impact (300, 1000, 2000 chars)
- Reranker vs. no reranker
- Location: `code/evaluation/retrieval_eval/`

### 3. Faithfulness & Citation Verification
- NLI-based claim verification (DeBERTa-v3-large-mnli)
- Automated claim extraction + entailment scoring
- Cross-LLM faithfulness comparison
- Location: `code/evaluation/faithfulness/`

### 4. Corrective RAG (CRAG)
- Retrieval quality evaluator + query refinement loop
- Based on Yan et al., 2024
- Location: `code/evaluation/crag/`

### 5. End-to-End Evaluation
- RAGAS metrics (faithfulness, relevance, precision, recall)
- Long-context (128K) vs. chunked RAG comparison
- Cost/latency/accuracy tradeoff analysis
- Location: `code/evaluation/ragas_eval/`

## Repository Structure

```
open-project-m2-citeright/
├── code/                         # All source code
│   ├── packages/                 #   core RAG pipeline (shared, api, worker)
│   ├── evaluation/               #   CS-552 research contribution
│   │   ├── gold_dataset/         #     annotated Q&A pairs
│   │   ├── retrieval_eval/       #     retrieval metrics
│   │   ├── faithfulness/         #     citation verification
│   │   ├── crag/                 #     corrective RAG
│   │   ├── ragas_eval/           #     RAGAS + long-context comparison
│   │   └── long_context/         #     128K baseline
│   ├── frontend/                 #   Next.js UI
│   ├── scripts/                  #   utilities & benchmarking
│   ├── docs/                     #   design docs & guides
│   ├── pyproject.toml            #   Python project config
│   └── docker-compose.yml        #   service definitions
├── notebooks/                    # One notebook per team member
│   ├── submit.sh                 #   RCP job launcher
│   └── *.ipynb                   #   individual contributions
├── report/                       # LaTeX reports
│   ├── proposal/                 #   due May 3
│   ├── literature_review/        #   due May 3
│   ├── progress_report/          #   due May 24
│   └── final_report/             #   due June 7
└── README.md
```

## Deadlines

| Milestone | Due | Weight |
|-----------|-----|--------|
| Proposal + Lit Review | May 3, 2026 | 10% |
| Preliminary Results | May 24, 2026 | 10% |
| Final Submission | June 7, 2026 | 50% |

## Running Notebooks

```bash
cd notebooks
# Set GASPAR to your EPFL username, GROUP to your team (e.g. g07)
./submit.sh          # launches 1-GPU interactive Jupyter on RCP
# Then: runai port-forward <job-name> --port 8888:8888
# Open http://localhost:8888 (token: cs552)
```

See [rcp_support/README.md](rcp_support/README.md) for full setup instructions.

## Team

| Member | SCIPER | Role |
|--------|--------|------|
| Elie Bruno | 355932 | Retrieval ablation (embeddings, chunking, reranking) |
| Andrea Trugenberger | 357615 | Citation verification (NLI pipeline, cross-LLM comparison) |
| Faruk Zahiragic | 415360 | Corrective RAG and threshold ablation |
| Yusif Askari | 413862 | RAGAS and long-context baseline |

All members contribute to the gold evaluation dataset.
