# CS-552 — Faithful RAG: Citation Accuracy & Retrieval Robustness in Domain-Specific Scientific Literature

**Course**: CS-552 Modern Natural Language Processing, Spring 2026, EPFL  
**Project type**: Open Project  
**Team**: 4 members  
**Base**: Forked from [cs433_ml_project_2](https://github.com/eliemada/cs433_ml_project_2) (Evidence-Based Policy Support Through RAG)

---

## Research Question

> When RAG systems generate answers with citations from scientific papers, do those citations actually support the claims? How do retrieval strategy, chunk granularity, and corrective mechanisms affect faithfulness?

## What We Inherited (from CS-433 ML Project)

A production-grade RAG pipeline for IP/innovation policy research:
- 999 academic papers parsed via Dolphin 1.5 VLM (PDF -> markdown)
- Hybrid chunking (coarse ~2000 chars + fine ~300 chars) with FAISS indexes
- Neural reranking (ZeroEntropy) + multi-LLM generation (OpenRouter)
- FastAPI backend + Next.js frontend with citations

See the original [README.md](README.md) for full architecture details.

## What We Added (CS-552 Contribution)

### 1. Gold Evaluation Dataset
- 50-100 expert-annotated Q&A pairs with ground-truth passages
- Location: `evaluation/gold_dataset/`

### 2. Retrieval Ablation Study
- Comparison across embedding models (text-embedding-3-small, BGE-M3, E5-large)
- Chunk size impact (300, 1000, 2000 chars)
- Reranker vs. no reranker
- Location: `evaluation/retrieval_eval/`

### 3. Faithfulness & Citation Verification
- NLI-based claim verification (DeBERTa-v3-large-mnli)
- Automated claim extraction + entailment scoring
- Cross-LLM faithfulness comparison
- Location: `evaluation/faithfulness/`

### 4. Corrective RAG (CRAG)
- Retrieval quality evaluator + query refinement loop
- Based on Yan et al., 2024
- Location: `evaluation/crag/`

### 5. End-to-End Evaluation
- RAGAS metrics (faithfulness, relevance, precision, recall)
- Long-context (128K) vs. chunked RAG comparison
- Cost/latency/accuracy tradeoff analysis
- Location: `evaluation/ragas_eval/`

## Repository Structure

```
cs552-faithful-rag/
├── packages/                     # [inherited] core RAG pipeline
│   ├── shared/rag_pipeline/      #   retrieval, chunking, embeddings
│   ├── api/                      #   FastAPI backend
│   └── worker/                   #   GPU PDF parser
├── frontend/                     # [inherited] Next.js UI
├── evaluation/                   # [NEW] CS-552 research contribution
│   ├── gold_dataset/             #   annotated Q&A pairs
│   ├── retrieval_eval/           #   retrieval metrics
│   ├── faithfulness/             #   citation verification
│   ├── crag/                     #   corrective RAG
│   ├── ragas_eval/               #   RAGAS + long-context comparison
│   └── long_context/             #   128K baseline
├── individual_notebooks/         # [NEW] one notebook per team member
│   ├── submit.sh                 #   RCP job launcher
│   └── *.ipynb                   #   individual contributions
├── reports/                      # [NEW] LaTeX reports
│   ├── proposal/                 #   due May 3
│   ├── literature_review/        #   due May 3
│   ├── progress_report/          #   due May 24
│   └── final_report/             #   due June 7
└── README.md                     # original project docs
```

## Deadlines

| Milestone | Due | Weight |
|-----------|-----|--------|
| Proposal + Lit Review | May 3, 2026 | 10% |
| Preliminary Results | May 24, 2026 | 10% |
| Final Submission | June 7, 2026 | 50% |

## Running Notebooks

```bash
cd individual_notebooks
./submit.sh  # launches RCP environment (details TBD)
```

## Team

| Member | Role |
|--------|------|
| TBD | Retrieval & chunking ablation |
| TBD | Faithfulness & citation verification |
| TBD | Corrective RAG (CRAG) |
| TBD | RAGAS evaluation & long-context comparison |
