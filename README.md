# Evidence-Based Policy Support Through RAG

[![CI](https://github.com/eliemada/cs433_ml_project_2/actions/workflows/ci.yml/badge.svg)](https://github.com/eliemada/cs433_ml_project_2/actions/workflows/ci.yml)
[![Release](https://github.com/eliemada/cs433_ml_project_2/actions/workflows/release.yml/badge.svg)](https://github.com/eliemada/cs433_ml_project_2/actions/workflows/release.yml)
[![uv Workspace](https://camo.githubusercontent.com/bec95721f735c7ba951b338419fcb6a4810be2f3c0fc50b2e0b322f7dcd65bc9/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f61737472616c2d73682f75762f6d61696e2f6173736574732f62616467652f76302e6a736f6e)](https://github.com/astral-sh/uv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![ESLint](https://img.shields.io/badge/eslint-3A33D1?style=flat-square&logo=eslint&logoColor=white)](https://eslint.org/)
[![Ruff](https://img.shields.io/badge/ruff-enabled-informational?logo=ruff&style=flat-square)](https://docs.astral.sh/ruff/)

**Live demo:** https://ml-poc.eliebruno.com/

![Frontend walkthrough](docs/assets/rag-demo.gif)

A monorepo that powers an end-to-end Retrieval-Augmented Generation system for intellectual property and innovation policy research. The stack ingests OpenAlex metadata, parses PDFs with a multimodal VLM, builds hybrid coarse/fine chunk indexes, and serves bilingual answers with traceable citations through a FastAPI backend and a Next.js 16 frontend.

## Table of Contents

- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Repository Layout](#repository-layout)
- [Data & Processing Pipeline](#data--processing-pipeline)
- [Retrieval & LLM Serving](#retrieval--llm-serving)
- [Frontend Experience](#frontend-experience)
- [Running the Stack Locally](#running-the-stack-locally)
- [Environment Variables](#environment-variables)
- [Tooling & Automation](#tooling--automation)
- [Documentation, Notebooks & Reports](#documentation-notebooks--reports)
- [Testing & Quality](#testing--quality)
- [Current Status & Roadmap](#current-status--roadmap)
- [Contributors](#contributors)
- [License](#license)
- [Citation](#citation)

## Overview

- 4,920 OpenAlex works targeted (economics, innovation, IP). 999 PDFs have been parsed into rich markdown with layout-aware metadata and figures that live under `s3://cs433-rag-project2/processed/`.
- Distributed GPU workers run the Dolphin 1.5 model end-to-end (PDF → layout → markdown → figures) and upload failures/debug logs back to S3.
- Chunking uses a hybrid semantic strategy (≈50k coarse + 200k fine chunks today) with benchmarking reports justifying the configuration (`docs/markdown_chunking.md`, `reports/*.html`).
- Retrieval combines FAISS indexes (OpenAI `text-embedding-3-small`, 1,536-dim) with optional ZeroEntropy reranking before prompting OpenRouter-hosted models through LiteLLM.
- A bilingual (EN/FR) Next.js interface (`frontend/`) exposes structured answers with executive summaries, detailed analysis, policy implications, and inline citations.

## Architecture at a Glance

```
        ┌───────────────┐      ┌────────────────────┐      ┌────────────────────┐
        │ OpenAlex APIs │─────▶│ Metadata + PDF S3   │─────▶│ GPU PDF Workers    │
        │ (filters, MFA)│      │ (raw_pdfs/, raw_meta│      │ Dolphin 1.5 parsing│
        └───────────────┘      └────────┬────────────┘      └─────────┬──────────┘
                                         │                           │
                                         ▼                           ▼
                                 ┌──────────────┐            ┌────────────────┐
                                 │ Markdown +   │            │ Markdown Chunker│
                                 │ figures/JSON │            │ Hybrid coarse/fine
                                 └────────┬─────┘            │ (S3 chunks/)
                                          │                  └─────────┬───────┘
                                          ▼                            │
                                   ┌──────────────┐                ┌──────────────┐
                                   │ Embeddings   │──────────────▶│ FAISS indexes │
                                   │ OpenAI API   │ (text-embedding-3-small)     │
                                   └──────────────┘                └────────┬─────┘
                                                                            ▼
                                                                   ┌─────────────────┐
                                                                   │ FastAPI backend │
                                                                   │ Hybrid retrieval│
                                                                   │ + ZeroEntropy   │
                                                                   └────────┬────────┘
                                                                            ▼
                                                                   ┌─────────────────┐
                                                                   │ Next.js frontend│
                                                                   │ OpenRouter LLMs │
                                                                   └─────────────────┘
```

## Repository Layout

```
project-2-rag/
├── packages/
│   ├── shared/rag_pipeline/   # Shared library (ingestion, pdf parsing, chunking, retrieval, benchmarking)
│   ├── api/                   # FastAPI service (LiteLLM/OpenRouter chat + search API)
│   └── worker/                # Distributed GPU worker + embedding/indexing scripts
├── frontend/                  # Next.js 16 bilingual chat UI
├── scripts/                   # Ops utilities (S3 helpers, tests, benchmarking)
├── data/                      # Local OpenAlex snapshots / IP policy PDFs
├── docs/                      # Designs, setup guides, benchmarking docs
├── notebooks/                 # Individual team member notebooks
├── report/                    # LaTeX reports (proposal, lit review, progress, final)
├── docker-compose.yml         # API + worker runtime definitions
├── Makefile                   # Worker Docker build/push helpers
└── CI.md / RELEASING.md       # Automation and release checklists
```

### Monorepo packages

- `packages/shared/rag_pipeline/` – Core Python library used by both API and workers:
  - `openalex/` metadata/PDF downloader (filters, polite rate limiting, Sci-Hub fallback).
  - `pdf_parsing/` Dolphin pipeline (image extraction, layout parsing, element recognition, markdown/JSON output).
  - `rag/` chunking, embeddings, FAISS + ZeroEntropy retriever, CLI helpers.
  - `benchmarking/` metrics + Plotly report builder for chunking experiments.
- `packages/api/api/` – FastAPI app (`main.py`) with `/search`, `/chat`, `/models`, `/pdf/{paper_id}` plus policy-focused prompts and model catalog (`config.py`, `prompts.py`). Uses LiteLLM to talk to OpenRouter for GPT-5 / Claude / Gemini / DeepSeek models.
- `packages/worker/worker/` – GPU-friendly scripts (`distributed_worker.py`, `process_pdfs_batch.py`, `chunk_all_documents.py`, `embed_and_index.py`, `batch_chunk_markdown.py`, `check_worker_status.py`) that run on EC2 Spot or Vast.ai machines. Dockerfile bundles PyTorch, Transformers, OpenCV, and the Dolphin weights.

## Data & Processing Pipeline

### 1. OpenAlex harvesting (`rag_pipeline/openalex/`)

- `config.py` defines filters (default: `primary_topic.id=t10856`, `open_access.is_oa=true`), polite delays, and output paths.
- `fetcher.py` + `downloader.py` orchestrate cursor-based pagination, metadata parquet exports, and optional Sci-Hub fallback for missing PDFs.
- Outputs land in `data/openalex/` when run locally and `raw_pdfs/`, `raw_metadata/` once synced to S3.

### 2. GPU PDF parsing (`rag_pipeline/pdf_parsing/`, `packages/worker/worker/distributed_worker.py`)

- Dolphin 1.5 VLM runs entirely inside the worker Docker image (CUDA 11.8) and emits:
  - Markdown with preserved section hierarchy, tables, equations, citation markers.
  - Page-level JSON metadata and extracted figure PNGs.
- Workers shard PDFs deterministically via modulo partitioning, process 3 PDFs in parallel, upload successes to `processed/{paper_id}/` and failure logs to `failures/worker-*.json`.

### 3. Hybrid markdown chunking (`rag_pipeline/rag/markdown_chunker.py`, `docs/markdown_chunking.md`)

- Produces both coarse chunks (~2k chars, 10% overlap) and fine chunks (~300 chars, 20% overlap).
- Section-aware splitting keeps semantic boundaries and preserves heading hierarchies for later UI rendering.
- `scripts/dev/test_markdown_chunking.py` validates S3 loading (`scripts/utils/markdown_s3_loader.py`) and chunk quality.

### 4. Embedding + indexing (`packages/worker/worker/embed_and_index.py`)

- Uses OpenAI `text-embedding-3-small` (1,536 dim) with batching and token-cost estimation utilities.
- Normalizes vectors for cosine similarity and builds FAISS `IndexFlatIP`.
- Uploads indexes + metadata maps to `s3://cs433-rag-project2/indexes/{coarse,fine}*.faiss`.

### 5. Storage layout

```
cs433-rag-project2/
├── raw_pdfs/                  # Input PDFs from OpenAlex
├── processed/<paper_id>/      # Markdown, metadata.json, figures/
├── chunks/                    # Chunk JSON dumps (coarse & fine)
├── indexes/                   # FAISS indexes + metadata
└── failures/worker-*.json     # Worker crash/trace data
```

## Retrieval & LLM Serving

- `rag_pipeline/rag/retriever.py` loads FAISS indexes from S3, embeds queries via OpenAI, and optionally reranks with ZeroEntropy (`zerank-1`).
- FastAPI service (`packages/api/api/main.py`) exposes:
  - `/health`, `/` – readiness and metadata.
  - `/search` – vector search, returning chunk text + section hierarchy with latency metrics.
  - `/chat` – retrieves top chunks then formats a policy prompt (`api/prompts.py`) before calling LiteLLM → OpenRouter. Supports 9 models across fast/balanced/premium tiers; fallback logic retries with the default `openrouter/openai/gpt-5-mini`.
  - `/models` – surfaces runtime-available LLMs to the frontend.
  - `/pdf/{paper_id}` – S3 presigned URL for the underlying PDF.
- Responses are normalized into `SearchResponse` and `ChatResponse` dataclasses to keep the frontend contract stable.

## Frontend Experience

- Located in `frontend/` (Next.js 16 + React 19 + Tailwind 4 beta). Key pieces:
  - `src/app/[lang]/page.tsx` – language-aware entrypoint using `getDictionary`.
  - `src/components/ChatInterface.tsx` – orchestrates sidebar, model selector, streaming states, and structured results parsing (Executive Summary, Detailed Analysis, Key References).
  - `src/lib/api.ts` – fetch helpers for `/health`, `/search`, `/chat`, `/models`, `/pdf`.
  - `src/dictionaries/` + `src/i18n-config.ts` – English/French translations for UI chrome.
- Features:
  - Model switcher with tiers, persisted default from `/models`.
  - Structured response rendering (bullets, collapsible sections, citations list).
  - Presigned PDF download buttons (coming from `/pdf`).
  - Responsive sidebar for saved chats (placeholder) and onboarding text.

## Running the Stack Locally

### Requirements

- Python 3.11+ and [uv](https://github.com/astral-sh/uv) for workspace management.
- Node.js 20+ (or latest LTS) + npm for the frontend.
- Docker + NVIDIA Container Toolkit for GPU worker images (optional but required for Dolphin).
- AWS credentials with access to `cs433-rag-project2` (or your own bucket).
- API keys: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, optional `ZEROENTROPY_API_KEY`.

### 1. Bootstrap the Python workspace

```bash
# From repo root
cp .env.example .env        # fill in API keys + S3 settings
uv sync                     # installs shared/api/worker environments
```

### 2. Running the API Framework

The FastAPI backend serves the retrieval and chat endpoints. To run it:

```bash
# Set required environment variables
export OPENAI_API_KEY=your_openai_key
export OPENROUTER_API_KEY=your_openrouter_key
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
export AWS_DEFAULT_REGION=us-east-1  # or your preferred region
export S3_BUCKET=cs433-rag-project2  # or your bucket name

# Launch the API server
uv run --package api uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will:
- Load FAISS indexes from S3 at startup
- Serve endpoints at `http://localhost:8000`
- Provide API documentation at `http://localhost:8000/docs`

Check health status: `curl http://localhost:8000/health`

### 3. Running the Frontend

The Next.js 16 application provides a bilingual (EN/FR) chat interface:

```bash
# Navigate to frontend directory
cd frontend

# Copy and configure environment variables
cp .env.local.example .env.local

# Edit .env.local and set:
# NEXT_PUBLIC_API_URL=http://localhost:8000

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at:
- English: `http://localhost:3000/en`
- French: `http://localhost:3000/fr`

For production builds:
```bash
npm run build
npm start
```

### 4. PDF Processing (GPU Required)

**⚠️ Important:** PDF processing with the Dolphin 1.5 model requires machines with NVIDIA GPUs. This is more complex than running the API or frontend.

#### Requirements:
- NVIDIA GPU with CUDA support (tested with CUDA 11.8)
- Docker with NVIDIA Container Toolkit
- S3 or S3-compatible storage (e.g., AWS S3, Infomaniak kDrive, MinIO)

#### Storage Configuration:

The system supports any S3-compatible API. Configure via environment variables:

```bash
# For AWS S3:
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
export S3_INPUT_BUCKET=your-input-bucket
export S3_OUTPUT_BUCKET=your-output-bucket

# For S3-compatible services (e.g., Infomaniak):
export AWS_ACCESS_KEY_ID=your_infomaniak_key
export AWS_SECRET_ACCESS_KEY=your_infomaniak_secret
export AWS_ENDPOINT_URL=https://s3.infomaniak.com  # or your S3-compatible endpoint
export S3_INPUT_BUCKET=your-bucket
export S3_OUTPUT_BUCKET=your-bucket
```

**Note:** When using S3-compatible services like Infomaniak, ensure your client library supports custom endpoints. The `boto3` library used in this project respects the `AWS_ENDPOINT_URL` environment variable.

#### Running the PDF Worker:

```bash
# Set worker configuration
export WORKER_ID=0
export TOTAL_WORKERS=1
export CONCURRENT_PDFS=3  # Adjust based on GPU memory

# Run worker locally (requires GPU)
uv run --package worker python packages/worker/worker/distributed_worker.py

# Or using Docker (recommended for production)
docker compose run --rm --gpus all pdf-worker
```

Workers will:
- Download PDFs from `S3_INPUT_BUCKET/raw_pdfs/`
- Process them using Dolphin 1.5 VLM
- Upload results to `S3_OUTPUT_BUCKET/processed/`
- Upload failure logs to `S3_OUTPUT_BUCKET/failures/`

For distributed processing across multiple GPU machines:
```bash
# On machine 1:
export WORKER_ID=0 TOTAL_WORKERS=3
docker compose run --rm --gpus all pdf-worker

# On machine 2:
export WORKER_ID=1 TOTAL_WORKERS=3
docker compose run --rm --gpus all pdf-worker

# On machine 3:
export WORKER_ID=2 TOTAL_WORKERS=3
docker compose run --rm --gpus all pdf-worker
```

### 5. Embed chunks / rebuild FAISS

After processing PDFs, generate embeddings and build FAISS indexes:

```bash
export OPENAI_API_KEY=your_key

# Generate embeddings and create indexes
uv run --package worker python packages/worker/worker/embed_and_index.py \
  --chunk-type both          # coarse, fine, or both

# Dry run to estimate cost first
uv run --package worker python packages/worker/worker/embed_and_index.py \
  --chunk-type both --dry-run
```

This will create FAISS indexes in `S3_BUCKET/indexes/`.

### 6. Docker Compose (API + GPU worker)

For containerized deployment:

```bash
# Run API only (no GPU required)
docker compose up rag-api

# Run PDF worker (requires NVIDIA runtime)
docker compose run --rm --gpus all pdf-worker
```

The worker image includes Dolphin weights and all dependencies. See `Makefile` for custom image builds.

## Environment Variables

| Variable | Used by | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | API, worker, embedding script | Required for embeddings + initial FAISS query encoding. |
| `OPENROUTER_API_KEY` | API | Required for LLM completions through LiteLLM. |
| `ZEROENTROPY_API_KEY` | API (optional) | Enables reranking via ZeroEntropy (if unset, retrieval falls back to FAISS-only). |
| `S3_BUCKET` / `S3_INPUT_BUCKET` / `S3_OUTPUT_BUCKET` | API & worker | Buckets holding raw PDFs, processed markdown, chunk JSON, indexes. Defaults to `cs433-rag-project2`. |
| `CHUNK_TYPE` | API | Chooses which FAISS index (`coarse` or `fine`) to load. |
| `FAISS_CANDIDATES` | API | Number of vectors pulled from FAISS before reranking (default 75). |
| `WORKER_ID` / `TOTAL_WORKERS` | worker | Deterministic sharding for distributed PDF processing. |
| `CONCURRENT_PDFS` | worker | Thread pool size per worker (default 3). |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | API & worker | Required to download/upload data from S3. |
| `NEXT_PUBLIC_API_URL` | frontend | Points the UI to the FastAPI base URL. |

`.env.example`, `scripts/.env.example`, and `frontend/.env.local.example` ship with sane defaults for each layer.

## Tooling & Automation

- **uv workspace** (`pyproject.toml`, `uv.lock`) keeps shared dependencies pinned and ensures API/worker install only what they need.
- **Dockerfiles**:
  - `packages/api/Dockerfile` – slim Python 3.11 image (≈800 MB) without GPU deps, syncs only the API package with rag_pipeline `[vector,cloud]` extra.
  - `packages/worker/Dockerfile` – CUDA 11.8 runtime with PyTorch, Transformers, OpenCV, Dolphin weights preloaded (~4.5 GB).
- **docker-compose.yml** wires `rag-api` and `pdf-worker` together, including health checks and GPU reservations.
- **Makefile** automates worker image build/push/test loops.
- **CI.md** documents how the UV + PEX based CI/CD pipeline builds deterministic artifacts; `scripts/ci/build_pex.sh` shows how to produce single-file deployables.

## Documentation, Notebooks & Reports

- `docs/plans/*.md` capture design decisions (distributed PDF parsing, chunking & embedding pipeline, LiteLLM/OpenRouter setup).
- `docs/markdown_chunking.md` describes the chunking system, evaluation metrics, and S3 layout in detail.
- `docs/benchmarking_system.md` covers the professional-grade chunking benchmarking framework and how to generate Plotly reports via `scripts/benchmarking/generate_chunking_report.py`.
- `docs/LITELLM_SETUP.md` and `docs/checkpoint_2025-11-10.md` summarize infra bring-up milestones.
- Chunking benchmark results are stored in `evaluation/chunking_benchmark_detailed.json`.
- `scripts/dev/test_retriever.py`, `scripts/dev/test_local_processing.py`, and `scripts/dev/test_markdown_chunking.py` provide lightweight smoke tests for each pipeline stage.

## Testing & Quality

```bash
# Run Python unit tests (packages/*/tests + repo-level tests/)
uv run pytest

# Lint & type-check shared code
uv run ruff check
uv run mypy packages/shared/rag_pipeline

# Frontend lint
cd frontend
npm run lint
```

Benchmarking and evaluation scripts live under `scripts/benchmarking/` and produce HTML dashboards for regression tracking.

## Current Status & Roadmap

**What exists today**
- ✅ 999/4,920 papers parsed into markdown + figures, synced to S3.
- ✅ Hybrid coarse/fine chunking strategy benchmarked with professional tooling; FAISS indexes for `coarse` and `fine` chunks stored in `indexes/`.
- ✅ FastAPI backend serving `/search`, `/chat`, `/models`, `/pdf` with OpenRouter + LiteLLM and optional ZeroEntropy reranking.
- ✅ Next.js bilingual UI with structured response rendering, model selection, and presigned PDF links.
- ✅ Dockerized GPU workers with deterministic work slicing and built-in failure reporting.

**In progress / next**
1. Finish embedding/indexing the remaining corpus (4,920 target) and optionally migrate to a managed vector DB (Pinecone or Weaviate) once scale requires.
2. Expand evaluation to ≥100 policy queries and integrate hallucination tracking in CI (see `docs/benchmarking_system.md` for current KPIs).
3. Add real-time PDF upload + ad-hoc indexing for user-provided documents.
4. Tighten metadata enrichment (author/year/citation formatting) and surface it in the frontend citations list.
5. Deploy monitoring on inference cost + latency and add caching for repeated queries.

## Contributors

- **Elie Bruno** – École Polytechnique Fédérale de Lausanne (EPFL)
- **Andrea Trugenberger** – EPFL
- **Youssef Chelaifa** – EPFL

## License

MIT-style placeholder. Add or update a `LICENSE` file if a different license is required for distribution.

## Citation

If you use this system in your research, please cite:

```bibtex
@article{bruno2025rag,
  title={Evidence-Based Policy Support Through RAG: A Critical Analysis of Bias and Reliability in AI-Assisted Patent Research},
  author={Bruno, Elie and Trugenberger, Andrea and Chelaifa, Youssef},
  year={2025},
  institution={École Polytechnique Fédérale de Lausanne (EPFL)}
}
```
