"""Shared infrastructure for all evaluation contributions.

Modules
-------
paths
    Resolves where artifacts live (RCP scratch / local archive / HF download).
data_loader
    High-level loaders: chunks, FAISS, parsed markdown, OpenAlex metadata, gold Q&A.
models
    Dual-path generation dispatcher (local vLLM vs api LiteLLM/OpenRouter) and
    embedding / NLI factories.
"""

from evaluation.common.data_loader import (
    artifact_root,
    iter_all_chunks,
    list_paper_ids,
    load_chunk_metadata,
    load_faiss_index,
    load_gold_qa,
    load_paper_chunks,
    load_paper_markdown,
    load_paper_openalex,
)
from evaluation.common.models import (
    API_MODELS_DEFAULT,
    LOCAL_MODELS_DEFAULT,
    available_models,
    generate,
    load_embedder,
    load_nli_classifier,
)

__all__ = [
    "API_MODELS_DEFAULT",
    "LOCAL_MODELS_DEFAULT",
    "artifact_root",
    "available_models",
    "generate",
    "iter_all_chunks",
    "list_paper_ids",
    "load_chunk_metadata",
    "load_embedder",
    "load_faiss_index",
    "load_gold_qa",
    "load_nli_classifier",
    "load_paper_chunks",
    "load_paper_markdown",
    "load_paper_openalex",
]
