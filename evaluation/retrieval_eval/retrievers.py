"""Retriever configurations for the M2 / M3 ablation.

Four embedder families crossed with two chunk granularities and ±ZeroEntropy
reranking, giving 16 named configs:

* ``openai`` — ``text-embedding-3-small``, the original 1536-dim FAISS index.
* ``bge_m3`` — ``BAAI/bge-m3`` dense single-vector (1024-dim).
* ``e5_large`` — ``intfloat/e5-large-v2`` dense single-vector (1024-dim).
* ``colbert`` — ``colbert-ir/colbertv2.0`` late-interaction multi-vector
  via PyLate + PLAID (residual quantization, nbits=4 by default).

The first three share the same ``HybridRetriever`` + ``FAISSRetriever`` code
path. ColBERTv2 is a different paradigm: the indexer (``scripts.build_colbert_index``)
emits a PyLate PLAID folder rather than a flat FAISS file, and the retriever
(``ColBERTRetriever``) scores via MaxSim instead of L2 / inner product.
``HybridRetriever`` composes any object satisfying :class:`BaseRetriever`,
so a single ablation pipeline drives both.

``RetrieverAdapter`` normalises ``HybridRetriever.search`` output to a plain
list of ``{chunk_id, paper_id, score, rank}`` dicts. That matches the
contract of ``evaluate_retrieval.evaluate_retriever`` and keeps the
``SearchResult`` dataclass out of the metric layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from rag_pipeline.rag.colbert_retriever import (
    DEFAULT_MODEL_ID as DEFAULT_COLBERT_MODEL_ID,
)
from rag_pipeline.rag.colbert_retriever import ColBERTRetriever
from rag_pipeline.rag.embedder import Embedder, SentenceTransformerEmbedder
from rag_pipeline.rag.retriever import HybridRetriever, ZeroEntropyReranker
from rag_pipeline.rag.retriever_base import BaseRetriever

from evaluation.gold_dataset._validator import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"

# HF ids for dense single-vector encoders. ``openai`` is server-side and
# ``colbert`` uses its own backend, so they're not in this table.
_HF_MODEL_IDS: Final[dict[str, str]] = {
    "bge_m3": "BAAI/bge-m3",
    "e5_large": "intfloat/e5-large-v2",
}

# Embedder families that use the dense FAISS path. ``colbert`` is excluded;
# it has its own ``ColBERTRetriever`` and PLAID-formatted index.
_DENSE_FAMILIES: Final[frozenset[str]] = frozenset({"openai", "bge_m3", "e5_large"})


@dataclass(frozen=True)
class RetrieverConfig:
    """One named configuration in the ablation.

    ``embedder`` is the family label that also forms the index filename
    prefix (e.g. ``"bge_m3"`` → ``bge_m3_coarse.faiss``,
    ``"colbert"`` → ``colbert_coarse/`` PLAID folder).  The legacy
    OpenAI configs use ``"openai"`` for the label but no prefix on disk,
    matching the original index filenames.
    """

    name: str
    chunk_type: str  # "coarse" | "fine"
    use_reranker: bool
    embedder: str = "openai"  # "openai" | "bge_m3" | "e5_large" | "colbert"

    def requires_zeroentropy(self) -> bool:
        return self.use_reranker

    def requires_openai(self) -> bool:
        return self.embedder == "openai"

    def is_colbert(self) -> bool:
        return self.embedder == "colbert"

    def index_basename(self) -> str:
        if self.embedder == "openai":
            return self.chunk_type
        return f"{self.embedder}_{self.chunk_type}"


def _make_configs() -> tuple[RetrieverConfig, ...]:
    """Cross every embedder family with both granularities and ±reranker."""
    embedder_order: tuple[str, ...] = ("openai", "bge_m3", "e5_large", "colbert")
    out: list[RetrieverConfig] = []
    for embedder in embedder_order:
        for chunk in ("coarse", "fine"):
            for rerank in (False, True):
                suffix = "rerank" if rerank else "faiss"
                if embedder == "openai":
                    name = f"{chunk}_{suffix}"
                else:
                    name = f"{embedder}_{chunk}_{suffix}"
                out.append(
                    RetrieverConfig(
                        name=name,
                        chunk_type=chunk,
                        use_reranker=rerank,
                        embedder=embedder,
                    )
                )
    return tuple(out)


CONFIGS: Final[tuple[RetrieverConfig, ...]] = _make_configs()
CONFIGS_BY_NAME: Final[dict[str, RetrieverConfig]] = {c.name: c for c in CONFIGS}


class RetrieverAdapter:
    """Thin wrapper exposing ``search(query, k) -> list[dict]``.

    Holds a single ``HybridRetriever`` and the config that determines whether
    the reranker is called. One adapter per (embedder, chunk_type, ±reranker).
    """

    def __init__(self, hybrid: HybridRetriever, config: RetrieverConfig):
        self._hybrid = hybrid
        self.config = config

    def search(self, query: str, k: int) -> list[dict]:
        results = self._hybrid.search(query, top_k=k, use_reranker=self.config.use_reranker)
        return [
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "score": float(r.score),
                "rank": r.rank,
            }
            for r in results
        ]


def _build_embedder(config: RetrieverConfig) -> Optional[Embedder]:
    """Materialise the dense encoder for ``config``. ``None`` for OpenAI default."""
    if config.embedder == "openai":
        return None
    hf_id = _HF_MODEL_IDS[config.embedder]
    return SentenceTransformerEmbedder(hf_id, short_name=config.embedder)


def _build_base_retriever(
    config: RetrieverConfig,
    *,
    indexes_dir: Path,
    openai_api_key: Optional[str],
    colbert_model_id: str,
    colbert_device: Optional[str],
) -> BaseRetriever:
    """Dispatch on backend family to construct the base retriever."""
    if config.is_colbert():
        return ColBERTRetriever.from_path(
            indexes_dir=indexes_dir,
            chunk_type=config.chunk_type,
            model_id=colbert_model_id,
            device=colbert_device,
        )

    # Dense single-vector families share the FAISS backend via HybridRetriever.
    # We unpack the inner FAISSRetriever so HybridRetriever stays the
    # reranker-composition layer (not nested).
    embedder = _build_embedder(config)
    from rag_pipeline.rag.retriever import FAISSRetriever  # local import keeps modules tight

    return FAISSRetriever.from_path(
        indexes_dir=indexes_dir,
        chunk_type=config.chunk_type,
        openai_api_key=openai_api_key,
        embedder=embedder,
        index_basename=config.index_basename(),
    )


def load_adapter(
    config_name: str,
    *,
    indexes_dir: Path = DEFAULT_INDEXES_DIR,
    openai_api_key: str | None = None,
    zeroentropy_api_key: str | None = None,
    colbert_model_id: str = DEFAULT_COLBERT_MODEL_ID,
    colbert_device: Optional[str] = None,
) -> RetrieverAdapter:
    """Construct an adapter for the given config name.

    Per-family preconditions:

    * ``openai``: needs ``OPENAI_API_KEY`` (encodes queries server-side).
    * ``bge_m3``, ``e5_large``: run entirely locally after the FAISS index
      is built.
    * ``colbert``: runs entirely locally; the PLAID index folder must
      already exist (build it with ``scripts.build_colbert_index``).

    All ``*_rerank`` configs additionally need ``ZEROENTROPY_API_KEY``.

    Raises ``RuntimeError`` when a required key or artefact is missing.
    """
    config = CONFIGS_BY_NAME[config_name]

    if config.requires_openai():
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError(
                f"Config {config_name!r} uses OpenAI query embeddings but "
                "OPENAI_API_KEY is not set."
            )

    if config.requires_zeroentropy():
        zeroentropy_api_key = zeroentropy_api_key or os.environ.get("ZEROENTROPY_API_KEY")
        if not zeroentropy_api_key:
            raise RuntimeError(
                f"Config {config_name!r} uses the ZeroEntropy reranker but "
                "ZEROENTROPY_API_KEY is not set. Use a *_faiss variant or "
                "provide the key."
            )

    base = _build_base_retriever(
        config,
        indexes_dir=indexes_dir,
        openai_api_key=openai_api_key,
        colbert_model_id=colbert_model_id,
        colbert_device=colbert_device,
    )
    reranker = (
        ZeroEntropyReranker(api_key=zeroentropy_api_key)
        if config.use_reranker and zeroentropy_api_key
        else None
    )
    hybrid = HybridRetriever(base_retriever=base, reranker=reranker)
    logger.info(
        "Loaded retriever config %s (embedder=%s, chunk_type=%s, rerank=%s)",
        config.name,
        config.embedder,
        config.chunk_type,
        config.use_reranker,
    )
    return RetrieverAdapter(hybrid, config)
