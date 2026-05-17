"""Retriever configurations for the M2 ablation.

Three embedder families crossed with two chunk granularities and ±ZeroEntropy
reranking. The OpenAI ``text-embedding-3-small`` row was indexed end-to-end
first; the BGE-M3 and E5-large rows ride on indices built by
``scripts.build_hf_index``.

ColBERTv2 stays out: it is a late-interaction multi-vector retriever, not a
drop-in dense-vector swap, and would need a separate index format (PLAID).

The adapter normalises ``HybridRetriever.search`` output to a plain list of
``{chunk_id, paper_id, score, rank}`` dicts. That matches the contract of
``evaluate_retrieval.evaluate_retriever`` and keeps the ``SearchResult``
dataclass out of the metric layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from rag_pipeline.rag.embedder import Embedder, SentenceTransformerEmbedder
from rag_pipeline.rag.retriever import HybridRetriever

from evaluation.gold_dataset._validator import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"

# Map an embedder family to its Hugging Face id. ``None`` means OpenAI (server-side).
_HF_MODEL_IDS: Final[dict[str, str]] = {
    "bge_m3": "BAAI/bge-m3",
    "e5_large": "intfloat/e5-large-v2",
}


@dataclass(frozen=True)
class RetrieverConfig:
    """One named configuration in the ablation.

    ``embedder`` is the family label that also forms the index filename
    prefix (e.g. ``"bge_m3"`` → ``bge_m3_coarse.faiss``). The legacy
    OpenAI configs use ``"openai"`` for the label but no prefix on disk,
    matching the original index filenames.
    """

    name: str
    chunk_type: str  # "coarse" | "fine"
    use_reranker: bool
    embedder: str = "openai"  # "openai" | "bge_m3" | "e5_large"

    def requires_zeroentropy(self) -> bool:
        return self.use_reranker

    def requires_openai(self) -> bool:
        return self.embedder == "openai"

    def index_basename(self) -> str:
        if self.embedder == "openai":
            return self.chunk_type
        return f"{self.embedder}_{self.chunk_type}"


def _make_configs() -> tuple[RetrieverConfig, ...]:
    """Cross every embedder family with both granularities and ±reranker."""
    embedder_order: tuple[str, ...] = ("openai", "bge_m3", "e5_large")
    out: list[RetrieverConfig] = []
    for embedder in embedder_order:
        for chunk in ("coarse", "fine"):
            for rerank in (False, True):
                if embedder == "openai":
                    name = f"{chunk}_{'rerank' if rerank else 'faiss'}"
                else:
                    name = f"{embedder}_{chunk}_{'rerank' if rerank else 'faiss'}"
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
    """Materialise the encoder for ``config``. ``None`` means OpenAI default."""
    if config.embedder == "openai":
        return None
    hf_id = _HF_MODEL_IDS[config.embedder]
    return SentenceTransformerEmbedder(hf_id, short_name=config.embedder)


def load_adapter(
    config_name: str,
    *,
    indexes_dir: Path = DEFAULT_INDEXES_DIR,
    openai_api_key: str | None = None,
    zeroentropy_api_key: str | None = None,
) -> RetrieverAdapter:
    """Construct an adapter for the given config name.

    Configs whose embedder is OpenAI need ``OPENAI_API_KEY`` to encode queries.
    Configs with a Hugging Face embedder run entirely locally (after the FAISS
    index has been built). The ZeroEntropy key is needed only for ±rerank.

    Raises ``RuntimeError`` when a required key is missing.
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

    embedder = _build_embedder(config)
    hybrid = HybridRetriever.from_path(
        indexes_dir=indexes_dir,
        openai_api_key=openai_api_key,
        zeroentropy_api_key=zeroentropy_api_key if config.use_reranker else None,
        chunk_type=config.chunk_type,
        embedder=embedder,
        index_basename=config.index_basename(),
    )
    logger.info(
        "Loaded retriever config %s (embedder=%s, chunk_type=%s, rerank=%s)",
        config.name,
        config.embedder,
        config.chunk_type,
        config.use_reranker,
    )
    return RetrieverAdapter(hybrid, config)
