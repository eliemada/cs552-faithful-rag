"""Retriever configurations for the M2 ablation.

Wraps :class:`rag_pipeline.rag.retriever.HybridRetriever` (FAISS + ZeroEntropy)
into four named configurations that vary along two axes already supported by
the existing infrastructure: chunk granularity (coarse vs. fine) and whether
the ZeroEntropy reranker is in the loop.

The remaining axes the proposal listed (alternative embedding models — BGE-M3,
E5-large, ColBERTv2) are deferred to M3: they each require building a new
46k-chunk FAISS index from scratch, which is hours of GPU + API time and not
on the M2 critical path.

The adapter normalises the ``HybridRetriever.search`` output to a plain list
of ``{chunk_id, paper_id, score, rank}`` dicts. That matches the contract of
``evaluate_retrieval.evaluate_retriever`` and avoids leaking the
``SearchResult`` dataclass into the metric layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rag_pipeline.rag.retriever import HybridRetriever

from evaluation.gold_dataset._validator import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_INDEXES_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "indexes"


@dataclass(frozen=True)
class RetrieverConfig:
    """One named configuration in the ablation."""

    name: str
    chunk_type: str  # "coarse" | "fine"
    use_reranker: bool

    def requires_zeroentropy(self) -> bool:
        return self.use_reranker


CONFIGS: Final[tuple[RetrieverConfig, ...]] = (
    RetrieverConfig(name="coarse_faiss", chunk_type="coarse", use_reranker=False),
    RetrieverConfig(name="coarse_rerank", chunk_type="coarse", use_reranker=True),
    RetrieverConfig(name="fine_faiss", chunk_type="fine", use_reranker=False),
    RetrieverConfig(name="fine_rerank", chunk_type="fine", use_reranker=True),
)

CONFIGS_BY_NAME: Final[dict[str, RetrieverConfig]] = {c.name: c for c in CONFIGS}


class RetrieverAdapter:
    """Thin wrapper exposing ``search(query, k) -> list[dict]``.

    Holds a single ``HybridRetriever`` and the config that determines whether
    the reranker is called. One adapter per (chunk_type, ±reranker) combination.
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


def load_adapter(
    config_name: str,
    *,
    indexes_dir: Path = DEFAULT_INDEXES_DIR,
    openai_api_key: str | None = None,
    zeroentropy_api_key: str | None = None,
) -> RetrieverAdapter:
    """Construct an adapter for the given config name.

    The OpenAI key is required (queries are embedded with
    ``text-embedding-3-small``). The ZeroEntropy key is required only if the
    config uses the reranker; configs without it work offline-after-embedding.

    Raises ``RuntimeError`` if a needed key is missing — this is meant to be
    called from a CLI, so a clear error beats a silent fallback.
    """
    config = CONFIGS_BY_NAME[config_name]
    openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to embed queries for FAISS retrieval. "
            "Set it in the environment or pass openai_api_key explicitly."
        )
    if config.requires_zeroentropy():
        zeroentropy_api_key = zeroentropy_api_key or os.environ.get("ZEROENTROPY_API_KEY")
        if not zeroentropy_api_key:
            raise RuntimeError(
                f"Config {config_name!r} uses the ZeroEntropy reranker but "
                "ZEROENTROPY_API_KEY is not set. Use a non-rerank config "
                "(coarse_faiss / fine_faiss) or provide the key."
            )

    hybrid = HybridRetriever.from_path(
        indexes_dir=indexes_dir,
        openai_api_key=openai_api_key,
        zeroentropy_api_key=zeroentropy_api_key if config.use_reranker else None,
        chunk_type=config.chunk_type,
    )
    logger.info("Loaded retriever config %s (chunk_type=%s, rerank=%s)",
                config.name, config.chunk_type, config.use_reranker)
    return RetrieverAdapter(hybrid, config)
