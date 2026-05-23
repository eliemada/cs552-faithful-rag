"""ColBERTv2 late-interaction retriever wrapping PyLate.

PyLate (lighton.ai) is the cleanest sentence-transformers-style wrapper
around the ColBERT family. We use it instead of RAGatouille because:

* PyLate's API is layered exactly like our existing
  ``SentenceTransformerEmbedder`` flow (a model, an index, a retriever),
  which keeps the abstraction parallel to FAISS-side code.
* PyLate exposes the model/index/retriever as separate objects, so the
  research code can inspect token-level encodings or swap PLAID for
  WARP / ScaNN without rewriting the wrapper.
* It is maintained by the team behind several recent ColBERT-style
  models (Reason-ModernColBERT, JaColBERT-X), so the wrapper tracks
  upstream improvements.

ColBERTv2 cannot reuse ``FAISSRetriever``: it emits one vector *per
token* rather than per chunk, and scores via MaxSim over a compressed
PLAID index. The class below implements :class:`BaseRetriever` directly,
plugging into ``HybridRetriever`` the same way ``FAISSRetriever`` does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, Optional, Union

from rag_pipeline.rag.retriever_base import BaseRetriever

if TYPE_CHECKING:
    from pylate import models as _pylate_models  # noqa: F401
    from rag_pipeline.rag.retriever import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID: Final[str] = "colbert-ir/colbertv2.0"
# 4-bit residual quantization matches the PLAID convention from the paper.
DEFAULT_NBITS: Final[int] = 4
# Top-k candidates pulled from the PLAID index before downstream rerank/cut.
DEFAULT_CANDIDATES: Final[int] = 75


class ColBERTRetriever(BaseRetriever):
    """Late-interaction retriever over a PyLate PLAID index.

    The constructor loads three things from disk:

    * the PyLate ``ColBERT`` model (downloaded from HF the first time)
    * the PLAID index folder produced by ``scripts.build_colbert_index``
    * the JSON metadata sidecar that maps each ``chunk_id`` back to its
      paper-level fields (``paper_id``, ``paper_title``, ``text``, ...)

    The metadata layout is identical to the one
    ``scripts.build_hf_index`` writes for the dense FAISS indices, so a
    single :class:`SearchResult` shape flows through the eval pipeline
    regardless of the underlying retriever.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        index_folder: Union[str, Path],
        index_name: str = "index",
        metadata_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        from pylate import indexes, models, retrieve  # heavy import, defer

        index_folder = Path(index_folder)
        metadata_path = Path(metadata_path)
        if not index_folder.is_dir():
            raise FileNotFoundError(f"ColBERT index folder missing: {index_folder}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"ColBERT metadata missing: {metadata_path}")

        logger.info("Loading ColBERT model %s on device=%s", model_id, device or "auto")
        self._model: "_pylate_models.ColBERT" = models.ColBERT(
            model_name_or_path=model_id,
            device=device,
        )

        logger.info("Opening PLAID index at %s/%s", index_folder, index_name)
        self._index = indexes.PLAID(
            index_folder=str(index_folder),
            index_name=index_name,
            override=False,
        )
        self._retriever = retrieve.ColBERT(index=self._index)
        self._metadata: dict[str, dict] = json.loads(metadata_path.read_text())
        logger.info("Loaded %d chunk-metadata entries", len(self._metadata))

    @classmethod
    def from_path(
        cls,
        indexes_dir: Union[str, Path],
        chunk_type: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
    ) -> "ColBERTRetriever":
        """Convenience constructor matching the dense-retriever conventions.

        Looks for ``indexes_dir / "colbert_<chunk_type>"`` (the PLAID index
        folder) and ``indexes_dir / "colbert_<chunk_type>_metadata.json"``.
        """
        indexes_dir = Path(indexes_dir)
        basename = f"colbert_{chunk_type}"
        return cls(
            model_id=model_id,
            index_folder=indexes_dir / basename,
            metadata_path=indexes_dir / f"{basename}_metadata.json",
            device=device,
        )

    def search(self, query: str, top_k: int = 50) -> "list[SearchResult]":
        # Local import keeps the module importable without dragging in the
        # dense FAISS stack and avoids a circular dependency.
        from rag_pipeline.rag.retriever import SearchResult

        query_embeddings = self._model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        # PyLate's retriever takes ``list[list | ndarray | Tensor]`` (a
        # batch); ``model.encode`` returns a single ndarray for a 1-item
        # batch, so wrap it once.
        scores = self._retriever.retrieve(
            queries_embeddings=[query_embeddings[0]],
            k=top_k,
        )
        if not scores:
            return []
        hits = scores[0]
        out: list[SearchResult] = []
        for rank, hit in enumerate(hits):
            # ``RerankResult`` is a ``TypedDict`` with ``id`` and ``score``.
            chunk_id = str(hit["id"])
            meta = self._metadata.get(chunk_id)
            if meta is None:
                # Indexed chunk not in metadata sidecar — should not happen
                # if the indexer ran cleanly, but skip rather than crash a
                # 37-query sweep.
                logger.warning("No metadata for chunk_id=%s, skipping", chunk_id)
                continue
            out.append(
                SearchResult(
                    chunk_id=chunk_id,
                    paper_id=meta["paper_id"],
                    paper_title=meta.get("paper_title", ""),
                    text=meta.get("text", ""),
                    section_hierarchy=meta.get("section_hierarchy", []),
                    score=float(hit["score"]),
                    rank=rank,
                )
            )
        return out
