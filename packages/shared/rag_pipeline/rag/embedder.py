"""Embedder abstraction for the retrieval ablation.

A small protocol that lets ``FAISSRetriever`` work with any encoder, not
just OpenAI. The two methods deliberately separate *query* and *passage*
encoding because some sentence-transformers families (E5, GTR) require
different prefixes for each role.

Implementations:

* :class:`OpenAIEmbedderAdapter` wraps the existing ``OpenAIEmbedder``
  and exposes the protocol-shaped API.
* :class:`SentenceTransformerEmbedder` runs any sentence-transformers
  checkpoint (BGE-M3, E5-large, ...) locally with optional MPS / CUDA
  acceleration and applies model-specific prefixes.

Both implementations L2-normalise their outputs so the result drops
straight into a FAISS ``IndexFlatL2`` over unit vectors, which is the
convention the OpenAI indices already use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from rag_pipeline.rag.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface required by ``FAISSRetriever`` and the indexer."""

    @property
    def name(self) -> str:
        """Short, file-safe identifier (e.g. ``"openai-3-small"``, ``"bge-m3"``)."""
        ...

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        ...

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Return an ``(N, dim)`` ``float32`` array, L2-normalised."""
        ...

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        """Return an ``(N, dim)`` ``float32`` array, L2-normalised."""
        ...


# ---------- OpenAI adapter -------------------------------------------------


class OpenAIEmbedderAdapter:
    """Adapter that exposes :class:`OpenAIEmbedder` through :class:`Embedder`."""

    def __init__(self, openai_embedder: "OpenAIEmbedder", name: str = "openai-3-small"):
        self._inner = openai_embedder
        self._name = name
        self._dim = int(self._inner.get_embedding_dimension())

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._inner.generate_embeddings_batch(texts)
        arr = np.asarray(embeddings, dtype=np.float32)
        # OpenAI returns unit vectors, but defensively normalise so the FAISS
        # metric assumption holds even if a future model breaks that.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(queries)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode(passages)


# ---------- sentence-transformers adapter ---------------------------------


# Model-specific prefix conventions. The E5 family explicitly trains with
# these prefixes; BGE-M3 expects raw text on both sides.
_QUERY_PREFIX: dict[str, str] = {
    "intfloat/e5-large-v2": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-small-v2": "query: ",
}
_PASSAGE_PREFIX: dict[str, str] = {
    "intfloat/e5-large-v2": "passage: ",
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
    "intfloat/e5-small-v2": "passage: ",
}


def _pick_device(prefer: str | None = None) -> str:
    """Return ``"mps"`` on Apple Silicon, ``"cuda"`` on NVIDIA, ``"cpu"`` else."""
    if prefer:
        return prefer
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class SentenceTransformerEmbedder:
    """A sentence-transformers checkpoint behind the :class:`Embedder` API.

    Parameters
    ----------
    model_id:
        Hugging Face identifier (e.g. ``"BAAI/bge-m3"``, ``"intfloat/e5-large-v2"``).
    short_name:
        File-safe label written to index filenames and JSON metadata. Falls
        back to a sanitised ``model_id`` when omitted.
    device:
        Override device. Default is ``"mps"`` on Apple Silicon, ``"cuda"`` on
        NVIDIA, ``"cpu"`` otherwise.
    batch_size:
        Forwarded to ``SentenceTransformer.encode``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        short_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int | None = 512,
    ):
        from sentence_transformers import SentenceTransformer  # heavy import, defer

        self._model_id = model_id
        self._name = short_name or model_id.replace("/", "_").replace("-", "_").lower()
        self._device = _pick_device(device)
        self._batch_size = batch_size
        self._query_prefix = _QUERY_PREFIX.get(model_id, "")
        self._passage_prefix = _PASSAGE_PREFIX.get(model_id, "")

        logger.info("Loading %s on device=%s", model_id, self._device)
        self._model = SentenceTransformer(model_id, device=self._device, trust_remote_code=True)
        # Cap context length: BGE-M3 defaults to 8192 which makes attention O(n^2)
        # blow up on long chunks. 512 tokens is the SBERT-family standard and the
        # chunks themselves are designed to fit.
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"sentence-transformers reported no dimension for {model_id}")
        self._dim = int(dim)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        arr = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(queries, self._query_prefix)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode(passages, self._passage_prefix)
