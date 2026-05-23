"""Retriever protocol shared by FAISS- and ColBERT-style backends.

``FAISSRetriever`` (dense single-vector) and ``ColBERTRetriever``
(late-interaction multi-vector) are very different machines — different
embedder shape, different index, different scoring algorithm — so they
do not share a common base class. They do, however, share the same
*interface*: a ``search(query, top_k)`` method returning a ranked list
of :class:`SearchResult` objects.

``HybridRetriever`` composes any object satisfying this protocol with an
optional reranker, so a single eval pipeline drives both retriever
families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rag_pipeline.rag.retriever import SearchResult


@runtime_checkable
class BaseRetriever(Protocol):
    """Common interface implemented by FAISS- and ColBERT-style retrievers."""

    def search(self, query: str, top_k: int = 50) -> "list[SearchResult]":
        """Return up to ``top_k`` ranked results for ``query``.

        Implementations must return :class:`SearchResult` instances in
        descending relevance order (rank 0 = best). The ``score`` field is
        backend-specific (cosine similarity, MaxSim, etc.); only the rank
        ordering is portable across implementations.
        """
        ...
