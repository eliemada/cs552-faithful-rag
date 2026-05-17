"""
Hybrid retriever with FAISS + ZeroEntropy reranking.

Usage:
    from rag_pipeline.rag.retriever import HybridRetriever

    retriever = HybridRetriever.from_s3(
        bucket_name="cs433-rag-project2",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        zeroentropy_api_key=os.environ["ZEROENTROPY_API_KEY"]
    )

    results = retriever.search("What are the effects of climate change?", top_k=10)
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import requests

from rag_pipeline.rag.embedder import Embedder, OpenAIEmbedderAdapter
from rag_pipeline.rag.openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with metadata."""

    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    section_hierarchy: List[str]
    score: float
    rank: int


class FAISSRetriever:
    """FAISS-based vector similarity search."""

    def __init__(self, index: faiss.Index, metadata: Dict[str, Dict], embedder: Embedder):
        """
        Initialize FAISS retriever.

        Args:
            index: FAISS index
            metadata: Dict mapping index position to chunk metadata
            embedder: any object satisfying the :class:`Embedder` protocol
        """
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    @classmethod
    def from_path(
        cls,
        indexes_dir: Union[str, Path],
        chunk_type: str,
        openai_api_key: Optional[str] = None,
        *,
        embedder: Optional[Embedder] = None,
        index_basename: Optional[str] = None,
    ) -> "FAISSRetriever":
        """Load a FAISS retriever from a directory holding ``<basename>.faiss``
        and ``<basename>_metadata.json``.

        By default the basename is ``<chunk_type>`` (the OpenAI layout).
        Alternative embedders use ``<embedder.name>_<chunk_type>``, e.g.
        ``bge_m3_coarse.faiss``.

        If ``embedder`` is omitted, the OpenAI ``text-embedding-3-small``
        encoder is instantiated from ``openai_api_key`` so existing callers
        keep working.
        """
        indexes_dir = Path(indexes_dir)
        basename = index_basename or chunk_type
        index_path = indexes_dir / f"{basename}.faiss"
        metadata_path = indexes_dir / f"{basename}_metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index missing: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata missing: {metadata_path}")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text())
        if embedder is None:
            if not openai_api_key:
                raise ValueError("Either embedder or openai_api_key must be provided.")
            embedder = OpenAIEmbedderAdapter(
                OpenAIEmbedder(api_key=openai_api_key, model="text-embedding-3-small")
            )

        logger.info("Loaded %s index with %d vectors from %s", basename, index.ntotal, indexes_dir)
        return cls(index, metadata, embedder)

    @classmethod
    def from_s3(
        cls, bucket_name: str, chunk_type: str, openai_api_key: str, index_prefix: str = "indexes/"
    ) -> "FAISSRetriever":
        """Load a FAISS retriever from S3 (legacy — kept for the original
        ``cs433-rag-project2`` bucket, which is being retired)."""
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for S3 loading. Install it or use from_path() instead."
            ) from exc

        s3_client = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            index_path = f.name
        try:
            logger.info(
                "Downloading index from s3://%s/%s%s.faiss", bucket_name, index_prefix, chunk_type
            )
            s3_client.download_file(bucket_name, f"{index_prefix}{chunk_type}.faiss", index_path)
            index = faiss.read_index(index_path)
        finally:
            os.unlink(index_path)

        logger.info(
            "Downloading metadata from s3://%s/%s%s_metadata.json",
            bucket_name,
            index_prefix,
            chunk_type,
        )
        response = s3_client.get_object(
            Bucket=bucket_name, Key=f"{index_prefix}{chunk_type}_metadata.json"
        )
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        embedder: Embedder = OpenAIEmbedderAdapter(
            OpenAIEmbedder(api_key=openai_api_key, model="text-embedding-3-small")
        )

        logger.info("Loaded %s index with %d vectors", chunk_type, index.ntotal)
        return cls(index, metadata, embedder)

    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Embed query (encoder returns L2-normalised float32 vectors)
        query_vector = self.embedder.encode_queries([query])

        # Search
        distances, indices = self.index.search(query_vector, top_k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # No more results
                break

            meta = self.metadata[str(idx)]
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    paper_id=meta["paper_id"],
                    paper_title=meta["paper_title"],
                    text=meta["text"],
                    section_hierarchy=meta["section_hierarchy"],
                    score=float(score),
                    rank=rank,
                )
            )

        return results


class ZeroEntropyReranker:
    """ZeroEntropy reranking API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.zeroentropy.dev/v1"):
        """
        Initialize ZeroEntropy reranker.

        Args:
            api_key: ZeroEntropy API key
            base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url

    def rerank(
        self, query: str, results: List[SearchResult], top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank search results using ZeroEntropy.

        Args:
            query: Original search query
            results: List of SearchResult to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult
        """
        if not results:
            return []

        # Prepare documents for reranking
        documents = [r.text for r in results]

        # Call ZeroEntropy API
        response = requests.post(
            f"{self.base_url}/models/rerank",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": "zerank-1",  # ZeroEntropy reranking model (or "zerank-1-small" for faster)
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"ZeroEntropy rerank failed: {response.status_code} - {response.text}")
            # Fallback: return original results
            return results[:top_k]

        # Parse response
        reranked_data = response.json()

        # Build reranked results
        reranked_results = []
        for rank, item in enumerate(reranked_data.get("results", [])):
            original_idx = item["index"]
            original_result = results[original_idx]

            reranked_results.append(
                SearchResult(
                    chunk_id=original_result.chunk_id,
                    paper_id=original_result.paper_id,
                    paper_title=original_result.paper_title,
                    text=original_result.text,
                    section_hierarchy=original_result.section_hierarchy,
                    score=item.get("relevance_score", original_result.score),
                    rank=rank,
                )
            )

        return reranked_results


class HybridRetriever:
    """
    Hybrid retriever combining FAISS retrieval with ZeroEntropy reranking.

    Flow:
    1. FAISS retrieves top-N candidates (fast, embedding similarity)
    2. ZeroEntropy reranks to top-K (accurate, relevance scoring)
    """

    def __init__(
        self,
        faiss_retriever: FAISSRetriever,
        reranker: Optional[ZeroEntropyReranker] = None,
        faiss_candidates: int = 75,
    ):
        """
        Initialize hybrid retriever.

        Args:
            faiss_retriever: FAISS retriever for initial search
            reranker: Optional ZeroEntropy reranker
            faiss_candidates: Number of candidates to retrieve from FAISS
        """
        self.faiss_retriever = faiss_retriever
        self.reranker = reranker
        self.faiss_candidates = faiss_candidates

    @classmethod
    def from_path(
        cls,
        indexes_dir: Union[str, Path],
        openai_api_key: Optional[str] = None,
        zeroentropy_api_key: Optional[str] = None,
        chunk_type: str = "coarse",
        faiss_candidates: int = 75,
        *,
        embedder: Optional[Embedder] = None,
        index_basename: Optional[str] = None,
    ) -> "HybridRetriever":
        """Load a hybrid retriever from a local directory of FAISS indexes.

        Backwards-compat default: the OpenAI ``text-embedding-3-small`` index
        named ``<chunk_type>.faiss``. Pass ``embedder`` and ``index_basename``
        to load an alternative-embedder index (e.g. ``bge_m3_coarse.faiss``).
        """
        faiss_retriever = FAISSRetriever.from_path(
            indexes_dir=indexes_dir,
            chunk_type=chunk_type,
            openai_api_key=openai_api_key,
            embedder=embedder,
            index_basename=index_basename,
        )
        reranker = ZeroEntropyReranker(api_key=zeroentropy_api_key) if zeroentropy_api_key else None
        return cls(faiss_retriever, reranker, faiss_candidates)

    @classmethod
    def from_s3(
        cls,
        bucket_name: str,
        openai_api_key: str,
        zeroentropy_api_key: Optional[str] = None,
        chunk_type: str = "coarse",
        faiss_candidates: int = 75,
    ) -> "HybridRetriever":
        """Load a hybrid retriever from S3 (legacy — bucket retired)."""
        faiss_retriever = FAISSRetriever.from_s3(
            bucket_name=bucket_name, chunk_type=chunk_type, openai_api_key=openai_api_key
        )
        reranker = ZeroEntropyReranker(api_key=zeroentropy_api_key) if zeroentropy_api_key else None
        return cls(faiss_retriever, reranker, faiss_candidates)

    def search(self, query: str, top_k: int = 10, use_reranker: bool = True) -> List[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of final results
            use_reranker: Whether to use ZeroEntropy reranking

        Returns:
            List of SearchResult objects
        """
        # Step 1: FAISS retrieval
        candidates = self.faiss_retriever.search(query, self.faiss_candidates)

        # Step 2: Reranking (if available and enabled)
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, candidates, top_k)
        else:
            results = candidates[:top_k]

        return results

    def search_with_context(
        self, query: str, top_k: int = 10, context_window: int = 1
    ) -> List[Dict]:
        """
        Search and include surrounding chunks for context.

        Args:
            query: Search query
            top_k: Number of results
            context_window: Number of chunks before/after to include

        Returns:
            List of result dicts with context
        """
        results = self.search(query, top_k)

        # TODO: Implement context expansion by loading adjacent chunks
        # For now, return results as-is
        return [
            {
                "chunk_id": r.chunk_id,
                "paper_id": r.paper_id,
                "paper_title": r.paper_title,
                "text": r.text,
                "section_hierarchy": r.section_hierarchy,
                "score": r.score,
                "rank": r.rank,
            }
            for r in results
        ]


# Convenience function for quick usage
def create_retriever(
    openai_api_key: Optional[str] = None,
    zeroentropy_api_key: Optional[str] = None,
    bucket_name: str = "cs433-rag-project2",
) -> HybridRetriever:
    """
    Create a hybrid retriever with default settings.

    Args:
        openai_api_key: OpenAI API key (defaults to env var)
        zeroentropy_api_key: ZeroEntropy API key (defaults to env var)
        bucket_name: S3 bucket name

    Returns:
        Configured HybridRetriever
    """
    openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    zeroentropy_key = zeroentropy_api_key or os.environ.get("ZEROENTROPY_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not provided")

    return HybridRetriever.from_s3(
        bucket_name=bucket_name, openai_api_key=openai_key, zeroentropy_api_key=zeroentropy_key
    )
