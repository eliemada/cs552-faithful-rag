"""
Corrective RAG (CRAG) Implementation — Person 3

Based on: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

Pipeline:
1. Retrieve documents with standard FAISS + optional reranker
2. Score retrieval quality with a lightweight evaluator
3. If CORRECT: use retrieved docs as-is
4. If AMBIGUOUS: refine query, re-retrieve
5. If INCORRECT: fallback (web search or broader retrieval)

The evaluator can be:
- A prompted LLM (zero-shot relevance scoring)
- A fine-tuned small classifier
- An NLI-based relevance checker
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetrievalQuality(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class CRAGConfig:
    confidence_threshold_high: float = 0.7
    confidence_threshold_low: float = 0.3
    max_retries: int = 2
    use_web_fallback: bool = False


@dataclass(frozen=True)
class CRAGResult:
    original_query: str
    quality: RetrievalQuality
    confidence: float
    refined_query: str | None
    retrieval_rounds: int
    final_documents: list[dict]


def evaluate_retrieval_quality(
    query: str,
    documents: list[dict],
    config: CRAGConfig = CRAGConfig(),
) -> tuple[RetrievalQuality, float]:
    """
    Score whether retrieved documents are relevant to the query.

    TODO: Implement using one of:
    - LLM-prompted relevance scoring
    - Cross-encoder from HuggingFace (e.g., ms-marco-MiniLM)
    - NLI-based relevance check

    Returns:
        (quality_label, confidence_score)
    """
    raise NotImplementedError("Implement retrieval quality evaluator")


def refine_query(query: str, failed_documents: list[dict]) -> str:
    """
    Reformulate query when retrieval quality is AMBIGUOUS.

    TODO: Implement using LLM-based query rewriting.
    """
    raise NotImplementedError("Implement query refinement")


def corrective_rag(
    query: str,
    retriever_fn,
    config: CRAGConfig = CRAGConfig(),
) -> CRAGResult:
    """
    Full CRAG pipeline.

    Args:
        query: user question
        retriever_fn: callable(query, k) -> list[{id, text, score}]
        config: CRAG configuration
    """
    current_query = query
    retrieval_rounds = 0

    for attempt in range(config.max_retries + 1):
        retrieval_rounds += 1
        documents = retriever_fn(current_query, k=10)
        quality, confidence = evaluate_retrieval_quality(current_query, documents, config)

        if quality == RetrievalQuality.CORRECT:
            return CRAGResult(
                original_query=query,
                quality=quality,
                confidence=confidence,
                refined_query=current_query if current_query != query else None,
                retrieval_rounds=retrieval_rounds,
                final_documents=documents,
            )

        if quality == RetrievalQuality.AMBIGUOUS and attempt < config.max_retries:
            current_query = refine_query(current_query, documents)
            continue

        # INCORRECT or exhausted retries
        if config.use_web_fallback:
            # TODO: implement web search fallback
            pass

        return CRAGResult(
            original_query=query,
            quality=quality,
            confidence=confidence,
            refined_query=current_query if current_query != query else None,
            retrieval_rounds=retrieval_rounds,
            final_documents=documents,
        )

    # Should not reach here
    return CRAGResult(
        original_query=query,
        quality=RetrievalQuality.INCORRECT,
        confidence=0.0,
        refined_query=current_query,
        retrieval_rounds=retrieval_rounds,
        final_documents=[],
    )


if __name__ == "__main__":
    print("Corrective RAG module ready.")
    print("Usage: implement evaluate_retrieval_quality() and refine_query(),")
    print("then see individual notebook for full experiments.")
