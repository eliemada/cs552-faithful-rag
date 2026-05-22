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
import logging
from typing import Iterable

import math

from evaluation.common import available_models, generate

logger = logging.getLogger(__name__)


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

    Uses a HuggingFace cross-encoder (ms-marco-MiniLM-L-6-v2) to score
    (query, doc) pairs, takes the best score, normalises to [0, 1], and
    maps to CORRECT / AMBIGUOUS / INCORRECT via the config thresholds.

    Returns:
        (quality_label, confidence_score)
    """
    if not documents:
        return RetrievalQuality.INCORRECT, 0.0

    scorer = _get_cross_encoder()
    pairs = [(query, _doc_text(doc)) for doc in documents]
    scores = scorer.predict(pairs, show_progress_bar=False)
    best = float(max(scores)) if len(scores) else 0.0
    norm = _normalize_score(best)

    if norm >= config.confidence_threshold_high:
        return RetrievalQuality.CORRECT, norm
    if norm >= config.confidence_threshold_low:
        return RetrievalQuality.AMBIGUOUS, norm
    return RetrievalQuality.INCORRECT, norm


def refine_query(query: str, failed_documents: list[dict]) -> str:
    """
    Reformulate query when retrieval quality is AMBIGUOUS.

    Asks a local LLM to rewrite the question, conditioned on snippets of
    the unsatisfying retrieved docs. Falls back to the original query if
    no model is available or generation fails.
    """
    model_spec = _pick_refine_model()
    if model_spec is None:
        return query

    snippets = _summarize_docs(failed_documents, max_docs=3, max_chars=800)
    prompt = (
        "Rewrite the user question to retrieve better evidence. "
        "Keep the original intent, add missing specifics, and avoid new facts.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved snippets:\n{snippets}\n\n"
        "Rewritten question (one sentence):"
    )

    try:
        refined = generate(model_spec, prompt, max_tokens=64, temperature=0.2).strip()
    except Exception as exc:  # pragma: no cover - defensive fallback
        _warn_refine_once(exc)
        return query

    return refined or query


_REFINE_WARNED = False


def _warn_refine_once(exc: Exception) -> None:
    global _REFINE_WARNED
    if _REFINE_WARNED:
        return
    logger.warning("Query refinement failed: %s", exc)
    _REFINE_WARNED = True


def _doc_text(doc: dict) -> str:
    if isinstance(doc, dict):
        return str(doc.get("text", ""))
    return str(getattr(doc, "text", ""))


def _summarize_docs(docs: Iterable[dict], *, max_docs: int, max_chars: int) -> str:
    lines = []
    for idx, doc in enumerate(docs):
        if idx >= max_docs:
            break
        text = _doc_text(doc).replace("\n", " ").strip()
        lines.append(f"[{idx + 1}] {text[:max_chars]}")
    return "\n".join(lines)


def _normalize_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    # Cross-encoders sometimes return logits; squash to 0-1 range.
    return 1.0 / (1.0 + math.exp(-score))


_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sentence-transformers is required for CRAG scoring") from exc

    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _CROSS_ENCODER = CrossEncoder(model_id)
    return _CROSS_ENCODER


def _pick_refine_model() -> str | None:
    specs = available_models(include_api=False)
    if not specs:
        return None
    return specs[0]


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
