"""Corrective RAG (CRAG) implementation — Person 3.

Based on: "Corrective Retrieval Augmented Generation" (Yan et al., 2024).

Pipeline
--------
1. Retrieve documents with FAISS + optional reranker.
2. Score retrieval quality with a cross-encoder.
3. If CORRECT  → use the retrieved docs as-is.
4. If AMBIGUOUS → refine the query with an LLM and re-retrieve.
5. If INCORRECT (or AMBIGUOUS retries exhausted) → abstain: drop the bad
   documents (``final_documents=[]``, ``abstained=True``) so the downstream
   generator emits an explicit "not enough evidence" answer rather than
   grounding on low-relevance context. See ``INCORRECT_BRANCH_PLAN.md``
   (Option A); off-corpus web fallback is deliberately out of scope.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from evaluation.common import available_models, generate

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RetrievalQuality(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class CRAGConfig:
    """Configuration for the CRAG pipeline.

    Every previously-hardcoded knob lives here so experiments can sweep
    thresholds, swap the cross-encoder, or override the refine LLM
    without touching module code.
    """

    # Quality thresholds applied to the normalised cross-encoder score.
    confidence_threshold_high: float = 0.7
    confidence_threshold_low: float = 0.3

    # Retry budget for AMBIGUOUS retrievals (each retry runs refine + re-retrieve).
    max_retries: int = 2

    # Top-k for each retrieval call inside the pipeline.
    retrieval_k: int = 10

    # When True (default), an INCORRECT verdict (or exhausted AMBIGUOUS
    # retries) abstains: the result carries no documents and ``abstained=True``
    # so the generator can emit an explicit "no answer" instead of grounding
    # on low-relevance context. Set False to keep the legacy behaviour of
    # returning the low-quality documents unchanged.
    use_abstain_fallback: bool = True

    # The answer emitted on abstain. Mirrors the unanswerable stratum in the
    # gold set ("Not covered by the corpus.") so abstain precision/recall is
    # measurable against existing ground truth.
    abstain_message: str = "I don't have enough evidence in the corpus to answer this faithfully."

    # Cross-encoder for relevance scoring. ms-marco-MiniLM is fast (~25M
    # params) and well-calibrated for ad-hoc (query, passage) relevance.
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Query-refinement prompt knobs.
    refine_max_docs: int = 3
    refine_max_chars_per_doc: int = 800
    refine_max_tokens: int = 64
    refine_temperature: float = 0.2

    # Optional override for the refine LLM. When None, the first local
    # model from `available_models(include_api=False)` is used (Qwen 7B).
    refine_model_spec: str | None = None


@dataclass(frozen=True)
class CRAGResult:
    original_query: str
    quality: RetrievalQuality
    confidence: float
    refined_query: str | None
    retrieval_rounds: int
    final_documents: list[dict]
    # True when the pipeline gave up on low-relevance retrieval and chose to
    # abstain rather than ground an answer on bad context. Implies
    # ``final_documents == []``.
    abstained: bool = False


# Type alias for any callable taking `(query, k)` and returning hit dicts.
Retriever = Callable[..., list[dict]]


def evaluate_retrieval_quality(
    query: str,
    documents: list[dict],
    config: CRAGConfig = CRAGConfig(),
) -> tuple[RetrievalQuality, float]:
    """Score retrieval quality and map to {CORRECT, AMBIGUOUS, INCORRECT}.

    Scores every (query, doc) pair with the configured cross-encoder,
    keeps the best score, normalises to [0, 1] (sigmoid when the model
    returns raw logits), and thresholds against the two confidence cuts.
    """
    if not documents:
        return RetrievalQuality.INCORRECT, 0.0

    scorer = _get_cross_encoder(config.cross_encoder_model)
    pairs = [(query, _doc_text(doc)) for doc in documents]
    scores = scorer.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
    best = float(max(scores))
    norm = _sigmoid_if_logit(best)

    if norm >= config.confidence_threshold_high:
        return RetrievalQuality.CORRECT, norm
    if norm >= config.confidence_threshold_low:
        return RetrievalQuality.AMBIGUOUS, norm
    return RetrievalQuality.INCORRECT, norm


def refine_query(
    query: str,
    failed_documents: list[dict],
    config: CRAGConfig = CRAGConfig(),
) -> str:
    """Rewrite the query with an LLM, conditioned on unsatisfying snippets.

    Falls back to the original query when no local model is reachable or
    when generation raises. The first failure logs at WARNING; subsequent
    failures stay at DEBUG to avoid log spam during a sweep.
    """
    model_spec = _pick_refine_model(config.refine_model_spec)
    if model_spec is None:
        return query

    snippets = _summarize_docs(
        failed_documents,
        max_docs=config.refine_max_docs,
        max_chars=config.refine_max_chars_per_doc,
    )
    prompt = (
        "Rewrite the user question to retrieve better evidence. "
        "Keep the original intent, add missing specifics, and avoid new facts.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved snippets:\n{snippets}\n\n"
        "Rewritten question (one sentence):"
    )

    try:
        refined = generate(
            model_spec,
            prompt,
            max_tokens=config.refine_max_tokens,
            temperature=config.refine_temperature,
        ).strip()
    except Exception as exc:
        logger.debug("Query refinement failed: %s", exc, exc_info=True)
        _emit_refine_failure_warning()
        return query

    return refined or query


def corrective_rag(
    query: str,
    retriever_fn: Retriever,
    config: CRAGConfig = CRAGConfig(),
) -> CRAGResult:
    """Full CRAG pipeline.

    Args:
        query: user question.
        retriever_fn: callable ``(query: str, k: int) -> list[dict]``.
        config: pipeline configuration (see :class:`CRAGConfig`).
    """
    current_query = query
    retrieval_rounds = 0

    for attempt in range(config.max_retries + 1):
        retrieval_rounds += 1
        documents = retriever_fn(current_query, k=config.retrieval_k)
        quality, confidence = evaluate_retrieval_quality(current_query, documents, config)

        refined_marker = current_query if current_query != query else None

        if quality == RetrievalQuality.CORRECT:
            return CRAGResult(
                original_query=query,
                quality=quality,
                confidence=confidence,
                refined_query=refined_marker,
                retrieval_rounds=retrieval_rounds,
                final_documents=documents,
                abstained=False,
            )

        if quality == RetrievalQuality.AMBIGUOUS and attempt < config.max_retries:
            current_query = refine_query(current_query, documents, config)
            continue

        # INCORRECT, or AMBIGUOUS retries exhausted. Per INCORRECT_BRANCH_PLAN.md
        # (Option A), abstain: drop the low-relevance docs so the generator emits
        # an explicit "no answer" rather than grounding on bad context.
        if config.use_abstain_fallback:
            return CRAGResult(
                original_query=query,
                quality=quality,
                confidence=confidence,
                refined_query=refined_marker,
                retrieval_rounds=retrieval_rounds,
                final_documents=[],
                abstained=True,
            )

        return CRAGResult(
            original_query=query,
            quality=quality,
            confidence=confidence,
            refined_query=refined_marker,
            retrieval_rounds=retrieval_rounds,
            final_documents=documents,
            abstained=False,
        )

    # Unreachable: the loop always returns. Kept as a defensive fallback
    # so static analysers don't complain about a missing return path.
    return CRAGResult(
        original_query=query,
        quality=RetrievalQuality.INCORRECT,
        confidence=0.0,
        refined_query=current_query,
        retrieval_rounds=retrieval_rounds,
        final_documents=[],
    )


def finalize_answer(
    result: CRAGResult,
    answer_fn: Callable[[CRAGResult], str],
    config: CRAGConfig = CRAGConfig(),
) -> str:
    """Resolve a :class:`CRAGResult` into a final answer string.

    When the pipeline abstained, returns ``config.abstain_message`` and never
    calls the generator — this is what makes the INCORRECT branch *do* something
    measurable (abstain vs. hallucinate). Otherwise delegates to ``answer_fn``,
    which receives the result (with ``final_documents``) and returns the
    generated answer. Keeping the generator injectable makes this testable
    without an LLM.
    """
    if result.abstained:
        return config.abstain_message
    return answer_fn(result)


# ---------- private helpers ----------


def _doc_text(doc: Any) -> str:
    """Best-effort text extraction from a retrieval result.

    Accepts dict-shaped hits (``{"text": "..."}``) or any object with a
    ``.text`` attribute. Returns an empty string when neither is present
    so the caller can still build a (query, doc) pair.
    """
    if isinstance(doc, dict):
        return str(doc.get("text", ""))
    return str(getattr(doc, "text", ""))


def _summarize_docs(
    docs: Iterable[dict],
    *,
    max_docs: int,
    max_chars: int,
) -> str:
    """Build a compact, prompt-friendly summary of retrieved snippets."""
    lines: list[str] = []
    for idx, doc in enumerate(docs):
        if idx >= max_docs:
            break
        text = _doc_text(doc).replace("\n", " ").strip()
        lines.append(f"[{idx + 1}] {text[:max_chars]}")
    return "\n".join(lines)


def _sigmoid_if_logit(score: float) -> float:
    """Return ``score`` unchanged if already a [0, 1] probability, else
    squash through a logistic sigmoid (cross-encoders sometimes return
    raw logits depending on the head)."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


@functools.lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Lazily load a cross-encoder, cached per (model_name) across calls.

    ``maxsize=4`` covers ablations that compare a handful of scorers in
    the same process without thrashing GPU memory.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for CRAG retrieval scoring "
            "(`uv add sentence-transformers`)."
        ) from exc

    logger.info("Loading cross-encoder %s", model_name)
    return CrossEncoder(model_name)


def _pick_refine_model(preferred: str | None) -> str | None:
    """Select the LLM for query refinement.

    Honours ``preferred`` when it's locally available; otherwise picks
    the first local spec. Returns ``None`` when nothing is reachable so
    the caller can short-circuit to the original query.
    """
    specs = available_models(include_api=False)
    if not specs:
        return None
    if preferred and preferred in specs:
        return preferred
    return specs[0]


@functools.cache
def _emit_refine_failure_warning() -> None:
    """Log the first refinement failure at WARNING; subsequent failures
    are suppressed by the cache. Full exception detail is still emitted
    at DEBUG level by the call site."""
    logger.warning(
        "Query refinement failed; falling back to the original query. "
        "Subsequent failures are suppressed — re-run with DEBUG logging "
        "for full tracebacks."
    )


if __name__ == "__main__":
    print("Corrective RAG module ready.")
    print("Use `corrective_rag(query, retriever_fn)` from the notebook.")
