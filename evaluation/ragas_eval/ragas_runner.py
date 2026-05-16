"""RAGAS metric runner.

Computes the four canonical RAGAS metrics over a list of
:class:`pipelines.RagasSample` rows:

* **Faithfulness** — fraction of generated claims directly supported by
  the retrieved contexts. Catches answer-side hallucinations.
* **Answer relevancy** — how well the answer addresses the question (LLM
  reverse-generates plausible questions from the answer; cosine of those
  embeddings against the original question).
* **Context precision** — among the retrieved passages, fraction that
  contain information relevant to the ground-truth answer. Catches
  retriever noise.
* **Context recall** — fraction of the ground-truth answer's claims that
  are entailed by retrieved contexts. Catches retriever gaps.

The judge LLM is wired to OpenRouter via langchain-openai (uses
``OPENROUTER_API_KEY``); embeddings go through the OpenAI client
directly with ``OPENAI_API_KEY`` (OpenRouter doesn't proxy embeddings
the same way).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Final

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

from evaluation.ragas_eval.pipelines import RagasSample

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL: Final[str] = "openai/gpt-4o-mini"
DEFAULT_EMBED_MODEL: Final[str] = "text-embedding-3-small"
METRIC_NAMES: Final[tuple[str, ...]] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _build_metrics() -> list:
    """Fresh metric instances per evaluate() call so LLMs/embeddings can vary.

    Result column names produced by these classes (must match METRIC_NAMES):
      Faithfulness                   → 'faithfulness'
      AnswerRelevancy                → 'answer_relevancy'
      ContextPrecision               → 'context_precision'
      LLMContextRecall               → 'context_recall'
    """
    return [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        LLMContextRecall(),
    ]


@dataclass(frozen=True)
class RagasResult:
    """Outcome of one RAGAS evaluation run."""

    per_sample: list[dict]  # one dict per input sample with all metric values
    aggregate: dict[str, float]  # metric → mean over samples
    n: int


def _build_judge_llm(model: str) -> Any:
    """Wrap a langchain ChatOpenAI pointing at OpenRouter for RAGAS's judge calls."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for the RAGAS judge LLM. "
            "Set it in the environment or pass a different llm to evaluate_samples."
        )
    chat = ChatOpenAI(
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(api_key),
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat)


def _build_embeddings(model: str) -> Any:
    """OpenAI embeddings for answer_relevancy (OpenRouter doesn't proxy embeddings)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for RAGAS answer-relevancy embeddings. "
            "answer_relevancy can also be skipped by overriding `metrics`."
        )
    embedder = OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))
    return LangchainEmbeddingsWrapper(embedder)


def evaluate_samples(
    samples: list[RagasSample],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    metrics: list | None = None,
) -> RagasResult:
    """Run RAGAS evaluate over the samples and return a tidy result.

    Empty input is allowed and short-circuits to a zero-rows result so
    the CLI doesn't have to guard.
    """
    if not samples:
        return RagasResult(per_sample=[], aggregate={m: 0.0 for m in METRIC_NAMES}, n=0)

    rows = [s.to_ragas_dict() for s in samples]
    dataset = Dataset.from_list(rows)

    llm = _build_judge_llm(judge_model)
    embeddings = _build_embeddings(embed_model)

    logger.info("Running RAGAS over %d samples with judge=%s", len(samples), judge_model)
    result = evaluate(
        dataset=dataset,
        metrics=metrics if metrics is not None else _build_metrics(),
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # evaluate() can return either an EvaluationResult or an Executor depending
    # on the lazy/eager mode; both expose to_pandas() in current ragas builds.
    df = result.to_pandas()  # ty: ignore[unresolved-attribute]
    per_sample: list[dict] = []
    for sample, row in zip(samples, df.to_dict(orient="records")):
        record: dict[str, str | float | None] = {
            "query_id": sample.query_id,
            "pipeline": sample.pipeline,
            "question": sample.question,
        }
        for m in METRIC_NAMES:
            if m in row:
                value = row[m]
                # RAGAS returns NaN for failed/skipped samples; pass through as
                # None so it doesn't poison aggregates downstream.
                record[m] = float(value) if value == value else None
        per_sample.append(record)

    aggregate: dict[str, float] = {}
    for m in METRIC_NAMES:
        values = [r[m] for r in per_sample if r.get(m) is not None]
        aggregate[m] = sum(values) / len(values) if values else 0.0
    return RagasResult(per_sample=per_sample, aggregate=aggregate, n=len(samples))
