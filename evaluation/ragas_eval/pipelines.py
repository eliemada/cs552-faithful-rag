"""Answer-generation pipelines fed into RAGAS.

Two pipelines that both emit ``(answer, contexts)`` for a given gold question:

* :func:`run_rag_pipeline` — the system under evaluation. Retrieves top-k
  chunks with the existing :class:`HybridRetriever` (FAISS + optional
  ZeroEntropy rerank) and asks the answer-LLM to respond *only* from those
  chunks.
* :func:`run_long_context_pipeline` — the proposal-committed baseline.
  Skips retrieval entirely; concatenates every paper cited by the gold
  question (full ``document.md``) into a single context block and asks
  the answer-LLM to respond. Requires a long-context-capable model
  (≥ 128k tokens for our largest papers).

The output schema is whatever RAGAS' :class:`datasets.Dataset` ingests::

    {
      "question":      str,
      "answer":        str,
      "contexts":      list[str],
      "ground_truth":  str,
    }

We never let the LLM "see" the gold answer or any annotation metadata —
only the retrieved (or full-document) context. That's the whole point of
the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from evaluation.common.models import generate
from evaluation.gold_dataset._validator import REPO_ROOT

DEFAULT_PROCESSED_DIR: Final[Path] = REPO_ROOT / "data" / "s3_archive" / "processed"

ANSWER_PROMPT: Final[str] = """\
You are answering a research question using ONLY the passages provided below. \
If the passages do not contain the answer, say so explicitly — do not draw on \
prior knowledge.

Passages:
{contexts}

Question: {question}

Answer in 1-3 sentences. Be precise."""


@dataclass(frozen=True)
class RagasSample:
    """One row in the dataset RAGAS consumes."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    # Provenance for debugging / failure analysis. Not consumed by RAGAS itself.
    pipeline: str
    query_id: str

    def to_ragas_dict(self) -> dict[str, str | list[str]]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


class Retriever(Protocol):
    """Structural type for any retriever exposing the eval ``search`` contract."""

    def search(self, query: str, k: int) -> list[dict]: ...


def _format_contexts(chunks: list[str]) -> str:
    """Render retrieved chunks as a numbered passage list for the prompt."""
    return "\n\n".join(f"[Passage {i + 1}]\n{c}" for i, c in enumerate(chunks))


def _read_paper_markdown(paper_id: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> str:
    """Load the LF-normalised ``document.md`` for one paper."""
    path = processed_dir / paper_id / "document.md"
    return path.read_text().replace("\r\n", "\n").replace("\r", "\n")


def run_rag_pipeline(
    *,
    question: str,
    ground_truth: str,
    query_id: str,
    retriever: Retriever,
    chunk_lookup: dict[str, str],
    answer_model: str = "api:openrouter/openai/gpt-4o-mini",
    top_k: int = 5,
) -> RagasSample:
    """Retrieve top-k chunks and generate an answer constrained to them.

    ``chunk_lookup`` maps ``chunk_id`` to the chunk's text. The retriever's
    own ``search`` returns chunk IDs + paper IDs + scores, but for RAGAS we
    need the text, so the caller pre-builds the lookup once (see
    :func:`build_chunk_lookup`).
    """
    hits = retriever.search(question, k=top_k)
    contexts = [chunk_lookup[h["chunk_id"]] for h in hits if h["chunk_id"] in chunk_lookup]
    prompt = ANSWER_PROMPT.format(contexts=_format_contexts(contexts), question=question)
    answer = generate(answer_model, prompt, max_tokens=400, temperature=0.0)
    return RagasSample(
        question=question,
        answer=answer.strip(),
        contexts=contexts,
        ground_truth=ground_truth,
        pipeline="rag",
        query_id=query_id,
    )


def run_long_context_pipeline(
    *,
    question: str,
    ground_truth: str,
    query_id: str,
    paper_ids: list[str],
    answer_model: str = "api:openrouter/google/gemini-2.5-flash",
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    max_chars_per_paper: int | None = None,
) -> RagasSample:
    """Concatenate the full text of every cited paper as context, then answer.

    Default model is Gemini 2.5 Flash — 1M-token window comfortably handles
    several full papers. Override to any long-context model accessible via
    the project's :mod:`evaluation.common.models` ``api:`` spec.

    ``max_chars_per_paper`` is an optional truncation knob for budget
    control during early experiments; leave as ``None`` for the real
    baseline run.
    """
    docs = [_read_paper_markdown(pid, processed_dir) for pid in paper_ids]
    if max_chars_per_paper is not None:
        docs = [d[:max_chars_per_paper] for d in docs]
    # Each paper becomes one "context" so RAGAS can score them separately
    # if it wants to (context_precision treats the list as ranked retrievals).
    prompt = ANSWER_PROMPT.format(contexts=_format_contexts(docs), question=question)
    answer = generate(answer_model, prompt, max_tokens=400, temperature=0.0)
    return RagasSample(
        question=question,
        answer=answer.strip(),
        contexts=docs,
        ground_truth=ground_truth,
        pipeline="long_context",
        query_id=query_id,
    )


def build_chunk_lookup(chunk_metadata_path: Path) -> dict[str, str]:
    """Build chunk_id → text from one of the aggregated metadata JSON files.

    Pass either ``coarse_metadata.json`` or ``fine_metadata.json`` —
    matched to whichever ``chunk_type`` the retriever was loaded with.
    """
    import json

    raw = json.loads(chunk_metadata_path.read_text())
    return {entry["chunk_id"]: entry["text"] for entry in raw.values()}
