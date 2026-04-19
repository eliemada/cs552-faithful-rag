"""
RAGAS Evaluation Runner — Person 4

Runs end-to-end evaluation using the RAGAS framework.
Compares chunked RAG vs long-context approaches.

Metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Install: pip install ragas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalConfig:
    gold_dataset_path: Path = Path("evaluation/gold_dataset/gold_qa.json")
    output_dir: Path = Path("evaluation/ragas_eval/results")
    models_to_test: tuple[str, ...] = (
        "openrouter/google/gemini-2.5-pro",
        "openrouter/anthropic/claude-sonnet-4",
        "openrouter/deepseek/deepseek-chat",
    )


def run_ragas_evaluation(config: EvalConfig = EvalConfig()) -> None:
    """
    Run RAGAS evaluation on the RAG pipeline.

    TODO: Implement:
    1. Load gold dataset
    2. For each question, run RAG pipeline to get (answer, contexts)
    3. Build RAGAS Dataset
    4. Evaluate with ragas.evaluate()
    5. Save results

    Example:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    """
    raise NotImplementedError("Implement RAGAS evaluation")


def run_long_context_baseline(config: EvalConfig = EvalConfig()) -> None:
    """
    Compare against long-context approach (128K context window).

    Instead of chunked retrieval, stuff full paper text into the context.

    TODO: Implement:
    1. For each question, identify relevant papers
    2. Concatenate full paper text (up to 128K tokens)
    3. Send to LLM with the question
    4. Evaluate with same RAGAS metrics
    5. Compare cost + latency + accuracy
    """
    raise NotImplementedError("Implement long-context baseline")


if __name__ == "__main__":
    print("RAGAS evaluation runner ready.")
    print("Install: pip install ragas datasets")
    print("See individual notebook for full experiments.")
