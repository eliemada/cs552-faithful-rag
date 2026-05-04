"""
Faithfulness & Citation Verification — Person 2

Pipeline:
1. Take a generated RAG answer
2. Extract individual claims
3. For each claim, check if the cited passage supports it (NLI)
4. Classify: SUPPORTED / NOT_SUPPORTED / FABRICATED_SOURCE

Models:
- DeBERTa-v3-large-mnli (HuggingFace) for NLI
- LLM-based claim extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VerificationLabel(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    FABRICATED_SOURCE = "fabricated_source"
    NO_CITATION = "no_citation"


@dataclass(frozen=True)
class Claim:
    text: str
    cited_source_id: str | None
    cited_passage: str | None


@dataclass(frozen=True)
class VerificationResult:
    claim: Claim
    label: VerificationLabel
    confidence: float
    explanation: str


def extract_claims(answer: str) -> list[str]:
    """
    Extract individual factual claims from a generated answer.

    TODO: Implement using either:
    - Rule-based sentence splitting + filtering
    - LLM-prompted claim decomposition (FActScore-style)
    """
    raise NotImplementedError("Implement claim extraction")


def verify_claim_nli(
    claim: str,
    passage: str,
    model=None,
    tokenizer=None,
) -> tuple[VerificationLabel, float]:
    """
    Use NLI model to check if passage entails the claim.

    TODO: Load DeBERTa-v3-large-mnli from HuggingFace:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large-mnli"
        )
    """
    raise NotImplementedError("Implement NLI verification")


def compute_faithfulness_score(results: list[VerificationResult]) -> dict:
    """Aggregate faithfulness metrics across all claims."""
    if not results:
        return {"faithfulness": 0.0, "total_claims": 0}

    total = len(results)
    supported = sum(1 for r in results if r.label == VerificationLabel.SUPPORTED)
    fabricated = sum(1 for r in results if r.label == VerificationLabel.FABRICATED_SOURCE)

    return {
        "faithfulness": supported / total,
        "supported_ratio": supported / total,
        "not_supported_ratio": sum(1 for r in results if r.label == VerificationLabel.NOT_SUPPORTED)
        / total,
        "fabricated_ratio": fabricated / total,
        "no_citation_ratio": sum(1 for r in results if r.label == VerificationLabel.NO_CITATION)
        / total,
        "total_claims": total,
    }


if __name__ == "__main__":
    print("Faithfulness scorer ready.")
    print("Usage: implement extract_claims() and verify_claim_nli(),")
    print("then see individual notebook for full experiments.")
