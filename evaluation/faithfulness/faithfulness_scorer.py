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
    
    Currently uses rule-based sentence splitting (placeholder).
    TODO: Replace with LLM-prompted claim decomposition (FActScore-style)
          once OpenRouter API key is available.
    
    Args:
        answer: The full text of the generated RAG answer
    
    Returns:
        A list of atomic claim strings
    """
    import re

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    
    # Filter out very short fragments and empty strings
    claims = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    return claims


def verify_claim_nli(
    claim: str,
    passage: str,
    nli_pipeline=None,
) -> tuple[VerificationLabel, float]:
    """
    Use NLI model to check if passage entails the claim.
    
    Args:
        claim: The atomic claim to verify
        passage: The cited passage that supposedly supports the claim
        nli_pipeline: Optional pre-loaded HuggingFace pipeline (for efficiency).
                      If None, loads the model on first call.
    
    Returns:
        Tuple of (VerificationLabel, confidence score)
    """
    from transformers import pipeline

    # Load model if not provided (allows reuse across many calls)
    if nli_pipeline is None:
        nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small"
        )

    # Run NLI: does the passage entail, contradict, or stay neutral to the claim?
    result = nli_pipeline(
        passage,
        candidate_labels=["entailment", "contradiction", "neutral"],
        hypothesis_template="This text means that {}."
    )
    
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    # Map NLI labels to our citation labels
    if top_label == "entailment" and top_score > 0.5:
        return VerificationLabel.SUPPORTED, round(top_score, 3)
    elif top_label == "contradiction":
        return VerificationLabel.NOT_SUPPORTED, round(top_score, 3)
    else:
        return VerificationLabel.NOT_SUPPORTED, round(top_score, 3)


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
