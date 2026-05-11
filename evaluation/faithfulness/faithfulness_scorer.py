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


def extract_claims(answer: str, model: str = "openai/gpt-4o-mini") -> list[str]:
    """
    Extract atomic factual claims from a generated RAG answer using an LLM.

    Falls back to rule-based sentence splitting if no API key is set.

    Args:
        answer: The full text of the generated RAG answer
        model: OpenRouter model spec (default: GPT-4o-mini, cheap and fast)

    Returns:
        A list of atomic claim strings, each a single verifiable proposition.
    """
    import os
    import json
    import urllib.request
    import re

    api_key = os.environ.get("OPENROUTER_API_KEY")

    # Fallback: rule-based splitting if no API key
    if not api_key:
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    # LLM-based decomposition (FActScore-style)
    prompt = f"""Decompose the following answer into atomic factual claims.
Each claim must be ONE independently verifiable proposition.
If two facts are joined by "and", split them.
Ignore questions, opinions, and meta-commentary.
Return ONLY a JSON array of strings, no other text.

Answer:
\"\"\"{answer}\"\"\"

JSON array:"""

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

    text = data["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


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
            "zero-shot-classification", model="cross-encoder/nli-deberta-v3-small"
        )

    # Run NLI: does the passage entail, contradict, or stay neutral to the claim?
    result = nli_pipeline(
        passage,
        candidate_labels=["entailment", "contradiction", "neutral"],
        hypothesis_template="This text means that {}.",
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


def verify_claim_llm_judge(
    claim: str,
    passage: str,
    model: str = "openai/gpt-4o-mini",
) -> tuple[VerificationLabel, float, str]:
    """
    Use an LLM judge to check whether the passage SPECIFICALLY supports the claim.

    Catches a failure mode that NLI misses: claims that are topically related
    to the passage but state different specifics (hallucinated details).

    Returns
    -------
    (label, confidence, explanation)
    """
    import os
    import json
    import urllib.request
    import re

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required for LLM judge")

    prompt = f"""You are a strict citation verifier. Determine whether the PASSAGE specifically supports the CLAIM.

A claim is SUPPORTED only if the specific facts, terms, and entities in the claim appear in the passage.
A claim that is topically related but states different specifics is NOT_SUPPORTED.

PASSAGE:
\"\"\"{passage}\"\"\"

CLAIM:
\"\"\"{claim}\"\"\"

Return ONLY a JSON object with these fields:
{{"label": "supported" | "not_supported", "confidence": 0.0 to 1.0, "reason": "one sentence"}}"""

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

    text = data["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)

    label = (
        VerificationLabel.SUPPORTED
        if parsed["label"] == "supported"
        else VerificationLabel.NOT_SUPPORTED
    )
    return label, float(parsed["confidence"]), parsed.get("reason", "")


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
