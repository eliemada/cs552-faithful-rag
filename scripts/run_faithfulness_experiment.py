"""
Cross-LLM faithfulness experiment with dual verification (NLI + LLM judge).

Compares NLI-based and LLM-judge faithfulness scores to expose hallucination
patterns NLI misses.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load API key from .env
env = (ROOT / ".env").read_text()
for line in env.splitlines():
    if line.startswith("OPENROUTER_API_KEY="):
        os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
        break

from evaluation.faithfulness.faithfulness_scorer import (
    extract_claims,
    verify_claim_nli,
    verify_claim_llm_judge,
    VerificationLabel,
)


def ask_llm(question: str, model: str) -> str:
    """Ask an LLM to answer a question with no retrieval."""
    prompt = f"Answer this research question concisely (2-3 sentences):\n\n{question}"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def evaluate_answer(answer: str, gold_passage: str, nli_pipeline) -> dict:
    """Extract claims and run BOTH NLI and LLM judge on each."""
    claims = extract_claims(answer)
    results = []
    for claim in claims:
        nli_label, nli_conf = verify_claim_nli(claim, gold_passage, nli_pipeline=nli_pipeline)
        judge_label, judge_conf, reason = verify_claim_llm_judge(claim, gold_passage)
        results.append({
            "claim": claim,
            "nli_label": nli_label.value,
            "nli_confidence": nli_conf,
            "judge_label": judge_label.value,
            "judge_confidence": judge_conf,
            "judge_reason": reason,
        })
    n_supported_nli = sum(1 for r in results if r["nli_label"] == "supported")
    n_supported_judge = sum(1 for r in results if r["judge_label"] == "supported")
    return {
        "n_claims": len(results),
        "nli_faithfulness": n_supported_nli / len(results) if results else 0.0,
        "judge_faithfulness": n_supported_judge / len(results) if results else 0.0,
        "claims": results,
    }


def main():
    models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-haiku",
        "deepseek/deepseek-chat",
    ]

    gold = json.loads(
        (ROOT / "evaluation/gold_dataset/contributions/andrea.json").read_text()
    )

    print("Loading NLI model...")
    from transformers import pipeline
    nli = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

    pair = next(p for p in gold if p["difficulty"] != "unanswerable")
    question = pair["question"]
    gold_passage = pair["claims"][0]["supporting_spans"][0]["quote"]

    print(f"\nQuestion: {question}\n")
    print(f"Gold passage: {gold_passage[:200]}...\n")

    summary = {}
    for model in models:
        print(f"\n{'=' * 60}\n{model}\n{'=' * 60}")
        try:
            answer = ask_llm(question, model)
            print(f"\nAnswer: {answer}\n")
            result = evaluate_answer(answer, gold_passage, nli)
            print(f"\nPer-claim breakdown:")
            for c in result["claims"]:
                marker_nli = "✓" if c["nli_label"] == "supported" else "✗"
                marker_judge = "✓" if c["judge_label"] == "supported" else "✗"
                print(f"  NLI:{marker_nli}  Judge:{marker_judge}  {c['claim'][:80]}")
                if c["judge_label"] != "supported":
                    print(f"     └─ Judge reason: {c['judge_reason']}")
            print(f"\nNLI faithfulness:   {result['nli_faithfulness']:.1%}")
            print(f"Judge faithfulness: {result['judge_faithfulness']:.1%}")
            summary[model] = result
        except Exception as e:
            print(f"Error: {e}")
            summary[model] = {"error": str(e)}

    print(f"\n\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"{'Model':<35} {'NLI':<10} {'Judge':<10}")
    for model, res in summary.items():
        if "error" in res:
            print(f"{model:<35} ERROR")
        else:
            print(f"{model:<35} {res['nli_faithfulness']:<10.1%} {res['judge_faithfulness']:<10.1%}")


if __name__ == "__main__":
    main()
