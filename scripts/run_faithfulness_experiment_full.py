"""
Cross-LLM faithfulness experiment — full gold dataset.

Extends scripts/run_faithfulness_experiment.py from n=1 (single question on
Shapiro 2001) to all answerable questions in evaluation/gold_dataset/gold_qa.json.

For each (question, model) pair:
  1. Ask the model the question (closed-book, no retrieval)
  2. Extract atomic claims from the answer
  3. Verify each claim with BOTH NLI and LLM judge against the gold passage
  4. Aggregate faithfulness scores

Output:
  - Per-question/per-model raw results -> evaluation/faithfulness/results/02_full_results.json
  - Summary table printed to terminal
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load OPENROUTER_API_KEY from .env if not already in environment
if "OPENROUTER_API_KEY" not in os.environ:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
                break

from evaluation.faithfulness.faithfulness_scorer import (  # noqa: E402
    extract_claims,
    verify_claim_nli,
    verify_claim_llm_judge,
)


def ask_llm(question: str, model: str) -> str:
    prompt = f"Answer this research question concisely (2-3 sentences):\n\n{question}"
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
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def evaluate_answer(answer: str, gold_passage: str, nli_pipeline) -> dict:
    claims = extract_claims(answer)
    results = []
    for claim in claims:
        nli_label, nli_conf = verify_claim_nli(claim, gold_passage, nli_pipeline=nli_pipeline)
        try:
            judge_label, judge_conf, reason = verify_claim_llm_judge(claim, gold_passage)
            judge_label_str = judge_label.value
        except Exception as e:
            judge_label_str = "error"
            judge_conf = 0.0
            reason = str(e)
        results.append(
            {
                "claim": claim,
                "nli_label": nli_label.value,
                "nli_confidence": nli_conf,
                "judge_label": judge_label_str,
                "judge_confidence": judge_conf,
                "judge_reason": reason,
            }
        )
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

    gold_path = ROOT / "evaluation/gold_dataset/gold_qa.json"
    gold = json.loads(gold_path.read_text())

    # Filter to answerable questions that have at least one claim with a span
    pairs = [
        q for q in gold
        if q.get("claims")
        and q["difficulty"] != "unanswerable"
        and q["claims"][0].get("supporting_spans")
    ]
    print(f"Loaded {len(pairs)} answerable questions x {len(models)} models = "
          f"{len(pairs) * len(models)} runs\n")

    print("Loading NLI model...")
    from transformers import pipeline
    nli = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

    all_results = []
    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        gold_passage = pair["claims"][0]["supporting_spans"][0]["quote"]
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(pairs)}] {pair['id']} ({pair['difficulty']}, {pair['category']})")
        print(f"# Q: {question[:100]}...")
        print(f"{'#' * 70}")

        for model in models:
            print(f"\n--- {model} ---")
            try:
                answer = ask_llm(question, model)
                print(f"Answer: {answer[:200]}...")
                result = evaluate_answer(answer, gold_passage, nli)
                print(f"  Claims: {result['n_claims']}  "
                      f"NLI: {result['nli_faithfulness']:.1%}  "
                      f"Judge: {result['judge_faithfulness']:.1%}")
                all_results.append({
                    "question_id": pair["id"],
                    "difficulty": pair["difficulty"],
                    "category": pair["category"],
                    "model": model,
                    "answer": answer,
                    "n_claims": result["n_claims"],
                    "nli_faithfulness": result["nli_faithfulness"],
                    "judge_faithfulness": result["judge_faithfulness"],
                    "claims": result["claims"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "question_id": pair["id"],
                    "model": model,
                    "error": str(e),
                })
            time.sleep(0.3)

    # Save raw results
    out_dir = ROOT / "evaluation/faithfulness/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_full_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nRaw results saved to {out_path}")

    # Aggregate per-model
    print(f"\n\n{'=' * 70}")
    print("AGGREGATE FAITHFULNESS (mean across all questions)")
    print(f"{'=' * 70}")
    print(f"{'Model':<35} {'NLI':>12} {'Judge':>12} {'Gap':>10}")
    print("-" * 70)
    for model in models:
        rows = [r for r in all_results if r.get("model") == model and "error" not in r]
        if not rows:
            print(f"{model:<35} NO DATA")
            continue
        nli_avg = sum(r["nli_faithfulness"] for r in rows) / len(rows)
        judge_avg = sum(r["judge_faithfulness"] for r in rows) / len(rows)
        gap = nli_avg - judge_avg
        print(f"{model:<35} {nli_avg:>11.1%} {judge_avg:>11.1%} {gap:>+9.1%}")
    print(f"{'=' * 70}")
    print("Gap = NLI faithfulness - Judge faithfulness")
    print("Larger gap = more hallucinated specifics that NLI misses")


if __name__ == "__main__":
    main()
