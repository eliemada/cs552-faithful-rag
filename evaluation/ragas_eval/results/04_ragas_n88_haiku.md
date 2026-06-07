# RAGAS three-arm ablation — M3 canonical (n = 88, Haiku 4.5 judge)

Full **88** evaluable gold · retriever `e5_rerank_s800_o0` · driver `scripts/run_ragas_sweep.py`

| arm | `faithfulness` | `answer_relevancy` | `context_precision` | `context_recall` | $/query |
|---|---|---|---|---|---|
| RAG · gemini-2.5-flash | 0.806 | 0.825 | 0.797 | 0.847 | 0.00029 |
| RAG · gpt-4o-mini | 0.836 | 0.835 | 0.807 | 0.864 | 0.00032 |
| LC · gemini-2.5-flash | **0.920** | 0.731 | **0.864** | **0.875** | 0.00370 |

**Report findings (Haiku):** LC wins `faithfulness` (+11.4 pp vs RAG-Gemini) and context metrics; RAG-Gemini leads `answer_relevancy` (+9.4 pp vs LC). Generator swap on RAG moves `answer_relevancy` by only ~+1 pp at n = 88 (vs +13.3 pp at the intermediate n = 30 sample).

Cross-judge Sonnet 4.6 rows: see `report/final_report/main.tex` Appendix Table `tab:ragas-app`.
