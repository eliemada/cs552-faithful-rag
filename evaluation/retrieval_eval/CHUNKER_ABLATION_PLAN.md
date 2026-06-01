# Chunker ablation — M3 plan (TA-feedback-driven)

**Owner:** Elie · **Target:** numbers landed by 2026-06-04 · **For §Results/Retrieval in the final report.**

## Why this ablation

TA feedback on M2:

> Since you've already pinpointed the chunker as the dominant bottleneck across both the multi-hop hit@10 drop and the open recall gap, it would really strengthen the work to add a quick experiment varying chunk size or overlap (or a semantic/recursive chunker) to show you can actually close some of that gap before the final report.

M2 result that motivates this: multi-hop hit@10 falls from **1.000 on coarse** (~2000 chars) to **0.667–0.833 on fine** (~300 chars). The current `MarkdownChunker` uses two hard-coded granularities (`coarse_target_size=2000` / `fine_target_size=300`, `coarse_overlap_pct=0.10` / `fine_overlap_pct=0.20`); nothing in between is tested. The chunk-level table in `comparison.md` shows fine_faiss bottoming out at hit@5=0.250 — a clear opening.

## Anchor — best M2 config (do not re-vary)

Per `comparison.md`, the corpus-level SOTA is **`e5_large_coarse_rerank`** (hit@10 0.973, MRR 0.878). Fix:

- embedder = E5-large
- reranker = ZeroEntropy ON
- evaluator = paper-level + chunk-level metrics on the 93-claim gold (was 37 in M2 — re-run uses the post-#48/#47 gold).

The ablation then varies **only** chunker dimensions on top of this anchor.

## Minimum-viable grid

10 new configs, ~1 dim of variation at a time, anchored on E5-large coarse+rerank.

### A. Fixed-size sweep (8 configs) — closes the size/overlap gap

| variant key             | chunk_size (chars) | overlap |
|-------------------------|--------------------|---------|
| `e5_rerank_s200_o0`     | 200                | 0       |
| `e5_rerank_s200_o100`   | 200                | 100 (50%) |
| `e5_rerank_s400_o0`     | 400                | 0       |
| `e5_rerank_s400_o200`   | 400                | 200 (50%) |
| `e5_rerank_s600_o0`     | 600                | 0       |
| `e5_rerank_s600_o300`   | 600                | 300 (50%) |
| `e5_rerank_s800_o0`     | 800                | 0       |
| `e5_rerank_s800_o400`   | 800                | 400 (50%) |

Drops 25% overlap to keep the grid small. If 0% vs 50% looks linear we can add 25% mid-points; if non-monotonic we add them anyway.

### B. Chunker-strategy variants (2 configs) — closes "semantic/recursive" ask

Use `DocumentChunker` from `packages/shared/rag_pipeline/rag/chunking.py` (already implements `recursive_chunking` and `semantic_chunking` — no new chunker code needed).

| variant key                  | strategy                                              |
|------------------------------|-------------------------------------------------------|
| `e5_rerank_recursive_400`    | `recursive_chunking(chunk_size=400, chunk_overlap=80)` (LangChain `RecursiveCharacterTextSplitter` family) |
| `e5_rerank_semantic`         | `semantic_chunking()` (sentence-embedding-similarity splitter) |

`400` chosen as the recursive baseline because it's the midpoint of the fixed-size sweep — keeps comparison clean.

## Pipeline cost estimate

Per config the work is: chunk-JSON regen → FAISS rebuild → eval-loop. From M2 timings (RCP, 1× A100 typical):

- Chunk-JSON regen across 36 papers: ~1–2 min/config
- FAISS rebuild (E5-large, ~50k chunks/config): ~3–5 min/config
- Eval on 93 queries × top-20: ~30s/config
- **10 configs → ~60–80 min total** on RCP. Fits in one job submission.

## Code changes needed (Elie — one PR)

1. **`scripts/generate_chunk_variants.py`** (new, ~80 lines)
   - Reads `data/s3_archive/papers/<paper_id>/document.md`.
   - For each variant in the grid, instantiates `DocumentChunker` and writes `data/s3_archive/chunks/<paper_id>_<variant_key>.json` in the same schema `MarkdownChunker.chunk_document()` produces.
   - Important: schema must include `chunk_id`, `paper_id`, `text`, `start_char`, `end_char` so `gold_resolver.py` can re-score chunk-level metrics correctly.

2. **`scripts/build_hf_index.py`** — already accepts `--chunk-type`; extend to accept any string (not just `coarse`/`fine`) and emit `<model>_<variant_key>.faiss`. Tiny change.

3. **`evaluation/retrieval_eval/retrievers.py`** — append 10 `RetrieverConfig` entries to `CONFIGS`, all `embedder=e5_large`, `reranker=True`, `chunk_type=<variant_key>`.

4. **`scripts/run_retrieval_ablation.py`** — no change; auto-picks up new `CONFIGS`.

5. **`evaluation/retrieval_eval/results/comparison_chunker.md`** — new output table written by `comparison.py` filtered on the new configs.

## RCP launch (single job, sequenced)

```bash
# from repo root, inside the rcp_support container
uv run python -m scripts.generate_chunk_variants --grid evaluation/retrieval_eval/CHUNKER_GRID.json
for v in s200_o0 s200_o100 s400_o0 s400_o200 s600_o0 s600_o300 s800_o0 s800_o400 recursive_400 semantic; do
  uv run python -m scripts.build_hf_index --model e5-large --chunk-type $v
done
uv run python -m scripts.run_retrieval_ablation --only-prefix e5_rerank_
```

## Report integration (§Results/Retrieval)

- 1 new table: chunker variants × {hit@5, hit@10, MRR, chunk-recall@10}, with the M2 baseline row (`e5_large_coarse_rerank`) at top for reference.
- 1 new plot: hit@10 vs chunk_size with two lines (overlap 0% / 50%), and two reference horizontal lines for recursive + semantic.
- 1-paragraph takeaway: which variant closes the multi-hop gap (if any); if none does, that itself is a publishable negative finding (the open-recall gap is fundamental, not a chunker-config issue).

## Out of scope (intentional)

- Re-running the full 16-config M2 grid with new chunkers — no signal-to-cost ratio improvement.
- Sweeping other embedders — we fix the SOTA from M2 because the ask is to close the bottleneck, not re-validate embedder choice.
- Hierarchical/parent-child chunking — too much engineering for 6 days.

## Risks

- **Gold-span resolver compatibility (HIGH):** `gold_resolver.py` maps gold character spans to chunk IDs assuming MarkdownChunker's chunk schema. If `DocumentChunker` chunks don't carry `start_char`/`end_char`, chunk-level metrics will silently NaN. Mitigation: smoke-test on 5 papers before kicking the full grid.
- **Semantic chunker dependency (MEDIUM):** `semantic_chunking()` may require a sentence-embedding model loaded at chunk time. Check `chunking.py` for HF deps before launch.
- **Multi-hop sample is small (LOW):** only ~5 multi-hop questions in the 93-pair gold — single-config noise may dominate. Report ranges, not point estimates.
