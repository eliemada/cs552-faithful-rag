#!/usr/bin/env bash
# Local runner for the M3 chunker ablation.
#
# Replaces rcp_support/submit_chunker_ablation.sh when RCP queue is
# congested. Designed to be fired and walked away from. Apple Silicon
# (MPS) is auto-detected; CUDA also works.
#
# TIMING (measured on M-series MPS, batch=64, ~22 passages/sec):
#
#   variant         chunks    ETA
#   s200_o0       199,891    ~2h30
#   s200_o100     199,891    ~2h30
#   s400_o0       150,548    ~1h55
#   s400_o200     150,548    ~1h55
#   s600_o0       120,589    ~1h30
#   s600_o300     120,589    ~1h30
#   s800_o0        99,119    ~1h15
#   s800_o400      99,119    ~1h15
#   recursive_400 295,328    ~3h45
#   ──────────────────────────────
#   FULL GRID                ~18h on MPS / ~30min on A100
#
# Plus ~5min/config for the eval step (88 queries × HTTPS rerank).
#
# A trimmed grid skipping the high-overlap duplicates fits in ~10h:
#
#   VARIANTS="s200_o0 s400_o0 s600_o0 s800_o0 recursive_400" \
#       ./scripts/run_chunker_ablation_local.sh
#
# Skips work that's already done so the script is safe to interrupt
# and rerun: per-variant index builds and per-config eval runs are
# individually idempotent. State lives in:
#
#   data/s3_archive/indexes/e5_large_<variant>.faiss[+metadata.json]
#   evaluation/retrieval_eval/results/e5_rerank_<variant>.json
#
# Progress is logged to logs/chunker_ablation_<timestamp>.log alongside
# the live terminal output.
#
# Run (full grid, ~18h on MPS)::
#
#     ./scripts/run_chunker_ablation_local.sh
#
# Override the device or variant subset::
#
#     DEVICE=cpu ./scripts/run_chunker_ablation_local.sh
#     VARIANTS="s400_o0 recursive_400" ./scripts/run_chunker_ablation_local.sh
#
# To resume after Ctrl-C, just rerun — the script skips finished work.

set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

# Variants to run. Override via env: VARIANTS="s400_o0 recursive_400" ./...
DEFAULT_VARIANTS="s200_o0 s200_o100 s400_o0 s400_o200 s600_o0 s600_o300 s800_o0 s800_o400 recursive_400"
read -r -a VARIANTS <<<"${VARIANTS:-${DEFAULT_VARIANTS}}"

# ---- env ----------------------------------------------------------------------

if [[ ! -f .env ]]; then
    echo "ERROR: .env missing. Need ZEROENTROPY_API_KEY for the reranker." >&2
    exit 1
fi

# shellcheck disable=SC2046
export $(grep -E '^(ZEROENTROPY|OPENAI|OPENROUTER)_API_KEY=' .env | xargs -I {} echo {})

if [[ -z "${ZEROENTROPY_API_KEY:-}" ]]; then
    echo "ERROR: ZEROENTROPY_API_KEY missing from .env. The reranker step needs it." >&2
    exit 1
fi

# Auto-detect device unless overridden.
DEVICE="${DEVICE:-}"
if [[ -z "${DEVICE}" ]]; then
    DEVICE="$(uv run python - <<'PY'
import torch
if torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
)"
fi
echo "Device: ${DEVICE}"

# ---- log ----------------------------------------------------------------------

mkdir -p logs
LOG="logs/chunker_ablation_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG}"
exec > >(tee -a "${LOG}") 2>&1

# ---- prereqs ------------------------------------------------------------------

echo
echo "==> Step 0: sanity checks"
n_papers=$(ls data/s3_archive/processed 2>/dev/null | wc -l | tr -d ' ')
n_coarse=$(ls data/s3_archive/chunks 2>/dev/null | grep -c '_coarse\.json$' || true)
n_variant_first=$(ls data/s3_archive/chunks 2>/dev/null | grep -c "_${VARIANTS[0]}\.json$" || true)
echo "  papers under processed/        : ${n_papers}"
echo "  *_coarse.json (M2 haystack)     : ${n_coarse}"
echo "  *_${VARIANTS[0]}.json (variant 0): ${n_variant_first}"

if [[ "${n_papers}" -lt 100 ]]; then
    echo "ERROR: only ${n_papers} papers under processed/. Expected ~1410. Pull from HF first:" >&2
    echo "  uv run python -c \"from huggingface_hub import snapshot_download; snapshot_download('citeright/corpus', repo_type='dataset', local_dir='data/s3_archive', allow_patterns=['processed/**'])\"" >&2
    exit 1
fi

if [[ "${n_variant_first}" -lt 100 ]]; then
    echo "==> regenerating chunk variants from processed/"
    uv run python -m scripts.generate_chunk_variants \
        --processed-dir data/s3_archive/processed \
        --out-dir data/s3_archive/chunks \
        --workers 8
fi

# ---- index builds (one per variant) ------------------------------------------

echo
echo "==> Step 1: build 9 E5-large variant indexes (skips already-built)"

build_one_index() {
    local v="$1"
    local out="data/s3_archive/indexes/e5_large_${v}.faiss"
    local meta="data/s3_archive/indexes/e5_large_${v}_metadata.json"
    if [[ -f "${out}" && -f "${meta}" ]]; then
        echo "  skip: ${out} already exists"
        return 0
    fi
    echo "  building e5_large_${v} on ${DEVICE} (batch=${BATCH_SIZE:-64}) ..."
    uv run python -m scripts.build_hf_index \
        --model e5-large \
        --chunk-type "${v}" \
        --restrict-to-papers-with coarse \
        --batch-size "${BATCH_SIZE:-64}" \
        --device "${DEVICE}"
}

for v in "${VARIANTS[@]}"; do
    build_one_index "${v}"
done

# ---- evals (one per config) --------------------------------------------------

echo
echo "==> Step 2: evaluate 9 e5_rerank_* configs (skips already-evaluated)"

eval_one_config() {
    local v="$1"
    local cfg="e5_rerank_${v}"
    local out="evaluation/retrieval_eval/results/${cfg}.json"
    if [[ -f "${out}" ]]; then
        echo "  skip: ${out} already exists"
        return 0
    fi
    echo "  evaluating ${cfg} ..."
    uv run python -m evaluation.retrieval_eval.evaluate_retrieval \
        --config "${cfg}" \
        --quiet
}

for v in "${VARIANTS[@]}"; do
    eval_one_config "${v}"
done

# ---- comparison table --------------------------------------------------------

echo
echo "==> Step 3: regenerate comparison.md/comparison.json"
uv run python -m scripts.run_retrieval_ablation

echo
echo "==> Done. Headline results:"
grep -E "e5_rerank_|coarse_rerank|fine_rerank" evaluation/retrieval_eval/results/comparison.md | head -20 || true

echo
echo "==> Full log: ${LOG}"
echo "==> See:    evaluation/retrieval_eval/results/comparison.md"
