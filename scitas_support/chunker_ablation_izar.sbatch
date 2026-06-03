#!/bin/bash
#SBATCH --job-name=cs552-chunker-ablation
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/chunker-%j.out
#SBATCH --error=slurm_logs/chunker-%j.err

# SCITAS izar SLURM submission for the M3 chunker-ablation experiment.
# Mirrors rcp_support/submit_chunker_ablation.sh logic but adapts it
# to SCITAS conventions: module load, /scratch/izar/$USER as work root,
# uv for Python toolchain, V100 GPU.
#
# Submit from any login node (izar1) after cloning the repo to
# /scratch/izar/$USER/cs552-faithful-rag and placing .env at the repo
# root (must contain ZEROENTROPY_API_KEY=...).
#
# Idempotent: re-running skips finished indices and re-uses cached
# chunk JSONs and FAISS files. Safe after preemption / Ctrl-C.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. paths + environment
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/scratch/izar/$USER/cs552-faithful-rag}"
SCRATCH_ROOT="/scratch/izar/$USER"

export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv_cache"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${REPO_DIR%/*}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[$(date)] ERROR: repo missing at ${REPO_DIR}." >&2
    echo "Run the bootstrap first (clone + .env)." >&2
    exit 1
fi

cd "${REPO_DIR}"
mkdir -p slurm_logs

# ---------------------------------------------------------------------------
# 1. module env — CUDA 12.1 matches torch's bundled runtime closely; gcc
#    is for any C-extension builds uv may trigger. We deliberately do NOT
#    load a python module — uv installs and pins its own interpreter.
# ---------------------------------------------------------------------------
module purge
module load gcc/11.3.0
module load cuda/12.1.1

echo "[$(date)] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 2. secrets — read from repo-root .env (committed-out via .gitignore)
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -E '^(ZEROENTROPY_API_KEY|HF_TOKEN)=' .env | xargs -I{} echo {})
fi
if [[ -z "${ZEROENTROPY_API_KEY:-}" ]]; then
    echo "[$(date)] ERROR: ZEROENTROPY_API_KEY missing (.env or env var)." >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# 3. python env via uv
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Pin Python to 3.12 — fast-plaid (1.3.0.290) ships wheels only for cp310/311/312/313;
# uv otherwise grabs the latest CPython (3.14) which has no compatible wheel.
uv python pin 3.12
uv sync --all-packages

# ---------------------------------------------------------------------------
# 4. corpus check (pre-staged via rsync from local — skip HF to dodge 429)
# ---------------------------------------------------------------------------
n_papers=$(ls data/s3_archive/processed/ 2>/dev/null | wc -l | tr -d ' ')
n_coarse=$(ls data/s3_archive/chunks/*_coarse.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] papers in processed/: ${n_papers}, coarse chunks: ${n_coarse}"
if [[ "${n_papers}" -lt 1000 ]]; then
    echo "[$(date)] ERROR: expected ~1410 papers, got ${n_papers}. Pre-stage with rsync from local repo." >&2
    exit 1
fi

# Drop any stale (52-paper-haystack) variant artefacts so we regen from the
# now-larger processed/ set. The M2 coarse/fine indexes are kept.
rm -f data/s3_archive/chunks/*_s[0-9]*_o[0-9]*.json
rm -f data/s3_archive/chunks/*_recursive_*.json
rm -f data/s3_archive/indexes/e5_large_s[0-9]*_o[0-9]*.faiss
rm -f data/s3_archive/indexes/e5_large_s[0-9]*_o[0-9]*_metadata.json
rm -f data/s3_archive/indexes/e5_large_recursive_*.faiss
rm -f data/s3_archive/indexes/e5_large_recursive_*_metadata.json

# ---------------------------------------------------------------------------
# 5. generate chunk variants (CPU, ~6s for the corpus)
# ---------------------------------------------------------------------------
uv run python -m scripts.generate_chunk_variants \
    --processed-dir data/s3_archive/processed \
    --out-dir data/s3_archive/chunks \
    --workers 8

# ---------------------------------------------------------------------------
# 6. build one E5-large FAISS index per variant (GPU)
# ---------------------------------------------------------------------------
VARIANTS=(s200_o0 s200_o100 s400_o0 s400_o200 s600_o0 s600_o300 s800_o0 s800_o400 recursive_400)
for v in "${VARIANTS[@]}"; do
    out="data/s3_archive/indexes/e5_large_${v}.faiss"
    if [[ -f "${out}" ]]; then
        echo "[$(date)] skip: ${out} already exists"
        continue
    fi
    echo "[$(date)] ==> building e5_large_${v}"
    uv run python -m scripts.build_hf_index \
        --model e5-large \
        --chunk-type "${v}" \
        --restrict-to-papers-with coarse \
        --batch-size 256 \
        --device cuda
done

# ---------------------------------------------------------------------------
# 7. evaluate (calls ZeroEntropy reranker over HTTPS)
# ---------------------------------------------------------------------------
rm -f evaluation/retrieval_eval/results/e5_rerank_*.json
rm -f evaluation/retrieval_eval/results/comparison.json
rm -f evaluation/retrieval_eval/results/comparison.md

uv run python -m scripts.run_retrieval_ablation --run-missing

# ---------------------------------------------------------------------------
# 8. report
# ---------------------------------------------------------------------------
echo "[$(date)] ==> indices:"
ls -lh data/s3_archive/indexes/ | grep e5_large_ || true
echo "[$(date)] ==> results:"
ls -lh evaluation/retrieval_eval/results/ | grep e5_rerank_ || true
echo "[$(date)] done."
