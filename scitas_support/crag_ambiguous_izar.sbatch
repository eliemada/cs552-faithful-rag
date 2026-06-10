#!/bin/bash
#SBATCH --job-name=cs552-crag-ablation
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/crag-%j.out
#SBATCH --error=slurm_logs/crag-%j.err

# SCITAS izar SLURM submission for Faruk's M3 CRAG threshold-sweep
# (branch: feat/m3-implementation-ragas-crag).
#
# Drives scripts/run_crag_ablation.py — probes each gold question once
# (retrieve + cross-encoder score) and analytically sweeps the τ_high × τ_low
# grid. Probe is GPU-bound (e5-large embedder + ms-marco-MiniLM cross-encoder
# via sentence-transformers); reranker is the ZeroEntropy HTTPS API.
#
# Submit from any izar1 login node after cloning the repo to
# /scratch/izar/$USER/cs552-faithful-rag and placing .env at the repo root
# (must contain ZEROENTROPY_API_KEY=...; HF_TOKEN= optional but recommended
# to avoid 429s on the cluster IP).
#
# Idempotent: probe cache and outputs are written under evaluation/crag/results.
# Safe after preemption / Ctrl-C — rerun and the script picks up the cache.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. paths + environment
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/scratch/izar/$USER/cs552-faithful-rag}"
SCRATCH_ROOT="/scratch/izar/$USER"
GIT_REF="${GIT_REF:-feat/m3-implementation-ragas-crag}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-e5_large_coarse_rerank}"
RETRIEVAL_K="${RETRIEVAL_K:-10}"

export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv_cache"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${REPO_DIR%/*}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[$(date)] ERROR: repo missing at ${REPO_DIR}." >&2
    echo "Bootstrap first:" >&2
    echo "  git clone git@github.com:eliemada/cs552-faithful-rag.git ${REPO_DIR}" >&2
    echo "  cd ${REPO_DIR} && cp ~/.env-cs552 .env  # or scp it up" >&2
    exit 1
fi

cd "${REPO_DIR}"
mkdir -p slurm_logs evaluation/crag/results

# Make sure we're on Faruk's branch (fetch + checkout). Won't clobber local
# work — the SCITAS clone is read-mostly.
git fetch --quiet origin "${GIT_REF}" || true
git checkout --quiet "${GIT_REF}" 2>/dev/null || git checkout --quiet "origin/${GIT_REF}"
echo "[$(date)] HEAD: $(git rev-parse --short HEAD) on ${GIT_REF}"

# ---------------------------------------------------------------------------
# 1. module env — CUDA 12.1 matches torch's bundled runtime closely. uv pins
#    its own Python, so we don't load a python module.
# ---------------------------------------------------------------------------
module purge
module load gcc/11.3.0
module load cuda/12.1.1

echo "[$(date)] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
nvidia-smi || true

# ---------------------------------------------------------------------------
# 2. secrets — read from repo-root .env
# ---------------------------------------------------------------------------
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -E '^(ZEROENTROPY_API_KEY|HF_TOKEN|OPENAI_API_KEY|OPENROUTER_API_KEY)=' .env | xargs -I{} echo {})
fi
if [[ -z "${ZEROENTROPY_API_KEY:-}" ]]; then
    echo "[$(date)] ERROR: ZEROENTROPY_API_KEY missing — the ${RETRIEVER_CONFIG} adapter calls the reranker over HTTPS." >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# 3. python env via uv (cp310/311/312/313 wheels for fast-plaid require 3.12)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python pin 3.12
uv sync --all-packages

# ---------------------------------------------------------------------------
# 4. dataset / index sanity (script needs e5_large_coarse.faiss for the SOTA
#    retriever config; pre-stage via rsync from local if missing).
# ---------------------------------------------------------------------------
n_papers=$(ls data/s3_archive/processed/ 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] papers in processed/: ${n_papers}"
if [[ "${n_papers}" -lt 1000 ]]; then
    echo "[$(date)] ERROR: expected ~1410 papers in data/s3_archive/processed/, got ${n_papers}." >&2
    echo "Pre-stage the corpus (rsync from local clone):" >&2
    echo "  rsync -avhP data/s3_archive/ izar:${REPO_DIR}/data/s3_archive/" >&2
    exit 1
fi

required_index="data/s3_archive/indexes/e5_large_coarse.faiss"
if [[ "${RETRIEVER_CONFIG}" == "e5_large_coarse_rerank" && ! -f "${required_index}" ]]; then
    echo "[$(date)] WARN: ${required_index} missing — building it now."
    uv run python -m scripts.build_hf_index \
        --model e5-large \
        --chunk-type coarse \
        --batch-size 256 \
        --device cuda
fi

# ---------------------------------------------------------------------------
# 5. run the CRAG threshold ablation (~10 min on V100 for ~93 gold questions)
# ---------------------------------------------------------------------------
OUTPUT_JSON="evaluation/crag/results/threshold_sweep_${SLURM_JOB_ID:-local}.json"
PROBE_CACHE="evaluation/crag/results/probe_${RETRIEVER_CONFIG}.json"

echo "[$(date)] ==> running CRAG ablation (retriever=${RETRIEVER_CONFIG}, k=${RETRIEVAL_K})"
uv run python -m scripts.run_crag_ablation \
    --retriever-config "${RETRIEVER_CONFIG}" \
    --retrieval-k "${RETRIEVAL_K}" \
    --probe-cache "${PROBE_CACHE}" \
    --output "${OUTPUT_JSON}"

# ---------------------------------------------------------------------------
# 6. report
# ---------------------------------------------------------------------------
echo "[$(date)] ==> outputs:"
ls -lh "${OUTPUT_JSON}" "${OUTPUT_JSON%.json}.md" "${PROBE_CACHE}" 2>/dev/null || true
echo "[$(date)] done."
