#!/bin/bash
#SBATCH --job-name=cs552-ragas-sweep
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/ragas-%j.out
#SBATCH --error=slurm_logs/ragas-%j.err

# SCITAS izar SLURM submission for Faruk's M3 RAGAS n=88 scale-up
# (driver: scripts/run_ragas_sweep.py on main).
#
# Runs the M2 SOTA retriever (`e5_large_coarse_rerank`) over the full
# 88-question gold and a long-context arm sharing the same gold answers,
# then scores both with Claude Haiku 4.5 RAGAS.
#
# The workload is API-bound (OpenRouter Gemini/GPT-4o-mini answer LLMs,
# Anthropic Haiku judge, ZeroEntropy reranker over HTTPS). The 1 GPU is
# only used for query-embedding via the e5-large encoder — it's mostly
# idle during the API waits, but keeps the env consistent with the
# existing crag/chunker sbatch pattern.
#
# Submit from any izar1 login node after cloning the repo to
# /scratch/izar/$USER/cs552-faithful-rag and placing .env at the repo
# root (must contain OPENROUTER_API_KEY, OPENAI_API_KEY,
# ZEROENTROPY_API_KEY, and either ANTHROPIC_API_KEY or routing Haiku
# through OpenRouter).
#
# Idempotent: each config writes its own ragas_<config>.json under
# --out-dir; an existing output is skipped (unless --overwrite). Safe
# after preemption / Ctrl-C — rerun and finished configs are reused.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. paths + environment
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/scratch/izar/$USER/cs552-faithful-rag}"
SCRATCH_ROOT="/scratch/izar/$USER"
GIT_REF="${GIT_REF:-main}"
RAGAS_CONFIGS="${RAGAS_CONFIGS:-sota}"
RAGAS_JUDGE="${RAGAS_JUDGE:-anthropic/claude-haiku-4-5}"
RAGAS_OUT_DIR="${RAGAS_OUT_DIR:-evaluation/ragas_eval/results/m3_sweep}"
RAGAS_SAMPLE="${RAGAS_SAMPLE:-0}"  # 0 = full gold
# Optional RAG answer-LLM override. Empty = script default
# (DEFAULT_ANSWER_MODEL_RAG = api:openrouter/openai/gpt-4o-mini).
# To run the Gemini RAG arm: RAGAS_RAG_MODEL=api:openrouter/google/gemini-2.5-flash
RAGAS_RAG_MODEL="${RAGAS_RAG_MODEL:-}"

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
mkdir -p slurm_logs "${RAGAS_OUT_DIR}"

# Pin to the requested ref (default: main). Uses `reset --hard` rather
# than `checkout` so the SCITAS clone re-aligns with remote even if a
# prior rsync left tracked files modified locally — the clone is treated
# as read-mostly and any local changes to tracked files are discarded.
# Untracked files (logs, results, .env) are preserved by reset --hard.
git fetch --quiet origin "${GIT_REF}" || true
git reset --hard --quiet "origin/${GIT_REF}"
echo "[$(date)] HEAD: $(git rev-parse --short HEAD) on ${GIT_REF}"

# ---------------------------------------------------------------------------
# 1. module env — same pattern as the crag / chunker sbatch
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
    export $(grep -E '^(OPENROUTER_API_KEY|OPENAI_API_KEY|ZEROENTROPY_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN)=' .env | xargs -I{} echo {})
fi
for k in OPENROUTER_API_KEY OPENAI_API_KEY ZEROENTROPY_API_KEY; do
    if [[ -z "${!k:-}" ]]; then
        echo "[$(date)] ERROR: ${k} missing — run_ragas_sweep needs it." >&2
        exit 1
    fi
done
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
uv python pin 3.12
# --all-extras pulls in the optional `eval` extra on packages/shared
# (langchain-openai, ragas, etc.) that scripts/run_ragas_sweep needs.
uv sync --all-packages --all-extras
# ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed
# in langchain-community 0.4.x. Fresh resolves grab 0.4.2 (broken); local
# dev .venv has 0.3.31 (works). Pin the downgrade until ragas patches the
# import or we move to a fixed ragas release.
uv pip install "langchain-community<0.4"

# ---------------------------------------------------------------------------
# 4. dataset / index sanity — `sota` preset uses e5_large_coarse_rerank,
#    which needs e5_large_coarse.faiss + corpus markdowns for the LC arm.
# ---------------------------------------------------------------------------
n_papers=$(ls data/s3_archive/processed/ 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date)] papers in processed/: ${n_papers}"
if [[ "${n_papers}" -lt 1000 ]]; then
    echo "[$(date)] ERROR: expected ~1410 papers in data/s3_archive/processed/, got ${n_papers}." >&2
    echo "Pre-stage the corpus (rsync from local clone):" >&2
    echo "  rsync -avhP data/s3_archive/ izar:${REPO_DIR}/data/s3_archive/" >&2
    exit 1
fi

case "${RAGAS_CONFIGS}" in
    sota|*e5_large_coarse_rerank*)
        required_index="data/s3_archive/indexes/e5_large_coarse.faiss"
        ;;
    *e5_rerank_s800_o0*)
        required_index="data/s3_archive/indexes/e5_large_s800_o0.faiss"
        ;;
    *)
        required_index=""
        ;;
esac
if [[ -n "${required_index}" && ! -f "${required_index}" ]]; then
    echo "[$(date)] ERROR: ${required_index} missing." >&2
    echo "Pre-stage from local (index files are large; debug QoS cannot rebuild them in 1h)." >&2
    echo "  rsync -avhP data/s3_archive/indexes/$(basename ${required_index%.*}).* \\" >&2
    echo "        izar:${REPO_DIR}/data/s3_archive/indexes/" >&2
    exit 1
fi

# Optional RAG-answer-model override forwarded to the script.
rag_model_flag=()
if [[ -n "${RAGAS_RAG_MODEL}" ]]; then
    rag_model_flag=(--answer-model-rag "${RAGAS_RAG_MODEL}")
fi

# Sanity-print the plan before paying for any API calls.
echo "[$(date)] ==> dry-run plan (--list-only):"
uv run python -m scripts.run_ragas_sweep \
    --configs "${RAGAS_CONFIGS}" \
    --sample "${RAGAS_SAMPLE}" \
    --judge-model "${RAGAS_JUDGE}" \
    --out-dir "${RAGAS_OUT_DIR}" \
    "${rag_model_flag[@]}" \
    --list-only

# ---------------------------------------------------------------------------
# 5. run the RAGAS sweep — API-bound; expect ~20-60 min wall for n=88
# ---------------------------------------------------------------------------
echo "[$(date)] ==> running RAGAS sweep (configs=${RAGAS_CONFIGS}, sample=${RAGAS_SAMPLE}, judge=${RAGAS_JUDGE}, rag_model=${RAGAS_RAG_MODEL:-default})"
uv run python -m scripts.run_ragas_sweep \
    --configs "${RAGAS_CONFIGS}" \
    --sample "${RAGAS_SAMPLE}" \
    --judge-model "${RAGAS_JUDGE}" \
    --out-dir "${RAGAS_OUT_DIR}" \
    "${rag_model_flag[@]}"

# ---------------------------------------------------------------------------
# 6. report
# ---------------------------------------------------------------------------
echo "[$(date)] ==> outputs:"
ls -lh "${RAGAS_OUT_DIR}" 2>/dev/null || true
echo "[$(date)] done."
