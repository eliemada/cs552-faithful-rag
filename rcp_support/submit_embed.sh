#!/bin/bash
# CS-552 — Run:AI launcher for the BGE-M3 + E5-large embed run.
#
# Builds four FAISS indices end-to-end on a single A100:
#   bge_m3_coarse, bge_m3_fine, e5_large_coarse, e5_large_fine
#
# This is a training-style (non-interactive) job. It clones the repo into
# /scratch, runs the embed pipeline, and exits when all four indices land
# in data/s3_archive/indexes/. The script is idempotent: if a previous
# run wrote one of the indices, the orchestrator skips it.
#
# After the job finishes, the indices live under
# /scratch/<repo>/data/s3_archive/indexes/. Pull them locally with rsync
# over runai bash, or push them to the citeright/corpus HF dataset.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="gaspar"              # <-- YOUR GASPAR EPFL username.
GROUP="gXX"                  # <-- YOUR TEAM, e.g. g07.
GIT_REF="feat/elie-marimo-notebook"  # branch/tag with the embedder code
# ==============================================

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_embed.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi
if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_embed.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-embed-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
REPO_URL="${REPO_URL:-https://github.com/eliemada/cs552-faithful-rag.git}"
REPO_DIR="/scratch/cs552-faithful-rag"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"

# The full command run inside the pod. Quoted as a heredoc-ish single-arg
# bash -lc so /scratch state survives between steps.
EMBED_COMMAND=$(cat <<'INNER'
set -euo pipefail
mkdir -p /scratch/hf_cache /scratch/uv_cache
export HF_HOME=/scratch/hf_cache
export UV_CACHE_DIR=/scratch/uv_cache

# 1. fetch or update the repo
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch --all --tags
git checkout "${GIT_REF}"
git pull --ff-only || true

# 2. set up uv (the course image may or may not have it)
if ! command -v uv >/dev/null 2>&1; then
    pip install --no-cache-dir uv
fi
uv sync --all-packages

# 3. confirm chunks corpus is present (it's gitignored — pull from HF if absent)
if [[ ! -d data/s3_archive/chunks ]] || [[ -z "$(ls data/s3_archive/chunks 2>/dev/null)" ]]; then
    echo "Chunks corpus missing under data/s3_archive/chunks/. Downloading from HF..."
    uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "citeright/corpus",
    repo_type="dataset",
    local_dir="data/s3_archive",
    allow_patterns=["chunks/*", "indexes/*"],
)
PY
fi

# 4. nvidia-smi sanity
nvidia-smi || true

# 5. build the four indices
uv run python -m scripts.build_all_hf_indices --device cuda --batch-size 256

# 6. show what we produced
ls -lh data/s3_archive/indexes/
INNER
)

echo ">>> Submitting embed job ${JOB_NAME} (1× ${NODE})"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment UV_CACHE_DIR=/scratch/uv_cache \
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_DIR="${REPO_DIR}" \
  --environment GIT_REF="${GIT_REF}" \
  --environment EMBED_COMMAND="${EMBED_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --command -- /bin/bash -lc 'ln -sf "$(command -v python3)" /usr/local/bin/python; eval "${EMBED_COMMAND}"'

cat <<EOF

>>> Embed job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

When the job finishes, the four indices land in:
  /scratch/cs552-faithful-rag/data/s3_archive/indexes/
    bge_m3_coarse.faiss        bge_m3_coarse_metadata.json
    bge_m3_fine.faiss          bge_m3_fine_metadata.json
    e5_large_coarse.faiss      e5_large_coarse_metadata.json
    e5_large_fine.faiss        e5_large_fine_metadata.json

Pull them locally (~1.9 GB total) so the evaluator can score them:
  runai bash ${JOB_NAME} -p ${PROJECT}
  # then from inside, scp / curl / push to citeright/corpus HF dataset
EOF
