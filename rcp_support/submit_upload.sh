#!/bin/bash
# CS-552 — Run:AI launcher that pushes the freshly-built alt-embedder
# FAISS indices to the citeright/corpus HF dataset.
#
# Requires HF_TOKEN with write access to citeright/corpus. Read it from
# the environment of whoever runs the submitter; do NOT bake it into
# this file.
#
# This is a tiny job (~1.9 GB upload), not GPU-bound. We still request 1
# GPU because the course quota is per-allocation and the small job will
# finish in a couple of minutes.

set -euo pipefail

GASPAR="${GASPAR:-gaspar}"
GROUP="${GROUP:-g68}"
# Tracks the latest branch that ships the upload script's default patterns
# (dense + ColBERT). After ``feat/colbert-integration`` lands, bump this to
# ``main``.
GIT_REF="${GIT_REF:-feat/colbert-integration}"

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: GASPAR=<username> ./rcp_support/submit_upload.sh" >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN env var must be set (write scope on citeright/corpus)." >&2
    echo "  HF_TOKEN=hf_... GASPAR=${GASPAR} ./rcp_support/submit_upload.sh" >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-upload-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
REPO_URL="${REPO_URL:-https://github.com/eliemada/cs552-faithful-rag.git}"
REPO_DIR="/scratch/cs552-faithful-rag"
SCRATCH_PVC="course-cs-552-scratch-${GROUP}"

UPLOAD_COMMAND=$(cat <<'INNER'
set -euo pipefail
git config --global --add safe.directory '*'
cd "${REPO_DIR}"
git fetch --all --tags
git checkout "${GIT_REF}"
git pull --ff-only || true
uv sync --all-packages
uv run python -m scripts.push_indexes_to_hf
INNER
)

echo ">>> Submitting upload job ${JOB_NAME}"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment UV_CACHE_DIR=/scratch/uv_cache \
  --environment HF_TOKEN="${HF_TOKEN}" \
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_DIR="${REPO_DIR}" \
  --environment GIT_REF="${GIT_REF}" \
  --environment UPLOAD_COMMAND="${UPLOAD_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --command -- /bin/bash -lc 'ln -sf "$(command -v python3)" /usr/local/bin/python; eval "${UPLOAD_COMMAND}"'

cat <<EOF

>>> Upload job submitted: ${JOB_NAME}

Stream logs:       runai training logs -f ${JOB_NAME} -p ${PROJECT}
Stop:              runai training delete ${JOB_NAME} -p ${PROJECT}

Once the upload finishes (~5 min), pull the indices locally with:

    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'citeright/corpus',
    repo_type='dataset',
    local_dir='data/s3_archive',
    allow_patterns=[
        'indexes/bge_m3_*',
        'indexes/e5_large_*',
        'indexes/colbert_*',
    ],
)"
EOF
