#!/bin/bash
# CS-552 — Run:AI launcher for the ColBERTv2 PLAID-index build.
#
# Builds both granularities end-to-end on a single A100:
#   colbert_coarse, colbert_fine
#
# This is the late-interaction sibling of submit_embed.sh: the dense
# BGE-M3 / E5-large indices are produced by submit_embed.sh; the ColBERT
# PLAID indices live here because they need a separate model, a separate
# index format, and a separate evaluator path (BaseRetriever protocol).
#
# Output: two folders under data/s3_archive/indexes/
#   colbert_coarse/             colbert_coarse_metadata.json
#   colbert_fine/               colbert_fine_metadata.json
# plus the standard chunk metadata sidecars matching the dense layout.

set -euo pipefail

GASPAR="${GASPAR:-gaspar}"
GROUP="${GROUP:-g68}"
GIT_REF="${GIT_REF:-main}"

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: GASPAR=<username> ./rcp_support/submit_colbert.sh" >&2
    exit 1
fi
if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: set GROUP to your team number (e.g. g68)." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-colbert-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
REPO_URL="${REPO_URL:-https://github.com/eliemada/cs552-faithful-rag.git}"
REPO_DIR="/scratch/cs552-faithful-rag"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"

COLBERT_COMMAND=$(cat <<'INNER'
set -euo pipefail
mkdir -p /scratch/hf_cache /scratch/uv_cache
export HF_HOME=/scratch/hf_cache
export UV_CACHE_DIR=/scratch/uv_cache

# Same dubious-ownership escape hatch we use in submit_embed.sh.
git config --global --add safe.directory '*'

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch --all --tags
git checkout "${GIT_REF}"
git pull --ff-only || true

if ! command -v uv >/dev/null 2>&1; then
    pip install --no-cache-dir uv
fi
uv sync --all-packages

# Confirm chunks corpus is present (pull from citeright/corpus if not).
if [[ ! -d data/s3_archive/chunks ]] || [[ -z "$(ls data/s3_archive/chunks 2>/dev/null)" ]]; then
    echo "Chunks corpus missing. Pulling from HF citeright/corpus..."
    uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "citeright/corpus",
    repo_type="dataset",
    local_dir="data/s3_archive",
    allow_patterns=["chunks/*"],
)
PY
fi

nvidia-smi || true

uv run python -m scripts.build_all_colbert_indices --device cuda --batch-size 64

ls -lh data/s3_archive/indexes/ | head -40
INNER
)

echo ">>> Submitting ColBERT build job ${JOB_NAME} (1× ${NODE})"

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
  --environment REPO_URL="${REPO_URL}" \
  --environment REPO_DIR="${REPO_DIR}" \
  --environment GIT_REF="${GIT_REF}" \
  --environment COLBERT_COMMAND="${COLBERT_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --command -- /bin/bash -lc 'ln -sf "$(command -v python3)" /usr/local/bin/python; eval "${COLBERT_COMMAND}"'

cat <<EOF

>>> ColBERT job submitted: ${JOB_NAME}

Watch it start:    runai training describe ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai training logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai training delete ${JOB_NAME} -p ${PROJECT}

When the job finishes, the indices land under
  /scratch/cs552-faithful-rag/data/s3_archive/indexes/
    colbert_coarse/    colbert_coarse_metadata.json
    colbert_fine/      colbert_fine_metadata.json

Push them to the team's HF dataset (citeright/corpus) with the existing
upload launcher; the script's patterns already cover colbert_* paths
once HF_TOKEN is set:

  HF_TOKEN=hf_... GASPAR=${GASPAR} ./rcp_support/submit_upload.sh
EOF
