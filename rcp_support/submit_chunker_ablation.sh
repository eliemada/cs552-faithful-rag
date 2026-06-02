#!/bin/bash
# CS-552 — Run:AI launcher for the M3 chunker-ablation experiment.
#
# Builds 9 new E5-large FAISS indices, one per chunker variant (s200_o0,
# s200_o100, s400_o0, s400_o200, s600_o0, s600_o300, s800_o0, s800_o400,
# recursive_400), then runs the ablation evaluator over the 93-question
# gold benchmark.
#
# Anchored on the M2 SOTA config (e5_large + ZeroEntropy reranker); only
# the chunker is varied. See evaluation/retrieval_eval/CHUNKER_ABLATION_PLAN.md
# for the full plan, decision rationale, and report integration notes.
#
# The variant chunk JSONs are generated inside the pod (CPU only, ~6 s
# for 1410 papers × 9 variants), so this script does NOT depend on
# pre-uploaded chunks beyond the original coarse/fine set that the embed
# job already produced.

set -euo pipefail

# ============== EDIT OR OVERRIDE VIA ENV ==============
GASPAR="${GASPAR:-gaspar}"
GROUP="${GROUP:-g68}"
GIT_REF="${GIT_REF:-main}"
# ======================================================

# The eval step calls the ZeroEntropy reranker over HTTPS, so the API key
# must travel into the pod. We read it from a local .env (preferred) or
# from the launcher's own environment as a fallback. The key is never
# committed.
if [[ -z "${ZEROENTROPY_API_KEY:-}" && -f .env ]]; then
    ZEROENTROPY_API_KEY="$(grep -E '^ZEROENTROPY_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
    export ZEROENTROPY_API_KEY
fi
if [[ -z "${ZEROENTROPY_API_KEY:-}" ]]; then
    echo "ERROR: ZEROENTROPY_API_KEY is not set and not present in .env. The reranker step needs it." >&2
    echo "Either:  export ZEROENTROPY_API_KEY=ze_..." >&2
    echo "Or add it to .env:  ZEROENTROPY_API_KEY=ze_..." >&2
    exit 1
fi

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_chunker_ablation.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi
if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_chunker_ablation.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-chunker-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"
REPO_URL="${REPO_URL:-https://github.com/eliemada/cs552-faithful-rag.git}"
REPO_DIR="/scratch/cs552-faithful-rag"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"

CHUNKER_COMMAND=$(cat <<'INNER'
set -euo pipefail
mkdir -p /scratch/hf_cache /scratch/uv_cache
export HF_HOME=/scratch/hf_cache
export UV_CACHE_DIR=/scratch/uv_cache

git config --global --add safe.directory '*'

# 1. fetch or update the repo
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git fetch --all --tags
git checkout "${GIT_REF}"
git pull --ff-only || true

# 2. set up uv
if ! command -v uv >/dev/null 2>&1; then
    pip install --no-cache-dir uv
fi
uv sync --all-packages

# 3. ensure base corpus (processed + existing coarse/fine chunks) is on /scratch.
#    Always run snapshot_download to add any newly-uploaded papers (idempotent;
#    HF skips files we already have locally). The previous run pulled only 52
#    papers because allow_patterns=["processed/*", ...] is a single-level glob
#    that misses processed/<paper_id>/document.md (two levels deep). We now use
#    "**" doublestar globs so all nested files are eligible.
echo "Syncing corpus from HF (recursive globs)..."
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "citeright/corpus",
    repo_type="dataset",
    local_dir="data/s3_archive",
    allow_patterns=["processed/**", "chunks/**", "indexes/**"],
)
PY
echo "Papers in processed/: $(ls data/s3_archive/processed/ 2>/dev/null | wc -l)"

# 3b. delete the previous (52-paper-haystack) variant chunks + indexes so the
#     next steps regenerate them from the now-larger processed/ set. Without
#     this, generate_chunk_variants --skip-existing and the build loop would
#     keep reusing the broken artefacts.
rm -f data/s3_archive/chunks/*_s[0-9]*_o[0-9]*.json
rm -f data/s3_archive/chunks/*_recursive_*.json
rm -f data/s3_archive/indexes/e5_large_s[0-9]*_o[0-9]*.faiss
rm -f data/s3_archive/indexes/e5_large_s[0-9]*_o[0-9]*_metadata.json
rm -f data/s3_archive/indexes/e5_large_recursive_*.faiss
rm -f data/s3_archive/indexes/e5_large_recursive_*_metadata.json
echo "Cleared previous variant artefacts; remaining indexes:"
ls data/s3_archive/indexes/ | head -20

# 4. generate the 9 chunker variants from processed/<paper>/document.md
#    This is CPU-only and takes ~6 seconds for the full corpus.
uv run python -m scripts.generate_chunk_variants \
    --processed-dir data/s3_archive/processed \
    --out-dir data/s3_archive/chunks \
    --workers 8

# 5. nvidia-smi sanity
nvidia-smi || true

# 6. build one E5-large index per variant. --restrict-to-papers-with coarse
#    keeps the haystack identical to the M2 SOTA baseline for fair
#    comparison.
VARIANTS=(s200_o0 s200_o100 s400_o0 s400_o200 s600_o0 s600_o300 s800_o0 s800_o400 recursive_400)
for v in "${VARIANTS[@]}"; do
    out="data/s3_archive/indexes/e5_large_${v}.faiss"
    if [[ -f "${out}" ]]; then
        echo "skip: ${out} already exists"
        continue
    fi
    echo "==> building e5_large_${v}"
    uv run python -m scripts.build_hf_index \
        --model e5-large \
        --chunk-type "${v}" \
        --restrict-to-papers-with coarse \
        --batch-size 256 \
        --device cuda
done

# 6b. drop any stale per-config eval JSONs from the previous (broken) run so
#     --run-missing actually re-evaluates them against the freshly-built
#     indexes. The committed M2 baselines (16 configs) are left alone.
rm -f evaluation/retrieval_eval/results/e5_rerank_*.json
rm -f evaluation/retrieval_eval/results/comparison.json
rm -f evaluation/retrieval_eval/results/comparison.md

# 7. run the ablation evaluator on the new configs.
#    --run-missing runs every config in CONFIGS that lacks a JSON, which is
#    exactly the 9 new e5_rerank_* configs (the 16 M2 configs already have
#    result files in evaluation/retrieval_eval/results/).
uv run python -m scripts.run_retrieval_ablation --run-missing

# 8. show what we produced
echo "==> indices:"
ls -lh data/s3_archive/indexes/ | grep e5_large_
echo "==> results:"
ls -lh evaluation/retrieval_eval/results/ | grep e5_rerank_
INNER
)

echo ">>> Submitting chunker-ablation job ${JOB_NAME} (1× ${NODE})"

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
  --environment CHUNKER_COMMAND="${CHUNKER_COMMAND}" \
  --environment ZEROENTROPY_API_KEY="${ZEROENTROPY_API_KEY}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --command -- /bin/bash -lc 'ln -sf "$(command -v python3)" /usr/local/bin/python; eval "${CHUNKER_COMMAND}"'

cat <<EOF

>>> Chunker-ablation job submitted: ${JOB_NAME}

Watch it start:    runai training describe ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai training logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai training delete ${JOB_NAME} -p ${PROJECT}

Outputs land under /scratch/cs552-faithful-rag/:
  data/s3_archive/indexes/e5_large_{s200_o0,...,recursive_400}.faiss   (9 new FAISS indices)
  evaluation/retrieval_eval/results/e5_rerank_{s200_o0,...,recursive_400}.json   (9 new eval files)
  evaluation/retrieval_eval/results/comparison.{json,md}                  (refreshed with new rows)

Pull results locally to slot into the M3 report:
  runai training bash ${JOB_NAME} -p ${PROJECT}
  # then from inside: tar + scp the results/ dir
EOF
