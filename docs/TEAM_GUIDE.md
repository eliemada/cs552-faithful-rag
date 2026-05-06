# CiteRight — Teammate Guide

Operational handbook for the four CS-552 contributors (Elie, Andrea, Faruk, Yusif).
The [CS552_README.md](../CS552_README.md) covers *what* the project is; this doc
covers *how* to actually work on it day-to-day.

> **Project state:** the heavy ingestion pipeline (OpenAlex → Dolphin → embeddings)
> already ran. Results are frozen on HuggingFace at
> [`citeright/corpus`](https://huggingface.co/datasets/citeright/corpus) — 3 GB,
> 71 762 files. The original AWS S3 bucket is being retired; do not depend on it.

---

## Table of contents

- [1. One-time laptop setup](#1-one-time-laptop-setup)
- [2. Launch your RCP pod](#2-launch-your-rcp-pod)
- [3. Connect to the pod](#3-connect-to-the-pod)
- [4. Filling in your notebook](#4-filling-in-your-notebook)
- [5. Where the data lives](#5-where-the-data-lives)
- [6. Generation: local vs API models](#6-generation-local-vs-api-models)
- [7. Gold dataset (everyone contributes)](#7-gold-dataset-everyone-contributes)
- [8. Committing changes](#8-committing-changes)
- [9. GPU etiquette](#9-gpu-etiquette)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. One-time laptop setup

### EPFL VPN
Cisco Secure Client → `vpn.epfl.ch` → GASPAR creds. Required for any RCP command.
If you also use Tailscale, **disable "Use Tailscale DNS"** while running `runai`
— Tailscale's MagicDNS hijacks the lookup for `rcpepfl.run.ai`.

### Run:AI CLI
1. Browser → https://rcpepfl.run.ai/ → SSO with EPFL.
2. Top-right `?` icon → **Researcher CLI** → macOS / Linux → download.
3. ```bash
   chmod +x ./runai && sudo mv ./runai /usr/local/bin/runai
   runai version
   ```

If you get **AADSTS50105** at SSO, your account is not yet provisioned —
email `nlp-cs552-spring2026-ta-team@groupes.epfl.ch` with your sciper and
gaspar. Team CiteRight is **group g68**.

### Kubeconfig + first login
```bash
mkdir -p ~/.kube
curl -o ~/.kube/config https://wiki.rcp.epfl.ch/public/files/kube-config.yaml
runai config cluster rcp-caas-prod
runai login
runai project set course-cs-552-<your-gaspar>
```

### HuggingFace token
Needed once, only if you want to upload artifacts (datasets, model checkpoints).
Read-only access (e.g. downloading the corpus) does not require auth.

```bash
hf auth login   # paste a token from https://huggingface.co/settings/tokens
```

---

## 2. Launch your RCP pod

The launcher script lives at [`notebooks/submit.sh`](../notebooks/submit.sh).
**Do not edit it permanently** — `GROUP="g68"` is correct for everyone, and
`GASPAR` is overridable via env var so you don't have to touch the file:

```bash
cd notebooks
GASPAR=<your-gaspar> ./submit.sh
# or with a custom suffix:
GASPAR=<your-gaspar> ./submit.sh exp1
```

The job name will look like `cs552-<gaspar>-g68-lab-<HHMMSS>`. Watch it boot:

```bash
runai workspace describe <job-name>
# wait for Phase: Running (~30 s – 3 min)
```

Each job gets:
- 1 × A100 40 GB GPU (course cap)
- 3 PVCs mounted: `/scratch` (group), `/shared-ro` (course datasets/models),
  `/shared-rw` (campus-wide writable)
- Course image `registry.rcp.epfl.ch/course-cs-552/base-vllm:v1` with PyTorch
  2.8.0+cu128, vLLM 0.11.0, transformers 4.57.0, sentence-transformers 5.4.1,
  faiss-cpu, JupyterLab — all pre-installed at the **system** Python, no venv.

---

## 3. Connect to the pod

Three options, all to the same pod simultaneously.

### Jupyter (the deliverable path — graders use this)
```bash
runai port-forward <job-name> --port 8888:8888
# new browser tab → http://localhost:8888 (token: cs552)
```

### Shell
```bash
runai workspace bash <job-name>
# inside the pod, drops you in /scratch
```

### VS Code attached
Install Microsoft's **Kubernetes** + **Remote Development** extensions, attach
to pod `<job-name>-0-0`, open folder `/scratch/citeright`. Best for editing.

---

## 4. Filling in your notebook

Each of you owns one file under `notebooks/`:

| Member  | File                                       | Contribution                       |
|---------|--------------------------------------------|------------------------------------|
| Elie    | `elie_bruno_355932.ipynb`                  | Retrieval ablation                 |
| Andrea  | `andrea_trugenberger_357615.ipynb`         | Citation faithfulness (NLI)        |
| Faruk   | `faruk_zahiragic_415360.ipynb`             | Corrective RAG + threshold sweep   |
| Yusif   | `yusif_askari_413862.ipynb`                | RAGAS + 128 K long-context         |

### First-time setup inside the pod
```bash
git config --global --add safe.directory /scratch/citeright   # one-off
cd /scratch
[ -d citeright ] || git clone https://github.com/eliemada/cs552-faithful-rag.git citeright
cd citeright && git pull
```

### The bootstrap cell — don't modify
Every notebook starts with the same bootstrap cell (`%pip install …` + corpus
download + smoke test). It is **generated** by
[`scripts/dev/generate_notebook_scaffolds.py`](../scripts/dev/generate_notebook_scaffolds.py).
If a teammate finds an issue with the bootstrap, fix the generator and rerun
it — never edit the four notebooks one-by-one:

```bash
uv run python scripts/dev/generate_notebook_scaffolds.py
```

That regenerates all four files consistently.

### Where to write *your* code
Replace the `raise NotImplementedError` placeholders inside each `## ` section
of your notebook. Keep cells small (2–10 lines each); push reusable logic into
the right module under `evaluation/`:

```
evaluation/
├── common/             # shared by everyone — extend, don't duplicate
│   ├── data_loader.py  # corpus loaders
│   ├── models.py       # generate(), load_embedder(), load_nli_classifier()
│   └── paths.py
├── retrieval_eval/     # Elie
├── faithfulness/       # Andrea
├── crag/               # Faruk
├── ragas_eval/         # Yusif
└── long_context/       # Yusif
```

Pattern: notebook = orchestrator (load data, sweep configs, plot results).
Library = primitives. If you find yourself copy-pasting between notebooks,
that code belongs in `evaluation/common/`.

---

## 5. Where the data lives

### Public HuggingFace dataset
[https://huggingface.co/datasets/citeright/corpus](https://huggingface.co/datasets/citeright/corpus)

```
chunks/<paper_id>_<coarse|fine>.json    # 1 998 files — pre-computed semantic chunks
indexes/<coarse|fine>.faiss             # FAISS IndexFlatIP, 1 536-d OpenAI embeddings
indexes/<coarse|fine>_metadata.json     # row → chunk lookup
processed/<paper_id>/document.md        # Dolphin-1.5 parsed markdown
processed/<paper_id>/metadata.json      # page-level layout
raw_metadata/<paper_id>.json            # OpenAlex bibliographic record
```

Original PDFs are **not** redistributed (license-conservative). Refetch them
from OpenAlex via the `id` field in `raw_metadata/<paper_id>.json` if needed.

### How notebooks load data
```python
from evaluation.common import (
    artifact_root,
    iter_all_chunks,
    list_paper_ids,
    load_chunk_metadata,
    load_faiss_index,
    load_gold_qa,
    load_paper_chunks,
    load_paper_markdown,
    load_paper_openalex,
)

# Resolves automatically:
#  1. $CITERIGHT_DATA_DIR (set by submit.sh on RCP)
#  2. /scratch/citeright_artifacts/
#  3. data/s3_archive/  (laptop, if you have a local mirror)
#  4. HuggingFace download (last resort, ~3 GB)
ART = artifact_root()                              # Path

faiss_idx = load_faiss_index("coarse")             # ~50K vectors
meta      = load_chunk_metadata("coarse")          # dict[int, dict]
md        = load_paper_markdown("00002_W2122361802")
gold      = load_gold_qa()                         # list[dict]
```

First call on a fresh pod triggers the HF download into `/scratch/citeright_artifacts/`.
Subsequent calls hit the cache instantly.

---

## 6. Generation: local vs API models

We use a **dual-path dispatcher** ([`evaluation/common/models.py`](../evaluation/common/models.py)).
A model spec is a string with one of two prefixes; the dispatcher picks the
backend automatically:

```python
from evaluation.common import generate, available_models

# Local (vLLM on the A100). These three always work:
text = generate("local:Qwen/Qwen2.5-7B-Instruct", "Question: ...")
text = generate("local:meta-llama/Meta-Llama-3.1-8B-Instruct", "...")
text = generate("local:mistralai/Mistral-7B-Instruct-v0.3", "...")

# API (LiteLLM → OpenRouter). Only enabled if OPENROUTER_API_KEY is set.
text = generate("api:openrouter/openai/gpt-4o-mini", "...")

# Helper — returns local + (api if key is set):
specs = available_models()
```

### API keys belong **inside the notebook**, not in `submit.sh`
The course rubric forbids hardcoded tokens in the launcher. Read keys via
`getpass` inside your notebook so they never enter git history:

```python
import os, getpass
for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
    if var not in os.environ:
        v = getpass.getpass(f"{var} (blank to skip): ").strip()
        if v: os.environ[var] = v
```

Local models always work; API arms gracefully skip if no key.

### VRAM constraints
The 40 GB A100 fits one ≤14 B model in fp16/bf16. **Don't try to load two
models at once** — the dispatcher's `_VLLMRegistry` unloads the previous model
before loading a new one. The right pattern for cross-model comparisons is:

```python
for spec in available_models():
    answers = [generate(spec, q) for q in gold_questions]   # this unloads/loads
    save(spec, answers)                                     # to /scratch
# Score offline once all answers are persisted.
```

---

## 7. Gold dataset (everyone contributes)

Target: **50 expert-annotated Q&A pairs** by **2026-05-20** (~12–13 each).

- Schema: [`evaluation/gold_dataset/README.md`](../evaluation/gold_dataset/README.md)
- File:   [`evaluation/gold_dataset/gold_qa.json`](../evaluation/gold_dataset/gold_qa.json)
- Categories: `policy_impact`, `methodology`, `comparison`, `factual`, `multi_hop`
- Difficulty: `single-hop`, `multi-hop`, `unanswerable`

When annotating: pick a real chunk from the corpus, write a question whose
answer requires that chunk, record the chunk ID in `gold_passages`. Don't
write the answer first and search for support — that's circular.

Do annotations on a feature branch, PR, review each other.

---

## 8. Committing changes

We use **squash-merge PRs**. Branch naming: `feat/...`, `fix/...`,
`refactor/...`, `docs/...`.

```bash
git checkout -b feat/<short-description>
# ... edit, test ...
git add <specific files>          # avoid `git add -A`
git commit -m "feat: <what + why>"
git push -u origin feat/<...>
gh pr create --title "..." --body "..."
gh pr merge <num> --squash --delete-branch
```

### Don't commit edits to `submit.sh`
The submitted version has `GASPAR="${GASPAR:-gaspar}"` and `GROUP="${GROUP:-g68}"`
on purpose. If you change them locally for testing, do not commit. If you
need a permanent change to the launcher (env vars, image, mounts), open a PR
explicitly — TAs may need to adapt.

To make local edits invisible to git:
```bash
git update-index --skip-worktree notebooks/submit.sh
# undo with: git update-index --no-skip-worktree notebooks/submit.sh
```

### What goes in the repo, what doesn't
- ✅ Notebooks, library code, scripts, reports, gold data, generator
- ❌ Trained checkpoints, large eval outputs, FAISS indexes, datasets — those
     go to `/scratch` (transient) or HuggingFace (durable)
- ❌ `.venv/` (already gitignored)

---

## 9. GPU etiquette

The course supports up to 75 groups on shared infrastructure. A few habits:

- **Delete idle pods.** If you walk away for more than ~30 min, run
  `runai delete workspace <job-name>`. Resubmit in 5 s when you return.
- **Use interactive pods only for development.** For long ablation runs,
  submit a Run:AI **training job** instead (see `rcp_support/submit_train.sh`).
  Training jobs are preemptible — your code must checkpoint to `/scratch`.
- **Plan around deadlines.** Expect long queues around May 24 and June 7.
- **Don't ask for >1 GPU.** Course cap is 1 GPU per group at a time.

---

## 10. Troubleshooting

### `runai login` → DNS timeout
Tailscale or another VPN is hijacking DNS. Disable Tailscale's "Use
Tailscale DNS", or briefly disconnect Tailscale, then retry.

### `runai login` → AADSTS50105
Account not provisioned for the RCP Run:AI app. Email
`nlp-cs552-spring2026-ta-team@groupes.epfl.ch`.

### `git pull` → "dubious ownership"
Inside the pod, run once:
```bash
git config --global --add safe.directory /scratch/citeright
```

### `ModuleNotFoundError: torch` (after activating `.venv`)
Don't activate the repo's `.venv` inside the pod — the course image's
**system Python** has torch+vllm+transformers pre-installed:
```bash
deactivate                    # if a venv prompt is showing
rm -rf .venv                  # optional cleanup
which python                  # should be /usr/local/bin/python
python -c "import torch; print(torch.cuda.is_available())"   # True
```

### `pip install .` at the repo root → setuptools "multiple top-level packages"
Don't run that. Notebooks use `sys.path.insert(0, '.')` — file-based imports.
You don't need to install the project as a package.

### `runai workspace bash` → `failed to read frame header: EOF`
Shell session died (network blip, idle timeout). Just open a new one:
```bash
runai workspace bash <job-name>
```
The pod itself is unaffected.

### Corpus download is noisy
`snapshot_download` prints a progress bar per file (71K files = 71K lines).
That's a known UX limitation. Set `HF_HUB_DISABLE_PROGRESS_BARS=1` if you
want quiet output. The download itself is fine.

### Job stuck `Pending` for >5 min
GPU contention or a scheduler issue. Check with:
```bash
runai workload describe <job-name>     # see Pending reason in events
```
If it's been pending for >10 min around a deadline, ask in Ed.

---

## When in doubt

- Operational issues with RCP / Run:AI: see [`rcp_support/README.md`](../rcp_support/README.md)
  (the canonical reference written by course staff).
- Project / data / library questions: ping Elie on Slack/Discord.
- Course / grading questions: post in Ed Discussion.
