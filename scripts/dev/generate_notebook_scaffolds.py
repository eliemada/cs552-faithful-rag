"""Generate the four CS-552 deliverable notebooks (one per teammate).

Each notebook shares a common bootstrap (deps + artefact download + smoke test)
so the grader can ``Run All`` on a clean RCP pod with no manual setup. The
contribution-specific sections sketch the experiments described in the proposal.

Run::

    uv run python scripts/dev/generate_notebook_scaffolds.py

This is a code-generator: rerunning it overwrites the four target notebooks. Do
not put real experimental results here — those go in the per-member notebook
once the team starts filling them in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import nbformat

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


@dataclass(frozen=True)
class Member:
    first: str
    last: str
    sciper: int
    contribution_title: str
    contribution_blurb: str
    sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Sequence of ``(section_title, section_body_markdown)`` pairs."""

    @property
    def filename(self) -> str:
        return f"{self.first.lower()}_{self.last.lower()}_{self.sciper}.ipynb"


# ---- common cells (shared verbatim across all four notebooks) -----------------


def title_cells(m: Member) -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            f"# {m.contribution_title}\n\n"
            f"**{m.first} {m.last}** &middot; SCIPER {m.sciper} &middot; "
            "CS-552 Spring 2026 &middot; Team CiteRight\n\n"
            f"{m.contribution_blurb}\n\n"
            "---\n\n"
            "This notebook is part of the **Faithful RAG** Open Project. "
            "It runs end-to-end inside the RCP pod launched by "
            "[`notebooks/submit.sh`](./submit.sh) with no manual setup: dependencies "
            "install in the bootstrap cell, the corpus downloads from "
            "[`citeright/corpus`](https://huggingface.co/datasets/citeright/corpus) "
            "into `/scratch`, and all generation runs locally on the 40 GB A100 by "
            "default. API-based comparisons are optional and skipped if no key is "
            "provided."
        ),
    ]


def bootstrap_cells() -> list[nbformat.NotebookNode]:
    install_cmd = (
        "%pip install --quiet "
        "huggingface_hub sentence-transformers faiss-cpu "
        "datasets transformers accelerate "
        "litellm tqdm pandas matplotlib"
    )
    bootstrap = '\n'.join([
        "import logging",
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')",
        "",
        "# Locate repo root (notebook lives in <repo>/notebooks/).",
        "REPO_ROOT = Path.cwd().resolve()",
        "while REPO_ROOT.name == 'notebooks' or not (REPO_ROOT / 'pyproject.toml').exists():",
        "    if REPO_ROOT.parent == REPO_ROOT:",
        "        raise RuntimeError('Could not find repo root (no pyproject.toml above notebook).')",
        "    REPO_ROOT = REPO_ROOT.parent",
        "if str(REPO_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(REPO_ROOT))",
        "",
        "# On RCP submit.sh sets these; default for laptop dev otherwise.",
        "os.environ.setdefault('CITERIGHT_HF_REPO', 'citeright/corpus')",
        "if Path('/scratch').is_dir():",
        "    os.environ.setdefault('CITERIGHT_DATA_DIR', '/scratch/citeright_artifacts')",
        "    os.environ.setdefault('HF_HOME', '/scratch/hf_cache')",
        "",
        "from evaluation.common import (",
        "    artifact_root, available_models, generate, ",
        "    load_chunk_metadata, load_faiss_index, load_gold_qa, load_paper_markdown,",
        ")",
        "",
        "ART = artifact_root()",
        "print(f'Corpus root: {ART}')",
        "print(f'Available models: {len(available_models())}')",
    ])
    optional_keys = '\n'.join([
        "# Optional: paid-API comparison arms.",
        "# Skip these cells if you only want the local-model path (always works).",
        "import getpass",
        "for var in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY'):",
        "    if var not in os.environ:",
        "        val = getpass.getpass(f'{var} (leave blank to skip): ').strip()",
        "        if val:",
        "            os.environ[var] = val",
        "print('Local models always available; API arms enabled iff keys above were provided.')",
    ])
    return [
        nbformat.v4.new_markdown_cell(
            "## Setup\n\n"
            "Installs Python dependencies (idempotent on rerun), wires `sys.path` "
            "so we can `from evaluation.common import ...`, and resolves the corpus "
            "cache. On RCP this finds `/scratch/citeright_artifacts`; on a laptop it "
            "falls back to `data/s3_archive/` if you've cloned a local mirror, "
            "otherwise downloads from HuggingFace."
        ),
        nbformat.v4.new_code_cell(install_cmd),
        nbformat.v4.new_code_cell(bootstrap),
        nbformat.v4.new_markdown_cell(
            "### Optional: API keys\n\n"
            "The notebook works with **only** local models. If you want the "
            "OpenAI / OpenRouter comparison arms (e.g. GPT-5, Claude, DeepSeek), "
            "paste the keys when prompted below. Leave blank to skip."
        ),
        nbformat.v4.new_code_cell(optional_keys),
    ]


def smoke_test_cells() -> list[nbformat.NotebookNode]:
    smoke = '\n'.join([
        "# Confirm the corpus loads end-to-end.",
        "import faiss",
        "",
        "meta = load_chunk_metadata('coarse')",
        "index = load_faiss_index('coarse')",
        "gold = load_gold_qa()",
        "",
        "print(f'Coarse FAISS rows: {index.ntotal}')",
        "print(f'Coarse metadata entries: {len(meta)}')",
        "print(f'Gold Q&A pairs: {len(gold)}')",
        "assert index.ntotal == len(meta), 'FAISS / metadata size mismatch'",
        "",
        "# Show one chunk so we know the schema.",
        "first = meta[0]",
        "print(f\"\\nExample chunk:\\n  paper_id: {first['paper_id']}\")",
        "print(f\"  section : {first.get('section_hierarchy', [])}\")",
        "print(f\"  text    : {first['text'][:200]}...\")",
    ])
    return [
        nbformat.v4.new_markdown_cell(
            "## Smoke test\n\n"
            "Loads the FAISS index, the row-→-chunk metadata, and the gold "
            "evaluation set. If this cell errors, the rest of the notebook will "
            "not run — see `evaluation/common/data_loader.py` for the resolution "
            "rules."
        ),
        nbformat.v4.new_code_cell(smoke),
    ]


def member_section_cells(m: Member) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    for title, body in m.sections:
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{body}"))
        cells.append(nbformat.v4.new_code_cell("# TODO: implement\nraise NotImplementedError"))
    return cells


def closing_cells() -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            "## Results & discussion\n\n"
            "Summarise the headline numbers, ablations, and comparisons here. "
            "The 4-page report references plots and tables produced above."
        ),
        nbformat.v4.new_markdown_cell(
            "## Reproduction\n\n"
            "From a clean clone of the repo:\n\n"
            "```bash\n"
            "cd notebooks && ./submit.sh         # launch the RCP pod\n"
            "runai port-forward <job> --port 8888:8888\n"
            "# open http://localhost:8888 (token: cs552), then Run All on this notebook\n"
            "```\n\n"
            "All artefacts download automatically from "
            "[`citeright/corpus`](https://huggingface.co/datasets/citeright/corpus); "
            "no S3 or AWS access is required."
        ),
    ]


# ---- per-member contribution sections ----------------------------------------


MEMBERS: tuple[Member, ...] = (
    Member(
        first="Elie",
        last="Bruno",
        sciper=355932,
        contribution_title="Retrieval Ablation: Embedders, Chunk Granularity, and Reranking",
        contribution_blurb=(
            "Compares three bi-encoders (OpenAI `text-embedding-3-small`, BGE-M3, "
            "E5-large) and ColBERTv2 late-interaction across coarse and fine chunks, "
            "with and without a cross-encoder reranker, scored on 50 gold queries "
            "with Precision@k, Recall@k, MRR, and Hit Rate."
        ),
        sections=(
            (
                "Build / load embeddings",
                "Embed every chunk with each candidate model. The OpenAI vectors "
                "are pre-built (FAISS in `indexes/coarse.faiss`); BGE-M3 and "
                "E5-large run locally via `sentence-transformers`. Cache to "
                "`/scratch` so the second pass is free.",
            ),
            (
                "Run gold queries",
                "For each (embedder × granularity × ±reranker) configuration, "
                "retrieve top-50 for every gold question. Save retrieved chunk "
                "IDs and scores to `evaluation/retrieval_eval/results/`.",
            ),
            (
                "Compute metrics",
                "Use `evaluation.retrieval_eval.evaluate_retrieval.evaluate_retriever`. "
                "Tabulate Precision@{5,10,20}, Recall@{5,10,20}, MRR, Hit@k.",
            ),
            (
                "Plot ablation table & failure analysis",
                "Bar plot per metric × granularity. Pull a handful of failure "
                "cases (gold not in top-20) and inspect what was retrieved instead.",
            ),
        ),
    ),
    Member(
        first="Andrea",
        last="Trugenberger",
        sciper=357615,
        contribution_title="Citation Faithfulness: Atomic Claim Extraction + NLI Verification",
        contribution_blurb=(
            "Generates RAG answers with several LLMs, decomposes each answer into "
            "atomic claims, and uses DeBERTa-v3-large MNLI to label every "
            "claim–citation pair as supported / not-supported / fabricated-source. "
            "Reports faithfulness per generator following the ALCE protocol."
        ),
        sections=(
            (
                "Generate cited answers",
                "For each gold question, run the standard RAG pipeline (top-k "
                "retrieval + prompt) with every model in `available_models()`. "
                "Persist `(question, answer, retrieved_chunks)` triples.",
            ),
            (
                "Atomic claim extraction",
                "Decompose answers via `evaluation.faithfulness.faithfulness_scorer."
                "extract_claims`. Use a prompted LLM (FActScore-style) for now; "
                "swap in a fine-tuned splitter later if needed.",
            ),
            (
                "NLI scoring",
                "Run `verify_claim_nli` per claim against its cited passage. "
                "DeBERTa-v3-large MNLI returns `(entailment, neutral, "
                "contradiction)` — map to the four-way label space.",
            ),
            (
                "Cross-LLM faithfulness comparison",
                "Aggregate `compute_faithfulness_score` across generators. Plot "
                "supported / not-supported / fabricated ratios per model. Discuss "
                "the local-vs-API gap.",
            ),
        ),
    ),
    Member(
        first="Faruk",
        last="Zahiragic",
        sciper=415360,
        contribution_title="Corrective RAG: Retrieval-Quality Evaluator and Threshold Ablation",
        contribution_blurb=(
            "Implements Yan et al. (2024) Corrective RAG with a lightweight "
            "retrieval-quality evaluator and query refinement loop. Sweeps the "
            "confidence thresholds and reports the impact on faithfulness, "
            "answer relevance, and latency."
        ),
        sections=(
            (
                "Baseline RAG outputs",
                "Run vanilla retrieve-and-generate on the gold set with the "
                "default model. Save baseline answers + retrieval traces.",
            ),
            (
                "Implement the retrieval-quality evaluator",
                "Score each retrieval as CORRECT / AMBIGUOUS / INCORRECT using "
                "either an NLI cross-encoder or a prompted LLM. See "
                "`evaluation.crag.corrective_rag.evaluate_retrieval_quality`.",
            ),
            (
                "Query refinement loop",
                "When AMBIGUOUS, refine the query (LLM rewriting in "
                "`refine_query`) and re-retrieve. Cap rounds at 2 to bound cost.",
            ),
            (
                "Threshold sweep",
                "Sweep the high/low confidence thresholds in "
                "`CRAGConfig`. For each (τ_high, τ_low), report Δ-faithfulness "
                "vs baseline, retrieval-round count, and latency. Plot the "
                "Pareto curve.",
            ),
        ),
    ),
    Member(
        first="Yusif",
        last="Askari",
        sciper=413862,
        contribution_title="End-to-End Evaluation: RAGAS Metrics and 128 K Long-Context Baseline",
        contribution_blurb=(
            "Scores every pipeline variant with RAGAS (faithfulness, answer "
            "relevancy, context precision/recall) and benchmarks them against a "
            "long-context baseline that stuffs full papers into a 128 K window. "
            "Reports the cost / latency / quality trade-off."
        ),
        sections=(
            (
                "Run the full pipeline on the gold set",
                "For each generator, collect `(question, answer, contexts, "
                "ground_truth)` tuples. Use the same prompt template across "
                "configurations so RAGAS scores are comparable.",
            ),
            (
                "RAGAS scoring",
                "Build a `datasets.Dataset` from the tuples and call "
                "`ragas.evaluate(...)` with the four metrics. Cache scores to "
                "`evaluation/ragas_eval/results/`.",
            ),
            (
                "Long-context baseline",
                "For each gold question identify the relevant papers, "
                "concatenate their `document.md` (truncating to 128 K tokens), "
                "and answer in one shot with a 128 K-context model. Score with "
                "the same RAGAS metrics.",
            ),
            (
                "Cost / latency / accuracy trade-off",
                "Plot RAGAS faithfulness vs (i) tokens per query and (ii) "
                "wall-clock latency. Compare chunked-RAG configurations to the "
                "long-context baseline. Discuss when each wins.",
            ),
        ),
    ),
)


def build_notebook(m: Member) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb.cells = [
        *title_cells(m),
        *bootstrap_cells(),
        *smoke_test_cells(),
        *member_section_cells(m),
        *closing_cells(),
    ]
    return nb


def write_all(members: Iterable[Member]) -> None:
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for m in members:
        nb = build_notebook(m)
        path = NOTEBOOKS_DIR / m.filename
        with path.open("w") as fh:
            nbformat.write(nb, fh)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    write_all(MEMBERS)
