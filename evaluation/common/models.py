"""Dual-path generation: vLLM (local) and LiteLLM (API) under one ``generate()``.

A *model spec* is a string with one of two prefixes:

* ``local:<HF_id>`` — served by vLLM on the local GPU. Single model resident at a
  time; switching specs unloads the previous one to free VRAM.
* ``api:<provider/model>`` — routed through LiteLLM to OpenRouter. Requires
  ``OPENROUTER_API_KEY`` to be set; otherwise we raise so the caller can skip.

This decoupling lets every notebook iterate over a list of model specs and stay
honest about which results came from which backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageInfo:
    """Token + cost data for one LLM call.

    ``cost_usd`` is ``0.0`` for ``local:`` backends (compute on-cluster, no
    per-call API charge) and computed via ``litellm.completion_cost`` for
    ``api:`` backends. ``model`` is the underlying model id without the
    ``local:`` / ``api:`` prefix.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    model: str


# Models that fit in a single 40 GB A100 in bf16/fp16 (≤ 14 B params).
LOCAL_MODELS_DEFAULT: Final[tuple[str, ...]] = (
    "local:Qwen/Qwen2.5-7B-Instruct",
    "local:meta-llama/Meta-Llama-3.1-8B-Instruct",
    "local:mistralai/Mistral-7B-Instruct-v0.3",
)

# Larger / closed-source models reachable only over the API.
API_MODELS_DEFAULT: Final[tuple[str, ...]] = (
    "api:openrouter/openai/gpt-4o-mini",
    "api:openrouter/anthropic/claude-haiku-4-5",
    "api:openrouter/deepseek/deepseek-chat",
)

_LOCAL_PREFIX: Final[str] = "local:"
_API_PREFIX: Final[str] = "api:"


def available_models(*, include_api: bool = True) -> list[str]:
    """Return the model specs that can run in the current environment.

    API-only specs are filtered out when no ``OPENROUTER_API_KEY`` is configured,
    so a grader without keys still gets a non-empty list.
    """
    specs = list(LOCAL_MODELS_DEFAULT)
    if include_api and os.environ.get("OPENROUTER_API_KEY"):
        specs.extend(API_MODELS_DEFAULT)
    return specs


def generate(
    model_spec: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
) -> str:
    """Generate a completion using the backend implied by ``model_spec``."""
    text, _ = generate_with_usage(
        model_spec,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
    )
    return text


def generate_with_usage(
    model_spec: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
) -> tuple[str, UsageInfo]:
    """Like :func:`generate` but also returns token / cost data.

    Used by evaluation pipelines that need to track per-call cost (e.g. the
    RAGAS RAG-vs-long-context comparison). Existing callers that only need
    the text should keep using :func:`generate`.
    """
    if model_spec.startswith(_LOCAL_PREFIX):
        return _generate_vllm(
            model_spec.removeprefix(_LOCAL_PREFIX),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
    if model_spec.startswith(_API_PREFIX):
        return _generate_litellm(
            model_spec.removeprefix(_API_PREFIX),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
    raise ValueError(f"Unknown model_spec prefix in {model_spec!r}; expected 'local:' or 'api:'")


# ---------- embeddings & NLI ----------


def load_embedder(name: str = "BAAI/bge-m3") -> Any:
    """Load a sentence-transformers embedder (BGE-M3, E5-large, GTE, ...)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedder %s", name)
    return SentenceTransformer(name, trust_remote_code=True)


def load_nli_classifier(
    name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
) -> tuple[Any, Any]:
    """Load an NLI classifier for faithfulness scoring.

    Returns ``(model, tokenizer)``. The default checkpoint is a strong DeBERTa-v3-large
    fine-tune on MNLI+FEVER+ANLI; override to ``microsoft/deberta-large-mnli`` etc.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI classifier %s", name)
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    return mdl, tok


# ---------- backend implementations ----------


class _VLLMRegistry:
    """Single-slot vLLM cache; switching models unloads the previous one."""

    _current_id: str | None = None
    _current_llm: Any = None

    @classmethod
    def get(cls, hf_id: str) -> Any:
        if cls._current_id == hf_id and cls._current_llm is not None:
            return cls._current_llm

        if cls._current_llm is not None:
            logger.info("Unloading vLLM model %s", cls._current_id)
            cls._current_llm = None
            cls._current_id = None
            _empty_cuda_cache()

        from vllm import LLM

        gpu_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
        max_len = int(os.environ.get("VLLM_MAX_LEN", "8192"))
        logger.info("Loading vLLM model %s (gpu_util=%.2f, max_len=%d)", hf_id, gpu_util, max_len)
        cls._current_llm = LLM(
            model=hf_id,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            trust_remote_code=True,
        )
        cls._current_id = hf_id
        return cls._current_llm


def _generate_vllm(
    hf_id: str,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
) -> tuple[str, UsageInfo]:
    from vllm import SamplingParams

    llm = _VLLMRegistry.get(hf_id)
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    full_prompt = f"{system.strip()}\n\n{prompt}" if system else prompt
    [out] = llm.generate([full_prompt], params, use_tqdm=False)
    text = out.outputs[0].text
    prompt_tokens = len(getattr(out, "prompt_token_ids", []) or [])
    completion_tokens = len(getattr(out.outputs[0], "token_ids", []) or [])
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=0.0,  # local inference; compute cost not counted here
        model=hf_id,
    )
    return text, usage


def _generate_litellm(
    model_id: str,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
) -> tuple[str, UsageInfo]:
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Either provide it inside the notebook "
            "or restrict to local: models via available_models(include_api=False)."
        )

    import litellm  # type: ignore[import-untyped]

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = litellm.completion(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    text = resp.choices[0].message.content
    usage_obj = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0)
    # OpenRouter populates ``usage.cost`` with the actual amount it charged the
    # account (USD), which is the most accurate figure available. Prefer it
    # when present; fall back to litellm's static pricing table for providers
    # that don't return cost in the response.
    cost_usd = 0.0
    or_cost = getattr(usage_obj, "cost", None)
    if or_cost is not None:
        cost_usd = float(or_cost)
    else:
        try:
            cost_usd = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception as exc:  # noqa: BLE001 — litellm raises a grab-bag of errors for missing pricing
            logger.warning("litellm.completion_cost failed for %s: %s", model_id, exc)
            cost_usd = 0.0
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        model=model_id,
    )
    return text, usage


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
