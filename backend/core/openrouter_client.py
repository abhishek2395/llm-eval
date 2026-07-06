"""
openrouter_client.py  —  Universal runner via OpenRouter.

One API key. One endpoint. Every model: Claude, GPT, Gemini, DeepSeek,
Llama, Grok, Mistral, Qwen, and 300+ more.

OpenRouter is OpenAI-SDK-compatible, so we use the openai package with
a custom base_url. No extra dependencies needed.

Usage:
    from openrouter_client import OpenRouterRunner, fetch_model_catalog

    runner = OpenRouterRunner()
    responses = runner.run_batch("anthropic/claude-sonnet-4-5", prompts)

    catalog = fetch_model_catalog()          # live list of all available models
    cheap = catalog.cheapest(n=10)           # 10 lowest cost-per-token models
    frontier = catalog.by_provider("openai") # all OpenAI models
"""

import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

import openai

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
    OPENROUTER_HEADERS,
    MAX_TOKENS,
    TEMPERATURE,
    PROVIDER_META,
)

logger = logging.getLogger(__name__)


# ── Model metadata ────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Metadata for a single model returned by the OpenRouter catalog API."""
    id: str
    name: str
    provider: str          # e.g. "anthropic", "openai", "google"
    context_length: int
    input_price_per_1m: float   # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    description: str = ""
    is_free: bool = False

    @property
    def provider_display(self) -> str:
        return PROVIDER_META.get(self.provider, {}).get("name", self.provider.title())

    @property
    def provider_color(self) -> str:
        return PROVIDER_META.get(self.provider, {}).get("color", "#64748b")

    @property
    def cost_per_1k_answers(self) -> float:
        """Estimated cost for 1000 average answers (150 in / 250 out tokens)."""
        in_cost  = (150 / 1_000_000) * self.input_price_per_1m
        out_cost = (250 / 1_000_000) * self.output_price_per_1m
        return (in_cost + out_cost) * 1000

    @property
    def answers_for_20_usd(self) -> int:
        """How many average answers $20 of API credits buys."""
        if self.cost_per_1k_answers <= 0:
            return 999_999
        return int(20 / (self.cost_per_1k_answers / 1000))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "provider_display": self.provider_display,
            "context_length": self.context_length,
            "input_price_per_1m": self.input_price_per_1m,
            "output_price_per_1m": self.output_price_per_1m,
            "is_free": self.is_free,
            "cost_per_1k_answers": round(self.cost_per_1k_answers, 4),
            "answers_for_20_usd": self.answers_for_20_usd,
        }


class ModelCatalog:
    """
    Live catalog of all models available on OpenRouter.
    Fetched once at startup and cached in memory.
    """

    def __init__(self, models: list[ModelInfo]):
        self._models = models
        self._by_id = {m.id: m for m in models}

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models)

    def get(self, model_id: str) -> Optional[ModelInfo]:
        return self._by_id.get(model_id)

    def all(self) -> list[ModelInfo]:
        return list(self._models)

    def by_provider(self, provider: str) -> list[ModelInfo]:
        p = provider.lower()
        return [m for m in self._models if m.provider.lower() == p]

    def providers(self) -> list[str]:
        return sorted(set(m.provider for m in self._models))

    def cheapest(self, n: int = 10, exclude_free: bool = False) -> list[ModelInfo]:
        src = [m for m in self._models if not m.is_free] if exclude_free else self._models
        return sorted(src, key=lambda m: m.input_price_per_1m)[:n]

    def free(self) -> list[ModelInfo]:
        return [m for m in self._models if m.is_free]

    def search(self, query: str) -> list[ModelInfo]:
        q = query.lower()
        return [m for m in self._models
                if q in m.id.lower() or q in m.name.lower() or q in m.provider.lower()]

    def top_by_context(self, n: int = 10) -> list[ModelInfo]:
        return sorted(self._models, key=lambda m: m.context_length, reverse=True)[:n]


def fetch_model_catalog(api_key: str = OPENROUTER_API_KEY) -> ModelCatalog:
    """
    Fetch the live model catalog from OpenRouter.
    Returns a ModelCatalog with full metadata for all available models.
    Falls back to a hardcoded minimal catalog if the API is unreachable.
    """
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — returning fallback catalog.")
        return _fallback_catalog()

    try:
        resp = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        models = []
        for m in data:
            raw_id = m.get("id", "")
            provider = raw_id.split("/")[0] if "/" in raw_id else "unknown"

            pricing  = m.get("pricing", {})
            in_price  = float(pricing.get("prompt",     0)) * 1_000_000
            out_price = float(pricing.get("completion", 0)) * 1_000_000

            models.append(ModelInfo(
                id=raw_id,
                name=m.get("name", raw_id),
                provider=provider,
                context_length=int(m.get("context_length", 0)),
                input_price_per_1m=round(in_price, 4),
                output_price_per_1m=round(out_price, 4),
                description=m.get("description", ""),
                is_free=(in_price == 0 and out_price == 0),
            ))

        # Sort: paid frontier first, then budget, then free
        models.sort(key=lambda m: (m.is_free, m.input_price_per_1m == 0, -m.context_length))
        logger.info(f"Fetched {len(models)} models from OpenRouter catalog.")
        return ModelCatalog(models)

    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter catalog: {e}. Using fallback.")
        return _fallback_catalog()


def _fallback_catalog() -> ModelCatalog:
    """Minimal hardcoded catalog used when the API is unreachable."""
    fallback = [
        ModelInfo("anthropic/claude-sonnet-4-5",          "Claude Sonnet 4.5",      "anthropic",  200_000, 3.00,  15.00),
        ModelInfo("anthropic/claude-haiku-4-5",           "Claude Haiku 4.5",       "anthropic",  200_000, 0.80,  4.00),
        ModelInfo("openai/gpt-4o",                        "GPT-4o",                 "openai",     128_000, 2.50,  10.00),
        ModelInfo("openai/gpt-4o-mini",                   "GPT-4o Mini",            "openai",     128_000, 0.15,  0.60),
        ModelInfo("openai/o4-mini",                       "o4-mini",                "openai",     128_000, 1.10,  4.40),
        ModelInfo("google/gemini-2.5-pro",                "Gemini 2.5 Pro",         "google",   1_000_000, 1.25,  10.00),
        ModelInfo("google/gemini-2.0-flash-001",          "Gemini 2.0 Flash",       "google",   1_048_576, 0.10,  0.40),
        ModelInfo("deepseek/deepseek-chat",               "DeepSeek V3",            "deepseek", 163_840,   0.27,  1.10),
        ModelInfo("deepseek/deepseek-r1",                 "DeepSeek R1",            "deepseek",  65_536,   0.55,  2.19),
        ModelInfo("x-ai/grok-3",                         "Grok 3",                 "x-ai",      131_072,  3.00,  15.00),
        ModelInfo("meta-llama/llama-3.3-70b-instruct",   "Llama 3.3 70B",          "meta-llama",131_072,  0.12,  0.30),
        ModelInfo("mistralai/mistral-large",             "Mistral Large",           "mistralai", 131_072,  2.00,  6.00),
        ModelInfo("qwen/qwen-2.5-72b-instruct",          "Qwen 2.5 72B",           "qwen",      131_072,  0.35,  0.40),
    ]
    return ModelCatalog(fallback)


# ── Response dataclass (same interface as old ModelResponse) ──────────────────

@dataclass
class ModelResponse:
    model: str
    prompt_id: str
    prompt_text: str
    response_text: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    provider: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


# ── OpenRouter runner ─────────────────────────────────────────────────────────

class OpenRouterRunner:
    """
    Universal runner for any model available on OpenRouter.
    Uses the OpenAI SDK with a custom base_url — no extra dependencies.

    One instance handles Claude, GPT, Gemini, DeepSeek, Llama, etc.
    """

    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set.\n"
                "Get a free key at https://openrouter.ai/keys\n"
                "Then: export OPENROUTER_API_KEY='sk-or-...'"
            )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )

    def run(
        self,
        model: str,
        prompt_id: str,
        prompt_text: str,
        system_prompt: str = "You are a helpful and precise assistant.",
        max_retries: int = 3,
    ) -> ModelResponse:
        """Run one prompt through one model. Returns a ModelResponse."""
        provider = model.split("/")[0] if "/" in model else "unknown"
        last_error = None

        for attempt in range(max_retries):
            try:
                start = time.time()
                completion = self.client.chat.completions.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt_text},
                    ],
                )
                latency_ms = (time.time() - start) * 1000

                return ModelResponse(
                    model=model,
                    provider=provider,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    response_text=completion.choices[0].message.content or "",
                    latency_ms=latency_ms,
                    input_tokens=getattr(completion.usage, "prompt_tokens", 0),
                    output_tokens=getattr(completion.usage, "completion_tokens", 0),
                )

            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"[{model}] Rate limit (attempt {attempt+1}). Waiting {wait}s…")
                time.sleep(wait)
                last_error = "rate_limit"

            except openai.APIStatusError as e:
                # 402 = insufficient credits, 404 = model not found
                code = e.status_code
                if code == 402:
                    logger.error(f"[{model}] Insufficient OpenRouter credits.")
                    return ModelResponse(model=model, provider=provider,
                                        prompt_id=prompt_id, prompt_text=prompt_text,
                                        response_text="", latency_ms=0,
                                        error=f"insufficient_credits ({code})")
                if code == 404:
                    logger.error(f"[{model}] Model not found on OpenRouter.")
                    return ModelResponse(model=model, provider=provider,
                                        prompt_id=prompt_id, prompt_text=prompt_text,
                                        response_text="", latency_ms=0,
                                        error=f"model_not_found ({code})")
                logger.error(f"[{model}] API error {code}: {e}")
                last_error = f"api_error_{code}"
                break

            except Exception as e:
                logger.error(f"[{model}] Unexpected error: {e}")
                last_error = str(e)
                break

        return ModelResponse(
            model=model, provider=provider,
            prompt_id=prompt_id, prompt_text=prompt_text,
            response_text="", latency_ms=0,
            error=last_error or "max_retries_exceeded",
        )

    def run_batch(
        self,
        model: str,
        prompts: list[dict],
        system_prompt: str = "You are a helpful and precise assistant.",
        delay_between: float = 0.4,
    ) -> list[ModelResponse]:
        """Run all prompts for one model. Returns list of ModelResponse."""
        results = []
        n = len(prompts)
        for i, p in enumerate(prompts):
            logger.info(f"  [{model}]  {i+1}/{n}  prompt={p['id']}")
            result = self.run(model, p["id"], p["prompt"], system_prompt)
            results.append(result)
            if delay_between > 0 and i < n - 1:
                time.sleep(delay_between)
        return results

    def run_all_models(
        self,
        models: list[str],
        prompts: list[dict],
        system_prompt: str = "You are a helpful and precise assistant.",
        delay_between_models: float = 1.0,
    ) -> list[ModelResponse]:
        """Run multiple models against all prompts. Returns flat list."""
        all_responses = []
        for i, model in enumerate(models):
            logger.info(f"\n── Model {i+1}/{len(models)}: {model}")
            responses = self.run_batch(model, prompts, system_prompt)
            success = sum(1 for r in responses if r.success)
            logger.info(f"   {success}/{len(responses)} successful")
            all_responses.extend(responses)
            if delay_between_models > 0 and i < len(models) - 1:
                time.sleep(delay_between_models)
        return all_responses
