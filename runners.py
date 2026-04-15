"""
runners.py — Async model runners for Claude and OpenAI.
Handles retries, rate limiting, and response normalization.
"""

import asyncio
import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import anthropic
import openai

from config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MAX_TOKENS, TEMPERATURE
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Normalized response from any model provider."""
    model: str
    prompt_id: str
    prompt_text: str
    response_text: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


# ── Claude Runner ─────────────────────────────────────────────────────────────

class ClaudeRunner:
    def __init__(self, api_key: str = ANTHROPIC_API_KEY):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Export it as an environment variable.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def run(
        self,
        model: str,
        prompt_id: str,
        prompt_text: str,
        system_prompt: str = "You are a helpful and precise assistant.",
        max_retries: int = 3,
    ) -> ModelResponse:
        """Run a single prompt through a Claude model with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                start = time.time()
                message = self.client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                latency_ms = (time.time() - start) * 1000

                return ModelResponse(
                    model=model,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    response_text=message.content[0].text,
                    latency_ms=latency_ms,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                )

            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Claude rate limit hit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                last_error = "rate_limit"

            except anthropic.APIError as e:
                logger.error(f"Claude API error for {prompt_id}: {e}")
                last_error = str(e)
                break

        return ModelResponse(
            model=model,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            response_text="",
            latency_ms=0,
            error=last_error or "max_retries_exceeded",
        )

    def run_batch(
        self,
        model: str,
        prompts: list[dict],
        system_prompt: str = "You are a helpful and precise assistant.",
        delay_between: float = 0.5,
    ) -> list[ModelResponse]:
        """Run all prompts for a model sequentially with a small delay."""
        results = []
        for i, p in enumerate(prompts):
            logger.info(f"  [{i+1}/{len(prompts)}] Running {model} on {p['id']}...")
            result = self.run(model, p["id"], p["prompt"], system_prompt)
            results.append(result)
            if delay_between > 0:
                time.sleep(delay_between)
        return results


# ── OpenAI Runner ─────────────────────────────────────────────────────────────

class OpenAIRunner:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Export it as an environment variable.")
        self.client = openai.OpenAI(api_key=api_key)

    def run(
        self,
        model: str,
        prompt_id: str,
        prompt_text: str,
        system_prompt: str = "You are a helpful and precise assistant.",
        max_retries: int = 3,
    ) -> ModelResponse:
        """Run a single prompt through an OpenAI model with retry logic."""
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
                        {"role": "user", "content": prompt_text},
                    ],
                )
                latency_ms = (time.time() - start) * 1000

                return ModelResponse(
                    model=model,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    response_text=completion.choices[0].message.content,
                    latency_ms=latency_ms,
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                )

            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"OpenAI rate limit hit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                last_error = "rate_limit"

            except openai.APIError as e:
                logger.error(f"OpenAI API error for {prompt_id}: {e}")
                last_error = str(e)
                break

        return ModelResponse(
            model=model,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            response_text="",
            latency_ms=0,
            error=last_error or "max_retries_exceeded",
        )

    def run_batch(
        self,
        model: str,
        prompts: list[dict],
        system_prompt: str = "You are a helpful and precise assistant.",
        delay_between: float = 0.5,
    ) -> list[ModelResponse]:
        """Run all prompts for a model sequentially."""
        results = []
        for i, p in enumerate(prompts):
            logger.info(f"  [{i+1}/{len(prompts)}] Running {model} on {p['id']}...")
            result = self.run(model, p["id"], p["prompt"], system_prompt)
            results.append(result)
            if delay_between > 0:
                time.sleep(delay_between)
        return results
