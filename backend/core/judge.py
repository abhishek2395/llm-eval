"""
judge.py — LLM-as-judge scoring engine.
Uses Claude via OpenRouter to evaluate model responses.
Only requires OPENROUTER_API_KEY — no separate Anthropic key needed.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import openai

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    JUDGE_MODEL, JUDGE_TEMPERATURE, RUBRIC
)

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """Structured scores from the LLM judge for one response."""
    prompt_id: str
    model: str
    # Quality dimensions
    accuracy: int = 0
    hallucination_resistance: int = 0
    relevance: int = 0
    instruction_following: int = 0
    conciseness: int = 0
    task_completion: int = 0
    composite_score: float = 0.0
    # Explanations
    rationale: str = ""
    notable_issues: str = ""
    one_line_verdict: str = ""
    judge_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.judge_error is None

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "model": self.model,
            "accuracy": self.accuracy,
            "hallucination_resistance": self.hallucination_resistance,
            "relevance": self.relevance,
            "instruction_following": self.instruction_following,
            "conciseness": self.conciseness,
            "task_completion": self.task_completion,
            "composite_score": round(self.composite_score, 3),
            "rationale": self.rationale,
            "notable_issues": self.notable_issues,
            "one_line_verdict": self.one_line_verdict,
            "judge_error": self.judge_error,
        }


class LLMJudge:
    """
    Evaluates model responses using the LLM-as-judge pattern.
    Routes through OpenRouter — only OPENROUTER_API_KEY required.
    JUDGE_MODEL in config.py controls which model acts as judge
    (defaults to anthropic/claude-sonnet-4-5).
    """

    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set.\n"
                "Add it to your .env file:  OPENROUTER_API_KEY=sk-or-...\n"
                "Get a key at: https://openrouter.ai/keys"
            )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )

    def _build_judge_prompt(
        self,
        prompt_text: str,
        response_text: str,
        ground_truth: str,
        category: str,
    ) -> str:
        return f"""
CATEGORY: {category}

ORIGINAL QUESTION:
{prompt_text}

GROUND TRUTH / EXPECTED ANSWER:
{ground_truth}

MODEL RESPONSE TO EVALUATE:
{response_text}

{RUBRIC}
""".strip()

    def score(
        self,
        prompt_id: str,
        model: str,
        prompt_text: str,
        response_text: str,
        ground_truth: str,
        category: str,
        max_retries: int = 3,
    ) -> JudgeScore:
        """Score a single model response. Returns a JudgeScore with all dimensions."""

        if not response_text:
            return JudgeScore(
                prompt_id=prompt_id,
                model=model,
                judge_error="empty_response_skipped",
            )

        judge_prompt = self._build_judge_prompt(
            prompt_text, response_text, ground_truth, category
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=JUDGE_MODEL,
                    max_tokens=512,
                    temperature=JUDGE_TEMPERATURE,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a rigorous LLM evaluation expert. "
                                "You always respond with valid JSON only, exactly as specified. "
                                "No preamble, no markdown fences, just the JSON object."
                            ),
                        },
                        {"role": "user", "content": judge_prompt},
                    ],
                )

                raw = completion.choices[0].message.content.strip()

                # Strip markdown code fences if model adds them
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                data = json.loads(raw)

                return JudgeScore(
                    prompt_id=prompt_id,
                    model=model,
                    accuracy=int(data.get("accuracy", 0)),
                    hallucination_resistance=int(data.get("hallucination_resistance", 0)),
                    relevance=int(data.get("relevance", 0)),
                    instruction_following=int(data.get("instruction_following", 5)),
                    conciseness=int(data.get("conciseness", 3)),
                    task_completion=int(data.get("task_completion", 3)),
                    composite_score=float(data.get("composite_score", 0.0)),
                    rationale=data.get("rationale", ""),
                    notable_issues=data.get("notable_issues", "none"),
                    one_line_verdict=data.get("one_line_verdict", ""),
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Judge JSON parse error (attempt {attempt+1}): {e}")
                last_error = f"json_parse_error: {e}"
                time.sleep(1)

            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Judge rate limit. Waiting {wait}s...")
                time.sleep(wait)
                last_error = "rate_limit"

            except openai.APIStatusError as e:
                logger.error(f"Judge API error {e.status_code}: {e}")
                last_error = f"api_error_{e.status_code}"
                break

            except Exception as e:
                logger.error(f"Judge unexpected error: {e}")
                last_error = str(e)
                break

        return JudgeScore(
            prompt_id=prompt_id,
            model=model,
            judge_error=last_error or "max_retries_exceeded",
        )

    def score_batch(
        self,
        responses: list,
        prompts_lookup: dict,
        delay_between: float = 0.3,
    ) -> list[JudgeScore]:
        """Score a list of ModelResponse objects. Returns JudgeScore list."""
        scores = []
        total = len(responses)

        for i, resp in enumerate(responses):
            prompt_meta = prompts_lookup.get(resp.prompt_id, {})
            logger.info(
                f"  Judging [{i+1}/{total}] model={resp.model} prompt={resp.prompt_id}..."
            )

            score = self.score(
                prompt_id=resp.prompt_id,
                model=resp.model,
                prompt_text=resp.prompt_text,
                response_text=resp.response_text,
                ground_truth=prompt_meta.get("ground_truth", ""),
                category=prompt_meta.get("category", "general"),
            )
            scores.append(score)

            if delay_between > 0:
                time.sleep(delay_between)

        return scores
