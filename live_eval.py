from __future__ import annotations

"""
live_eval.py — Real-time eval engine for the Streamlit dashboard.

Uses a generator pattern so each completed (response + score) pair
can be yielded back to the dashboard immediately, enabling live
progress bars and incremental chart updates.

Usage:
    for update in run_model_live(model_id, api_key):
        # update is a dict with keys:
        #   type: "progress" | "result" | "done" | "error"
        #   prompt_idx, prompt_total, prompt_id (for progress/result)
        #   response_row, score_row  (for result — DataFrames rows as dicts)
        #   message  (for error/done)
        update_ui(update)
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd

from config import (
    PROMPTS_FILE, RESULTS_DIR, RESULTS_FILE, SCORES_FILE,
    OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    JUDGE_MODEL, JUDGE_TEMPERATURE, RUBRIC,
    VALUE_WEIGHTS,
)

logger = logging.getLogger(__name__)

# ── Cache key: model → set of already-evaluated prompt IDs ───────────────────
CACHE_FILE = RESULTS_DIR / "eval_cache.json"


def _load_cache() -> dict:
    """Load the prompt-level cache: {model_id: [prompt_id, ...]}"""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def already_evaluated(model: str, prompt_id: str) -> tuple[dict | None, dict | None]:
    """
    Check if (model, prompt_id) already exists in the saved CSVs.
    Returns (response_row_dict, score_row_dict) or (None, None).
    """
    if not RESULTS_FILE.exists() or not SCORES_FILE.exists():
        return None, None
    try:
        r = pd.read_csv(RESULTS_FILE)
        s = pd.read_csv(SCORES_FILE)
        rrow = r[(r["model"] == model) & (r["prompt_id"] == prompt_id)]
        srow = s[(s["model"] == model) & (s["prompt_id"] == prompt_id)]
        if not rrow.empty and not srow.empty:
            return rrow.iloc[0].to_dict(), srow.iloc[0].to_dict()
    except Exception:
        pass
    return None, None


def _call_model(client, model: str, prompt: str,
                system: str = "You are a helpful and precise assistant.") -> dict:
    """Single model call. Returns dict with response_text, latency_ms, tokens."""
    import openai
    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        latency_ms = (time.time() - start) * 1000
        text = completion.choices[0].message.content or ""
        in_tok  = getattr(completion.usage, "prompt_tokens", 0)
        out_tok = getattr(completion.usage, "completion_tokens", 0)

        # first_token approximation: 30% of total latency (streaming not used here)
        first_tok_ms = latency_ms * 0.30

        return {
            "response_text": text,
            "total_latency_ms": round(latency_ms, 1),
            "first_token_latency_ms": round(first_tok_ms, 1),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "tokens_per_second": round(out_tok / (latency_ms / 1000), 1) if out_tok and latency_ms else 0,
            "verbosity_ratio": round(out_tok / in_tok, 2) if in_tok else 0,
            "refused": False,
            "error": None,
        }
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return {
            "response_text": "",
            "total_latency_ms": round(latency_ms, 1),
            "first_token_latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0,
            "tokens_per_second": 0, "verbosity_ratio": 0,
            "refused": True, "error": str(e),
        }


def _call_judge(client, prompt_text: str, response_text: str,
                ground_truth: str, category: str) -> dict:
    """Call LLM judge. Returns score dict."""
    import openai

    judge_prompt = f"""
CATEGORY: {category}

ORIGINAL QUESTION:
{prompt_text}

GROUND TRUTH / EXPECTED ANSWER:
{ground_truth}

MODEL RESPONSE TO EVALUATE:
{response_text}

{RUBRIC}
""".strip()

    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            temperature=JUDGE_TEMPERATURE,
            messages=[
                {"role": "system", "content":
                 "You are a rigorous LLM evaluator. Respond with valid JSON only."},
                {"role": "user", "content": judge_prompt},
            ],
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

        dims = ["accuracy", "hallucination_resistance", "relevance",
                "instruction_following", "conciseness", "task_completion"]
        scores = {d: int(data.get(d, 3)) for d in dims}

        # Value index
        w = VALUE_WEIGHTS
        weighted = sum(scores[d] * w.get(d, 0) for d in dims)
        out_tok = 250  # approximate for value index calc
        verbosity_penalty = max(1.0, (out_tok / 150) / 3.0)
        value_index = round(weighted / verbosity_penalty, 3)

        composite = round(sum(scores.values()) / len(dims), 3)

        return {
            **scores,
            "composite_score": float(data.get("composite_score", composite)),
            "value_index": value_index,
            "useful_density": round((composite / max(out_tok, 1)) * 100, 3),
            "consistency_score": 0.85,  # single-run estimate
            "rationale": data.get("rationale", ""),
            "notable_issues": data.get("notable_issues", "none"),
            "one_line_verdict": data.get("one_line_verdict", ""),
            "judge_error": None,
        }
    except Exception as e:
        return {
            "accuracy": 0, "hallucination_resistance": 0, "relevance": 0,
            "instruction_following": 0, "conciseness": 0, "task_completion": 0,
            "composite_score": 0.0, "value_index": 0.0, "useful_density": 0.0,
            "consistency_score": 0.0, "rationale": "", "notable_issues": "",
            "one_line_verdict": "", "judge_error": str(e),
        }


def _append_to_csv(row: dict, path: Path, key_cols: list[str]):
    """Append one row to a CSV, replacing if (model, prompt_id) already exists."""
    new_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        # Drop existing row for same model+prompt
        mask = pd.Series([True] * len(existing))
        for col in key_cols:
            if col in existing.columns:
                mask &= existing[col] == row.get(col, "")
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def run_model_live(
    model: str,
    api_key: str,
    prompts: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """
    Generator that evaluates one model against all prompts.
    Yields a dict for each step so the dashboard can update in real time.

    Yield shapes:
        {"type": "start",    "model": str, "total": int}
        {"type": "cached",   "model": str, "prompt_id": str, "idx": int,
                             "total": int, "response_row": dict, "score_row": dict}
        {"type": "progress", "model": str, "prompt_id": str, "idx": int,
                             "total": int, "stage": "inference"|"judging"}
        {"type": "result",   "model": str, "prompt_id": str, "idx": int,
                             "total": int, "response_row": dict, "score_row": dict}
        {"type": "done",     "model": str, "total": int}
        {"type": "error",    "model": str, "message": str}
    """
    import openai

    if not api_key:
        yield {"type": "error", "model": model, "message": "API key not set."}
        return

    if prompts is None:
        if not PROMPTS_FILE.exists():
            yield {"type": "error", "model": model, "message": "prompts.json not found."}
            return
        prompts = json.loads(PROMPTS_FILE.read_text())

    client = openai.OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
    )

    provider = model.split("/")[0] if "/" in model else "unknown"
    total = len(prompts)
    yield {"type": "start", "model": model, "total": total}

    for idx, prompt in enumerate(prompts):
        pid = prompt["id"]

        # ── Check cache first ─────────────────────────────────────────────
        cached_r, cached_s = already_evaluated(model, pid)
        if cached_r and cached_s:
            yield {
                "type": "cached",
                "model": model, "prompt_id": pid,
                "idx": idx, "total": total,
                "response_row": cached_r,
                "score_row": cached_s,
            }
            continue

        # ── Inference ─────────────────────────────────────────────────────
        yield {"type": "progress", "model": model, "prompt_id": pid,
               "idx": idx, "total": total, "stage": "inference"}

        r = _call_model(client, model, prompt["prompt"])

        response_row = {
            "model": model,
            "provider": provider,
            "prompt_id": pid,
            "category": prompt.get("category", ""),
            "prompt_text": prompt["prompt"],
            "response_text": r["response_text"],
            **{k: r[k] for k in ["total_latency_ms", "first_token_latency_ms",
                                   "input_tokens", "output_tokens",
                                   "tokens_per_second", "verbosity_ratio",
                                   "refused", "error"]},
        }

        # ── Judge scoring ─────────────────────────────────────────────────
        yield {"type": "progress", "model": model, "prompt_id": pid,
               "idx": idx, "total": total, "stage": "judging"}

        s = _call_judge(
            client,
            prompt_text=prompt["prompt"],
            response_text=r["response_text"],
            ground_truth=prompt.get("ground_truth", ""),
            category=prompt.get("category", "general"),
        )

        score_row = {
            "model": model,
            "prompt_id": pid,
            **s,
        }

        # ── Persist ───────────────────────────────────────────────────────
        _append_to_csv(response_row, RESULTS_FILE, ["model", "prompt_id"])
        _append_to_csv(score_row,    SCORES_FILE,  ["model", "prompt_id"])

        yield {
            "type": "result",
            "model": model, "prompt_id": pid,
            "idx": idx, "total": total,
            "response_row": response_row,
            "score_row": score_row,
        }

    yield {"type": "done", "model": model, "total": total}


def build_efficiency_row(model: str, responses: list[dict], scores: list[dict]) -> dict:
    """
    Compute the efficiency summary row for a model from its raw results.
    Same shape as efficiency_summary.csv so the dashboard can merge seamlessly.
    """
    from config import API_PRICING, PRO_PLAN_LIMITS

    r_df = pd.DataFrame(responses)
    s_df = pd.DataFrame(scores)

    avg_in  = float(r_df["input_tokens"].mean())  if not r_df.empty else 0
    avg_out = float(r_df["output_tokens"].mean()) if not r_df.empty else 0

    pricing = API_PRICING.get(model, {"input": 1.0, "output": 3.0})
    cost_per = (avg_in / 1e6 * pricing.get("input", 0)) + \
               (avg_out / 1e6 * pricing.get("output", 0))
    api_ans = int(20.0 / cost_per) if cost_per > 0 else 999_999

    plan = PRO_PLAN_LIMITS.get(model, {})
    provider_parts = model.split("/")
    provider_raw = provider_parts[0] if len(provider_parts) > 1 else "unknown"

    from config import PROVIDER_META
    provider_display = PROVIDER_META.get(provider_raw, {}).get("name", provider_raw.title())

    return {
        "model": model,
        "provider": provider_display,
        "plan": plan.get("plan", "API only"),
        "plan_cost_usd": plan.get("$/mo", 0),
        "daily_msg_limit": plan.get("daily_msgs", 999),
        "composite_score": round(s_df["composite_score"].mean(), 2) if not s_df.empty else 0,
        "value_index": round(s_df["value_index"].mean(), 3) if "value_index" in s_df else 0,
        "consistency_score": round(s_df["consistency_score"].mean(), 3) if "consistency_score" in s_df else 0,
        "task_completion_avg": round(s_df["task_completion"].mean(), 2) if "task_completion" in s_df else 0,
        "refusal_rate_pct": round(float(r_df["refused"].mean()) * 100, 1) if not r_df.empty else 0,
        "avg_total_latency_ms": round(float(r_df["total_latency_ms"].mean()), 0) if not r_df.empty else 0,
        "avg_first_token_ms": round(float(r_df["first_token_latency_ms"].mean()), 0) if not r_df.empty else 0,
        "avg_tokens_per_sec": round(float(r_df["tokens_per_second"].mean()), 1) if not r_df.empty else 0,
        "avg_input_tokens": round(avg_in, 0),
        "avg_output_tokens": round(avg_out, 0),
        "avg_verbosity_ratio": round(float(r_df["verbosity_ratio"].mean()), 2) if not r_df.empty else 0,
        "useful_density": round(s_df["useful_density"].mean(), 4) if "useful_density" in s_df else 0,
        "api_cost_per_answer_usd": round(cost_per, 6),
        "api_answers_for_20usd": api_ans,
        "input_price_per_1m": pricing.get("input", 0),
        "output_price_per_1m": pricing.get("output", 0),
    }
