"""
generate_demo_data.py — Generates realistic synthetic results for testing
the dashboard without requiring API keys. Includes all efficiency metrics.

Run: python generate_demo_data.py
Then: streamlit run dashboard.py
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    RESULTS_FILE, SCORES_FILE, PROMPTS_FILE, RESULTS_DIR,
    API_PRICING, PRO_PLAN_LIMITS, VALUE_WEIGHTS,
)

random.seed(42)
np.random.seed(42)

MODELS = [
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
]

# ── Realistic model profiles ──────────────────────────────────────────────────
PROFILES = {
    "anthropic/claude-sonnet-4-5": {
        "accuracy":                (4.3, 0.5),
        "hallucination_resistance": (4.7, 0.4),
        "relevance":               (4.4, 0.5),
        "instruction_following":   (4.6, 0.4),
        "conciseness":             (3.7, 0.7),
        "task_completion":         (4.6, 0.4),
        "total_latency_ms":        (2100, 450),
        "first_token_latency_ms":  (580, 140),
        "input_tokens":            (120, 38),
        "output_tokens":           (285, 75),
        "consistency_std":         (0.13, 0.04),
        "refusal_prob":            0.03,
    },
    "openai/gpt-4o-mini": {
        "accuracy":                (3.9, 0.7),
        "hallucination_resistance": (3.8, 0.8),
        "relevance":               (4.0, 0.6),
        "instruction_following":   (4.1, 0.6),
        "conciseness":             (4.2, 0.6),
        "task_completion":         (3.9, 0.7),
        "total_latency_ms":        (970, 200),
        "first_token_latency_ms":  (310, 75),
        "input_tokens":            (110, 32),
        "output_tokens":           (190, 65),
        "consistency_std":         (0.22, 0.07),
        "refusal_prob":            0.07,
    },
    "google/gemini-2.0-flash-001": {
        "accuracy":                (4.1, 0.6),
        "hallucination_resistance": (4.0, 0.7),
        "relevance":               (4.2, 0.5),
        "instruction_following":   (4.0, 0.6),
        "conciseness":             (4.3, 0.5),
        "task_completion":         (4.2, 0.5),
        "total_latency_ms":        (780, 180),
        "first_token_latency_ms":  (250, 60),
        "input_tokens":            (105, 30),
        "output_tokens":           (200, 60),
        "consistency_std":         (0.18, 0.06),
        "refusal_prob":            0.05,
    },
    "deepseek/deepseek-chat": {
        "accuracy":                (4.0, 0.7),
        "hallucination_resistance": (3.9, 0.7),
        "relevance":               (4.1, 0.6),
        "instruction_following":   (3.9, 0.7),
        "conciseness":             (4.0, 0.6),
        "task_completion":         (4.0, 0.6),
        "total_latency_ms":        (1400, 350),
        "first_token_latency_ms":  (420, 110),
        "input_tokens":            (115, 35),
        "output_tokens":           (230, 70),
        "consistency_std":         (0.20, 0.06),
        "refusal_prob":            0.06,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "accuracy":                (3.8, 0.7),
        "hallucination_resistance": (3.6, 0.8),
        "relevance":               (3.9, 0.7),
        "instruction_following":   (3.8, 0.7),
        "conciseness":             (3.9, 0.7),
        "task_completion":         (3.8, 0.7),
        "total_latency_ms":        (1100, 280),
        "first_token_latency_ms":  (350, 90),
        "input_tokens":            (108, 33),
        "output_tokens":           (210, 65),
        "consistency_std":         (0.25, 0.08),
        "refusal_prob":            0.08,
    },
}

SAMPLE_RESPONSES = {
    "factual": "Canberra is the capital of Australia, with a population of approximately 460,000 people. It was purpose-built as the capital following a compromise between Sydney and Melbourne.",
    "reasoning": "Time = Distance / Speed = 150 miles / 60 mph = 2.5 hours (2 hours and 30 minutes).",
    "coding": "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return [seen[complement], i]\n        seen[n] = i",
    "hallucination_test": "No such treaty exists. I'm not aware of a 'Copenhagen AI Safety Treaty' from 2019. Rather than speculate, I'd recommend checking official UN or EU treaty databases.",
    "summarization": "Machine learning is an AI approach that uses statistical methods to let computers learn from data and improve at tasks without explicit per-task programming.",
    "instruction_following": "• Improves cardiovascular health and reduces heart disease risk\n• Boosts mood and reduces anxiety via endorphin release\n• Builds muscle strength and increases bone density",
}


def clip_score(val: float) -> int:
    return int(np.clip(round(val), 1, 5))


def sample(profile: dict, key: str) -> float:
    mu, sigma = profile[key]
    return np.random.normal(mu, sigma)


def generate_responses(prompts: list) -> pd.DataFrame:
    rows = []
    for prompt in prompts:
        for model in MODELS:
            p = PROFILES[model]
            refused = random.random() < p["refusal_prob"]
            total_lat = max(300.0, sample(p, "total_latency_ms"))
            first_tok = max(100.0, min(total_lat * 0.4, sample(p, "first_token_latency_ms")))
            in_tok = max(30, int(sample(p, "input_tokens")))
            out_tok = 0 if refused else max(20, int(sample(p, "output_tokens")))
            tps = round(out_tok / (total_lat / 1000), 1) if out_tok > 0 else 0.0

            rows.append({
                "model": model,
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "prompt_text": prompt["prompt"],
                "response_text": "" if refused else SAMPLE_RESPONSES.get(prompt["category"], "Sample response."),
                "total_latency_ms": round(total_lat, 1),
                "first_token_latency_ms": round(first_tok, 1),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "tokens_per_second": tps,
                "verbosity_ratio": round(out_tok / in_tok, 2) if in_tok > 0 else 0.0,
                "refused": refused,
                "error": "refused" if refused else None,
            })
    return pd.DataFrame(rows)


def generate_scores(prompts: list, responses_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    verdicts = {
        "anthropic/claude-sonnet-4-5": [
            "Accurate and thorough — strong choice for complex, high-stakes tasks.",
            "No fabricated facts; completes task fully with clear reasoning.",
            "High-quality output; slightly verbose but all content earns its place.",
        ],
        "openai/gpt-4o-mini": [
            "Fast and concise — solid for everyday straightforward queries.",
            "Mostly accurate with minor imprecision; good value for simple tasks.",
            "Efficient output but hedges slightly on nuanced edge cases.",
        ],
        "google/gemini-2.0-flash-001": [
            "Very fast first-token latency and consistently concise responses.",
            "Good accuracy on factual tasks; strong instruction following.",
            "Excellent cost-efficiency — one of the best value-per-token models.",
        ],
        "deepseek/deepseek-chat": [
            "Strong reasoning at a fraction of frontier pricing.",
            "Competitive quality on coding and analysis tasks.",
            "Good all-rounder with very low cost per answer.",
        ],
        "meta-llama/llama-3.3-70b-instruct": [
            "Open-weight model with solid general capability.",
            "Competitive on reasoning; occasionally needs more guidance.",
            "Good budget option for volume tasks.",
        ],
    }

    for prompt in prompts:
        for model in MODELS:
            p = PROFILES[model]
            resp_row = responses_df[
                (responses_df["prompt_id"] == prompt["id"]) &
                (responses_df["model"] == model)
            ]
            refused = bool(resp_row.iloc[0]["refused"]) if not resp_row.empty else False

            if refused:
                rows.append({
                    "prompt_id": prompt["id"], "model": model,
                    "accuracy": 1, "hallucination_resistance": 5,
                    "relevance": 1, "instruction_following": 1,
                    "conciseness": 5, "task_completion": 1,
                    "composite_score": 2.33,
                    "value_index": 1.0, "useful_density": 0.0,
                    "consistency_score": 0.9,
                    "rationale": "Model refused or excessively hedged without completing the task.",
                    "notable_issues": "refusal/over-hedging",
                    "one_line_verdict": "Refused to answer — not useful for this task.",
                    "judge_error": None,
                })
                continue

            acc   = clip_score(sample(p, "accuracy"))
            hall  = clip_score(sample(p, "hallucination_resistance"))
            rel   = clip_score(sample(p, "relevance"))
            instr = clip_score(sample(p, "instruction_following"))
            conc  = clip_score(sample(p, "conciseness"))
            comp_task = clip_score(sample(p, "task_completion"))

            if "hallucination" in prompt["id"]:
                hall = random.choice([5, 5, 4]) if "claude" in model or "gemini" in model else random.choice([3, 4, 4])
                acc  = random.choice([5, 4])    if "claude" in model else random.choice([3, 4])

            composite = round(float(np.mean([acc, hall, rel, instr, conc, comp_task])), 3)

            out_toks = int(resp_row.iloc[0]["output_tokens"]) if not resp_row.empty else 200
            in_toks  = int(resp_row.iloc[0]["input_tokens"])  if not resp_row.empty else 100
            verb = out_toks / in_toks if in_toks > 0 else 1.0

            w = VALUE_WEIGHTS
            weighted = (
                acc * w["accuracy"] +
                hall * w["hallucination_resistance"] +
                rel * w["relevance"] +
                comp_task * w["task_completion"] +
                instr * w["instruction_following"] +
                conc * w["conciseness"]
            )
            verbosity_penalty = max(1.0, verb / 3.0)
            value_index = round(weighted / verbosity_penalty, 3)
            useful_density = round((composite / max(out_toks, 1)) * 100, 3)
            consistency = round(max(0.0, 1.0 - float(np.random.normal(
                p["consistency_std"][0], p["consistency_std"][1]))), 3)

            rows.append({
                "prompt_id": prompt["id"], "model": model,
                "accuracy": acc, "hallucination_resistance": hall,
                "relevance": rel, "instruction_following": instr,
                "conciseness": conc, "task_completion": comp_task,
                "composite_score": composite,
                "value_index": value_index,
                "useful_density": useful_density,
                "consistency_score": consistency,
                "rationale": "Scores reflect quality against ground truth and task requirements.",
                "notable_issues": "none" if random.random() > 0.25 else random.choice([
                    "minor verbosity", "slight hedge", "imprecise wording"
                ]),
                "one_line_verdict": random.choice(verdicts[model]),
                "judge_error": None,
            })
    return pd.DataFrame(rows)


def compute_efficiency_summary(responses_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    merged = responses_df.merge(
        scores_df[["prompt_id", "model", "composite_score", "value_index",
                   "useful_density", "consistency_score", "task_completion"]],
        on=["prompt_id", "model"], how="left"
    )

    # Real OpenRouter pricing ($/1M tokens, April 2025)
    OR_PRICING = {
        "anthropic/claude-sonnet-4-5":          {"input": 3.00,  "output": 15.00, "provider": "Anthropic",  "plan": "Claude Pro",    "plan_cost": 20, "daily_msgs": 100},
        "openai/gpt-4o-mini":                    {"input": 0.15,  "output": 0.60,  "provider": "OpenAI",     "plan": "ChatGPT Plus",  "plan_cost": 20, "daily_msgs": 999},
        "google/gemini-2.0-flash-001":           {"input": 0.10,  "output": 0.40,  "provider": "Google",     "plan": "Gemini Advanced","plan_cost": 20, "daily_msgs": 500},
        "deepseek/deepseek-chat":                {"input": 0.27,  "output": 1.10,  "provider": "DeepSeek",   "plan": "DeepSeek API",  "plan_cost": 0,  "daily_msgs": 999},
        "meta-llama/llama-3.3-70b-instruct":    {"input": 0.12,  "output": 0.30,  "provider": "Meta/Llama", "plan": "API only",      "plan_cost": 0,  "daily_msgs": 999},
    }

    rows = []
    for model in MODELS:
        m = merged[merged["model"] == model]
        pricing = OR_PRICING.get(model, {"input": 1.0, "output": 3.0, "provider": model.split("/")[0],
                                          "plan": "API only", "plan_cost": 0, "daily_msgs": 999})

        avg_in  = float(m["input_tokens"].mean())
        avg_out = float(m["output_tokens"].mean())
        cost_per_answer = (avg_in / 1e6 * pricing["input"]) + (avg_out / 1e6 * pricing["output"])
        api_answers = int(20.0 / cost_per_answer) if cost_per_answer > 0 else 999_999

        rows.append({
            "model": model,
            "provider": pricing["provider"],
            "plan": pricing["plan"],
            "plan_cost_usd": pricing["plan_cost"],
            "daily_msg_limit": pricing["daily_msgs"],
            "composite_score": round(float(m["composite_score"].mean()), 2),
            "value_index": round(float(m["value_index"].mean()), 3),
            "consistency_score": round(float(m["consistency_score"].mean()), 3),
            "task_completion_avg": round(float(m["task_completion"].mean()), 2),
            "refusal_rate_pct": round(float(m["refused"].mean()) * 100, 1),
            "avg_total_latency_ms": round(float(m["total_latency_ms"].mean()), 0),
            "avg_first_token_ms": round(float(m["first_token_latency_ms"].mean()), 0),
            "avg_tokens_per_sec": round(float(m["tokens_per_second"].mean()), 1),
            "avg_input_tokens": round(avg_in, 0),
            "avg_output_tokens": round(avg_out, 0),
            "avg_verbosity_ratio": round(float(m["verbosity_ratio"].mean()), 2),
            "useful_density": round(float(m["useful_density"].mean()), 4),
            "api_cost_per_answer_usd": round(cost_per_answer, 6),
            "api_answers_for_20usd": api_answers,
            "input_price_per_1m": pricing["input"],
            "output_price_per_1m": pricing["output"],
        })
    return pd.DataFrame(rows)


def main():
    print("Generating enhanced demo data with efficiency metrics...")
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    responses_df = generate_responses(prompts)
    scores_df    = generate_scores(prompts, responses_df)
    summary_df   = compute_efficiency_summary(responses_df, scores_df)

    responses_df.to_csv(RESULTS_FILE, index=False)
    scores_df.to_csv(SCORES_FILE, index=False)
    summary_df.to_csv(RESULTS_DIR / "efficiency_summary.csv", index=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_df.to_csv(RESULTS_DIR / f"responses_{ts}.csv", index=False)
    scores_df.to_csv(RESULTS_DIR / f"scores_{ts}.csv", index=False)

    print(f"✓ {len(responses_df)} responses | {len(scores_df)} scores | {len(summary_df)} model summaries\n")
    cols = ["model", "composite_score", "value_index", "avg_total_latency_ms",
            "avg_tokens_per_sec", "avg_output_tokens", "refusal_rate_pct", "api_answers_for_20usd"]
    pd.set_option("display.width", 120)
    print(summary_df[cols].to_string(index=False))
    print("\nRun: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
