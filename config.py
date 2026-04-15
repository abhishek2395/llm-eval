"""
config.py — Central configuration for the LLM Evaluation Framework.
Edit model lists, scoring rubric, Pro plan constants, and output paths here.
"""

import os
from pathlib import Path

# ── Load .env automatically ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # export OPENROUTER_API_KEY manually if dotenv not installed

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

PROMPTS_FILE = DATA_DIR / "prompts.json"
RESULTS_FILE = RESULTS_DIR / "results.csv"
SCORES_FILE = RESULTS_DIR / "scores.csv"

# ── Models ───────────────────────────────────────────────────────────────────
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    # "claude-opus-4-20250514",
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    # "gpt-4o",
]

ALL_MODELS = CLAUDE_MODELS + OPENAI_MODELS

JUDGE_MODEL = "anthropic/claude-sonnet-4-5"  # OpenRouter format: provider/model

# ── API settings ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# OpenRouter endpoint — OpenAI-compatible, one key for 300+ models
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Headers OpenRouter recommends for ranking/attribution
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/abhishek/llm-eval-framework",
    "X-Title": "LLM Eval Framework",
}

MAX_TOKENS = 1024
TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.1

# Number of repeated runs per prompt for consistency scoring (set to 1 to skip)
CONSISTENCY_RUNS = 3

# ── OpenRouter curated presets ────────────────────────────────────────────────
# These are the model IDs as they appear on OpenRouter.
# Used as defaults when user hasn't specified models.
# Full catalog: https://openrouter.ai/models
OPENROUTER_PRESET_MODELS = {
    "frontier": [
        "anthropic/claude-sonnet-4-5",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-chat",
        "x-ai/grok-3",
    ],
    "budget": [
        "anthropic/claude-haiku-4-5",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "reasoning": [
        "anthropic/claude-sonnet-4-5",
        "openai/o4-mini",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-r1",
    ],
    "open_source": [
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large",
        "qwen/qwen-2.5-72b-instruct",
        "deepseek/deepseek-chat",
    ],
}

# Provider display names + brand colors for the dashboard
PROVIDER_META = {
    "anthropic":  {"name": "Anthropic",  "color": "#d97706"},
    "openai":     {"name": "OpenAI",      "color": "#10b981"},
    "google":     {"name": "Google",      "color": "#3b82f6"},
    "deepseek":   {"name": "DeepSeek",    "color": "#8b5cf6"},
    "meta-llama": {"name": "Meta",        "color": "#f97316"},
    "mistralai":  {"name": "Mistral",     "color": "#ec4899"},
    "x-ai":       {"name": "xAI / Grok",  "color": "#06b6d4"},
    "qwen":       {"name": "Alibaba",     "color": "#ef4444"},
    "cohere":     {"name": "Cohere",      "color": "#84cc16"},
    "perplexity": {"name": "Perplexity",  "color": "#a78bfa"},
}

# ── Pro Plan Pricing & Limits ─────────────────────────────────────────────────
# Used to compute "value per $20/month" efficiency metrics.
# Update these if plan limits change.
PRO_PLAN_COST_USD = 20.0

PRO_PLAN_LIMITS = {
    # Approximate daily message caps (varies; conservative estimates)
    "claude-sonnet-4-20250514":  {"daily_msgs": 100,  "plan": "Claude Pro",    "$/mo": 20},
    "claude-opus-4-20250514":    {"daily_msgs": 50,   "plan": "Claude Pro",    "$/mo": 20},
    "gpt-4o-mini":               {"daily_msgs": 999,  "plan": "ChatGPT Plus",  "$/mo": 20},
    "gpt-4o":                    {"daily_msgs": 80,   "plan": "ChatGPT Plus",  "$/mo": 20},
}

# API pricing per 1M tokens (for cost-per-answer estimates)
# Source: anthropic.com/pricing and openai.com/pricing (April 2025)
API_PRICING = {
    "claude-sonnet-4-20250514":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4-20250514":    {"input": 15.00, "output": 75.00},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"input": 2.50,  "output": 10.00},
}

# ── Scoring rubric ───────────────────────────────────────────────────────────
SCORE_DIMENSIONS = [
    "accuracy",
    "hallucination_resistance",
    "relevance",
    "instruction_following",
    "conciseness",
    "task_completion",
]

RUBRIC = """
You are an expert LLM evaluator benchmarking models for real-world $20/month plan users.
Score the following response on SIX dimensions. Use the scales exactly as defined.

DIMENSION 1 — ACCURACY (1–5)
  5: Fully correct, all key facts match ground truth
  4: Mostly correct, minor omissions or imprecision
  3: Partially correct — some right, some wrong
  2: Mostly incorrect but contains a correct element
  1: Completely wrong or fabricated

DIMENSION 2 — HALLUCINATION RESISTANCE (1–5)
  5: No hallucinations; correctly declines if premise is false
  4: No fabricated facts; slight overconfidence but no false claims
  3: Minor unsupported claim present
  2: One clear hallucination or fabricated fact
  1: Multiple hallucinations or confidently wrong

DIMENSION 3 — RELEVANCE (1–5)
  5: Directly and completely answers the question asked
  4: Answers the question with minor tangents
  3: Partially answers; partially off-topic
  2: Mostly off-topic or misunderstood the question
  1: Does not address the question at all

DIMENSION 4 — INSTRUCTION FOLLOWING (1–5)
  5: Follows all formatting/structural instructions perfectly (default 5 if none given)
  4: Follows most instructions, minor deviation
  3: Partially follows instructions
  2: Ignores most instructions
  1: Completely ignores instructions

DIMENSION 5 — CONCISENESS (1–5)
  Rate whether the response length is appropriate for the task. Penalize bloat.
  5: Response is exactly as long as needed — no padding, no fluff
  4: Slightly verbose but all content adds value
  3: Noticeable padding, filler phrases, or unnecessary repetition
  2: Response is 2x longer than it needs to be with significant filler
  1: Massively over-generated; buries the answer in noise

DIMENSION 6 — TASK COMPLETION (1–5)
  Did the model actually finish the task, or did it hedge, truncate, or refuse?
  5: Task 100% completed — answer is actionable and self-contained
  4: Task mostly completed; minor gap
  3: Task half-done — answer requires follow-up to be usable
  2: Task barely attempted; mostly disclaimers or hedging
  1: Refused, gave up, or produced an unusable response

Respond ONLY in this exact JSON format — no preamble, no markdown fences:
{
  "accuracy": <int 1-5>,
  "hallucination_resistance": <int 1-5>,
  "relevance": <int 1-5>,
  "instruction_following": <int 1-5>,
  "conciseness": <int 1-5>,
  "task_completion": <int 1-5>,
  "composite_score": <float, average of all six dimensions>,
  "rationale": "<2-3 sentences explaining the scores>",
  "notable_issues": "<specific problems found, or 'none'>",
  "one_line_verdict": "<single sentence a non-technical user would understand>"
}
"""

# ── Value Index weights ───────────────────────────────────────────────────────
# Composite score is a simple mean, but Value Index weights dimensions
# by what matters most for a $20/month subscription decision.
VALUE_WEIGHTS = {
    "accuracy":                0.25,
    "hallucination_resistance": 0.20,
    "relevance":               0.20,
    "task_completion":         0.20,
    "instruction_following":   0.10,
    "conciseness":             0.05,
}

# ── Report settings ───────────────────────────────────────────────────────────
REPORT_TITLE = "LLM Evaluation Report"
AUTHOR = "Abhishek — AI Quality Engineering Portfolio"

# ── Model display aliases ─────────────────────────────────────────────────────
# Maps OpenRouter model IDs → clean display names used everywhere in the UI.
# Add any new model here to give it a custom label.
# If a model ID isn't listed, short_name() falls back to auto-parsing.
MODEL_ALIASES: dict[str, str] = {
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "anthropic/claude-opus-4-5":          "Claude Opus 4.5",
    "anthropic/claude-sonnet-4-5":        "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4-5":         "Claude Haiku 4.5",
    "anthropic/claude-opus-4-20250514":   "Claude Opus 4",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai/gpt-4o":                      "GPT-4o",
    "openai/gpt-4o-mini":                 "GPT-4o Mini",
    "openai/o3":                          "GPT o3",
    "openai/o3-mini":                     "GPT o3-mini",
    "openai/o4-mini":                     "GPT o4-mini",
    "openai/gpt-4.5-preview":             "GPT-4.5",
    # ── Google ────────────────────────────────────────────────────────────────
    "google/gemini-2.5-pro":              "Gemini 2.5 Pro",
    "google/gemini-2.5-flash":            "Gemini 2.5 Flash",
    "google/gemini-2.0-flash-001":        "Gemini 2.0 Flash",
    "google/gemini-flash-1.5":            "Gemini 1.5 Flash",
    "google/gemini-pro-1.5":              "Gemini 1.5 Pro",
    # ── DeepSeek ──────────────────────────────────────────────────────────────
    "deepseek/deepseek-chat":             "DeepSeek V3",
    "deepseek/deepseek-r1":               "DeepSeek R1",
    "deepseek/deepseek-r1-zero":          "DeepSeek R1 Zero",
    # ── Meta / Llama ──────────────────────────────────────────────────────────
    "meta-llama/llama-3.3-70b-instruct":  "Llama 3.3 70B",
    "meta-llama/llama-3.1-405b-instruct": "Llama 3.1 405B",
    "meta-llama/llama-3.1-70b-instruct":  "Llama 3.1 70B",
    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistralai/mistral-large":            "Mistral Large",
    "mistralai/mistral-medium":           "Mistral Medium",
    "mistralai/mixtral-8x22b-instruct":   "Mixtral 8×22B",
    # ── xAI / Grok ────────────────────────────────────────────────────────────
    "x-ai/grok-3":                        "Grok 3",
    "x-ai/grok-3-mini":                   "Grok 3 Mini",
    "x-ai/grok-2-1212":                   "Grok 2",
    # ── Alibaba / Qwen ────────────────────────────────────────────────────────
    "qwen/qwen-2.5-72b-instruct":         "Qwen 2.5 72B",
    "qwen/qwq-32b":                       "QwQ 32B",
    # ── Cohere ────────────────────────────────────────────────────────────────
    "cohere/command-r-plus":              "Command R+",
    "cohere/command-r":                   "Command R",
    # ── Perplexity ────────────────────────────────────────────────────────────
    "perplexity/llama-3.1-sonar-large-128k-online": "Sonar Large",
    "perplexity/llama-3.1-sonar-small-128k-online": "Sonar Small",
}
