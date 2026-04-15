"""
evaluate.py — Main orchestrator for the LLM Evaluation Framework.

Usage:
    python evaluate.py                          # Run all configured models
    python evaluate.py --models claude-sonnet-4-20250514 gpt-4o-mini
    python evaluate.py --prompts data/prompts.json --skip-judge
    python evaluate.py --load-results           # Skip inference, re-score existing results
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    ALL_MODELS, CLAUDE_MODELS, OPENAI_MODELS,
    OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    PROMPTS_FILE, RESULTS_FILE, SCORES_FILE,
    RESULTS_DIR, OPENROUTER_PRESET_MODELS,
)
from openrouter_client import OpenRouterRunner, fetch_model_catalog
from judge import LLMJudge
from reporter import generate_markdown_report, generate_summary_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} prompts from {path}")
    return data


def run_inference(
    prompts: list[dict],
    models: list[str],
    system_prompt: str = "You are a helpful and precise assistant.",
) -> list:
    """
    Run all specified models against all prompts via OpenRouter.
    Falls back to direct Anthropic/OpenAI runners if OPENROUTER_API_KEY is not set.
    Returns list of ModelResponse.
    """
    # ── OpenRouter path (preferred) ───────────────────────────────────────────
    if OPENROUTER_API_KEY:
        logger.info(f"\n── OpenRouter: running {len(models)} models × {len(prompts)} prompts")
        runner = OpenRouterRunner()
        return runner.run_all_models(models, prompts, system_prompt)

    # ── Legacy fallback: direct Anthropic + OpenAI ────────────────────────────
    logger.warning("OPENROUTER_API_KEY not set — falling back to direct provider runners.")
    logger.warning("For multi-provider support, set OPENROUTER_API_KEY instead.")

    all_responses = []
    from runners import ClaudeRunner, OpenAIRunner

    claude_models = [m for m in models if "claude" in m]
    openai_models = [m for m in models if m not in claude_models]

    if claude_models and ANTHROPIC_API_KEY:
        runner = ClaudeRunner()
        for model in claude_models:
            logger.info(f"\nModel: {model}")
            all_responses.extend(runner.run_batch(model, prompts, system_prompt))

    if openai_models and OPENAI_API_KEY:
        runner = OpenAIRunner()
        for model in openai_models:
            logger.info(f"\nModel: {model}")
            all_responses.extend(runner.run_batch(model, prompts, system_prompt))

    return all_responses


def responses_to_dataframe(responses: list) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in responses])


def scores_to_dataframe(scores: list) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in scores])


def save_results(responses_df: pd.DataFrame, scores_df: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save timestamped versions
    resp_path = RESULTS_DIR / f"responses_{timestamp}.csv"
    scores_path = RESULTS_DIR / f"scores_{timestamp}.csv"
    responses_df.to_csv(resp_path, index=False)
    scores_df.to_csv(scores_path, index=False)

    # Also overwrite the "latest" files for the dashboard
    responses_df.to_csv(RESULTS_FILE, index=False)
    scores_df.to_csv(SCORES_FILE, index=False)

    logger.info(f"\nResults saved:")
    logger.info(f"  Responses → {resp_path}")
    logger.info(f"  Scores    → {scores_path}")
    logger.info(f"  Latest    → {RESULTS_FILE}, {SCORES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Framework")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="OpenRouter model IDs to evaluate (e.g. anthropic/claude-sonnet-4-5 openai/gpt-4o)"
    )
    parser.add_argument(
        "--preset", choices=list(OPENROUTER_PRESET_MODELS.keys()), default=None,
        help="Use a curated model preset: frontier | budget | reasoning | open_source"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available OpenRouter models and exit"
    )
    parser.add_argument(
        "--list-providers", type=str, default=None, metavar="PROVIDER",
        help="List all models for a specific provider (e.g. --list-providers google)"
    )
    parser.add_argument(
        "--prompts", type=Path, default=PROMPTS_FILE,
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Run inference only, skip LLM-as-judge scoring"
    )
    parser.add_argument(
        "--load-results", type=Path, default=None,
        help="Skip inference, load existing responses CSV and re-score"
    )
    parser.add_argument(
        "--system-prompt", type=str,
        default="You are a helpful and precise assistant.",
        help="System prompt to use for all model calls"
    )
    args = parser.parse_args()

    # ── --list-models / --list-providers ──────────────────────────────────────
    if args.list_models or args.list_providers:
        catalog = fetch_model_catalog()
        models_to_show = (catalog.by_provider(args.list_providers)
                          if args.list_providers else catalog.all())
        print(f"\n{'ID':<52} {'Provider':<14} {'Input $/1M':>10} {'Out $/1M':>10} {'Context':>10}")
        print("─" * 100)
        for m in models_to_show[:100]:  # cap at 100 for readability
            free_tag = " [FREE]" if m.is_free else ""
            print(f"{m.id:<52} {m.provider_display:<14} "
                  f"{m.input_price_per_1m:>10.3f} {m.output_price_per_1m:>10.3f} "
                  f"{m.context_length:>10,}{free_tag}")
        print(f"\nShowing {min(100, len(models_to_show))} of {len(models_to_show)} models.")
        print("Full catalog: https://openrouter.ai/models")
        sys.exit(0)

    # ── Resolve model list ────────────────────────────────────────────────────
    if args.models:
        selected_models = args.models
    elif args.preset:
        selected_models = OPENROUTER_PRESET_MODELS[args.preset]
        logger.info(f"Using preset '{args.preset}': {selected_models}")
    else:
        # Default: 2 models for quick demo
        selected_models = [
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o-mini",
        ]
        logger.info(f"No --models or --preset specified. Using defaults: {selected_models}")

    logger.info("=" * 60)
    logger.info("  LLM EVALUATION FRAMEWORK")
    logger.info("=" * 60)

    prompts = load_prompts(args.prompts)
    prompts_lookup = {p["id"]: p for p in prompts}

    # ── Step 1: Inference ──────────────────────────────────────────────────
    if args.load_results:
        logger.info(f"\nLoading existing responses from {args.load_results}...")
        responses_df = pd.read_csv(args.load_results)
        from openrouter_client import ModelResponse
        responses = [
            ModelResponse(
                model=row["model"],
                prompt_id=row["prompt_id"],
                prompt_text=row["prompt_text"],
                response_text=row["response_text"],
                latency_ms=row.get("latency_ms", 0),
                error=row.get("error") if pd.notna(row.get("error")) else None,
            )
            for _, row in responses_df.iterrows()
        ]
    else:
        logger.info(f"\nRunning inference on {len(prompts)} prompts × {len(selected_models)} models...")
        responses = run_inference(prompts, selected_models, args.system_prompt)
        responses_df = responses_to_dataframe(responses)

    if responses_df.empty:
        logger.error("No responses collected. Check API keys and model availability.")
        sys.exit(1)

    logger.info(f"\nTotal responses: {len(responses_df)}")
    success_rate = responses_df["error"].isna().mean() * 100
    logger.info(f"Success rate: {success_rate:.1f}%")

    # ── Step 2: Judge scoring ─────────────────────────────────────────────
    scores_df = pd.DataFrame()

    if not args.skip_judge:
        logger.info("\n── Running LLM-as-judge evaluation...")
        judge = LLMJudge()
        successful_responses = [r for r in responses if r.success]
        scores = judge.score_batch(successful_responses, prompts_lookup)
        scores_df = scores_to_dataframe(scores)

        # Print quick summary
        if not scores_df.empty:
            logger.info("\n── Score Summary (mean per model):")
            summary = generate_summary_stats(scores_df)
            print(summary.to_string())

    # ── Step 3: Save ──────────────────────────────────────────────────────
    save_results(responses_df, scores_df)

    # ── Step 4: Markdown report ───────────────────────────────────────────
    if not scores_df.empty:
        report_path = generate_markdown_report(
            responses_df=responses_df,
            scores_df=scores_df,
            prompts=prompts,
        )
        logger.info(f"\nMarkdown report → {report_path}")

    logger.info("\nDone. Run `streamlit run dashboard.py` to explore results.")


if __name__ == "__main__":
    main()
