"""
reporter.py — Report generation for the LLM Evaluation Framework.
Generates Markdown summary reports and Pandas summary stats.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import REPORTS_DIR, REPORT_TITLE, AUTHOR


def generate_summary_stats(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with mean scores per model."""
    score_cols = [
        "accuracy",
        "hallucination_resistance",
        "relevance",
        "instruction_following",
        "composite_score",
    ]
    available = [c for c in score_cols if c in scores_df.columns]
    summary = (
        scores_df[scores_df["judge_error"].isna()]
        .groupby("model")[available]
        .mean()
        .round(2)
        .sort_values("composite_score", ascending=False)
    )
    return summary


def generate_category_breakdown(scores_df: pd.DataFrame, responses_df: pd.DataFrame) -> pd.DataFrame:
    """Break down composite scores by model × category."""
    if responses_df.empty or scores_df.empty:
        return pd.DataFrame()

    # Join scores with category info from responses
    merged = scores_df.merge(
        responses_df[["prompt_id", "model"]].drop_duplicates(),
        on=["prompt_id", "model"],
        how="left",
    )
    # We need category — load from prompts if available, else skip
    return merged


def _model_section(model: str, scores_df: pd.DataFrame, responses_df: pd.DataFrame) -> str:
    """Build a markdown section for a single model."""
    model_scores = scores_df[scores_df["model"] == model]
    model_responses = responses_df[responses_df["model"] == model]

    if model_scores.empty:
        return f"### {model}\n\n_No scores available._\n\n"

    valid = model_scores[model_scores["judge_error"].isna()]

    lines = [
        f"### {model}",
        "",
        f"**Prompts evaluated:** {len(valid)} / {len(model_responses)}  ",
        f"**Mean composite score:** {valid['composite_score'].mean():.2f} / 5.00  ",
        f"**Mean accuracy:** {valid['accuracy'].mean():.2f}  ",
        f"**Mean hallucination resistance:** {valid['hallucination_resistance'].mean():.2f}  ",
        f"**Mean relevance:** {valid['relevance'].mean():.2f}  ",
        f"**Mean instruction following:** {valid['instruction_following'].mean():.2f}  ",
        "",
    ]

    # Latency
    if "latency_ms" in model_responses.columns:
        avg_lat = model_responses["latency_ms"].mean()
        lines.append(f"**Avg latency:** {avg_lat:.0f}ms  ")

    # Token usage
    if "output_tokens" in model_responses.columns:
        avg_tokens = model_responses["output_tokens"].mean()
        lines.append(f"**Avg output tokens:** {avg_tokens:.0f}  ")

    lines.append("")

    # Per-prompt details
    lines.append("**Per-prompt scores:**")
    lines.append("")
    lines.append("| Prompt ID | Category | Accuracy | Halluc. | Relevance | Composite | Issues |")
    lines.append("|-----------|----------|----------|---------|-----------|-----------|--------|")

    for _, row in valid.iterrows():
        issues = str(row.get("notable_issues", "none"))
        if len(issues) > 40:
            issues = issues[:37] + "..."
        lines.append(
            f"| {row['prompt_id']} | — | "
            f"{row['accuracy']} | {row['hallucination_resistance']} | "
            f"{row['relevance']} | {row['composite_score']:.2f} | {issues} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_markdown_report(
    responses_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    prompts: list[dict],
) -> Path:
    """Generate a full Markdown evaluation report and save it to reports/."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_report_{timestamp}.md"

    summary = generate_summary_stats(scores_df)
    models = scores_df["model"].unique().tolist() if not scores_df.empty else []
    categories = list({p["category"] for p in prompts})

    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Author:** {AUTHOR}  ",
        f"**Models evaluated:** {', '.join(models)}  ",
        f"**Total prompts:** {len(prompts)}  ",
        f"**Categories:** {', '.join(categories)}  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "### Leaderboard (composite score, descending)",
        "",
        "| Rank | Model | Composite | Accuracy | Halluc. Resistance | Relevance |",
        "|------|-------|-----------|----------|-------------------|-----------|",
    ]

    for rank, (model, row) in enumerate(summary.iterrows(), 1):
        lines.append(
            f"| {rank} | `{model}` | **{row.get('composite_score', 0):.2f}** | "
            f"{row.get('accuracy', 0):.2f} | {row.get('hallucination_resistance', 0):.2f} | "
            f"{row.get('relevance', 0):.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Each prompt was run through all configured models with a fixed system prompt and temperature (0.3).",
        "Responses were then evaluated by an LLM judge (Claude) using a structured 4-dimension rubric:",
        "",
        "- **Accuracy (1–5):** Correctness relative to ground truth",
        "- **Hallucination Resistance (1–5):** Absence of fabricated facts",
        "- **Relevance (1–5):** How directly the response addresses the question",
        "- **Instruction Following (1–5):** Adherence to any formatting or structural instructions",
        "",
        "**Composite score** = mean of the four dimensions.",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
    ]

    for model in models:
        lines.append(_model_section(model, scores_df, responses_df))

    lines += [
        "---",
        "",
        "## Notable Findings",
        "",
        "_(Auto-generated — review and edit before sharing)_",
        "",
    ]

    # Auto-flag hallucination test results
    hall_scores = scores_df[
        (scores_df["prompt_id"].str.contains("hallucination")) &
        scores_df["judge_error"].isna()
    ]
    if not hall_scores.empty:
        lines.append("### Hallucination Test Results")
        lines.append("")
        for _, row in hall_scores.iterrows():
            flag = "✅" if row["hallucination_resistance"] >= 4 else "⚠️"
            lines.append(
                f"- {flag} **{row['model']}** on `{row['prompt_id']}`: "
                f"hallucination_resistance = {row['hallucination_resistance']}/5"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "_Report generated by LLM Evaluation Framework — github.com/yourusername/llm-eval-framework_",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
