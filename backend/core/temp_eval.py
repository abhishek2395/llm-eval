"""
temp_eval.py — temperature sensitivity runs.

Runs the same prompts at multiple temperatures (default 0.0 / 0.3 / 0.7),
judges each response, and measures how much a model's quality swings with
sampling temperature. High variance = unpredictable under temperature.

Persistence: results/temp_runs.csv keyed (model, prompt_id, temperature).
Generator events mirror live_eval.py for the same SSE bridge.
"""

from __future__ import annotations

import json
from typing import Generator

import pandas as pd

from config import (
    OPENROUTER_BASE_URL, OPENROUTER_HEADERS, PROMPTS_FILE, RESULTS_DIR,
)
from live_eval import _append_to_csv, _call_judge, _call_model

TEMP_RUNS_FILE = RESULTS_DIR / "temp_runs.csv"
DEFAULT_TEMPS = [0.0, 0.3, 0.7]


def default_prompts() -> list[dict]:
    """One prompt per category — keeps a full sweep affordable."""
    if not PROMPTS_FILE.exists():
        return []
    prompts = json.loads(PROMPTS_FILE.read_text())
    seen: set[str] = set()
    picked = []
    for p in prompts:
        if p["category"] not in seen:
            seen.add(p["category"])
            picked.append(p)
    return picked


def _cached(model: str, prompt_id: str, temp: float) -> dict | None:
    if not TEMP_RUNS_FILE.exists():
        return None
    try:
        df = pd.read_csv(TEMP_RUNS_FILE)
        row = df[(df["model"] == model) & (df["prompt_id"] == prompt_id)
                 & (df["temperature"] == temp)]
        if row.empty:
            return None
        d = row.iloc[0].to_dict()
        if (isinstance(d.get("error"), str) and d["error"]) or \
           (isinstance(d.get("judge_error"), str) and d["judge_error"]):
            return None  # retry failures
        return d
    except Exception:
        return None


def run_temp_sensitivity(
    model: str,
    api_key: str,
    temps: list[float] | None = None,
    prompts: list[dict] | None = None,
    judge_model: str | None = None,
) -> Generator[dict, None, None]:
    import openai

    temps = temps or DEFAULT_TEMPS
    prompts = prompts if prompts is not None else default_prompts()
    if not prompts:
        yield {"type": "error", "model": model, "message": "No prompts available."}
        return

    client = openai.OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=OPENROUTER_HEADERS,
    )

    combos = [(p, t) for p in prompts for t in temps]
    total = len(combos)
    yield {"type": "start", "model": model, "total": total,
           "temps": temps, "prompts": [p["id"] for p in prompts]}

    for idx, (prompt, temp) in enumerate(combos):
        pid = prompt["id"]
        cached = _cached(model, pid, temp)
        if cached:
            yield {"type": "cached", "model": model, "prompt_id": pid,
                   "temperature": temp, "idx": idx, "total": total, "row": cached}
            continue

        yield {"type": "progress", "model": model, "prompt_id": pid,
               "temperature": temp, "idx": idx, "total": total, "stage": "inference"}
        r = _call_model(client, model, prompt["prompt"], temperature=temp)

        yield {"type": "progress", "model": model, "prompt_id": pid,
               "temperature": temp, "idx": idx, "total": total, "stage": "judging"}
        if r["error"]:
            s = {"composite_score": 0.0, "judge_error": None}
        else:
            s = _call_judge(
                client,
                prompt_text=prompt["prompt"],
                response_text=r["response_text"],
                ground_truth=prompt.get("ground_truth", ""),
                category=prompt.get("category", "general"),
                output_tokens=r["output_tokens"],
                judge_model=judge_model,
            )

        row = {
            "model": model,
            "prompt_id": pid,
            "temperature": temp,
            "category": prompt.get("category", ""),
            "response_text": r["response_text"],
            "output_tokens": r["output_tokens"],
            "total_latency_ms": r["total_latency_ms"],
            "composite_score": s.get("composite_score", 0.0),
            "one_line_verdict": s.get("one_line_verdict", ""),
            "error": r["error"],
            "judge_error": s.get("judge_error"),
        }
        _append_to_csv(row, TEMP_RUNS_FILE, ["model", "prompt_id", "temperature"])
        yield {"type": "result", "model": model, "prompt_id": pid,
               "temperature": temp, "idx": idx, "total": total, "row": row}

    yield {"type": "done", "model": model, "total": total}


def sensitivity_report(model: str) -> dict:
    """Per-prompt composite across temps + variance, and per-temp means."""
    if not TEMP_RUNS_FILE.exists():
        return {"model": model, "prompts": [], "temp_means": {}}
    df = pd.read_csv(TEMP_RUNS_FILE)
    df = df[(df["model"] == model) & df["judge_error"].isna() & df["error"].isna()]
    if df.empty:
        return {"model": model, "prompts": [], "temp_means": {}}

    prompts_out = []
    for pid in sorted(df["prompt_id"].unique()):
        sub = df[df["prompt_id"] == pid]
        by_temp = {float(t): round(float(v), 2)
                   for t, v in zip(sub["temperature"], sub["composite_score"])}
        vals = list(by_temp.values())
        std = float(pd.Series(vals).std(ddof=0)) if len(vals) > 1 else 0.0
        prompts_out.append({
            "prompt_id": pid,
            "category": sub.iloc[0].get("category", ""),
            "by_temp": by_temp,
            "spread": round(max(vals) - min(vals), 2) if vals else 0.0,
            "std": round(std, 3),
        })
    prompts_out.sort(key=lambda p: -p["std"])

    temp_means = {
        float(t): round(float(g["composite_score"].mean()), 3)
        for t, g in df.groupby("temperature")
    }
    return {"model": model, "prompts": prompts_out, "temp_means": temp_means}
