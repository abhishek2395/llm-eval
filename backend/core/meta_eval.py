"""
meta_eval.py — the eval of the eval.

Four instruments for answering "can you trust these scores?":

1. Judge calibration — score hand-authored responses with KNOWN quality
   (planted hallucinations, eloquent-but-wrong, bloat, format violations…)
   and measure whether the judge lands inside the expected score ranges.
2. Judge repeatability — judge the same stored responses N times, report
   per-dimension standard deviation.
3. Model consistency — regenerate the same prompts N times and measure how
   stable the judged quality is (replaces V1's hardcoded 0.85).
4. Statistical confidence — bootstrap 95% CIs on each model's composite,
   with explicitly flagged statistical ties and self-judgment warnings.
"""

from __future__ import annotations

import json
from typing import Generator

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, JUDGE_MODEL, OPENROUTER_BASE_URL, OPENROUTER_HEADERS,
    RESULTS_DIR, RESULTS_FILE, SCORES_FILE, SCORE_DIMENSIONS,
)
from live_eval import _append_to_csv, _call_judge, _call_model

CALIBRATION_FILE = DATA_DIR / "calibration_set.json"
CALIBRATION_RUNS_FILE = RESULTS_DIR / "calibration_runs.csv"
REPEATABILITY_FILE = RESULTS_DIR / "repeatability_runs.csv"
CONSISTENCY_FILE = RESULTS_DIR / "consistency_runs.csv"


def _client(api_key: str):
    import openai

    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL,
                         default_headers=OPENROUTER_HEADERS)


# ── 1. Judge calibration ──────────────────────────────────────────────────────

def load_calibration() -> list[dict]:
    if not CALIBRATION_FILE.exists():
        return []
    return json.loads(CALIBRATION_FILE.read_text())


def _check_item(item: dict, scores: dict) -> tuple[bool, list[dict]]:
    """Compare judge scores against the item's expected ranges."""
    checks: list[dict] = []
    for dim, (lo, hi) in item.get("expected", {}).items():
        got = scores.get(dim)
        ok = got is not None and lo <= float(got) <= hi
        checks.append({"dimension": dim, "expected": [lo, hi],
                       "got": got, "ok": bool(ok)})
    if "expect_refused" in item:
        got_ref = bool(scores.get("model_refused", False))
        checks.append({"dimension": "model_refused",
                       "expected": item["expect_refused"],
                       "got": got_ref, "ok": got_ref == item["expect_refused"]})
    return all(c["ok"] for c in checks), checks


def run_calibration_live(judge: str, api_key: str) -> Generator[dict, None, None]:
    items = load_calibration()
    if not items:
        yield {"type": "error", "message": "No calibration set found."}
        return
    client = _client(api_key)

    # (judge, cal_id) pairs already run successfully
    existing: set[tuple[str, str]] = set()
    if CALIBRATION_RUNS_FILE.exists():
        df = pd.read_csv(CALIBRATION_RUNS_FILE)
        ok = df[df["judge_error"].isna()] if "judge_error" in df.columns else df
        existing = {(r["judge_model"], r["cal_id"]) for _, r in ok.iterrows()}

    total = len(items)
    yield {"type": "start", "judge": judge, "total": total}
    for idx, item in enumerate(items):
        if (judge, item["id"]) in existing:
            yield {"type": "cached", "judge": judge, "cal_id": item["id"],
                   "idx": idx, "total": total}
            continue
        yield {"type": "progress", "judge": judge, "cal_id": item["id"],
               "idx": idx, "total": total}
        s = _call_judge(
            client,
            prompt_text=item["prompt"],
            response_text=item["response"],
            ground_truth=item.get("ground_truth", ""),
            category=item.get("category", "general"),
            output_tokens=max(len(item["response"]) // 4, 1),
            judge_model=judge,
        )
        passed, checks = _check_item(item, s)
        row = {
            "judge_model": judge,
            "cal_id": item["id"],
            "label": item["label"],
            "failure_mode_tested": item["failure_mode_tested"],
            "passed": bool(passed and not s.get("judge_error")),
            "checks": json.dumps(checks),
            "scores": json.dumps({d: s.get(d) for d in
                                  SCORE_DIMENSIONS + ["composite_score", "model_refused"]}),
            "judge_error": s.get("judge_error"),
        }
        _append_to_csv(row, CALIBRATION_RUNS_FILE, ["judge_model", "cal_id"])
        yield {"type": "result", "judge": judge, "cal_id": item["id"],
               "idx": idx, "total": total, "row": row}
    yield {"type": "done", "judge": judge, "total": total}


def calibration_report() -> dict:
    items = {i["id"]: i for i in load_calibration()}
    if not CALIBRATION_RUNS_FILE.exists():
        return {"judges": [], "n_items": len(items)}
    df = pd.read_csv(CALIBRATION_RUNS_FILE)
    df = df[df["judge_error"].isna()] if "judge_error" in df.columns else df

    judges_out = []
    for judge in sorted(df["judge_model"].unique()):
        sub = df[df["judge_model"] == judge]
        dim_stats: dict[str, dict[str, int]] = {}
        failures = []
        for _, r in sub.iterrows():
            checks = json.loads(r["checks"])
            for c in checks:
                d = dim_stats.setdefault(c["dimension"], {"pass": 0, "fail": 0})
                d["pass" if c["ok"] else "fail"] += 1
            if not r["passed"]:
                failures.append({
                    "cal_id": r["cal_id"],
                    "label": r["label"],
                    "failure_mode_tested": r["failure_mode_tested"],
                    "checks": [c for c in checks if not c["ok"]],
                    "scores": json.loads(r["scores"]),
                })
        judges_out.append({
            "judge": judge,
            "items_run": int(len(sub)),
            "pass_rate": round(float(sub["passed"].mean()), 3) if len(sub) else 0,
            "per_dimension": {
                dim: {**v, "accuracy": round(v["pass"] / (v["pass"] + v["fail"]), 3)}
                for dim, v in sorted(dim_stats.items())
            },
            "failures": failures,
        })
    return {"judges": judges_out, "n_items": len(items)}


# ── 2. Judge repeatability ────────────────────────────────────────────────────

def _repeatability_targets(k: int = 5) -> list[dict]:
    """Deterministic pick: first valid response per category, up to k."""
    if not RESULTS_FILE.exists():
        return []
    r = pd.read_csv(RESULTS_FILE)
    r = r[r["error"].isna() & r["response_text"].notna()]
    r = r.sort_values(["category", "prompt_id", "model"])
    picked = r.groupby("category").head(1).head(k)
    return picked.to_dict("records")


def run_repeatability_live(judge: str, api_key: str,
                           trials: int = 3) -> Generator[dict, None, None]:
    targets = _repeatability_targets()
    if not targets:
        yield {"type": "error", "message": "No stored responses to re-judge."}
        return
    client = _client(api_key)

    existing: set[tuple[str, str, int]] = set()
    if REPEATABILITY_FILE.exists():
        df = pd.read_csv(REPEATABILITY_FILE)
        existing = {(r["judge_model"], r["response_key"], int(r["trial"]))
                    for _, r in df.iterrows()}

    combos = [(t, n) for t in targets for n in range(trials)]
    total = len(combos)
    yield {"type": "start", "judge": judge, "total": total}
    for idx, (t, trial) in enumerate(combos):
        key = f"{t['model']}::{t['prompt_id']}"
        if (judge, key, trial) in existing:
            yield {"type": "cached", "idx": idx, "total": total}
            continue
        yield {"type": "progress", "judge": judge, "key": key, "trial": trial,
               "idx": idx, "total": total}
        s = _call_judge(
            client,
            prompt_text=t["prompt_text"],
            response_text=t["response_text"],
            ground_truth="",
            category=t.get("category", "general"),
            output_tokens=int(t.get("output_tokens", 250) or 250),
            judge_model=judge,
        )
        row = {"judge_model": judge, "response_key": key, "trial": trial,
               **{d: s.get(d) for d in SCORE_DIMENSIONS},
               "composite_score": s.get("composite_score"),
               "judge_error": s.get("judge_error")}
        _append_to_csv(row, REPEATABILITY_FILE,
                       ["judge_model", "response_key", "trial"])
        yield {"type": "result", "idx": idx, "total": total, "row": row}
    yield {"type": "done", "judge": judge, "total": total}


def repeatability_report() -> dict:
    if not REPEATABILITY_FILE.exists():
        return {"judges": []}
    df = pd.read_csv(REPEATABILITY_FILE)
    df = df[df["judge_error"].isna()] if "judge_error" in df.columns else df
    out = []
    for judge in sorted(df["judge_model"].unique()):
        sub = df[df["judge_model"] == judge]
        dim_sigma = {}
        for d in SCORE_DIMENSIONS + ["composite_score"]:
            stds = sub.groupby("response_key")[d].std(ddof=0).dropna()
            if len(stds):
                dim_sigma[d] = round(float(stds.mean()), 3)
        worst = (
            sub.groupby("response_key")["composite_score"]
            .agg(["min", "max", "count"])
            .assign(spread=lambda x: x["max"] - x["min"])
            .sort_values("spread", ascending=False)
            .head(3)
            .reset_index()
        )
        out.append({
            "judge": judge,
            "responses": int(sub["response_key"].nunique()),
            "trials": int(sub.groupby("response_key")["trial"].count().max()),
            "sigma_per_dimension": dim_sigma,
            "worst_spreads": [
                {"response_key": r["response_key"],
                 "min": round(float(r["min"]), 2),
                 "max": round(float(r["max"]), 2),
                 "spread": round(float(r["spread"]), 2)}
                for _, r in worst.iterrows()
            ],
        })
    return {"judges": out}


# ── 3. Model consistency (real — replaces V1's hardcoded 0.85) ───────────────

def run_consistency_live(model: str, api_key: str,
                         trials: int = 3) -> Generator[dict, None, None]:
    from temp_eval import default_prompts

    prompts = default_prompts()
    if not prompts:
        yield {"type": "error", "message": "No prompts available."}
        return
    client = _client(api_key)

    existing: set[tuple[str, str, int]] = set()
    if CONSISTENCY_FILE.exists():
        df = pd.read_csv(CONSISTENCY_FILE)
        ok = df[df["judge_error"].isna() & df["error"].isna()]
        existing = {(r["model"], r["prompt_id"], int(r["trial"]))
                    for _, r in ok.iterrows()}

    combos = [(p, n) for p in prompts for n in range(trials)]
    total = len(combos)
    yield {"type": "start", "model": model, "total": total}
    for idx, (prompt, trial) in enumerate(combos):
        pid = prompt["id"]
        if (model, pid, trial) in existing:
            yield {"type": "cached", "idx": idx, "total": total}
            continue
        yield {"type": "progress", "model": model, "prompt_id": pid,
               "trial": trial, "idx": idx, "total": total, "stage": "inference"}
        r = _call_model(client, model, prompt["prompt"])
        yield {"type": "progress", "model": model, "prompt_id": pid,
               "trial": trial, "idx": idx, "total": total, "stage": "judging"}
        s = ({"composite_score": 0.0, "judge_error": None} if r["error"] else
             _call_judge(client, prompt_text=prompt["prompt"],
                         response_text=r["response_text"],
                         ground_truth=prompt.get("ground_truth", ""),
                         category=prompt.get("category", "general"),
                         output_tokens=r["output_tokens"]))
        row = {"model": model, "prompt_id": pid, "trial": trial,
               "composite_score": s.get("composite_score", 0.0),
               "output_tokens": r["output_tokens"],
               "error": r["error"], "judge_error": s.get("judge_error")}
        _append_to_csv(row, CONSISTENCY_FILE, ["model", "prompt_id", "trial"])
        yield {"type": "result", "idx": idx, "total": total, "row": row}
    yield {"type": "done", "model": model, "total": total}


def real_consistency(model: str) -> float | None:
    """Measured consistency ∈ [0,1]: 1 − mean per-prompt composite σ ÷ 2."""
    if not CONSISTENCY_FILE.exists():
        return None
    df = pd.read_csv(CONSISTENCY_FILE)
    df = df[(df["model"] == model) & df["error"].isna() & df["judge_error"].isna()]
    if df.empty:
        return None
    stds = df.groupby("prompt_id")["composite_score"].std(ddof=0).dropna()
    if not len(stds):
        return None
    return round(max(0.0, 1.0 - float(stds.mean()) / 2.0), 3)


def consistency_report(model: str) -> dict:
    if not CONSISTENCY_FILE.exists():
        return {"model": model, "prompts": [], "consistency": None}
    df = pd.read_csv(CONSISTENCY_FILE)
    df = df[(df["model"] == model) & df["error"].isna() & df["judge_error"].isna()]
    prompts = []
    for pid in sorted(df["prompt_id"].unique()):
        vals = df[df["prompt_id"] == pid]["composite_score"].tolist()
        prompts.append({"prompt_id": pid,
                        "scores": [round(float(v), 2) for v in vals],
                        "std": round(float(np.std(vals)), 3)})
    prompts.sort(key=lambda p: -p["std"])
    return {"model": model, "prompts": prompts, "consistency": real_consistency(model)}


# ── 4. Statistical confidence ────────────────────────────────────────────────

def confidence_report(n_boot: int = 2000, seed: int = 42) -> dict:
    if not SCORES_FILE.exists():
        return {"models": [], "ties": [], "judge": JUDGE_MODEL}
    s = pd.read_csv(SCORES_FILE)
    s = s[s["judge_error"].isna()]
    rng = np.random.default_rng(seed)
    judge_provider = JUDGE_MODEL.split("/")[0]

    models = []
    for m in sorted(s["model"].unique()):
        vals = s[s["model"] == m]["composite_score"].to_numpy()
        if len(vals) < 2:
            continue
        boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        models.append({
            "model": m,
            "n_prompts": int(len(vals)),
            "mean": round(float(vals.mean()), 3),
            "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3),
            "self_judged": m.split("/")[0] == judge_provider,
        })
    models.sort(key=lambda x: -x["mean"])

    ties = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            if a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]:
                ties.append({"model_a": a["model"], "model_b": b["model"]})
    return {"models": models, "ties": ties, "judge": JUDGE_MODEL,
            "n_boot": n_boot}
