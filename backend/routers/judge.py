"""
/judge — multi-judge panel.

GET /judge/stream?judge=<id>&models=a,b   SSE: re-judge existing responses
                                          with another judge model
GET /judge/scores                         all panel scores (judge_scores.csv)
GET /judge/agreement                      inter-judge agreement + disagreements

The default judge's scores live in scores.csv (judge = config.JUDGE_MODEL);
panel judges' scores are stored separately in results/judge_scores.csv with
a `judge_model` column. Agreement is computed across the union.
"""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from config import (
    JUDGE_MODEL, JUDGE_TEMPERATURE, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS, PROMPTS_FILE, RESULTS_DIR, RESULTS_FILE, SCORES_FILE,
)
from live_eval import _append_to_csv, _call_judge

from ._common import clean, df_records, model_meta

router = APIRouter(prefix="/judge", tags=["judge"])

JUDGE_SCORES_FILE = RESULTS_DIR / "judge_scores.csv"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(clean(payload), ensure_ascii=False, default=str)}\n\n"


def _prompts_lookup() -> dict:
    if not PROMPTS_FILE.exists():
        return {}
    return {p["id"]: p for p in json.loads(PROMPTS_FILE.read_text())}


@router.get("/stream")
def judge_stream(
    judge: str = Query(..., description="OpenRouter model ID to use as judge"),
    models: str = Query(..., description="Comma-separated model IDs to re-judge"),
    api_key: str | None = Query(None),
):
    """Re-judge the stored responses of the given models with another judge."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(400, "No OpenRouter API key configured.")
    if not RESULTS_FILE.exists():
        raise HTTPException(404, "No responses to judge yet.")

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    r = pd.read_csv(RESULTS_FILE)
    rows = r[r["model"].isin(model_list)]
    rows = rows[rows["error"].isna() & (rows["response_text"].notna())]
    lookup = _prompts_lookup()

    # Skip (judge, model, prompt) triples already judged
    existing: set[tuple[str, str, str]] = set()
    if JUDGE_SCORES_FILE.exists():
        j = pd.read_csv(JUDGE_SCORES_FILE)
        ok = j[j["judge_error"].isna()] if "judge_error" in j.columns else j
        existing = {
            (row["judge_model"], row["model"], row["prompt_id"])
            for _, row in ok.iterrows()
        }

    def event_gen():
        import openai

        client = openai.OpenAI(
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )
        todo = [
            row
            for _, row in rows.iterrows()
            if (judge, row["model"], row["prompt_id"]) not in existing
        ]
        total = len(todo)
        skipped = len(rows) - total
        yield _sse({"type": "start", "judge": judge, "total": total, "cached": skipped})

        for i, row in enumerate(todo):
            pid = row["prompt_id"]
            meta = lookup.get(pid, {})
            yield _sse({"type": "progress", "judge": judge, "model": row["model"],
                        "prompt_id": pid, "idx": i, "total": total})
            s = _call_judge(
                client,
                prompt_text=row["prompt_text"],
                response_text=row["response_text"],
                ground_truth=meta.get("ground_truth", ""),
                category=meta.get("category", "general"),
                output_tokens=int(row.get("output_tokens", 250) or 250),
                judge_model=judge,
            )
            score_row = {"judge_model": judge, "model": row["model"],
                         "prompt_id": pid, **s}
            _append_to_csv(score_row, JUDGE_SCORES_FILE,
                           ["judge_model", "model", "prompt_id"])
            yield _sse({"type": "result", "judge": judge, "model": row["model"],
                        "prompt_id": pid, "idx": i, "total": total,
                        "score_row": score_row})

        yield _sse({"type": "done", "judge": judge, "total": total})

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers=SSE_HEADERS)


@router.get("/scores")
def judge_scores():
    if not JUDGE_SCORES_FILE.exists():
        return {"scores": [], "judges": []}
    j = pd.read_csv(JUDGE_SCORES_FILE)
    return {
        "scores": df_records(j),
        "judges": sorted(j["judge_model"].dropna().unique().tolist()),
    }


@router.get("/agreement")
def agreement():
    """
    Inter-judge agreement per model, across the default judge (scores.csv)
    and every panel judge (judge_scores.csv).

    agreement = 1 - (mean |composite diff| between judge pairs) / 4
    disagreements = per-prompt judge pairs differing by > 1.0 composite.
    """
    if not SCORES_FILE.exists():
        return {"judges": [], "models": [], "disagreements": []}

    base = pd.read_csv(SCORES_FILE)
    base = base[base["judge_error"].isna()].copy()
    base["judge_model"] = JUDGE_MODEL

    frames = [base[["judge_model", "model", "prompt_id", "composite_score"]]]
    if JUDGE_SCORES_FILE.exists():
        j = pd.read_csv(JUDGE_SCORES_FILE)
        j = j[j["judge_error"].isna()]
        frames.append(j[["judge_model", "model", "prompt_id", "composite_score"]])

    allj = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["judge_model", "model", "prompt_id"], keep="last"
    )
    judges = sorted(allj["judge_model"].unique().tolist())
    if len(judges) < 2:
        return {"judges": judges, "models": [], "disagreements": [],
                "note": "Run a second judge to compute agreement."}

    pivot = allj.pivot_table(index=["model", "prompt_id"],
                             columns="judge_model",
                             values="composite_score")
    pivot = pivot.dropna(thresh=2)  # need ≥2 judges on the same row

    models_out = []
    disagreements = []
    for model in sorted({m for m, _ in pivot.index}):
        sub = pivot.loc[model]
        diffs = []
        for a_i in range(len(judges)):
            for b_i in range(a_i + 1, len(judges)):
                a, b = judges[a_i], judges[b_i]
                if a not in sub.columns or b not in sub.columns:
                    continue
                pair = sub[[a, b]].dropna()
                for pid, row in pair.iterrows():
                    d = abs(row[a] - row[b])
                    diffs.append(d)
                    if d > 1.0:
                        disagreements.append({
                            "model": model, "prompt_id": pid,
                            "judge_a": a, "judge_b": b,
                            "score_a": round(float(row[a]), 2),
                            "score_b": round(float(row[b]), 2),
                            "diff": round(float(d), 2),
                        })
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            per_judge = {
                j: round(float(sub[j].mean()), 3)
                for j in judges if j in sub.columns and sub[j].notna().any()
            }
            models_out.append({
                "model": model,
                "meta": model_meta(model),
                "agreement": round(1 - mean_diff / 4, 3),
                "mean_abs_diff": round(mean_diff, 3),
                "prompts_compared": int(len(sub)),
                "per_judge_composite": per_judge,
            })

    return clean({
        "judges": judges,
        "models": models_out,
        "disagreements": sorted(disagreements, key=lambda d: -d["diff"]),
    })
