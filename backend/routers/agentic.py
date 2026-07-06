"""
/agentic — Tier-2 agentic harness endpoints.

GET /agentic/tasks                     task definitions
GET /agentic/stream?model=<id>  (SSE)  run all tasks for one model
GET /agentic/results                   all trajectory rows + per-model summary
DELETE /agentic/model/{id}             drop one model's agentic rows
"""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from agentic_eval import AGENTIC_RUNS_FILE, load_tasks, run_agentic_live
from config import OPENROUTER_API_KEY

from ._common import clean, model_meta

router = APIRouter(prefix="/agentic", tags=["agentic"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(clean(payload), ensure_ascii=False, default=str)}\n\n"


@router.get("/tasks")
def tasks():
    return {"tasks": load_tasks()}


@router.get("/stream")
def agentic_stream(
    model: str = Query(..., description="OpenRouter model ID"),
    judge: str | None = Query(None),
    api_key: str | None = Query(None),
):
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(400, "No OpenRouter API key configured.")

    def event_gen():
        for update in run_agentic_live(model, key, judge_model=judge):
            yield _sse(update)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers=SSE_HEADERS)


def _rows() -> list[dict]:
    if not AGENTIC_RUNS_FILE.exists():
        return []
    df = pd.read_csv(AGENTIC_RUNS_FILE)
    out = []
    for _, r in df.iterrows():
        d = clean(r.to_dict())
        try:
            d["transcript"] = json.loads(d.get("transcript") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["transcript"] = []
        out.append(d)
    return out


@router.get("/results")
def results():
    rows = _rows()
    ok = [r for r in rows if not r.get("error") and not r.get("judge_error")]
    summary = []
    for m in sorted({r["model"] for r in ok}):
        mine = [r for r in ok if r["model"] == m]
        n = len(mine)
        calls = sum(r["tool_calls_total"] for r in mine)
        valid = sum(r["tool_calls_valid"] for r in mine)
        summary.append({
            "model": m,
            "meta": model_meta(m),
            "tasks_run": n,
            "success_avg": round(sum(r["task_success"] for r in mine) / n, 2),
            "composite_avg": round(sum(r["composite_score"] for r in mine) / n, 3),
            "efficiency_avg": round(sum(r["tool_efficiency"] for r in mine) / n, 2),
            "honesty_avg": round(sum(r["honesty"] for r in mine) / n, 2),
            "avg_steps": round(sum(r["steps"] for r in mine) / n, 1),
            "validity_pct": round(valid / calls * 100, 1) if calls else 100.0,
            "avg_tokens": round(sum(r["input_tokens"] + r["output_tokens"] for r in mine) / n),
        })
    summary.sort(key=lambda s: -s["composite_avg"])
    return {"rows": rows, "summary": summary}


@router.delete("/model/{model_id:path}")
def delete_agentic(model_id: str):
    if not AGENTIC_RUNS_FILE.exists():
        raise HTTPException(404, "No agentic runs yet.")
    df = pd.read_csv(AGENTIC_RUNS_FILE)
    if model_id not in df["model"].values:
        raise HTTPException(404, f"No agentic rows for '{model_id}'.")
    df[df["model"] != model_id].to_csv(AGENTIC_RUNS_FILE, index=False)
    return {"deleted": model_id}
