"""
/eval — live evaluation streaming.

GET /eval/stream?model=<openrouter-id>  →  text/event-stream

Streams the V1 live_eval.py generator events verbatim as SSE `data:` frames:
    start → cached|progress|result (per prompt) → efficiency → done
    (or error)

The `efficiency` event is V2-only: the computed efficiency_summary row for
the model, emitted just before `done` so the frontend can update the verdict
banner without a second request.
"""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from config import OPENROUTER_API_KEY
from live_eval import build_efficiency_row, run_model_live

from ._common import clean, model_meta
from .models import enrich_pricing

router = APIRouter(prefix="/eval", tags=["eval"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/railway)
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(clean(payload), ensure_ascii=False, default=str)}\n\n"


@router.get("/stream")
def stream_eval(
    model: str = Query(..., description="OpenRouter model ID, e.g. anthropic/claude-sonnet-4-5"),
    api_key: str | None = Query(None, description="Override the server-side OpenRouter key"),
    judge: str | None = Query(None, description="Override the default judge model"),
    prompt_set: str | None = Query(None, description="Run only this named prompt set"),
):
    """Evaluate one model against all prompts, streaming progress as SSE."""
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="No OpenRouter API key configured.")

    prompts = None
    if prompt_set:
        from .prompts import _load, _load_sets

        sets = _load_sets()
        if prompt_set not in sets:
            raise HTTPException(404, f"Prompt set '{prompt_set}' not found.")
        wanted = set(sets[prompt_set])
        prompts = [p for p in _load() if p["id"] in wanted]
        if not prompts:
            raise HTTPException(422, f"Prompt set '{prompt_set}' is empty.")

    def event_gen():
        responses: list[dict] = []
        scores: list[dict] = []
        for update in run_model_live(model, key, prompts=prompts, judge_model=judge):
            if update["type"] in ("result", "cached"):
                responses.append(update["response_row"])
                # Judge-errored rows carry 0-scores — keep them out of the
                # efficiency averages (V1 dashboard did the same filter).
                srow = update["score_row"]
                if not srow.get("judge_error") or pd.isna(srow.get("judge_error")):
                    scores.append(srow)
            if update["type"] == "done" and responses:
                ok = [x for x in responses if not x.get("error")]
                if ok:
                    eff = enrich_pricing(build_efficiency_row(model, ok, scores), model)
                    yield _sse({"type": "efficiency", "model": model,
                                "meta": model_meta(model), "row": eff})
            yield _sse(update)

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=SSE_HEADERS)
