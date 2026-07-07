"""
/meta-eval — the eval of the eval.

GET /meta-eval/calibration/items            the golden calibration set
GET /meta-eval/calibration/stream?judge=    SSE: judge the calibration set
GET /meta-eval/calibration/results          per-judge accuracy report
GET /meta-eval/repeatability/stream?judge=  SSE: judge 5 responses × 3 trials
GET /meta-eval/repeatability/results        per-dimension sigma report
GET /meta-eval/consistency/stream?model=    SSE: regenerate prompts × 3 trials
GET /meta-eval/consistency/results?model=   real consistency report
GET /meta-eval/confidence                   bootstrap CIs + ties + self-judge flags
"""

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from config import JUDGE_MODEL, OPENROUTER_API_KEY
from meta_eval import (
    calibration_report, confidence_report, consistency_report,
    load_calibration, repeatability_report, run_calibration_live,
    run_consistency_live, run_repeatability_live,
)

from ._common import clean

router = APIRouter(prefix="/meta-eval", tags=["meta-eval"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(clean(payload), ensure_ascii=False, default=str)}\n\n"


def _key(api_key: str | None) -> str:
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise HTTPException(400, "No OpenRouter API key configured.")
    return key


def _stream(gen):
    return StreamingResponse((_sse(u) for u in gen),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/calibration/items")
def calibration_items():
    return {"items": load_calibration(), "default_judge": JUDGE_MODEL}


@router.get("/calibration/stream")
def calibration_stream(judge: str = Query(JUDGE_MODEL),
                       api_key: str | None = Query(None)):
    return _stream(run_calibration_live(judge, _key(api_key)))


@router.get("/calibration/results")
def calibration_results():
    return clean(calibration_report())


@router.get("/repeatability/stream")
def repeatability_stream(judge: str = Query(JUDGE_MODEL),
                         trials: int = Query(3, ge=2, le=5),
                         api_key: str | None = Query(None)):
    return _stream(run_repeatability_live(judge, _key(api_key), trials=trials))


@router.get("/repeatability/results")
def repeatability_results():
    return clean(repeatability_report())


@router.get("/consistency/stream")
def consistency_stream(model: str = Query(...),
                       trials: int = Query(3, ge=2, le=5),
                       api_key: str | None = Query(None)):
    return _stream(run_consistency_live(model, _key(api_key), trials=trials))


@router.get("/consistency/results")
def consistency_results(model: str = Query(...)):
    return clean(consistency_report(model))


@router.get("/confidence")
def confidence():
    return clean(confidence_report())
