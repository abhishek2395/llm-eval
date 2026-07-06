"""
main.py — LLM Eval Framework V2 backend (FastAPI).

All V1 Python logic lives unchanged in core/ and is imported flat
(`from config import ...`) via the sys.path insert below, so the V1
modules never needed their imports rewritten.

Run:  uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Make backend/core/ importable as flat modules BEFORE any router imports.
CORE_DIR = Path(__file__).parent / "core"
sys.path.insert(0, str(CORE_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from config import (  # noqa: E402
    JUDGE_MODEL, MODEL_ALIASES, OPENROUTER_API_KEY,
    OPENROUTER_PRESET_MODELS, PROMPTS_FILE, PROVIDER_META,
    SCORE_DIMENSIONS, SCORES_FILE, VALUE_WEIGHTS,
)
from routers import eval as eval_router  # noqa: E402
from routers import judge as judge_router  # noqa: E402
from routers import models as models_router  # noqa: E402
from routers import prompts as prompts_router  # noqa: E402
from routers import results as results_router  # noqa: E402

app = FastAPI(
    title="LLM Eval Framework API",
    version="2.0.0",
    description="Quality · Efficiency · Value-per-$20 · 300+ models via OpenRouter",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval_router.router)
app.include_router(judge_router.router)
app.include_router(models_router.router)
app.include_router(prompts_router.router)
app.include_router(results_router.router)


@app.get("/health", tags=["meta"])
def health():
    import json

    import pandas as pd

    n_models = 0
    if SCORES_FILE.exists():
        s = pd.read_csv(SCORES_FILE)
        n_models = int(s["model"].nunique()) if not s.empty else 0
    n_prompts = len(json.loads(PROMPTS_FILE.read_text())) if PROMPTS_FILE.exists() else 0
    return {
        "status": "ok",
        "version": "2.0.0",
        "api_key_set": bool(OPENROUTER_API_KEY),
        "models_evaluated": n_models,
        "prompts": n_prompts,
    }


@app.get("/meta", tags=["meta"])
def meta():
    """Shared constants — single source of truth mirrored by frontend lib/."""
    return {
        "aliases": MODEL_ALIASES,
        "provider_meta": PROVIDER_META,
        "dimensions": SCORE_DIMENSIONS,
        "value_weights": VALUE_WEIGHTS,
        "judge_model": JUDGE_MODEL,
        "presets": OPENROUTER_PRESET_MODELS,
    }
