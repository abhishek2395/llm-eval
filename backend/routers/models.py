"""
/models — OpenRouter catalog + models with results.

GET  /models/catalog      searchable/filterable live catalog (cached 5 min)
GET  /models/active       models that have rows in scores.csv, with display meta
GET  /models/aliases      MODEL_ALIASES map (id → display name)
"""

import time

import pandas as pd
from fastapi import APIRouter, Query

from config import MODEL_ALIASES, OPENROUTER_PRESET_MODELS, SCORES_FILE
from openrouter_client import fetch_model_catalog

from ._common import clean, model_meta

router = APIRouter(prefix="/models", tags=["models"])

# ── Catalog cache (module-level, 5 min TTL — same as V1 dashboard) ───────────
_cache: dict = {"ts": 0.0, "catalog": None}
CATALOG_TTL_S = 300


def get_catalog():
    now = time.time()
    if _cache["catalog"] is None or (now - _cache["ts"]) > CATALOG_TTL_S:
        _cache["catalog"] = fetch_model_catalog()
        _cache["ts"] = now
    return _cache["catalog"]


def enrich_pricing(row: dict, model: str) -> dict:
    """
    Replace build_efficiency_row's fallback pricing with real OpenRouter
    catalog pricing (V1's API_PRICING only knew 4 legacy model IDs).
    """
    try:
        info = get_catalog().get(model)
    except Exception:
        return row
    if info and (info.input_price_per_1m or info.output_price_per_1m):
        avg_in = row.get("avg_input_tokens") or 0
        avg_out = row.get("avg_output_tokens") or 0
        cost = (avg_in / 1e6 * info.input_price_per_1m) + (avg_out / 1e6 * info.output_price_per_1m)
        row.update({
            "input_price_per_1m": info.input_price_per_1m,
            "output_price_per_1m": info.output_price_per_1m,
            "api_cost_per_answer_usd": round(cost, 6),
            "api_answers_for_20usd": int(20.0 / cost) if cost > 0 else 999_999,
            "context_length": info.context_length,
        })
    return row


@router.get("/catalog")
def catalog(
    search: str | None = Query(None, description="Substring match on id/name/provider"),
    provider: str | None = Query(None),
    tier: str = Query("all", description="all | free | lt1 | 1to5 | gt5 ($/1M input)"),
    limit: int = Query(100, ge=1, le=500),
):
    cat = get_catalog()
    models = cat.search(search) if search else cat.all()
    if provider and provider != "all":
        models = [m for m in models if m.provider == provider]
    if tier == "free":
        models = [m for m in models if m.is_free]
    elif tier == "lt1":
        models = [m for m in models if 0 < m.input_price_per_1m < 1]
    elif tier == "1to5":
        models = [m for m in models if 1 <= m.input_price_per_1m <= 5]
    elif tier == "gt5":
        models = [m for m in models if m.input_price_per_1m > 5]

    # Provider distribution over the FULL catalog (for the bar chart)
    provider_counts: dict[str, int] = {}
    for m in cat.all():
        provider_counts[m.provider] = provider_counts.get(m.provider, 0) + 1

    return {
        "total_in_catalog": len(cat),
        "matched": len(models),
        "providers": cat.providers(),
        "provider_counts": provider_counts,
        "models": [m.to_dict() for m in models[:limit]],
    }


@router.get("/active")
def active_models():
    """Models that currently have judged rows in scores.csv."""
    if not SCORES_FILE.exists():
        return {"models": []}
    s = pd.read_csv(SCORES_FILE)
    if s.empty:
        return {"models": []}
    out = []
    for m in sorted(s["model"].dropna().unique().tolist()):
        meta = model_meta(m)
        m_rows = s[s["model"] == m]
        valid = m_rows[m_rows["judge_error"].isna()] if "judge_error" in m_rows.columns else m_rows
        meta["prompts_evaluated"] = int(len(valid))
        meta["composite_score"] = clean(round(valid["composite_score"].mean(), 3)) if not valid.empty else None
        out.append(meta)
    return {"models": out}


@router.get("/aliases")
def aliases():
    return {"aliases": MODEL_ALIASES, "presets": OPENROUTER_PRESET_MODELS}
