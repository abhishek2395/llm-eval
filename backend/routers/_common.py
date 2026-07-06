"""
Shared helpers for all routers — JSON sanitation and model display names.
"""

import math
from typing import Any

import pandas as pd

from config import MODEL_ALIASES, PROVIDER_META


def clean(obj: Any) -> Any:
    """
    Make an object strictly JSON-safe:
    - NaN / inf → None  (json.dumps would emit invalid `NaN` literals)
    - numpy scalars → native Python types
    """
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean(obj.item())  # numpy scalar → python scalar
        except Exception:
            return str(obj)
    return obj


def df_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → JSON-safe list of row dicts."""
    if df is None or df.empty:
        return []
    return clean(df.to_dict("records"))


def short_name(model: str) -> str:
    """Clean display name for a model ID (same logic as V1 dashboard.py)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    lower = model.lower()
    for k, v in MODEL_ALIASES.items():
        if k.lower() == lower:
            return v
    part = model.split("/")[-1] if "/" in model else model
    seg = part.split("-")
    return seg[0].capitalize() if len(seg) < 3 else f"{seg[0].capitalize()} {seg[1].capitalize()}"


def provider_of(model: str) -> str:
    return model.split("/")[0] if "/" in model else "unknown"


def color_of(model: str) -> str:
    return PROVIDER_META.get(provider_of(model), {}).get("color", "#64748b")


def model_meta(model: str) -> dict:
    """Full display metadata for one model ID."""
    prov = provider_of(model)
    meta = PROVIDER_META.get(prov, {})
    return {
        "id": model,
        "alias": short_name(model),
        "provider": prov,
        "provider_display": meta.get("name", prov.title()),
        "color": meta.get("color", "#64748b"),
    }
