"""
/prompts — CRUD over data/prompts.json + named prompt sets + bulk import.
"""

import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import DATA_DIR, PROMPTS_FILE

router = APIRouter(prefix="/prompts", tags=["prompts"])

SETS_FILE = DATA_DIR / "prompt_sets.json"


class PromptIn(BaseModel):
    id: str | None = None
    category: str = "general"
    difficulty: str = "medium"
    prompt: str = Field(min_length=1)
    ground_truth: str = ""


class Prompt(PromptIn):
    id: str


def _load() -> list[dict]:
    if not PROMPTS_FILE.exists():
        return []
    return json.loads(PROMPTS_FILE.read_text())


def _save(prompts: list[dict]) -> None:
    PROMPTS_FILE.write_text(json.dumps(prompts, indent=4, ensure_ascii=False))


def _make_id(category: str, existing: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "prompt"
    n = 1
    while f"{slug}_q{n}" in existing:
        n += 1
    return f"{slug}_q{n}"


class PromptSet(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    prompt_ids: list[str]


def _load_sets() -> dict[str, list[str]]:
    if not SETS_FILE.exists():
        return {}
    return json.loads(SETS_FILE.read_text())


def _save_sets(sets: dict[str, list[str]]) -> None:
    SETS_FILE.write_text(json.dumps(sets, indent=2, ensure_ascii=False))


@router.get("/sets")
def list_sets() -> list[PromptSet]:
    return [PromptSet(name=n, prompt_ids=ids) for n, ids in _load_sets().items()]


@router.post("/sets", status_code=201)
def save_set(body: PromptSet) -> PromptSet:
    known = {p["id"] for p in _load()}
    missing = [i for i in body.prompt_ids if i not in known]
    if missing:
        raise HTTPException(422, f"Unknown prompt ids: {missing}")
    sets = _load_sets()
    sets[body.name] = body.prompt_ids
    _save_sets(sets)
    return body


@router.delete("/sets/{name}", status_code=204)
def delete_set(name: str) -> None:
    sets = _load_sets()
    if name not in sets:
        raise HTTPException(404, f"Prompt set '{name}' not found.")
    del sets[name]
    _save_sets(sets)


@router.post("/bulk", status_code=201)
def bulk_import(body: list[PromptIn]) -> list[Prompt]:
    """Bulk import (e.g. parsed CSV rows). Skips ids that already exist."""
    prompts = _load()
    ids = {p["id"] for p in prompts}
    created: list[Prompt] = []
    for item in body:
        pid = item.id or _make_id(item.category, ids)
        if pid in ids:
            continue
        new = {"id": pid, "category": item.category, "difficulty": item.difficulty,
               "prompt": item.prompt, "ground_truth": item.ground_truth}
        prompts.append(new)
        ids.add(pid)
        created.append(Prompt(**new))
    _save(prompts)
    return created


@router.get("")
def list_prompts() -> list[Prompt]:
    return [Prompt(**{"difficulty": "medium", "ground_truth": "", **p}) for p in _load()]


@router.post("", status_code=201)
def add_prompt(body: PromptIn) -> Prompt:
    prompts = _load()
    ids = {p["id"] for p in prompts}
    pid = body.id or _make_id(body.category, ids)
    if pid in ids:
        raise HTTPException(409, f"Prompt id '{pid}' already exists.")
    new = {"id": pid, "category": body.category, "difficulty": body.difficulty,
           "prompt": body.prompt, "ground_truth": body.ground_truth}
    prompts.append(new)
    _save(prompts)
    return Prompt(**new)


@router.put("/{pid}")
def update_prompt(pid: str, body: PromptIn) -> Prompt:
    prompts = _load()
    for i, p in enumerate(prompts):
        if p["id"] == pid:
            updated = {"id": pid, "category": body.category, "difficulty": body.difficulty,
                       "prompt": body.prompt, "ground_truth": body.ground_truth}
            prompts[i] = updated
            _save(prompts)
            return Prompt(**updated)
    raise HTTPException(404, f"Prompt '{pid}' not found.")


@router.delete("/{pid}", status_code=204)
def delete_prompt(pid: str) -> None:
    prompts = _load()
    remaining = [p for p in prompts if p["id"] != pid]
    if len(remaining) == len(prompts):
        raise HTTPException(404, f"Prompt '{pid}' not found.")
    _save(remaining)
