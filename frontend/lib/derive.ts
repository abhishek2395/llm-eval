/**
 * derive.ts — pure helpers that turn raw rows into chart-ready aggregates.
 */

import type {
  Dimension,
  EfficiencyRow,
  Prompt,
  ResponseRow,
  ScoreRow,
} from "./types";
import { DIMENSIONS } from "./types";

export function validScores(scores: ScoreRow[]): ScoreRow[] {
  return scores.filter((s) => s.judge_error == null);
}

export function modelsIn(scores: ScoreRow[]): string[] {
  return [...new Set(scores.map((s) => s.model))].sort();
}

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

/** Per-model mean of each score dimension. */
export function dimensionMeans(
  scores: ScoreRow[],
  models: string[],
): Map<string, Record<Dimension, number>> {
  const out = new Map<string, Record<Dimension, number>>();
  for (const m of models) {
    const rows = scores.filter((s) => s.model === m);
    const rec = {} as Record<Dimension, number>;
    for (const d of DIMENSIONS) rec[d] = mean(rows.map((r) => Number(r[d]) || 0));
    out.set(m, rec);
  }
  return out;
}

export function modelMean(
  rows: (ScoreRow | ResponseRow)[],
  model: string,
  field: string,
): number {
  const vals = rows
    .filter((r) => r.model === model)
    .map((r) => Number((r as unknown as Record<string, unknown>)[field]) || 0);
  return mean(vals);
}

/** category for a prompt_id, from the prompt library. */
export function categoryOf(prompts: Prompt[], promptId: string): string {
  return prompts.find((p) => p.id === promptId)?.category ?? "—";
}

/** Per model × category mean composite score. */
export function categoryMeans(
  scores: ScoreRow[],
  prompts: Prompt[],
  models: string[],
): { categories: string[]; byModel: Map<string, Map<string, number>> } {
  const categories = [...new Set(prompts.map((p) => p.category))];
  const byModel = new Map<string, Map<string, number>>();
  for (const m of models) {
    const catMap = new Map<string, number>();
    for (const c of categories) {
      const ids = new Set(prompts.filter((p) => p.category === c).map((p) => p.id));
      const rows = scores.filter((s) => s.model === m && ids.has(s.prompt_id));
      if (rows.length) catMap.set(c, mean(rows.map((r) => r.composite_score)));
    }
    byModel.set(m, catMap);
  }
  return { categories, byModel };
}

export function sortByValueIndex(summary: EfficiencyRow[]): EfficiencyRow[] {
  return [...summary].sort((a, b) => b.value_index - a.value_index);
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
export const fmt = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : "—");
