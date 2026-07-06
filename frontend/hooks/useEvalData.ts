"use client";

/**
 * useEvalData — the dashboard's single source of truth.
 *
 * Loads responses/scores/summary + prompts once, then supports live merging
 * of SSE rows as an eval runs, and non-destructive hide/restore of models
 * (hidden list persisted to localStorage).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { modelsIn, validScores } from "@/lib/derive";
import type {
  EfficiencyRow,
  Prompt,
  ResponseRow,
  ScoreRow,
} from "@/lib/types";

const HIDDEN_KEY = "llm-eval:hidden-models";

export function useEvalData() {
  const [responses, setResponses] = useState<ResponseRow[]>([]);
  const [scores, setScores] = useState<ScoreRow[]>([]);
  const [summary, setSummary] = useState<EfficiencyRow[]>([]);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<string[]>([]);

  // hydrate hidden list from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(HIDDEN_KEY);
      if (raw) setHidden(JSON.parse(raw));
    } catch {
      /* ignore */
    }
  }, []);

  const persistHidden = useCallback((next: string[]) => {
    setHidden(next);
    try {
      localStorage.setItem(HIDDEN_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [r, p] = await Promise.all([api.results(), api.prompts()]);
      setResponses(r.responses);
      setScores(r.scores);
      setSummary(r.summary);
      setPrompts(p);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  /** Merge one live SSE result/cached row pair into state. */
  const mergeRow = useCallback((resp: ResponseRow, score: ScoreRow) => {
    const key = (r: { model: string; prompt_id: string }) =>
      `${r.model}::${r.prompt_id}`;
    setResponses((prev) => [
      ...prev.filter((r) => key(r) !== key(resp)),
      resp,
    ]);
    setScores((prev) => [...prev.filter((s) => key(s) !== key(score)), score]);
  }, []);

  /** Replace/insert the efficiency summary row for a model. */
  const mergeEfficiency = useCallback((row: EfficiencyRow) => {
    setSummary((prev) => [...prev.filter((r) => r.model !== row.model), row]);
  }, []);

  const hideModel = useCallback(
    (model: string) => persistHidden([...new Set([...hidden, model])]),
    [hidden, persistHidden],
  );
  const restoreModel = useCallback(
    (model: string) => persistHidden(hidden.filter((m) => m !== model)),
    [hidden, persistHidden],
  );

  const valid = useMemo(() => validScores(scores), [scores]);
  const evaluatedModels = useMemo(() => modelsIn(valid), [valid]);
  const activeModels = useMemo(
    () => evaluatedModels.filter((m) => !hidden.includes(m)),
    [evaluatedModels, hidden],
  );
  const hiddenModels = useMemo(
    () => evaluatedModels.filter((m) => hidden.includes(m)),
    [evaluatedModels, hidden],
  );

  return {
    responses,
    scores,
    valid,
    summary,
    prompts,
    loading,
    loadError,
    evaluatedModels,
    activeModels,
    hiddenModels,
    refresh,
    mergeRow,
    mergeEfficiency,
    hideModel,
    restoreModel,
  };
}

export type EvalData = ReturnType<typeof useEvalData>;
