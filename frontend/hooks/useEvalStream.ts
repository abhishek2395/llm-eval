"use client";

/**
 * useEvalStream — consumes GET /eval/stream (SSE) for one model at a time,
 * exposing per-prompt progress and forwarding rows to the data store.
 */

import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { EfficiencyRow, ResponseRow, ScoreRow, SSEEvent } from "@/lib/types";

export interface RunProgress {
  model: string;
  idx: number;
  total: number;
  completed: number;
  cached: number;
  failures: number;
  /** live (non-cached) token usage — feeds the running cost tracker */
  inTokens: number;
  outTokens: number;
  stage: "starting" | "inference" | "judging" | "done" | "error";
  promptId?: string;
  lastScore?: number;
  message?: string;
}

interface StreamCallbacks {
  onRow: (resp: ResponseRow, score: ScoreRow) => void;
  onEfficiency: (row: EfficiencyRow) => void;
  onDone?: (model: string) => void;
}

export function useEvalStream({ onRow, onEfficiency, onDone }: StreamCallbacks) {
  const [runs, setRuns] = useState<Record<string, RunProgress>>({});
  const [queue, setQueue] = useState<string[]>([]);
  const sourceRef = useRef<EventSource | null>(null);
  const runningRef = useRef(false);
  const queueRef = useRef<string[]>([]);

  const update = useCallback(
    (model: string, patch: Partial<RunProgress>) =>
      setRuns((prev) => {
        const base: RunProgress = prev[model] ?? {
          model,
          idx: 0,
          total: 0,
          completed: 0,
          cached: 0,
          failures: 0,
          inTokens: 0,
          outTokens: 0,
          stage: "starting",
        };
        return { ...prev, [model]: { ...base, ...patch, model } };
      }),
    [],
  );

  const startNext = useCallback(() => {
    if (runningRef.current) return;
    const next = queueRef.current.shift();
    setQueue([...queueRef.current]);
    if (!next) return;

    runningRef.current = true;
    update(next, { stage: "starting" });
    const es = new EventSource(api.evalStreamUrl(next));
    sourceRef.current = es;

    const finish = () => {
      es.close();
      sourceRef.current = null;
      runningRef.current = false;
      startNext(); // pull the next queued model, if any
    };

    es.onmessage = (msg) => {
      let ev: SSEEvent;
      try {
        ev = JSON.parse(msg.data) as SSEEvent;
      } catch {
        return;
      }
      switch (ev.type) {
        case "start":
          update(ev.model, { total: ev.total, stage: "inference" });
          break;
        case "progress":
          update(ev.model, {
            idx: ev.idx,
            total: ev.total,
            promptId: ev.prompt_id,
            stage: ev.stage,
          });
          break;
        case "cached":
        case "result": {
          onRow(ev.response_row, ev.score_row);
          const failed = Boolean(ev.response_row.error || ev.score_row.judge_error);
          setRuns((prev) => {
            const cur = prev[ev.model];
            const completed = (cur?.completed ?? 0) + 1;
            const cached = (cur?.cached ?? 0) + (ev.type === "cached" ? 1 : 0);
            const failures = (cur?.failures ?? 0) + (failed ? 1 : 0);
            const live = ev.type === "result"; // cached rows were paid for previously
            const inTokens =
              (cur?.inTokens ?? 0) + (live ? Number(ev.response_row.input_tokens) || 0 : 0);
            const outTokens =
              (cur?.outTokens ?? 0) + (live ? Number(ev.response_row.output_tokens) || 0 : 0);
            return {
              ...prev,
              [ev.model]: {
                ...(cur ??
                  ({
                    model: ev.model,
                    idx: 0,
                    failures: 0,
                    inTokens: 0,
                    outTokens: 0,
                    stage: "inference",
                  } as RunProgress)),
                model: ev.model,
                idx: ev.idx,
                total: ev.total,
                completed,
                cached,
                failures,
                inTokens,
                outTokens,
                stage: "inference",
                promptId: ev.prompt_id,
                lastScore: ev.score_row.composite_score,
              },
            };
          });
          break;
        }
        case "efficiency":
          onEfficiency(ev.row);
          break;
        case "done":
          update(ev.model, { stage: "done" });
          onDone?.(ev.model);
          finish();
          break;
        case "error":
          update(ev.model, { stage: "error", message: ev.message });
          finish();
          break;
      }
    };

    es.onerror = () => {
      // EventSource fires error on normal close too — only flag if not done
      setRuns((prev) => {
        const cur = prev[next];
        if (cur && cur.stage !== "done") {
          return {
            ...prev,
            [next]: { ...cur, stage: "error", message: "stream disconnected" },
          };
        }
        return prev;
      });
      finish();
    };
  }, [onRow, onEfficiency, onDone, update]);

  /** Queue one or more models for sequential evaluation. */
  const run = useCallback(
    (models: string[]) => {
      queueRef.current = [
        ...queueRef.current,
        ...models.filter((m) => !queueRef.current.includes(m)),
      ];
      setQueue([...queueRef.current]);
      startNext();
    },
    [startNext],
  );

  const dismissRun = useCallback((model: string) => {
    setRuns((prev) => {
      const next = { ...prev };
      delete next[model];
      return next;
    });
  }, []);

  const isRunning = Object.values(runs).some(
    (r) => r.stage !== "done" && r.stage !== "error",
  );

  return { run, runs, queue, isRunning, dismissRun };
}
