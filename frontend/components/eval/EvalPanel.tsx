"use client";

/**
 * EvalPanel — add models (search aliases or type any OpenRouter ID),
 * run live evals over SSE, hide/restore evaluated models.
 */

import { useMemo, useRef, useState } from "react";
import { Plus, RotateCcw, X } from "lucide-react";
import { MODEL_ALIASES, shortName } from "@/lib/aliases";
import { mcolor } from "@/lib/colors";
import type { RunProgress } from "@/hooks/useEvalStream";

export type CostEstimator = (
  model: string,
  inTokens: number,
  outTokens: number,
) => number | null;

function ProgressBar({
  run,
  total,
  estimateCost,
}: {
  run: RunProgress;
  total: number;
  estimateCost?: CostEstimator;
}) {
  const cost =
    estimateCost && (run.inTokens || run.outTokens)
      ? estimateCost(run.model, run.inTokens, run.outTokens)
      : null;
  const pct =
    run.stage === "done"
      ? 100
      : total > 0
        ? Math.round((run.completed / total) * 100)
        : 5;
  const label =
    run.stage === "error"
      ? `✕ ${run.message ?? "failed"}`
      : run.stage === "done"
        ? run.failures > 0
          ? `⚠ done — ${run.failures}/${run.completed} failed (auto-retries next run)`
          : `✓ complete — ${run.completed} prompts (${run.cached} cached)`
        : run.stage === "judging"
          ? `⚖ judging ${run.promptId ?? ""} (${run.completed}/${total || "…"})`
          : run.stage === "inference"
            ? `→ calling model ${run.promptId ?? ""} (${run.completed}/${total || "…"})`
            : "starting…";
  const color =
    run.stage === "error"
      ? "var(--red)"
      : run.stage === "done"
        ? run.failures > 0
          ? "var(--amber)"
          : "var(--green)"
        : "var(--amber)";

  return (
    <div className="mb-2">
      <div className="mb-1 flex items-baseline justify-between">
        <span
          className="font-mono text-[0.75rem] font-semibold"
          style={{ color: mcolor(run.model) }}
        >
          {shortName(run.model)}
        </span>
        <span className="font-mono text-[0.68rem]" style={{ color }}>
          {label}
          {run.lastScore != null && run.stage !== "done" && run.stage !== "error"
            ? ` · last score ${run.lastScore.toFixed(2)}`
            : ""}
          {run.inTokens + run.outTokens > 0 &&
            ` · ${(run.inTokens + run.outTokens).toLocaleString()} tok${
              cost != null ? ` ≈ $${cost.toFixed(4)}` : ""
            }`}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

export function EvalPanel({
  activeModels,
  hiddenModels,
  evaluatedModels,
  keySet,
  runs,
  queue,
  onRun,
  onHide,
  onRestore,
  estimateCost,
}: {
  activeModels: string[];
  hiddenModels: string[];
  evaluatedModels: string[];
  keySet: boolean;
  runs: Record<string, RunProgress>;
  queue: string[];
  onRun: (models: string[]) => void;
  onHide: (model: string) => void;
  onRestore: (model: string) => void;
  estimateCost?: CostEstimator;
}) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = Object.keys(MODEL_ALIASES).filter(
      (m) => !evaluatedModels.includes(m),
    );
    if (!q) return pool.slice(0, 8);
    return pool
      .filter(
        (m) =>
          m.toLowerCase().includes(q) ||
          MODEL_ALIASES[m].toLowerCase().includes(q),
      )
      .slice(0, 8);
  }, [query, evaluatedModels]);

  const isValidCustomId = /^[\w.-]+\/[\w.:-]+$/.test(query.trim());
  const activeRuns = Object.values(runs).filter((r) => r.stage !== "done" || true);

  const startEval = (model: string) => {
    setQuery("");
    inputRef.current?.blur();
    onRun([model]);
  };

  return (
    <section>
      <div className="section-hdr">⚡ Model selection &amp; evaluation</div>

      {/* Active model chips */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {activeModels.map((m) => (
          <span
            key={m}
            className="inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 font-mono text-[0.75rem]"
            style={{
              background: `${mcolor(m)}14`,
              borderColor: `${mcolor(m)}40`,
              color: mcolor(m),
            }}
          >
            {shortName(m)}
            <button
              aria-label={`hide ${m}`}
              className="text-mute transition-colors hover:text-red"
              onClick={() => onHide(m)}
            >
              <X size={13} />
            </button>
          </span>
        ))}
        {activeModels.length === 0 && (
          <span className="text-[0.8rem] text-mute">
            No active models — add one below to begin comparing.
          </span>
        )}
      </div>

      {/* Hidden models restore */}
      {hiddenModels.length > 0 && (
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[0.65rem] uppercase tracking-[0.1em] text-mute">
            hidden:
          </span>
          {hiddenModels.map((m) => (
            <button
              key={m}
              className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-bg3 px-2.5 py-1 font-mono text-[0.7rem] text-mute transition-colors hover:text-ink"
              onClick={() => onRestore(m)}
            >
              <RotateCcw size={11} />
              {shortName(m)}
            </button>
          ))}
        </div>
      )}

      {/* Search + add */}
      <div className="relative flex gap-2">
        <div className="relative flex-1">
          <input
            ref={inputRef}
            className="panel-input w-full font-mono text-[0.8rem]"
            placeholder="Search models or type any OpenRouter ID — e.g. google/gemini-2.5-pro"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setTimeout(() => setFocused(false), 150)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && isValidCustomId) startEval(query.trim());
            }}
          />
          {focused && (suggestions.length > 0 || isValidCustomId) && (
            <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-line bg-bg2 shadow-xl">
              {isValidCustomId && !suggestions.includes(query.trim()) && (
                <button
                  className="flex w-full items-center gap-2 px-3 py-2 text-left font-mono text-[0.75rem] text-amber hover:bg-bg3"
                  onMouseDown={() => startEval(query.trim())}
                >
                  <Plus size={13} /> evaluate “{query.trim()}”
                </button>
              )}
              {suggestions.map((m) => (
                <button
                  key={m}
                  className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-bg3"
                  onMouseDown={() => startEval(m)}
                >
                  <span
                    className="font-mono text-[0.75rem]"
                    style={{ color: mcolor(m) }}
                  >
                    {MODEL_ALIASES[m]}
                  </span>
                  <span className="font-mono text-[0.68rem] text-mute">{m}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <span
          className={`pill self-center ${keySet ? "pill-green" : "pill-red"}`}
        >
          {keySet ? "🔑 key set" : "⚠ no key"}
        </span>
      </div>

      <div className="mt-2 text-[0.75rem] text-mute">
        {queue.length > 0 && (
          <span className="font-mono">
            queued: {queue.map(shortName).join(", ")} ·{" "}
          </span>
        )}
        Adding a model runs all prompts through it live — cached prompts are
        instant and free.
      </div>

      {/* Live run progress */}
      {activeRuns.length > 0 && (
        <div className="mt-4">
          {activeRuns.map((r) => (
            <ProgressBar
              key={r.model}
              run={r}
              total={r.total}
              estimateCost={estimateCost}
            />
          ))}
        </div>
      )}
    </section>
  );
}
