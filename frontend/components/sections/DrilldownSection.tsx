"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { shortName } from "@/lib/aliases";
import { mcolor, scoreColor } from "@/lib/colors";
import { fmt } from "@/lib/derive";
import { DIMENSIONS, type Prompt, type ResponseRow, type ScoreRow } from "@/lib/types";

const SORT_OPTIONS = [
  "composite_score",
  "value_index",
  "hallucination_resistance",
  "task_completion",
  "conciseness",
] as const;

function ModelCell({
  model,
  score,
  response,
}: {
  model: string;
  score?: ScoreRow;
  response?: ResponseRow;
}) {
  const [showResponse, setShowResponse] = useState(false);
  const mc = mcolor(model);

  return (
    <div className="min-w-0">
      <div
        className="mb-1.5 font-mono text-[0.75rem] font-semibold"
        style={{ color: mc }}
      >
        {shortName(model)}
      </div>
      {score ? (
        <>
          <div
            className="font-mono text-[1.8rem] font-bold"
            style={{ color: scoreColor(score.composite_score) }}
          >
            {fmt(score.composite_score)}
            <span className="text-[0.9rem] text-mute">/5</span>
          </div>
          <div className="mt-1">
            {DIMENSIONS.map((d) => {
              const val = score[d];
              const pc =
                val >= 4 ? "pill-green" : val >= 3 ? "pill-amber" : "pill-red";
              return (
                <span key={d} className={`pill ${pc}`}>
                  {d.replaceAll("_", " ")}: {val}
                </span>
              );
            })}
          </div>
          {response && (
            <div className="mt-1.5 font-mono text-[0.7rem] leading-[1.8] text-mute">
              ⏱ {fmt(response.total_latency_ms, 0)}ms · first-tok{" "}
              {fmt(response.first_token_latency_ms, 0)}ms ·{" "}
              {fmt(response.tokens_per_second, 0)} tok/s
              <br />
              {fmt(response.output_tokens, 0)} output-tokens ·{" "}
              {fmt(response.verbosity_ratio, 1)}× verbose
              {response.refused && (
                <>
                  <br />
                  <span className="text-red">⚠ refused</span>
                </>
              )}
            </div>
          )}
          {(score.model_refused || score.judge_confidence != null) && (
            <div className="mt-1.5">
              {score.model_refused && (
                <span className="pill pill-red">model refused</span>
              )}
              {score.judge_confidence != null && (
                <span className="pill pill-mute">
                  judge conf {Number(score.judge_confidence).toFixed(2)}
                </span>
              )}
            </div>
          )}
          {score.one_line_verdict && (
            <div
              className="mt-2 rounded-r border-l-2 bg-white/[0.03] px-2.5 py-1.5 text-[0.78rem] leading-normal"
              style={{ borderLeftColor: mc }}
            >
              {score.one_line_verdict}
            </div>
          )}
          {response?.response_text && (
            <button
              className="mt-2 font-mono text-[0.7rem] text-mute hover:text-ink"
              onClick={() => setShowResponse((s) => !s)}
            >
              {showResponse ? "▾ hide response" : "▸ response text"}
            </button>
          )}
          {showResponse && response?.response_text && (
            <div className="mt-1.5 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-line bg-bg3 p-3 text-[0.8rem] leading-relaxed text-mute">
              {response.response_text}
            </div>
          )}
        </>
      ) : (
        <div className="font-mono text-[0.75rem] text-mute">no result</div>
      )}
    </div>
  );
}

export function DrilldownSection({
  models,
  valid,
  responses,
  prompts,
}: {
  models: string[];
  valid: ScoreRow[];
  responses: ResponseRow[];
  prompts: Prompt[];
}) {
  const [sortBy, setSortBy] =
    useState<(typeof SORT_OPTIONS)[number]>("composite_score");
  const [catFilter, setCatFilter] = useState<string>("all");
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const categories = useMemo(
    () => [...new Set(prompts.map((p) => p.category))],
    [prompts],
  );

  const promptIds = useMemo(() => {
    const withScores = new Set(valid.map((s) => s.prompt_id));
    let ids = prompts.filter((p) => withScores.has(p.id));
    if (catFilter !== "all") ids = ids.filter((p) => p.category === catFilter);
    // order prompts by the best model score for the selected sort key
    return ids
      .map((p) => ({
        p,
        best: Math.max(
          ...valid.filter((s) => s.prompt_id === p.id).map((s) => Number(s[sortBy]) || 0),
          0,
        ),
      }))
      .sort((a, b) => b.best - a.best)
      .map((x) => x.p);
  }, [prompts, valid, catFilter, sortBy]);

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">Per-prompt drill-down</div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label className="font-mono text-[0.7rem] uppercase tracking-[0.08em] text-mute">
          Sort by
        </label>
        <select
          className="panel-input font-mono text-[0.75rem]"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
        >
          {SORT_OPTIONS.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <label className="ml-2 font-mono text-[0.7rem] uppercase tracking-[0.08em] text-mute">
          Category
        </label>
        <select
          className="panel-input font-mono text-[0.75rem]"
          value={catFilter}
          onChange={(e) => setCatFilter(e.target.value)}
        >
          <option value="all">all</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        {promptIds.map((p) => {
          const rows = valid.filter((s) => s.prompt_id === p.id);
          const isOpen = open[p.id] ?? false;
          return (
            <div key={p.id} className="rounded-lg border border-line bg-bg3">
              <button
                className="flex w-full flex-wrap items-center gap-2 px-4 py-3 text-left transition-colors hover:border-amber/30"
                onClick={() => setOpen((o) => ({ ...o, [p.id]: !isOpen }))}
              >
                {isOpen ? (
                  <ChevronDown size={14} className="flex-shrink-0 text-mute" />
                ) : (
                  <ChevronRight size={14} className="flex-shrink-0 text-mute" />
                )}
                <span className="font-mono text-[0.8rem] font-semibold">{p.id}</span>
                <span className="pill pill-mute">{p.category}</span>
                <span className="pill pill-mute">{p.difficulty}</span>
                <span className="flex-1" />
                {models.map((m) => {
                  const row = rows.find((r) => r.model === m);
                  if (!row) return null;
                  const mc = mcolor(m);
                  return (
                    <span
                      key={m}
                      className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[0.7rem]"
                      style={{
                        background: `${mc}18`,
                        borderColor: `${mc}44`,
                      }}
                    >
                      <span style={{ color: mc, fontWeight: 600 }}>
                        {shortName(m).split(" ")[0]}
                      </span>
                      <span className="text-mute">·</span>
                      <span
                        style={{
                          color: scoreColor(row.composite_score),
                          fontWeight: 600,
                        }}
                      >
                        {fmt(row.composite_score)}
                      </span>
                    </span>
                  );
                })}
              </button>

              {isOpen && (
                <div className="border-t border-line px-4 py-4">
                  <div className="mb-4 grid gap-4 lg:grid-cols-2">
                    <div>
                      <div className="card-title">Prompt</div>
                      <div className="text-[0.85rem] leading-relaxed">{p.prompt}</div>
                    </div>
                    <div>
                      <div className="card-title">Ground truth</div>
                      <div className="text-[0.85rem] leading-relaxed text-mute">
                        {p.ground_truth || "—"}
                      </div>
                    </div>
                  </div>
                  <div
                    className="grid gap-5"
                    style={{
                      gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
                    }}
                  >
                    {models.map((m) => (
                      <ModelCell
                        key={m}
                        model={m}
                        score={rows.find((r) => r.model === m)}
                        response={responses.find(
                          (r) => r.model === m && r.prompt_id === p.id,
                        )}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
