"use client";

/**
 * Agentic harness — multi-turn tool-use trajectories, judged on the
 * trajectory (task success, tool efficiency, honesty, reasoning).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Check, Play, X } from "lucide-react";
import { MODEL_ALIASES, shortName } from "@/lib/aliases";
import { api } from "@/lib/api";
import { mcolor, scoreColor } from "@/lib/colors";
import { fmt } from "@/lib/derive";
import type { AgenticRow, AgenticSummary, AgenticTask } from "@/lib/types";

interface RunState {
  model: string;
  completed: number;
  total: number;
  taskId?: string;
  stage: "starting" | "acting" | "judging" | "done" | "error";
  message?: string;
}

export default function AgenticPage() {
  const [tasks, setTasks] = useState<AgenticTask[]>([]);
  const [rows, setRows] = useState<AgenticRow[]>([]);
  const [summary, setSummary] = useState<AgenticSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [run, setRun] = useState<RunState | null>(null);
  const [selected, setSelected] = useState<{ model: string; task: string } | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const refresh = useCallback(() => {
    return Promise.all([api.agenticTasks(), api.agenticResults()])
      .then(([t, r]) => {
        setTasks(t.tasks);
        setRows(r.rows);
        setSummary(r.summary);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    void refresh();
    return () => esRef.current?.close();
  }, [refresh]);

  const evaluatedModels = useMemo(
    () => summary.map((s) => s.model),
    [summary],
  );

  const start = (m: string) => {
    if (!m.trim()) return;
    esRef.current?.close();
    setRun({ model: m, completed: 0, total: 0, stage: "starting" });
    const es = new EventSource(api.agenticStreamUrl(m.trim()));
    esRef.current = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "start")
        setRun((r) => r && { ...r, total: ev.total, stage: "acting" });
      else if (ev.type === "progress")
        setRun((r) => r && { ...r, taskId: ev.task_id, stage: ev.stage });
      else if (ev.type === "result" || ev.type === "cached")
        setRun((r) => r && { ...r, completed: r.completed + 1 });
      else if (ev.type === "done") {
        setRun((r) => r && { ...r, stage: "done" });
        es.close();
        void refresh();
      } else if (ev.type === "error") {
        setRun((r) => r && { ...r, stage: "error", message: ev.message });
        es.close();
      }
    };
    es.onerror = () => {
      setRun((r) =>
        r && r.stage !== "done" ? { ...r, stage: "error", message: "stream disconnected" } : r,
      );
      es.close();
    };
  };

  const rowFor = (m: string, t: string) =>
    rows.find(
      (r) => r.model === m && r.task_id === t && !r.error && !r.judge_error,
    );
  const selectedRow = selected ? rowFor(selected.model, selected.task) : undefined;
  const selectedTask = selected ? tasks.find((t) => t.id === selected.task) : undefined;

  return (
    <div className="mx-auto max-w-[1300px]">
      <header className="pb-2 pt-1">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
          LLM Eval Framework
        </div>
        <h1 className="flex items-center gap-2 text-[1.6rem] font-semibold leading-tight">
          <Bot size={24} className="text-amber" /> Agentic Harness
        </h1>
        <div className="text-[0.82rem] text-mute">
          Multi-turn tool-use trajectories in a deterministic mock sandbox —
          judged on task success, tool efficiency, honesty, and reasoning.
        </div>
      </header>

      <hr className="hdivider" />

      {error && (
        <div className="mb-4 rounded-lg border border-red/30 bg-red/5 p-4 font-mono text-[0.8rem] text-red">
          Backend unreachable: {error}
        </div>
      )}

      {/* Run controls */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <input
          className="panel-input min-w-72 font-mono text-[0.78rem]"
          list="agentic-models"
          placeholder="Model to run — e.g. deepseek/deepseek-chat"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && start(model)}
        />
        <datalist id="agentic-models">
          {Object.keys(MODEL_ALIASES).map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
        <button
          className="panel-btn panel-btn-primary flex items-center gap-1.5"
          disabled={!model.trim() || run?.stage === "acting" || run?.stage === "judging"}
          onClick={() => start(model)}
        >
          <Play size={12} /> Run {tasks.length} tasks
        </button>
        <span className="font-mono text-[0.68rem] mt-1 text-mute">
          completed tasks are cached — failed ones auto-retry
        </span>
      </div>

      {run && (
        <div className="mb-4 lg:w-2/3">
          <div className="mb-1 flex justify-between font-mono text-[0.7rem]">
            <span style={{ color: mcolor(run.model) }}>{shortName(run.model)}</span>
            <span className="text-mute">
              {run.stage === "done"
                ? `✓ ${run.completed} tasks complete`
                : run.stage === "error"
                  ? `✕ ${run.message}`
                  : `${run.stage} ${run.taskId ?? ""} (${run.completed}/${run.total || "…"})`}
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${run.stage === "done" ? 100 : run.total ? (run.completed / run.total) * 100 : 5}%`,
                background: run.stage === "error" ? "var(--red)" : "var(--amber)",
              }}
            />
          </div>
        </div>
      )}

      {/* Leaderboard */}
      {summary.length > 0 && (
        <>
          <div className="section-hdr">Agentic leaderboard</div>
          <div className="mb-6 overflow-x-auto rounded-lg border border-line">
            <table className="w-full border-collapse font-mono text-[0.75rem]">
              <thead>
                <tr className="border-b-2 border-line bg-bg3">
                  {["Model", "Composite", "Success", "Efficiency", "Honesty", "Avg steps", "Call validity", "Avg tokens"].map((h) => (
                    <th key={h} className="px-3 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {summary.map((s) => (
                  <tr key={s.model} className="border-b border-line last:border-0">
                    <td className="px-3 py-2 font-semibold" style={{ color: s.meta.color }}>
                      {s.meta.alias}
                    </td>
                    <td className="px-3 py-2 font-semibold" style={{ color: scoreColor(s.composite_avg) }}>
                      {fmt(s.composite_avg)}
                    </td>
                    <td className="px-3 py-2">{fmt(s.success_avg)}/5</td>
                    <td className="px-3 py-2">{fmt(s.efficiency_avg)}/5</td>
                    <td className="px-3 py-2">{fmt(s.honesty_avg)}/5</td>
                    <td className="px-3 py-2 text-mute">{s.avg_steps}</td>
                    <td className="px-3 py-2 text-mute">{fmt(s.validity_pct, 0)}%</td>
                    <td className="px-3 py-2 text-mute">{s.avg_tokens.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Task × model grid */}
          <div className="section-hdr">Task × model — click a cell for the trajectory</div>
          <div className="mb-6 overflow-x-auto rounded-lg border border-line">
            <table className="w-full border-collapse font-mono text-[0.72rem]">
              <thead>
                <tr className="border-b-2 border-line bg-bg3">
                  <th className="px-3 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                    Task
                  </th>
                  {evaluatedModels.map((m) => (
                    <th key={m} className="px-3 py-2 text-left text-[0.65rem]" style={{ color: mcolor(m) }}>
                      {shortName(m).split(" ").slice(0, 2).join(" ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t.id} className="border-b border-line last:border-0">
                    <td className="px-3 py-2 text-mute">{t.id}</td>
                    {evaluatedModels.map((m) => {
                      const r = rowFor(m, t.id);
                      const isSel = selected?.model === m && selected?.task === t.id;
                      return (
                        <td key={m} className="px-3 py-1.5">
                          {r ? (
                            <button
                              className={`rounded-md border px-2 py-0.5 font-semibold transition-colors ${isSel ? "border-amber" : "border-transparent"}`}
                              style={{
                                color: scoreColor(r.composite_score),
                                background: "rgba(255,255,255,0.04)",
                              }}
                              onClick={() => setSelected({ model: m, task: t.id })}
                            >
                              {fmt(r.composite_score, 1)}
                            </button>
                          ) : (
                            <span className="text-mute">—</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Trajectory viewer */}
      {selected && selectedRow && selectedTask && (
        <>
          <div className="section-hdr">
            Trajectory — {shortName(selected.model)} · {selected.task}
          </div>
          <div className="card mb-6">
            <div className="mb-3 text-[0.85rem]">{selectedTask.goal}</div>
            <div className="mb-3 flex flex-wrap gap-1.5">
              <span className={`pill ${selectedRow.task_success >= 4 ? "pill-green" : selectedRow.task_success >= 3 ? "pill-amber" : "pill-red"}`}>
                success {selectedRow.task_success}/5
              </span>
              <span className={`pill ${selectedRow.tool_efficiency >= 4 ? "pill-green" : selectedRow.tool_efficiency >= 3 ? "pill-amber" : "pill-red"}`}>
                efficiency {selectedRow.tool_efficiency}/5
              </span>
              <span className={`pill ${selectedRow.honesty >= 4 ? "pill-green" : "pill-red"}`}>
                honesty {selectedRow.honesty}/5
              </span>
              <span className="pill pill-mute">{selectedRow.steps} steps</span>
              <span className="pill pill-mute">
                {selectedRow.tool_calls_valid}/{selectedRow.tool_calls_total} valid calls
              </span>
              <span className="pill pill-mute">{fmt(selectedRow.latency_ms / 1000, 1)}s</span>
              <span className="pill pill-mute">
                {(selectedRow.input_tokens + selectedRow.output_tokens).toLocaleString()} tok
              </span>
            </div>

            {selectedRow.transcript.length ? (
              <div className="mb-3 flex flex-col gap-1.5">
                {selectedRow.transcript.map((c, i) => (
                  <div key={i} className="rounded-lg border border-line bg-bg3 px-3 py-2 font-mono text-[0.72rem]">
                    <span className="text-mute">{i + 1}.</span>{" "}
                    {c.valid ? (
                      <Check size={11} className="inline text-green" />
                    ) : (
                      <X size={11} className="inline text-red" />
                    )}{" "}
                    <span className="text-blue">{c.tool}</span>(
                    <span className="text-mute">{JSON.stringify(c.args)}</span>)
                    <span className="text-mute"> → {c.result.slice(0, 160)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="mb-3 font-mono text-[0.72rem] text-mute">
                no tool calls — answered directly
              </div>
            )}

            <div className="mb-1 font-mono text-[0.65rem] uppercase tracking-[0.1em] text-mute">
              Final answer
            </div>
            <div className="mb-3 whitespace-pre-wrap rounded-lg border border-line bg-bg3 p-3 text-[0.82rem]">
              {selectedRow.final_answer || "(none)"}
            </div>
            {selectedRow.one_line_verdict && (
              <div
                className="rounded-r border-l-2 bg-white/[0.03] px-2.5 py-1.5 text-[0.78rem]"
                style={{ borderLeftColor: mcolor(selected.model) }}
              >
                ⚖ {selectedRow.one_line_verdict}
              </div>
            )}
          </div>
        </>
      )}

      {/* Task definitions */}
      <div className="section-hdr">Task definitions</div>
      <div className="grid gap-3 pb-8" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        {tasks.map((t) => (
          <div key={t.id} className="card">
            <div className="card-title">{t.id}</div>
            <div className="mb-2 text-[0.82rem]">{t.goal}</div>
            <div className="mb-2">
              {t.tools.map((tool) => (
                <span key={tool} className="pill pill-blue">{tool}</span>
              ))}
              <span className="pill pill-mute">max {t.max_steps} steps</span>
            </div>
            <div className="text-[0.72rem] leading-relaxed text-mute">
              {t.success_criteria}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
