"use client";

/**
 * JudgePanel — run additional judge models over the stored responses and
 * show inter-judge agreement + flagged disagreements.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Scale } from "lucide-react";
import { shortName } from "@/lib/aliases";
import { api } from "@/lib/api";
import { mcolor } from "@/lib/colors";
import { fmt } from "@/lib/derive";
import type { AgreementResponse } from "@/lib/types";

const JUDGE_SUGGESTIONS = [
  "anthropic/claude-sonnet-5",
  "openai/gpt-4o",
  "google/gemini-2.5-pro",
  "deepseek/deepseek-r1",
];

interface JudgeRun {
  judge: string;
  completed: number;
  total: number;
  stage: "starting" | "judging" | "done" | "error";
  message?: string;
}

export function JudgePanel({ activeModels }: { activeModels: string[] }) {
  const [open, setOpen] = useState(false);
  const [judge, setJudge] = useState(JUDGE_SUGGESTIONS[0]);
  const [run, setRun] = useState<JudgeRun | null>(null);
  const [agreement, setAgreement] = useState<AgreementResponse | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const loadAgreement = useCallback(() => {
    api.judgeAgreement().then(setAgreement).catch(() => setAgreement(null));
  }, []);

  useEffect(() => {
    if (open) loadAgreement();
    return () => esRef.current?.close();
  }, [open, loadAgreement]);

  const start = () => {
    if (!judge.trim() || !activeModels.length) return;
    esRef.current?.close();
    setRun({ judge, completed: 0, total: 0, stage: "starting" });
    const es = new EventSource(api.judgeStreamUrl(judge.trim(), activeModels));
    esRef.current = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "start")
        setRun((r) => r && { ...r, total: ev.total, stage: "judging" });
      else if (ev.type === "result")
        setRun((r) => r && { ...r, completed: r.completed + 1 });
      else if (ev.type === "done") {
        setRun((r) => r && { ...r, stage: "done" });
        es.close();
        loadAgreement();
      }
    };
    es.onerror = () => {
      setRun((r) => (r && r.stage !== "done" ? { ...r, stage: "error", message: "stream disconnected" } : r));
      es.close();
    };
  };

  const judges = agreement?.judges ?? [];
  const pct = useMemo(
    () => (run && run.total ? Math.round((run.completed / run.total) * 100) : 0),
    [run],
  );

  return (
    <section>
      <button
        className="flex w-full items-center gap-2 rounded-lg border border-line bg-bg2 px-4 py-3 text-left font-mono text-[0.8rem] text-mute transition-colors hover:text-ink"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Scale size={13} />
        Multi-judge panel
        {judges.length > 1 && ` — ${judges.length} judges active`}
      </button>

      {open && (
        <div className="mt-3 rounded-lg border border-line bg-bg2 p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <input
              className="panel-input min-w-64 font-mono text-[0.78rem]"
              list="judge-suggestions"
              value={judge}
              onChange={(e) => setJudge(e.target.value)}
              placeholder="judge model ID"
            />
            <datalist id="judge-suggestions">
              {JUDGE_SUGGESTIONS.map((j) => (
                <option key={j} value={j} />
              ))}
            </datalist>
            <button
              className="panel-btn panel-btn-primary"
              disabled={!judge.trim() || !activeModels.length || run?.stage === "judging"}
              onClick={start}
            >
              ⚖ Judge {activeModels.length} models
            </button>
            <span className="font-mono text-[0.68rem] text-mute">
              re-scores stored responses with a second judge — no model calls
            </span>
          </div>

          {run && (
            <div className="mb-4">
              <div className="mb-1 flex justify-between font-mono text-[0.7rem]">
                <span style={{ color: mcolor(run.judge) }}>{shortName(run.judge)}</span>
                <span className="text-mute">
                  {run.stage === "done"
                    ? `✓ judged ${run.completed} responses`
                    : run.stage === "error"
                      ? `✕ ${run.message}`
                      : `${run.completed}/${run.total || "…"}`}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${run.stage === "done" ? 100 : pct}%`,
                    background:
                      run.stage === "error" ? "var(--red)" : "var(--purple)",
                  }}
                />
              </div>
            </div>
          )}

          {agreement?.note && (
            <div className="font-mono text-[0.75rem] text-mute">{agreement.note}</div>
          )}

          {agreement && agreement.models.length > 0 && (
            <>
              <div className="card-title">Inter-judge agreement</div>
              <div className="mb-4 overflow-x-auto rounded-lg border border-line">
                <table className="w-full border-collapse font-mono text-[0.73rem]">
                  <thead>
                    <tr className="border-b-2 border-line bg-bg3">
                      <th className="px-3 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                        Model
                      </th>
                      {judges.map((j) => (
                        <th
                          key={j}
                          className="px-3 py-2 text-left text-[0.65rem]"
                          style={{ color: mcolor(j) }}
                        >
                          {shortName(j)}
                        </th>
                      ))}
                      <th className="px-3 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                        Agreement
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {agreement.models.map((m) => (
                      <tr key={m.model} className="border-b border-line last:border-0">
                        <td className="px-3 py-1.5" style={{ color: m.meta.color }}>
                          {m.meta.alias}
                        </td>
                        {judges.map((j) => (
                          <td key={j} className="px-3 py-1.5">
                            {m.per_judge_composite[j] != null
                              ? fmt(m.per_judge_composite[j])
                              : "—"}
                          </td>
                        ))}
                        <td
                          className="px-3 py-1.5 font-semibold"
                          style={{
                            color:
                              m.agreement >= 0.9
                                ? "var(--green)"
                                : m.agreement >= 0.75
                                  ? "var(--amber)"
                                  : "var(--red)",
                          }}
                        >
                          {fmt(m.agreement)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {agreement.disagreements.length > 0 && (
                <>
                  <div className="card-title">
                    Flagged disagreements — judges differ by &gt; 1.0
                  </div>
                  <div className="flex flex-col gap-1.5">
                    {agreement.disagreements.slice(0, 10).map((d, i) => (
                      <div
                        key={i}
                        className="rounded-lg border border-red/25 bg-red/5 px-3 py-2 font-mono text-[0.72rem]"
                      >
                        <span style={{ color: mcolor(d.model) }}>
                          {shortName(d.model)}
                        </span>{" "}
                        · {d.prompt_id} —{" "}
                        <span style={{ color: mcolor(d.judge_a) }}>
                          {shortName(d.judge_a)}: {d.score_a}
                        </span>{" "}
                        vs{" "}
                        <span style={{ color: mcolor(d.judge_b) }}>
                          {shortName(d.judge_b)}: {d.score_b}
                        </span>{" "}
                        <span className="text-red">Δ {d.diff}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
