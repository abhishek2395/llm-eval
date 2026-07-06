"use client";

/**
 * Compare — side-by-side diff viewer, per-prompt radar, head-to-head win
 * rates, and run-over-run regression detection.
 */

import { useEffect, useMemo, useState } from "react";
import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { api } from "@/lib/api";
import { hexToRgba, mcolor, scoreColor, PALETTE } from "@/lib/colors";
import { wordDiff } from "@/lib/diff";
import { fmt, validScores } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import {
  DIMENSIONS,
  type Prompt,
  type ResponseRow,
  type ScoreRow,
} from "@/lib/types";

export default function ComparePage() {
  const [responses, setResponses] = useState<ResponseRow[]>([]);
  const [scores, setScores] = useState<ScoreRow[]>([]);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [runs, setRuns] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [promptId, setPromptId] = useState("");
  const [regressionRun, setRegressionRun] = useState("");
  const [regression, setRegression] = useState<
    { model: string; before: number; after: number; delta: number }[] | null
  >(null);

  useEffect(() => {
    Promise.all([api.results(), api.prompts(), api.runs()])
      .then(([r, p, rn]) => {
        setResponses(r.responses);
        setScores(r.scores);
        setPrompts(p);
        setRuns(rn.runs);
      })
      .catch((e) => setError(e.message));
  }, []);

  const valid = useMemo(() => validScores(scores), [scores]);
  const models = useMemo(
    () => [...new Set(valid.map((s) => s.model))].sort(),
    [valid],
  );

  useEffect(() => {
    if (models.length >= 2 && !modelA) {
      setModelA(models[0]);
      setModelB(models[1]);
    }
    if (prompts.length && !promptId) setPromptId(prompts[0].id);
  }, [models, prompts, modelA, promptId]);

  const respOf = (m: string) =>
    responses.find((r) => r.model === m && r.prompt_id === promptId);
  const scoreOf = (m: string) =>
    valid.find((s) => s.model === m && s.prompt_id === promptId);

  const diff = useMemo(() => {
    const a = respOf(modelA)?.response_text ?? "";
    const b = respOf(modelB)?.response_text ?? "";
    if (!a && !b) return null;
    return wordDiff(a, b);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelA, modelB, promptId, responses]);

  // Head-to-head: pairwise win counts per prompt on composite score
  const h2h = useMemo(() => {
    const wins = new Map<string, Map<string, number>>();
    const games = new Map<string, Map<string, number>>();
    for (const pid of new Set(valid.map((s) => s.prompt_id))) {
      const rows = valid.filter((s) => s.prompt_id === pid);
      for (const x of rows) {
        for (const y of rows) {
          if (x.model === y.model) continue;
          if (!wins.has(x.model)) wins.set(x.model, new Map());
          if (!games.has(x.model)) games.set(x.model, new Map());
          const g = games.get(x.model)!;
          g.set(y.model, (g.get(y.model) ?? 0) + 1);
          if (x.composite_score > y.composite_score) {
            const w = wins.get(x.model)!;
            w.set(y.model, (w.get(y.model) ?? 0) + 1);
          }
        }
      }
    }
    return { wins, games };
  }, [valid]);

  const winRate = (a: string, b: string): string => {
    const g = h2h.games.get(a)?.get(b) ?? 0;
    if (!g) return "—";
    const w = h2h.wins.get(a)?.get(b) ?? 0;
    return `${Math.round((w / g) * 100)}%`;
  };

  const overallWinPct = (m: string): number => {
    let w = 0, g = 0;
    for (const [, count] of h2h.games.get(m) ?? []) g += count;
    for (const [, count] of h2h.wins.get(m) ?? []) w += count;
    return g ? (w / g) * 100 : 0;
  };

  const leaderboard = [...models].sort((a, b) => overallWinPct(b) - overallWinPct(a));

  const checkRegression = async (stamp: string) => {
    setRegressionRun(stamp);
    if (!stamp) {
      setRegression(null);
      return;
    }
    const old = await api.run(stamp);
    const oldValid = validScores(old.scores);
    const rows: { model: string; before: number; after: number; delta: number }[] = [];
    for (const m of models) {
      const oldRows = oldValid.filter((s) => s.model === m);
      const newRows = valid.filter((s) => s.model === m);
      if (!oldRows.length || !newRows.length) continue;
      const before = oldRows.reduce((a, s) => a + s.composite_score, 0) / oldRows.length;
      const after = newRows.reduce((a, s) => a + s.composite_score, 0) / newRows.length;
      rows.push({ model: m, before, after, delta: after - before });
    }
    setRegression(rows.sort((a, b) => a.delta - b.delta));
  };

  const renderDiff = (parts: { text: string; kind: string }[], side: "left" | "right") => (
    <div className="max-h-96 overflow-y-auto whitespace-pre-wrap rounded-lg border border-line bg-bg3 p-3 text-[0.82rem] leading-relaxed">
      {parts.map((p, i) => (
        <span
          key={i}
          style={
            p.kind === "same"
              ? undefined
              : {
                  background:
                    side === "left" ? "rgba(239,68,68,0.18)" : "rgba(34,197,94,0.18)",
                  borderRadius: 3,
                }
          }
        >
          {p.text}
        </span>
      ))}
    </div>
  );

  const radarFor = (pid: string): Data[] =>
    [modelA, modelB].filter(Boolean).map((m) => {
      const s = valid.find((x) => x.model === m && x.prompt_id === pid);
      const vals = DIMENSIONS.map((d) => (s ? Number(s[d]) : 0));
      const labs = DIMENSIONS.map((d) => d.replaceAll("_", " "));
      return {
        type: "scatterpolar",
        r: [...vals, vals[0]],
        theta: [...labs, labs[0]],
        fill: "toself",
        name: shortName(m),
        line: { color: mcolor(m), width: 2 },
        fillcolor: hexToRgba(mcolor(m)),
      } as Data;
    });

  if (error) {
    return (
      <div className="mx-auto max-w-[1100px] pt-4 font-mono text-[0.8rem] text-red">
        Backend unreachable: {error}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1300px]">
      <header className="pb-2 pt-1">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
          LLM Eval Framework
        </div>
        <h1 className="text-[1.6rem] font-semibold leading-tight">Compare</h1>
        <div className="text-[0.82rem] text-mute">
          Side-by-side diff · per-prompt radar · head-to-head win rate · regression
          detection
        </div>
      </header>

      <hr className="hdivider" />

      {/* Selectors */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <select
          className="panel-input font-mono text-[0.75rem]"
          value={modelA}
          onChange={(e) => setModelA(e.target.value)}
          style={{ color: mcolor(modelA) }}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {shortName(m)}
            </option>
          ))}
        </select>
        <span className="font-mono text-[0.7rem] text-mute">vs</span>
        <select
          className="panel-input font-mono text-[0.75rem]"
          value={modelB}
          onChange={(e) => setModelB(e.target.value)}
          style={{ color: mcolor(modelB) }}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {shortName(m)}
            </option>
          ))}
        </select>
        <span className="font-mono text-[0.7rem] text-mute">on</span>
        <select
          className="panel-input min-w-64 flex-1 font-mono text-[0.75rem]"
          value={promptId}
          onChange={(e) => setPromptId(e.target.value)}
        >
          {prompts.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id} — {p.prompt.slice(0, 70)}
            </option>
          ))}
        </select>
      </div>

      {/* Side-by-side diff */}
      <div className="grid gap-4 lg:grid-cols-2">
        {[modelA, modelB].map((m, side) => {
          const s = scoreOf(m);
          const parts = side === 0 ? diff?.left : diff?.right;
          return (
            <div key={`${m}-${side}`}>
              <div className="mb-1.5 flex items-baseline justify-between">
                <span
                  className="font-mono text-[0.85rem] font-semibold"
                  style={{ color: mcolor(m) }}
                >
                  {shortName(m)}
                </span>
                {s && (
                  <span
                    className="font-mono text-[1rem] font-bold"
                    style={{ color: scoreColor(s.composite_score) }}
                  >
                    {fmt(s.composite_score)}/5
                  </span>
                )}
              </div>
              {parts ? (
                renderDiff(parts, side === 0 ? "left" : "right")
              ) : (
                <div className="rounded-lg border border-line bg-bg3 p-3 font-mono text-[0.75rem] text-mute">
                  no response recorded
                </div>
              )}
              {s?.one_line_verdict && (
                <div
                  className="mt-2 rounded-r border-l-2 bg-white/[0.03] px-2.5 py-1.5 text-[0.78rem]"
                  style={{ borderLeftColor: mcolor(m) }}
                >
                  {s.one_line_verdict}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-2 font-mono text-[0.68rem] text-mute">
        <span style={{ background: "rgba(239,68,68,0.18)", padding: "1px 4px", borderRadius: 3 }}>
          red
        </span>{" "}
        only in {shortName(modelA)} ·{" "}
        <span style={{ background: "rgba(34,197,94,0.18)", padding: "1px 4px", borderRadius: 3 }}>
          green
        </span>{" "}
        only in {shortName(modelB)}
      </div>

      {/* Per-prompt radar */}
      <hr className="hdivider" />
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="section-hdr">Per-prompt radar — {promptId}</div>
          <Plot
            data={radarFor(promptId)}
            layout={baseLayout({
              polar: {
                radialaxis: {
                  visible: true,
                  range: [0, 5],
                  tickfont: { size: 9, color: PALETTE.textMute },
                  gridcolor: PALETTE.border,
                  linecolor: PALETTE.border,
                },
                angularaxis: {
                  tickfont: { size: 10, color: PALETTE.textMute },
                  gridcolor: PALETTE.border,
                  linecolor: PALETTE.border,
                },
                bgcolor: "rgba(0,0,0,0)",
              },
              height: 320,
              margin: { l: 60, r: 60, t: 20, b: 20 },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        {/* Head-to-head table */}
        <div>
          <div className="section-hdr">Head-to-head win rate — row beats column</div>
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full border-collapse font-mono text-[0.72rem]">
              <thead>
                <tr className="border-b-2 border-line bg-bg3">
                  <th className="px-2.5 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                    win %
                  </th>
                  {leaderboard.map((m) => (
                    <th
                      key={m}
                      className="px-2.5 py-2 text-left text-[0.65rem]"
                      style={{ color: mcolor(m) }}
                    >
                      {shortName(m).split(" ")[0]}
                    </th>
                  ))}
                  <th className="px-2.5 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute">
                    overall
                  </th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((a) => (
                  <tr key={a} className="border-b border-line last:border-0">
                    <td
                      className="px-2.5 py-1.5 font-semibold"
                      style={{ color: mcolor(a) }}
                    >
                      {shortName(a)}
                    </td>
                    {leaderboard.map((b) => (
                      <td key={b} className="px-2.5 py-1.5 text-mute">
                        {a === b ? "·" : winRate(a, b)}
                      </td>
                    ))}
                    <td className="px-2.5 py-1.5 font-semibold text-amber">
                      {Math.round(overallWinPct(a))}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Regression detection */}
      <hr className="hdivider" />
      <div className="section-hdr">Regression detection — current vs historical run</div>
      <div className="mb-3 flex items-center gap-2">
        <select
          className="panel-input font-mono text-[0.75rem]"
          value={regressionRun}
          onChange={(e) => void checkRegression(e.target.value)}
        >
          <option value="">select a historical run…</option>
          {runs.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>
      {regression && (
        <div className="overflow-x-auto rounded-lg border border-line lg:w-2/3">
          <table className="w-full border-collapse font-mono text-[0.75rem]">
            <thead>
              <tr className="border-b-2 border-line bg-bg3">
                {["Model", "Run " + regressionRun, "Current", "Δ", ""].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left text-[0.6rem] uppercase tracking-[0.1em] text-mute"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {regression.map((r) => (
                <tr key={r.model} className="border-b border-line last:border-0">
                  <td className="px-3 py-1.5" style={{ color: mcolor(r.model) }}>
                    {shortName(r.model)}
                  </td>
                  <td className="px-3 py-1.5 text-mute">{fmt(r.before)}</td>
                  <td className="px-3 py-1.5">{fmt(r.after)}</td>
                  <td
                    className="px-3 py-1.5 font-semibold"
                    style={{
                      color:
                        r.delta < -0.5
                          ? "var(--red)"
                          : r.delta > 0
                            ? "var(--green)"
                            : "var(--text-mute)",
                    }}
                  >
                    {r.delta >= 0 ? "+" : ""}
                    {fmt(r.delta)}
                  </td>
                  <td className="px-3 py-1.5">
                    {r.delta < -0.5 && <span className="pill pill-red">regression</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
