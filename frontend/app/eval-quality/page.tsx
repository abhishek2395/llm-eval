"use client";

/**
 * Eval Quality — the eval of the eval. Judge calibration against golden
 * tests, bootstrap confidence intervals, judge repeatability, measured
 * model consistency, and honest limitations.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { Data } from "plotly.js";
import { ShieldCheck, Play } from "lucide-react";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { API_BASE } from "@/lib/api";
import { mcolor, PALETTE } from "@/lib/colors";
import { fmt } from "@/lib/derive";
import { withVisitorKey } from "@/lib/key";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";

interface ConfidenceModel {
  model: string;
  n_prompts: number;
  mean: number;
  ci_low: number;
  ci_high: number;
  self_judged: boolean;
}
interface CalCheck {
  dimension: string;
  expected: [number, number] | boolean;
  got: number | boolean | null;
  ok: boolean;
}
interface CalJudge {
  judge: string;
  items_run: number;
  pass_rate: number;
  per_dimension: Record<string, { pass: number; fail: number; accuracy: number }>;
  failures: { cal_id: string; label: string; failure_mode_tested: string; checks: CalCheck[] }[];
}
interface RepJudge {
  judge: string;
  responses: number;
  trials: number;
  sigma_per_dimension: Record<string, number>;
}
interface ConsistencyReport {
  model: string;
  consistency: number | null;
  prompts: { prompt_id: string; scores: number[]; std: number }[];
}

function useSSERun(onDone: () => void) {
  const [run, setRun] = useState<{ completed: number; total: number; stage: string } | null>(null);
  const esRef = useRef<EventSource | null>(null);
  useEffect(() => () => esRef.current?.close(), []);
  const start = (url: string) => {
    esRef.current?.close();
    setRun({ completed: 0, total: 0, stage: "starting" });
    const es = new EventSource(withVisitorKey(url));
    esRef.current = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "start") setRun((r) => r && { ...r, total: ev.total, stage: "running" });
      else if (ev.type === "result" || ev.type === "cached")
        setRun((r) => r && { ...r, completed: r.completed + 1 });
      else if (ev.type === "done") {
        setRun((r) => r && { ...r, stage: "done" });
        es.close();
        onDone();
      } else if (ev.type === "error") {
        setRun((r) => r && { ...r, stage: "error" });
        es.close();
      }
    };
    es.onerror = () => {
      setRun((r) => (r && r.stage !== "done" ? { ...r, stage: "error" } : r));
      es.close();
    };
  };
  return { run, start };
}

function MiniProgress({ run }: { run: { completed: number; total: number; stage: string } | null }) {
  if (!run) return null;
  const pct = run.stage === "done" ? 100 : run.total ? (run.completed / run.total) * 100 : 5;
  return (
    <div className="my-2 lg:w-1/2">
      <div className="mb-1 text-right font-mono text-[0.65rem] text-mute">
        {run.stage === "done" ? `✓ ${run.completed} done` : run.stage === "error" ? "✕ failed" : `${run.completed}/${run.total || "…"}`}
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: run.stage === "error" ? "var(--red)" : "var(--purple)" }}
        />
      </div>
    </div>
  );
}

const LIMITATIONS = [
  "Sample size: 15 prompts per model. Bootstrap CIs above make the implied precision explicit — most adjacent leaderboard positions are statistical ties.",
  "Single default judge: composite scores come from one judge model. Self-judged rows (judge and model share a provider) are flagged; the multi-judge panel on the dashboard is the mitigation.",
  "Judge calibration is necessary, not sufficient: passing 16 golden tests bounds obvious failure modes (eloquence bias, length leniency, hallucination blindness) but cannot prove correctness on arbitrary responses.",
  "The agentic sandbox uses deterministic mock tools — it measures tool-use competence, not real-world API robustness.",
  "Rows judged before the rubric upgrade lack refusal/confidence fields; old April-era scores were judged without ground truths (visible as multi-judge disagreements).",
  "Consistency is measured only for models where the repeat-run panel has been executed; other models show a legacy estimate.",
];

export default function EvalQualityPage() {
  const [confidence, setConfidence] = useState<{ models: ConfidenceModel[]; ties: { model_a: string; model_b: string }[]; judge: string } | null>(null);
  const [calibration, setCalibration] = useState<{ judges: CalJudge[]; n_items: number } | null>(null);
  const [repeatability, setRepeatability] = useState<{ judges: RepJudge[] } | null>(null);
  const [consModel, setConsModel] = useState("deepseek/deepseek-chat");
  const [consReport, setConsReport] = useState<ConsistencyReport | null>(null);
  const [calJudge, setCalJudge] = useState("");

  const refresh = useCallback(() => {
    fetch(`${API_BASE}/meta-eval/confidence`).then((r) => r.json()).then(setConfidence).catch(() => {});
    fetch(`${API_BASE}/meta-eval/calibration/results`).then((r) => r.json()).then(setCalibration).catch(() => {});
    fetch(`${API_BASE}/meta-eval/repeatability/results`).then((r) => r.json()).then(setRepeatability).catch(() => {});
  }, []);
  const loadConsistency = useCallback((m: string) => {
    fetch(`${API_BASE}/meta-eval/consistency/results?model=${encodeURIComponent(m)}`)
      .then((r) => r.json()).then(setConsReport).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    loadConsistency("deepseek/deepseek-chat");
  }, [refresh, loadConsistency]);

  const calRun = useSSERun(refresh);
  const repRun = useSSERun(refresh);
  const consRun = useSSERun(() => loadConsistency(consModel));

  const ciTrace: Data[] = confidence
    ? [
        {
          type: "scatter",
          mode: "markers",
          y: confidence.models.map((m) => shortName(m.model)),
          x: confidence.models.map((m) => m.mean),
          error_x: {
            type: "data",
            symmetric: false,
            array: confidence.models.map((m) => m.ci_high - m.mean),
            arrayminus: confidence.models.map((m) => m.mean - m.ci_low),
            color: PALETTE.textMute,
            thickness: 1.5,
            width: 6,
          },
          marker: {
            size: 12,
            color: confidence.models.map((m) => mcolor(m.model)),
            line: { width: 2, color: PALETTE.bg3 },
          },
          text: confidence.models.map((m) => `${fmt(m.mean)} [${fmt(m.ci_low)}, ${fmt(m.ci_high)}]`),
          textfont: { family: MONO, size: 10 },
        } as unknown as Data,
      ]
    : [];

  return (
    <div className="mx-auto max-w-[1300px]">
      <header className="pb-2 pt-1">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
          LLM Eval Framework
        </div>
        <h1 className="flex items-center gap-2 text-[1.6rem] font-semibold leading-tight">
          <ShieldCheck size={24} className="text-amber" /> Eval Quality
        </h1>
        <div className="text-[0.82rem] text-mute">
          The eval of the eval — can you trust these scores? Judge calibration
          against golden tests, statistical confidence, repeatability, and
          measured consistency.
        </div>
      </header>

      <hr className="hdivider" />

      {/* 1 — Statistical confidence */}
      <div className="section-hdr">
        Statistical confidence — bootstrap 95% CIs on composite score
      </div>
      {confidence && (
        <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
          <Plot
            data={ciTrace}
            layout={baseLayout({
              height: 60 + confidence.models.length * 44,
              showlegend: false,
              margin: { l: 150, r: 30, t: 8, b: 30 },
              xaxis: { range: [3.5, 5.05], title: { text: "composite (95% CI)", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
          <div>
            <div className="card">
              <div className="card-title">What the intervals say</div>
              <div className="mb-2 text-[0.82rem] leading-relaxed">
                {confidence.ties.length} of{" "}
                {(confidence.models.length * (confidence.models.length - 1)) / 2}{" "}
                pairwise comparisons are <b>statistical ties</b> at n=
                {confidence.models[0]?.n_prompts ?? "?"} prompts — overlapping
                intervals mean the leaderboard order between those models is
                not statistically established.
              </div>
              <div className="mb-1 font-mono text-[0.65rem] uppercase tracking-[0.1em] text-mute">
                Self-judgment flags
              </div>
              {confidence.models.filter((m) => m.self_judged).map((m) => (
                <span key={m.model} className="pill pill-amber">
                  {shortName(m.model)} — judged by same provider
                </span>
              ))}
              <div className="mt-2 text-[0.72rem] text-mute">
                Judge: {shortName(confidence.judge)} · mitigation: multi-judge
                panel on the dashboard
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2 — Judge calibration */}
      <hr className="hdivider" />
      <div className="section-hdr">
        Judge calibration — {calibration?.n_items ?? 16} golden tests with known answers
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          className="panel-input min-w-64 font-mono text-[0.78rem]"
          placeholder="judge to calibrate (empty = default)"
          value={calJudge}
          onChange={(e) => setCalJudge(e.target.value)}
        />
        <button
          className="panel-btn panel-btn-primary flex items-center gap-1.5"
          onClick={() =>
            calRun.start(
              `${API_BASE}/meta-eval/calibration/stream${calJudge.trim() ? `?judge=${encodeURIComponent(calJudge.trim())}` : ""}`,
            )
          }
        >
          <Play size={12} /> Run calibration
        </button>
        <span className="font-mono text-[0.68rem] text-mute">
          planted hallucinations · eloquent-but-wrong · bloat · format violations · refusals
        </span>
      </div>
      <MiniProgress run={calRun.run} />
      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
        {calibration?.judges.map((j) => (
          <div key={j.judge} className="card">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="font-mono text-[0.85rem] font-semibold" style={{ color: mcolor(j.judge) }}>
                {shortName(j.judge)}
              </span>
              <span
                className="font-mono text-[1.4rem] font-semibold"
                style={{ color: j.pass_rate >= 0.95 ? "var(--green)" : j.pass_rate >= 0.8 ? "var(--amber)" : "var(--red)" }}
              >
                {Math.round(j.pass_rate * 100)}%
              </span>
            </div>
            {Object.entries(j.per_dimension).map(([dim, s]) => (
              <div key={dim} className="score-row">
                <span className="score-label">{dim.replaceAll("_", " ")}</span>
                <div className="score-bar-bg">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${s.accuracy * 100}%`,
                      background: s.accuracy >= 0.99 ? "var(--green)" : s.accuracy >= 0.75 ? "var(--amber)" : "var(--red)",
                    }}
                  />
                </div>
                <span className="score-val">{s.pass}/{s.pass + s.fail}</span>
              </div>
            ))}
            {j.failures.map((f) => (
              <div key={f.cal_id} className="mt-2 rounded-lg border border-red/25 bg-red/5 px-3 py-2 font-mono text-[0.7rem]">
                ✗ {f.cal_id} <span className="text-mute">({f.label})</span>
                {f.checks.map((c, i) => (
                  <div key={i} className="text-mute">
                    {c.dimension}: expected {JSON.stringify(c.expected)}, got{" "}
                    <span className="text-red">{String(c.got)}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* 3 — Repeatability + 4 — Consistency */}
      <hr className="hdivider" />
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <div className="section-hdr">Judge repeatability — same response, 3 trials</div>
          <button
            className="panel-btn panel-btn-primary mb-2 flex items-center gap-1.5"
            onClick={() => repRun.start(`${API_BASE}/meta-eval/repeatability/stream`)}
          >
            <Play size={12} /> Run repeatability
          </button>
          <MiniProgress run={repRun.run} />
          {repeatability?.judges.map((j) => (
            <div key={j.judge} className="card mt-2">
              <div className="card-title" style={{ color: mcolor(j.judge) }}>
                {shortName(j.judge)} — {j.responses} responses × {j.trials} trials
              </div>
              {Object.entries(j.sigma_per_dimension).map(([dim, sig]) => (
                <div key={dim} className="flex justify-between font-mono text-[0.73rem]">
                  <span className="text-mute">{dim.replaceAll("_", " ")}</span>
                  <span style={{ color: sig <= 0.1 ? "var(--green)" : sig <= 0.3 ? "var(--amber)" : "var(--red)" }}>
                    σ = {fmt(sig, 3)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div>
          <div className="section-hdr">
            Model consistency — same prompt regenerated ×3 (replaces V1&apos;s hardcoded 0.85)
          </div>
          <div className="mb-2 flex flex-wrap gap-2">
            <input
              className="panel-input min-w-56 font-mono text-[0.75rem]"
              value={consModel}
              onChange={(e) => {
                setConsModel(e.target.value);
                loadConsistency(e.target.value);
              }}
            />
            <button
              className="panel-btn panel-btn-primary flex items-center gap-1.5"
              onClick={() =>
                consRun.start(`${API_BASE}/meta-eval/consistency/stream?model=${encodeURIComponent(consModel)}`)
              }
            >
              <Play size={12} /> Measure
            </button>
          </div>
          <MiniProgress run={consRun.run} />
          {consReport && consReport.consistency != null && (
            <div className="card mt-2">
              <div className="mb-2 flex items-baseline justify-between">
                <span className="font-mono text-[0.85rem] font-semibold" style={{ color: mcolor(consReport.model) }}>
                  {shortName(consReport.model)}
                </span>
                <span className="font-mono text-[1.4rem] font-semibold text-green">
                  {fmt(consReport.consistency, 3)}
                </span>
              </div>
              {consReport.prompts.slice(0, 7).map((p) => (
                <div key={p.prompt_id} className="flex justify-between font-mono text-[0.72rem]">
                  <span className="text-mute">{p.prompt_id}</span>
                  <span>
                    [{p.scores.join(", ")}]{" "}
                    <span style={{ color: p.std <= 0.15 ? "var(--green)" : "var(--amber)" }}>σ {fmt(p.std, 2)}</span>
                  </span>
                </div>
              ))}
              <div className="mt-2 text-[0.7rem] text-mute">
                consistency = 1 − mean per-prompt σ ÷ 2 · feeds the dashboard&apos;s
                consistency chart for this model
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 5 — Limitations */}
      <hr className="hdivider" />
      <div className="section-hdr">Known limitations — stated, not hidden</div>
      <div className="mb-8 flex flex-col gap-2">
        {LIMITATIONS.map((l, i) => (
          <div key={i} className="rounded-lg border border-line bg-bg2 px-4 py-2.5 text-[0.8rem] leading-relaxed text-mute">
            {i + 1}. {l}
          </div>
        ))}
      </div>
    </div>
  );
}
