"use client";

import { useCallback, useEffect, useState } from "react";
import { CatalogBrowser } from "@/components/eval/CatalogBrowser";
import { EvalPanel, type CostEstimator } from "@/components/eval/EvalPanel";
import { JudgePanel } from "@/components/eval/JudgePanel";
import { TempPanel } from "@/components/eval/TempPanel";
import { CategorySection } from "@/components/sections/CategorySection";
import { DrilldownSection } from "@/components/sections/DrilldownSection";
import { EfficiencySection } from "@/components/sections/EfficiencySection";
import { KpiStrip } from "@/components/sections/KpiStrip";
import { QualitySection } from "@/components/sections/QualitySection";
import { ValueSection } from "@/components/sections/ValueSection";
import { VerdictBanner } from "@/components/sections/VerdictBanner";
import { useEvalData } from "@/hooks/useEvalData";
import { useEvalStream } from "@/hooks/useEvalStream";
import { api } from "@/lib/api";
import { getVisitorKey } from "@/lib/key";

const SECTIONS = [
  "Overview",
  "Verdict",
  "Quality",
  "Efficiency",
  "Value",
  "Category",
  "Drill-down",
] as const;
type SectionName = (typeof SECTIONS)[number];
const SECTIONS_KEY = "llm-eval:hidden-sections";

export default function Dashboard() {
  const data = useEvalData();
  const [keySet, setKeySet] = useState(false);
  const [hiddenSections, setHiddenSections] = useState<SectionName[]>([]);
  const [focusId, setFocusId] = useState<string | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SECTIONS_KEY);
      if (raw) setHiddenSections(JSON.parse(raw));
    } catch {
      /* ignore */
    }
  }, []);

  const setSections = (next: SectionName[]) => {
    setHiddenSections(next);
    try {
      localStorage.setItem(SECTIONS_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const show = (s: SectionName) => !hiddenSections.includes(s);
  const toggleSection = (s: SectionName) =>
    setSections(show(s) ? [...hiddenSections, s] : hiddenSections.filter((x) => x !== s));

  useEffect(() => {
    api
      .health()
      .then((h) => setKeySet(h.api_key_set || Boolean(getVisitorKey())))
      .catch(() => setKeySet(Boolean(getVisitorKey())));
  }, []);

  const { refresh } = data;
  const stream = useEvalStream({
    onRow: data.mergeRow,
    onEfficiency: data.mergeEfficiency,
    onDone: useCallback(() => void refresh(), [refresh]),
  });

  // ?run=a,b (⌘K run/surprise-me) and ?focus=<prompt_id> (⌘K prompt search)
  const { run } = stream;
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const runParam = params.get("run");
    const focusParam = params.get("focus");
    if (runParam || focusParam) window.history.replaceState(null, "", "/");
    if (runParam) run(runParam.split(",").filter(Boolean));
    if (focusParam) setFocusId(focusParam);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const estimateCost: CostEstimator = useCallback(
    (model, inTokens, outTokens) => {
      const s = data.summary.find((x) => x.model === model);
      if (!s?.input_price_per_1m && !s?.output_price_per_1m) return null;
      return (
        (inTokens / 1e6) * (s?.input_price_per_1m ?? 0) +
        (outTokens / 1e6) * (s?.output_price_per_1m ?? 0)
      );
    },
    [data.summary],
  );

  const models = data.activeModels;
  const activeSummary = data.summary.filter((s) => models.includes(s.model));
  const activeValid = data.valid.filter((s) => models.includes(s.model));
  const activeResponses = data.responses.filter((r) => models.includes(r.model));

  return (
    <div className="mx-auto max-w-[1400px]">
      {/* Top bar */}
      <header className="flex items-end justify-between pb-2 pt-1">
        <div>
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
            LLM Eval Framework
          </div>
          <h1 className="text-[1.6rem] font-semibold leading-tight">
            Model Comparison Dashboard
          </h1>
          <div className="text-[0.82rem] text-mute">
            Quality · Efficiency · Value-per-$20 · 300+ models via OpenRouter
          </div>
        </div>
      </header>

      <hr className="hdivider" />

      <EvalPanel
        activeModels={models}
        hiddenModels={data.hiddenModels}
        evaluatedModels={data.evaluatedModels}
        keySet={keySet}
        runs={stream.runs}
        queue={stream.queue}
        onRun={stream.run}
        onHide={data.hideModel}
        onRestore={data.restoreModel}
        estimateCost={estimateCost}
      />

      <hr className="hdivider" />

      <CatalogBrowser onPick={(m) => stream.run([m])} />

      <div className="mt-3">
        <JudgePanel activeModels={models} />
      </div>

      <div className="mt-3">
        <TempPanel activeModels={models} />
      </div>

      {data.loading ? (
        <div className="py-16 text-center font-mono text-[0.8rem] text-mute">
          loading results…
        </div>
      ) : data.loadError ? (
        <div className="mt-6 rounded-lg border border-red/30 bg-red/5 p-4 font-mono text-[0.8rem] text-red">
          Backend unreachable: {data.loadError} — is uvicorn running on :8000?
        </div>
      ) : models.length === 0 ? (
        <div className="mt-6 rounded-lg border border-line bg-bg2 p-6 text-center text-[0.85rem] text-mute">
          Add a model above (or restore a hidden one) to see the comparison.
        </div>
      ) : (
        <>
          <hr className="hdivider" />
          {/* Section visibility chips */}
          <div className="mb-1 flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[0.6rem] uppercase tracking-[0.12em] text-mute">
              sections:
            </span>
            {SECTIONS.map((s) => (
              <button
                key={s}
                className={`pill ${show(s) ? "pill-amber" : "pill-mute"}`}
                onClick={() => toggleSection(s)}
              >
                {s}
              </button>
            ))}
            <button className="pill pill-mute" onClick={() => setSections([])}>
              expand all
            </button>
            <button className="pill pill-mute" onClick={() => setSections([...SECTIONS])}>
              collapse all
            </button>
          </div>
          {show("Overview") && (
            <KpiStrip
              models={models}
              valid={activeValid}
              responses={activeResponses}
              summary={activeSummary}
            />
          )}
          {show("Verdict") && <VerdictBanner summary={activeSummary} />}
          {show("Quality") && <QualitySection models={models} valid={activeValid} />}
          {show("Efficiency") && (
            <EfficiencySection
              models={models}
              responses={activeResponses}
              summary={activeSummary}
            />
          )}
          {show("Value") && <ValueSection summary={activeSummary} />}
          {show("Category") && (
            <CategorySection models={models} valid={activeValid} prompts={data.prompts} />
          )}
          {show("Drill-down") && (
            <DrilldownSection
              models={models}
              valid={activeValid}
              responses={activeResponses}
              prompts={data.prompts}
              focusId={focusId}
            />
          )}
        </>
      )}

      <hr className="hdivider" />
      <footer className="flex items-center justify-between pb-6 font-mono text-[0.68rem] text-mute">
        <span>LLM Eval Framework · Abhishek · AI Quality Engineering Portfolio</span>
        <span>V2 — FastAPI + Next.js · CSV persistence · one OpenRouter key</span>
      </footer>
    </div>
  );
}
