"use client";

import { useCallback, useEffect, useState } from "react";
import { CatalogBrowser } from "@/components/eval/CatalogBrowser";
import { EvalPanel, type CostEstimator } from "@/components/eval/EvalPanel";
import { JudgePanel } from "@/components/eval/JudgePanel";
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

export default function Dashboard() {
  const data = useEvalData();
  const [keySet, setKeySet] = useState(false);

  useEffect(() => {
    api
      .health()
      .then((h) => setKeySet(h.api_key_set))
      .catch(() => setKeySet(false));
  }, []);

  const { refresh } = data;
  const stream = useEvalStream({
    onRow: data.mergeRow,
    onEfficiency: data.mergeEfficiency,
    onDone: useCallback(() => void refresh(), [refresh]),
  });

  // ?run=a,b — set by the ⌘K palette ("surprise me" / run model) from any page
  const { run } = stream;
  useEffect(() => {
    const param = new URLSearchParams(window.location.search).get("run");
    if (param) {
      window.history.replaceState(null, "", "/");
      run(param.split(",").filter(Boolean));
    }
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
          <KpiStrip
            models={models}
            valid={activeValid}
            responses={activeResponses}
            summary={activeSummary}
          />
          <VerdictBanner summary={activeSummary} />
          <QualitySection models={models} valid={activeValid} />
          <EfficiencySection
            models={models}
            responses={activeResponses}
            summary={activeSummary}
          />
          <ValueSection summary={activeSummary} />
          <CategorySection models={models} valid={activeValid} prompts={data.prompts} />
          <DrilldownSection
            models={models}
            valid={activeValid}
            responses={activeResponses}
            prompts={data.prompts}
          />
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
