"use client";

import { mcolor } from "@/lib/colors";
import { fmt, modelMean } from "@/lib/derive";
import type { EfficiencyRow, ResponseRow, ScoreRow } from "@/lib/types";

function Metric({
  label,
  value,
  delta,
  deltaColor,
}: {
  label: string;
  value: string;
  delta?: string;
  deltaColor?: string;
}) {
  return (
    <div className="rounded-[10px] border border-line bg-bg3 p-4">
      <div className="font-mono text-[0.7rem] uppercase tracking-[0.08em] text-mute">
        {label}
      </div>
      <div className="mt-1 font-mono text-[1.4rem] font-semibold text-ink">
        {value}
      </div>
      {delta && (
        <div
          className="font-mono text-[0.75rem]"
          style={{ color: deltaColor ?? "var(--text-mute)" }}
        >
          {delta}
        </div>
      )}
    </div>
  );
}

export function KpiStrip({
  models,
  valid,
  responses,
  summary,
}: {
  models: string[];
  valid: ScoreRow[];
  responses: ResponseRow[];
  summary: EfficiencyRow[];
}) {
  const composites = models.map((m) => modelMean(valid, m, "composite_score"));
  const bestComp = Math.max(...composites, 0);
  const bestVi = Math.max(...summary.map((s) => s.value_index), 0);

  return (
    <section>
      <div className="section-hdr">Overview</div>
      {models.map((model, i) => {
        const comp = composites[i];
        const vi = modelMean(valid, model, "value_index");
        const lat = modelMean(
          responses.filter((r) => r.total_latency_ms > 0),
          model,
          "total_latency_ms",
        );
        const eff = summary.find((s) => s.model === model);
        const ref = eff?.refusal_rate_pct ?? 0;
        return (
          <div key={model} className="mb-4">
            <div
              className="mb-1.5 mt-3 font-mono text-[0.68rem] uppercase tracking-[0.12em]"
              style={{ color: mcolor(model) }}
            >
              {model}
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Metric
                label="Quality"
                value={`${fmt(comp)}/5`}
                delta={
                  comp === bestComp
                    ? `best in set`
                    : `${fmt(comp - bestComp)} vs best`
                }
                deltaColor={comp === bestComp ? "var(--green)" : "var(--red)"}
              />
              <Metric
                label="Value Index"
                value={fmt(vi)}
                delta={eff && eff.value_index === bestVi ? "best" : undefined}
                deltaColor="var(--green)"
              />
              <Metric label="Avg Latency" value={lat ? `${fmt(lat, 0)} ms` : "—"} />
              <Metric
                label="Refusal Rate"
                value={`${fmt(ref, 0)}%`}
                delta={ref < 10 ? "↓ good" : "↑ high"}
                deltaColor={ref < 10 ? "var(--green)" : "var(--red)"}
              />
            </div>
          </div>
        );
      })}
    </section>
  );
}
