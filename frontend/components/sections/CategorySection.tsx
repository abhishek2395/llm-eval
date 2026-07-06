"use client";

import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { mcolor } from "@/lib/colors";
import { categoryMeans, mean } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import type { Prompt, ScoreRow } from "@/lib/types";

export function CategorySection({
  models,
  valid,
  prompts,
}: {
  models: string[];
  valid: ScoreRow[];
  prompts: Prompt[];
}) {
  const { categories, byModel } = categoryMeans(valid, prompts, models);
  if (!categories.length) return null;

  const traces: Data[] = models.map(
    (m) =>
      ({
        type: "bar",
        name: shortName(m),
        x: categories,
        y: categories.map((c) => byModel.get(m)?.get(c) ?? null),
        marker: { color: mcolor(m) },
      }) as Data,
  );

  // Hallucination focus — trap prompts only
  const hallIds = new Set(
    prompts.filter((p) => p.category === "hallucination_test").map((p) => p.id),
  );
  const hallMeans = models
    .map((m) => ({
      model: m,
      v: mean(
        valid
          .filter((s) => s.model === m && hallIds.has(s.prompt_id))
          .map((s) => s.hallucination_resistance),
      ),
    }))
    .filter((r) => r.v > 0);

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">Performance by category</div>

      <Plot
        data={traces}
        layout={baseLayout({
          height: 300,
          barmode: "group",
          bargap: 0.25,
          bargroupgap: 0.1,
          margin: { l: 42, r: 8, t: 8, b: 32 },
          yaxis: { range: [0, 5.4], title: { text: "Composite score", font: { size: 10 } } },
          legend: {
            orientation: "h",
            y: 1.12,
            x: 0,
            bgcolor: "rgba(0,0,0,0)",
            borderwidth: 0,
            font: { size: 11 },
          },
        })}
        config={PLOT_CONFIG}
        className="w-full"
        useResizeHandler
      />

      {hallMeans.length > 0 && (
        <div className="mt-2 lg:w-2/3">
          <div className="card-title">
            Hallucination resistance — trap prompts only
          </div>
          <Plot
            data={[
              {
                type: "bar",
                x: hallMeans.map((r) => shortName(r.model)),
                y: hallMeans.map((r) => r.v),
                marker: { color: hallMeans.map((r) => mcolor(r.model)) },
                text: hallMeans.map((r) => `${r.v.toFixed(2)}/5`),
                textposition: "outside",
                textfont: { family: MONO, size: 13 },
              } as Data,
            ]}
            layout={baseLayout({
              height: 220,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 28 },
              yaxis: { range: [0, 5.8] },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      )}
    </section>
  );
}
