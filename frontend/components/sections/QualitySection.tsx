"use client";

import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { hexToRgba, mcolor, scoreColor, PALETTE } from "@/lib/colors";
import { dimensionMeans, fmt } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import { DIMENSIONS, type ScoreRow } from "@/lib/types";

const dimLabel = (d: string) =>
  d.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());

export function QualitySection({
  models,
  valid,
}: {
  models: string[];
  valid: ScoreRow[];
}) {
  const means = dimensionMeans(valid, models);

  const radarTraces: Data[] = models.map((m) => {
    const rec = means.get(m)!;
    const vals = DIMENSIONS.map((d) => rec[d]);
    const labs = DIMENSIONS.map(dimLabel);
    return {
      type: "scatterpolar",
      r: [...vals, vals[0]],
      theta: [...labs, labs[0]],
      fill: "toself",
      name: shortName(m),
      line: { color: mcolor(m), width: 2 },
      fillcolor: hexToRgba(mcolor(m)),
      opacity: 0.9,
    } as Data;
  });

  const heatY = models.map(shortName);
  const heatZ = models.map((m) => DIMENSIONS.map((d) => means.get(m)![d]));
  const heatX = DIMENSIONS.map((d) =>
    d.replaceAll("_", " ").replace("hallucination resistance", "halluc. resist."),
  );

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">Quality analysis</div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="card-title">Dimension radar</div>
          <Plot
            data={radarTraces}
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
              height: 340,
              margin: { l: 60, r: 60, t: 20, b: 20 },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        <div>
          <div className="card-title">Score heatmap — model × dimension</div>
          <Plot
            data={[
              {
                type: "heatmap",
                z: heatZ,
                x: heatX,
                y: heatY,
                colorscale: [
                  [0, "#1a1a2e"],
                  [0.3, "#7c2d12"],
                  [0.6, "#d97706"],
                  [1, "#22c55e"],
                ],
                zmin: 1,
                zmax: 5,
                text: heatZ.map((row) => row.map((v) => fmt(v))) as unknown as string[],
                texttemplate: "%{text}",
                textfont: { size: 13, family: MONO },
                hoverongaps: false,
                showscale: true,
                colorbar: {
                  tickfont: { size: 9, color: PALETTE.textMute },
                  outlinecolor: PALETTE.border,
                  outlinewidth: 1,
                  thickness: 12,
                  len: 0.9,
                },
              } as unknown as Data,
            ]}
            layout={baseLayout({
              height: 340,
              margin: { l: 110, r: 60, t: 10, b: 60 },
              xaxis: { tickangle: -20, tickfont: { size: 10 } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      </div>

      <div className="card-title mt-2">Dimension breakdown</div>
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}
      >
        {models.map((m) => {
          const rec = means.get(m)!;
          return (
            <div key={m} className="card">
              <div className="card-title" style={{ color: mcolor(m) }}>
                {shortName(m)}
              </div>
              {DIMENSIONS.map((d) => {
                const val = rec[d];
                const pct = ((val - 1) / 4) * 100;
                const col = scoreColor(val);
                return (
                  <div key={d} className="score-row">
                    <span className="score-label">{d.replaceAll("_", " ")}</span>
                    <div className="score-bar-bg">
                      <div
                        className="score-bar-fill"
                        style={{ width: `${Math.max(pct, 0)}%`, background: col }}
                      />
                    </div>
                    <span className="score-val" style={{ color: col }}>
                      {fmt(val)}
                    </span>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </section>
  );
}
