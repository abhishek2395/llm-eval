"use client";

import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { mcolor, PALETTE } from "@/lib/colors";
import { fmtInt } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import type { EfficiencyRow } from "@/lib/types";

export function ValueSection({ summary }: { summary: EfficiencyRow[] }) {
  if (!summary.length) return null;

  const viSorted = [...summary].sort((a, b) => a.value_index - b.value_index);
  const apiSorted = [...summary].sort(
    (a, b) => a.api_answers_for_20usd - b.api_answers_for_20usd,
  );
  const udSorted = [...summary].sort((a, b) => a.useful_density - b.useful_density);
  const csSorted = [...summary].sort(
    (a, b) => a.consistency_score - b.consistency_score,
  );
  const meanLat =
    summary.reduce((a, r) => a + r.avg_total_latency_ms, 0) / summary.length;

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">
        Value for $20/month
        <span className="ml-2 text-[0.55rem] normal-case tracking-normal text-mute">
          Value Index = weighted quality ÷ verbosity penalty · API answers = $20 ÷
          cost-per-call
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div>
          <div className="card-title">Value Index (quality per token-dollar)</div>
          <Plot
            data={[
              {
                type: "bar",
                orientation: "h",
                x: viSorted.map((r) => r.value_index),
                y: viSorted.map((r) => shortName(r.model)),
                marker: {
                  color: viSorted.map((r) => r.value_index),
                  colorscale: [
                    [0, "#1a1a2e"],
                    [0.4, PALETTE.amberDim],
                    [1, PALETTE.amber],
                  ],
                  showscale: false,
                },
                text: viSorted.map((r) => r.value_index.toFixed(3)),
                textposition: "outside",
                textfont: { family: MONO, size: 12, color: PALETTE.text },
              } as Data,
            ]}
            layout={baseLayout({
              height: 220,
              margin: { l: 110, r: 60, t: 8, b: 8 },
              xaxis: {
                range: [0, Math.max(...summary.map((r) => r.value_index)) * 1.25],
              },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        <div>
          <div className="card-title">API answers you can get for $20</div>
          <Plot
            data={[
              {
                type: "bar",
                orientation: "h",
                x: apiSorted.map((r) => r.api_answers_for_20usd),
                y: apiSorted.map((r) => shortName(r.model)),
                marker: {
                  color: apiSorted.map((r) => r.api_answers_for_20usd),
                  colorscale: [
                    [0, "#0f4c5c"],
                    [1, PALETTE.teal],
                  ],
                  showscale: false,
                },
                text: apiSorted.map((r) => fmtInt(r.api_answers_for_20usd)),
                textposition: "outside",
                textfont: { family: MONO, size: 12, color: PALETTE.text },
              } as Data,
            ]}
            layout={baseLayout({
              height: 220,
              margin: { l: 110, r: 80, t: 8, b: 8 },
              xaxis: {
                range: [
                  0,
                  Math.max(...summary.map((r) => r.api_answers_for_20usd)) * 1.2,
                ],
              },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        <div>
          <div className="card-title">Quality vs latency trade-off</div>
          <Plot
            data={summary.map(
              (r) =>
                ({
                  type: "scatter",
                  mode: "text+markers",
                  x: [r.avg_total_latency_ms],
                  y: [r.composite_score],
                  name: shortName(r.model),
                  marker: {
                    size: 22,
                    color: mcolor(r.model),
                    line: { width: 2, color: PALETTE.bg3 },
                  },
                  text: [shortName(r.model)],
                  textposition: "top center",
                  textfont: { size: 10, family: MONO, color: mcolor(r.model) },
                }) as Data,
            )}
            layout={baseLayout({
              height: 220,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 32 },
              xaxis: { title: { text: "Avg latency (ms)", font: { size: 10 } } },
              yaxis: {
                title: { text: "Composite score", font: { size: 10 } },
                range: [0, 5.2],
              },
              shapes: [
                {
                  type: "line",
                  x0: meanLat,
                  x1: meanLat,
                  y0: 0,
                  y1: 5.2,
                  line: { dash: "dot", color: PALETTE.border, width: 1 },
                },
                {
                  type: "line",
                  xref: "paper",
                  x0: 0,
                  x1: 1,
                  y0: 3.5,
                  y1: 3.5,
                  line: { dash: "dot", color: PALETTE.border, width: 1 },
                },
              ],
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      </div>

      <div className="card-title mt-4">
        API pricing — input vs output cost per 1M tokens
      </div>
      <Plot
        data={[
          {
            type: "bar",
            name: "Input $/1M",
            x: summary.map((r) => shortName(r.model)),
            y: summary.map((r) => r.input_price_per_1m),
            marker: { color: summary.map((r) => mcolor(r.model)) },
            opacity: 0.65,
            text: summary.map((r) => `$${r.input_price_per_1m.toFixed(2)}`),
            textposition: "outside",
            textfont: { family: MONO, size: 10 },
          } as Data,
          {
            type: "bar",
            name: "Output $/1M",
            x: summary.map((r) => shortName(r.model)),
            y: summary.map((r) => r.output_price_per_1m),
            marker: { color: summary.map((r) => mcolor(r.model)) },
            opacity: 1.0,
            text: summary.map((r) => `$${r.output_price_per_1m.toFixed(2)}`),
            textposition: "outside",
            textfont: { family: MONO, size: 10 },
          } as Data,
        ]}
        layout={baseLayout({
          height: 280,
          barmode: "group",
          margin: { l: 42, r: 8, t: 8, b: 40 },
          yaxis: { title: { text: "USD per 1M tokens", font: { size: 10 } } },
        })}
        config={PLOT_CONFIG}
        className="w-full"
        useResizeHandler
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="card-title">
            Useful density — quality per 100 output tokens
          </div>
          <Plot
            data={[
              {
                type: "bar",
                x: udSorted.map((r) => shortName(r.model)),
                y: udSorted.map((r) => r.useful_density),
                marker: {
                  color: udSorted.map((r) => r.useful_density),
                  colorscale: [
                    [0, "#0f2027"],
                    [1, PALETTE.purple],
                  ],
                  showscale: false,
                },
                text: udSorted.map((r) => r.useful_density.toFixed(4)),
                textposition: "outside",
                textfont: { family: MONO, size: 12 },
              } as Data,
            ]}
            layout={baseLayout({
              height: 220,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 30 },
              yaxis: { title: { text: "score / 100 tokens", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
        <div>
          <div className="card-title">
            Consistency score (0–1, higher = more predictable output)
          </div>
          <Plot
            data={[
              {
                type: "bar",
                x: csSorted.map((r) => shortName(r.model)),
                y: csSorted.map((r) => r.consistency_score),
                marker: {
                  color: csSorted.map((r) => r.consistency_score),
                  colorscale: [
                    [0, "#0f2027"],
                    [1, PALETTE.green],
                  ],
                  showscale: false,
                },
                text: csSorted.map((r) => r.consistency_score.toFixed(3)),
                textposition: "outside",
                textfont: { family: MONO, size: 12 },
              } as Data,
            ]}
            layout={baseLayout({
              height: 220,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 30 },
              yaxis: { range: [0, 1.15], title: { text: "score", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      </div>
    </section>
  );
}
