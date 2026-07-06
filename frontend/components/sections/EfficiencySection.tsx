"use client";

import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { shortName } from "@/lib/aliases";
import { hexToRgba, mcolor } from "@/lib/colors";
import { fmt, mean } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import type { EfficiencyRow, ResponseRow } from "@/lib/types";

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function EfficiencySection({
  models,
  responses,
  summary,
}: {
  models: string[];
  responses: ResponseRow[];
  summary: EfficiencyRow[];
}) {
  if (!responses.length) return null;

  const byModel = (m: string) => responses.filter((r) => r.model === m);

  const violinTraces: Data[] = models
    .map((m) => {
      const lats = byModel(m)
        .map((r) => r.total_latency_ms)
        .filter((v) => v > 0);
      if (!lats.length) return null;
      return {
        type: "violin",
        y: lats,
        name: shortName(m),
        box: { visible: true },
        meanline: { visible: true },
        line: { color: mcolor(m) },
        fillcolor: hexToRgba(mcolor(m), 0.13),
        opacity: 0.9,
      } as Data;
    })
    .filter(Boolean) as Data[];

  const ftl = models.map((m) => mean(byModel(m).map((r) => r.first_token_latency_ms)));
  const tot = models.map((m) => mean(byModel(m).map((r) => r.total_latency_ms)));
  const names = models.map(shortName);
  const colors = models.map(mcolor);

  const tps = models.map((m) =>
    mean(byModel(m).map((r) => r.tokens_per_second).filter((v) => v > 0)),
  );

  const boxTraces = (field: "output_tokens" | "verbosity_ratio"): Data[] =>
    models
      .map((m) => {
        const vals = byModel(m)
          .map((r) => r[field])
          .filter((v) => v > 0);
        if (!vals.length) return null;
        return {
          type: "box",
          y: vals,
          name: shortName(m),
          marker: { color: mcolor(m) },
          line: { color: mcolor(m) },
          fillcolor: hexToRgba(mcolor(m), 0.13),
        } as Data;
      })
      .filter(Boolean) as Data[];

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">Compute efficiency &amp; speed</div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div>
          <div className="card-title">Latency distribution (ms)</div>
          <Plot
            data={violinTraces}
            layout={baseLayout({
              height: 280,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 28 },
              yaxis: { title: { text: "ms", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        <div>
          <div className="card-title">First-token vs total latency (avg ms)</div>
          <Plot
            data={[
              {
                type: "bar",
                x: names,
                y: ftl,
                name: "First token",
                marker: { color: colors },
                opacity: 0.6,
              } as Data,
              {
                type: "bar",
                x: names,
                y: tot,
                name: "Total",
                marker: { color: colors },
                opacity: 1.0,
              } as Data,
            ]}
            layout={baseLayout({
              height: 280,
              barmode: "group",
              margin: { l: 42, r: 8, t: 8, b: 28 },
              yaxis: { title: { text: "ms", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>

        <div>
          <div className="card-title">Tokens per second</div>
          <Plot
            data={[
              {
                type: "bar",
                x: names,
                y: tps,
                marker: { color: colors },
                text: tps.map((v) => v.toFixed(0)),
                textposition: "outside",
                textfont: { family: MONO, size: 11 },
              } as Data,
            ]}
            layout={baseLayout({
              height: 280,
              showlegend: false,
              margin: { l: 42, r: 8, t: 28, b: 28 },
              yaxis: { title: { text: "tokens/sec", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="card-title">Output tokens per response</div>
          <Plot
            data={boxTraces("output_tokens")}
            layout={baseLayout({
              height: 250,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 28 },
              yaxis: { title: { text: "tokens", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
        <div>
          <div className="card-title">
            Verbosity ratio (output ÷ input tokens) — lower is more efficient
          </div>
          <Plot
            data={boxTraces("verbosity_ratio")}
            layout={baseLayout({
              height: 250,
              showlegend: false,
              margin: { l: 42, r: 8, t: 8, b: 28 },
              yaxis: { title: { text: "ratio", font: { size: 10 } } },
            })}
            config={PLOT_CONFIG}
            className="w-full"
            useResizeHandler
          />
        </div>
      </div>

      {/* Latency percentiles + context utilisation */}
      <div className="card-title mt-4">
        Latency percentiles &amp; context utilisation
      </div>
      <div className="overflow-x-auto rounded-lg border border-line lg:w-3/4">
        <table className="w-full border-collapse font-mono text-[0.75rem]">
          <thead>
            <tr className="border-b-2 border-line bg-bg3">
              {["Model", "p50", "p90", "p99", "retries", "ctx window", "ctx used"].map((h) => (
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
            {models.map((m) => {
              const lats = byModel(m)
                .map((r) => r.total_latency_ms)
                .filter((v) => v > 0)
                .sort((a, b) => a - b);
              const eff = summary.find((s) => s.model === m);
              const ctx = eff?.context_length;
              const util =
                ctx && eff ? ((eff.avg_input_tokens + eff.avg_output_tokens) / ctx) * 100 : null;
              return (
                <tr key={m} className="border-b border-line last:border-0">
                  <td className="px-3 py-1.5" style={{ color: mcolor(m) }}>
                    {shortName(m)}
                  </td>
                  <td className="px-3 py-1.5">{fmt(percentile(lats, 0.5), 0)}ms</td>
                  <td className="px-3 py-1.5">{fmt(percentile(lats, 0.9), 0)}ms</td>
                  <td className="px-3 py-1.5 text-amber">
                    {fmt(percentile(lats, 0.99), 0)}ms
                  </td>
                  <td className="px-3 py-1.5 text-mute">
                    {(() => {
                      const withRetries = byModel(m).filter((r) => r.retries != null);
                      if (!withRetries.length) return "—";
                      const total = withRetries.reduce((a, r) => a + (r.retries ?? 0), 0);
                      return total === 0 ? "0" : `${total} (${((total / withRetries.length) * 100).toFixed(0)}%/call)`;
                    })()}
                  </td>
                  <td className="px-3 py-1.5 text-mute">
                    {ctx ? `${Math.round(ctx / 1000)}K` : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-mute">
                    {util != null ? `${util.toFixed(2)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
