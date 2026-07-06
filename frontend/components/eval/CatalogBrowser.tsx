"use client";

/**
 * CatalogBrowser — searchable/filterable OpenRouter catalog (300+ models),
 * with provider distribution chart. Collapsed by default.
 */

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { Data } from "plotly.js";
import { Plot } from "@/components/charts/Plot";
import { api } from "@/lib/api";
import { PROVIDER_META, PALETTE } from "@/lib/colors";
import { fmtInt } from "@/lib/derive";
import { baseLayout, MONO, PLOT_CONFIG } from "@/lib/plotly";
import type { CatalogResponse } from "@/lib/types";

const TIERS = [
  { value: "all", label: "All prices" },
  { value: "free", label: "Free" },
  { value: "lt1", label: "< $1/1M" },
  { value: "1to5", label: "$1–5/1M" },
  { value: "gt5", label: "> $5/1M" },
];

export function CatalogBrowser({ onPick }: { onPick: (model: string) => void }) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [provider, setProvider] = useState("all");
  const [tier, setTier] = useState("all");
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    const t = setTimeout(() => {
      api
        .catalog({ search: search || undefined, provider, tier, limit: 30 })
        .then(setData)
        .catch(() => setData(null))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [open, search, provider, tier]);

  const providerBar = useMemo(() => {
    if (!data) return null;
    const top = Object.entries(data.provider_counts)
      .sort((a, b) => a[1] - b[1])
      .slice(-12);
    return {
      x: top.map(([, c]) => c),
      y: top.map(([p]) => PROVIDER_META[p]?.name ?? p),
      colors: top.map(([p]) => PROVIDER_META[p]?.color ?? PALETTE.textMute),
      counts: top.map(([, c]) => c),
    };
  }, [data]);

  return (
    <section>
      <button
        className="flex w-full items-center gap-2 rounded-lg border border-line bg-bg2 px-4 py-3 text-left font-mono text-[0.8rem] text-mute transition-colors hover:text-ink"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        🗂 Browse OpenRouter catalog
        {data ? ` (${data.total_in_catalog} models)` : " (300+ models)"}
      </button>

      {open && (
        <div className="mt-3 rounded-lg border border-line bg-bg2 p-4">
          <div className="mb-4 flex flex-wrap gap-2">
            <input
              className="panel-input min-w-52 flex-1 font-mono text-[0.78rem]"
              placeholder="gemini, llama, deepseek…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select
              className="panel-input font-mono text-[0.75rem]"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              <option value="all">All providers</option>
              {data?.providers.map((p) => (
                <option key={p} value={p}>
                  {PROVIDER_META[p]?.name ?? p}
                </option>
              ))}
            </select>
            <select
              className="panel-input font-mono text-[0.75rem]"
              value={tier}
              onChange={(e) => setTier(e.target.value)}
            >
              {TIERS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>

          {loading ? (
            <div className="py-8 text-center font-mono text-[0.75rem] text-mute">
              loading catalog…
            </div>
          ) : data && data.models.length ? (
            <>
              <div className="overflow-x-auto rounded-lg border border-line">
                <table className="w-full border-collapse text-[0.8rem]">
                  <thead>
                    <tr className="border-b-2 border-line bg-bg3">
                      {["Model ID", "Provider", "In / Out $/1M", "Context", "Ans/$20", ""].map(
                        (h) => (
                          <th
                            key={h}
                            className="px-2.5 py-2 text-left font-mono text-[0.6rem] font-normal uppercase tracking-[0.1em] text-mute"
                          >
                            {h}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {data.models.map((m) => (
                      <tr key={m.id} className="border-b border-line last:border-0">
                        <td className="px-2.5 py-1.5 font-mono text-[0.75rem]">
                          {m.id}
                        </td>
                        <td className="px-2.5 py-1.5">
                          <span
                            className="text-[0.72rem] font-semibold"
                            style={{
                              color:
                                PROVIDER_META[m.provider]?.color ??
                                PALETTE.textMute,
                            }}
                          >
                            {m.provider_display}
                          </span>
                        </td>
                        <td className="px-2.5 py-1.5 font-mono text-[0.72rem] text-mute">
                          {m.is_free ? (
                            <span className="pill pill-green">FREE</span>
                          ) : (
                            `$${m.input_price_per_1m.toFixed(3)} / $${m.output_price_per_1m.toFixed(3)}`
                          )}
                        </td>
                        <td className="px-2.5 py-1.5 font-mono text-[0.72rem] text-mute">
                          {m.context_length >= 1000
                            ? `${Math.round(m.context_length / 1000)}K`
                            : m.context_length}
                        </td>
                        <td className="px-2.5 py-1.5 font-mono text-[0.72rem] text-amber">
                          {m.is_free ? "∞" : fmtInt(m.answers_for_20_usd)}
                        </td>
                        <td className="px-2.5 py-1.5">
                          <button
                            className="panel-btn px-2.5 py-1 text-[0.65rem]"
                            onClick={() => onPick(m.id)}
                          >
                            ＋ eval
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-2 font-mono text-[0.7rem] text-mute">
                Showing {data.models.length} of {data.matched} matches ·{" "}
                {data.total_in_catalog} models in catalog
              </div>

              {providerBar && (
                <>
                  <div className="card-title mt-4">Models available by provider</div>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        orientation: "h",
                        x: providerBar.x,
                        y: providerBar.y,
                        marker: { color: providerBar.colors },
                        text: providerBar.counts.map(String),
                        textposition: "outside",
                        textfont: { family: MONO, size: 10, color: PALETTE.text },
                      } as Data,
                    ]}
                    layout={baseLayout({
                      height: 320,
                      showlegend: false,
                      margin: { l: 90, r: 50, t: 8, b: 28 },
                      xaxis: {
                        title: { text: "models available", font: { size: 10 } },
                      },
                    })}
                    config={PLOT_CONFIG}
                    className="w-full"
                    useResizeHandler
                  />
                </>
              )}
            </>
          ) : (
            <div className="py-8 text-center font-mono text-[0.75rem] text-mute">
              No models match your filters.
            </div>
          )}
        </div>
      )}
    </section>
  );
}
