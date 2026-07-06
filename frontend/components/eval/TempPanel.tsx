"use client";

/**
 * TempPanel — temperature sensitivity: same prompts at 0.0 / 0.3 / 0.7,
 * per-prompt score spread and per-temperature means.
 */

import { useRef, useState } from "react";
import { ChevronDown, ChevronRight, Thermometer } from "lucide-react";
import { shortName } from "@/lib/aliases";
import { API_BASE } from "@/lib/api";
import { mcolor } from "@/lib/colors";
import { fmt } from "@/lib/derive";
import { withVisitorKey } from "@/lib/key";

interface TempReport {
  model: string;
  prompts: {
    prompt_id: string;
    category: string;
    by_temp: Record<string, number>;
    spread: number;
    std: number;
  }[];
  temp_means: Record<string, number>;
}

interface TempRun {
  completed: number;
  total: number;
  stage: "starting" | "running" | "done" | "error";
  label?: string;
}

export function TempPanel({ activeModels }: { activeModels: string[] }) {
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState("");
  const [run, setRun] = useState<TempRun | null>(null);
  const [report, setReport] = useState<TempReport | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const loadReport = (m: string) => {
    fetch(`${API_BASE}/eval/temp/results?model=${encodeURIComponent(m)}`)
      .then((r) => r.json())
      .then(setReport)
      .catch(() => setReport(null));
  };

  const start = () => {
    const m = model.trim();
    if (!m) return;
    esRef.current?.close();
    setRun({ completed: 0, total: 0, stage: "starting" });
    const es = new EventSource(
      withVisitorKey(`${API_BASE}/eval/temp/stream?model=${encodeURIComponent(m)}`),
    );
    esRef.current = es;
    es.onmessage = (msg) => {
      const ev = JSON.parse(msg.data);
      if (ev.type === "start")
        setRun((r) => r && { ...r, total: ev.total, stage: "running" });
      else if (ev.type === "result" || ev.type === "cached")
        setRun((r) => r && {
          ...r,
          completed: r.completed + 1,
          label: `${ev.prompt_id} @ ${ev.temperature}`,
        });
      else if (ev.type === "done") {
        setRun((r) => r && { ...r, stage: "done" });
        es.close();
        loadReport(m);
      } else if (ev.type === "error") {
        setRun((r) => r && { ...r, stage: "error", label: ev.message });
        es.close();
      }
    };
    es.onerror = () => {
      setRun((r) => (r && r.stage !== "done" ? { ...r, stage: "error" } : r));
      es.close();
    };
  };

  const temps = report ? Object.keys(report.temp_means).sort() : [];

  return (
    <section>
      <button
        className="flex w-full items-center gap-2 rounded-lg border border-line bg-bg2 px-4 py-3 text-left font-mono text-[0.8rem] text-mute transition-colors hover:text-ink"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Thermometer size={13} />
        Temperature sensitivity — 0.0 / 0.3 / 0.7
      </button>

      {open && (
        <div className="mt-3 rounded-lg border border-line bg-bg2 p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <input
              className="panel-input min-w-64 font-mono text-[0.78rem]"
              list="temp-models"
              placeholder="model to sweep"
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                if (activeModels.includes(e.target.value)) loadReport(e.target.value);
              }}
            />
            <datalist id="temp-models">
              {activeModels.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            <button
              className="panel-btn panel-btn-primary"
              disabled={!model.trim() || run?.stage === "running"}
              onClick={start}
            >
              🌡 Run sweep
            </button>
            <span className="font-mono text-[0.68rem] text-mute">
              one prompt per category × 3 temperatures
            </span>
          </div>

          {run && (
            <div className="mb-4 lg:w-2/3">
              <div className="mb-1 flex justify-between font-mono text-[0.7rem]">
                <span className="text-mute">{run.label ?? "…"}</span>
                <span className="text-mute">
                  {run.stage === "done"
                    ? `✓ ${run.completed} runs`
                    : run.stage === "error"
                      ? "✕ failed"
                      : `${run.completed}/${run.total || "…"}`}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${run.stage === "done" ? 100 : run.total ? (run.completed / run.total) * 100 : 5}%`,
                    background: run.stage === "error" ? "var(--red)" : "var(--teal)",
                  }}
                />
              </div>
            </div>
          )}

          {report && report.prompts.length > 0 && (
            <>
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span
                  className="font-mono text-[0.78rem] font-semibold"
                  style={{ color: mcolor(report.model) }}
                >
                  {shortName(report.model)}
                </span>
                {temps.map((t) => (
                  <span key={t} className="pill pill-mute">
                    t={t}: {fmt(report.temp_means[t])}
                  </span>
                ))}
              </div>
              <div className="overflow-x-auto rounded-lg border border-line">
                <table className="w-full border-collapse font-mono text-[0.73rem]">
                  <thead>
                    <tr className="border-b-2 border-line bg-bg3">
                      {["Prompt", ...temps.map((t) => `t=${t}`), "spread", "σ"].map((h) => (
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
                    {report.prompts.map((p) => (
                      <tr key={p.prompt_id} className="border-b border-line last:border-0">
                        <td className="px-3 py-1.5 text-mute">
                          {p.prompt_id}
                          <span className="ml-1.5 text-[0.62rem]">{p.category}</span>
                        </td>
                        {temps.map((t) => (
                          <td key={t} className="px-3 py-1.5">
                            {p.by_temp[t] != null ? fmt(p.by_temp[t]) : "—"}
                          </td>
                        ))}
                        <td className="px-3 py-1.5">{fmt(p.spread)}</td>
                        <td
                          className="px-3 py-1.5 font-semibold"
                          style={{
                            color:
                              p.std > 0.5
                                ? "var(--red)"
                                : p.std > 0.25
                                  ? "var(--amber)"
                                  : "var(--green)",
                          }}
                        >
                          {fmt(p.std, 3)}
                          {p.std > 0.5 && " ⚠"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
