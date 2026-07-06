"use client";

/**
 * Report — shareable exports: self-contained HTML, structured JSON,
 * README leaderboard snippet, PDF via print.
 */

import { useEffect, useState } from "react";
import { Copy, Check, Download, ExternalLink, FileText, Printer } from "lucide-react";
import { api, exportUrls } from "@/lib/api";
import { fmt, fmtInt, sortByValueIndex } from "@/lib/derive";
import type { EfficiencyRow } from "@/lib/types";

export default function ReportPage() {
  const [summary, setSummary] = useState<EfficiencyRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api
      .summary()
      .then((r) => setSummary(r.summary))
      .catch((e) => setError(e.message));
  }, []);

  const copyMarkdown = async () => {
    const res = await fetch(exportUrls.markdown);
    await navigator.clipboard.writeText(await res.text());
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const ranked = sortByValueIndex(summary);
  const medals = ["🥇", "🥈", "🥉"];

  return (
    <div className="mx-auto max-w-[1100px]">
      <header className="pb-2 pt-1">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
          LLM Eval Framework
        </div>
        <h1 className="text-[1.6rem] font-semibold leading-tight">
          Shareable Report
        </h1>
        <div className="text-[0.82rem] text-mute">
          Exports work without the dashboard — send the HTML file, paste the
          markdown into a README, or feed the JSON to anything.
        </div>
      </header>

      <hr className="hdivider" />

      {error && (
        <div className="mb-4 rounded-lg border border-red/30 bg-red/5 p-4 font-mono text-[0.8rem] text-red">
          Backend unreachable: {error}
        </div>
      )}

      {/* Export actions */}
      <div className="mb-6 grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))" }}>
        <a
          className="card block transition-colors hover:border-amber/40"
          href={exportUrls.html}
          target="_blank"
          rel="noreferrer"
        >
          <div className="card-title flex items-center gap-1.5">
            <ExternalLink size={12} /> Self-contained HTML
          </div>
          <div className="text-[0.8rem] text-mute">
            One file, no server — save with ⌘S and share it anywhere.
          </div>
        </a>
        <a
          className="card block transition-colors hover:border-amber/40"
          href={exportUrls.json}
        >
          <div className="card-title flex items-center gap-1.5">
            <Download size={12} /> Structured JSON
          </div>
          <div className="text-[0.8rem] text-mute">
            Responses + scores + efficiency summary, typed end-to-end.
          </div>
        </a>
        <button className="card block text-left transition-colors hover:border-amber/40" onClick={() => void copyMarkdown()}>
          <div className="card-title flex items-center gap-1.5">
            {copied ? <Check size={12} className="text-green" /> : <Copy size={12} />}
            {copied ? "Copied!" : "README leaderboard"}
          </div>
          <div className="text-[0.8rem] text-mute">
            Markdown table snippet — paste straight into your GitHub README.
          </div>
        </button>
        <a
          className="card block transition-colors hover:border-amber/40"
          href={exportUrls.html}
          target="_blank"
          rel="noreferrer"
        >
          <div className="card-title flex items-center gap-1.5">
            <Printer size={12} /> PDF
          </div>
          <div className="text-[0.8rem] text-mute">
            Open the HTML report → ⌘P → save as PDF (print styles included).
          </div>
        </a>
      </div>

      {/* Leaderboard preview */}
      <div className="section-hdr flex items-center gap-2">
        <FileText size={12} /> Leaderboard preview
      </div>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full border-collapse font-mono text-[0.78rem]">
          <thead>
            <tr className="border-b-2 border-line bg-bg3">
              {["Rank", "Model", "Value Index", "Quality", "Latency", "Speed", "Ans/$20"].map((h) => (
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
            {ranked.map((row, i) => (
              <tr key={row.model} className="border-b border-line last:border-0">
                <td className="px-3 py-2 text-[1rem]">{medals[i] ?? i + 1}</td>
                <td className="px-3 py-2">
                  <span style={{ color: row.meta.color }} className="font-semibold">
                    {row.meta.alias}
                  </span>
                  <span className="ml-2 text-[0.65rem] text-mute">{row.model}</span>
                </td>
                <td className="px-3 py-2 font-semibold text-amber">
                  {fmt(row.value_index)}
                </td>
                <td className="px-3 py-2">{fmt(row.composite_score)}/5</td>
                <td className="px-3 py-2">{fmt(row.avg_total_latency_ms, 0)}ms</td>
                <td className="px-3 py-2">{fmt(row.avg_tokens_per_sec, 0)} tok/s</td>
                <td className="px-3 py-2">{fmtInt(row.api_answers_for_20usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
