"use client";

import { shortName } from "@/lib/aliases";
import { fmt, fmtInt, sortByValueIndex } from "@/lib/derive";
import type { EfficiencyRow } from "@/lib/types";

const USE_CASES: Record<string, { wins: string[]; loses: string[] }> = {
  "anthropic/claude-sonnet-4-5": {
    wins: ["Complex reasoning", "Code review", "Long-form writing", "Hallucination-sensitive tasks"],
    loses: ["High-volume tasks", "Latency-critical apps", "Budget-first API usage"],
  },
  "openai/gpt-4o-mini": {
    wins: ["Fast Q&A", "Simple drafts", "High-volume repetitive tasks", "Low-latency apps"],
    loses: ["Deep analysis", "Nuanced reasoning", "Hallucination traps"],
  },
  "google/gemini-2.0-flash-001": {
    wins: ["Fastest first-token", "Long context tasks", "Cost-efficient volume", "Instruction following"],
    loses: ["Deepest reasoning", "Edge-case accuracy"],
  },
  "deepseek/deepseek-chat": {
    wins: ["Coding tasks", "Cost-efficient analysis", "Math & reasoning", "STEM queries"],
    loses: ["Latency-sensitive use", "Western cultural context"],
  },
  "meta-llama/llama-3.3-70b-instruct": {
    wins: ["Open-weight transparency", "No-lock-in pipeline", "Budget bulk tasks"],
    loses: ["Frontier accuracy", "Hallucination resistance", "Reliability at edge cases"],
  },
  __default__: {
    wins: ["General purpose tasks"],
    loses: ["Specialized benchmarks"],
  },
};

const STYLES = ["verdict-winner", "verdict-runner", "verdict-neutral"];
const MEDALS = ["🥇", "🥈", "🥉"];

function StatCell({ label, value, big }: { label: string; value: string; big?: boolean }) {
  return (
    <div>
      <div className="font-mono text-[0.62rem] uppercase tracking-[0.1em] text-mute">
        {label}
      </div>
      <div
        className={`font-mono font-semibold ${big ? "text-2xl" : "text-[1.1rem] font-medium"}`}
      >
        {value}
      </div>
    </div>
  );
}

export function VerdictBanner({ summary }: { summary: EfficiencyRow[] }) {
  const ranked = sortByValueIndex(summary);
  if (!ranked.length) return null;

  return (
    <section>
      <hr className="hdivider" />
      <div className="section-hdr">
        Subscription verdict — which model for your $20/month?
      </div>
      <div
        className="grid gap-3"
        style={{
          gridTemplateColumns: `repeat(auto-fit, minmax(280px, 1fr))`,
        }}
      >
        {ranked.map((row, i) => {
          const cases = USE_CASES[row.model] ?? USE_CASES.__default__;
          const vi = row.value_index;
          const lat = row.avg_total_latency_ms;
          const ref = row.refusal_rate_pct;
          const vc = vi >= 3.5 ? "pill-green" : vi >= 2.5 ? "pill-amber" : "pill-red";
          const lc = lat < 1200 ? "pill-green" : lat < 2500 ? "pill-amber" : "pill-red";
          const rc = ref < 10 ? "pill-green" : ref < 20 ? "pill-amber" : "pill-red";
          return (
            <div key={row.model} className={`verdict-wrap ${STYLES[Math.min(i, 2)]}`}>
              <div className="mb-3 flex items-baseline gap-2.5">
                <span className="text-xl">{MEDALS[Math.min(i, 2)]}</span>
                <span className="font-mono text-[0.95rem] font-semibold">
                  {shortName(row.model)}
                </span>
              </div>
              <div className="mb-2.5 font-mono text-[0.72rem] text-mute">
                {row.plan} · $20/mo · ~{row.daily_msg_limit} msgs/day
              </div>

              <div className="mb-4 grid grid-cols-2 gap-x-3 gap-y-1.5">
                <div style={{ color: i === 0 ? "var(--amber)" : "var(--text)" }}>
                  <StatCell label="Value Index" value={fmt(vi)} big />
                </div>
                <StatCell label="Quality" value={fmt(row.composite_score)} big />
                <StatCell label="Latency" value={`${fmt(lat, 0)}ms`} />
                <StatCell label="API ans./$20" value={fmtInt(row.api_answers_for_20usd)} />
              </div>

              <div className="mb-2.5">
                <span className={`pill ${vc}`}>VI {fmt(vi)}</span>
                <span className={`pill ${lc}`}>{fmt(lat, 0)}ms latency</span>
                <span className={`pill ${rc}`}>{fmt(ref, 0)}% refusal</span>
                <span className="pill pill-mute">{fmt(row.avg_tokens_per_sec, 0)} tok/s</span>
                <span className="pill pill-mute">{fmt(row.avg_output_tokens, 0)} out-tokens</span>
                <span className="pill pill-mute">{fmt(row.avg_verbosity_ratio, 1)}× verbose</span>
                <span className="pill pill-mute">consistency {fmt(row.consistency_score)}</span>
              </div>

              <div className="mb-1 font-mono text-[0.72rem] uppercase tracking-[0.08em] text-mute">
                Best for
              </div>
              <div className="mb-2">
                {cases.wins.map((w) => (
                  <span key={w} className="pill pill-green">{w}</span>
                ))}
              </div>
              <div className="mb-1 font-mono text-[0.72rem] uppercase tracking-[0.08em] text-mute">
                Watch out for
              </div>
              <div>
                {cases.loses.map((w) => (
                  <span key={w} className="pill pill-red">{w}</span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
