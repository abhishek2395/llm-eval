/**
 * colors.ts — instrument-panel palette + provider color system.
 * Mirrors backend core/config.py PROVIDER_META — keep in sync.
 */

export const PALETTE = {
  bg: "#0a0e1a",
  bg2: "#111827",
  bg3: "#1a2235",
  border: "#1e2d45",
  text: "#e2e8f0",
  textMute: "#64748b",
  amber: "#f59e0b",
  amberDim: "#92400e",
  teal: "#14b8a6",
  blue: "#3b82f6",
  red: "#ef4444",
  green: "#22c55e",
  purple: "#a855f7",
} as const;

export interface ProviderMeta {
  name: string;
  color: string;
}

export const PROVIDER_META: Record<string, ProviderMeta> = {
  anthropic: { name: "Anthropic", color: "#d97706" },
  openai: { name: "OpenAI", color: "#10b981" },
  google: { name: "Google", color: "#3b82f6" },
  deepseek: { name: "DeepSeek", color: "#8b5cf6" },
  "meta-llama": { name: "Meta", color: "#f97316" },
  mistralai: { name: "Mistral", color: "#ec4899" },
  "x-ai": { name: "xAI / Grok", color: "#06b6d4" },
  qwen: { name: "Alibaba", color: "#ef4444" },
  cohere: { name: "Cohere", color: "#84cc16" },
  perplexity: { name: "Perplexity", color: "#a78bfa" },
};

/** Provider color for a model ID ("provider/model" format). */
export function mcolor(model: string): string {
  const provider = model.includes("/") ? model.split("/")[0] : model.split("-")[0];
  return PROVIDER_META[provider]?.color ?? PALETTE.textMute;
}

/** Green/amber/red banding for a 1–5 score (same thresholds as V1). */
export function scoreColor(v: number, lo = 1, hi = 5): string {
  const t = (v - lo) / (hi - lo);
  if (t >= 0.7) return PALETTE.green;
  if (t >= 0.4) return PALETTE.amber;
  return PALETTE.red;
}

export function hexToRgba(hex: string, alpha = 0.15): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
