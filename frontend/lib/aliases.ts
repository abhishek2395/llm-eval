/**
 * aliases.ts — OpenRouter model ID → display name.
 * Mirrors backend core/config.py MODEL_ALIASES — keep in sync.
 */

export const MODEL_ALIASES: Record<string, string> = {
  // ── Anthropic ──────────────────────────────────────────────────────────────
  "anthropic/claude-opus-4-5": "Claude Opus 4.5",
  "anthropic/claude-sonnet-4-5": "Claude Sonnet 4.5",
  "anthropic/claude-haiku-4-5": "Claude Haiku 4.5",
  // 2026 dot-format OpenRouter IDs
  "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
  "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
  "anthropic/claude-sonnet-5": "Claude Sonnet 5",
  "anthropic/claude-opus-4.8-fast": "Claude Opus 4.8 Fast",
  "google/gemini-3.5-flash": "Gemini 3.5 Flash",
  "anthropic/claude-opus-4-20250514": "Claude Opus 4",
  "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
  // ── OpenAI ─────────────────────────────────────────────────────────────────
  "openai/gpt-4o": "GPT-4o",
  "openai/gpt-4o-mini": "GPT-4o Mini",
  "openai/o3": "GPT o3",
  "openai/o3-mini": "GPT o3-mini",
  "openai/o4-mini": "GPT o4-mini",
  "openai/gpt-4.5-preview": "GPT-4.5",
  // ── Google ─────────────────────────────────────────────────────────────────
  "google/gemini-2.5-pro": "Gemini 2.5 Pro",
  "google/gemini-2.5-flash": "Gemini 2.5 Flash",
  "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
  "google/gemini-flash-1.5": "Gemini 1.5 Flash",
  "google/gemini-pro-1.5": "Gemini 1.5 Pro",
  // ── DeepSeek ───────────────────────────────────────────────────────────────
  "deepseek/deepseek-chat": "DeepSeek V3",
  "deepseek/deepseek-r1": "DeepSeek R1",
  "deepseek/deepseek-r1-zero": "DeepSeek R1 Zero",
  // ── Meta / Llama ───────────────────────────────────────────────────────────
  "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
  "meta-llama/llama-3.1-405b-instruct": "Llama 3.1 405B",
  "meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B",
  // ── Mistral ────────────────────────────────────────────────────────────────
  "mistralai/mistral-large": "Mistral Large",
  "mistralai/mistral-medium": "Mistral Medium",
  "mistralai/mixtral-8x22b-instruct": "Mixtral 8×22B",
  // ── xAI / Grok ─────────────────────────────────────────────────────────────
  "x-ai/grok-3": "Grok 3",
  "x-ai/grok-3-mini": "Grok 3 Mini",
  "x-ai/grok-2-1212": "Grok 2",
  // ── Alibaba / Qwen ─────────────────────────────────────────────────────────
  "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
  "qwen/qwq-32b": "QwQ 32B",
  // ── Cohere ─────────────────────────────────────────────────────────────────
  "cohere/command-r-plus": "Command R+",
  "cohere/command-r": "Command R",
  // ── Perplexity ─────────────────────────────────────────────────────────────
  "perplexity/llama-3.1-sonar-large-128k-online": "Sonar Large",
  "perplexity/llama-3.1-sonar-small-128k-online": "Sonar Small",
};

/**
 * Clean display name for a model ID — exact alias, case-insensitive alias,
 * then auto-parse fallback (same logic as V1 dashboard.py short_name).
 */
export function shortName(model: string): string {
  if (MODEL_ALIASES[model]) return MODEL_ALIASES[model];
  const lower = model.toLowerCase();
  for (const [k, v] of Object.entries(MODEL_ALIASES)) {
    if (k.toLowerCase() === lower) return v;
  }
  const part = model.includes("/") ? model.split("/").pop()! : model;
  const seg = part.split("-");
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  return seg.length < 3 ? cap(seg[0]) : `${cap(seg[0])} ${cap(seg[1])}`;
}
