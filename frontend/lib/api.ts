/**
 * api.ts — typed client for the FastAPI backend.
 */

import type {
  AgreementResponse,
  CatalogResponse,
  HealthResponse,
  JudgeScoreRow,
  MetaResponse,
  Prompt,
  PromptSet,
  ResultsResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function send<T>(
  method: "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${method} ${path} → ${res.status}`);
  return (res.status === 204 ? undefined : res.json()) as Promise<T>;
}

export const api = {
  health: () => get<HealthResponse>("/health"),
  meta: () => get<MetaResponse>("/meta"),
  results: () => get<ResultsResponse>("/results"),
  summary: () => get<{ summary: ResultsResponse["summary"] }>("/results/summary"),
  runs: () => get<{ runs: string[] }>("/results/runs"),
  run: (stamp: string) =>
    get<Pick<ResultsResponse, "responses" | "scores">>(`/results/runs/${stamp}`),
  deleteModelResults: (modelId: string) =>
    send<{ deleted: string }>("DELETE", `/results/model/${modelId}`),

  activeModels: () =>
    get<{ models: import("./types").ModelMeta[] }>("/models/active"),
  catalog: (params?: {
    search?: string;
    provider?: string;
    tier?: string;
    limit?: number;
  }) => {
    const q = new URLSearchParams();
    if (params?.search) q.set("search", params.search);
    if (params?.provider) q.set("provider", params.provider);
    if (params?.tier) q.set("tier", params.tier);
    if (params?.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return get<CatalogResponse>(`/models/catalog${qs ? `?${qs}` : ""}`);
  },

  prompts: () => get<Prompt[]>("/prompts"),
  addPrompt: (p: Omit<Prompt, "id"> & { id?: string }) =>
    send<Prompt>("POST", "/prompts", p),
  updatePrompt: (id: string, p: Omit<Prompt, "id">) =>
    send<Prompt>("PUT", `/prompts/${id}`, p),
  deletePrompt: (id: string) => send<void>("DELETE", `/prompts/${id}`),

  sets: () => get<PromptSet[]>("/prompts/sets"),
  saveSet: (s: PromptSet) => send<PromptSet>("POST", "/prompts/sets", s),
  deleteSet: (name: string) =>
    send<void>("DELETE", `/prompts/sets/${encodeURIComponent(name)}`),
  bulkImport: (prompts: (Omit<Prompt, "id"> & { id?: string })[]) =>
    send<Prompt[]>("POST", "/prompts/bulk", prompts),

  judgeScores: () =>
    get<{ scores: JudgeScoreRow[]; judges: string[] }>("/judge/scores"),
  judgeAgreement: () => get<AgreementResponse>("/judge/agreement"),
  judgeStreamUrl: (judge: string, models: string[]) =>
    `${API_BASE}/judge/stream?judge=${encodeURIComponent(judge)}&models=${encodeURIComponent(models.join(","))}`,

  /** URL for the SSE eval stream (consumed with EventSource). */
  evalStreamUrl: (model: string) =>
    `${API_BASE}/eval/stream?model=${encodeURIComponent(model)}`,
};

/** Direct-download / share URLs (opened in a new tab, not fetched). */
export const exportUrls = {
  json: `${API_BASE}/results/export`,
  html: `${API_BASE}/results/report.html`,
  markdown: `${API_BASE}/results/embed.md`,
};
