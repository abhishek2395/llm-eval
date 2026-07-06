/**
 * types.ts — end-to-end API types.
 * Shapes mirror the FastAPI backend (Pydantic models + CSV row dicts).
 */

// ── Rows (CSV-backed) ─────────────────────────────────────────────────────────

export interface ResponseRow {
  model: string;
  provider: string;
  prompt_id: string;
  category: string;
  prompt_text: string;
  response_text: string;
  total_latency_ms: number;
  first_token_latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  tokens_per_second: number;
  verbosity_ratio: number;
  refused: boolean;
  error: string | null;
}

export interface ScoreRow {
  prompt_id: string;
  model: string;
  accuracy: number;
  hallucination_resistance: number;
  relevance: number;
  instruction_following: number;
  conciseness: number;
  task_completion: number;
  composite_score: number;
  value_index: number;
  useful_density: number;
  consistency_score: number;
  rationale: string;
  notable_issues: string;
  one_line_verdict: string;
  judge_error: string | null;
}

export type Dimension =
  | "accuracy"
  | "hallucination_resistance"
  | "relevance"
  | "instruction_following"
  | "conciseness"
  | "task_completion";

export const DIMENSIONS: Dimension[] = [
  "accuracy",
  "hallucination_resistance",
  "relevance",
  "instruction_following",
  "conciseness",
  "task_completion",
];

export interface ModelMeta {
  id: string;
  alias: string;
  provider: string;
  provider_display: string;
  color: string;
  prompts_evaluated?: number;
  composite_score?: number | null;
}

export interface EfficiencyRow {
  model: string;
  provider: string;
  plan: string;
  plan_cost_usd: number;
  daily_msg_limit: number;
  composite_score: number;
  value_index: number;
  consistency_score: number;
  task_completion_avg: number;
  refusal_rate_pct: number;
  avg_total_latency_ms: number;
  avg_first_token_ms: number;
  avg_tokens_per_sec: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  avg_verbosity_ratio: number;
  useful_density: number;
  api_cost_per_answer_usd: number;
  api_answers_for_20usd: number;
  input_price_per_1m: number;
  output_price_per_1m: number;
  context_length?: number;
  meta: ModelMeta;
}

// ── Prompts ───────────────────────────────────────────────────────────────────

export interface Prompt {
  id: string;
  category: string;
  difficulty: string;
  prompt: string;
  ground_truth: string;
}

export interface PromptSet {
  name: string;
  prompt_ids: string[];
}

// ── Multi-judge ───────────────────────────────────────────────────────────────

export interface JudgeScoreRow extends ScoreRow {
  judge_model: string;
}

export interface AgreementModel {
  model: string;
  meta: ModelMeta;
  agreement: number;
  mean_abs_diff: number;
  prompts_compared: number;
  per_judge_composite: Record<string, number>;
}

export interface Disagreement {
  model: string;
  prompt_id: string;
  judge_a: string;
  judge_b: string;
  score_a: number;
  score_b: number;
  diff: number;
}

export interface AgreementResponse {
  judges: string[];
  models: AgreementModel[];
  disagreements: Disagreement[];
  note?: string;
}

// ── Catalog ───────────────────────────────────────────────────────────────────

export interface CatalogModel {
  id: string;
  name: string;
  provider: string;
  provider_display: string;
  context_length: number;
  input_price_per_1m: number;
  output_price_per_1m: number;
  is_free: boolean;
  cost_per_1k_answers: number;
  answers_for_20_usd: number;
}

export interface CatalogResponse {
  total_in_catalog: number;
  matched: number;
  providers: string[];
  provider_counts: Record<string, number>;
  models: CatalogModel[];
}

// ── Results payloads ──────────────────────────────────────────────────────────

export interface ResultsResponse {
  responses: ResponseRow[];
  scores: ScoreRow[];
  summary: EfficiencyRow[];
}

export interface HealthResponse {
  status: string;
  version: string;
  api_key_set: boolean;
  models_evaluated: number;
  prompts: number;
}

export interface MetaResponse {
  aliases: Record<string, string>;
  provider_meta: Record<string, { name: string; color: string }>;
  dimensions: Dimension[];
  value_weights: Record<Dimension, number>;
  judge_model: string;
  presets: Record<string, string[]>;
}

// ── SSE events (discriminated union on `type`) ────────────────────────────────

export interface SSEStart {
  type: "start";
  model: string;
  total: number;
}
export interface SSEProgress {
  type: "progress";
  model: string;
  prompt_id: string;
  idx: number;
  total: number;
  stage: "inference" | "judging";
}
export interface SSEResult {
  type: "result" | "cached";
  model: string;
  prompt_id: string;
  idx: number;
  total: number;
  response_row: ResponseRow;
  score_row: ScoreRow;
}
export interface SSEEfficiency {
  type: "efficiency";
  model: string;
  meta: ModelMeta;
  row: EfficiencyRow;
}
export interface SSEDone {
  type: "done";
  model: string;
  total: number;
}
export interface SSEError {
  type: "error";
  model: string;
  message: string;
}

export type SSEEvent =
  | SSEStart
  | SSEProgress
  | SSEResult
  | SSEEfficiency
  | SSEDone
  | SSEError;
