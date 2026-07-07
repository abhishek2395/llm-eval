# LLM Eval Framework V2

**Which model actually deserves your $20/month?** A full-stack LLM evaluation
framework: one OpenRouter key → 300+ models, LLM-as-judge scoring on 6 quality
dimensions, live SSE-streamed evals, a multi-judge reliability panel, and an
agentic tool-use harness that judges *trajectories*, not just answers.

![Dashboard](docs/screenshots/dashboard.png)

## Why

Leaderboards tell you which model tops MMLU. They don't tell you which model
gives **you** the most value per dollar on **your** prompts — or whether it
fabricates an answer when its tools come back empty. This framework answers
both, with receipts: every score traces back to a stored response, a judge
rationale, and (for agentic tasks) a full tool-call transcript.

## Findings from real runs (July 2026)

These came out of actually using the framework — not hypotheticals:

- **Judge choice moves scores by half a point.** Claude Sonnet 5 judges the
  same responses ~0.3–0.5 composite lower than Claude Sonnet 4.5 (inter-judge
  agreement 0.74–0.79). The biggest disagreements (up to Δ3.17/5) were on
  responses originally judged *without* ground truths — the stricter judge
  with ground truths exposed plausible-but-wrong answers that scored 4+ blind.
- **Frontier-adjacent models ace single-turn tool prompts but diverge on
  trajectories.** In the agentic harness, Claude Sonnet 4.5, DeepSeek V3 and
  GPT-4o Mini ran perfect trajectories. Gemini 3.5 Flash hit the fictional-city
  trap honestly (searched 4×, never fabricated) but **never delivered a final
  answer** — a failure mode invisible to single-turn evals. Llama 3.3 70B
  searched the web for "the capital of France" (tool restraint failure) and
  got the two-city weather arithmetic wrong.
- **Failed rows must not be cached.** V1 wrote judge/API failures to the CSV
  and treated them as cached forever. V2 auto-retries them — which silently
  healed rows that had been broken for months.
- **Most leaderboard positions are statistical ties.** Bootstrap 95% CIs at
  n=15 prompts show 12 of 15 pairwise model comparisons overlap — the Eval
  Quality tab states this instead of implying false precision. Meanwhile the
  judge itself passes 16/16 golden calibration tests (planted hallucinations,
  eloquent-but-wrong answers, length-bias probes) with perfect repeatability
  (σ=0.000 across 3 trials); a budget judge (GPT-4o Mini) scores 94%, failing
  exactly where you'd fear — leniency on half-finished work.

## Features

| | |
|---|---|
| **Live eval panel** | Type any OpenRouter model ID → SSE-streamed progress per prompt, live token + cost ticker, charts update as results land. Cached prompts are instant and free. |
| **6-dimension LLM-as-judge** | accuracy, hallucination resistance, relevance, instruction following, conciseness, task completion — plus a composite and a Value Index (quality ÷ verbosity penalty). |
| **Agentic harness** | 5 trajectory tasks in a deterministic mock-tool sandbox (calculator, search, files, weather). Real function-calling loops; judged on task success, tool efficiency, honesty, reasoning. |
| **Multi-judge panel** | Re-score stored responses with any second judge; inter-judge agreement + flagged disagreements >1.0. |
| **Eval Quality tab** | The eval of the eval: 16-item judge calibration suite with known-answer golden tests, bootstrap 95% CIs with statistical-tie detection, judge repeatability σ, measured model consistency (replacing V1's hardcoded estimate), self-judgment flags, and stated limitations. |
| **Compare** | Word-level diff of two models on one prompt, per-prompt radar, pairwise head-to-head win-rate matrix, regression detection vs historical runs. |
| **Prompt library** | CRUD + category/difficulty tags, CSV import, named prompt sets, no-ground-truth mode. Includes an `agentic` category (tool-call JSON, planning, restraint, error recovery, temporal reasoning). |
| **Exports** | Self-contained HTML report, structured JSON, README leaderboard snippet, PDF via print styles. |
| **⌘K palette** | Navigate, run models, "surprise me" (3 random catalog models), search prompts, export. |

![Agentic harness](docs/screenshots/agentic.png)

## Architecture

```
┌────────────────────────┐         ┌──────────────────────────────┐
│  Next.js 15 (App Router)         │  FastAPI                     │
│  TypeScript strict     │  REST   │                              │
│  Tailwind v4           │◄───────►│  routers/  eval · judge ·    │
│  react-plotly.js       │   SSE   │            agentic · models ·│
│  ⌘K palette (cmdk)     │◄────────│            prompts · results │
└────────────────────────┘         │  core/     V1 logic, intact: │
                                   │    live_eval.py  (generator) │
        one shared contract:       │    agentic_eval.py (harness) │
        lib/types.ts ⇆ Pydantic    │    openrouter_client.py      │
                                   │    judge.py · config.py      │
                                   └──────────────┬───────────────┘
                                                  │ one API key
                                                  ▼
                                   OpenRouter → 300+ models
                                   (Claude, GPT, Gemini, DeepSeek,
                                    Llama, Grok, Mistral, Qwen…)

        persistence: plain CSVs (results/, no database) —
        responses · scores · judge_scores · agentic_runs
```

The V1 Python engine (generator-based live eval, judge, OpenRouter client)
lives unchanged in `backend/core/` — V2 wraps it in FastAPI and streams the
generator's events (`start → cached|progress|result → efficiency → done`)
straight through as Server-Sent Events. The frontend consumes them with a
typed `EventSource` hook and merges rows into chart state as they arrive.

## Quickstart

**Backend** (Python 3.11+, [uv](https://docs.astral.sh/uv/) recommended):

```bash
cd backend
uv venv --python 3.12 .venv && uv pip install -r requirements.txt --python .venv/bin/python
echo "OPENROUTER_API_KEY=sk-or-..." > .env       # get one at openrouter.ai/keys
.venv/bin/uvicorn main:app --port 8000
```

**Frontend** (Node 18+):

```bash
cd frontend
npm install
npm run dev                                       # → http://localhost:3000
```

The repo ships with real evaluated results, so the dashboard is populated on
first load. Add any model live from the search box — only un-evaluated
prompts cost tokens.

## Stack decisions

| Decision | Why |
|---|---|
| FastAPI over rewriting the engine | V1's generator pattern maps 1:1 onto SSE; zero logic rewritten |
| Next.js over Streamlit | V1 needed 200+ lines of CSS injection and `st.rerun()` hacks for what React does natively; sharing a Streamlit app means "install Python" |
| react-plotly.js over Recharts | Exact visual parity — violin plots and heatmaps have no Recharts equivalent |
| CSVs over a database | Evals are append-mostly, human-inspectable, and diff-able; a DB adds ops for no analytical win at this scale |
| One OpenRouter key | 300+ models through one OpenAI-compatible endpoint; provider lock-in deferred indefinitely |

## Repo layout

```
backend/
  core/          V1 engine (unchanged) + agentic_eval.py
  routers/       eval · judge · agentic · models · prompts · results
  data/          prompts.json · agentic_tasks.json · prompt_sets.json
  results/       CSV persistence (responses, scores, judge panel, trajectories)
frontend/
  app/           dashboard · /agentic · /compare · /prompts · /report
  components/    sections/ · eval/ · charts/ · layout/
  lib/           typed API client · aliases · colors · plotly theme · diff
  hooks/         useEvalData (store) · useEvalStream (SSE)
```

---

Built by [Abhishek](https://github.com/abhishek2395) as an AI quality
engineering portfolio project. V1 (Streamlit) history is on the `main` branch.
