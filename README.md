# LLM Eval Framework

> Compare 300+ language models side-by-side — quality scores, latency, cost efficiency, and hallucination resistance — all from a single dark-mode dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b?style=flat-square)
![OpenRouter](https://img.shields.io/badge/OpenRouter-300%2B%20models-6c47ff?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What it does

Select any models from OpenRouter's catalog, click **Run Evaluation**, and the framework:

1. Sends each prompt to the model via the OpenRouter API
2. Passes the response to an LLM judge (Claude) that scores it on 6 dimensions
3. Streams results live into the dashboard as each prompt completes
4. Builds comparison charts across quality, latency, cost, and value metrics

---

## Dashboard

- **Overview** — per-model KPI strip (composite score, value index, latency, refusal rate)
- **Radar chart** — 6-dimension quality fingerprint per model
- **Latency vs Quality** — scatter plot to find the speed/quality sweet spot
- **Cost Efficiency** — answers-per-$20 and value index comparisons
- **Category Breakdown** — heatmap of scores across prompt categories
- **Per-prompt Drill-down** — read every response and judge rationale side-by-side
- **Live eval panel** — select up to 5 models from a dropdown, run evals, watch scores stream in real time

---

## Scoring dimensions

Each response is scored 1–5 on six dimensions by the LLM judge:

| Dimension | What it measures |
|---|---|
| Accuracy | Correctness against ground truth |
| Hallucination Resistance | Absence of fabricated facts |
| Relevance | How directly the question is answered |
| Instruction Following | Adherence to formatting / structural instructions |
| Conciseness | Signal-to-noise ratio of the response |
| Task Completion | Whether the full task was actually completed |

A **composite score** and **value index** (quality per token cost) are derived from these.

---

## Architecture

```
prompts.json
     │
     ▼
OpenRouter API  ──►  300+ models (GPT-4o, Claude, Gemini, Llama, Grok, DeepSeek…)
     │
     ▼
Response Store  (Pandas DataFrame + results/results.csv)
     │
     ▼
LLM Judge  (Claude via OpenRouter — 6-dimension structured rubric)
     │
     ▼
Score Store  (results/scores.csv)
     │
   ┌─┴─────────────┐
   ▼               ▼
Streamlit      Efficiency
Dashboard      Summary CSV
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/abhishek2395/llm-eval.git
cd llm-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your OpenRouter API key

```bash
cp .env.example .env
# then edit .env and paste your key
```

```
OPENROUTER_API_KEY=sk-or-...
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Generate demo data (optional)

Populates the dashboard with sample results so you can explore it before running real evals:

```bash
python generate_demo_data.py
```

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

---

## Running evaluations

### From the dashboard (recommended)

1. Open the dashboard at `http://localhost:8501`
2. Use the **Model Selection** dropdown to pick up to 5 models
3. Click **▶ Run Evaluation** — scores stream in live as each prompt completes
4. Charts update automatically when the run finishes

### From the CLI

```bash
# Run all models defined in config.py
python evaluate.py

# Run specific models
python evaluate.py --models openai/gpt-4o-mini google/gemini-2.0-flash

# Skip judge scoring (inference only)
python evaluate.py --skip-judge

# Re-score from an existing responses file
python evaluate.py --load-results results/responses_20260410_000726.csv
```

---

## Project structure

```
llm-eval/
├── dashboard.py          # Streamlit dashboard — all charts and live eval UI
├── live_eval.py          # Generator-based real-time eval engine
├── config.py             # Models, pricing, rubric, OpenRouter settings
├── evaluate.py           # CLI orchestrator
├── runners.py            # API runners with retry / backoff
├── judge.py              # LLM-as-judge implementation
├── openrouter_client.py  # OpenRouter catalog browser
├── reporter.py           # Markdown report generator
├── generate_demo_data.py # Synthetic demo data for local exploration
├── prompts.json          # Prompt dataset with ground truths
├── data/prompts.json     # (same, used by CLI path)
├── results/              # CSV output — responses, scores, efficiency summary
└── reports/              # Auto-generated Markdown evaluation reports
```

---

## Adding prompts

Edit `prompts.json`. Each entry:

```json
{
  "id": "unique_id",
  "category": "factual | reasoning | coding | hallucination_test",
  "prompt": "Your question here",
  "ground_truth": "The expected correct answer",
  "difficulty": "easy | medium | hard"
}
```

---

## License

MIT
