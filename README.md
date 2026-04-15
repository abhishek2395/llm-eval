# LLM Evaluation & Testing Framework

> A production-grade framework for benchmarking LLMs on accuracy, hallucination resistance, and relevance using the **LLM-as-judge** pattern.

Built as a portfolio project demonstrating AI Quality Engineering, ETL testing principles applied to LLM pipelines, and full-stack Python development.

---

## Features

- **Multi-provider support** — Claude (Anthropic) and OpenAI models evaluated side-by-side
- **LLM-as-judge scoring** — Claude evaluates responses on a structured 4-dimension rubric
- **Hallucination testing** — dedicated prompt category with trap questions and false premises
- **Streamlit dashboard** — interactive charts, per-prompt drill-down, latency vs quality scatter
- **Exportable reports** — auto-generated Markdown reports + CSV exports
- **Historical runs** — timestamped result files, compare runs over time
- **Retry logic** — exponential backoff on rate limits for both providers

---

## Architecture

```
prompts.json
    │
    ├──► Claude Runner (claude-sonnet, claude-opus)
    └──► OpenAI Runner (gpt-4o, gpt-4o-mini)
              │
              ▼
        Response Store (Pandas DataFrame + CSV)
              │
              ▼
        LLM Judge (Claude) — structured 4-dim rubric
              │
              ▼
        Score Store (scores.csv)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
Streamlit          Markdown Report
Dashboard           + CSV Export
```

**Scoring dimensions (1–5 each):**
| Dimension | What it measures |
|---|---|
| Accuracy | Correctness vs ground truth |
| Hallucination Resistance | Absence of fabricated facts |
| Relevance | How directly the question is answered |
| Instruction Following | Adherence to formatting/structural instructions |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### 3. Add your prompts

Edit `data/prompts.json`. Each prompt needs:

```json
{
  "id": "unique_id",
  "category": "factual | reasoning | coding | hallucination_test | ...",
  "prompt": "Your question here",
  "ground_truth": "The expected correct answer",
  "difficulty": "easy | medium | hard"
}
```

---

## Usage

### Run a full evaluation

```bash
python evaluate.py
```

### Run specific models only

```bash
python evaluate.py --models claude-sonnet-4-20250514 gpt-4o-mini
```

### Run inference only (no judge scoring)

```bash
python evaluate.py --skip-judge
```

### Re-score existing results (no API calls for inference)

```bash
python evaluate.py --load-results results/responses_20250601_120000.csv
```

### Launch the dashboard

```bash
streamlit run dashboard.py
```

---

## Project Structure

```
llm_eval_framework/
├── config.py          # Models, paths, rubric, settings
├── runners.py         # ClaudeRunner + OpenAIRunner with retry logic
├── judge.py           # LLMJudge — LLM-as-judge pattern implementation
├── evaluate.py        # Main orchestrator CLI
├── reporter.py        # Markdown report + summary stats generator
├── dashboard.py       # Streamlit dashboard (5 tabs)
├── requirements.txt
├── data/
│   └── prompts.json   # Prompt dataset with ground truths
├── results/           # CSV output — responses + scores, timestamped
└── reports/           # Auto-generated Markdown evaluation reports
```

---

## Extending the Framework

### Add a new model provider

1. Create a new runner class in `runners.py` following the `ClaudeRunner` pattern
2. Add the model IDs to `config.py`
3. Import and call in `evaluate.py`'s `run_inference()`

### Customize the scoring rubric

Edit `RUBRIC` in `config.py`. The judge expects JSON back with the same keys — update `JudgeScore` in `judge.py` if you add dimensions.

### Add prompt categories

Add prompts to `data/prompts.json` with a new `category` value. No code changes needed.

---

## Resume Talking Points

**For AI PM roles:**
- Designed evaluation methodology using LLM-as-judge pattern (Meta/Anthropic research)
- Defined 4-dimension rubric mapping to product quality metrics
- Built reporting pipeline producing stakeholder-ready comparison reports
- Identified hallucination patterns across model families

**For AI Engineer / QA roles:**
- Implemented async-capable multi-provider API abstraction with retry/backoff
- Built ETL pipeline: prompt ingestion → model inference → judge scoring → structured output
- Applied data quality validation principles (ground truth comparison, anomaly detection)
- Designed for extensibility: new providers drop in without touching orchestration logic

---

## License

MIT
