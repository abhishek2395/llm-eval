"""
/results — serve the CSV persistence layer as JSON.

GET    /results                 everything: responses + scores + efficiency summary
GET    /results/summary         efficiency summary only (recomputed live)
GET    /results/runs            timestamped historical run files
GET    /results/runs/{stamp}    one historical run (responses + scores)
DELETE /results/model/{id}      permanently remove one model's rows from the CSVs
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from config import RESULTS_DIR, RESULTS_FILE, SCORES_FILE
from live_eval import build_efficiency_row

from ._common import clean, df_records, model_meta
from .models import enrich_pricing

router = APIRouter(prefix="/results", tags=["results"])


def _load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
    s = pd.read_csv(SCORES_FILE) if SCORES_FILE.exists() else pd.DataFrame()
    return r, s


def _summary_rows(r: pd.DataFrame, s: pd.DataFrame) -> list[dict]:
    """Recompute the efficiency summary per model from raw rows.

    Judge-errored score rows are excluded (their 0-scores would drag every
    mean down); models with no valid scores are skipped entirely.
    """
    if r.empty or s.empty:
        return []
    valid = s[s["judge_error"].isna()] if "judge_error" in s.columns else s
    # Infra-errored responses (401s, timeouts) are not model behavior — keep
    # them out of the latency/token averages.
    r_ok = r[r["error"].isna()] if "error" in r.columns else r
    rows = []
    for m in sorted(valid["model"].dropna().unique().tolist()):
        resp = r_ok[r_ok["model"] == m].to_dict("records")
        sc = valid[valid["model"] == m].to_dict("records")
        if resp and sc:
            row = enrich_pricing(build_efficiency_row(m, resp, sc), m)
            # True refusals (model declining in text, judged by the rubric's
            # model_refused flag) replace the legacy infra-error-based rate.
            if "model_refused" in valid.columns:
                m_rows = valid[valid["model"] == m]
                row["refusal_rate_pct"] = round(
                    float(m_rows["model_refused"].fillna(False).astype(bool).mean()) * 100, 1
                )
            avg_conf = None
            if "judge_confidence" in valid.columns:
                cvals = valid[valid["model"] == m]["judge_confidence"].dropna()
                if len(cvals):
                    avg_conf = round(float(cvals.mean()), 3)
            row["judge_confidence_avg"] = avg_conf
            # Measured consistency (repeat-run panel) replaces the V1 estimate
            from meta_eval import real_consistency

            measured = real_consistency(m)
            if measured is not None:
                row["consistency_score"] = measured
                row["consistency_measured"] = True
            row["meta"] = model_meta(m)
            rows.append(clean(row))
    return rows


@router.get("")
def all_results():
    r, s = _load_csvs()
    return {
        "responses": df_records(r),
        "scores": df_records(s),
        "summary": _summary_rows(r, s),
    }


@router.get("/summary")
def summary():
    r, s = _load_csvs()
    return {"summary": _summary_rows(r, s)}


@router.get("/runs")
def list_runs():
    runs = sorted(RESULTS_DIR.glob("scores_*.csv"), reverse=True)
    return {"runs": [p.stem.replace("scores_", "") for p in runs]}


@router.get("/runs/{stamp}")
def get_run(stamp: str):
    s_path = RESULTS_DIR / f"scores_{stamp}.csv"
    r_path = RESULTS_DIR / f"responses_{stamp}.csv"
    if not s_path.exists():
        raise HTTPException(404, f"Run '{stamp}' not found.")
    s = pd.read_csv(s_path)
    r = pd.read_csv(r_path) if r_path.exists() else pd.DataFrame()
    return {"responses": df_records(r), "scores": df_records(s)}


@router.get("/export")
def export_json():
    """Structured JSON export as a download."""
    from datetime import datetime

    from fastapi.responses import JSONResponse

    r, s = _load_csvs()
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "llm-eval v2",
        "responses": df_records(r),
        "scores": df_records(s),
        "summary": _summary_rows(r, s),
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": 'attachment; filename="llm-eval-export.json"'},
    )


def _leaderboard_rows() -> list[dict]:
    r, s = _load_csvs()
    return sorted(_summary_rows(r, s), key=lambda x: -x["value_index"])


@router.get("/embed.md")
def embed_markdown():
    """Markdown leaderboard snippet for a GitHub README."""
    from datetime import datetime

    from fastapi.responses import PlainTextResponse

    rows = _leaderboard_rows()
    lines = [
        "## LLM Eval Leaderboard",
        "",
        "| Rank | Model | Value Index | Quality | Latency | Ans/$20 |",
        "|-----:|-------|------------:|--------:|--------:|--------:|",
    ]
    medals = ["🥇", "🥈", "🥉"]
    for i, row in enumerate(rows):
        rank = medals[i] if i < 3 else str(i + 1)
        lines.append(
            f"| {rank} | {row['meta']['alias']} | {row['value_index']:.2f} "
            f"| {row['composite_score']:.2f}/5 | {row['avg_total_latency_ms']:.0f}ms "
            f"| {row['api_answers_for_20usd']:,} |"
        )
    lines += ["", f"*Generated by [llm-eval](https://github.com/abhishek2395/llm-eval) · {datetime.now():%Y-%m-%d}*"]
    return PlainTextResponse("\n".join(lines))


@router.get("/report.html")
def report_html():
    """Self-contained HTML report — no server needed, share the file."""
    from datetime import datetime

    from fastapi.responses import HTMLResponse

    rows = _leaderboard_rows()
    r, s = _load_csvs()
    valid = s[s["judge_error"].isna()] if "judge_error" in s.columns else s

    def bar(v: float, vmax: float, color: str) -> str:
        pct = 0 if vmax <= 0 else round(v / vmax * 100)
        return (f'<div style="background:rgba(255,255,255,0.07);border-radius:3px;height:8px;width:160px;display:inline-block;vertical-align:middle;">'
                f'<div style="width:{pct}%;height:100%;border-radius:3px;background:{color};"></div></div>')

    vmax = max((x["value_index"] for x in rows), default=1)
    body_rows = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, row in enumerate(rows):
        m = row["meta"]
        body_rows += f"""
<tr style="border-bottom:1px solid #1e2d45;">
  <td style="padding:10px 12px;font-size:18px;">{medals[i] if i < 3 else i + 1}</td>
  <td style="padding:10px 12px;"><span style="color:{m['color']};font-weight:600;">{m['alias']}</span>
      <span style="color:#64748b;font-size:11px;"> {row['model']}</span></td>
  <td style="padding:10px 12px;font-weight:600;color:#f59e0b;">{row['value_index']:.2f} {bar(row['value_index'], vmax, m['color'])}</td>
  <td style="padding:10px 12px;">{row['composite_score']:.2f}/5</td>
  <td style="padding:10px 12px;">{row['avg_total_latency_ms']:.0f}ms</td>
  <td style="padding:10px 12px;">{row['avg_tokens_per_sec']:.0f} tok/s</td>
  <td style="padding:10px 12px;">{row['refusal_rate_pct']:.0f}%</td>
  <td style="padding:10px 12px;">{row['api_answers_for_20usd']:,}</td>
</tr>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>LLM Eval Report — {datetime.now():%Y-%m-%d}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ background:#0a0e1a; color:#e2e8f0; font-family: -apple-system, 'Segoe UI', sans-serif;
         margin:0; padding:40px 24px; }}
  .mono {{ font-family: ui-monospace, 'SF Mono', Menlo, monospace; }}
  table {{ border-collapse:collapse; width:100%; max-width:1100px; margin:0 auto;
           font-size:14px; }}
  th {{ text-align:left; padding:10px 12px; color:#64748b; font-size:10px;
        text-transform:uppercase; letter-spacing:0.1em; border-bottom:2px solid #1e2d45; }}
  @media print {{ body {{ background:#fff; color:#111; }} }}
</style></head><body>
<div style="max-width:1100px;margin:0 auto 28px;">
  <div class="mono" style="color:#f59e0b;font-size:11px;letter-spacing:0.18em;">LLM EVAL FRAMEWORK</div>
  <h1 style="margin:4px 0;font-size:26px;">Model Comparison Report</h1>
  <div style="color:#64748b;font-size:13px;">
    {len(rows)} models · {int(valid['prompt_id'].nunique()) if not valid.empty else 0} prompts ·
    6-dimension LLM-as-judge · generated {datetime.now():%Y-%m-%d %H:%M}
  </div>
</div>
<table class="mono">
<thead><tr><th>Rank</th><th>Model</th><th>Value Index</th><th>Quality</th>
<th>Latency</th><th>Speed</th><th>Refusals</th><th>Ans/$20</th></tr></thead>
<tbody>{body_rows}</tbody>
</table>
<div style="max-width:1100px;margin:24px auto;color:#64748b;font-size:11px;" class="mono">
  Value Index = weighted quality ÷ verbosity penalty · single OpenRouter key · CSV persistence ·
  github.com/abhishek2395/llm-eval
</div>
</body></html>"""
    return HTMLResponse(
        html,
        headers={"Content-Disposition": 'inline; filename="llm-eval-report.html"'},
    )


@router.delete("/model/{model_id:path}")
def delete_model_results(model_id: str):
    """Destructive: drop every row for this model from results.csv + scores.csv."""
    r, s = _load_csvs()
    if (r.empty or model_id not in r["model"].values) and \
       (s.empty or model_id not in s["model"].values):
        raise HTTPException(404, f"No results for model '{model_id}'.")
    if not r.empty:
        r[r["model"] != model_id].to_csv(RESULTS_FILE, index=False)
    if not s.empty:
        s[s["model"] != model_id].to_csv(SCORES_FILE, index=False)
    return {"deleted": model_id}
