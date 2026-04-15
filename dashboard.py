"""
dashboard.py  —  LLM Evaluation Framework · Modern Dark Dashboard
Run:  streamlit run dashboard.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    RESULTS_FILE, SCORES_FILE, PROMPTS_FILE, RESULTS_DIR,
    PROVIDER_META, OPENROUTER_API_KEY, MODEL_ALIASES,
)
from openrouter_client import fetch_model_catalog
from live_eval import run_model_live, build_efficiency_row

# ── Palette & theme constants ─────────────────────────────────────────────────
BG        = "#0a0e1a"
BG2       = "#111827"
BG3       = "#1a2235"
BORDER    = "#1e2d45"
TEXT      = "#e2e8f0"
TEXT_MUTE = "#64748b"
AMBER     = "#f59e0b"
AMBER_DIM = "#92400e"
TEAL      = "#14b8a6"
BLUE      = "#3b82f6"
RED       = "#ef4444"
GREEN     = "#22c55e"
PURPLE    = "#a855f7"

MODEL_COLORS = {
    "claude-sonnet-4-20250514": AMBER,
    "gpt-4o-mini":              TEAL,
    "gpt-4o":                   BLUE,
    "claude-opus-4-20250514":   PURPLE,
}

def mcolor(model: str) -> str:
    # OpenRouter format: "provider/model-name"
    provider = model.split("/")[0] if "/" in model else model.split("-")[0]
    meta = PROVIDER_META.get(provider, {})
    if meta:
        return meta["color"]
    # Fallback: scan for partial matches
    for k, v in MODEL_COLORS.items():
        if k in model:
            return v
    return TEXT_MUTE

def short_name(model: str) -> str:
    """
    Return a clean display name for a model ID.
    Checks MODEL_ALIASES first (exact match), then auto-parses as fallback.
    Examples:
        deepseek/deepseek-chat        → DeepSeek V3
        openai/o4-mini                → GPT o4-mini
        anthropic/claude-sonnet-4-5   → Claude Sonnet 4.5
        some/unknown-model-xyz        → Unknown-model (auto-parsed)
    """
    # 1. Exact alias match
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # 2. Case-insensitive match (handles minor ID variations)
    lower = model.lower()
    for k, v in MODEL_ALIASES.items():
        if k.lower() == lower:
            return v
    # 3. Auto-parse fallback: take slug after "/"
    part = model.split("/")[-1] if "/" in model else model
    seg  = part.split("-")
    return seg[0].capitalize() if len(seg) < 3 else f"{seg[0].capitalize()} {seg[1].capitalize()}"

def score_color(v: float, lo: float = 1, hi: float = 5) -> str:
    t = (v - lo) / (hi - lo)
    if t >= 0.7: return GREEN
    if t >= 0.4: return AMBER
    return RED

def pct_color(v: float, good_low=True) -> str:
    """good_low=True means lower is better (e.g. refusal rate, latency)."""
    return GREEN if (v < 15 if good_low else v > 3.5) else (AMBER if (v < 30 if good_low else v > 2.5) else RED)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Mono', monospace", color=TEXT, size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickfont=dict(size=10)),
)

def apply_layout(fig, **overrides):
    fig.update_layout(**{**PLOTLY_LAYOUT, **overrides})
    return fig

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert a 6-digit hex color string to an rgba() string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval · Dashboard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif;
}}

/* Hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stSidebar"] {{ background: {BG2} !important; border-right: 1px solid {BORDER}; }}
section.main > div {{ padding-top: 1rem !important; }}

/* Metric overrides */
[data-testid="metric-container"] {{
    background: {BG3} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}}
[data-testid="metric-container"] label {{
    color: {TEXT_MUTE} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.75rem !important; }}

/* Card */
.card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}
.card-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXT_MUTE};
    margin-bottom: 0.75rem;
}}

/* Verdict banner */
.verdict-wrap {{
    border-radius: 14px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 0.75rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}}
.verdict-wrap::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}}
.verdict-winner  {{ background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.3); --accent: {AMBER}; }}
.verdict-runner  {{ background: rgba(20,184,166,0.05); border-color: rgba(20,184,166,0.25); --accent: {TEAL}; }}
.verdict-neutral {{ background: rgba(255,255,255,0.03); border-color: {BORDER}; --accent: {TEXT_MUTE}; }}

/* Pill badges */
.pill {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    margin: 2px 3px 2px 0;
    border: 1px solid;
}}
.pill-green  {{ color: {GREEN};  border-color: rgba(34,197,94,0.3);  background: rgba(34,197,94,0.08); }}
.pill-amber  {{ color: {AMBER};  border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.08); }}
.pill-red    {{ color: {RED};    border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.08); }}
.pill-blue   {{ color: {BLUE};   border-color: rgba(59,130,246,0.3); background: rgba(59,130,246,0.08); }}
.pill-mute   {{ color: {TEXT_MUTE}; border-color: {BORDER}; background: rgba(255,255,255,0.03); }}

/* Score bar */
.score-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 5px 0;
    font-size: 0.8rem;
}}
.score-label {{
    color: {TEXT_MUTE};
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    width: 170px;
    flex-shrink: 0;
}}
.score-bar-bg {{
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}}
.score-bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}}
.score-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: {TEXT};
    width: 28px;
    text-align: right;
    flex-shrink: 0;
}}

/* Section header */
.section-hdr {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: {TEXT_MUTE};
    padding: 0.4rem 0 0.6rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1rem;
}}

/* Per-prompt table rows */
.prompt-row {{
    background: {BG3};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: border-color 0.2s;
}}
.prompt-row:hover {{ border-color: {AMBER}40; }}

/* Divider */
.hdivider {{ border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0; }}

/* Selectbox / multiselect */
div[data-baseweb="select"] > div {{
    background-color: {BG3} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
.stSelectbox label, .stMultiSelect label {{
    color: {TEXT_MUTE} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace !important;
}}
button[data-testid="baseButton-secondary"] {{
    background: {BG3} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}}
button[data-testid="baseButton-secondary"]:hover {{
    border-color: {AMBER}80 !important;
    color: {AMBER} !important;
}}

/* Expander */
details summary {{
    color: {TEXT_MUTE} !important;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace !important;
}}
[data-testid="stExpander"] {{
    background: {BG3} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_all():
    r = pd.read_csv(RESULTS_FILE)  if RESULTS_FILE.exists()  else pd.DataFrame()
    s = pd.read_csv(SCORES_FILE)   if SCORES_FILE.exists()   else pd.DataFrame()
    e_path = RESULTS_DIR / "efficiency_summary.csv"
    e = pd.read_csv(e_path) if e_path.exists() else pd.DataFrame()
    p = json.loads(PROMPTS_FILE.read_text()) if PROMPTS_FILE.exists() else []
    runs = sorted(RESULTS_DIR.glob("scores_*.csv"), reverse=True)
    return r, s, e, p, runs


responses_df, scores_df, summary_df, prompts, runs = load_all()

# Soft guard — page still renders even with empty CSVs (user can add models live)
valid = scores_df[scores_df["judge_error"].isna()].copy() \
        if not scores_df.empty else pd.DataFrame()
models_all = sorted(valid["model"].unique().tolist()) if not valid.empty else []



# ── Session state: master data store ─────────────────────────────────────────
def _init_state(responses_df, scores_df, summary_df):
    """Seed session state from disk on first load."""
    if "master_responses" not in st.session_state:
        st.session_state.master_responses = responses_df.copy()
    if "master_scores" not in st.session_state:
        st.session_state.master_scores = scores_df.copy()
    if "master_summary" not in st.session_state:
        st.session_state.master_summary = summary_df.copy()
    if "hidden_models" not in st.session_state:
        st.session_state.hidden_models = []
    if "eval_log" not in st.session_state:
        st.session_state.eval_log = []
    if "extra_models" not in st.session_state:
        st.session_state.extra_models = []  # custom model IDs added via text input

_init_state(responses_df, scores_df, summary_df)

def get_active_models() -> list[str]:
    all_m = sorted(st.session_state.master_scores["model"].unique().tolist())             if not st.session_state.master_scores.empty else []
    return [m for m in all_m if m not in st.session_state.hidden_models]

def rebuild_efficiency_summary():
    """Recompute efficiency_summary from master DataFrames."""
    from generate_demo_data import compute_efficiency_summary
    if st.session_state.master_responses.empty or st.session_state.master_scores.empty:
        return
    try:
        new_e = compute_efficiency_summary(
            st.session_state.master_responses,
            st.session_state.master_scores,
        )
        st.session_state.master_summary = new_e
    except Exception as exc:
        st.warning(f"Could not rebuild efficiency summary: {exc}")

# ── Live eval panel ───────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>⚡ Model Selection & Evaluation</div>", unsafe_allow_html=True)

MAX_MODELS = 5
effective_key = OPENROUTER_API_KEY or ""

# Build model option list: custom extras → already evaluated → popular from aliases
_evaluated    = sorted(st.session_state.master_scores["model"].unique().tolist()) \
                if not st.session_state.master_scores.empty else []
_popular      = [m for m in sorted(MODEL_ALIASES.keys()) if m not in _evaluated]
_extra        = [m for m in st.session_state.extra_models
                 if m not in _evaluated and m not in _popular]
_all_options  = _extra + _evaluated + _popular

active_now = get_active_models()

# ── Row 1: dropdown  |  Run button  |  key status ────────────────────────────
sel_c, run_c, key_c = st.columns([5, 2, 1])

with sel_c:
    _default_sel = [m for m in active_now if m in _all_options]
    selected_models = st.multiselect(
        "Models",
        options=_all_options,
        default=_default_sel,
        format_func=short_name,
        placeholder="Select up to 5 models to compare…",
        label_visibility="collapsed",
        key="model_selector",
    )
    if len(selected_models) > MAX_MODELS:
        st.warning(f"Max {MAX_MODELS} models allowed — using the first {MAX_MODELS}.")
        selected_models = selected_models[:MAX_MODELS]

# Sync hidden_models: deselected → hide, re-selected → restore
for m in active_now:
    if m not in selected_models and m not in st.session_state.hidden_models:
        st.session_state.hidden_models.append(m)
for m in selected_models:
    if m in st.session_state.hidden_models:
        st.session_state.hidden_models.remove(m)

_needs_eval = [m for m in selected_models if m not in _evaluated]

with run_c:
    run_clicked = st.button(
        "▶ Run Evaluation",
        disabled=not effective_key or not _needs_eval,
        type="primary",
        use_container_width=True,
    )

with key_c:
    if effective_key:
        st.markdown('<span class="pill pill-green">🔑 key set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-red">⚠ no key</span>', unsafe_allow_html=True)

# ── Row 2: custom model text input ───────────────────────────────────────────
cust_c, add_c = st.columns([5, 2])
with cust_c:
    custom_model_id = st.text_input(
        "Custom model",
        placeholder="Type a model ID not in the list — e.g. google/gemini-2.5-pro",
        label_visibility="collapsed",
        key="custom_model_input",
    )
with add_c:
    add_custom_clicked = st.button(
        "＋ Add to list",
        disabled=not custom_model_id.strip() or len(selected_models) >= MAX_MODELS,
        use_container_width=True,
    )

if add_custom_clicked and custom_model_id.strip():
    cm = custom_model_id.strip()
    if cm not in st.session_state.extra_models:
        st.session_state.extra_models.insert(0, cm)
    # Remove from hidden so it can be auto-selected
    if cm in st.session_state.hidden_models:
        st.session_state.hidden_models.remove(cm)
    st.rerun()

# ── Status hint ───────────────────────────────────────────────────────────────
if _needs_eval:
    names = ", ".join(short_name(m) for m in _needs_eval)
    st.caption(f"⚡ Ready to evaluate: **{names}** — click **▶ Run Evaluation** to start.")
elif selected_models:
    st.caption(f"✓ {len(selected_models)} model(s) loaded with results. Charts updated below ↓")
else:
    st.caption("Select models from the dropdown above to begin comparing.")

st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

# ── Live eval execution ───────────────────────────────────────────────────────
if run_clicked and _needs_eval:
    if not effective_key:
        st.error("Set your OpenRouter API key in .env first.")
    else:
        prompts_list = json.loads(PROMPTS_FILE.read_text()) if PROMPTS_FILE.exists() else []
        if not prompts_list:
            st.error("No prompts found — ensure prompts.json exists.")
        else:
            for model_to_add in _needs_eval:
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;'
                    f'color:{AMBER};margin-bottom:8px;">'
                    f'Evaluating: <b>{short_name(model_to_add)}</b>'
                    f' <span style="color:{TEXT_MUTE};">({model_to_add})</span>'
                    f' · {len(prompts_list)} prompts</div>',
                    unsafe_allow_html=True,
                )

                progress_bar  = st.progress(0, text="Starting…")
                status_text   = st.empty()
                partial_table = st.empty()

                new_responses, new_scores = [], []
                completed = 0
                cached_count = 0

                gen = run_model_live(model_to_add, effective_key, prompts_list)
                for update in gen:
                    t = update["type"]

                    if t == "error":
                        st.error(f"Eval error: {update['message']}")
                        break

                    elif t == "start":
                        status_text.markdown(
                            f'<span style="font-size:0.75rem;color:{TEXT_MUTE};">'
                            f'Starting {update["total"]} prompts…</span>',
                            unsafe_allow_html=True,
                        )

                    elif t == "cached":
                        new_responses.append(update["response_row"])
                        new_scores.append(update["score_row"])
                        cached_count += 1
                        completed += 1
                        pct = completed / len(prompts_list)
                        progress_bar.progress(pct, text=f"Cached: {update['prompt_id']} ({completed}/{len(prompts_list)})")

                    elif t == "progress":
                        stage_label = "🔄 Calling model…" if update["stage"] == "inference" else "⚖️ Judging…"
                        status_text.markdown(
                            f'<span style="font-size:0.75rem;color:{TEXT_MUTE};">'
                            f'{stage_label} <code>{update["prompt_id"]}</code> '
                            f'({update["idx"]+1}/{update["total"]})</span>',
                            unsafe_allow_html=True,
                        )

                    elif t == "result":
                        new_responses.append(update["response_row"])
                        new_scores.append(update["score_row"])
                        completed += 1
                        pct = completed / len(prompts_list)

                        if new_scores:
                            partial_df = pd.DataFrame(new_scores)
                            cols_show = [c for c in ["prompt_id", "composite_score", "accuracy",
                                                      "hallucination_resistance", "task_completion"]
                                         if c in partial_df.columns]
                            partial_table.dataframe(
                                partial_df[cols_show],
                                use_container_width=True,
                                height=min(250, 50 + 35 * len(new_scores)),
                            )

                        sc = update["score_row"].get("composite_score", 0)
                        progress_bar.progress(pct, text=f"✓ {update['prompt_id']} · score {sc:.2f} ({completed}/{len(prompts_list)})")

                    elif t == "done":
                        progress_bar.progress(1.0, text=f"✅ Complete — {completed} prompts ({cached_count} cached)")
                        status_text.empty()

                        if new_responses:
                            new_r_df = pd.DataFrame(new_responses)
                            new_s_df = pd.DataFrame(new_scores)

                            mr = st.session_state.master_responses
                            ms = st.session_state.master_scores
                            mr = mr[mr["model"] != model_to_add] if not mr.empty else mr
                            ms = ms[ms["model"] != model_to_add] if not ms.empty else ms

                            st.session_state.master_responses = pd.concat([mr, new_r_df], ignore_index=True)
                            st.session_state.master_scores    = pd.concat([ms, new_s_df], ignore_index=True)

                            eff_row = build_efficiency_row(model_to_add, new_responses, new_scores)
                            me = st.session_state.master_summary
                            if not me.empty:
                                me = me[me["model"] != model_to_add]
                            st.session_state.master_summary = pd.concat(
                                [me, pd.DataFrame([eff_row])], ignore_index=True
                            )

                            st.session_state.eval_log.append({
                                "model": model_to_add,
                                "prompts": completed,
                                "cached": cached_count,
                                "avg_score": round(new_s_df["composite_score"].mean(), 2),
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                            })

                        st.success(f"**{short_name(model_to_add)}** evaluated successfully.")

            st.rerun()

# ── Pull active data from session state for all chart sections ────────────────
responses_df  = st.session_state.master_responses
scores_df     = st.session_state.master_scores
summary_df    = st.session_state.master_summary
valid         = scores_df[scores_df["judge_error"].isna()].copy() \
                if not scores_df.empty else pd.DataFrame()
prompts       = json.loads(PROMPTS_FILE.read_text()) if PROMPTS_FILE.exists() else []

# ── Live model catalog (cached 5 min) ─────────────────────────────────────────
@st.cache_data(ttl=300)
def get_catalog():
    return fetch_model_catalog()

# ── Top bar ───────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])

with hcol1:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 0.25rem">
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                   text-transform:uppercase;letter-spacing:0.18em;color:{AMBER};">
        LLM EVAL FRAMEWORK
      </span><br>
      <span style="font-size:1.6rem;font-weight:600;line-height:1.2;">
        Model Comparison Dashboard
      </span><br>
      <span style="color:{TEXT_MUTE};font-size:0.82rem;">
        Quality · Efficiency · Value-per-$20 · 300+ models via OpenRouter
      </span>
    </div>
    """, unsafe_allow_html=True)

with hcol2:
    run_labels = ["Latest run"] + [r.stem.replace("scores_", "") for r in runs]
    selected_run = st.selectbox("Run", run_labels, label_visibility="visible")
    if selected_run != "Latest run" and runs:
        idx = run_labels.index(selected_run) - 1
        scores_df = pd.read_csv(runs[idx])
        rpath = RESULTS_DIR / runs[idx].name.replace("scores_", "responses_")
        if rpath.exists():
            responses_df = pd.read_csv(rpath)
        valid = scores_df[scores_df["judge_error"].isna()].copy()

st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

# ── Model catalog browser ──────────────────────────────────────────────────────
with st.expander("🗂  Browse & select models from OpenRouter catalog (300+ models)", expanded=False):
    catalog = get_catalog()

    cat_c1, cat_c2, cat_c3, cat_c4 = st.columns([2, 1.5, 1.5, 1])
    with cat_c1:
        cat_search = st.text_input("Search models", placeholder="gemini, llama, deepseek…",
                                   label_visibility="visible")
    with cat_c2:
        provider_opts = ["All providers"] + sorted(catalog.providers())
        cat_provider = st.selectbox("Provider", provider_opts, label_visibility="visible")
    with cat_c3:
        price_filter = st.selectbox("Price tier", ["All", "Free", "< $1/1M tokens",
                                                    "$1–5/1M tokens", "> $5/1M tokens"],
                                    label_visibility="visible")
    with cat_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        show_n = st.number_input("Show top N", min_value=5, max_value=100, value=20, step=5)

    # Filter catalog
    display_models = catalog.all()
    if cat_search:
        display_models = catalog.search(cat_search)
    if cat_provider != "All providers":
        display_models = [m for m in display_models if m.provider == cat_provider]
    if price_filter == "Free":
        display_models = [m for m in display_models if m.is_free]
    elif price_filter == "< $1/1M tokens":
        display_models = [m for m in display_models if 0 < m.input_price_per_1m < 1]
    elif price_filter == "$1–5/1M tokens":
        display_models = [m for m in display_models if 1 <= m.input_price_per_1m <= 5]
    elif price_filter == "> $5/1M tokens":
        display_models = [m for m in display_models if m.input_price_per_1m > 5]

    display_models = display_models[:int(show_n)]

    if display_models:
        # Render as styled table
        table_rows = ""
        for m in display_models:
            pc = m.provider_color
            free_badge = f'<span style="color:#22c55e;font-size:0.65rem;border:1px solid #22c55e33;padding:1px 5px;border-radius:10px;">FREE</span>' if m.is_free else ""
            price_str = "Free" if m.is_free else f"${m.input_price_per_1m:.3f} / ${m.output_price_per_1m:.3f}"
            ctx_str = f"{m.context_length // 1000}K" if m.context_length >= 1000 else str(m.context_length)
            ans_str = f"{m.answers_for_20_usd:,}" if not m.is_free else "∞"
            table_rows += f"""
<tr style="border-bottom:1px solid {BORDER};">
  <td style="padding:6px 8px;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:{TEXT};">{m.id}</td>
  <td style="padding:6px 8px;"><span style="color:{pc};font-size:0.72rem;font-weight:600;">{m.provider_display}</span></td>
  <td style="padding:6px 8px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:{TEXT_MUTE};">{price_str} {free_badge}</td>
  <td style="padding:6px 8px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:{TEXT_MUTE};">{ctx_str}</td>
  <td style="padding:6px 8px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:{AMBER};">{ans_str}</td>
</tr>"""

        st.markdown(f"""
<div style="overflow-x:auto;border:1px solid {BORDER};border-radius:8px;">
<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
  <thead>
    <tr style="background:{BG3};border-bottom:2px solid {BORDER};">
      <th style="padding:8px;text-align:left;font-family:'IBM Plex Mono',monospace;
                 font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTE};">Model ID</th>
      <th style="padding:8px;text-align:left;font-family:'IBM Plex Mono',monospace;
                 font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTE};">Provider</th>
      <th style="padding:8px;text-align:left;font-family:'IBM Plex Mono',monospace;
                 font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTE};">Input / Output $/1M tok</th>
      <th style="padding:8px;text-align:left;font-family:'IBM Plex Mono',monospace;
                 font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTE};">Context</th>
      <th style="padding:8px;text-align:left;font-family:'IBM Plex Mono',monospace;
                 font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTE};">Ans/$20 API</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>
</div>
<div style="margin-top:8px;font-size:0.7rem;color:{TEXT_MUTE};font-family:'IBM Plex Mono',monospace;">
  Showing {len(display_models)} of {len(catalog)} models · To eval: 
  <code style="background:{BG3};padding:2px 6px;border-radius:4px;">python evaluate.py --models {display_models[0].id if display_models else 'model/id'} ...</code>
</div>
""", unsafe_allow_html=True)

        # Provider breakdown
        prov_counts = {}
        for m in catalog.all():
            prov_counts[m.provider_display] = prov_counts.get(m.provider_display, 0) + 1
        prov_df = pd.DataFrame(list(prov_counts.items()), columns=["provider", "count"]) \
                    .sort_values("count", ascending=True).tail(12)
        fig_prov = go.Figure(go.Bar(
            x=prov_df["count"], y=prov_df["provider"], orientation="h",
            marker_color=[PROVIDER_META.get(p.lower().replace(" ", "-").replace("alibaba", "qwen")
                          .replace("xai / grok", "x-ai").replace("meta", "meta-llama"), {}).get("color", TEXT_MUTE)
                          for p in prov_df["provider"]],
            text=prov_df["count"], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color=TEXT),
        ))
        apply_layout(fig_prov, height=320, showlegend=False,
                     margin=dict(l=8, r=50, t=8, b=8))
        fig_prov.update_xaxes(title_text="models available", title_font=dict(size=10))
        st.markdown(f"<div class='card-title' style='margin-top:1rem;'>Models available by provider</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(fig_prov, use_container_width=True)
    else:
        st.info("No models match your filters.")

st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

# ── Guard: need at least one evaluated model before showing charts ─────────────
if not selected_models or all(m not in _evaluated for m in selected_models):
    st.info("Select models above and click **▶ Run Evaluation** to generate results.")
    st.stop()

# Only chart models that have been evaluated
selected_models = [m for m in selected_models if m in _evaluated]
if not selected_models:
    st.stop()

st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)

# Filter everything for charts
filt   = valid[valid["model"].isin(selected_models)].copy()
filt_r = responses_df[responses_df["model"].isin(selected_models)].copy() \
         if not responses_df.empty else pd.DataFrame()
filt_e = summary_df[summary_df["model"].isin(selected_models)].copy() \
         if not summary_df.empty else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-hdr'>Overview</div>", unsafe_allow_html=True)

best_comp = filt.groupby("model")["composite_score"].mean().max() if not filt.empty else 0

for model in selected_models:
    m_s = filt[filt["model"] == model]
    m_r = filt_r[filt_r["model"] == model] if not filt_r.empty else pd.DataFrame()
    m_e = filt_e[filt_e["model"] == model] if not filt_e.empty else pd.DataFrame()

    mc   = mcolor(model)
    comp = m_s["composite_score"].mean() if not m_s.empty else 0
    vi   = m_s["value_index"].mean()     if "value_index" in m_s.columns else 0
    lat  = m_r["total_latency_ms"].mean() if not m_r.empty and "total_latency_ms" in m_r.columns else 0
    ref  = m_e["refusal_rate_pct"].values[0] if not m_e.empty and "refusal_rate_pct" in m_e.columns else 0

    st.markdown(
        f"<div style='font-size:0.68rem;color:{mc};font-family:\"IBM Plex Mono\",monospace;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin:0.75rem 0 0.25rem;'>"
        f"{model}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Quality", f"{comp:.2f}/5",
                  delta=f"+{comp-best_comp:.2f}" if comp == best_comp else f"{comp-best_comp:.2f}")
    with c2:
        best_vi_flag = (not filt_e.empty and not m_e.empty and
                        "value_index" in filt_e.columns and vi == filt_e["value_index"].max())
        st.metric("Value Index", f"{vi:.2f}", delta="best" if best_vi_flag else None)
    with c3:
        st.metric("Avg Latency", f"{lat:.0f} ms" if lat else "—")
    with c4:
        st.metric("Refusal Rate", f"{ref:.0f}%" if ref else "0%",
                  delta="↓ good" if ref < 10 else "↑ high",
                  delta_color="normal" if ref < 10 else "inverse")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VERDICT BANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("<div class='section-hdr'>Subscription Verdict — which model for your $20/month?</div>",
            unsafe_allow_html=True)

if not filt_e.empty:
    ranked = filt_e.sort_values("value_index", ascending=False).reset_index(drop=True)
    styles = ["verdict-winner", "verdict-runner", "verdict-neutral"]
    medals = ["🥇", "🥈", "🥉"]

    USE_CASES = {
        "anthropic/claude-sonnet-4-5": {
            "wins":  ["Complex reasoning", "Code review", "Long-form writing", "Hallucination-sensitive tasks"],
            "loses": ["High-volume tasks", "Latency-critical apps", "Budget-first API usage"],
        },
        "openai/gpt-4o-mini": {
            "wins":  ["Fast Q&A", "Simple drafts", "High-volume repetitive tasks", "Low-latency apps"],
            "loses": ["Deep analysis", "Nuanced reasoning", "Hallucination traps"],
        },
        "google/gemini-2.0-flash-001": {
            "wins":  ["Fastest first-token", "Long context tasks", "Cost-efficient volume", "Instruction following"],
            "loses": ["Deepest reasoning", "Edge-case accuracy"],
        },
        "deepseek/deepseek-chat": {
            "wins":  ["Coding tasks", "Cost-efficient analysis", "Math & reasoning", "STEM queries"],
            "loses": ["Latency-sensitive use", "Western cultural context"],
        },
        "meta-llama/llama-3.3-70b-instruct": {
            "wins":  ["Open-weight transparency", "No-lock-in pipeline", "Budget bulk tasks"],
            "loses": ["Frontier accuracy", "Hallucination resistance", "Reliability at edge cases"],
        },
        # fallback for any other model
        "__default__": {
            "wins":  ["General purpose tasks"],
            "loses": ["Specialized benchmarks"],
        },
    }

    vcols = st.columns(len(ranked))
    for i, (_, row) in enumerate(ranked.iterrows()):
        model = row["model"]
        style = styles[min(i, 2)]
        medal = medals[min(i, 2)]
        cases = USE_CASES.get(model, USE_CASES["__default__"])

        vi    = row.get("value_index", 0)
        comp  = row.get("composite_score", 0)
        lat   = row.get("avg_total_latency_ms", 0)
        ftl   = row.get("avg_first_token_ms", 0)
        tps   = row.get("avg_tokens_per_sec", 0)
        ref   = row.get("refusal_rate_pct", 0)
        out_t = row.get("avg_output_tokens", 0)
        vr    = row.get("avg_verbosity_ratio", 0)
        api_n = row.get("api_answers_for_20usd", 0)
        cons  = row.get("consistency_score", 0)
        plan  = row.get("plan", "—")
        dlim  = row.get("daily_msg_limit", "—")

        vc = "pill-green" if vi >= 3.5 else "pill-amber" if vi >= 2.5 else "pill-red"
        lc = "pill-green" if lat < 1200 else "pill-amber" if lat < 2500 else "pill-red"
        rc = "pill-green" if ref < 10 else "pill-amber" if ref < 20 else "pill-red"

        wins_html  = "".join(f'<span class="pill pill-green">{w}</span>' for w in cases["wins"])
        loses_html = "".join(f'<span class="pill pill-red">{w}</span>'   for w in cases["loses"])

        with vcols[i]:
            st.markdown(f"""
<div class="verdict-wrap {style}">
  <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:0.9rem;">
    <span style="font-size:1.3rem;">{medal}</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:0.95rem;">{model}</span>
  </div>
  <div style="font-size:0.72rem;color:{TEXT_MUTE};margin-bottom:0.6rem;
              font-family:'IBM Plex Mono',monospace;">{plan} · $20/mo · ~{dlim} msgs/day</div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-bottom:1rem;">
    <div>
      <div style="font-size:0.62rem;color:{TEXT_MUTE};text-transform:uppercase;
                  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;">Value Index</div>
      <div style="font-size:1.5rem;font-weight:600;font-family:'IBM Plex Mono',monospace;
                  color:{'#f59e0b' if i==0 else TEXT};">{vi:.2f}</div>
    </div>
    <div>
      <div style="font-size:0.62rem;color:{TEXT_MUTE};text-transform:uppercase;
                  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;">Quality</div>
      <div style="font-size:1.5rem;font-weight:600;font-family:'IBM Plex Mono',monospace;">{comp:.2f}</div>
    </div>
    <div>
      <div style="font-size:0.62rem;color:{TEXT_MUTE};text-transform:uppercase;
                  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;">Latency</div>
      <div style="font-size:1.1rem;font-weight:500;font-family:'IBM Plex Mono',monospace;">{lat:.0f}ms</div>
    </div>
    <div>
      <div style="font-size:0.62rem;color:{TEXT_MUTE};text-transform:uppercase;
                  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;">API ans./$20</div>
      <div style="font-size:1.1rem;font-weight:500;font-family:'IBM Plex Mono',monospace;">{api_n:,}</div>
    </div>
  </div>

  <div style="margin-bottom:0.6rem;">
    <span class="pill {vc}">VI {vi:.2f}</span>
    <span class="pill {lc}">{lat:.0f}ms latency</span>
    <span class="pill {rc}">{ref:.0f}% refusal</span>
    <span class="pill pill-mute">{tps:.0f} tok/s</span>
    <span class="pill pill-mute">{out_t:.0f} out-tokens</span>
    <span class="pill pill-mute">{vr:.1f}× verbose</span>
    <span class="pill pill-mute">consistency {cons:.2f}</span>
  </div>

  <div style="font-size:0.72rem;color:{TEXT_MUTE};margin-bottom:4px;
              font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
              letter-spacing:0.08em;">Best for</div>
  <div style="margin-bottom:0.5rem;">{wins_html}</div>
  <div style="font-size:0.72rem;color:{TEXT_MUTE};margin-bottom:4px;
              font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
              letter-spacing:0.08em;">Watch out for</div>
  <div>{loses_html}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QUALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("<div class='section-hdr'>Quality Analysis</div>", unsafe_allow_html=True)

q_left, q_right = st.columns([1, 1])

DIMS = ["accuracy", "hallucination_resistance", "relevance",
        "instruction_following", "conciseness", "task_completion"]
avail_dims = [d for d in DIMS if d in filt.columns]

# ── Radar chart ───────────────────────────────────────────────────────────────
with q_left:
    st.markdown("<div class='card-title'>Dimension radar</div>", unsafe_allow_html=True)
    if avail_dims:
        means = filt.groupby("model")[avail_dims].mean().reset_index()
        fig_r = go.Figure()
        for _, row in means.iterrows():
            mc = mcolor(row["model"])
            vals = [row[d] for d in avail_dims] + [row[avail_dims[0]]]
            labs = [d.replace("_", " ").title() for d in avail_dims] + \
                   [avail_dims[0].replace("_", " ").title()]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=labs, fill="toself",
                name=short_name(row["model"]),
                line=dict(color=mc, width=2),
                fillcolor=hex_to_rgba(mc) if mc.startswith("#") else mc,
                opacity=0.9,
            ))
        apply_layout(fig_r,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5],
                                tickfont=dict(size=9, color=TEXT_MUTE),
                                gridcolor=BORDER, linecolor=BORDER),
                angularaxis=dict(tickfont=dict(size=10, color=TEXT_MUTE),
                                 gridcolor=BORDER, linecolor=BORDER),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=340, margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

# ── Score heatmap ─────────────────────────────────────────────────────────────
with q_right:
    st.markdown("<div class='card-title'>Score heatmap — model × dimension</div>",
                unsafe_allow_html=True)
    if avail_dims:
        heat = filt.groupby("model")[avail_dims].mean()
        col_labels = [d.replace("_", " ").replace("hallucination resistance", "halluc. resist.")
                      for d in avail_dims]
        fig_h = go.Figure(go.Heatmap(
            z=heat.values,
            x=col_labels,
            y=[short_name(m) for m in heat.index],
            colorscale=[[0, "#1a1a2e"], [0.3, "#7c2d12"], [0.6, "#d97706"], [1, "#22c55e"]],
            zmin=1, zmax=5,
            text=[[f"{v:.2f}" for v in row] for row in heat.values],
            texttemplate="%{text}",
            textfont=dict(size=13, family="IBM Plex Mono"),
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                tickfont=dict(size=9, color=TEXT_MUTE),
                outlinecolor=BORDER,
                outlinewidth=1,
                thickness=12,
                len=0.9,
            ),
        ))
        apply_layout(fig_h, height=340, margin=dict(l=20, r=60, t=10, b=20))
        fig_h.update_xaxes(tickangle=-20, tickfont=dict(size=10))
        st.plotly_chart(fig_h, use_container_width=True)

# ── Score bars per model ──────────────────────────────────────────────────────
st.markdown("<div class='card-title' style='margin-top:0.5rem'>Dimension breakdown</div>",
            unsafe_allow_html=True)
bar_cols = st.columns(len(selected_models))

for i, model in enumerate(selected_models):
    m_s = filt[filt["model"] == model]
    with bar_cols[i]:
        short = short_name(model)
        mc    = mcolor(model)
        rows_html = ""
        for dim in avail_dims:
            val  = m_s[dim].mean() if not m_s.empty else 0
            pct  = (val - 1) / 4 * 100
            col  = score_color(val)
            rows_html += f"""
<div class="score-row">
  <span class="score-label">{dim.replace('_',' ')}</span>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{pct:.0f}%;background:{col};"></div>
  </div>
  <span class="score-val" style="color:{col}">{val:.2f}</span>
</div>"""
        st.markdown(f"""
<div class="card">
  <div class="card-title" style="color:{mc}">{short}</div>
  {rows_html}
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EFFICIENCY & SPEED
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("<div class='section-hdr'>Compute Efficiency & Speed</div>", unsafe_allow_html=True)

if not filt_r.empty:
    e1, e2, e3 = st.columns(3)

    # Latency violin
    with e1:
        st.markdown("<div class='card-title'>Latency distribution (ms)</div>",
                    unsafe_allow_html=True)
        lat_data = filt_r[filt_r["total_latency_ms"] > 0].copy()
        lat_data["short"] = lat_data["model"].str.split("-").str[0]
        fig_lat = go.Figure()
        for model in selected_models:
            md = lat_data[lat_data["model"] == model]["total_latency_ms"]
            if not md.empty:
                fig_lat.add_trace(go.Violin(
                    y=md, name=short_name(model),
                    box_visible=True, meanline_visible=True,
                    line_color=mcolor(model),
                    fillcolor=hex_to_rgba(mcolor(model), 0.13),
                    opacity=0.9,
                ))
        apply_layout(fig_lat, height=280, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=8))
        fig_lat.update_yaxes(title_text="ms", title_font=dict(size=10))
        st.plotly_chart(fig_lat, use_container_width=True)

    # First-token vs total grouped bar
    with e2:
        st.markdown("<div class='card-title'>First-token vs total latency (avg ms)</div>",
                    unsafe_allow_html=True)
        lat_comp = filt_r.groupby("model").agg(
            first_token=("first_token_latency_ms", "mean"),
            total=("total_latency_ms", "mean"),
        ).reset_index()
        lat_comp["short"] = lat_comp["model"].str.split("-").str[0]
        fig_ftl = go.Figure()
        for col_name, col_label, opacity in [("first_token", "First token", 0.6),
                                              ("total", "Total", 1.0)]:
            fig_ftl.add_trace(go.Bar(
                x=lat_comp["short"],
                y=lat_comp[col_name],
                name=col_label,
                marker_color=[mcolor(m) for m in lat_comp["model"]],
                opacity=opacity,
            ))
        apply_layout(fig_ftl, height=280, barmode="group",
                     margin=dict(l=8, r=8, t=8, b=8))
        fig_ftl.update_yaxes(title_text="ms", title_font=dict(size=10))
        st.plotly_chart(fig_ftl, use_container_width=True)

    # Tokens/sec + verbosity
    with e3:
        st.markdown("<div class='card-title'>Tokens/sec · verbosity ratio · output tokens</div>",
                    unsafe_allow_html=True)
        tps_data = filt_r[filt_r["tokens_per_second"] > 0].groupby("model").agg(
            tps=("tokens_per_second", "mean"),
            verb=("verbosity_ratio", "mean"),
            out=("output_tokens", "mean"),
        ).reset_index()
        fig_tps = go.Figure()
        fig_tps.add_trace(go.Bar(
            x=[short_name(m) for m in tps_data["model"]],
            y=tps_data["tps"],
            name="tok/s",
            marker_color=[mcolor(m) for m in tps_data["model"]],
            text=[f"{v:.0f}" for v in tps_data["tps"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        apply_layout(fig_tps, height=280, showlegend=False,
                     margin=dict(l=8, r=8, t=28, b=8))
        fig_tps.update_yaxes(title_text="tokens/sec", title_font=dict(size=10))
        st.plotly_chart(fig_tps, use_container_width=True)

    # Verbosity + output token box plots side by side
    vb1, vb2 = st.columns(2)
    with vb1:
        st.markdown("<div class='card-title'>Output tokens per response</div>",
                    unsafe_allow_html=True)
        fig_out = go.Figure()
        for model in selected_models:
            md = filt_r[(filt_r["model"] == model) & (filt_r["output_tokens"] > 0)]["output_tokens"]
            if not md.empty:
                fig_out.add_trace(go.Box(
                    y=md, name=short_name(model),
                    marker_color=mcolor(model),
                    line_color=mcolor(model),
                    fillcolor=hex_to_rgba(mcolor(model), 0.13),
                ))
        apply_layout(fig_out, height=250, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=8))
        fig_out.update_yaxes(title_text="tokens", title_font=dict(size=10))
        st.plotly_chart(fig_out, use_container_width=True)

    with vb2:
        st.markdown("<div class='card-title'>Verbosity ratio (output ÷ input tokens) — lower is more efficient</div>",
                    unsafe_allow_html=True)
        fig_vr = go.Figure()
        for model in selected_models:
            md = filt_r[(filt_r["model"] == model) & (filt_r["verbosity_ratio"] > 0)]["verbosity_ratio"]
            if not md.empty:
                fig_vr.add_trace(go.Box(
                    y=md, name=short_name(model),
                    marker_color=mcolor(model),
                    line_color=mcolor(model),
                    fillcolor=hex_to_rgba(mcolor(model), 0.13),
                ))
        apply_layout(fig_vr, height=250, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=8))
        fig_vr.update_yaxes(title_text="ratio", title_font=dict(size=10))
        st.plotly_chart(fig_vr, use_container_width=True)

else:
    st.info("Run inference (not just scoring) to see timing metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VALUE FOR $20
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("""
<div class='section-hdr'>
  Value for $20/month
  <span style="font-size:0.55rem;color:#64748b;margin-left:8px;">
    Value Index = weighted quality ÷ verbosity penalty · API answers = $20 ÷ cost-per-call
  </span>
</div>""", unsafe_allow_html=True)

if not filt_e.empty:
    v1, v2, v3 = st.columns(3)

    # Value Index bar
    with v1:
        st.markdown("<div class='card-title'>Value Index (quality per token-dollar)</div>",
                    unsafe_allow_html=True)
        vi_data = filt_e.sort_values("value_index")
        fig_vi = go.Figure(go.Bar(
            x=vi_data["value_index"],
            y=[short_name(m) for m in vi_data["model"]],
            orientation="h",
            marker=dict(
                color=vi_data["value_index"],
                colorscale=[[0, "#1a1a2e"], [0.4, AMBER_DIM], [1, AMBER]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in vi_data["value_index"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color=TEXT),
        ))
        apply_layout(fig_vi, height=220, margin=dict(l=8, r=60, t=8, b=8))
        fig_vi.update_xaxes(range=[0, filt_e["value_index"].max() * 1.25])
        st.plotly_chart(fig_vi, use_container_width=True)

    # API answers for $20
    with v2:
        st.markdown("<div class='card-title'>API answers you can get for $20</div>",
                    unsafe_allow_html=True)
        api_data = filt_e.sort_values("api_answers_for_20usd")
        fig_api = go.Figure(go.Bar(
            x=api_data["api_answers_for_20usd"],
            y=[short_name(m) for m in api_data["model"]],
            orientation="h",
            marker=dict(
                color=api_data["api_answers_for_20usd"],
                colorscale=[[0, "#0f4c5c"], [1, TEAL]],
                showscale=False,
            ),
            text=[f"{int(v):,}" for v in api_data["api_answers_for_20usd"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color=TEXT),
        ))
        apply_layout(fig_api, height=220, margin=dict(l=8, r=80, t=8, b=8))
        fig_api.update_xaxes(range=[0, filt_e["api_answers_for_20usd"].max() * 1.2])
        st.plotly_chart(fig_api, use_container_width=True)

    # Scatter: quality vs latency trade-off
    with v3:
        st.markdown("<div class='card-title'>Quality vs latency trade-off</div>",
                    unsafe_allow_html=True)
        fig_ql = go.Figure()
        for _, row in filt_e.iterrows():
            mc = mcolor(row["model"])
            fig_ql.add_trace(go.Scatter(
                x=[row["avg_total_latency_ms"]],
                y=[row["composite_score"]],
                mode="markers+text",
                name=short_name(row["model"]),
                marker=dict(size=22, color=mc, line=dict(width=2, color=BG3)),
                text=[short_name(row["model"])],
                textposition="top center",
                textfont=dict(size=10, family="IBM Plex Mono", color=mc),
            ))
        apply_layout(fig_ql, height=220, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=8))
        fig_ql.update_xaxes(title_text="Avg latency (ms)", title_font=dict(size=10))
        fig_ql.update_yaxes(title_text="Composite score", range=[0, 5.2],
                            title_font=dict(size=10))
        # Add quadrant annotations
        x_mid = filt_e["avg_total_latency_ms"].mean()
        fig_ql.add_vline(x=x_mid, line_dash="dot", line_color=BORDER, line_width=1)
        fig_ql.add_hline(y=3.5,   line_dash="dot", line_color=BORDER, line_width=1)
        st.plotly_chart(fig_ql, use_container_width=True)

    # Pricing bar — input + output side by side
    if "input_price_per_1m" in filt_e.columns:
        st.markdown("<div class='card-title' style='margin-top:1rem;'>API pricing — input vs output cost per 1M tokens</div>",
                    unsafe_allow_html=True)
        pricing_melt = filt_e[["model", "input_price_per_1m", "output_price_per_1m"]].copy()
        pricing_melt["short"] = pricing_melt["model"].apply(
            lambda m: m.split("/")[-1].replace("-instruct","").replace("-001","")[:22]
        )
        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(
            name="Input $/1M",
            x=pricing_melt["short"],
            y=pricing_melt["input_price_per_1m"],
            marker_color=[mcolor(m) for m in pricing_melt["model"]],
            opacity=0.65,
            text=[f"${v:.2f}" for v in pricing_melt["input_price_per_1m"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_price.add_trace(go.Bar(
            name="Output $/1M",
            x=pricing_melt["short"],
            y=pricing_melt["output_price_per_1m"],
            marker_color=[mcolor(m) for m in pricing_melt["model"]],
            opacity=1.0,
            text=[f"${v:.2f}" for v in pricing_melt["output_price_per_1m"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        apply_layout(fig_price, height=280, barmode="group",
                     margin=dict(l=8, r=8, t=8, b=40))
        fig_price.update_yaxes(title_text="USD per 1M tokens", title_font=dict(size=10))
        st.plotly_chart(fig_price, use_container_width=True)

    ud1, ud2 = st.columns(2)
    with ud1:
        st.markdown("<div class='card-title'>Useful density — quality per 100 output tokens</div>",
                    unsafe_allow_html=True)
        ud_data = filt_e.sort_values("useful_density")
        fig_ud = go.Figure(go.Bar(
            x=[short_name(m) for m in ud_data["model"]],
            y=ud_data["useful_density"],
            marker=dict(
                color=ud_data["useful_density"],
                colorscale=[[0, "#0f2027"], [1, PURPLE]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in ud_data["useful_density"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12),
        ))
        apply_layout(fig_ud, height=220, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=30))
        fig_ud.update_yaxes(title_text="score / 100 tokens", title_font=dict(size=10))
        st.plotly_chart(fig_ud, use_container_width=True)

    with ud2:
        st.markdown("<div class='card-title'>Consistency score (0–1, higher = more predictable output)</div>",
                    unsafe_allow_html=True)
        cs_data = filt_e.sort_values("consistency_score")
        fig_cs = go.Figure(go.Bar(
            x=[short_name(m) for m in cs_data["model"]],
            y=cs_data["consistency_score"],
            marker=dict(
                color=cs_data["consistency_score"],
                colorscale=[[0, "#0f2027"], [1, GREEN]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in cs_data["consistency_score"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12),
        ))
        apply_layout(fig_cs, height=220, showlegend=False,
                     margin=dict(l=8, r=8, t=8, b=30))
        fig_cs.update_yaxes(range=[0, 1.15], title_text="score", title_font=dict(size=10))
        st.plotly_chart(fig_cs, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CATEGORY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("<div class='section-hdr'>Performance by Category</div>",
            unsafe_allow_html=True)

if "category" in filt_r.columns:
    merged_cat = filt.merge(
        filt_r[["prompt_id", "model", "category"]].drop_duplicates(),
        on=["prompt_id", "model"], how="left"
    )
    cat_data = merged_cat.groupby(["model", "category"])["composite_score"].mean().reset_index()

    fig_cat = px.bar(
        cat_data,
        x="category", y="composite_score", color="model",
        barmode="group",
        color_discrete_map={m: mcolor(m) for m in selected_models},
        labels={"composite_score": "Composite Score", "category": "", "model": ""},
    )
    apply_layout(fig_cat, height=300, margin=dict(l=8, r=8, t=8, b=8))
    fig_cat.update_yaxes(range=[0, 5.4])
    fig_cat.update_xaxes(tickangle=0)
    fig_cat.update_layout(
        legend=dict(orientation="h", y=1.08, x=0),
        bargap=0.25, bargroupgap=0.1,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Hallucination focus
    hall_data = merged_cat[merged_cat["category"] == "hallucination_test"] \
        if "hallucination_test" in merged_cat.get("category", pd.Series()).values else pd.DataFrame()

    if not hall_data.empty and "hallucination_resistance" in hall_data.columns:
        h_means = hall_data.groupby("model")["hallucination_resistance"].mean().reset_index()
        hcols = st.columns([2, 1])
        with hcols[0]:
            st.markdown("<div class='card-title'>Hallucination resistance — trap prompts only</div>",
                        unsafe_allow_html=True)
            fig_hall = go.Figure(go.Bar(
                x=[short_name(m) for m in h_means["model"]],
                y=h_means["hallucination_resistance"],
                marker=dict(
                    color=[mcolor(m) for m in h_means["model"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.2f}/5" for v in h_means["hallucination_resistance"]],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=13),
            ))
            apply_layout(fig_hall, height=220, showlegend=False,
                         margin=dict(l=8, r=8, t=8, b=8))
            fig_hall.update_yaxes(range=[0, 5.8])
            st.plotly_chart(fig_hall, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PER-PROMPT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
st.markdown("<div class='section-hdr'>Per-Prompt Drill-Down</div>", unsafe_allow_html=True)

prompt_ids = filt["prompt_id"].unique().tolist()
prompts_lookup = {p["id"]: p for p in prompts}

# Sort control
sort_col, filter_col, _ = st.columns([1.5, 1.5, 3])
with sort_col:
    sort_by = st.selectbox("Sort by", ["composite_score", "value_index",
                                        "hallucination_resistance", "task_completion",
                                        "conciseness"], label_visibility="visible")
with filter_col:
    cats_avail = (
        list(set(filt_r["category"].dropna().tolist()))
        if not filt_r.empty and "category" in filt_r.columns
        else []
    )
    cat_filter = st.multiselect("Category", cats_avail, default=cats_avail,
                                label_visibility="visible")

# Merge category back
if not filt_r.empty and "category" in filt_r.columns:
    filt_with_cat = filt.merge(
        filt_r[["prompt_id", "model", "category"]].drop_duplicates(),
        on=["prompt_id", "model"], how="left"
    )
    if cat_filter:
        filt_with_cat = filt_with_cat[filt_with_cat["category"].isin(cat_filter)]
else:
    filt_with_cat = filt.copy()

# Build pivot-style rows per prompt_id
for pid in filt_with_cat["prompt_id"].unique():
    prompt_meta = prompts_lookup.get(pid, {})
    cat = prompt_meta.get("category", "—")
    difficulty = prompt_meta.get("difficulty", "—")

    rows_for_pid = filt_with_cat[filt_with_cat["prompt_id"] == pid].sort_values(
        sort_by, ascending=False
    )

    # Build model score chips
    chips_html = ""
    for _, row in rows_for_pid.iterrows():
        mc   = mcolor(row["model"])
        comp = row.get("composite_score", 0)
        chips_html += f"""
<span style="display:inline-block;background:{mc}18;border:1px solid {mc}44;
             border-radius:6px;padding:3px 8px;margin:2px;font-size:0.7rem;
             font-family:'IBM Plex Mono',monospace;">
  <span style="color:{mc};font-weight:600;">{row['model'].split('-')[0]}</span>
  <span style="color:{TEXT_MUTE};margin:0 3px;">·</span>
  <span style="color:{score_color(comp)};font-weight:600;">{comp:.2f}</span>
</span>"""

    with st.expander(
        f"**{pid}** · `{cat}` · `{difficulty}` {chips_html}",
        expanded=False,
    ):
        # Prompt + ground truth
        if prompt_meta:
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"<div class='card-title'>Prompt</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.85rem;line-height:1.6;'>{prompt_meta.get('prompt','')}</div>",
                            unsafe_allow_html=True)
            with p2:
                st.markdown(f"<div class='card-title'>Ground truth</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.85rem;line-height:1.6;color:{TEXT_MUTE};'>{prompt_meta.get('ground_truth','')}</div>",
                            unsafe_allow_html=True)
            st.markdown("<hr class='hdivider' style='margin:0.75rem 0'>", unsafe_allow_html=True)

        # Side-by-side model comparison
        model_cols = st.columns(len(selected_models))
        for ci, model in enumerate(selected_models):
            mrow = rows_for_pid[rows_for_pid["model"] == model]
            mresp = filt_r[(filt_r["prompt_id"] == pid) & (filt_r["model"] == model)] \
                    if not filt_r.empty else pd.DataFrame()
            mc = mcolor(model)

            with model_cols[ci]:
                st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;color:{mc};"
                            f"font-size:0.75rem;font-weight:600;margin-bottom:6px;'>"
                            f"{short_name(model)}</div>", unsafe_allow_html=True)

                if not mrow.empty:
                    r = mrow.iloc[0]
                    comp = r.get("composite_score", 0)
                    st.markdown(
                        f"<div style='font-size:1.8rem;font-weight:700;"
                        f"font-family:IBM Plex Mono,monospace;color:{score_color(comp)};'>"
                        f"{comp:.2f}<span style='font-size:0.9rem;color:{TEXT_MUTE};'>/5</span></div>",
                        unsafe_allow_html=True,
                    )
                    # Dim pills
                    pills = ""
                    for dim in avail_dims:
                        val = r.get(dim, 0)
                        pc  = "pill-green" if val >= 4 else "pill-amber" if val >= 3 else "pill-red"
                        pills += f'<span class="pill {pc}">{dim.replace("_"," ")}: {val}</span>'
                    st.markdown(pills, unsafe_allow_html=True)

                    # Timing
                    if not mresp.empty:
                        rr = mresp.iloc[0]
                        lat  = rr.get("total_latency_ms", 0)
                        ftl  = rr.get("first_token_latency_ms", 0)
                        tps  = rr.get("tokens_per_second", 0)
                        out  = rr.get("output_tokens", 0)
                        vr   = rr.get("verbosity_ratio", 0)
                        ref  = rr.get("refused", False)
                        refused_html = '<br><span style="color:#ef4444;">⚠ refused</span>' if ref else ""
                        st.markdown(
                            f"<div style='font-size:0.7rem;color:{TEXT_MUTE};"
                            f"font-family:IBM Plex Mono,monospace;margin-top:6px;line-height:1.8;'>"
                            f"⏱ {lat:.0f}ms · first-tok {ftl:.0f}ms · "
                            f"{tps:.0f} tok/s<br>{out:.0f} output-tokens · {vr:.1f}× verbose"
                            f"{refused_html}</div>",
                            unsafe_allow_html=True,
                        )

                    # Verdict + rationale
                    verdict = r.get("one_line_verdict", "")
                    if verdict:
                        st.markdown(
                            f"<div style='margin-top:8px;padding:7px 10px;"
                            f"background:rgba(255,255,255,0.03);border-left:2px solid {mc};"
                            f"border-radius:4px;font-size:0.78rem;line-height:1.5;'>"
                            f"{verdict}</div>",
                            unsafe_allow_html=True,
                        )

                    # Model response
                    resp_text = mresp.iloc[0].get("response_text", "") if not mresp.empty else ""
                    if resp_text:
                        with st.expander("Response text"):
                            st.markdown(
                                f"<div style='font-size:0.82rem;line-height:1.6;"
                                f"color:{TEXT_MUTE};'>{resp_text}</div>",
                                unsafe_allow_html=True,
                            )


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)
# ── Eval history log ─────────────────────────────────────────────────────────
if st.session_state.get("eval_log"):
    with st.expander(f"📋 Session eval log ({len(st.session_state.eval_log)} runs)", expanded=False):
        log_df = pd.DataFrame(st.session_state.eval_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            color:{TEXT_MUTE};font-size:0.68rem;font-family:'IBM Plex Mono',monospace;
            padding-bottom:1rem;">
  <span>LLM Eval Framework · Abhishek · AI Quality Engineering Portfolio</span>
  <span>Add models live above · python evaluate.py for batch runs</span>
</div>
""", unsafe_allow_html=True)
