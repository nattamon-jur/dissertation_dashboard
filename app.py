"""
AI Adoption Returns Explorer
==========================================================
Dissertation: Predicting Revenue Growth and Cost Reduction from Business AI Adoption
 
Run: streamlit run dashboard.py
Requires: shared_pipeline.py run (FULL run, sample_frac=None) for both
          targets, so trained XGBoost models exist in
          pipeline_outputs/models/, and DATA_PATH points at the source CSV.
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import re
import json
 
MODELS_DIR = "models"
DATA_PATH = "dashboard_data.csv"
TARGET_1 = "revenue_growth_percent"
TARGET_2 = "cost_reduction_percent"
TARGET_1_LABEL = "Predicted revenue growth"
TARGET_2_LABEL = "Predicted cost reduction"
TARGET_1_SHORT = "revenue growth"
TARGET_2_SHORT = "cost reduction"
 
TOOL_NAME = "AI Adoption Returns Explorer"
TOOL_DESCRIPTION = (
    "See how a company's AI adoption choices connect to revenue growth and cost reduction, and what drives each prediction."
)
 
N_MAIN_CONTROLS = 5
 
# ═══════════════════════════════════════════════════════════════════════════
# FONT SCALE — one place to control every text size, with a clear
# hierarchy: big bold numbers for what matters most, small muted labels
# for supporting context.
# ═══════════════════════════════════════════════════════════════════════════
# Sizes tuned for a ~1440px desktop viewport. The previous scale was set
# for a much smaller test window and rendered oversized on a real monitor,
# pushing the driver panels below the fold. CSS media queries further down
# step these down again on narrower screens.
F_TOOL_NAME = 34           # top-of-page tool name
F_TOOL_DESC = 16           # top-of-page one-line description
F_PANEL_TITLE = 20         # "Company profile", "What's driving..." titles
F_PANEL_SUBTITLE = 14      # "Results update as you change these"
F_METRIC_LABEL = 16        # "Predicted revenue growth" / "Predicted cost reduction"
F_METRIC_VALUE = 38        # the big % number — still the loudest element on the page
F_METRIC_CAPTION = 13      # "Typical range X-Y%"
F_INPUT_LABEL = 14         # slider/dropdown labels
F_CHART_LABEL = 14         # driver bar feature names
F_CHART_VALUE = 14         # driver bar value labels
F_INSIGHT = 15             # key insight callout text
F_LEGEND = 13              # color legend text
F_FOOTER = 12              # disclaimer footer
F_BUTTON = 14              # Reset / collapse / preset controls
F_SECTION_LABEL = 11       # small uppercase group labels, e.g. "KEY DETAILS"
MAX_PAGE_WIDTH = 1440      # px — stops the layout stretching on wide monitors
 
# ── Colors (light theme) ─────────────────────────────────────────────────
BG = "#FFFFFF"
CARD_BG = "#F5F5F3"
CARD_BORDER = "#E5E5E2"
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#5F5E5A"
TEXT_MUTED = "#888780"
UP = "#378ADD"        # "increases prediction" — accent blue
DOWN = "#D85A30"      # "decreases prediction" — coral
# NOTE: colour here encodes DIRECTION of effect, not which outcome. This
# deliberately differs from the Chapter 4 figures, where blue/orange encode
# revenue growth vs cost reduction. Direction is the more useful encoding
# for a non-technical user reading a single company profile.
ACCENT = "#378ADD"
ACCENT_DARK = "#0C447C"
INSIGHT_BG = "#E6F1FB"
INSIGHT_TEXT = "#0C447C"
LEGEND_BG = "#F5F5F3"
 
# ── Base variable metadata ───────────────────────────────────────────────
BASE_CATEGORICALS = {
    "industry": ["Technology", "Finance", "Retail", "Manufacturing", "Healthcare",
                 "Consulting", "Education", "Logistics", "Agriculture"],
    "company_size": ["Startup", "SME", "Enterprise"],
    "region": ["North America", "Europe", "Asia", "Africa", "South America", "Oceania"],
    "ai_adoption_stage": ["none", "pilot", "partial", "full"],
    "ai_ethics_committee": ["Yes", "No"],
    "data_privacy_level": ["Low", "Medium", "High"],
}
BASE_NUMERICS = {
    # var: (is_int, is_pct) — value RANGES (min/max) are no longer hardcoded
    # here. They are computed from the actual dataset at runtime, since a
    # hardcoded guess (e.g. "num_employees maxes out around 5000") can be
    # wrong for a real dataset and cause a crash when a real data point
    # (or a default/preset derived from real data) falls outside it.
    "ai_adoption_rate": (False, True),
    "ai_budget_percentage": (False, True),
    "ai_investment_per_employee": (True, False),
    "ai_training_hours": (True, False),
    "years_using_ai": (False, False),
    "ai_projects_active": (True, False),
    "task_automation_rate": (True, True),
    "productivity_change_percent": (False, True),
    "ai_risk_management_score": (False, False),
    "regulatory_compliance_score": (False, False),
    "annual_revenue_usd_millions": (True, False),
    "num_employees": (True, False),
}
ALL_BASE_VARS = list(BASE_CATEGORICALS.keys()) + list(BASE_NUMERICS.keys())
 
# ── Example presets — every base variable covered ───────────────────────
PRESETS = {
    "Tech SME": dict(
        industry="Technology", company_size="SME", region="Europe",
        ai_adoption_rate=65, ai_adoption_stage="full",
        ai_budget_percentage=8.5, ai_investment_per_employee=1500, ai_training_hours=15,
        years_using_ai=3.0, ai_projects_active=8, task_automation_rate=45,
        productivity_change_percent=12.0, ai_risk_management_score=6.0,
        regulatory_compliance_score=7.0, ai_ethics_committee="Yes", data_privacy_level="High",
        annual_revenue_usd_millions=120, num_employees=800,
    ),
    "Manufacturing Enterprise": dict(
        industry="Manufacturing", company_size="Enterprise", region="Asia",
        ai_adoption_rate=40, ai_adoption_stage="partial",
        ai_budget_percentage=4.0, ai_investment_per_employee=800, ai_training_hours=8,
        years_using_ai=1.5, ai_projects_active=3, task_automation_rate=70,
        productivity_change_percent=6.0, ai_risk_management_score=7.5,
        regulatory_compliance_score=8.0, ai_ethics_committee="Yes", data_privacy_level="Medium",
        annual_revenue_usd_millions=350, num_employees=3500,
    ),
    "Retail Startup": dict(
        industry="Retail", company_size="Startup", region="North America",
        ai_adoption_rate=20, ai_adoption_stage="pilot",
        ai_budget_percentage=2.0, ai_investment_per_employee=300, ai_training_hours=2,
        years_using_ai=0.5, ai_projects_active=1, task_automation_rate=15,
        productivity_change_percent=2.0, ai_risk_management_score=2.0,
        regulatory_compliance_score=3.0, ai_ethics_committee="No", data_privacy_level="Low",
        annual_revenue_usd_millions=5, num_employees=25,
    ),
}
PRESET_OPTIONS = ["Dataset averages"] + list(PRESETS.keys())
DEFAULT_PRESET = "Dataset averages"
 
st.set_page_config(page_title=TOOL_NAME, layout="wide")
 
 
# ═══════════════════════════════════════════════════════════════════════════
# NAME HELPERS
# ═══════════════════════════════════════════════════════════════════════════
 
def get_base_variable(raw_name):
    name = re.sub(r"^(num__|cat__)", "", raw_name)
    for base in BASE_CATEGORICALS:
        if name.startswith(base + "_"):
            return base
    return name
 
 
def humanize_base_variable(base_var):
    OVERRIDES = {
        "num_employees": "Number of employees",
        "annual_revenue_usd_millions": "Annual revenue (USD millions)",
        "ai_investment_per_employee": "AI investment per employee (USD)",
        "ai_budget_percentage": "AI budget",
        "task_automation_rate": "Tasks automated",
        "productivity_change_percent": "Productivity change",
        "ai_adoption_rate": "AI adoption rate",
        "ai_training_hours": "AI training hours",
        "years_using_ai": "Years using AI",
        "ai_projects_active": "Active AI projects",
        "ai_risk_management_score": "AI risk management",
        "regulatory_compliance_score": "Regulatory compliance",
        "ai_adoption_stage": "AI adoption stage",
        "ai_ethics_committee": "AI ethics committee",
        "data_privacy_level": "Data privacy level",
        "company_size": "Company size",
        "industry": "Industry",
        "region": "Region",
    }
    return OVERRIDES.get(base_var, base_var.replace("_", " ").capitalize())
 
 
def humanize_feature_name(raw_name):
    base = get_base_variable(raw_name)
    base_label = humanize_base_variable(base)
    name = re.sub(r"^(num__|cat__)", "", raw_name)
    if raw_name.startswith("cat__") and name.startswith(base + "_"):
        value = name[len(base) + 1:]
        ACRONYMS = {"sme": "SME", "no": "No", "yes": "Yes"}
        value = ACRONYMS.get(value.lower(), value.capitalize())
        return f"{base_label}: {value}"
    return base_label
 
 
# ═══════════════════════════════════════════════════════════════════════════
# CACHED COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════
 
@st.cache_resource
def load_model(target_column):
    path = f"{MODELS_DIR}/{target_column}_xgboost.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
@st.cache_data
def load_dataset(data_path):
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)
 
 
@st.cache_data
def compute_numeric_ranges(_df):
    """Computes the ACTUAL min/max for every numeric variable from the real
    dataset, rather than relying on a hardcoded guess. This guarantees that
    any real data point — including the median used as a default, or a
    preset value — can never fall outside the slider's own bounds, which
    was the root cause of a prior crash (a hardcoded 'num_employees maxes
    around 5000' assumption didn't hold for this dataset's real values)."""
    ranges = {}
    for var, (is_int, is_pct) in BASE_NUMERICS.items():
        if var in _df.columns:
            lo, hi = _df[var].min(), _df[var].max()
            if is_int:
                ranges[var] = (int(np.floor(lo)), int(np.ceil(hi)))
            else:
                ranges[var] = (float(lo), float(hi))
    return ranges
 
 
@st.cache_data
def compute_defaults(_df, _ranges):
    defaults = {}
    for var, (is_int, is_pct) in BASE_NUMERICS.items():
        if var in _df.columns:
            med = _df[var].median()
            lo, hi = _ranges[var]
            med = min(max(med, lo), hi)  # safety clamp, should already hold
            defaults[var] = int(round(med)) if is_int else float(med)
    for var in BASE_CATEGORICALS:
        if var in _df.columns:
            defaults[var] = _df[var].mode()[0]
    return defaults
 
 
@st.cache_data
def compute_global_importance(_model, _df, target_name, sample_size=500):
    # `target_name` is included specifically so this cache entry is distinct
    # for revenue vs cost — `_model`/`_df` are underscore-prefixed and are
    # NOT hashed by st.cache_data, so without a differentiating hashed
    # argument, calling this for both models would incorrectly return the
    # SAME cached result for both (confirmed bug, now fixed).
    feature_cols = [c for c in _df.columns if c not in [TARGET_1, TARGET_2]]
    sample = _df[feature_cols].sample(min(sample_size, len(_df)), random_state=42)
    preprocessor = _model.named_steps["preprocessor"]
    xgb_model = _model.named_steps["model"]
    Xt = preprocessor.transform(sample)
    feature_names = list(preprocessor.get_feature_names_out())
    Xt_df = pd.DataFrame(Xt, columns=feature_names)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(Xt_df)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    agg = {}
    for fname, val in zip(feature_names, mean_abs):
        base = get_base_variable(fname)
        agg[base] = agg.get(base, 0.0) + float(val)
    return agg
 
 
@st.cache_data
def compute_main_controls(_imp_rev, _imp_cost, top_n=N_MAIN_CONTROLS):
    def normalize(d):
        max_v = max(d.values()) if d else 1.0
        return {k: v / max_v for k, v in d.items()}
    n_rev, n_cost = normalize(_imp_rev), normalize(_imp_cost)
    combined = {}
    for k in set(list(n_rev.keys()) + list(n_cost.keys())):
        combined[k] = n_rev.get(k, 0) + n_cost.get(k, 0)
    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return [k for k, _ in ranked[:top_n]]
 
 
@st.cache_data
def load_typical_ranges(path="residual_ranges.json"):
    """Residual-based prediction intervals (16th-84th percentile), precomputed
    on the full 150,000-observation dataset. Loaded rather than recomputed so
    the deployed app does not need the full CSV in memory."""
    with open(path) as f:
        return json.load(f)


TYPICAL_RANGES = load_typical_ranges()
 
 
model_revenue = load_model(TARGET_1)
model_cost = load_model(TARGET_2)
 
if model_revenue is None or model_cost is None:
    st.error(
        f"Trained models not found in `{MODELS_DIR}/`. Run `shared_pipeline.py` "
        f"with `sample_frac=None` for both targets first."
    )
    st.stop()
 
df = load_dataset(DATA_PATH)
if df is None:
    st.error(f"Dataset not found at `{DATA_PATH}`.")
    st.stop()
 
numeric_ranges = compute_numeric_ranges(df)
defaults = compute_defaults(df, numeric_ranges)
imp_rev = compute_global_importance(model_revenue, df, TARGET_1)
imp_cost = compute_global_importance(model_cost, df, TARGET_2)
main_controls = compute_main_controls(imp_rev, imp_cost)
advanced_vars = [v for v in ALL_BASE_VARS if v in defaults and v not in main_controls]
 
range_rev = tuple(TYPICAL_RANGES[TARGET_1])
range_cost = tuple(TYPICAL_RANGES[TARGET_2])
 
 
# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
 
if "filters_expanded" not in st.session_state:
    st.session_state.filters_expanded = True
if "preset" not in st.session_state:
    st.session_state.preset = DEFAULT_PRESET
 
 
def field_prefix(var):
    """Which session_state prefix a variable currently uses — depends on
    whether SHAP importance placed it on the main panel or in Advanced.
    (main_controls/advanced_vars are stable within a session since they only
    depend on the trained model + dataset, not on current input values.)"""
    return "main" if var in main_controls else "adv"
 
 
def apply_preset(preset_name):
    """Sets session_state values for every field. Must only be called BEFORE
    the relevant widgets are instantiated in the current script run (i.e.
    from a button handler followed immediately by st.rerun(), never after
    the widgets have already rendered in this same run). Numeric preset
    values are clamped into the dataset's real min/max, since a hand-picked
    example value could in principle fall outside a real dataset's actual
    range (e.g. a preset written for a small test dataset, applied to a
    real dataset with a narrower or wider spread)."""
    st.session_state.preset = preset_name
    values = defaults if preset_name == "Dataset averages" else PRESETS[preset_name]
    for var, val in values.items():
        if var in numeric_ranges:
            lo, hi = numeric_ranges[var]
            val = min(max(val, lo), hi)
        st.session_state[f"{field_prefix(var)}_{var}"] = val
 
 
def reset_advanced_to_defaults():
    """Resets ONLY the Advanced-panel fields to dataset defaults, leaving
    the main 5 controls (and whichever preset is selected) untouched."""
    for var in advanced_vars:
        if var in defaults:
            st.session_state[f"adv_{var}"] = defaults[var]
 
 
def initialize_widget_state():
    """Pre-populates session_state for every widget BEFORE any widget is
    created. This is the required Streamlit pattern for key-bound widgets:
    once a widget with a given key has rendered in a run, session_state[key]
    can no longer be reassigned from code — only user interaction or a
    st.rerun()-triggered fresh run can change it. Widgets are then created
    with ONLY `key=` (no `value=`/`index=`), so they read their initial
    value from this pre-populated session_state."""
    for var in ALL_BASE_VARS:
        if var not in defaults:
            continue
        key = f"{field_prefix(var)}_{var}"
        if key not in st.session_state:
            st.session_state[key] = defaults[var]
 
 
def clamp_stale_session_state():
    """Defensive safeguard, runs on EVERY rerun (not just first load): if a
    session_state value for a slider/dropdown is out of range or invalid —
    e.g. left over from an earlier app version, or from a previous dataset
    with a different value range — it is corrected BEFORE any widget is
    created, rather than letting Streamlit raise a crash when the widget
    renders. Ranges come from the ACTUAL dataset (numeric_ranges), and the
    fallback value is clamped too, so the fallback itself can never be the
    out-of-range value that caused the problem in the first place."""
    for var in BASE_NUMERICS:
        if var not in numeric_ranges:
            continue
        lo, hi = numeric_ranges[var]
        fallback = min(max(defaults.get(var, (lo + hi) / 2), lo), hi)
        for prefix in ("main", "adv"):
            key = f"{prefix}_{var}"
            if key in st.session_state:
                current = st.session_state[key]
                if not isinstance(current, (int, float)) or current < lo or current > hi:
                    st.session_state[key] = fallback
    for var, options in BASE_CATEGORICALS.items():
        for prefix in ("main", "adv"):
            key = f"{prefix}_{var}"
            if key in st.session_state and st.session_state[key] not in options:
                st.session_state[key] = defaults.get(var, options[0])
 
 
initialize_widget_state()
clamp_stale_session_state()
 
 
# ═══════════════════════════════════════════════════════════════════════════
# LIGHT THEME CSS
# ═══════════════════════════════════════════════════════════════════════════
 
st.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2/dist/tabler-icons.min.css">
<style>
    /* Hide Streamlit's Deploy toolbar and collapse its header. It is a
       development affordance rather than part of the artefact, and it
       consumes roughly 45px at the top of every screenshot. Comment this
       block out while developing if you need the hamburger menu (Rerun,
       Clear cache), then re-enable before screenshotting or deploying. */
    div[data-testid="stToolbar"] {{ display: none !important; }}
    header[data-testid="stHeader"] {{
        height: 0 !important;
        background: transparent !important;
    }}
 
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stMain"] {{ background-color: {BG}; }}
    div[data-testid="stMarkdownContainer"] p {{ color: {TEXT_PRIMARY}; font-size:{F_INPUT_LABEL}px; }}
    .stSlider label p, .stSelectbox label p {{ color: {TEXT_SECONDARY} !important; font-size:{F_INPUT_LABEL}px !important; }}
    /* Cap the content width. Without this the layout stretches edge-to-edge
       on a wide monitor, making the driver bars absurdly long and pushing
       the two outcome cards far apart. */
    .block-container {{
        max-width: {MAX_PAGE_WIDTH}px !important;
        padding-top: 1.5rem;   /* toolbar is hidden above, so no clearance needed */
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    div[data-testid="stExpander"] summary {{
        background-color: {CARD_BG} !important;
        border-radius: 8px !important;
        border: 1px solid {CARD_BORDER} !important;
        padding: 8px 12px !important;
    }}
    div[data-testid="stExpander"] summary:hover {{
        background-color: {INSIGHT_BG} !important;
        border-color: {ACCENT} !important;
    }}
    div[data-testid="stExpander"] summary p {{
        font-size:{F_INPUT_LABEL}px !important;
        color:{ACCENT} !important;
        font-weight:600 !important;
    }}
    div[data-testid="stExpander"] summary:hover p {{
        color: {ACCENT_DARK} !important;
    }}
    /* Arrow inherits its fill from the SVG rather than the parent text
       colour, so it has to be set separately from the label above. */
    div[data-testid="stExpander"] summary svg {{
        fill: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    /* The hover rule turns the summary background accent-blue, so the arrow
       must go white with the label or it vanishes against it. */
    div[data-testid="stExpander"] summary:hover svg {{
        fill: {ACCENT_DARK} !important;
        color: {ACCENT_DARK} !important;
    }}
    div[data-baseweb="slider"] div[role="slider"] {{ background-color: {ACCENT} !important; border-color: {ACCENT} !important; }}
 
    /* Force buttons and dropdowns to light theme explicitly — Streamlit's
       native components otherwise follow the browser/OS dark-mode
       preference independently of the custom colors above, which is what
       caused unreadable black buttons and low-contrast dropdown text. */
    .stButton button, button[kind="secondary"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
    .stButton button:hover, button[kind="secondary"]:hover {{
        background-color: {ACCENT} !important;
        color: #FFFFFF !important;
        border-color: {ACCENT} !important;
    }}
    .stButton button p, button[kind="secondary"] p {{
        color: inherit !important;
        font-size: {F_BUTTON}px !important;
    }}
    div[data-baseweb="select"] {{ background-color: {CARD_BG} !important; }}
    div[data-baseweb="select"] > div {{
        background-color: {CARD_BG} !important;
        color: {TEXT_PRIMARY} !important;
        border-color: {CARD_BORDER} !important;
        font-size: {F_INPUT_LABEL}px !important;
    }}
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {{
        color: {TEXT_PRIMARY} !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] {{ background-color: {BG} !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li {{ color: {TEXT_PRIMARY} !important; font-size: {F_INPUT_LABEL}px !important; }}
    div[data-baseweb="slider"] div[data-testid="stThumbValue"] {{ font-size: {F_INPUT_LABEL}px !important; color: {ACCENT} !important; }}
 
    /* Compress the input panel. The five main sliders each stack a label,
       a thumb value and a track; at default spacing they run taller than the
       whole results column, pushing Advanced Filters below the fold.
       Negative margins are needed because Streamlit's inter-element gap sits
       outside the element padding, so padding alone cannot reach it. */
    div[data-testid="stSlider"] {{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: -10px !important;
        margin-bottom: -10px !important;
    }}
    div[data-testid="stSlider"] label {{ margin-bottom: 2px !important; }}
    div[data-baseweb="slider"] {{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
    }}
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {{
        font-size: {F_INPUT_LABEL - 2}px !important;
        padding-bottom: 0 !important;
        top: 2px !important;
    }}
    div[data-testid="stSelectbox"] {{ margin-bottom: 2px !important; }}
    div[data-testid="stExpander"] {{ margin-top: 10px !important; }}
 
    /* Smooth hover feedback on the outcome + driver cards, so the page
       feels responsive without any animation that would distract. */
    .card-surface {{ transition: border-color 120ms ease, box-shadow 120ms ease; }}
    .card-surface:hover {{ border-color: #D8D8D4 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
 
    /* Responsive steps. Streamlit stacks its columns on narrow viewports
       by itself; these rules stop the type scale from staying desktop-sized
       when it does. */
    @media (max-width: 1200px) {{
        .tool-name {{ font-size: {int(F_TOOL_NAME * 0.85)}px !important; }}
        .metric-value {{ font-size: {int(F_METRIC_VALUE * 0.85)}px !important; }}
        .block-container {{ padding-left: 1.25rem; padding-right: 1.25rem; }}
    }}
    @media (max-width: 820px) {{
        .tool-name {{ font-size: {int(F_TOOL_NAME * 0.7)}px !important; }}
        .metric-value {{ font-size: {int(F_METRIC_VALUE * 0.72)}px !important; }}
        .tool-desc {{ font-size: {F_TOOL_DESC - 2}px !important; }}
        .block-container {{ padding-left: 0.9rem; padding-right: 0.9rem; }}
    }}
    div[data-testid="stTooltipContent"] {{
        background-color: {TEXT_PRIMARY} !important;
        color: #FFFFFF !important;
        font-size: {F_FOOTER}px !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
    }}
</style>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# HEADER — tool name + description (accent bar + divider treatment)
# ═══════════════════════════════════════════════════════════════════════════
 
st.markdown(
    f"<div style='width:36px; height:3px; background:{ACCENT}; border-radius:2px; margin-bottom:10px;'></div>"
    f"<div class='tool-name' style='font-size:{F_TOOL_NAME}px; font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:4px; line-height:1.15;'>{TOOL_NAME}</div>"
    # nowrap removed: it forced horizontal scrolling on any viewport
    # narrower than the sentence itself.
    f"<div class='tool-desc' style='font-size:{F_TOOL_DESC}px; color:{TEXT_SECONDARY}; margin-bottom:14px; max-width:110ch;'>{TOOL_DESCRIPTION}</div>"
    f"<div style='border-bottom:1px solid {CARD_BORDER}; margin-bottom:20px;'></div>",
    unsafe_allow_html=True,
)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
 
if st.session_state.filters_expanded:
    col_left, col_right = st.columns([1, 2.4], gap="medium")
else:
    col_left, col_right = st.columns([0.2, 3.2], gap="medium")
 
current_values = {}
 
with col_left:
    header_c1, header_c2, header_c3 = st.columns([2.4, 1, 1])
    with header_c1:
        if st.session_state.filters_expanded:
            # Subtitle sits directly under the heading rather than inside the
            # bordered panel, so it aligns with the Reset row instead of
            # pushing the first input down.
            st.markdown(
                f"<div style='font-size:{F_PANEL_TITLE}px; font-weight:600; color:{TEXT_PRIMARY}; padding-top:2px; line-height:1.2;'>Company profile</div>"
                f"<div style='font-size:{F_PANEL_SUBTITLE}px; color:{TEXT_MUTED}; margin-top:2px;'>Results update as change </div>",
                unsafe_allow_html=True)
    with header_c2:
        if st.session_state.filters_expanded:
            if st.button("Reset", key="reset_btn", width="stretch"):
                apply_preset(st.session_state.preset)
                st.rerun()
    with header_c3:
        # Label rather than a tooltip: a hover-only affordance is invisible
        # until discovered, which works against the design principle that
        # the tool should be usable without prior interaction (Section 3.8).
        label = "◀ Hide" if st.session_state.filters_expanded else "▶ Show"
        if st.button(label, key="toggle_filters", width="stretch"):
            st.session_state.filters_expanded = not st.session_state.filters_expanded
            st.rerun()
 
    # No fixed height: the panel grows to fit its content naturally, so the
    # "Advanced" section is always reachable via normal page scrolling. A
    # fixed-height internal-scroll container was tried first, but wasn't a
    # discoverable/reliable way to reach it in practice.
    with st.container(border=True):
 
        if st.session_state.filters_expanded:

            preset_choice = st.selectbox(
                "Preset", PRESET_OPTIONS,
                index=PRESET_OPTIONS.index(st.session_state.preset),
                key="preset_selector",
            )
            if preset_choice != st.session_state.preset:
                apply_preset(preset_choice)
                st.rerun()
 
            for var in main_controls:
                label = humanize_base_variable(var)
                if var in BASE_CATEGORICALS:
                    options = BASE_CATEGORICALS[var]
                    current_values[var] = st.selectbox(label, options, key=f"main_{var}")
                else:
                    is_int, is_pct = BASE_NUMERICS[var]
                    lo, hi = numeric_ranges[var]
                    step = 1 if is_int else 0.1
                    slider_label = f"{label} (%)" if is_pct else label
                    med = defaults.get(var)
                    hint = f"Dataset median: {med:,.0f}" if is_int else f"Dataset median: {med:,.1f}"
                    val = st.slider(slider_label, lo, hi, step=step,
                                    key=f"main_{var}", help=hint)
                    current_values[var] = val
 
            with st.expander(f"Advanced Filters"):
 
                for var in advanced_vars:
                    label = humanize_base_variable(var)
                    if var in BASE_CATEGORICALS:
                        options = BASE_CATEGORICALS[var]
                        current_values[var] = st.selectbox(label, options, key=f"adv_{var}")
                    else:
                        is_int, is_pct = BASE_NUMERICS[var]
                        lo, hi = numeric_ranges[var]
                        step = 1 if is_int else 0.1
                        slider_label = f"{label} (%)" if is_pct else label
                        med = defaults.get(var)
                        hint = f"Dataset median: {med:,.0f}" if is_int else f"Dataset median: {med:,.1f}"
                        current_values[var] = st.slider(slider_label, lo, hi, step=step,
                                                        key=f"adv_{var}", help=hint)
 
        else:
            for var in ALL_BASE_VARS:
                prefix = field_prefix(var)
                current_values[var] = st.session_state.get(f"{prefix}_{var}", defaults.get(var))
            st.markdown(
                f"<div style='writing-mode:vertical-rl; text-align:center; font-size:{F_PANEL_SUBTITLE}px; "
                f"color:{TEXT_MUTED}; padding:10px 0;'>Company profile</div>",
                unsafe_allow_html=True,
            )
 
for var in ALL_BASE_VARS:
    if var not in current_values:
        current_values[var] = defaults.get(var)
 
input_row = pd.DataFrame([current_values])
 
 
# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION + LOCAL SHAP
# ═══════════════════════════════════════════════════════════════════════════
 
def get_prediction_and_shap(model, input_row, top_n=5):
    pred = model.predict(input_row)[0]
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]
    Xt = preprocessor.transform(input_row)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f_{i}" for i in range(Xt.shape[1])]
    Xt_df = pd.DataFrame(Xt, columns=feature_names)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(Xt_df)[0]
    order = np.argsort(-np.abs(shap_vals))[:top_n]
    labels = [humanize_feature_name(feature_names[i]) for i in order]
    return pred, shap_vals[order], labels
 
 
def render_driver_bars(values, labels, scale_max=None):
    """Renders each feature's SHAP contribution as a labelled progress-style
    bar (feature name + signed value on one row, colored bar below).
 
    `scale_max` is passed in from the caller so BOTH outcome panels are
    drawn against ONE shared maximum. Scaling each panel independently
    made a 0.05 contribution fill the same width as a 0.13 contribution,
    which visually contradicted this study's central RQ2 finding that the
    two outcomes are driven by factors of genuinely different magnitude.
    Bar length is now comparable across the two panels, not just within one.
 
    Plain HTML/CSS rather than a charting library — renders natively in the
    same visual language as the surrounding cards, with no separate
    rendering engine to visually clash with the page theme.
 
    Returns an HTML string rather than rendering directly, so the caller can
    compose the full card in ONE st.markdown call. Splitting a card across
    several calls does not work: Streamlit sanitises each call separately,
    so an unclosed wrapper <div> is auto-closed on the spot and renders as
    an empty grey box, while the trailing </div> is silently discarded."""
    order = np.argsort(-np.abs(values))  # largest impact first, top to bottom
    values = np.array(values)[order]
    labels = np.array(labels)[order]
    max_abs = scale_max if scale_max else max(np.abs(values).max(), 1e-6)
 
    rows_html = ""
    for label, val in zip(labels, values):
        color = UP if val > 0 else DOWN
        width_pct = max(abs(val) / max_abs * 100, 1.5)  # floor keeps tiny bars visible
        sign = "+" if val > 0 else ""
        rows_html += f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between; font-size:{F_CHART_LABEL}px; margin-bottom:3px;">
            <span style="color:{TEXT_PRIMARY};">{label}</span>
            <span style="color:{color}; font-weight:500;">{sign}{val:.2f}</span>
          </div>
          <div style="height:6px; background:{CARD_BORDER}; border-radius:3px;">
            <div style="height:100%; width:{width_pct:.1f}%; background:{color}; border-radius:3px;"></div>
          </div>
        </div>
        """
    return rows_html
 
 
pred_rev, shap_rev_vals, shap_rev_labels = get_prediction_and_shap(model_revenue, input_row)
pred_cost, shap_cost_vals, shap_cost_labels = get_prediction_and_shap(model_cost, input_row)
 
# One scale across both panels — see render_driver_bars docstring.
shared_scale = max(np.abs(shap_rev_vals).max(), np.abs(shap_cost_vals).max(), 1e-6)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ═══════════════════════════════════════════════════════════════════════════
 
with col_right:
 
    m1, m2 = st.columns(2, gap="medium")
    for col, label, pred, rng, icon in [
        (m1, TARGET_1_LABEL, pred_rev, range_rev, "ti-trending-up"),
        (m2, TARGET_2_LABEL, pred_cost, range_cost, "ti-coin"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="card-surface" style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px; padding:18px 20px;">
                  <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                    <i class="ti {icon}" style="font-size:18px; color:{ACCENT};"></i>
                    <span style="font-size:{F_METRIC_LABEL}px; font-weight:700; color:{TEXT_PRIMARY};">{label}</span>
                  </div>
                  <div class="metric-value" style="font-size:{F_METRIC_VALUE}px; font-weight:600; color:{TEXT_PRIMARY}; line-height:1.1; margin-bottom:4px;">
                    {'+' if pred >= 0 else ''}{pred:.1f}%
                  </div>
                  <div style="font-size:{F_METRIC_CAPTION}px; color:{TEXT_MUTED}; line-height:1.35;">
                    Typically {pred + rng[0]:.1f}–{pred + rng[1]:.1f}%
                    <span style="opacity:0.8;">&middot; covers about two-thirds of similar companies</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
 
    def driver_card(title, values, labels, scale_max):
        """Builds one complete driver panel as a single HTML string.
 
        The subtitle matters: without it a user sees a negative top driver
        sitting above a positive headline prediction and reads the two as
        contradicting each other."""
        return f"""
        <div class="card-surface" style="background:{CARD_BG}; border:1px solid {CARD_BORDER};
                    border-radius:14px; padding:18px 20px 10px 20px;">
          <div style="font-size:{F_PANEL_TITLE}px; font-weight:600; color:{TEXT_PRIMARY}; margin-bottom:2px;">{title}</div>
          <div style="font-size:{F_METRIC_CAPTION}px; color:{TEXT_MUTED}; margin-bottom:12px;">How this profile differs from the dataset average</div>
          {render_driver_bars(values, labels, scale_max)}
        </div>
        """
 
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.markdown(driver_card(f"What's driving {TARGET_1_SHORT}",
                                shap_rev_vals, shap_rev_labels, shared_scale),
                    unsafe_allow_html=True)
    with d2:
        st.markdown(driver_card(f"What's driving {TARGET_2_SHORT}",
                                shap_cost_vals, shap_cost_labels, shared_scale),
                    unsafe_allow_html=True)
 
    # ── Color legend ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex; gap:24px; margin-top:10px; padding:9px 18px;
                    background:{LEGEND_BG}; border-radius:10px;">
          <div style="display:flex; align-items:center; gap:8px; font-size:{F_LEGEND}px; color:{TEXT_SECONDARY};">
            <div style="width:12px; height:12px; border-radius:3px; background:{UP};"></div>
            Increases prediction
          </div>
          <div style="display:flex; align-items:center; gap:8px; font-size:{F_LEGEND}px; color:{TEXT_SECONDARY};">
            <div style="width:12px; height:12px; border-radius:3px; background:{DOWN};"></div>
            Decreases prediction
          </div>
          <div style="font-size:{F_LEGEND}px; color:{TEXT_MUTED}; margin-left:auto;">
            Bar length is comparable across both outcomes
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # ── Auto-generated insight ───────────────────────────────────────────
    idx_rev = int(np.argmax(np.abs(shap_rev_vals)))
    idx_cost = int(np.argmax(np.abs(shap_cost_vals)))
    top_rev_label, top_rev_val = shap_rev_labels[idx_rev], shap_rev_vals[idx_rev]
    top_cost_label, top_cost_val = shap_cost_labels[idx_cost], shap_cost_vals[idx_cost]
 
    rev_label_map = dict(zip(shap_rev_labels, shap_rev_vals))
    cost_impact_on_rev = rev_label_map.get(top_cost_label, 0.0)
 
    if top_rev_label == top_cost_label:
        insight = f"<strong>{top_rev_label}</strong> is the dominant driver behind both outcomes for this company profile."
    elif abs(top_cost_val) > abs(cost_impact_on_rev) * 2:
        insight = (f"<strong>{top_cost_label}</strong> drives your {TARGET_2_SHORT} far more than your "
                   f"{TARGET_1_SHORT} — the two outcomes have different levers.")
    else:
        insight = (f"<strong>{top_rev_label}</strong> drives {TARGET_1_SHORT} most, while "
                   f"<strong>{top_cost_label}</strong> drives {TARGET_2_SHORT} most — different factors matter for each outcome.")
 
    st.markdown(
        f"""
        <div style="background:{INSIGHT_BG}; border-radius:12px; padding:12px 20px; margin-top:10px;
                    display:flex; align-items:flex-start; gap:10px;">
          <span style="font-size:{F_INSIGHT+2}px; color:{INSIGHT_TEXT};">ⓘ</span>
          <span style="font-size:{F_INSIGHT}px; color:{INSIGHT_TEXT}; line-height:1.5;">{insight}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
# ── Persistent disclaimer ────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="margin-top:14px; display:flex; align-items:flex-start; gap:8px;">
          <span style="font-size:{F_FOOTER+1}px; color:{TEXT_MUTED}; line-height:1.5;">&#9432;</span>
          <div style="font-size:{F_FOOTER}px; color:{TEXT_MUTED}; line-height:1.5;">
            Proof-of-concept prototype built on a synthetic dataset. Predictions illustrate how the
            underlying model responds to different inputs — they are not empirical claims about
            real-world AI returns, and this tool has not been tested with real users. Not investment
            or business advice.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
