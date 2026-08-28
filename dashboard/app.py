"""
RecoverIQ -- AI Revenue Recovery Decision Engine Dashboard
Phase 7: Streamlit Interactive Dashboard

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path regardless of execution directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.simulator import simulate_outcome
from dashboard.agent_examples import AGENT_EXAMPLES
from engine.decision_engine import decide_best_action
from guardrails.config import DEFAULT_GUARDRAIL_CONFIG
from guardrails.policy_engine import check_guardrails
from models.recovery_model import FEATURE_COLUMNS, load_recovery_model

DATA_PATH = ROOT_DIR / "data" / "sample_data" / "payments.csv"
CACHE_PATH = ROOT_DIR / "evaluation" / "dashboard_cache.json"

# Unified brand and semantic color palette (Stripe / AI Fintech aesthetic)
BRAND_PRIMARY = "#4f46e5"      # Indigo - Main brand accent
BRAND_PRIMARY_DARK = "#3730a3" # Deep Indigo
COLOR_SUCCESS = "#059669"      # Emerald - Recovery / Gains
COLOR_WARNING = "#d97706"      # Amber - Guardrails / Alerts
COLOR_DANGER = "#dc2626"       # Rose - Hard Decline / Failures
COLOR_INFO = "#0284c7"         # Sky Blue - Info
COLOR_NEUTRAL = "#64748b"      # Slate - Muted / Secondary

INTERVENTION_COLORS = {
    "retry": "#059669",         # Emerald
    "payment_link": "#2563eb",  # Blue
    "notification": "#7c3aed",  # Purple
    "escalate": "#d97706",      # Amber
    "stop": "#64748b",          # Slate
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


# -----------------------------------------------------------------------------
# Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_cached_evaluation(path: str = str(CACHE_PATH)) -> dict:
    """Load precomputed multi-split and 4-strategy progression metrics."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_payments_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load raw failed payments dataset for live aggregations & browsing."""
    df = pd.read_csv(path)
    return df


@st.cache_data
def compute_guardrail_stats(df: pd.DataFrame) -> dict:
    """
    Run lightweight check_guardrails once across the dataset to count safety overrides.
    Cached for near-instant retrieval on re-runs.
    """
    violations_count = 0
    rule_breakdown: dict[str, int] = {}

    for row in df.itertuples(index=False):
        # Check default proposed decision "retry"
        res = check_guardrails(row, proposed_decision="retry", retry_attempts_so_far=0)
        if not res["allowed"]:
            violations_count += 1
            for rule_str in res["violated_rules"]:
                rule_name = rule_str.split(":")[0].strip()
                rule_breakdown[rule_name] = rule_breakdown.get(rule_name, 0) + 1

    return {
        "total_violations": violations_count,
        "violation_pct": (violations_count / len(df)) * 100.0 if len(df) > 0 else 0.0,
        "rule_breakdown": rule_breakdown,
    }


@st.cache_data
def compute_sample_probabilities(df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """
    Compute ML recovery probabilities across a sample of rows for distribution plotting.
    Cached so model inference runs only once.
    """
    sample_df = df.iloc[:sample_size].copy()
    model = load_recovery_model()
    probs = model.predict_proba(sample_df[FEATURE_COLUMNS])[:, 1]
    sample_df["recovery_probability"] = probs
    return sample_df


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="RecoverIQ — AI Revenue Recovery Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for polished, unified fintech aesthetic
    st.markdown(
        f"""
        <style>
        /* Base typography & page styling */
        html, body, [class*="css"] {{
            font-family: {FONT_FAMILY};
        }}

        /* Section Headings */
        .section-header {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-color, inherit);
            margin-bottom: 2px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section-caption {{
            font-size: 0.85rem;
            color: var(--text-color, #94a3b8);
            opacity: 0.8;
            margin-bottom: 16px;
        }}

        /* Polished KPI Cards with Accent Top Border & Soft Shadow */
        .kpi-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 14px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease-in-out;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-card:hover {{
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        }}
        .kpi-border-indigo {{ border-top: 3px solid #4f46e5; }}
        .kpi-border-emerald {{ border-top: 3px solid #059669; }}
        .kpi-border-amber {{ border-top: 3px solid #d97706; }}
        .kpi-border-blue {{ border-top: 3px solid #2563eb; }}
        .kpi-border-slate {{ border-top: 3px solid #64748b; }}

        .kpi-label {{
            font-size: 0.72rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.7rem;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin-bottom: 4px;
        }}
        .kpi-subtext {{
            font-size: 0.78rem;
            font-weight: 500;
            color: #64748b;
        }}
        .kpi-subtext-success {{
            color: #059669;
            font-weight: 600;
        }}
        .kpi-subtext-warning {{
            color: #d97706;
            font-weight: 600;
        }}

        /* Subtle Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 9px;
            font-size: 0.78rem;
            font-weight: 600;
            border-radius: 9999px;
            letter-spacing: 0.02em;
        }}
        .badge-retry {{ background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
        .badge-payment_link {{ background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
        .badge-notification {{ background-color: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe; }}
        .badge-escalate {{ background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
        .badge-stop {{ background-color: #f8fafc; color: #475569; border: 1px solid #cbd5e1; }}
        .badge-guardrail-pass {{ background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
        .badge-guardrail-fail {{ background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}

        /* Transaction Container Pair */
        .flow-card {{
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
            height: 100%;
        }}
        .flow-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #f1f5f9;
        }}
        .flow-card-title {{
            font-size: 0.78rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Sidebar Styling */
        .sidebar-section-title {{
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-color, inherit);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 14px;
            margin-bottom: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load Data & Cache
    cache = load_cached_evaluation()
    df = load_payments_dataset()
    guardrail_stats = compute_guardrail_stats(df)

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
                <span style="font-size: 1.5rem;">⚡</span>
                <span style="font-size: 1.35rem; font-weight: 700; color: var(--text-color, inherit);">RecoverIQ</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-color, #94a3b8); opacity: 0.8; margin-bottom: 16px;">
                AI Revenue Recovery Decision Engine
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">About RecoverIQ</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.84rem; color: #334155; line-height: 1.5;">
                RecoverIQ dynamically evaluates Expected Value across 5 recovery actions (retry, payment link, notification, escalation, stop) while enforcing strict safety guardrails. Rather than an uncalibrated retry bot, it maximizes net recovery.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Guardrail Policies</div>', unsafe_allow_html=True)
        with st.expander("🛡️ Active Safety Rules", expanded=False):
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #334155; line-height: 1.6;">
                1. <b>High Amount</b>: > ₹{DEFAULT_GUARDRAIL_CONFIG['high_amount_threshold']:,} &rarr; <code>{DEFAULT_GUARDRAIL_CONFIG['high_amount_action']}</code><br>
                2. <b>Repeated Decline</b>: <code>hard_decline</code> &ge; {DEFAULT_GUARDRAIL_CONFIG['hard_decline_repeat_failures']} failures &rarr; <code>{DEFAULT_GUARDRAIL_CONFIG['hard_decline_repeat_action']}</code><br>
                3. <b>Max Retries</b>: &ge; {DEFAULT_GUARDRAIL_CONFIG['max_retry_attempts']} retries &rarr; <code>{DEFAULT_GUARDRAIL_CONFIG['max_retry_action']}</code><br>
                4. <b>Max Incentive</b>: Capped at {DEFAULT_GUARDRAIL_CONFIG['max_incentive_pct'] * 100:.0f}%<br>
                5. <b>Invalid Action</b>: Fallback &rarr; <code>{DEFAULT_GUARDRAIL_CONFIG['invalid_action_fallback']}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">System Information</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #475569; line-height: 1.7;">
                <div>📁 <b>Dataset:</b> {len(df):,} failed events</div>
                <div>🧠 <b>ML Model:</b> Logistic Regression (AUC 0.893)</div>
                <div>⚡ <b>Decisioning:</b> EV Argmax + Safety Layer</div>
                <div>🏷️ <b>Version:</b> Phase 7 Production Build</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Header Title
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <h1 style="font-size: 1.85rem; font-weight: 700; color: var(--text-color, inherit); margin: 0; letter-spacing: -0.02em;">
                    RecoverIQ Revenue Recovery Engine
                </h1>
                <span class="badge badge-retry" style="font-size: 0.72rem;">ACTIVE ENGINE</span>
            </div>
            <p style="font-size: 0.92rem; color: var(--text-color, #94a3b8); opacity: 0.85; margin-top: 4px; margin-bottom: 0;">
                Autonomous payment recovery decisioning, expected value optimization, and experimentation benchmarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. TOP KPI ROW (Polished Cards)
    # -------------------------------------------------------------------------
    rev_at_risk = float(df["amount"].sum())
    rev_iq = cache["progression"]["RecoverIQ"]
    rev_recovered = float(rev_iq["total_revenue_recovered"])
    recovery_rate = float(rev_iq["recovery_rate_pct"])
    incremental_rev = float(cache["multi_split_avg_incremental_revenue"])
    incremental_rate_pp = float(cache["multi_split_avg_incremental_rate_pp"])
    total_interventions = int(rev_iq["n"])
    guardrail_violations = guardrail_stats["total_violations"]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-slate">
                <div class="kpi-label">Revenue At Risk</div>
                <div class="kpi-value">₹{rev_at_risk:,.0f}</div>
                <div class="kpi-subtext">15,000 Failed Events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-indigo">
                <div class="kpi-label">Revenue Recovered</div>
                <div class="kpi-value" style="color: #4f46e5;">₹{rev_recovered:,.0f}</div>
                <div class="kpi-subtext">{recovery_rate:.2f}% of Exposure</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-emerald">
                <div class="kpi-label">Incremental Revenue</div>
                <div class="kpi-value" style="color: #059669;">+₹{incremental_rev:,.0f}</div>
                <div class="kpi-subtext kpi-subtext-success">+{incremental_rate_pp:.2f} pp vs Rule-Based</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-blue">
                <div class="kpi-label">Recovery Rate</div>
                <div class="kpi-value">{recovery_rate:.2f}%</div>
                <div class="kpi-subtext">{rev_iq['n_recovered']:,} Successful Events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col5:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-slate">
                <div class="kpi-label">Total Interventions</div>
                <div class="kpi-value">{total_interventions:,}</div>
                <div class="kpi-subtext">5 Action Modalities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col6:
        st.markdown(
            f"""
            <div class="kpi-card kpi-border-amber">
                <div class="kpi-label">Guardrail Overrides</div>
                <div class="kpi-value" style="color: #d97706;">{guardrail_violations:,}</div>
                <div class="kpi-subtext kpi-subtext-warning">{guardrail_stats['violation_pct']:.1f}% Forced Safe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. CHARTS & ANALYTICS SECTION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header">
            <span>📊</span> Performance Analytics & Recovery Dynamics
        </div>
        <div class="section-caption">
            Evaluation progression across decision policies and failure exposure distributions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_distribution = st.tabs(["Strategy Comparison & Decision Mix", "Exposure & Recovery Probabilities"])

    with tab_overview:
        col_strat, col_mix = st.columns([1.15, 0.85])

        with col_strat:
            # 4-Strategy progression comparison
            # Visual hierarchy: Muted grays for baselines, saturated brand indigo for RecoverIQ
            progression = cache["progression"]
            strat_names = {
                "naive_strategy": "Naive (Always Retry)",
                "rule_based_strategy": "Rule-Based",
                "ml_strategy": "ML / EV Unconstrained",
                "RecoverIQ": "RecoverIQ (ML + Guardrails)",
            }
            strat_df = pd.DataFrame([
                {
                    "Strategy": strat_names.get(k, k),
                    "Revenue Recovered (₹)": v["total_revenue_recovered"],
                    "Recovery Rate (%)": v["recovery_rate_pct"],
                    "Recovered Events": v["n_recovered"],
                    "BarColor": (
                        "#4f46e5" if k == "RecoverIQ"          # Brand Indigo (Emphasized)
                        else "#818cf8" if k == "ml_strategy"   # Soft Indigo/Purple
                        else "#94a3b8" if k == "rule_based_strategy" # Mid Slate Gray
                        else "#cbd5e1"                         # Light Slate Gray (Muted Baseline)
                    ),
                }
                for k, v in progression.items()
            ])

            fig_strat = go.Figure()
            fig_strat.add_trace(
                go.Bar(
                    x=strat_df["Strategy"],
                    y=strat_df["Revenue Recovered (₹)"],
                    marker=dict(
                        color=strat_df["BarColor"],
                        line=dict(color="#0f172a", width=0.5),
                    ),
                    text=[
                        f"₹{val:,.0f} ({rate:.1f}%)"
                        for val, rate in zip(strat_df["Revenue Recovered (₹)"], strat_df["Recovery Rate (%)"])
                    ],
                    textposition="auto",
                    textfont=dict(family=FONT_FAMILY, size=11),
                    hovertemplate="<b>%{x}</b><br>Revenue Recovered: ₹%{y:,.2f}<extra></extra>",
                )
            )
            fig_strat.update_layout(
                title=dict(
                    text="<b>Strategy Progression: Total Revenue Recovered</b>",
                    font=dict(family=FONT_FAMILY, size=14),
                ),
                yaxis=dict(
                    title=dict(text="Total Recovered (₹)", font=dict(family=FONT_FAMILY, size=12)),
                    gridcolor="rgba(150, 150, 150, 0.15)",
                    zeroline=False,
                ),
                xaxis=dict(
                    title="",
                    tickfont=dict(family=FONT_FAMILY, size=11),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=370,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_strat, width="stretch")

        with col_mix:
            # Decision mix donut chart with cohesive semantic colors
            mix_data = rev_iq["decision_mix"]
            mix_df = pd.DataFrame([
                {"Intervention": k.replace("_", " ").title(), "Count": v, "Raw": k}
                for k, v in mix_data.items()
            ])

            fig_mix = px.pie(
                mix_df,
                names="Intervention",
                values="Count",
                title="<b>RecoverIQ Intervention Allocation Mix</b>",
                color="Raw",
                color_discrete_map=INTERVENTION_COLORS,
                hole=0.5,
            )
            fig_mix.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(family=FONT_FAMILY, size=11),
                hovertemplate="<b>%{label}</b><br>Events: %{value:,} (%{percent})<extra></extra>",
                marker=dict(line=dict(color="#ffffff", width=2)),
            )
            fig_mix.update_layout(
                title=dict(
                    text="<b>RecoverIQ Intervention Allocation Mix</b>",
                    font=dict(family=FONT_FAMILY, size=14),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=370,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(family=FONT_FAMILY, size=11),
                ),
            )
            st.plotly_chart(fig_mix, width="stretch")

    with tab_distribution:
        col_fail, col_prob = st.columns([1, 1])

        with col_fail:
            # Revenue at risk by failure reason (monochromatic clean indigo/slate)
            fail_df = (
                df.groupby("failure_reason")
                .agg(Total_Amount=("amount", "sum"), Count=("amount", "count"))
                .reset_index()
                .sort_values(by="Total_Amount", ascending=True)
            )
            fail_df["Reason_Clean"] = fail_df["failure_reason"].str.replace("_", " ").str.title()

            fig_fail = px.bar(
                fail_df,
                x="Total_Amount",
                y="Reason_Clean",
                orientation="h",
                title="<b>Revenue At Risk by Failure Reason</b>",
                labels={"Total_Amount": "Total Exposure (₹)", "Reason_Clean": "Failure Reason"},
                color_discrete_sequence=["#4f46e5"],
                text=fail_df["Total_Amount"].apply(lambda x: f"₹{x:,.0f}"),
            )
            fig_fail.update_layout(
                title=dict(
                    text="<b>Revenue At Risk by Failure Reason</b>",
                    font=dict(family=FONT_FAMILY, size=14),
                ),
                xaxis=dict(
                    title=dict(text="Total Exposure (₹)", font=dict(family=FONT_FAMILY, size=12)),
                    gridcolor="rgba(150, 150, 150, 0.15)",
                ),
                yaxis=dict(title="", tickfont=dict(family=FONT_FAMILY, size=11)),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=370,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            fig_fail.update_traces(
                textposition="outside",
                textfont=dict(family=FONT_FAMILY, size=11),
            )
            st.plotly_chart(fig_fail, width="stretch")

        with col_prob:
            # Recovery probability distribution across sample
            sample_prob_df = compute_sample_probabilities(df, sample_size=500)
            fig_hist = px.histogram(
                sample_prob_df,
                x="recovery_probability",
                nbins=25,
                title="<b>ML Recovery Probability Distribution (500-Sample)</b>",
                labels={"recovery_probability": "Predicted Recovery Probability"},
                color_discrete_sequence=["#4f46e5"],
                opacity=0.85,
            )
            fig_hist.update_layout(
                title=dict(
                    text="<b>ML Recovery Probability Distribution (500-Sample)</b>",
                    font=dict(family=FONT_FAMILY, size=14),
                ),
                yaxis=dict(
                    title=dict(text="Payment Count", font=dict(family=FONT_FAMILY, size=12)),
                    gridcolor="rgba(150, 150, 150, 0.15)",
                ),
                xaxis=dict(
                    title=dict(text="P(Recovery | Features, Retry)", font=dict(family=FONT_FAMILY, size=12)),
                    gridcolor="rgba(150, 150, 150, 0.15)",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=370,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_hist, width="stretch")

    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. INTERACTIVE TRANSACTION VIEW (Polished Input -> Output Pair)
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header">
            <span>🔍</span> Interactive Transaction Decision Inspector
        </div>
        <div class="section-caption">
            Inspect live on-demand decisioning: evaluates Expected Value across all 5 candidate actions and applies safety guardrails.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "payment_select" not in st.session_state:
        st.session_state["payment_select"] = "PAY-0000016"

    # Preset Quick Select Buttons for demo showcase
    st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: var(--text-color, inherit); opacity: 0.85; margin-bottom: 6px;'>PRESET DEMO TRANSACTIONS:</div>", unsafe_allow_html=True)
    preset_cols = st.columns(5)

    def _set_payment(pid: str) -> None:
        st.session_state["payment_select"] = pid

    with preset_cols[0]:
        st.button("PAY-0000016\n(High Value Retry)", on_click=_set_payment, args=("PAY-0000016",), width="stretch")
    with preset_cols[1]:
        st.button("PAY-0000001\n(Transient Bank Failure)", on_click=_set_payment, args=("PAY-0000001",), width="stretch")
    with preset_cols[2]:
        st.button("PAY-0000003\n(Insufficient Funds)", on_click=_set_payment, args=("PAY-0000003",), width="stretch")
    with preset_cols[3]:
        st.button("PAY-0000005\n(Guardrail Override: Stop)", on_click=_set_payment, args=("PAY-0000005",), width="stretch")
    with preset_cols[4]:
        st.button("PAY-0000013\n(Card Expired Link)", on_click=_set_payment, args=("PAY-0000013",), width="stretch")

    # Search / Select box
    all_payment_ids = df["payment_id"].tolist()

    col_select, col_empty = st.columns([1, 1])
    with col_select:
        selected_id = st.selectbox(
            "Select or Search Payment ID:",
            options=all_payment_ids,
            key="payment_select",
        )

    # Get single row
    selected_row = df[df["payment_id"] == selected_id].iloc[0]

    # Run on-demand decision engine for just this single row
    decision_result = decide_best_action(selected_row)
    decision = decision_result["decision"]
    ev_val = decision_result["expected_value"]
    reason = decision_result["reason"]
    all_evaluated = decision_result["all_evaluated"]

    # Also simulate outcome deterministically for this row & decision
    sim_outcome = simulate_outcome(selected_row, decision)

    # Check guardrail status
    is_overridden = "overridden by guardrails" in reason

    # Render Two-Column Transaction Detail (Input -> Output Flow)
    col_details, col_engine = st.columns([1, 1.25])

    with col_details:
        st.markdown(
            f"""
            <div class="flow-card">
                <div class="flow-card-header">
                    <span class="flow-card-title">① Transaction Input</span>
                    <span class="badge badge-{decision}">{selected_row['payment_method'].upper()}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">{selected_row['payment_id']}</span>
                    <span style="font-size: 1.7rem; font-weight: 700; color: #0f172a;">₹{float(selected_row['amount']):,.2f}</span>
                </div>
                <div style="border-top: 1px solid #f1f5f9; padding-top: 10px;">
                    <table style="width: 100%; font-size: 0.86rem; color: #334155; line-height: 1.8;">
                        <tr><td style="color: #64748b;">Customer ID:</td><td><b>{selected_row['customer_id']}</b></td></tr>
                        <tr><td style="color: #64748b;">Failure Reason:</td><td><span style="color: #dc2626; font-weight: 600;">{selected_row['failure_reason']}</span></td></tr>
                        <tr><td style="color: #64748b;">Customer LTV:</td><td>₹{float(selected_row['customer_value']):,.2f}</td></tr>
                        <tr><td style="color: #64748b;">Customer Age:</td><td>{selected_row['customer_age']} yrs</td></tr>
                        <tr><td style="color: #64748b;">Historical Successes:</td><td><span style="color: #059669; font-weight: 600;">{selected_row['previous_successes']}</span></td></tr>
                        <tr><td style="color: #64748b;">Historical Failures:</td><td><span style="color: #dc2626; font-weight: 600;">{selected_row['previous_failures']}</span></td></tr>
                        <tr><td style="color: #64748b;">Subscription Tenure:</td><td>{selected_row['subscription_age']} months</td></tr>
                        <tr><td style="color: #64748b;">Days Since Last Attempt:</td><td>{selected_row['days_since_last_payment']} days</td></tr>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_engine:
        if is_overridden:
            guardrail_badge = '<span class="badge badge-guardrail-fail">🛡️ Overridden by Guardrails</span>'
        else:
            guardrail_badge = '<span class="badge badge-guardrail-pass">✓ Guardrails Approved</span>'

        st.markdown(
            f"""
            <div class="flow-card" style="border-left: 4px solid {INTERVENTION_COLORS.get(decision, '#4f46e5')};">
                <div class="flow-card-header">
                    <span class="flow-card-title">② Autonomous Decision Output</span>
                    {guardrail_badge}
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span class="badge badge-{decision}" style="font-size: 1.1rem; padding: 5px 14px;">{decision.replace('_', ' ').upper()}</span>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">EV: ₹{ev_val:,.2f}</span>
                </div>
                <div style="font-size: 0.86rem; color: #334155; background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px 12px; border-radius: 6px; margin-bottom: 12px; line-height: 1.5;">
                    <b style="color: #475569;">Decision Rationale:</b><br>{reason}
                </div>
                <div style="font-size: 0.84rem; color: #475569; display: flex; justify-content: space-between; align-items: center; padding-top: 4px;">
                    <span>Environment Realization:</span>
                    {'<span style="color: #059669; font-weight: 600;">✓ RECOVERED (₹' + f"{sim_outcome['recovered_amount']:,.2f}" + f" in {sim_outcome['recovery_time_hours']:.1f}h)</span>" if sim_outcome['payment_recovered'] else '<span style="color: #dc2626; font-weight: 600;">✗ NOT RECOVERED</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full All-Evaluated Interventions Matrix
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: var(--text-color, inherit); margin-bottom: 8px;'>Intervention Expected Value Matrix (All Candidate Modalities)</div>", unsafe_allow_html=True)

    matrix_rows = []
    for ev_item in all_evaluated:
        act = ev_item["intervention"]
        p_val = ev_item["probability_used"]
        cost_val = ev_item["cost"]
        ev_calc = ev_item["expected_value"]
        is_chosen = (act == decision)

        matrix_rows.append({
            "Status": "👉 CHOSEN" if is_chosen else "—",
            "Intervention": act.replace("_", " ").title(),
            "Recovery Probability": f"{p_val * 100:.2f}%",
            "Execution Cost": f"₹{cost_val:,.2f}",
            "Expected Value": f"₹{ev_calc:,.2f}",
            "Formula Calculation": f"({p_val:.3f} × ₹{float(selected_row['amount']):,.2f}) - ₹{cost_val:.0f}",
        })

    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(
        matrix_df,
        width="stretch",
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # 4. AI AGENT REASONING EXAMPLES
    # -------------------------------------------------------------------------
    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <span>🤖</span> AI Agent Reasoning Examples
        </div>
        <div class="section-caption">
            These are real captured LLM tool-calling outputs demonstrating the agent's reasoning process, shown statically here due to free-tier API rate limits.
        </div>
        """,
        unsafe_allow_html=True,
    )

    agent_cols = st.columns(len(AGENT_EXAMPLES))
    for col, eg in zip(agent_cols, AGENT_EXAMPLES):
        with col:
            p_id = eg["payment_id"]
            dec = eg["decision"]
            ev = eg["expected_value"]
            reason_text = eg["reason"]
            match = eg.get("phase3_match", False)

            match_badge = (
                '<span class="badge badge-guardrail-pass" style="font-size: 0.72rem; padding: 2px 7px;">'
                '✓ Matches Phase 3 Decision</span>'
                if match else ''
            )

            st.markdown(
                f"""
                <div class="flow-card" style="border-top: 3px solid {INTERVENTION_COLORS.get(dec, '#4f46e5')};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{p_id}</span>
                        {match_badge}
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span class="badge badge-{dec}" style="font-size: 0.88rem; padding: 4px 10px;">{dec.replace('_', ' ').upper()}</span>
                        <span style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">EV: ₹{ev:,.2f}</span>
                    </div>
                    <div style="font-size: 0.86rem; color: #334155; line-height: 1.6; background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px;">
                        <b style="color: #475569;">Agent Explanation:</b><br>{reason_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
