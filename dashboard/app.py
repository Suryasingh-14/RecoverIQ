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

# Color constants
COLOR_PRIMARY = "#4f46e5"     # Indigo
COLOR_SUCCESS = "#10b981"     # Emerald
COLOR_WARNING = "#f59e0b"     # Amber
COLOR_DANGER = "#ef4444"      # Rose
COLOR_INFO = "#06b6d4"        # Cyan
COLOR_DARK = "#1e293b"        # Slate

INTERVENTION_COLORS = {
    "retry": "#10b981",         # Emerald
    "payment_link": "#3b82f6",  # Blue
    "notification": "#8b5cf6",  # Purple
    "escalate": "#f59e0b",      # Amber
    "stop": "#64748b",          # Slate
}


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

    # Custom CSS for polished styling
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .metric-title {
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            color: #10b981;
            font-weight: 500;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 9999px;
            text-transform: uppercase;
        }
        .badge-retry { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
        .badge-payment_link { background-color: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
        .badge-notification { background-color: #ede9fe; color: #5b21b6; border: 1px solid #ddd6fe; }
        .badge-escalate { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
        .badge-stop { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
        .badge-guardrail-pass { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .badge-guardrail-fail { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
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
        st.title("⚡ RecoverIQ")
        st.caption("AI-Powered Revenue Recovery Decision Engine")

        st.markdown("### About RecoverIQ")
        st.info(
            "RecoverIQ is an AI revenue recovery decision engine that dynamically evaluates "
            "the Expected Value of multiple recovery actions (retry, payment links, customer "
            "notifications, human escalation, or stop) while enforcing strict business guardrails. "
            "Rather than an uncalibrated retry bot, RecoverIQ optimizes net recovered revenue "
            "and long-term customer experience."
        )

        st.divider()

        st.markdown("### Guardrail Policies")
        with st.expander("🛡️ Active Safety Rules", expanded=False):
            st.markdown(
                f"""
                1. **High Amount**: Transactions > ₹{DEFAULT_GUARDRAIL_CONFIG['high_amount_threshold']:,} forced to `{DEFAULT_GUARDRAIL_CONFIG['high_amount_action']}`.
                2. **Repeated Hard Decline**: `hard_decline` with ≥ {DEFAULT_GUARDRAIL_CONFIG['hard_decline_repeat_failures']} failures forced to `{DEFAULT_GUARDRAIL_CONFIG['hard_decline_repeat_action']}`.
                3. **Max Retries**: ≥ {DEFAULT_GUARDRAIL_CONFIG['max_retry_attempts']} retries blocked & forced to `{DEFAULT_GUARDRAIL_CONFIG['max_retry_action']}`.
                4. **Max Incentive**: Discounts capped at {DEFAULT_GUARDRAIL_CONFIG['max_incentive_pct'] * 100:.0f}%.
                5. **Invalid Action**: Non-whitelisted interventions fallback to `{DEFAULT_GUARDRAIL_CONFIG['invalid_action_fallback']}`.
                """
            )

        st.divider()
        st.markdown("### System Metadata")
        st.caption(f"📁 Dataset Size: **{len(df):,} failed payments**")
        st.caption(f"🧠 ML Model: **Logistic Regression (AUC 0.8932)**")
        st.caption(f"⚡ Pipeline: **EV Argmax + Trust & Safety Engine**")
        st.caption("🚀 Version: **Phase 7 Production Build**")

    # -------------------------------------------------------------------------
    # Header Title
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="margin-bottom: 0px; font-size: 2.2rem; font-weight: 800; color: #1e293b;">
                ⚡ RecoverIQ Dashboard
            </h1>
            <p style="font-size: 1.05rem; color: #64748b; margin-top: 4px;">
                Autonomous Payment Recovery Decisioning, Expected Value Optimization & Experimentation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 1. TOP KPI ROW
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
            <div class="metric-card">
                <div class="metric-title">Revenue At Risk</div>
                <div class="metric-value">₹{rev_at_risk:,.0f}</div>
                <div class="metric-subtitle" style="color: #64748b;">15,000 Failed Events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Revenue Recovered</div>
                <div class="metric-value" style="color: #4f46e5;">₹{rev_recovered:,.0f}</div>
                <div class="metric-subtitle">{recovery_rate:.2f}% Recovery Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Incremental Rev</div>
                <div class="metric-value" style="color: #10b981;">+₹{incremental_rev:,.0f}</div>
                <div class="metric-subtitle">+{incremental_rate_pp:.2f} pp vs Rule-Based</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Recovery Rate</div>
                <div class="metric-value">{recovery_rate:.2f}%</div>
                <div class="metric-subtitle" style="color: #4f46e5;">{rev_iq['n_recovered']:,} Recovered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Interventions</div>
                <div class="metric-value">{total_interventions:,}</div>
                <div class="metric-subtitle" style="color: #64748b;">5 Action Modalities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Guardrail Overrides</div>
                <div class="metric-value" style="color: #f59e0b;">{guardrail_violations:,}</div>
                <div class="metric-subtitle" style="color: #d97706;">{guardrail_stats['violation_pct']:.1f}% Forced Safe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. CHARTS & ANALYTICS SECTION
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Performance Analytics & Recovery Dynamics")

    tab_overview, tab_distribution = st.tabs(["Strategy Comparison & Decision Mix", "Exposure & Recovery Probabilities"])

    with tab_overview:
        col_strat, col_mix = st.columns([1.1, 0.9])

        with col_strat:
            # 4-Strategy progression comparison
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
                    "Color": (
                        "#4f46e5" if k == "RecoverIQ"
                        else "#06b6d4" if k == "ml_strategy"
                        else "#94a3b8" if k == "naive_strategy"
                        else "#64748b"
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
                        color=strat_df["Color"],
                        line=dict(color="#1e293b", width=1),
                    ),
                    text=[f"₹{val:,.0f}<br>({rate:.1f}%)" for val, rate in zip(strat_df["Revenue Recovered (₹)"], strat_df["Recovery Rate (%)"])],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>",
                )
            )
            fig_strat.update_layout(
                title="<b>Strategy Progression: Revenue Recovered</b>",
                yaxis_title="Total Recovered (₹)",
                xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_strat, use_container_width=True)

        with col_mix:
            # Decision mix donut chart
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
                color_discrete_map={
                    "retry": INTERVENTION_COLORS["retry"],
                    "payment_link": INTERVENTION_COLORS["payment_link"],
                    "notification": INTERVENTION_COLORS["notification"],
                    "escalate": INTERVENTION_COLORS["escalate"],
                    "stop": INTERVENTION_COLORS["stop"],
                },
                hole=0.45,
            )
            fig_mix.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Events: %{value:,} (%{percent})<extra></extra>",
            )
            fig_mix.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
                showlegend=True,
            )
            st.plotly_chart(fig_mix, use_container_width=True)

    with tab_distribution:
        col_fail, col_prob = st.columns([1, 1])

        with col_fail:
            # Revenue at risk by failure reason
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
                color="Total_Amount",
                color_continuous_scale="Purples",
                text=fail_df["Total_Amount"].apply(lambda x: f"₹{x:,.0f}"),
            )
            fig_fail.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            fig_fail.update_traces(textposition="outside")
            st.plotly_chart(fig_fail, use_container_width=True)

        with col_prob:
            # Recovery probability distribution across sample
            sample_prob_df = compute_sample_probabilities(df, sample_size=500)
            fig_hist = px.histogram(
                sample_prob_df,
                x="recovery_probability",
                nbins=25,
                title="<b>ML Recovery Probability Distribution (500-Payment Sample)</b>",
                labels={"recovery_probability": "P(Recovery | Retry Features)"},
                color_discrete_sequence=["#4f46e5"],
                opacity=0.85,
            )
            fig_hist.update_layout(
                yaxis_title="Payment Count",
                xaxis_title="Predicted Recovery Probability",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. INTERACTIVE TRANSACTION VIEW
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Interactive Transaction Decision Inspector")
    st.caption("Inspect live decisioning on individual payments: evaluates Expected Value across all 5 actions and applies trust & safety guardrails on-demand.")

    if "payment_select" not in st.session_state:
        st.session_state["payment_select"] = "PAY-0000016"

    # Preset Quick Select Buttons for demo showcase
    st.markdown("**Quick Preset Demos:**")
    preset_cols = st.columns(5)

    def _set_payment(pid: str) -> None:
        st.session_state["payment_select"] = pid

    with preset_cols[0]:
        st.button("PAY-0000016\n(High Value Retry)", on_click=_set_payment, args=("PAY-0000016",))
    with preset_cols[1]:
        st.button("PAY-0000001\n(Transient Bank Failure)", on_click=_set_payment, args=("PAY-0000001",))
    with preset_cols[2]:
        st.button("PAY-0000003\n(Insufficient Funds)", on_click=_set_payment, args=("PAY-0000003",))
    with preset_cols[3]:
        st.button("PAY-0000005\n(Guardrail Override: Stop)", on_click=_set_payment, args=("PAY-0000005",))
    with preset_cols[4]:
        st.button("PAY-0000013\n(Card Expired Link)", on_click=_set_payment, args=("PAY-0000013",))

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

    # Render Two-Column Transaction Detail
    col_details, col_engine = st.columns([1, 1.2])

    with col_details:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem; font-weight: 700; color: #0f172a;">{selected_row['payment_id']}</span>
                    <span class="badge badge-{decision}">{selected_row['payment_method'].upper()}</span>
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #1e293b; margin-bottom: 14px;">
                    ₹{float(selected_row['amount']):,.2f}
                </div>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid #f1f5f9;">
                <table style="width: 100%; font-size: 0.92rem; color: #334155; line-height: 1.8;">
                    <tr><td style="color: #64748b;">Customer ID:</td><td><b>{selected_row['customer_id']}</b></td></tr>
                    <tr><td style="color: #64748b;">Failure Reason:</td><td><b style="color: #dc2626;">{selected_row['failure_reason']}</b></td></tr>
                    <tr><td style="color: #64748b;">Customer LTV:</td><td>₹{float(selected_row['customer_value']):,.2f}</td></tr>
                    <tr><td style="color: #64748b;">Customer Age:</td><td>{selected_row['customer_age']} yrs</td></tr>
                    <tr><td style="color: #64748b;">Historical Successes:</td><td><span style="color: #16a34a; font-weight: 600;">{selected_row['previous_successes']}</span></td></tr>
                    <tr><td style="color: #64748b;">Historical Failures:</td><td><span style="color: #dc2626; font-weight: 600;">{selected_row['previous_failures']}</span></td></tr>
                    <tr><td style="color: #64748b;">Subscription Tenure:</td><td>{selected_row['subscription_age']} months</td></tr>
                    <tr><td style="color: #64748b;">Days Since Last Payment:</td><td>{selected_row['days_since_last_payment']} days</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_engine:
        # Decision status banner
        if is_overridden:
            guardrail_badge = '<span class="badge badge-guardrail-fail">🛡️ Guardrail Override</span>'
        else:
            guardrail_badge = '<span class="badge badge-guardrail-pass">✓ Guardrails Passed</span>'

        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 0.9rem; font-weight: 700; color: #64748b; text-transform: uppercase;">RecoverIQ Engine Decision</span>
                    {guardrail_badge}
                </div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <span class="badge badge-{decision}" style="font-size: 1.25rem; padding: 6px 16px;">{decision.replace('_', ' ').upper()}</span>
                    <span style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">EV: ₹{ev_val:,.2f}</span>
                </div>
                <div style="font-size: 0.92rem; color: #475569; background: #f8fafc; border-left: 4px solid {INTERVENTION_COLORS.get(decision, '#4f46e5')}; padding: 8px 12px; border-radius: 4px; margin-bottom: 14px;">
                    <b>Rationale:</b> {reason}
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-bottom: 4px;">
                    <b>Simulated Environment Outcome:</b>
                    {'<span style="color: #16a34a; font-weight: 700;">✓ RECOVERED (₹' + f"{sim_outcome['recovered_amount']:,.2f}" + f" in {sim_outcome['recovery_time_hours']:.1f}h)</span>" if sim_outcome['payment_recovered'] else '<span style="color: #dc2626; font-weight: 700;">✗ NOT RECOVERED</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Full All-Evaluated Interventions Matrix
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📋 Intervention Expected Value Matrix (All Evaluated Options)")

    matrix_rows = []
    for ev_item in all_evaluated:
        act = ev_item["intervention"]
        p_val = ev_item["probability_used"]
        cost_val = ev_item["cost"]
        ev_calc = ev_item["expected_value"]
        is_chosen = (act == decision)

        matrix_rows.append({
            "Selected": "👉 CHOSEN" if is_chosen else "",
            "Intervention": act.replace("_", " ").title(),
            "Recovery Probability": f"{p_val * 100:.2f}%",
            "Execution Cost (₹)": f"₹{cost_val:,.2f}",
            "Expected Value (₹)": f"₹{ev_calc:,.2f}",
            "Formula Calculation": f"({p_val:.3f} × ₹{float(selected_row['amount']):,.2f}) - ₹{cost_val:.0f}",
        })

    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(
        matrix_df,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # 4. AI AGENT REASONING EXAMPLES
    # -------------------------------------------------------------------------
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Agent Reasoning Examples")
    st.caption("These are real captured LLM tool-calling outputs demonstrating the agent's reasoning process, shown statically here due to free-tier API rate limits.")

    agent_cols = st.columns(len(AGENT_EXAMPLES))
    for col, eg in zip(agent_cols, AGENT_EXAMPLES):
        with col:
            p_id = eg["payment_id"]
            dec = eg["decision"]
            ev = eg["expected_value"]
            reason_text = eg["reason"]
            match = eg.get("phase3_match", False)

            match_badge = (
                '<span class="badge badge-guardrail-pass" style="font-size: 0.75rem; padding: 2px 8px;">'
                '✓ Matches Phase 3 Decision</span>'
                if match else ''
            )

            st.markdown(
                f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{p_id}</span>
                        {match_badge}
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span class="badge badge-{dec}" style="font-size: 0.95rem; padding: 4px 12px;">{dec.replace('_', ' ').upper()}</span>
                        <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">EV: ₹{ev:,.2f}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #334155; line-height: 1.6; background-color: #f8fafc; padding: 12px; border-radius: 8px; border-left: 3px solid {INTERVENTION_COLORS.get(dec, '#4f46e5')};">
                        <b>Agent Explanation:</b><br>{reason_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
