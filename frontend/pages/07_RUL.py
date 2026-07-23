"""
Manufacturing Digital Twin - RUL Forecast Page (Phase 7)
Time-series Remaining Useful Life forecasting per machine.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from auth import require_auth, render_sidebar_user, get_auth_headers

st.set_page_config(
    page_title="RUL Forecast | Manufacturing DT",
    page_icon="⏱️",
    layout="wide",
)

st.title("⏱️ Remaining Useful Life Forecast")
st.markdown(
    "XGBoost regression model predicts hours until equipment failure "
    "from rolling sensor windows — replaces the rule-based days-to-failure estimate."
)


def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")


API_BASE = get_api_base()

require_auth()
selected_plant = render_sidebar_user()

# ============================================================================
# DATA LOADERS
# ============================================================================

@st.cache_data(ttl=60)
def load_rul_summary(api_base: str, plant_id: str):
    try:
        resp = requests.get(
            f"{api_base}/rul-summary",
            params={"plant_id": plant_id},
            timeout=15,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as e:
        st.error(f"Failed to load RUL summary: {e}")
        return None


@st.cache_data(ttl=60)
def load_rul_forecast(api_base: str, machine_id: str, plant_id: str, hours: int):
    try:
        resp = requests.get(
            f"{api_base}/rul-forecast/{machine_id}",
            params={"plant_id": plant_id, "hours": hours},
            timeout=15,
        )
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Failed to load RUL forecast for {machine_id}: {e}")
        return None


# ============================================================================
# SUMMARY CARDS
# ============================================================================

with st.spinner("Loading RUL data..."):
    summary_df = load_rul_summary(API_BASE, selected_plant)

if summary_df is None or summary_df.empty:
    st.warning("No RUL data available. Make sure the backend is running.")
    st.stop()

st.markdown("### 🔋 Current RUL per Machine")

cols = st.columns(5)
RISK_COLORS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "GOOD": "🟢"}

for i, row in summary_df.iterrows():
    icon = RISK_COLORS.get(row["risk_level"], "⚪")
    rul_h = row["rul_hours"]
    days  = rul_h / 24
    delta_txt = f"±{(row['rul_upper'] - row['rul_lower']) / 2:.1f} h band"

    with cols[i % 5]:
        st.metric(
            label=f"{icon} {row['machine_id']}",
            value=f"{rul_h:.1f} h",
            delta=delta_txt,
            delta_color="off",
        )
        st.caption(f"{days:.1f} days · **{row['risk_level']}**")

# Risk legend
st.caption(
    "🔴 CRITICAL < 12 h  |  🟠 HIGH < 24 h  |  🟡 MEDIUM < 48 h  |  🟢 GOOD ≥ 48 h"
)

st.markdown("---")

# ============================================================================
# DEGRADATION CURVE
# ============================================================================

st.markdown("### 📉 RUL Degradation Trend")

machine_options = summary_df["machine_id"].tolist()
selected_machines = st.multiselect(
    "Select machines to compare",
    options=machine_options,
    default=machine_options,
)
lookback_h = st.slider("History window (hours)", min_value=6, max_value=48, value=48, step=6)

if not selected_machines:
    st.info("Select at least one machine above.")
    st.stop()

fig = go.Figure()
palette = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A"]


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


for idx, mid in enumerate(selected_machines):
    fc = load_rul_forecast(API_BASE, mid, selected_plant, lookback_h)
    if fc is None or fc.empty:
        continue

    color = palette[idx % len(palette)]

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["timestamp"], fc["timestamp"].iloc[::-1]]),
        y=pd.concat([fc["rul_upper"], fc["rul_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor=hex_to_rgba(color, 0.12),
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name=f"{mid} band",
    ))

    # RUL line
    fig.add_trace(go.Scatter(
        x=fc["timestamp"],
        y=fc["rul_hours"],
        mode="lines",
        name=mid,
        line=dict(color=color, width=2),
        hovertemplate="%{x|%H:%M}<br>RUL: %{y:.1f} h<extra>" + mid + "</extra>",
    ))

# Threshold lines
fig.add_hline(y=12, line_dash="dash", line_color="red",
              annotation_text="Critical (12 h)", annotation_position="bottom right")
fig.add_hline(y=24, line_dash="dash", line_color="orange",
              annotation_text="High (24 h)", annotation_position="bottom right")
fig.add_hline(y=48, line_dash="dot",  line_color="gold",
              annotation_text="Medium (48 h)", annotation_position="bottom right")

fig.update_layout(
    yaxis_title="Remaining Useful Life (hours)",
    xaxis_title="Time",
    height=420,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PER-MACHINE DETAIL TABLE
# ============================================================================

st.markdown("---")
st.markdown("### 📋 RUL Summary Table")

display = summary_df[["machine_id", "rul_hours", "rul_lower", "rul_upper", "risk_level"]].copy()
display["days_remaining"] = (display["rul_hours"] / 24).round(1)
display.columns = ["Machine", "RUL (h)", "Lower (h)", "Upper (h)", "Risk Level", "Days"]
display = display[["Machine", "RUL (h)", "Days", "Lower (h)", "Upper (h)", "Risk Level"]]

def color_risk(val):
    mapping = {"CRITICAL": "#ffcccc", "HIGH": "#ffe6cc", "MEDIUM": "#ffffcc", "GOOD": "#ccffcc"}
    return f"background-color: {mapping.get(val, '')}"

st.dataframe(
    display.style.map(color_risk, subset=["Risk Level"]),
    use_container_width=True,
    hide_index=True,
)

# ============================================================================
# MODEL INFO
# ============================================================================

st.markdown("---")
with st.expander("ℹ️ About the RUL model"):
    st.markdown("""
**Model:** XGBoost Regressor (`rul_model.joblib`)

**Target variable:** Hours remaining until failure (regression)

**Training data:**
- Fault episodes: latent health decays 1.0 → 0 over 48 readings; RUL target = `48 - step`
- Normal episodes: RUL target capped at 72 h (machine is healthy)
- 5 machines × (40 fault + 20 normal) episodes = 14,400 training rows

**Features:** Same rolling-window features as the XGBoost classifier —
`temperature`, `vibration`, `power_consumption`, 6-reading rolling mean & std, machine encoding.

**Confidence band:** ±20% of predicted RUL (minimum ±1 h).

**How it replaces the placeholder:** Previously `days_to_failure` was a random bucket keyed
only off failure probability. Now it comes from this model's per-reading prediction.
""")

st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>Last updated: {}</div>".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ),
    unsafe_allow_html=True,
)
