"""
Manufacturing Digital Twin - Maintenance Predictions Page
Shows predictive maintenance alerts and risk scores for equipment
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Maintenance Alerts | Manufacturing DT",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Predictive Maintenance Alerts")
st.markdown("AI-driven equipment failure predictions • Optimize maintenance scheduling")

def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")

API_BASE = get_api_base()

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data(ttl=60)
def load_risk_analysis(api_base: str):
    try:
        resp = requests.get(f"{api_base}/risk-analysis", timeout=15)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_sensor_data(api_base: str):
    try:
        resp = requests.get(f"{api_base}/sensor-readings", params={"hours": 24}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {api_base}.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_risk_score(row):
    temp_risk    = (row["temperature"] - 50) / 40 * 25
    vib_risk     = min(row["vibration"] / 5 * 40, 40)
    power_risk   = (row["power_consumption"] - 10) / 20 * 20
    downtime_risk = min(row["downtime_minutes"] / 30 * 15, 15)
    return min(temp_risk + vib_risk + power_risk + downtime_risk, 100)

with st.spinner("Loading maintenance data..."):
    df = load_sensor_data(API_BASE)
    risk_df = load_risk_analysis(API_BASE)

if df is None:
    st.warning("No data available. Make sure the backend is running.")
    st.stop()

# ============================================================================
# DUAL-SIGNAL RISK INTELLIGENCE
# ============================================================================

if risk_df is not None and not risk_df.empty:
    st.markdown("### 🤖 Dual-Signal Risk Intelligence")
    st.caption(
        "**XGBoost** (supervised) detects known failure patterns from labeled training data. "
        "**Isolation Forest** (unsupervised) flags any sensor combination that deviates from normal — "
        "no failure labels needed. Together they catch both known and novel fault modes."
    )

    # Grouped bar chart: XGBoost vs IF per machine
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Bar(
        name="XGBoost Failure Probability",
        x=risk_df["machine_id"],
        y=(risk_df["xgb_failure_probability"] * 100).round(1),
        marker_color="#EF553B",
        text=(risk_df["xgb_failure_probability"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig_dual.add_trace(go.Bar(
        name="Isolation Forest Anomaly Score",
        x=risk_df["machine_id"],
        y=(risk_df["if_anomaly_score"] * 100).round(1),
        marker_color="#636EFA",
        text=(risk_df["if_anomaly_score"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig_dual.update_layout(
        barmode="group",
        yaxis=dict(title="Risk Score (%)", range=[0, 110]),
        xaxis_title="Machine",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=[
            dict(type="line", x0=-0.5, x1=4.5, y0=60, y1=60,
                 line=dict(color="orange", dash="dash", width=1.5)),
        ],
        annotations=[
            dict(x=4.5, y=62, xref="x", yref="y", text="High risk (60%)",
                 showarrow=False, font=dict(color="orange", size=11), xanchor="right"),
        ],
    )
    st.plotly_chart(fig_dual, use_container_width=True)

    # Combined risk summary table
    display_df = risk_df[["machine_id", "xgb_failure_probability", "if_anomaly_score", "combined_risk"]].copy()
    display_df.columns = ["Machine", "XGBoost Prob", "IF Anomaly", "Combined Risk"]
    for col in ["XGBoost Prob", "IF Anomaly", "Combined Risk"]:
        display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("---")

latest_readings = df.sort_values("timestamp").groupby("machine_id").tail(1).copy()
latest_readings["risk_score"] = latest_readings.apply(calculate_risk_score, axis=1)

# ============================================================================
# ALERT SUMMARY (Top Row)
# ============================================================================

st.markdown("### 🚨 Critical & High-Risk Alerts")

col1, col2, col3 = st.columns(3)

critical_count = (latest_readings["risk_score"] >= 80).sum()
high_count     = (latest_readings["risk_score"] >= 60).sum() - critical_count
medium_count   = (latest_readings["risk_score"] >= 40).sum() - high_count - critical_count

with col1:
    st.metric("🔴 CRITICAL", f"{critical_count} machine(s)", "Require immediate attention")
with col2:
    st.metric("🟠 HIGH RISK", f"{high_count} machine(s)", "Schedule maintenance soon")
with col3:
    st.metric("🟡 MEDIUM", f"{medium_count} machine(s)", "Monitor closely")

st.markdown("---")

# ============================================================================
# RISK GAUGE FOR EACH MACHINE
# ============================================================================

st.markdown("### 📊 Machine Risk Assessment")

cols = st.columns(5)

for idx, (_, row) in enumerate(latest_readings.iterrows()):
    risk = row["risk_score"]
    if risk >= 80:
        color, status = "🔴", "CRITICAL"
    elif risk >= 60:
        color, status = "🟠", "HIGH"
    elif risk >= 40:
        color, status = "🟡", "MEDIUM"
    else:
        color, status = "🟢", "GOOD"

    with cols[idx % 5]:
        st.metric(f"{color} {row['machine_id']}", f"{risk:.0f}%", status)
        with st.expander("Details"):
            st.write(f"🌡️ Temp: {row['temperature']:.1f}°C")
            st.write(f"📳 Vibration: {row['vibration']:.2f} mm/s")
            st.write(f"⚡ Power: {row['power_consumption']:.1f} kW")
            st.write(f"⏸️ Downtime: {row['downtime_minutes']:.1f} min")

# ============================================================================
# DETAILED ALERTS TABLE
# ============================================================================

st.markdown("---")
st.markdown("### 📋 Detailed Alert Recommendations")

alerts_data = []
for _, row in latest_readings.iterrows():
    risk = row["risk_score"]
    if risk >= 80:
        action, days = "SCHEDULE IMMEDIATE MAINTENANCE - Replace bearings & check motor", 1
    elif risk >= 60:
        action, days = "Schedule maintenance within 1 week - Monitor vibration closely", 7
    elif risk >= 40:
        action, days = "Monitor equipment - Check lubrication and alignment", 14
    else:
        action, days = "Normal operation - Continue monitoring", 30

    alerts_data.append({
        "Machine": row["machine_id"],
        "Risk Score": f"{risk:.0f}%",
        "Temperature": f"{row['temperature']:.1f}°C",
        "Vibration": f"{row['vibration']:.2f} mm/s",
        "Recommended Action": action,
        "Est. Days to Failure": days
    })

alerts_df = pd.DataFrame(alerts_data)

def highlight_risk(val):
    risk = float(val.rstrip("%"))
    if risk >= 80:   return "background-color: #ffcccc"
    elif risk >= 60: return "background-color: #ffe6cc"
    elif risk >= 40: return "background-color: #ffffcc"
    else:            return "background-color: #ccffcc"

st.dataframe(alerts_df.style.map(highlight_risk, subset=["Risk Score"]),
             use_container_width=True, hide_index=True)

# ============================================================================
# TREND ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("### 📈 Risk Trend Over Time (Last 24 Hours)")

df_recent = df[df["timestamp"] >= datetime.now() - timedelta(hours=24)].copy()
df_recent["risk_score"] = df_recent.apply(calculate_risk_score, axis=1)

fig = px.line(
    df_recent, x="timestamp", y="risk_score", color="machine_id",
    title="Risk Score Evolution",
    labels={"timestamp": "Time", "risk_score": "Risk Score (%)", "machine_id": "Machine"},
    height=400
)
fig.add_hline(y=80, line_dash="dash", line_color="red",    annotation_text="Critical (80%)")
fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="High (60%)")
fig.add_hline(y=40, line_dash="dash", line_color="yellow", annotation_text="Medium (40%)")
fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAINTENANCE COST CALCULATOR
# ============================================================================

st.markdown("---")
st.markdown("### 💰 Maintenance Cost Impact Analysis")

st.info("""
**ROI of Predictive Maintenance:**
- Unplanned downtime costs: **$1,000–$5,000 per hour**
- Planned maintenance cost: **$500–$2,000 (includes labor + parts)**
- Early detection saves: **60–80% of downtime costs**
""")

critical_machines    = (latest_readings["risk_score"] >= 80).sum()
cost_if_unplanned    = critical_machines * 4 * 3000
cost_preventive      = critical_machines * 1500
savings              = cost_if_unplanned - cost_preventive

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💵 Risk if Unplanned Failure", f"${cost_if_unplanned:,.0f}")
with col2:
    st.metric("🔧 Preventive Maintenance Cost", f"${cost_preventive:,.0f}")
with col3:
    roi_pct = (savings / cost_if_unplanned * 100) if cost_if_unplanned > 0 else 0
    st.metric("💚 Potential Savings", f"${savings:,.0f}", delta=f"{roi_pct:.0f}% ROI")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

st.markdown("---")
st.markdown("### 📌 Maintenance Schedule Recommendations")

recommendations = []
for _, row in latest_readings.iterrows():
    risk = row["risk_score"]
    if risk >= 60:
        recommendations.append({
            "Priority": "🔴 HIGH" if risk >= 80 else "🟠 MEDIUM",
            "Machine": row["machine_id"],
            "Issue": f"High vibration ({row['vibration']:.2f} mm/s) + elevated temp ({row['temperature']:.1f}°C)",
            "Action": "Replace bearings, check motor alignment",
            "Estimated Downtime": "2-4 hours",
            "Next Check": "ASAP" if risk >= 80 else "Within 1 week"
        })

if recommendations:
    st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)
else:
    st.success("✅ All machines operating normally. No immediate maintenance required.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>Last updated: {}</div>".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
), unsafe_allow_html=True)
