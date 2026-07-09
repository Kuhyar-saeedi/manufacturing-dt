"""
Manufacturing Digital Twin - Maintenance Predictions Page
Shows predictive maintenance alerts and risk scores for equipment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Maintenance Alerts | Manufacturing DT",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Predictive Maintenance Alerts")
st.markdown("AI-driven equipment failure predictions • Optimize maintenance scheduling")

# Construct absolute path to database
import os
current_file = os.path.abspath(__file__)  # /frontend/pages/02_Predictions.py
pages_dir = os.path.dirname(current_file)  # /frontend/pages
frontend_dir = os.path.dirname(pages_dir)  # /frontend
project_dir = os.path.dirname(frontend_dir)  # /manufacturing-dt
backend_dir = os.path.join(project_dir, "backend")
DB_PATH = os.path.join(backend_dir, "factory.db")

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data(ttl=60)
def load_sensor_data():
    """Load sensor data and compute failure risk"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 500"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            st.error("❌ No data in database. Run sensor_simulator.py in backend folder.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return None

def calculate_risk_score(row):
    """
    Calculate failure risk score (0-100)
    Based on: temperature + vibration + power + downtime patterns
    """
    temp_risk = (row['temperature'] - 50) / 40 * 25  # 0-25 points
    vib_risk = min(row['vibration'] / 5 * 40, 40)    # 0-40 points
    power_risk = (row['power_consumption'] - 10) / 20 * 20  # 0-20 points
    downtime_risk = min(row['downtime_minutes'] / 30 * 15, 15)  # 0-15 points
    
    total_risk = temp_risk + vib_risk + power_risk + downtime_risk
    return min(total_risk, 100)

# Load data
df = load_sensor_data()

if df is None:
    st.warning("No data available. Please run the sensor simulator first.")
    st.stop()

# Calculate risk scores per machine (latest reading)
latest_readings = df.sort_values('timestamp').groupby('machine_id').tail(1)
latest_readings['risk_score'] = latest_readings.apply(calculate_risk_score, axis=1)

# ============================================================================
# ALERT SUMMARY (Top Row)
# ============================================================================

st.markdown("### 🚨 Critical & High-Risk Alerts")

col1, col2, col3 = st.columns(3)

critical_count = (latest_readings['risk_score'] >= 80).sum()
high_count = (latest_readings['risk_score'] >= 60).sum() - critical_count
medium_count = (latest_readings['risk_score'] >= 40).sum() - high_count

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

for idx, (machine_id, row) in enumerate(latest_readings.iterrows()):
    risk = row['risk_score']
    
    # Determine color
    if risk >= 80:
        color = "🔴"
        status = "CRITICAL"
    elif risk >= 60:
        color = "🟠"
        status = "HIGH"
    elif risk >= 40:
        color = "🟡"
        status = "MEDIUM"
    else:
        color = "🟢"
        status = "GOOD"
    
    with cols[idx % 5]:
        st.metric(
            f"{color} {row['machine_id']}",
            f"{risk:.0f}%",
            status
        )
        
        # Show contributing factors
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

# Create alert dataframe
alerts_data = []
for machine_id, row in latest_readings.iterrows():
    risk = row['risk_score']
    
    # Determine recommendation
    if risk >= 80:
        action = "SCHEDULE IMMEDIATE MAINTENANCE - Replace bearings & check motor"
        estimated_days = 1
    elif risk >= 60:
        action = "Schedule maintenance within 1 week - Monitor vibration closely"
        estimated_days = 7
    elif risk >= 40:
        action = "Monitor equipment - Check lubrication and alignment"
        estimated_days = 14
    else:
        action = "Normal operation - Continue monitoring"
        estimated_days = 30
    
    alerts_data.append({
        'Machine': row['machine_id'],
        'Risk Score': f"{risk:.0f}%",
        'Temperature': f"{row['temperature']:.1f}°C",
        'Vibration': f"{row['vibration']:.2f} mm/s",
        'Recommended Action': action,
        'Est. Days to Failure': estimated_days
    })

alerts_df = pd.DataFrame(alerts_data)

# Color the risk score column
def highlight_risk(val):
    risk = float(val.rstrip('%'))
    if risk >= 80:
        return 'background-color: #ffcccc'  # Light red
    elif risk >= 60:
        return 'background-color: #ffe6cc'  # Light orange
    elif risk >= 40:
        return 'background-color: #ffffcc'  # Light yellow
    else:
        return 'background-color: #ccffcc'  # Light green

styled_df = alerts_df.style.map(highlight_risk, subset=['Risk Score'])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ============================================================================
# TREND ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("### 📈 Risk Trend Over Time (Last 24 Hours)")

# Calculate risk scores for all recent readings
df_recent = df[df['timestamp'] >= datetime.now() - timedelta(hours=24)].copy()
df_recent['risk_score'] = df_recent.apply(calculate_risk_score, axis=1)

# Plot risk trends
import plotly.express as px

fig = px.line(
    df_recent,
    x='timestamp',
    y='risk_score',
    color='machine_id',
    title="Risk Score Evolution",
    labels={'timestamp': 'Time', 'risk_score': 'Risk Score (%)', 'machine_id': 'Machine'},
    height=400
)

# Add alert threshold lines
fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Critical (80%)")
fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="High (60%)")
fig.add_hline(y=40, line_dash="dash", line_color="yellow", annotation_text="Medium (40%)")

fig.update_layout(
    hovermode='x unified',
    margin=dict(l=0, r=0, t=30, b=0)
)

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

# Estimate cost savings
critical_machines = (latest_readings['risk_score'] >= 80).sum()
estimated_downtime_risk = critical_machines * 4  # hours
cost_if_unplanned = estimated_downtime_risk * 3000  # $3000/hour average
cost_preventive = critical_machines * 1500  # $1500 per maintenance
savings = cost_if_unplanned - cost_preventive

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💵 Risk if Unplanned Failure", f"${cost_if_unplanned:,.0f}")

with col2:
    st.metric("🔧 Preventive Maintenance Cost", f"${cost_preventive:,.0f}")

with col3:
    st.metric("💚 Potential Savings", f"${savings:,.0f}", delta=f"{(savings/cost_if_unplanned*100):.0f}% ROI")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

st.markdown("---")
st.markdown("### 📌 Maintenance Schedule Recommendations")

recommendations = []

for machine_id, row in latest_readings.iterrows():
    risk = row['risk_score']
    if risk >= 60:
        recommendations.append({
            'Priority': '🔴 HIGH' if risk >= 80 else '🟠 MEDIUM',
            'Machine': row['machine_id'],
            'Issue': f"High vibration ({row['vibration']:.2f} mm/s) + elevated temp ({row['temperature']:.1f}°C)",
            'Action': "Replace bearings, check motor alignment",
            'Estimated Downtime': "2-4 hours",
            'Next Check': "ASAP" if risk >= 80 else "Within 1 week"
        })

if recommendations:
    rec_df = pd.DataFrame(recommendations)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ All machines operating normally. No immediate maintenance required.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>Last updated: {}</div>".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
), unsafe_allow_html=True)
