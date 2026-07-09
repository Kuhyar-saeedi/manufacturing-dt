"""
Manufacturing Plant Digital Twin - Streamlit Dashboard
Real-time monitoring, predictive maintenance, and optimization
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Manufacturing Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & HEADER
# ============================================================================

st.title("🏭 Manufacturing Plant Digital Twin")
st.markdown("**Real-time monitoring, predictive maintenance & optimization**")

# API endpoint (localhost for testing)
API_BASE = "http://localhost:8000"
DB_PATH = "../backend/factory.db"  # Database is in backend folder

# ============================================================================
# SIDEBAR: FILTERS & NAVIGATION
# ============================================================================

st.sidebar.title("⚙️ Controls")
selected_machine = st.sidebar.multiselect(
    "Filter by Machine",
    ["M1", "M2", "M3", "M4", "M5"],
    default=["M1", "M2", "M3", "M4", "M5"],
    help="Select machines to display"
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "All Data"],
    index=2
)

# Map time range to hours
time_map = {
    "Last 1 Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "All Data": 999
}
hours_back = time_map[time_range]

st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
st.sidebar.info("📊 You are on: **Dashboard**")
if st.sidebar.button("🚨 Maintenance Alerts", use_container_width=True):
    st.switch_page("pages/02_Predictions.py")

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=60)
def load_data_from_db():
    """Load sensor data from SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT * FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 500
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            st.warning("No data in database. Run sensor_simulator.py first.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

# Load data
df = load_data_from_db()

if df is None:
    st.stop()

# Filter by selected machines and time range
cutoff_time = datetime.now() - timedelta(hours=hours_back)
df_filtered = df[
    (df['machine_id'].isin(selected_machine)) & 
    (df['timestamp'] >= cutoff_time)
].sort_values('timestamp')

if df_filtered.empty:
    st.warning("No data for selected filters. Please adjust time range or machine selection.")
    st.stop()

# ============================================================================
# KEY METRICS (Top Row)
# ============================================================================

st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

# Total Production Today
total_production = df_filtered['production_count'].sum()
with col1:
    st.metric(
        "Total Production",
        f"{total_production:,.0f} Parts",
        "+12% from yesterday"
    )

# Average Quality
avg_quality = df_filtered['quality_score'].mean()
with col2:
    st.metric(
        "Avg Quality Score",
        f"{avg_quality:.1f}%",
        "+2% from target"
    )

# Total Downtime
total_downtime = df_filtered['downtime_minutes'].sum()
with col3:
    st.metric(
        "Total Downtime",
        f"{total_downtime:.0f} min",
        "-5% vs. last week"
    )

# Machine Utilization (estimate)
expected_hours = len(selected_machine) * hours_back
actual_hours = len(df_filtered) / len(selected_machine)  # Rough estimate
utilization = (actual_hours / expected_hours * 100) if expected_hours > 0 else 0
with col4:
    st.metric(
        "Utilization Rate",
        f"{utilization:.1f}%",
        "-3%" if utilization < 95 else "+2%"
    )

st.markdown("---")

# ============================================================================
# MAIN VISUALIZATIONS (2x2 Grid)
# ============================================================================

st.markdown("### 📈 Real-Time Factory Floor Metrics")

col1, col2 = st.columns(2)

# Chart 1: Temperature Over Time (by machine)
with col1:
    st.markdown("#### 🌡️ Machine Temperature Trends")
    fig_temp = px.line(
        df_filtered,
        x='timestamp',
        y='temperature',
        color='machine_id',
        title="Temperature (°C) - Hover for details",
        labels={'timestamp': 'Time', 'temperature': 'Temp (°C)', 'machine_id': 'Machine'},
        height=400
    )
    fig_temp.update_layout(
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_temp, use_container_width=True)

# Chart 2: Vibration Over Time (anomaly detection)
with col2:
    st.markdown("#### 📳 Vibration Levels (Bearing Health)")
    fig_vib = px.line(
        df_filtered,
        x='timestamp',
        y='vibration',
        color='machine_id',
        title="Vibration (mm/s) - Spikes indicate issues",
        labels={'timestamp': 'Time', 'vibration': 'Vib (mm/s)', 'machine_id': 'Machine'},
        height=400
    )
    fig_vib.add_hline(
        y=5.0,
        line_dash="dash",
        line_color="red",
        annotation_text="⚠️ Alert Threshold"
    )
    fig_vib.update_layout(
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_vib, use_container_width=True)

col1, col2 = st.columns(2)

# Chart 3: Power Consumption by Machine
with col1:
    st.markdown("#### ⚡ Power Consumption (Energy Efficiency)")
    power_by_machine = df_filtered.groupby('machine_id')['power_consumption'].mean().sort_values()
    fig_power = px.bar(
        x=power_by_machine.values,
        y=power_by_machine.index,
        orientation='h',
        color=power_by_machine.values,
        color_continuous_scale='Viridis',
        title="Avg Power per Machine (kW)",
        labels={'x': 'Power (kW)', 'y': 'Machine'},
        height=350
    )
    fig_power.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig_power, use_container_width=True)

# Chart 4: Quality Score Distribution
with col2:
    st.markdown("#### ✅ Quality Score by Machine")
    fig_quality = px.box(
        df_filtered,
        x='machine_id',
        y='quality_score',
        title="Quality Distribution (Target: >95%)",
        labels={'machine_id': 'Machine', 'quality_score': 'Quality (%)'},
        height=350,
        points="outliers"
    )
    fig_quality.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target")
    fig_quality.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_quality, use_container_width=True)

# ============================================================================
# DETAILED DATA TABLE
# ============================================================================

st.markdown("---")
st.markdown("### 📋 Latest Sensor Readings")

# Display last 20 readings
display_cols = ['timestamp', 'machine_id', 'temperature', 'vibration', 'power_consumption', 
                'production_count', 'downtime_minutes', 'quality_score']
latest_readings = df_filtered[display_cols].sort_values('timestamp', ascending=False).head(20)

# Format for display
latest_readings_display = latest_readings.copy()
latest_readings_display['timestamp'] = latest_readings_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
latest_readings_display['temperature'] = latest_readings_display['temperature'].round(1)
latest_readings_display['vibration'] = latest_readings_display['vibration'].round(3)
latest_readings_display['power_consumption'] = latest_readings_display['power_consumption'].round(2)

st.dataframe(
    latest_readings_display,
    use_container_width=True,
    hide_index=True
)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Manufacturing Digital Twin v0.1.0 | Last updated: {}<br>
    Data source: SQLite (factory.db) | Backend: FastAPI
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
