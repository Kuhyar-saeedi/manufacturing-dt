"""
Manufacturing Digital Twin - Analytics & Reports
Historical trends, OEE breakdown, quality/energy analysis, and data export
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import os

st.set_page_config(
    page_title="Reports | Manufacturing DT",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analytics & Reports")
st.markdown("Historical performance analysis, KPI trends, and export tools")

def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")

API_BASE = get_api_base()

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_all_data(api_base: str):
    try:
        resp = requests.get(f"{api_base}/sensor-readings", params={"hours": 48}, timeout=15)
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

def calculate_oee(df: pd.DataFrame) -> dict:
    """Calculate Overall Equipment Effectiveness components."""
    total_time_min = df["timestamp"].nunique() * 1  # 1 min per reading approx
    availability = 1 - (df["downtime_minutes"].sum() / max(total_time_min * len(df["machine_id"].unique()), 1))
    performance = df["production_count"].sum() / max(len(df["machine_id"].unique()) * 100, 1)
    quality = df["quality_score"].mean() / 100
    oee = max(min(availability * performance * quality, 1.0), 0.0)
    return {
        "oee": oee,
        "availability": max(min(availability, 1.0), 0.0),
        "performance": max(min(performance, 1.0), 0.0),
        "quality": quality,
    }

with st.spinner("Loading analytics data..."):
    df_raw = load_all_data(API_BASE)

if df_raw is None:
    st.warning("No data available. Make sure the backend is running.")
    st.stop()

# ============================================================================
# SIDEBAR: FILTERS
# ============================================================================

st.sidebar.title("⚙️ Filters")

selected_machines = st.sidebar.multiselect(
    "Machines",
    options=sorted(df_raw["machine_id"].unique()),
    default=sorted(df_raw["machine_id"].unique()),
)

time_range = st.sidebar.selectbox(
    "Time Window",
    ["Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Last 48 Hours"],
    index=2,
)

time_map = {"Last 6 Hours": 6, "Last 12 Hours": 12, "Last 24 Hours": 24, "Last 48 Hours": 48}
hours_back = time_map[time_range]
cutoff = df_raw["timestamp"].max() - timedelta(hours=hours_back)

grouping = st.sidebar.selectbox(
    "Trend Grouping",
    ["Hourly", "Every 2 Hours", "Every 4 Hours"],
    index=0,
)
freq_map = {"Hourly": "1h", "Every 2 Hours": "2h", "Every 4 Hours": "4h"}
freq = freq_map[grouping]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing data for: **{time_range}**\n\nMachines: **{', '.join(selected_machines)}**")

# Apply filters
df = df_raw[
    (df_raw["timestamp"] >= cutoff) &
    (df_raw["machine_id"].isin(selected_machines))
].copy()

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ============================================================================
# TOP KPI STRIP
# ============================================================================

st.markdown("### 📊 Period Summary")

kpis = calculate_oee(df)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Overall OEE", f"{kpis['oee']*100:.1f}%", help="Availability × Performance × Quality")
with col2:
    st.metric("Availability", f"{kpis['availability']*100:.1f}%")
with col3:
    st.metric("Performance", f"{kpis['performance']*100:.1f}%")
with col4:
    st.metric("Avg Quality", f"{df['quality_score'].mean():.1f}%")
with col5:
    downtime_cost = df["downtime_minutes"].sum() * 50
    st.metric("Est. Downtime Cost", f"${downtime_cost:,.0f}")

st.markdown("---")

# ============================================================================
# TABBED DASHBOARD
# ============================================================================

tab_prod, tab_quality, tab_energy, tab_oee, tab_export = st.tabs([
    "🏭 Production", "✅ Quality", "⚡ Energy", "📐 OEE Breakdown", "💾 Export"
])

# ────────────────────────────────────────────────
# TAB 1: PRODUCTION TRENDS
# ────────────────────────────────────────────────

with tab_prod:
    st.markdown("#### Production Count Over Time")

    df_prod = (
        df.set_index("timestamp")
        .groupby("machine_id")["production_count"]
        .resample(freq)
        .sum()
        .reset_index()
    )

    fig_prod = px.line(
        df_prod, x="timestamp", y="production_count", color="machine_id",
        title=f"Parts Produced per Machine ({grouping})",
        labels={"timestamp": "Time", "production_count": "Parts", "machine_id": "Machine"},
        height=400,
    )
    fig_prod.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_prod, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Total Production by Machine")
        prod_total = df.groupby("machine_id")["production_count"].sum().reset_index()
        prod_total.columns = ["Machine", "Total Parts"]
        fig_bar = px.bar(
            prod_total, x="Machine", y="Total Parts",
            color="Total Parts", color_continuous_scale="Blues",
            title="Total Parts Produced",
            height=300,
        )
        fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### Downtime Distribution")
        downtime_by_machine = df.groupby("machine_id")["downtime_minutes"].sum().reset_index()
        downtime_by_machine.columns = ["Machine", "Total Downtime (min)"]
        fig_dt = px.pie(
            downtime_by_machine, names="Machine", values="Total Downtime (min)",
            title="Downtime Share by Machine",
            height=300,
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        fig_dt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_dt, use_container_width=True)

# ────────────────────────────────────────────────
# TAB 2: QUALITY
# ────────────────────────────────────────────────

with tab_quality:
    st.markdown("#### Quality Score Distribution by Machine")

    fig_box = px.box(
        df, x="machine_id", y="quality_score",
        color="machine_id",
        title="Quality Score Distribution (Target: >95%)",
        labels={"machine_id": "Machine", "quality_score": "Quality Score (%)"},
        height=400,
        points="outliers",
    )
    fig_box.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target (95%)")
    fig_box.add_hline(y=85, line_dash="dash", line_color="red",   annotation_text="Min Acceptable (85%)")
    fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Quality Trend Over Time")
        df_quality = (
            df.set_index("timestamp")
            .groupby("machine_id")["quality_score"]
            .resample(freq)
            .mean()
            .reset_index()
        )
        fig_qt = px.line(
            df_quality, x="timestamp", y="quality_score", color="machine_id",
            title="Avg Quality Score Over Time",
            labels={"timestamp": "Time", "quality_score": "Quality (%)", "machine_id": "Machine"},
            height=320,
        )
        fig_qt.add_hline(y=95, line_dash="dash", line_color="green")
        fig_qt.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_qt, use_container_width=True)

    with col2:
        st.markdown("#### Quality Stats by Machine")
        quality_stats = df.groupby("machine_id")["quality_score"].agg(
            Mean="mean", Std="std", Min="min", Max="max",
            Below95=lambda x: (x < 95).sum()
        ).round(1).reset_index()
        quality_stats.columns = ["Machine", "Mean (%)", "Std Dev", "Min (%)", "Max (%)", "Readings Below 95%"]
        st.dataframe(quality_stats, use_container_width=True, hide_index=True)

        # Defect rate insight
        total_readings = len(df)
        below_target = (df["quality_score"] < 95).sum()
        defect_rate = below_target / total_readings * 100 if total_readings > 0 else 0
        if defect_rate > 10:
            st.error(f"⚠️ {defect_rate:.1f}% of readings below quality target — investigate immediately.")
        elif defect_rate > 5:
            st.warning(f"🟡 {defect_rate:.1f}% of readings below quality target — monitor closely.")
        else:
            st.success(f"✅ {defect_rate:.1f}% of readings below quality target — on track.")

# ────────────────────────────────────────────────
# TAB 3: ENERGY
# ────────────────────────────────────────────────

with tab_energy:
    st.markdown("#### Power Consumption Analysis")

    df_energy = (
        df.set_index("timestamp")
        .groupby("machine_id")["power_consumption"]
        .resample(freq)
        .mean()
        .reset_index()
    )

    fig_energy = px.area(
        df_energy, x="timestamp", y="power_consumption", color="machine_id",
        title=f"Average Power Consumption Over Time (kW)",
        labels={"timestamp": "Time", "power_consumption": "Power (kW)", "machine_id": "Machine"},
        height=380,
    )
    fig_energy.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_energy, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Avg Power by Machine (kW)")
        avg_power = df.groupby("machine_id")["power_consumption"].mean().sort_values(ascending=False)
        fig_pw = px.bar(
            x=avg_power.index, y=avg_power.values,
            color=avg_power.values, color_continuous_scale="Viridis",
            labels={"x": "Machine", "y": "Avg Power (kW)"},
            height=280,
        )
        fig_pw.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_pw, use_container_width=True)

    with col2:
        st.markdown("#### Energy Efficiency (Parts per kWh)")
        # Efficiency = production_count / power_consumption per machine
        eff = df.groupby("machine_id").apply(
            lambda g: g["production_count"].sum() / max(g["power_consumption"].sum(), 1)
        ).reset_index()
        eff.columns = ["Machine", "Parts per kWh"]
        eff["Parts per kWh"] = eff["Parts per kWh"].round(2)
        fig_eff = px.bar(
            eff, x="Machine", y="Parts per kWh",
            color="Parts per kWh", color_continuous_scale="Greens",
            height=280,
        )
        fig_eff.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_eff, use_container_width=True)

    with col3:
        st.markdown("#### Energy Cost Estimate")
        electricity_rate = st.number_input(
            "Electricity Rate ($/kWh)", value=0.12, step=0.01, format="%.3f"
        )
        # Approximate: avg kW × hours × rate
        hours_in_period = hours_back
        total_kwh = df.groupby("machine_id")["power_consumption"].mean().sum() * hours_in_period
        total_cost = total_kwh * electricity_rate
        st.metric("Total kWh Consumed", f"{total_kwh:,.0f} kWh")
        st.metric("Estimated Energy Cost", f"${total_cost:,.2f}")
        st.caption(f"Based on {hours_in_period}h window × avg load")

# ────────────────────────────────────────────────
# TAB 4: OEE BREAKDOWN
# ────────────────────────────────────────────────

with tab_oee:
    st.markdown("#### OEE per Machine")
    st.markdown("*OEE = Availability × Performance × Quality (World-class: >85%)*")

    oee_rows = []
    for machine in sorted(df["machine_id"].unique()):
        mdf = df[df["machine_id"] == machine]
        m_oee = calculate_oee(mdf)
        oee_rows.append({
            "Machine": machine,
            "OEE (%)": round(m_oee["oee"] * 100, 1),
            "Availability (%)": round(m_oee["availability"] * 100, 1),
            "Performance (%)": round(m_oee["performance"] * 100, 1),
            "Quality (%)": round(m_oee["quality"] * 100, 1),
        })
    oee_df = pd.DataFrame(oee_rows)

    # Gauge charts row
    gauge_cols = st.columns(len(oee_rows))
    for idx, row in oee_df.iterrows():
        with gauge_cols[idx]:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row["OEE (%)"],
                title={"text": row["Machine"], "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0,  60], "color": "#ffcccc"},
                        {"range": [60, 85], "color": "#ffe6cc"},
                        {"range": [85, 100], "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "green", "width": 3},
                        "thickness": 0.75,
                        "value": 85,
                    },
                },
                number={"suffix": "%"},
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### OEE Component Breakdown")
    fig_oee_bar = px.bar(
        oee_df.melt(id_vars="Machine", value_vars=["Availability (%)", "Performance (%)", "Quality (%)"]),
        x="Machine", y="value", color="variable", barmode="group",
        title="OEE Components by Machine",
        labels={"value": "Score (%)", "variable": "Component"},
        height=360,
    )
    fig_oee_bar.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="World Class (85%)")
    fig_oee_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_oee_bar, use_container_width=True)

    st.dataframe(oee_df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────
# TAB 5: EXPORT
# ────────────────────────────────────────────────

with tab_export:
    st.markdown("### 💾 Export Data & Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Download Raw Sensor Data")
        csv_buffer = io.StringIO()
        export_cols = ["timestamp", "machine_id", "temperature", "vibration",
                       "power_consumption", "production_count", "downtime_minutes", "quality_score"]
        df[export_cols].sort_values("timestamp").to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download CSV (Sensor Data)",
            data=csv_buffer.getvalue(),
            file_name=f"manufacturing_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(df):,} rows × {len(export_cols)} columns")

    with col2:
        st.markdown("#### Download OEE Summary")
        oee_csv = io.StringIO()
        oee_df.to_csv(oee_csv, index=False)
        st.download_button(
            label="⬇️ Download CSV (OEE Summary)",
            data=oee_csv.getvalue(),
            file_name=f"oee_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📄 PDF Summary Report")

    try:
        from fpdf import FPDF

        def generate_pdf_report(df: pd.DataFrame, oee_df: pd.DataFrame, time_range: str) -> bytes:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_margins(15, 15, 15)

            # Header
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 12, "Manufacturing Digital Twin", ln=True, align="C")
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 8, f"Performance Report — {time_range}", ln=True, align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(6)

            # Summary KPIs
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Key Performance Indicators", ln=True)
            pdf.set_fill_color(230, 240, 255)
            pdf.set_font("Helvetica", "", 10)

            overall_oee = calculate_oee(df)
            kpi_data = [
                ("Period",             time_range),
                ("Machines Monitored", str(df["machine_id"].nunique())),
                ("Total Readings",     f"{len(df):,}"),
                ("Overall OEE",        f"{overall_oee['oee']*100:.1f}%"),
                ("Availability",       f"{overall_oee['availability']*100:.1f}%"),
                ("Performance",        f"{overall_oee['performance']*100:.1f}%"),
                ("Avg Quality Score",  f"{df['quality_score'].mean():.1f}%"),
                ("Total Downtime",     f"{df['downtime_minutes'].sum():.0f} min"),
                ("Total Production",   f"{df['production_count'].sum():,} parts"),
                ("Est. Downtime Cost", f"${df['downtime_minutes'].sum()*50:,.0f}"),
            ]

            for i, (label, value) in enumerate(kpi_data):
                fill = (i % 2 == 0)
                pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(90, 7, f"  {label}", border=0, fill=fill)
                pdf.cell(0, 7, value, ln=True, fill=fill)

            pdf.ln(6)

            # OEE by Machine
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "OEE by Machine", ln=True)
            pdf.set_font("Helvetica", "B", 9)
            headers = ["Machine", "OEE (%)", "Availability", "Performance", "Quality"]
            col_w = [30, 32, 32, 32, 32]
            pdf.set_fill_color(70, 130, 180)
            pdf.set_text_color(255, 255, 255)
            for h, w in zip(headers, col_w):
                pdf.cell(w, 7, h, border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 9)
            for i, row in oee_df.iterrows():
                fill = (i % 2 == 0)
                pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(30, 6, row["Machine"], border=1, fill=fill, align="C")
                pdf.cell(32, 6, f"{row['OEE (%)']:.1f}%", border=1, fill=fill, align="C")
                pdf.cell(32, 6, f"{row['Availability (%)']:.1f}%", border=1, fill=fill, align="C")
                pdf.cell(32, 6, f"{row['Performance (%)']:.1f}%", border=1, fill=fill, align="C")
                pdf.cell(32, 6, f"{row['Quality (%)']:.1f}%", border=1, fill=fill, align="C")
                pdf.ln()

            pdf.ln(6)

            # Quality summary
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "Quality Analysis", ln=True)
            pdf.set_font("Helvetica", "", 10)
            below95 = (df["quality_score"] < 95).sum()
            defect_rate = below95 / len(df) * 100
            pdf.cell(0, 6, f"Readings below 95% quality target: {below95:,} ({defect_rate:.1f}%)", ln=True)
            pdf.cell(0, 6, f"Min quality score: {df['quality_score'].min():.1f}%", ln=True)
            pdf.cell(0, 6, f"Max quality score: {df['quality_score'].max():.1f}%", ln=True)
            pdf.cell(0, 6, f"Std deviation: {df['quality_score'].std():.2f}%", ln=True)

            pdf.ln(6)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 5, "Manufacturing Digital Twin v0.1.0 | Generated by automated reporting system", ln=True, align="C")

            return pdf.output()

        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf_report(df, oee_df, time_range)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=bytes(pdf_bytes),
                file_name=f"manufacturing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("PDF ready for download!")

    except ImportError:
        st.info(
            "PDF export requires `fpdf2`. "
            "Install it with: `pip install fpdf2`  \n"
            "Then restart the app to enable PDF reports."
        )

    st.markdown("---")
    st.markdown("#### 📋 Data Preview")
    preview_cols = ["timestamp", "machine_id", "temperature", "vibration",
                    "power_consumption", "production_count", "quality_score"]
    st.dataframe(
        df[preview_cols].sort_values("timestamp", ascending=False).head(50),
        use_container_width=True, hide_index=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray; font-size: 12px;'>"
    f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Data window: {time_range} | Backend: {API_BASE}"
    f"</div>",
    unsafe_allow_html=True,
)
