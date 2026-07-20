"""
Manufacturing Digital Twin - Admin Dashboard
Cross-plant KPIs, fleet health overview, and alert heatmap (admin role only)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from auth import require_role, get_auth_headers, render_sidebar_user

st.set_page_config(
    page_title="Admin Dashboard | Manufacturing DT",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Admin Dashboard")
st.markdown("Cross-plant performance overview — admin access only")

def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")

API_BASE = get_api_base()

require_role("admin")
render_sidebar_user()

# ============================================================================
# LOAD CROSS-PLANT DATA
# ============================================================================

@st.cache_data(ttl=60)
def load_cross_plant(api_base: str, token: str):
    try:
        resp = requests.get(
            f"{api_base}/admin/cross-plant",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as e:
        st.error(f"Failed to load cross-plant data: {e}")
        return None


token = st.session_state.get("auth_token", "")
with st.spinner("Loading cross-plant summary..."):
    df = load_cross_plant(API_BASE, token)

if df is None or df.empty:
    st.warning("No cross-plant data available.")
    st.stop()

PLANT_LABELS = {"alpha": "Plant Alpha", "beta": "Plant Beta", "gamma": "Plant Gamma"}
df["Plant"] = df["plant_id"].map(PLANT_LABELS).fillna(df["plant_id"])

# ============================================================================
# FLEET-WIDE KPI SUMMARY
# ============================================================================

st.markdown("### Fleet-Wide KPIs (Last 24 Hours)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Plants", len(df))
col2.metric("Total Production", f"{df['total_production_24h'].sum():,} parts")
col3.metric("Avg OEE (fleet)", f"{df['average_oee'].mean() * 100:.1f}%")
col4.metric("Active Alerts (fleet)", int(df["active_alerts"].sum()))

st.markdown("---")

# ============================================================================
# PER-PLANT COMPARISON CHARTS
# ============================================================================

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Production by Plant")
    fig_prod = px.bar(
        df,
        x="Plant",
        y="total_production_24h",
        color="Plant",
        text="total_production_24h",
        labels={"total_production_24h": "Parts Produced (24 h)"},
        height=320,
    )
    fig_prod.update_traces(textposition="outside")
    fig_prod.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_prod, use_container_width=True)

with col_right:
    st.markdown("#### OEE by Plant")
    fig_oee = px.bar(
        df,
        x="Plant",
        y=df["average_oee"] * 100,
        color="Plant",
        text=(df["average_oee"] * 100).round(1).astype(str) + "%",
        labels={"y": "OEE (%)"},
        height=320,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_oee.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="World-class (85%)")
    fig_oee.update_traces(textposition="outside")
    fig_oee.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_oee, use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Avg Quality Score by Plant")
    fig_q = px.bar(
        df,
        x="Plant",
        y="avg_quality",
        color="Plant",
        text=df["avg_quality"].round(1).astype(str) + "%",
        labels={"avg_quality": "Avg Quality (%)"},
        height=300,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_q.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target (95%)")
    fig_q.update_traces(textposition="outside")
    fig_q.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_q, use_container_width=True)

with col_right:
    st.markdown("#### Active Maintenance Alerts by Plant")
    fig_alerts = px.bar(
        df,
        x="Plant",
        y="active_alerts",
        color="active_alerts",
        color_continuous_scale="RdYlGn_r",
        text="active_alerts",
        labels={"active_alerts": "Active Alerts"},
        height=300,
    )
    fig_alerts.update_traces(textposition="outside")
    fig_alerts.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig_alerts, use_container_width=True)

st.markdown("---")

# ============================================================================
# PLANT SUMMARY TABLE
# ============================================================================

st.markdown("### Plant Summary Table")

summary = df[["Plant", "total_machines", "total_production_24h", "average_oee", "active_alerts", "avg_quality"]].copy()
summary.columns = ["Plant", "Machines", "Production (24h)", "OEE", "Active Alerts", "Avg Quality (%)"]
summary["OEE"] = (summary["OEE"] * 100).round(1).astype(str) + "%"

st.dataframe(summary, use_container_width=True, hide_index=True)

# ============================================================================
# DEMO CREDENTIALS INFO
# ============================================================================

with st.expander("ℹ️ Role-Based Access Control"):
    st.markdown("""
**Accounts available in this demo:**

| Username | Password | Role | Access |
|----------|----------|------|--------|
| `admin` | `admin` | Admin | All 3 plants + this admin dashboard |
| `operator` | `operator` | Operator | Plant Alpha only |

**How it works:**
- JWT tokens signed with a secret key (env var `JWT_SECRET`)
- Operator's plant scope is embedded in the token — every API call is server-side filtered
- This admin page calls `GET /admin/cross-plant` which requires a valid admin-role token
- Streamlit page guards (`require_role`) block the page from rendering for non-admins
""")

st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
