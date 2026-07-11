"""
Manufacturing Digital Twin - Production Scheduling Optimizer
Greedy nearest-neighbor job sequencing to minimize changeover time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Scheduler | Manufacturing DT",
    page_icon="📅",
    layout="wide"
)

st.title("📅 Production Scheduling Optimizer")
st.markdown("Minimize changeover time using SMED-based job sequencing")

def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")

API_BASE = get_api_base()

# ============================================================================
# CONSTANTS
# ============================================================================

PRODUCT_TYPES = ["Part_A", "Part_B", "Part_C"]
MACHINES = ["M1", "M2", "M3", "M4", "M5"]

# Changeover matrix: minutes needed to switch between product types
CHANGEOVER_MATRIX = {
    ("Part_A", "Part_A"): 5,
    ("Part_A", "Part_B"): 15,
    ("Part_A", "Part_C"): 25,
    ("Part_B", "Part_A"): 15,
    ("Part_B", "Part_B"): 5,
    ("Part_B", "Part_C"): 20,
    ("Part_C", "Part_A"): 25,
    ("Part_C", "Part_B"): 20,
    ("Part_C", "Part_C"): 5,
}

def get_changeover(p1: str, p2: str) -> int:
    return CHANGEOVER_MATRIX.get((p1, p2), 20)

def total_changeover_time(jobs: list) -> int:
    if len(jobs) <= 1:
        return 0
    return sum(
        get_changeover(jobs[i - 1]["product_type"], jobs[i]["product_type"])
        for i in range(1, len(jobs))
    )

def nearest_neighbor_optimize(jobs: list) -> list:
    """Greedy nearest-neighbor: always pick next job with minimum changeover cost."""
    if len(jobs) <= 1:
        return jobs.copy()
    remaining = jobs.copy()
    optimized = [remaining.pop(0)]
    while remaining:
        current_type = optimized[-1]["product_type"]
        best_idx = min(
            range(len(remaining)),
            key=lambda i: get_changeover(current_type, remaining[i]["product_type"])
        )
        optimized.append(remaining.pop(best_idx))
    return optimized

def build_gantt_df(jobs: list, label: str) -> pd.DataFrame:
    """Convert job list to Gantt dataframe with calculated start/end times."""
    rows = []
    machine_clocks = {m: datetime(2026, 1, 1, 6, 0) for m in MACHINES}  # 06:00 shift start
    prev_by_machine = {}

    for job in jobs:
        machine = job["machine_id"]
        start = machine_clocks[machine]

        # Add changeover from previous product on same machine
        prev_type = prev_by_machine.get(machine)
        if prev_type:
            changeover = get_changeover(prev_type, job["product_type"])
            start += timedelta(minutes=changeover)
            machine_clocks[machine] = start

        end = start + timedelta(hours=job["duration_hours"])
        machine_clocks[machine] = end
        prev_by_machine[machine] = job["product_type"]

        rows.append({
            "Job": job["job_id"],
            "Machine": machine,
            "Product": job["product_type"],
            "Start": start,
            "Finish": end,
            "Quantity": job["quantity"],
            "Schedule": label,
        })
    return pd.DataFrame(rows)

# ============================================================================
# SESSION STATE: DEFAULT JOB LIST
# ============================================================================

DEFAULT_JOBS = [
    {"job_id": "J1", "product_type": "Part_A", "machine_id": "M1", "quantity": 100, "duration_hours": 2.0},
    {"job_id": "J2", "product_type": "Part_C", "machine_id": "M2", "quantity": 150, "duration_hours": 3.0},
    {"job_id": "J3", "product_type": "Part_B", "machine_id": "M1", "quantity": 80,  "duration_hours": 1.5},
    {"job_id": "J4", "product_type": "Part_A", "machine_id": "M3", "quantity": 200, "duration_hours": 4.0},
    {"job_id": "J5", "product_type": "Part_C", "machine_id": "M2", "quantity": 120, "duration_hours": 2.5},
    {"job_id": "J6", "product_type": "Part_B", "machine_id": "M3", "quantity": 90,  "duration_hours": 2.0},
]

if "jobs" not in st.session_state:
    st.session_state.jobs = DEFAULT_JOBS.copy()

# ============================================================================
# SIDEBAR: CONTROLS
# ============================================================================

st.sidebar.title("⚙️ Controls")
if st.sidebar.button("🔄 Reset to Default Jobs", use_container_width=True):
    st.session_state.jobs = DEFAULT_JOBS.copy()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Changeover Times")
st.sidebar.markdown("""
| From → To | A | B | C |
|-----------|---|---|---|
| **Part_A** | 5 | 15 | 25 |
| **Part_B** | 15 | 5 | 20 |
| **Part_C** | 25 | 20 | 5 |

*(minutes)*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 SMED Principle")
st.sidebar.info(
    "**Single-Minute Exchange of Die** — group similar jobs together to minimize "
    "the number of costly product transitions."
)

# ============================================================================
# JOB INPUT TABLE
# ============================================================================

st.markdown("### 📝 Job Queue (Edit, Add, or Delete Rows)")
st.markdown("Edit the table below to configure your production jobs. Add rows with the ➕ button.")

jobs_df = pd.DataFrame(st.session_state.jobs)

edited_df = st.data_editor(
    jobs_df,
    column_config={
        "job_id": st.column_config.TextColumn("Job ID", width="small"),
        "product_type": st.column_config.SelectboxColumn(
            "Product Type", options=PRODUCT_TYPES, width="medium"
        ),
        "machine_id": st.column_config.SelectboxColumn(
            "Machine", options=MACHINES, width="small"
        ),
        "quantity": st.column_config.NumberColumn(
            "Qty (parts)", min_value=1, max_value=1000, step=10, width="small"
        ),
        "duration_hours": st.column_config.NumberColumn(
            "Duration (h)", min_value=0.5, max_value=24.0, step=0.5, width="small"
        ),
    },
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="job_editor",
)

# Sync edits back to session state
st.session_state.jobs = edited_df.dropna(subset=["job_id"]).to_dict("records")

# ============================================================================
# OPTIMIZATION
# ============================================================================

st.markdown("---")

if len(st.session_state.jobs) < 2:
    st.warning("Add at least 2 jobs to run the optimizer.")
    st.stop()

original_jobs = st.session_state.jobs
optimized_jobs = nearest_neighbor_optimize(original_jobs)

original_changeover = total_changeover_time(original_jobs)
optimized_changeover = total_changeover_time(optimized_jobs)
time_saved = original_changeover - optimized_changeover
pct_saved = (time_saved / original_changeover * 100) if original_changeover > 0 else 0

# ============================================================================
# KPI METRICS
# ============================================================================

st.markdown("### ⚡ Optimization Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jobs Scheduled", len(original_jobs))
with col2:
    st.metric("Original Changeover", f"{original_changeover} min")
with col3:
    st.metric("Optimized Changeover", f"{optimized_changeover} min", delta=f"-{time_saved} min")
with col4:
    st.metric("Time Saved", f"{time_saved} min", delta=f"{pct_saved:.1f}% reduction")

# Estimated cost savings (assume $50/min opportunity cost)
cost_saved = time_saved * 50
st.success(
    f"✅ Optimized schedule saves **{time_saved} minutes** of changeover time, "
    f"worth approximately **${cost_saved:,.0f}** in production opportunity cost."
)

# ============================================================================
# OPTIMIZED JOB SEQUENCE
# ============================================================================

st.markdown("---")
st.markdown("### 🔀 Sequence Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Original Order")
    orig_seq = pd.DataFrame([
        {
            "Step": i + 1,
            "Job": j["job_id"],
            "Product": j["product_type"],
            "Machine": j["machine_id"],
            "Changeover": f"{get_changeover(original_jobs[i-1]['product_type'], j['product_type'])} min" if i > 0 else "—"
        }
        for i, j in enumerate(original_jobs)
    ])
    st.dataframe(orig_seq, use_container_width=True, hide_index=True)
    st.caption(f"Total changeover: **{original_changeover} min**")

with col2:
    st.markdown("#### Optimized Order")
    opt_seq = pd.DataFrame([
        {
            "Step": i + 1,
            "Job": j["job_id"],
            "Product": j["product_type"],
            "Machine": j["machine_id"],
            "Changeover": f"{get_changeover(optimized_jobs[i-1]['product_type'], j['product_type'])} min" if i > 0 else "—"
        }
        for i, j in enumerate(optimized_jobs)
    ])
    st.dataframe(opt_seq, use_container_width=True, hide_index=True)
    st.caption(f"Total changeover: **{optimized_changeover} min**")

# ============================================================================
# GANTT CHARTS
# ============================================================================

st.markdown("---")
st.markdown("### 📊 Production Gantt Charts")

tab1, tab2 = st.tabs(["✅ Optimized Schedule", "⚠️ Original Schedule"])

orig_gantt_df = build_gantt_df(original_jobs, "Original")
opt_gantt_df  = build_gantt_df(optimized_jobs, "Optimized")

COLOR_MAP = {"Part_A": "#1f77b4", "Part_B": "#ff7f0e", "Part_C": "#2ca02c"}

with tab1:
    fig_opt = px.timeline(
        opt_gantt_df,
        x_start="Start", x_end="Finish",
        y="Machine", color="Product",
        text="Job",
        title=f"Optimized Schedule (Total changeover: {optimized_changeover} min)",
        color_discrete_map=COLOR_MAP,
        height=350,
        hover_data={"Quantity": True, "Product": True, "Job": True},
    )
    fig_opt.update_yaxes(autorange="reversed")
    fig_opt.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_opt, use_container_width=True)

with tab2:
    fig_orig = px.timeline(
        orig_gantt_df,
        x_start="Start", x_end="Finish",
        y="Machine", color="Product",
        text="Job",
        title=f"Original Schedule (Total changeover: {original_changeover} min)",
        color_discrete_map=COLOR_MAP,
        height=350,
        hover_data={"Quantity": True, "Product": True, "Job": True},
    )
    fig_orig.update_yaxes(autorange="reversed")
    fig_orig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_orig, use_container_width=True)

# ============================================================================
# CHANGEOVER HEATMAP
# ============================================================================

st.markdown("---")
st.markdown("### 🗺️ Changeover Matrix (minutes)")

matrix_data = [[CHANGEOVER_MATRIX.get((r, c), 0) for c in PRODUCT_TYPES] for r in PRODUCT_TYPES]
fig_heat = go.Figure(data=go.Heatmap(
    z=matrix_data,
    x=PRODUCT_TYPES,
    y=PRODUCT_TYPES,
    colorscale="RdYlGn_r",
    text=matrix_data,
    texttemplate="%{text} min",
    showscale=True,
))
fig_heat.update_layout(
    title="Changeover Time Matrix (lower = better)",
    xaxis_title="To Product",
    yaxis_title="From Product",
    height=300,
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# SMED ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("### 🔩 SMED Changeover Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Changeover by Transition Type")
    transitions = {}
    for i in range(1, len(optimized_jobs)):
        key = f"{optimized_jobs[i-1]['product_type']} → {optimized_jobs[i]['product_type']}"
        transitions[key] = get_changeover(
            optimized_jobs[i-1]["product_type"], optimized_jobs[i]["product_type"]
        )

    if transitions:
        trans_df = pd.DataFrame(
            [{"Transition": k, "Changeover (min)": v} for k, v in transitions.items()]
        ).sort_values("Changeover (min)", ascending=False)
        fig_bar = px.bar(
            trans_df, x="Transition", y="Changeover (min)",
            color="Changeover (min)", color_continuous_scale="RdYlGn_r",
            height=300,
        )
        fig_bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("#### SMED Improvement Tips")
    st.info("""
**Reduce changeover time with:**

🔧 **Internal → External** — Move prep tasks (tooling, materials) to happen *while* the machine is still running

📋 **Standardize** — Create changeover checklists to avoid trial-and-error at shutdown

🗂️ **Group similar jobs** — Run all Part_A jobs before switching to Part_B (this optimizer does exactly this!)

⏱️ **Set targets** — Track each changeover; aim for <10 min per switch

💡 **Quick-change tooling** — Replace bolts with clamps, eliminate fine adjustments
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    f"Scheduler updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Algorithm: Greedy Nearest-Neighbor (O(n²))"
    "</div>",
    unsafe_allow_html=True
)
