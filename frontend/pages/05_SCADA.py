"""
Manufacturing Digital Twin - SCADA / OPC-UA Monitor
Live view of the OPC-UA server address space, node IDs, and subscription feed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from auth import require_auth, render_sidebar_user

st.set_page_config(
    page_title="SCADA / OPC-UA | Manufacturing DT",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 SCADA / OPC-UA Monitor")
st.markdown("Live OPC-UA server address space · node subscriptions · real-time tag values")


def get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("API_BASE", "http://localhost:8000")


API_BASE = get_api_base()

require_auth()
render_sidebar_user()

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data(ttl=5)
def load_opcua_status(api_base: str):
    try:
        resp = requests.get(f"{api_base}/opcua-status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


status = load_opcua_status(API_BASE)

# ============================================================================
# CONNECTION BANNER
# ============================================================================

col_status, col_endpoint, col_readings = st.columns([1, 3, 2])

if status is None:
    col_status.error("API unreachable")
elif status["connected"]:
    col_status.success("OPC-UA Connected")
else:
    col_status.warning("OPC-UA Connecting…")

if status:
    col_endpoint.markdown(f"**Endpoint:** `{status['endpoint']}`  \n**Namespace:** `urn:manufacturing:dt`")
    col_readings.metric("Readings ingested via OPC-UA", f"{status['readings_received']:,}")

st.divider()

if status is None:
    st.info("Cannot reach the backend. Make sure the FastAPI server is running.")
    st.stop()

# ============================================================================
# ADDRESS SPACE — node ID table
# ============================================================================

st.subheader("📋 OPC-UA Address Space")
st.caption("Every sensor tag exposed by the server. Node IDs are assigned at startup and remain stable for the process lifetime.")

nodes = status.get("nodes", [])
if nodes:
    df_nodes = pd.DataFrame(nodes)
    df_nodes["value"] = df_nodes["value"].apply(lambda v: round(float(v), 3) if v is not None else None)
    df_nodes["last_updated"] = pd.to_datetime(df_nodes["last_updated"]).dt.strftime("%H:%M:%S")
    df_nodes = df_nodes.rename(columns={
        "machine_id":   "Machine",
        "signal":       "Signal",
        "node_id":      "OPC-UA Node ID",
        "value":        "Current Value",
        "last_updated": "Last Updated",
    })

    # Pivot: machines as rows, signals as columns (value only)
    pivot = df_nodes.pivot(index="Machine", columns="Signal", values="Current Value").reset_index()

    st.dataframe(
        pivot,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🔖 Node ID Reference")
    st.caption("OPC-UA node identifiers for every machine tag — use these in any OPC-UA client (UaExpert, node-opcua, python-opcua).")
    node_ref = df_nodes[["Machine", "Signal", "OPC-UA Node ID", "Current Value", "Last Updated"]]
    st.dataframe(node_ref, use_container_width=True, hide_index=True)
else:
    st.info("No nodes discovered yet — the OPC-UA bridge is connecting…")

st.divider()

# ============================================================================
# PER-MACHINE LIVE VALUES
# ============================================================================

st.subheader("📡 Live Machine Tags")

if nodes:
    df_all = pd.DataFrame(nodes)
    machines = sorted(df_all["machine_id"].unique())
    cols = st.columns(len(machines))

    SIGNAL_UNITS = {
        "Temperature":      "°C",
        "Vibration":        "mm/s",
        "PowerConsumption": "kW",
        "ProductionCount":  "parts",
        "DowntimeMinutes":  "min",
        "QualityScore":     "%",
    }

    for col, machine_id in zip(cols, machines):
        col.markdown(f"**{machine_id}**")
        m_rows = df_all[df_all["machine_id"] == machine_id]
        for _, row in m_rows.iterrows():
            unit = SIGNAL_UNITS.get(row["signal"], "")
            val = round(float(row["value"]), 2) if row["value"] is not None else "—"
            col.metric(label=f"{row['signal']} ({unit})", value=val)

st.divider()

# ============================================================================
# HOW IT WORKS
# ============================================================================

with st.expander("ℹ️  How the OPC-UA integration works"):
    st.markdown("""
**Architecture (all in-process)**

```
FastAPI process
├── OPC-UA Server  (asyncua, opc.tcp://127.0.0.1:4840/manufacturing/)
│   └── update_loop()  — random-walk physics, writes node values every 5 s
│
└── OPC-UA Bridge Client  (asyncua subscription, 500 ms publish interval)
    └── DataChangeHandler  — batches 6 signals per machine, flushes to SQLite
```

**Address space**
- Namespace URI: `urn:manufacturing:dt`
- Objects → Machines → M1…M5 → Temperature, Vibration, PowerConsumption, ProductionCount, DowntimeMinutes, QualityScore

**Why OPC-UA?**
OPC-UA (IEC 62541) is the dominant machine-to-machine protocol in modern SCADA and IIoT systems. It provides a
typed, discoverable address space, pub/sub subscriptions, and optional security (X.509 certs + encrypted channels).
This integration uses **NoSecurity** on loopback — production deployments add `Basic256Sha256` certificates.

**Connecting your own OPC-UA client**
Any OPC-UA client can browse this server when running locally:
```
Endpoint: opc.tcp://127.0.0.1:4840/manufacturing/
Security: None
```
Tools: [UaExpert](https://www.unified-automation.com/products/development-tools/uaexpert.html), node-red (node-opcua), python asyncua.
""")

# Auto-refresh
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption(f"Last fetched: {datetime.now().strftime('%H:%M:%S')} · auto-refresh every 5 s with Streamlit TTL cache")
