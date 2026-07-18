"""
OPC-UA server exposing manufacturing plant sensor nodes.
Runs as a background asyncio task within the FastAPI process.

Address space layout:
  Objects/Machines/M1/Temperature, Vibration, PowerConsumption, ProductionCount, DowntimeMinutes, QualityScore
  Objects/Machines/M2/... (same)
  ... M3, M4, M5
"""

import asyncio
import logging
import numpy as np
from asyncua import Server, ua

# Suppress asyncua cert/security warnings — we use NoSecurity on loopback intentionally
logging.getLogger("asyncua").setLevel(logging.ERROR)

ENDPOINT = "opc.tcp://127.0.0.1:4840/manufacturing/"
NAMESPACE_URI = "urn:manufacturing:dt"
MACHINES = ["M1", "M2", "M3", "M4", "M5"]
SIGNALS = ["Temperature", "Vibration", "PowerConsumption", "ProductionCount", "DowntimeMinutes", "QualityScore"]

_MACHINE_BASELINES = {
    "M1": {"temp": 65.0, "vib": 2.5, "power": 15.0},
    "M2": {"temp": 65.0, "vib": 3.0, "power": 15.0},
    "M3": {"temp": 58.0, "vib": 2.2, "power": 14.0},
    "M4": {"temp": 70.0, "vib": 3.5, "power": 16.0},  # older machine
    "M5": {"temp": 62.0, "vib": 2.8, "power": 15.0},
}

# Event set when server is ready — lets the OPC-UA client know it can connect
server_ready = asyncio.Event()


class ManufacturingOPCServer:
    """OPC-UA server: 5 machines × 6 sensor variables, random-walk simulation."""

    def __init__(self):
        self.server = Server()
        # {machine_id: {signal_name: ua_node}}
        self.machine_nodes: dict[str, dict[str, object]] = {}
        # per-machine simulation state
        self._state = {
            m: {
                "temp":             float(b["temp"]  + np.random.normal(0, 2)),
                "vib":              float(b["vib"]   + np.random.uniform(0, 0.5)),
                "power":            float(b["power"] + np.random.normal(0, 1)),
                "production_count": 10,
                "downtime_minutes": 0.0,
                "quality_score":    93.0,
            }
            for m, b in _MACHINE_BASELINES.items()
        }

    async def init(self):
        await self.server.init()
        self.server.set_endpoint(ENDPOINT)
        self.server.set_server_name("Manufacturing Digital Twin OPC-UA Server")
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        idx = await self.server.register_namespace(NAMESPACE_URI)

        objects = self.server.nodes.objects
        machines_folder = await objects.add_folder(idx, "Machines")

        for machine_id in MACHINES:
            s = self._state[machine_id]
            machine_obj = await machines_folder.add_object(idx, machine_id)

            nodes = {
                "Temperature":      await machine_obj.add_variable(idx, "Temperature",     ua.Variant(round(s["temp"], 2),             ua.VariantType.Double)),
                "Vibration":        await machine_obj.add_variable(idx, "Vibration",        ua.Variant(round(s["vib"], 3),              ua.VariantType.Double)),
                "PowerConsumption": await machine_obj.add_variable(idx, "PowerConsumption", ua.Variant(round(s["power"], 2),            ua.VariantType.Double)),
                "ProductionCount":  await machine_obj.add_variable(idx, "ProductionCount",  ua.Variant(s["production_count"],           ua.VariantType.Int32)),
                "DowntimeMinutes":  await machine_obj.add_variable(idx, "DowntimeMinutes",  ua.Variant(round(s["downtime_minutes"], 1), ua.VariantType.Double)),
                "QualityScore":     await machine_obj.add_variable(idx, "QualityScore",     ua.Variant(round(s["quality_score"], 2),    ua.VariantType.Double)),
            }
            for node in nodes.values():
                await node.set_writable()
            self.machine_nodes[machine_id] = nodes

    async def start(self):
        await self.server.start()
        print(f"✅ OPC-UA server listening at {ENDPOINT}")
        server_ready.set()

    async def stop(self):
        await self.server.stop()
        print("🛑 OPC-UA server stopped")

    async def update_loop(self):
        """Random-walk: push new values to OPC-UA nodes every 5 seconds."""
        while True:
            await asyncio.sleep(5)
            try:
                for machine_id, s in self._state.items():
                    s["temp"]             = float(np.clip(s["temp"]  + np.random.normal(0, 1.5), 40, 95))
                    s["vib"]              = float(np.clip(s["vib"]   + np.random.normal(0, 0.3),  0.3, 10))
                    s["power"]            = float(np.clip(s["power"] + np.random.normal(0, 0.8),  5, 35))
                    s["production_count"] = int(np.random.randint(5, 20))
                    s["downtime_minutes"] = float(max(0.0, np.random.normal(0, 2)))
                    s["quality_score"]    = float(np.clip(np.random.normal(92, 5), 70, 100))

                    nodes = self.machine_nodes[machine_id]
                    await nodes["Temperature"].set_value(     ua.Variant(round(s["temp"], 2),             ua.VariantType.Double))
                    await nodes["Vibration"].set_value(       ua.Variant(round(s["vib"], 3),              ua.VariantType.Double))
                    await nodes["PowerConsumption"].set_value(ua.Variant(round(s["power"], 2),            ua.VariantType.Double))
                    await nodes["ProductionCount"].set_value( ua.Variant(s["production_count"],           ua.VariantType.Int32))
                    await nodes["DowntimeMinutes"].set_value( ua.Variant(round(s["downtime_minutes"], 1), ua.VariantType.Double))
                    await nodes["QualityScore"].set_value(    ua.Variant(round(s["quality_score"], 2),    ua.VariantType.Double))
            except Exception as exc:
                print(f"⚠️  OPC-UA server update error: {exc}")
