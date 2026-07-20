"""
OPC-UA bridge client — subscribes to the in-process OPC-UA server and
writes sensor readings into the database via the shared ingest function.

Node discovery is done by browsing: Objects/Machines/<M>/<Signal>
so node IDs are captured at runtime and exposed for the /opcua-status endpoint.
"""

import asyncio
from datetime import datetime
from asyncua import Client
from opcua_server import server_ready, ENDPOINT, MACHINES, SIGNALS

# Shared status dict — read by the /opcua-status FastAPI endpoint
opcua_status: dict = {
    "connected": False,
    "endpoint": ENDPOINT,
    "server_time": None,
    "nodes": [],          # [{machine_id, signal, node_id, value, last_updated}]
    "readings_received": 0,
}

_EXPECTED_SIGNALS = set(SIGNALS)


async def _browse_nodes(client) -> tuple[dict, list]:
    """
    Walk Objects → Machines → <machine> → <signal> and return:
      node_map : {ua.Node -> (machine_id, signal_name)}
      node_info: [{machine_id, signal, node_id, value, last_updated}]
    """
    node_map: dict = {}
    node_info: list = []

    objects = client.nodes.objects
    machines_folder = None
    for child in await objects.get_children():
        name = (await child.read_browse_name()).Name
        if name == "Machines":
            machines_folder = child
            break

    if machines_folder is None:
        raise RuntimeError("Machines folder not found in OPC-UA address space")

    for machine_node in await machines_folder.get_children():
        machine_id = (await machine_node.read_browse_name()).Name
        for var_node in await machine_node.get_children():
            signal = (await var_node.read_browse_name()).Name
            if signal not in _EXPECTED_SIGNALS:
                continue
            value = await var_node.read_value()
            node_map[var_node] = (machine_id, signal)
            node_info.append({
                "machine_id":   machine_id,
                "signal":       signal,
                "node_id":      var_node.nodeid.to_string(),
                "value":        value,
                "last_updated": datetime.now().isoformat(),
            })

    return node_map, node_info


class _DataChangeHandler:
    """
    Asyncua subscription handler — batches per-machine values and flushes
    to the database once all six signals for a machine have arrived.
    """

    def __init__(self, node_map: dict, ingest_fn, session_factory):
        self._node_map = node_map
        self._ingest_fn = ingest_fn          # sync: ingest_fn(db, machine_id, **kwargs)
        self._session_factory = session_factory
        self._buffers: dict[str, dict] = {m: {} for m in MACHINES}

    async def datachange_notification(self, node, val, data):
        if node not in self._node_map:
            return
        machine_id, signal = self._node_map[node]
        self._buffers[machine_id][signal] = val

        # Update live status value
        for entry in opcua_status["nodes"]:
            if entry["machine_id"] == machine_id and entry["signal"] == signal:
                entry["value"] = val
                entry["last_updated"] = datetime.now().isoformat()
                break

        # Flush once all 6 signals for this machine have arrived
        if len(self._buffers[machine_id]) >= 6:
            buf = self._buffers[machine_id].copy()
            self._buffers[machine_id] = {}
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._flush_sync, machine_id, buf)

    def _flush_sync(self, machine_id: str, buf: dict):
        db = self._session_factory()
        try:
            self._ingest_fn(
                db=db,
                machine_id=machine_id,
                plant_id="alpha",
                temperature=float(buf.get("Temperature", 65.0)),
                vibration=float(buf.get("Vibration", 2.5)),
                power_consumption=float(buf.get("PowerConsumption", 15.0)),
                production_count=int(buf.get("ProductionCount", 10)),
                downtime_minutes=float(buf.get("DowntimeMinutes", 0.0)),
                quality_score=float(buf.get("QualityScore", 93.0)),
            )
            opcua_status["readings_received"] += 1
        except Exception as exc:
            print(f"⚠️  OPC-UA DB flush error ({machine_id}): {exc}")
        finally:
            db.close()


async def run_opcua_bridge(ingest_fn, session_factory):
    """
    Long-running coroutine: wait for server, subscribe to all machine nodes,
    reconnect automatically on disconnect. Call as an asyncio.Task.
    """
    print("🔌 OPC-UA bridge: waiting for server...")
    await server_ready.wait()

    while True:
        try:
            async with Client(ENDPOINT) as client:
                opcua_status["connected"] = True
                print("✅ OPC-UA bridge connected")

                node_map, node_info = await _browse_nodes(client)
                opcua_status["nodes"] = node_info

                handler = _DataChangeHandler(node_map, ingest_fn, session_factory)
                sub = await client.create_subscription(500, handler)  # 500 ms publish interval
                await sub.subscribe_data_change(list(node_map.keys()))
                print(f"📡 Subscribed to {len(node_map)} OPC-UA nodes")

                # Block until disconnected
                while True:
                    opcua_status["server_time"] = datetime.now().isoformat()
                    await asyncio.sleep(10)

        except Exception as exc:
            opcua_status["connected"] = False
            print(f"⚠️  OPC-UA bridge error: {exc} — retrying in 5 s")
            await asyncio.sleep(5)
