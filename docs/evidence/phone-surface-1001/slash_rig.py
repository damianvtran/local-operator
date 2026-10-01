"""Drive every phone-sheet command through the REAL phone HTTP seam against a real runtime.

Real Session -> ServingSessionHandle -> RuntimeServer (loopback socket) <- MobileDaemon relay
(dial + request) <- httpx ASGI client on /api/sessions/<id>/command.  Run under env -i + isolated HOME.
"""
import asyncio, json, os, sys, tempfile
from pathlib import Path
ROOT = Path(os.environ["WT"])
sys.path.insert(0, str(ROOT))
iso = Path(os.environ["HOME"])
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(iso / ".local-operator")
os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
(iso / ".local-operator").mkdir(parents=True, exist_ok=True)
import httpx
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app, _dial
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

OPS = sys.argv[1].split(",")   # which ops to drive: slash,slash_result

async def main():
    d = iso / ".local-operator" / "sessions" / "rigsess00001"
    d.mkdir(parents=True)
    session = build_session(d, ScriptedStream([text_turn("ok")]*40))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(d))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    daemon = MobileDaemon(port=0, password="pw")
    try:
        record = None
        for _ in range(100):
            for rec, st in registry.scan(iso / ".local-operator"):
                if rec.session_id == session.session_id: record = rec
            if record: break
            await asyncio.sleep(0.05)
        entry = SessionEntry(record); daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        for _ in range(100):
            if entry.projection is not None: break
            await asyncio.sleep(0.05)
        transport = httpx.ASGITransport(app=build_app(daemon))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post("/login", data={"password": "pw"}); 
            cat = (await c.get("/api/commands")).json()["commands"]
            names = [x["name"] for x in cat]
            print(f"CATALOGUE offers {len(names)}")
            rows = {}
            for op in OPS:
                for x in cat:
                    name = x["name"]
                    args = {"goal": "ship the thing", "rename": "rig name", "model": "", "effort": "", "fast": "", "approvals": "ask", "mcp": "list", "context": "", "team": "", "agent": "", "compact": ""}.get(name, "")
                    try:
                        r = await asyncio.wait_for(c.post(f"/api/sessions/{session.session_id}/command",
                            json={"op": op, "command": name, "args": args, "images": []}), 20)
                        out = f"{r.status_code} {r.text[:110]}"
                    except Exception as e:
                        out = f"EXC {type(e).__name__}: {e}"
                    rows[(op, name)] = out
                    print(f"{op:12s} /{name:14s} {out}")
        dial.cancel()
    finally:
        server.close(); await session.dispose()
asyncio.run(main())
