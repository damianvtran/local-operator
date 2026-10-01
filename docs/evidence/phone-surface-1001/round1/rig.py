"""Round-1 remediation evidence: real runtime -> relay -> the SHIPPED app.

One rig: a real Session (scripted provider) behind ServingSessionHandle and
RuntimeServer on a loopback socket, dialled by a real MobileDaemon whose app is
served from the HEAD's own built bundle to headless Chrome. Prints the HTTP
receipts for the listing commands (the #1869 fold: `/team`, `/mcp`, `/agent`) and
captures rendered frames for the container tone, the wide-view control and the
ended-session copy.
"""
from __future__ import annotations

import asyncio, json, os, socket, sys, time, urllib.request
from pathlib import Path

WT = os.environ["WT"]; SP = os.environ["SP"]; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, WT)
import scripts.probe_isolation  # noqa: F401  (re-homes HOME/config, strips CMUX_*)

import httpx, uvicorn
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app, _dial
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.teams import TeamEditFields, TeamRegistry
from tests.e2e.harness import ScriptedStream, build_session, text_turn
from scripts.mobile_overflow_capture import Chrome, Page


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0)); return int(probe.getsockname()[1])


class NameManager:
    """The MCP listing's one seam: a manager that names its servers."""

    def get_all_server_names(self):  # noqa: ANN201
        return ["alpha-stdio", "beta-oauth"]


async def build():
    iso = config_dir()
    directory = iso / "sessions" / "r1evidence01"
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, ScriptedStream([text_turn("ok")] * 40))
    teams = TeamRegistry(iso)
    teams.create_team(
        TeamEditFields(
            name="release", label="Release", members=[{"role": "coder"}, {"role": "reviewer"}]
        )
    )
    session.team_registry = teams
    session.mcp_manager = NameManager()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    daemon = MobileDaemon(port=0, password="pw")
    record = None
    for _ in range(200):
        for rec, _state in registry.scan(iso):
            if rec.session_id == session.session_id:
                record = rec
        if record:
            break
        await asyncio.sleep(0.05)
    assert record is not None, "the runtime published no record"
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    dial = asyncio.ensure_future(_dial(daemon, entry))
    for _ in range(200):
        if entry.projection is not None:
            break
        await asyncio.sleep(0.05)
    return session, server, daemon, entry, dial, record


def browser_phase(port: int, session_id: str) -> dict:
    """SYNC, and run in a worker thread: the page fetches from the uvicorn server
    on the caller's loop, so a blocking browser block there starves it."""
    report: dict = {}
    base = f"http://127.0.0.1:{port}"
    chrome = None
    page = None
    try:
        chrome = Chrome()
        page = Page(chrome.target_ws())
        page.metrics(390, 844)
        page.goto(f"{base}/login")
        page.js(
            "(()=>{const i=document.querySelector('input[type=password]');"
            "i.value='pw';i.form.submit();})()"
        )
        time.sleep(2.5)

        def probe_notice() -> dict:
            return json.loads(
                page.js(
                    "(()=>{const n=document.querySelector('[role=status]');"
                    "if(!n)return JSON.stringify(null);const b=n.parentElement;const s=getComputedStyle(b);"
                    "return JSON.stringify({text:n.textContent,cls:b.className,bg:s.backgroundColor,fg:s.color,"
                    "draft:(document.querySelector('textarea')||{}).value});})()"
                )
                or "null"
            )

        def type_and_send(text: str) -> None:
            page.js(
                "(()=>{const t=document.querySelector('textarea');"
                "const set=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;"
                f"set.call(t,{json.dumps(text)});"
                "t.dispatchEvent(new Event('input',{bubbles:true}));})()"
            )
            time.sleep(0.5)
            page.js("document.querySelector('button[aria-label=\"send\"]').click()")
            time.sleep(3.0)

        def open_session() -> None:
            page.goto(f"{base}/#/"); time.sleep(2.0)
            page.js("localStorage.clear()")
            page.goto(f"{base}/#/s/{session_id}"); time.sleep(3.5)

        def probe_wide() -> dict:
            return json.loads(
                page.js(
                    "(()=>{const b=document.querySelector('button[aria-label=\"wide view\"]');"
                    "if(!b)return JSON.stringify(null);"
                    "return JSON.stringify({pressed:b.getAttribute('aria-pressed'),text:b.textContent,"
                    "cls:b.className,inner:innerWidth,stored:localStorage.getItem('lo-mobile-wide-view'),"
                    "meta:(document.querySelector('meta[name=viewport]')||{}).content});})()"
                )
                or "null"
            )

        open_session()
        type_and_send("/effort")
        report["refusal_receipt"] = probe_notice()
        page.shot(OUT / "r1-refusal-receipt.png")
        # A 200-REFUSAL THAT KEEPS THE DRAFT (UX round 1, U3): a bad argument is
        # answered with a usage line on the 200 path, and the whole command used to
        # be cleared there while the 422 path kept it.
        type_and_send("/model nonexistent-model-xyz")
        report["refused_argument"] = probe_notice()
        page.shot(OUT / "r1-refused-argument.png")
        open_session()
        type_and_send("/goal ship the phone fixes")
        report["success_receipt"] = probe_notice()
        page.shot(OUT / "r1-success-receipt.png")
        type_and_send("/team")
        report["team_receipt"] = probe_notice()
        page.shot(OUT / "r1-team-receipt.png")

        open_session()
        report["session_control_off"] = probe_wide()
        page.shot(OUT / "r1-session-wide-off.png")
        page.js("document.querySelector('button[aria-label=\"wide view\"]').click()")
        time.sleep(2.5)
        report["session_control_on"] = probe_wide()
        page.shot(OUT / "r1-session-wide-on.png")
        page.goto(f"{base}/#/"); time.sleep(2.5)
        report["list_control"] = probe_wide()
        page.shot(OUT / "r1-list-wide-on.png")
    finally:
        try:
            if page:
                page.close()
        except Exception:
            pass
        if chrome:
            chrome.close()
    return report


async def main():
    print("building the rig", flush=True)
    session, server, daemon, _entry, dial, _record = await build()
    print("rig built", flush=True)
    port = free_port()
    served = uvicorn.Server(
        uvicorn.Config(build_app(daemon), host="127.0.0.1", port=port, log_level="warning")
    )
    server_task = asyncio.ensure_future(served.serve())
    # ``served.started`` rather than a blocking urlopen probe: the probe would park
    # the very loop this server needs, and a server that failed to bind would look
    # exactly like a slow one.
    for _ in range(150):
        if served.started:
            break
        await asyncio.sleep(0.1)
    assert served.started, "uvicorn never started"
    print("uvicorn up", flush=True)

    report: dict = {}
    transport = httpx.ASGITransport(app=build_app(daemon))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/login", data={"password": "pw"})
        for command, args in (("team", ""), ("mcp", ""), ("agent", ""), ("context", ""),
                              ("goal", "ship it"), ("effort", "")):
            reply = await client.post(
                f"/api/sessions/{session.session_id}/command",
                json={"op": "slash_result", "command": command, "args": args, "images": []},
            )
            report[f"receipt:{command}"] = {"status": reply.status_code, **reply.json()}
    print(json.dumps(report, indent=1), flush=True)

    report.update(await asyncio.to_thread(browser_phase, port, session.session_id))
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    dial.cancel(); server_task.cancel(); server.close(); await session.dispose()


asyncio.run(main())
