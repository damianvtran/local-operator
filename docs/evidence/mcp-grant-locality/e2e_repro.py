"""End-to-end proof over the REAL control socket.

Stands up a live ``RuntimeServer``, dials it the way an attach terminal does
(loopback + the record's control key), and sends the exact ``slash_result``
frame a typed ``/mcp reauth notion`` produces. Runs the same frame against a
client that declares itself remote, so both sides of the locality decision are
exercised on the wire rather than through a direct function call.

No real MCP server and no real credential store are touched: the session's
manager is a double that records what the grant did. The point under test is
the DISPATCH — which path a routed grant verb takes, and what the invoker gets
back — not the OAuth exchange itself.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.session.runtime import registry  # noqa: E402
from local_operator.session.runtime.owned import OwnedSessionHandle  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402


class Cfg:
    auth = None
    url = "https://mcp.notion.com/mcp"


class Manager:
    """Records the grant's side effects instead of performing them."""

    def __init__(self) -> None:
        self.connected: list[str] = []
        self.disconnected: list[str] = []

    def get_server_config(self, name: str) -> Any:
        return Cfg()

    async def server_supports_oauth_login(self, cfg: Any) -> bool:
        return True

    async def disconnect_server(self, name: str) -> None:
        self.disconnected.append(name)

    async def connect_configured_server(self, name: str, *, timeout_ms: Any = None) -> Any:
        self.connected.append(name)
        return type("Conn", (), {"tools": [1, 2, 3, 4, 5]})()


class Session:
    """The slice of Session the runtime's /mcp handler reads."""

    def __init__(self) -> None:
        self.session_id = "evidence-1"
        self.model_label = "test/model"
        self.effective_model_label = "test/model"
        self.model = None
        self.conversation_name = "MCP grant evidence"
        self.is_streaming = False
        self.mcp_manager = Manager()
        self.notices: list[tuple[str, str]] = []
        from local_operator.harness.jobs import AsyncJobManager

        self.jobs = AsyncJobManager()

    # The seams OwnedSessionHandle installs at construction. None of them
    # matter to a slash command; they exist so the handle can be built.
    def set_approval_handler(self, handler: Any) -> None:
        return None

    def set_ask_handler(self, handler: Any) -> None:
        return None

    def subscribe(self, handler: Any) -> Any:
        return lambda: None

    def subscribe_admitted_commands(self, handler: Any) -> Any:
        return lambda: None

    def subscribe_rejected_steering(self, handler: Any) -> Any:
        return lambda: None

    async def _emit(self, event: Any) -> None:
        self.notices.append((getattr(event, "text", ""), getattr(event, "kind", "")))


async def dial(record: Any, locality: str | None) -> tuple[Any, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    auth: dict[str, Any] = {"key": record.control_key, "client": "attach"}
    if locality is not None:
        auth["locality"] = locality
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    await asyncio.wait_for(reader.readline(), timeout=5)  # welcome projection
    return reader, writer


async def send_slash(reader: Any, writer: Any, req: int, args: str) -> dict[str, Any]:
    frame = {"op": "slash_result", "req": req, "command": "mcp", "args": args}
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()
    for _ in range(40):
        raw = await asyncio.wait_for(reader.readline(), timeout=10)
        data = json.loads(raw.decode("utf-8", "replace"))
        if data.get("op") in ("result", "error") and data.get("req") == req:
            return data
    raise AssertionError("no reply")


async def main() -> int:
    # NEVER touch the developer's real ``auth.db``. ``reauth`` deletes the
    # stored grant before reconnecting, and the helper it calls resolves the
    # server name against the REAL config and writes the REAL credential
    # store, so an unpatched run of this script removes a live credential as a
    # side effect of gathering evidence. Patched at the definition site
    # because ``grants.py`` imports it lazily inside the call.
    import local_operator.mcp.auth as _auth

    forgotten: list[str] = []

    def _fake_logout(name: str, cwd: str) -> str | None:
        forgotten.append(name)
        return None

    _auth.mcp_logout_server = _fake_logout  # type: ignore[assignment]

    session = Session()
    loop = asyncio.get_running_loop()
    handle = OwnedSessionHandle(session, loop, cwd="/tmp")
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    failures = 0
    try:
        record = None
        for _ in range(100):
            await asyncio.sleep(0.05)
            # scan() yields (record, state) for EVERY live session on this
            # machine, so filter to the one this script published rather than
            # dialing a colleague's session.
            found = [
                rec
                for rec, state in registry.scan()
                if rec.session_id == "evidence-1" and state == "live"
            ]
            if found and found[0].control_port:
                record = found[0]
                break
        assert record is not None, "runtime never published a record"
        print(f"runtime listening on 127.0.0.1:{record.control_port}\n")

        # --- 1. a local attach client: the reported bug -----------------------
        reader, writer = await dial(record, locality=None)
        reply = await send_slash(reader, writer, 1, "reauth notion")
        text = reply["data"]["text"]
        print("[1] LOCAL client, `/mcp reauth notion`")
        print(f"    reply : {text}")
        print(f"    style : {reply['data']['style']}")
        if "run it from a terminal on that machine" in text:
            print("    FAIL  : still refusing a user on the machine that stores creds")
            failures += 1
        elif "authorizing MCP server 'notion'" not in text:
            print("    FAIL  : unexpected receipt")
            failures += 1
        else:
            print("    PASS  : the grant was started")

        # The receipt returns immediately; the exchange settles behind it.
        for _ in range(40):
            await asyncio.sleep(0.02)
            if session.mcp_manager.connected:
                break
        print(
            f"    effect: forgot={forgotten} "
            f"disconnected={session.mcp_manager.disconnected} "
            f"connected={session.mcp_manager.connected}"
        )
        if forgotten != ["notion"]:
            print("    FAIL  : reauth did not forget the stored grant first")
            failures += 1
        if session.mcp_manager.connected != ["notion"]:
            print("    FAIL  : the grant never reached the manager")
            failures += 1
        else:
            print("    PASS  : reauth forgot the old grant, then reconnected")
        print(f"    notice: {session.notices}")
        if not any("authenticated MCP server 'notion'" in n[0] for n in session.notices):
            print("    FAIL  : settled outcome never reached viewers")
            failures += 1
        else:
            print("    PASS  : outcome delivered as a NoticeEvent\n")
        writer.close()

        # --- 2. a relayed remote client: refusal preserved --------------------
        session.mcp_manager = Manager()
        reader, writer = await dial(record, locality="remote")
        reply = await send_slash(reader, writer, 2, "reauth notion")
        text = reply["data"]["text"]
        print("[2] REMOTE client (a future phone relay), same command")
        print(f"    reply : {text}")
        await asyncio.sleep(0.2)
        print(
            f"    effect: disconnected={session.mcp_manager.disconnected} "
            f"connected={session.mcp_manager.connected}"
        )
        if "run it from a terminal on that machine" not in text:
            print("    FAIL  : a remote device would open a browser nobody sees")
            failures += 1
        elif session.mcp_manager.connected or session.mcp_manager.disconnected:
            print("    FAIL  : refused but still touched the credential")
            failures += 1
        else:
            print("    PASS  : refused, and nothing was touched\n")
        writer.close()

        # --- 3. arity refusals still answer the typed string ------------------
        session.mcp_manager = Manager()
        reader, writer = await dial(record, locality=None)
        for req, args, want in (
            (3, "reauth", "usage: /mcp reauth <name>"),
            (4, "reauth a b", "takes one server name"),
            (5, "bogus x", "unknown mcp subcommand"),
        ):
            reply = await send_slash(reader, writer, req, args)
            got = reply["data"]["text"]
            ok = want in got
            print(f"[3] `/mcp {args}` -> {got}")
            if not ok:
                print(f"    FAIL  : expected {want!r}")
                failures += 1
        if session.mcp_manager.connected:
            print("    FAIL  : a malformed command still ran a grant")
            failures += 1
        else:
            print("    PASS  : malformed commands ran nothing\n")
        writer.close()

        # --- 4. the phone daemon dials as remote (review F1) ------------------
        # Byte-identical to mobile/daemon.py::_dial. Before the fix it sent
        # {"key": ...} alone, was classified local, and a phone's /mcp reauth
        # opened a browser on this desktop.
        session.mcp_manager = Manager()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        writer.write(json.dumps({"key": record.control_key, "locality": "remote"}).encode() + b"\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=5)
        reply = await send_slash(reader, writer, 6, "reauth notion")
        print(f"[4] DAEMON-class dial -> {reply['data']['text']}")
        await asyncio.sleep(0.2)
        if "run it from a terminal on that machine" not in reply["data"]["text"]:
            print("    FAIL  : the phone relay would open a browser on this desktop")
            failures += 1
        elif session.mcp_manager.connected or session.mcp_manager.disconnected:
            print("    FAIL  : refused but still touched the credential")
            failures += 1
        else:
            print("    PASS  : the relay is refused, and nothing was touched\n")
        writer.close()

        # --- 5. a slow capability probe must not hold the reader (F2/Q1) ------
        # The probe is up to three sequential 10s HTTP GETs. Awaited inside the
        # request it blew past ACK_TIMEOUT_S (15s) and parked every other op on
        # the connection behind it.
        class SlowProbe(Manager):
            async def server_supports_oauth_login(self, cfg: Any) -> bool:
                await asyncio.sleep(3600)
                return True

        session.mcp_manager = SlowProbe()
        reader, writer = await dial(record, locality=None)
        started = asyncio.get_running_loop().time()
        reply = await send_slash(reader, writer, 7, "login notion")
        elapsed = asyncio.get_running_loop().time() - started
        print("[5] `/mcp login` with an unreachable probe host")
        print(f"    receipt after {elapsed:.2f}s (ACK_TIMEOUT_S is 15s)")
        if elapsed > 5:
            print("    FAIL  : the probe is still blocking the request")
            failures += 1
        else:
            print("    PASS  : the receipt returned immediately")
        # And the connection is still usable while that grant hangs.
        started = asyncio.get_running_loop().time()
        await send_slash(reader, writer, 8, "list")
        elapsed = asyncio.get_running_loop().time() - started
        print(f"    a second op on the SAME connection answered in {elapsed:.2f}s")
        if elapsed > 5:
            print("    FAIL  : the serial reader is parked behind the grant")
            failures += 1
        else:
            print("    PASS  : the connection stayed responsive\n")
        writer.close()

        # --- 6. a superseded reauth must say the credential is gone (F6) ------
        # reauth is destructive before it is constructive. Cancelled between
        # the delete and the reconnect, the server has NO credential, and a
        # bare "cancelled" reads as "nothing changed".
        class Parked(Manager):
            def __init__(self) -> None:
                super().__init__()
                self.gate = asyncio.Event()

            async def connect_configured_server(self, name: str, *, timeout_ms: Any = None) -> Any:
                await self.gate.wait()
                return await super().connect_configured_server(name, timeout_ms=timeout_ms)

        session.mcp_manager = Parked()
        session.notices.clear()
        forgotten.clear()
        reader, writer = await dial(record, locality=None)
        await send_slash(reader, writer, 10, "reauth notion")
        await asyncio.sleep(0.1)
        # A second grant supersedes the first, cancelling it mid-flight.
        await send_slash(reader, writer, 11, "login other")
        await asyncio.sleep(0.2)
        print("[6] `/mcp reauth notion` superseded by `/mcp login other`")
        print(f"    forgot : {forgotten}")
        print(f"    notice : {[n[0] for n in session.notices]}")
        # Scenario 5 left a `/mcp login notion` parked on its probe; this
        # supersede cancels that one too, so filter to the REAUTH notice — the
        # one whose delete already happened.
        cancel_notes = [n[0] for n in session.notices if n[0].startswith("MCP reauth")]
        if not cancel_notes:
            print("    FAIL  : the superseded grant reported nothing")
            failures += 1
        elif "unauthenticated" not in cancel_notes[0]:
            print("    FAIL  : the user is not told the credential is gone")
            failures += 1
        else:
            print("    PASS  : the loss is named, with the recovery step\n")
        writer.close()
    finally:
        runtime.close()

    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
