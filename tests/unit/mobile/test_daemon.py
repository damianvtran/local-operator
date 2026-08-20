"""End-to-end over the real sockets: a fake registrant publishes a record,
the daemon adopts it, projections flow, control requests round-trip, and the
HTTP gate holds. These tests use the repo's config-dir isolation (conftest
sets LOCAL_OPERATOR_CONFIG_DIR to a tmp path)."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import registry
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, _dial, build_app
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import SessionProjection


class FakeHandle:
    """The minimal SessionHandle: static projection, echo answers."""

    def __init__(self) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake session",
            cwd="/tmp",
            model_label="anthropic/claude-opus-5",
        )
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    async def _record(self, name: str, *args, **kwargs) -> str:  # noqa: ANN202
        self.calls.append((name, args, kwargs))
        return f"{name} ok"

    async def prompt(self, text):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text):  # noqa: ANN001, ANN202
        return await self._record("steer", text)

    async def abort(self):  # noqa: ANN202
        return await self._record("abort")

    async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
        return await self._record("set_model", provider, model_id)

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return await self._record("set_effort", effort)

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        return await self._record("slash", command, args)

    async def new_conversation(self):  # noqa: ANN202
        return await self._record("new_conversation")

    async def resume_session(self, session_id):  # noqa: ANN001, ANN202
        return await self._record("resume_session", session_id)

    async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
        return await self._record("approval_answer", request_id, approved, remember)

    async def ask_answer(self, request_id, value):  # noqa: ANN001, ANN202
        return await self._record("ask_answer", request_id, value)

    async def refresh(self) -> None:
        pass


@pytest.mark.asyncio
async def test_registrant_publishes_and_daemon_adopts() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        # Wait for the record to appear.
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = registry.scan()
            if found:
                record, state = found[0]
                if state == "live":
                    break
            await asyncio.sleep(0.1)
        assert record is not None
        assert record.control_port > 0

        daemon = MobileDaemon(port=0, password="pw")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            for _ in range(50):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.1)
            assert entry.projection is not None
            assert entry.projection.conversation_name == "fake session"

            reply = await daemon.request(record.pid, "prompt", text="hello")
            assert reply["op"] == "ack"
            assert handle.calls[-1][0] == "prompt"

            reply = await daemon.request(record.pid, "set_effort", effort="high")
            assert "set_effort ok" in reply["detail"]
        finally:
            dial.cancel()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_wrong_key_is_rejected_silently() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = registry.scan()
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.1)
        assert record is not None

        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": "wrong"}).encode() + b"\n")
        await writer.drain()
        # The registrant closes without a reply: reading yields EOF.
        data = await asyncio.wait_for(reader.read(), timeout=5)
        assert data == b""
        writer.close()
    finally:
        registrant.close()


def test_http_gate_and_login_flow() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)

    assert client.get("/healthz").status_code == 200
    assert client.get("/api/sessions").status_code == 401
    root = client.get("/")
    assert root.status_code == 303
    assert root.headers["location"] == "/login"

    bad = client.post("/login", data={"password": "nope"})
    assert bad.status_code == 401

    good = client.post("/login", data={"password": "pw123"})
    assert good.status_code == 303
    assert "lop_mobile" in good.headers["set-cookie"]

    authed = client.get("/api/sessions")
    assert authed.status_code == 200
    assert authed.json() == {"sessions": []}


def test_unknown_session_command_is_a_409() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    reply = client.post("/api/sessions/424242/command", json={"op": "abort"})
    assert reply.status_code == 409


def test_slash_catalogue_excludes_terminal_chrome() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    names = [c["name"] for c in daemon.slash_commands()]
    assert "model" in names
    assert "effort" in names
    assert "resume" in names
    # TUI chrome never leaves the terminal.
    assert "exit" not in names
    assert "quit" not in names
    assert "clear" not in names
