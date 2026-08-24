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

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
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

    async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
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


def test_spawn_dir_gate_allows_home_and_tmp_only(tmp_path, monkeypatch) -> None:
    """The phone may start a session anywhere under home or in the system temp
    dir (a common scratch root), and nowhere else — the gate guards against a
    fat-fingered/traversed path, not against the owner."""
    from pathlib import Path

    from local_operator.mobile import daemon as daemon_mod
    from local_operator.mobile.daemon import _spawn_dir_allowed

    # pytest's tmp_path already lives UNDER the real system temp dir, so pin an
    # explicit fake tmp and home under it and assert against those bounds —
    # otherwise "outside" would still be a child of the real /tmp and allowed.
    home = tmp_path / "home"
    home.mkdir()
    fake_tmp = tmp_path / "scratch"
    fake_tmp.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(daemon_mod, "_tmp_dir", lambda: str(fake_tmp.resolve()))

    # Under home: allowed. Home itself: allowed.
    sub = home / "projects"
    sub.mkdir()
    assert _spawn_dir_allowed(home.resolve())
    assert _spawn_dir_allowed(sub.resolve())

    # The (fake) temp dir and a child of it: allowed.
    assert _spawn_dir_allowed(fake_tmp.resolve())
    child = fake_tmp / "work"
    child.mkdir()
    assert _spawn_dir_allowed(child.resolve())

    # Somewhere neither under home nor tmp: refused.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert not _spawn_dir_allowed(outside.resolve())


def test_directories_endpoint_offers_tmp(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    body = client.get("/api/directories").json()
    assert "home" in body
    assert "recent" in body
    # /tmp is offered as a scratch start dir beside home.
    assert body.get("tmp")


def test_image_bytes_reads_attachment_from_transcript(tmp_path, monkeypatch) -> None:
    """The image endpoint's helper decodes the Nth image block of a message
    from the on-disk transcript — the lazy source the phone fetches pixels
    from. Index counts IMAGE blocks only, matching _image_refs."""
    import asyncio as _asyncio
    import base64

    from local_operator.harness.types import ImageContent, Message
    from local_operator.mobile.daemon import _image_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    # _image_bytes imports config_dir from local_operator.paths at call time,
    # so patching the source module is what redirects it to the fake config.
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    session_id = "sess-img"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    raw = b"\x89PNG\r\n\x1a\nHELLO"
    message = Message.user(
        "look",
        [ImageContent(data=base64.b64encode(raw).decode(), mime_type="image/png")],
    )
    transcript = Transcript(directory)
    _asyncio.run(transcript.append_message(message))

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    found = _image_bytes(record, message.id, 0)
    assert found is not None
    data, mime = found
    assert data == raw
    assert mime == "image/png"

    # Out-of-range index and unknown entry both miss cleanly.
    assert _image_bytes(record, message.id, 1) is None
    assert _image_bytes(record, "nope", 0) is None


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
