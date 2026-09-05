"""The attach client against a real registrant: discovery, gating, identity,
correlation, and the no-reconnect contract."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.mobile.attach_client import AttachClient, find_owner_record
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer


class FakeHandle:
    def __init__(self, session_id: str = "sess-a") -> None:
        self._projection = SessionProjection(
            session_id=session_id,
            pid=0,
            kind="tui",
            conversation_name="owner chat",
            cwd="/tmp",
            model_label="test/model",
        )

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    async def refresh(self) -> None:
        pass

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        self._projection.transcript.append(
            TranscriptEntry(id=f"u{len(self._projection.transcript)}", kind="user", text=text)
        )
        return "prompt sent"

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        return "steering queued"

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        raise ValueError(f"/{command} is terminal-only here")

    # The rest of the SessionHandle surface the registrant's protocol demands;
    # these tests never drive them, but the protocol check is structural.
    async def abort(self):  # noqa: ANN202
        return "stopping"

    async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
        return "model"

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return "effort"

    async def new_conversation(self):  # noqa: ANN202
        raise ValueError("not here")

    async def resume_session(self, session_id):  # noqa: ANN001, ANN202
        raise ValueError("not here")

    async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
        return "done"

    async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
        return "done"


async def _wait_record() -> registry.SessionRecord:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan()
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.05)
    raise AssertionError("no live record")


def _marker(config: Path, session_id: str, pid: int) -> None:
    d = config / "sessions" / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / ".session.pid").write_text(str(pid))


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    return cfg


@pytest.mark.asyncio
async def test_discovery_finds_the_live_owner(config: Path) -> None:
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        _marker(config, "sess-a", os.getpid())
        found, owner = find_owner_record(config, "sess-a")
        assert found is not None
        assert found.pid == record.pid
        assert owner == os.getpid()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_discovery_without_owner_returns_none(config: Path) -> None:
    found, owner = find_owner_record(config, "never-started")
    assert found is None
    assert owner is None


@pytest.mark.asyncio
async def test_discovery_owner_without_record_reports_pid_only(config: Path) -> None:
    # A live pid holds the claim but publishes nothing (old binary,
    # registrant failed): the caller needs the pid for the refusal copy.
    _marker(config, "sess-x", os.getpid())
    found, owner = find_owner_record(config, "sess-x")
    assert found is None
    assert owner == os.getpid()


@pytest.mark.asyncio
async def test_protocol_gate_refuses_v1_records(config: Path) -> None:
    handle = FakeHandle("sess-old")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        _marker(config, "sess-old", os.getpid())
        # Degrade the published record to protocol 1, as an old binary would.
        record.protocol = 1
        registry.publish(record, root=config)
        found, owner = find_owner_record(config, "sess-old")
        # The gate reports the owner (refusal copy needs the pid) but no
        # dialable record.
        assert found is None
        assert owner == os.getpid()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_connect_rejects_protocol_one_before_dialing(config: Path) -> None:
    record = registry.SessionRecord(
        pid=1,
        kind="tui",
        session_id="s",
        conversation_name="",
        cwd="/tmp",
        model_label="",
        control_port=1,
        control_key="k",
        protocol=1,
    )
    client = AttachClient(lambda p: None, lambda reason: None)
    with pytest.raises(ConnectionError):
        await client.connect(record, "s")


@pytest.mark.asyncio
async def test_welcome_identity_mismatch_is_a_connection_error(config: Path) -> None:
    # The owner is hosting sess-a; the user asked for sess-b (a rebind raced
    # the heartbeat). The welcome projection must arbitrate against attaching.
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        with pytest.raises(ConnectionError) as excinfo:
            await client.connect(record, "sess-b")
        assert "another conversation" in str(excinfo.value)
    finally:
        r.close()


@pytest.mark.asyncio
async def test_prompt_ack_and_repaint_flow(config: Path) -> None:
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        projections: list[SessionProjection] = []
        disconnected: list[str] = []
        client = AttachClient(projections.append, disconnected.append)
        await client.connect(record, "sess-a")
        assert projections and projections[0].session_id == "sess-a"
        detail = await client.prompt("hello owner")
        assert detail == "prompt sent"
        # The broadcast repaint carries the user row the owner folded.
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if any(e.kind == "user" for e in projections[-1].transcript):
                break
            await asyncio.sleep(0.05)
        assert any(e.kind == "user" and e.text == "hello owner" for e in projections[-1].transcript)
        await client.detach()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_owner_death_fires_on_disconnected_once(config: Path) -> None:
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        disconnected: list[str] = []
        client = AttachClient(lambda p: None, disconnected.append)
        await client.connect(record, "sess-a")
        # Kill the owner through its public synchronous seam. This call runs
        # on the attach client's loop, not the registrant thread, preserving
        # the hard socket-death path while exercising the cross-loop boundary.
        r.close()
        r.close()  # repeated host teardown must remain a no-op
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if disconnected:
                break
            await asyncio.sleep(0.05)
        assert len(disconnected) == 1
        assert not client.connected
    finally:
        r.close()


@pytest.mark.asyncio
async def test_request_error_raises_runtime_error(config: Path) -> None:
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        await client.connect(record, "sess-a")
        with pytest.raises(RuntimeError) as excinfo:
            # An op the owned-handle rejects for every caller: the error
            # frame must surface as RuntimeError, not a silent None.
            await client.slash("nonexistent", "")
        assert "terminal-only" in str(excinfo.value)
        await client.detach()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_a_refused_frame_is_not_reported_as_an_oversized_one(config: Path) -> None:
    """The pump's disconnect reason names what actually happened.

    ``StreamReader.readline`` raises ``ValueError`` on a line overrun, and the
    pump's ``except ValueError`` was written for that. But the frame callbacks
    raise ``ValueError`` too — a follower store refusing an update as "not the
    next state sequence", pydantic validating a frame — and every one of those
    was logged as "owner sent a frame larger than the 1048576-byte line
    limit" for a frame of a few hundred bytes (#573's viewer, whose store was
    still the cold one because the sync had been refused). Only the READ can
    be oversized; a callback failure carries its own reason.
    """
    from local_operator.mobile.attach_client import OVERSIZED_FRAME_REASON
    from tests.unit.session.runtime.test_server import FakeHandle as RelayHandle

    # The relay-capable fake (session_id "s1"): this test needs an owner that
    # actually pushes an event frame at the client.
    handle = RelayHandle()
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        disconnected: list[str] = []

        def refuse(_data):  # noqa: ANN001, ANN202
            raise ValueError("frontend update is not the next state sequence")

        # The real shape: the sync installs, then the follower's store refuses
        # the next ordered update (a viewer left at the wrong epoch).
        client = AttachClient(
            lambda p: None,
            disconnected.append,
            frontend_state=True,
            on_frontend_sync=lambda _data: None,
            on_frontend_update=refuse,
        )
        await client.connect(record, "s1")
        handle._frontend.mutate(goal="anything that publishes an update")
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline and not disconnected:
            await asyncio.sleep(0.05)
        assert len(disconnected) == 1
        assert disconnected[0] != OVERSIZED_FRAME_REASON
        assert "not the next state sequence" in disconnected[0]
        assert not client.connected
    finally:
        r.close()


@pytest.mark.asyncio
async def test_abandon_closes_without_reporting_owner_loss(config: Path) -> None:
    """``abandon`` is the close for a host that is abandoning a connection it
    judged unusable and intends to keep running: the pump's ``finally`` must
    not report it as a disconnect, or the host's recovery loop would redial
    the very runtime it just gave up on."""
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        disconnected: list[str] = []
        client = AttachClient(lambda p: None, disconnected.append)
        await client.connect(record, "sess-a")
        client.abandon()
        await asyncio.sleep(0.2)
        assert disconnected == []
        assert not client.connected
        for _ in range(50):
            if r.attach_clients() == 0:
                break
            await asyncio.sleep(0.02)
        assert r.attach_clients() == 0
    finally:
        r.close()
