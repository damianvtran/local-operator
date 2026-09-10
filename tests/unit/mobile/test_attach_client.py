"""The attach client against a real registrant: discovery, gating, identity,
correlation, and the no-reconnect contract."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from pathlib import Path

import pytest

from local_operator.mobile.attach_client import (
    ACK_TIMEOUT_S,
    ASIDE_DEADLINE_S,
    AttachClient,
    OwnerAckTimeout,
    find_runtime_record,
)
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.providers.clients import STREAM_READ_TIMEOUT_S
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
        found, owner = find_runtime_record(config, "sess-a")
        assert found is not None
        assert found.pid == record.pid
        assert owner == os.getpid()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_discovery_without_owner_returns_none(config: Path) -> None:
    found, owner = find_runtime_record(config, "never-started")
    assert found is None
    assert owner is None


@pytest.mark.asyncio
async def test_discovery_owner_without_record_reports_pid_only(config: Path) -> None:
    # A live pid holds the claim but publishes nothing (old binary,
    # registrant failed): the caller needs the pid for the refusal copy.
    _marker(config, "sess-x", os.getpid())
    found, owner = find_runtime_record(config, "sess-x")
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
        found, owner = find_runtime_record(config, "sess-old")
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
        synced = asyncio.Event()
        client = AttachClient(
            lambda p: None,
            disconnected.append,
            frontend_state=True,
            on_frontend_sync=lambda _data: synced.set(),
            on_frontend_update=refuse,
        )
        await client.connect(record, "s1")
        # ``connect()`` returns at the WELCOME frame, but the server subscribes
        # this connection to the frontend only afterwards. A mutation landing in
        # that gap is folded into the sync snapshot instead of being relayed as
        # an update, so ``refuse`` never fires and no amount of waiting on the
        # deadline below can produce the frame. Gate on the sync itself.
        await asyncio.wait_for(synced.wait(), timeout=5)
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


@pytest.mark.asyncio
async def test_a_connection_reset_keeps_the_reset_reason_not_the_generic_one(
    config: Path,
) -> None:
    """Locks the pump's clause ORDER, which is load-bearing and invisible.

    ``ConnectionResetError`` and ``BrokenPipeError`` are ``ConnectionError``
    subclasses, and ``ConnectionError`` is an ``OSError``. So the pump's three
    handlers are ordered reset -> ConnectionError -> OSError, and every one of
    them is reachable only because of that order: hoisting the bare
    ``ConnectionError`` clause above the reset clause would silently retag a
    dead transport as "a frame this client refused", which is the opposite
    diagnosis — the host redials for a fresh snapshot instead of reporting the
    owner as gone. Nothing about the source says that, so this pins it
    (review round 1, F4).
    """
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        disconnected: list[str] = []
        client = AttachClient(lambda p: None, disconnected.append)
        await client.connect(record, "sess-a")
        # A transport death, raised out of the pump's read exactly as a peer
        # RST would surface it.
        assert client._reader is not None
        client._reader.set_exception(ConnectionResetError("peer reset"))
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline and not disconnected:
            await asyncio.sleep(0.05)
        assert disconnected == ["owner connection reset"]
        assert not client.connected
    finally:
        r.close()


class SlowEffortHandle(FakeHandle):
    """An owner that is alive and healthy but slow to answer.

    Both ops are real dispatched wire ops whose ack is sent only after the
    handle returns, so sleeping here reproduces exactly the shape of the
    reported bug: nothing wrong with the socket, the answer simply has not
    arrived yet. ``set_effort`` routes through ``_request_frame`` and
    ``fork_snapshot`` through ``_request_payload`` (``server.py``
    ``_PAYLOAD_OPS``) — the two helpers carry SEPARATE copies of the
    ``except`` ladder, so each needs its own behavioural cover.
    """

    #: Long enough to outlast the 0.05 s deadlines these tests use by an order
    #: of magnitude, short enough that the owner's inline dispatch drains
    #: inside the test rather than being torn down mid-await.
    ANSWER_AFTER_S = 0.5

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        await asyncio.sleep(self.ANSWER_AFTER_S)
        return "effort"

    async def fork_snapshot(self, message):  # noqa: ANN001, ANN202
        await asyncio.sleep(self.ANSWER_AFTER_S)
        return {"parent_id": "sess-a"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper", "op", "fields"),
    [
        ("_request_frame", "set_effort", {"effort": "hi"}),
        ("_request_payload", "fork_snapshot", {"message": ""}),
    ],
)
async def test_an_unanswered_request_is_a_timeout_not_a_lost_connection(
    config: Path, helper: str, op: str, fields: dict[str, str]
) -> None:
    """A slow owner is not a dead one, and it must not be reported as one.

    ``asyncio.wait_for`` raises ``TimeoutError``, which subclasses ``OSError``
    on 3.11+ and whose ``str()`` is ``''``. Caught by the arm meant for dead
    sockets it rendered as the dangling ``owner connection lost:`` this fix
    removes. See docs/design-aside-deadline.md §2.

    Parametrized over BOTH request helpers on purpose. They hold two separate
    copies of the same ``except`` ladder, so cover on one proves nothing about
    the other: an ``OSError`` arm inserted above ``_request_payload``'s
    ``TimeoutError`` arm restores the bug for all eight payload ops
    (``server.py`` ``_PAYLOAD_OPS``) while every frame-side test stays green.
    Asserting the ORDER by source offset does not catch it either — a broader
    ``except OSError`` sits above both arms without disturbing the offsets of
    either pinned substring. Only driving each helper to a real timeout does.
    This is design §7 risk 2, and it is what makes this a regression test.
    """
    handle = SlowEffortHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        await client.connect(record, "sess-a")
        request = getattr(client, helper)
        with pytest.raises(OwnerAckTimeout) as caught:
            await request(op, deadline_s=0.05, **fields)
        assert isinstance(caught.value, OwnerAckTimeout)
        assert isinstance(caught.value, ConnectionError)
        assert isinstance(caught.value, TimeoutError)
        assert str(caught.value)
        assert "owner connection lost" not in str(caught.value)
        assert op in str(caught.value)
        # Let the owner finish the op it is still parked on, so teardown does
        # not cancel a dispatch mid-await and emit a pending-task warning.
        await asyncio.sleep(SlowEffortHandle.ANSWER_AFTER_S + 0.2)
        await client.detach()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_only_the_aside_waits_longer_than_the_ack_budget(config: Path) -> None:
    """The long deadline is scoped to one op; the other 18 keep the 15 s budget.

    Asserted on the deadline PASSED, never by waiting one out: a test that
    really waited 15 s or 180 s would hang the suite (there is no
    ``pytest-timeout`` here).

    The stub below deliberately declares NO default for ``deadline_s``. A stub
    that repeats the production default answers for it: an op calling
    ``_request`` without the argument then records the STUB's value, so
    widening ``_request``'s own default to 180 s would leak the long deadline
    to all 19 ops with this test still green. Without a default the omission
    is a ``TypeError``, and the real default is asserted directly from the
    production signature.
    """
    assert ACK_TIMEOUT_S == 15.0
    assert ASIDE_DEADLINE_S == 180.0
    # Pin the DERIVATION, not just the number: the constant's comment says it
    # is matched to the provider layer's own budget for silence on a stream in
    # flight (design §3). Importing it here is free; importing it in
    # production would couple the transport to the provider package.
    assert ASIDE_DEADLINE_S == STREAM_READ_TIMEOUT_S
    # The seam every non-aside op relies on. Read from the signature because
    # no call site passes it, so nothing else can observe a change to it.
    assert inspect.signature(AttachClient._request).parameters["deadline_s"].default == (
        ACK_TIMEOUT_S
    )

    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        await client.connect(record, "sess-a")

        seen: list[tuple[str, float]] = []

        # Recorded BELOW ``_request``, not in place of it. ``_request`` is
        # where the default lives and it forwards ``deadline_s`` to
        # ``_request_frame`` explicitly, so intercepting here observes the
        # value production actually chose. Stubbing ``_request`` itself would
        # substitute the stub's own signature for the one under test.
        async def recorder(op: str, *, deadline_s: float, **fields) -> dict[str, object]:
            seen.append((op, deadline_s))
            return {"op": "ack", "detail": "recorded"}

        client._request_frame = recorder  # type: ignore[assignment]

        await client.complete_aside([])
        await client.abort()
        await client.request_stop()
        await client.set_effort("hi")

        deadlines = dict(seen)
        assert deadlines["complete_aside"] == ASIDE_DEADLINE_S == 180.0
        for op in ("abort", "stop", "set_effort"):
            assert deadlines[op] == ACK_TIMEOUT_S == 15.0
    finally:
        r.close()


@pytest.mark.asyncio
async def test_a_real_reset_still_reports_a_lost_connection(config: Path) -> None:
    """The disconnect path is untouched: a dead socket still says why.

    The complement of the timeout test — proving the new arm narrowed what the
    ``OSError`` arm catches without taking anything from it.
    """
    handle = FakeHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        await client.connect(record, "sess-a")

        assert client._writer is not None

        async def dead_drain() -> None:
            raise ConnectionResetError("peer reset")

        client._writer.drain = dead_drain  # type: ignore[method-assign]

        with pytest.raises(ConnectionError) as caught:
            await client._request_frame("set_effort", effort="hi")
        assert not isinstance(caught.value, OwnerAckTimeout)
        assert "owner connection lost:" in str(caught.value)
        assert str(caught.value).split("owner connection lost:", 1)[1].strip()
    finally:
        r.close()


@pytest.mark.asyncio
async def test_a_late_reply_for_an_abandoned_request_is_logged(
    config: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A completed provider answer being discarded must not be invisible.

    The waiter gave up and its future was popped, so the owner's ack arrives
    for a ``req`` nobody is holding. Dropping it is correct; dropping it
    silently loses the only trace that the work was paid for (design §4).
    """
    handle = SlowEffortHandle("sess-a")
    r = RuntimeServer(handle, kind="tui")
    r.start()
    try:
        record = await _wait_record()
        client = AttachClient(lambda p: None, lambda reason: None)
        await client.connect(record, "sess-a")

        with caplog.at_level(logging.WARNING, logger="local_operator.mobile.attach_client"):
            with pytest.raises(OwnerAckTimeout):
                await client._request_frame("set_effort", deadline_s=0.05, effort="hi")
            abandoned_req = client._req_seq
            assert not client._pending
            # The owner is still working; its ack lands once the handle
            # returns. Wait for the pump to see it rather than sleeping a
            # fixed span.
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                if "unknown request" in caplog.text:
                    break
                await asyncio.sleep(0.05)

        assert "unknown request" in caplog.text
        assert str(abandoned_req) in caplog.text
        await client.detach()
    finally:
        r.close()
