"""Multi-connection registrant: N authenticated clients on one control socket.

Protocol v2's core contract — the daemon plus up to ATTACH_MAX_CLIENTS attach
terminals, broadcast pushes, point-to-point acks, watch/unwatch accounting,
and the attach dispatch restrictions. These run against the REAL socket (a
FakeHandle), matching test_daemon.py's style, because the failure modes this
module guards (frame interleaving, eviction, registry leaks) only exist on a
real connection pair.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from local_operator.mobile import registry
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import (
    ATTACH_MAX_CLIENTS,
    PROTOCOL_VERSION,
    PendingRequest,
    SessionProjection,
    TranscriptEntry,
)


class FakeHandle:
    """Static projection; records dispatch calls for assertions."""

    def __init__(self) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake",
            cwd="/tmp",
            model_label="test/model",
        )
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._event_handler = None
        self.event_pending: PendingRequest | None = None

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    def subscribe_events(self, on_event):  # noqa: ANN001, ANN202
        self._event_handler = on_event
        return lambda: None

    def emit_event(self, event) -> None:  # noqa: ANN001
        if self._event_handler is not None:
            self._event_handler(event.model_dump(mode="json"))

    async def _record(self, name: str, *args: object, **kwargs: object) -> str:
        self.calls.append((name, args, kwargs))
        return f"{name} ok"

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        return await self._record("steer", text)

    async def recall_steer(self, command_id):  # noqa: ANN001, ANN202
        return await self._record("recall_steer", command_id)

    async def receive_peer_message(  # noqa: ANN001, ANN202
        self, text, *, mode="mailbox", wake=False, sender=None
    ):
        self.calls.append(
            ("receive_peer_message", (text,), {"mode": mode, "wake": wake, "sender": sender})
        )
        return "delivered to the mailbox (will be read on the next turn)"

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
        return await self._record("ask_answer", request_id, value, question_index)

    async def refresh(self) -> None:
        pass


class ConcurrentHandle(FakeHandle):
    """Owner-side admission model for real-socket multi-producer tests."""

    def __init__(self) -> None:
        super().__init__()
        self._admission_lock = asyncio.Lock()
        self._notify = None
        self.admitted: list[tuple[str, str]] = []

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        self._notify = on_projection
        return lambda: None

    async def _admit(self, kind: str, text: str) -> str:
        async with self._admission_lock:
            self.admitted.append((kind, text))
            self._projection.transcript.append(
                TranscriptEntry(
                    id=f"{kind}-{len(self.admitted)}",
                    kind="steer" if kind == "steer" else "user",
                    text=text,
                )
            )
            if self._notify is not None:
                self._notify()
            await asyncio.sleep(0)
            return f"{kind} admitted"

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        return await self._admit("prompt", text)

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        return await self._admit("steer", text)


async def _wait_record() -> registry.SessionRecord:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan()
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.05)
    raise AssertionError("registrant never published a live record")


async def _dial(record: registry.SessionRecord, *, client: str | None = None):
    """Open + auth one connection; consume the welcome projection."""
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    auth: dict[str, object] = {"key": record.control_key}
    if client is not None:
        auth["client"] = client
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    welcome = await asyncio.wait_for(reader.readline(), timeout=5)
    assert json.loads(welcome)["op"] == "projection"
    return reader, writer


async def _until(
    reader: asyncio.StreamReader, want_op: str, want_req: object = None, n: int = 30
) -> dict[str, Any]:
    """Read until a frame matching (op, req) arrives; skip broadcasts."""
    for _ in range(n):
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        s = raw.decode("utf-8", "replace").strip()
        if not s:
            continue
        frame = json.loads(s)
        if frame.get("op") == want_op and (want_req is None or frame.get("req") == want_req):
            return frame
    raise AssertionError(f"no {want_op} frame arrived")


@pytest.mark.asyncio
async def test_protocol_version_is_four_and_cap_constant() -> None:
    assert PROTOCOL_VERSION == 4
    assert ATTACH_MAX_CLIENTS == 4


@pytest.mark.asyncio
async def test_v4_event_client_gets_seed_and_events_daemon_gets_no_raw_frames() -> None:
    """Raw AgentEvents are opt-in attach frames; phone daemon stays byte-identical."""
    from local_operator.harness.types import AgentStartEvent, NoticeEvent

    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    daemon_writer = attach_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)
        attach_reader, attach_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        attach_writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await attach_writer.drain()
        assert json.loads(await attach_reader.readline())["op"] == "projection"
        seed = json.loads(await attach_reader.readline())
        assert seed["op"] == "attach_sync"
        assert seed["data"]["streaming"] is False

        handle.emit_event(AgentStartEvent(generation=9))
        handle.emit_event(NoticeEvent(text="live", kind="info"))
        first = await _until(attach_reader, "event")
        second = await _until(attach_reader, "event")
        assert [first["data"]["type"], second["data"]["type"]] == [
            "agent_start",
            "notice",
        ]
        # A projection refresh is the only owner push a daemon may see. There
        # is no event frame queued on its byte stream.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(daemon_reader.readline(), timeout=0.1)
    finally:
        for writer in (daemon_writer, attach_writer):
            if writer is not None:
                writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_event_pending_is_overlaid_only_for_event_clients() -> None:
    """A TUI approval reaches followers without changing phone daemon bytes."""
    handle = FakeHandle()
    pending = PendingRequest(
        request_id="approval-1", kind="approval", title="bash", detail="echo hi"
    )
    handle.event_pending = pending
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    daemon_writer = attach_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)
        attach_reader, attach_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port
        )
        attach_writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await attach_writer.drain()
        follower = json.loads(await attach_reader.readline())
        assert json.loads(await attach_reader.readline())["op"] == "attach_sync"
        # _dial consumed the daemon welcome; trigger a fresh ordinary repaint
        # and compare it to the event client's overlaid form.
        await registrant._push()
        daemon = json.loads(await daemon_reader.readline())
        follower_repaint = json.loads(await attach_reader.readline())
        assert daemon["data"]["pending"] is None
        assert follower["data"]["pending"]["request_id"] == "approval-1"
        assert follower_repaint["data"]["pending"]["request_id"] == "approval-1"
    finally:
        for writer in (daemon_writer, attach_writer):
            if writer is not None:
                writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_event_seed_covers_events_before_client_is_ready() -> None:
    """A mid-turn join gets open state once in attach_sync, then later events."""
    from local_operator.harness.types import AgentStartEvent, NoticeEvent

    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _wait_record()
        handle.emit_event(AgentStartEvent(generation=4))
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["op"] == "projection"
        seed = json.loads(await reader.readline())
        assert seed["op"] == "attach_sync"
        assert seed["data"]["streaming"] is True
        assert seed["data"]["generation"] == 4
        handle.emit_event(NoticeEvent(text="after seed", kind="info"))
        frame = await _until(reader, "event")
        assert frame["data"]["text"] == "after seed"
    finally:
        if writer is not None:
            writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_recall_steer_dispatches_by_command_id() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(
            json.dumps({"op": "recall_steer", "req": 8, "command_id": "m1"}).encode() + b"\n"
        )
        await writer.drain()
        assert (await _until(reader, "ack", 8))["detail"] == "recall_steer ok"
        assert handle.calls[-1][0:2] == ("recall_steer", ("m1",))
    finally:
        if writer is not None:
            writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_peer_message_dispatches_with_parsed_args() -> None:
    """A `lop send` peer_message reaches the handle with mode/wake/sender parsed."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        writer.write(
            json.dumps(
                {
                    "op": "peer_message",
                    "req": 11,
                    "text": "gates are green",
                    "mode": "mailbox",
                    "wake": True,
                    "sender": {"pid": 4242, "conversation_name": "peer-send design"},
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        ack = await _until(reader, "ack", 11)
        assert "mailbox" in ack["detail"]
        name, args, kwargs = handle.calls[-1]
        assert name == "receive_peer_message"
        assert args == ("gates are green",)
        assert kwargs["mode"] == "mailbox"
        assert kwargs["wake"] is True
        sender = cast("dict[str, Any]", kwargs["sender"])
        assert sender["pid"] == 4242
    finally:
        if writer is not None:
            writer.close()
        registrant.close()


class NoPeerHandle(FakeHandle):
    """An owner host that predates peer messaging: no receive_peer_message.

    The dispatch probes the capability with getattr, so a handle that simply
    lacks the method must surface the clear "cannot receive" error rather than
    an AttributeError — exactly the optional-capability contract recall_steer
    documents."""

    receive_peer_message = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_peer_message_on_handle_without_capability_errors_cleanly() -> None:
    handle = NoPeerHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        writer.write(
            json.dumps({"op": "peer_message", "req": 12, "text": "hi", "mode": "mailbox"}).encode()
            + b"\n"
        )
        await writer.drain()
        err = await _until(reader, "error", 12)
        assert "cannot receive peer messages" in err["message"]
    finally:
        if writer is not None:
            writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_daemon_and_attach_clients_coexist() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        ra, wa = await _dial(record, client="attach")
        ra2, wa2 = await _dial(record, client="attach")
        assert registrant.attach_clients() == 2
        # All three stay live across traffic from any of them.
        wa.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await wa.drain()
        await _until(ra, "ack", 1)
        wa2.write(json.dumps({"op": "ping", "req": 2}).encode() + b"\n")
        await wa2.drain()
        await _until(ra2, "ack", 2)
        assert registrant.attach_clients() == 2
        for w in (wd, wa, wa2):
            w.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_in_process_close_joins_delayed_projection_push() -> None:
    """Loop teardown leaves no coalesced repaint task behind."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    await registrant.start_in_process()
    registrant._schedule_push()
    await asyncio.sleep(0)
    task = registrant._push_task
    assert task is not None and not task.done()

    await registrant.aclose()

    assert task.done()
    assert registrant._push_task is None
    assert not registrant._push_scheduled
    assert registrant._heartbeat_task is None
    assert registrant._server is None

    # A second awaited close must join the completed shutdown rather than
    # returning early based only on the cross-thread close latch.
    await registrant.aclose()


@pytest.mark.asyncio
async def test_in_process_sync_close_schedules_cleanup_without_deadlock() -> None:
    """Legacy synchronous hosts may close from inside the owning loop."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    await registrant.start_in_process()
    registrant._schedule_push()
    await asyncio.sleep(0)

    registrant.close()
    registrant.close()
    task = registrant._shutdown_task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=2)

    assert registrant._heartbeat_task is None
    assert registrant._push_task is None
    assert registrant._server is None


@pytest.mark.asyncio
async def test_broadcast_reaches_every_client() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        ra, wa = await _dial(record, client="attach")
        # A mutation from the DAEMON must repaint the ATTACH client too.
        wd.write(json.dumps({"op": "set_effort", "req": 7, "effort": "high"}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 7)
        frame = await _until(ra, "projection")
        assert frame["data"]["session_id"] == "s1"
        wd.close()
        wa.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["prompt", "steer"])
async def test_concurrent_producers_ack_once_and_converge(op: str) -> None:
    """Daemon plus two attaches submit together through one transcript owner."""
    handle = ConcurrentHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        clients = [
            await _dial(record),
            await _dial(record, client="attach"),
            await _dial(record, client="attach"),
        ]
        texts = ["from mobile", "from attach one", "from attach two"]
        for req, ((_, writer), text) in enumerate(zip(clients, texts)):
            writer.write(json.dumps({"op": op, "req": req, "text": text}).encode() + b"\n")
        await asyncio.gather(*(writer.drain() for _, writer in clients))

        acks = await asyncio.gather(
            *(_until(reader, "ack", req) for req, (reader, _) in enumerate(clients))
        )
        assert [ack["req"] for ack in acks] == [0, 1, 2]
        assert sorted(text for kind, text in handle.admitted if kind == op) == sorted(texts)
        assert len(handle.admitted) == 3

        expected = [(kind, text) for kind, text in handle.admitted]

        async def reconciled(reader: asyncio.StreamReader) -> dict[str, Any]:
            for _ in range(10):
                projection = await _until(reader, "projection")
                if len(projection["data"]["transcript"]) == len(expected):
                    return projection
            raise AssertionError("viewer never received the reconciled projection")

        projections = await asyncio.gather(*(reconciled(reader) for reader, _ in clients))
        for projection in projections:
            rows = projection["data"]["transcript"]
            assert [
                ("steer" if row["kind"] == "steer" else "prompt", row["text"]) for row in rows
            ] == expected
        for _, writer in clients:
            writer.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_high_volume_event_relay_bounds_nonreader_and_preserves_healthy_order(
    monkeypatch,
) -> None:
    """One slow writer has one bounded task; a healthy peer sees ordered events."""
    from local_operator.harness.types import NoticeEvent
    from local_operator.mobile import registrant as registrant_module

    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    await registrant.start_in_process()
    slow_reader = slow_writer = healthy_writer = None
    blocked = asyncio.Event()
    original_send = registrant._send_to
    try:
        record = registrant._record
        slow_reader, slow_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        slow_writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await slow_writer.drain()
        assert json.loads(await slow_reader.readline())["op"] == "projection"
        assert json.loads(await slow_reader.readline())["op"] == "attach_sync"
        assert len(registrant._clients) == 1
        slow_conn = next(iter(registrant._clients.values()))

        healthy_reader, healthy_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        healthy_writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await healthy_writer.drain()
        assert json.loads(await healthy_reader.readline())["op"] == "projection"
        assert json.loads(await healthy_reader.readline())["op"] == "attach_sync"

        async def block_only_slow(conn, frame):  # noqa: ANN001, ANN202
            if conn is slow_conn and frame.get("op") == "event":
                await blocked.wait()
                return
            await original_send(conn, frame)

        monkeypatch.setattr(registrant, "_send_to", block_only_slow)
        total = registrant_module._EVENT_QUEUE_MAX * 2
        for index in range(total):
            handle.emit_event(NoticeEvent(text=f"event-{index}", kind="info"))
            # Healthy writer gets scheduling opportunities while the deliberately
            # blocked peer's one writer remains unable to consume its FIFO.
            await asyncio.sleep(0)

        # The slow client is dropped on overflow rather than retaining one task
        # per event. The only remaining event writer belongs to the healthy peer.
        assert id(slow_conn.writer) not in registrant._clients
        assert len(registrant._event_sends) <= 1
        assert slow_conn.event_queue.qsize() <= registrant_module._EVENT_QUEUE_MAX

        received = []
        deadline = asyncio.get_running_loop().time() + 2
        while len(received) < total and asyncio.get_running_loop().time() < deadline:
            frame = json.loads(await asyncio.wait_for(healthy_reader.readline(), timeout=1))
            if frame.get("op") == "event":
                received.append(frame["data"]["text"])
        assert received == [f"event-{index}" for index in range(total)]
    finally:
        blocked.set()
        for writer in (slow_writer, healthy_writer):
            if writer is not None:
                writer.close()
        await registrant.aclose()


@pytest.mark.asyncio
async def test_nonreading_socket_cannot_block_active_ack_and_projection() -> None:
    """A real authenticated peer that applies backpressure loses only itself."""
    handle = FakeHandle()
    # Each repaint is large enough that a handful fill the non-reader's kernel
    # window, while remaining below the protocol's one-megabyte frame limit.
    handle._projection.conversation_name = "x" * 700_000
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        slow_reader, slow_writer = await _dial(record, client="attach")
        del slow_reader  # authenticate and consume welcome, then never read again
        active_reader, active_writer = await _dial(record, client="attach")
        for req in range(12):
            active_writer.write(json.dumps({"op": "ping", "req": req}).encode() + b"\n")
            await active_writer.drain()
            ack = await asyncio.wait_for(_until(active_reader, "ack", req), timeout=2.5)
            assert ack["detail"] == "pong"
            projection = await asyncio.wait_for(_until(active_reader, "projection"), timeout=2.5)
            assert projection["data"]["session_id"] == "s1"
        active_writer.close()
        slow_writer.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_acks_stay_point_to_point_under_concurrent_ops() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        ra2, wa2 = await _dial(record, client="attach")
        # Two concurrent ops with overlapping req ids from two clients.
        wa.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        wa2.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await wa.drain()
        await wa2.drain()
        a1 = await _until(ra, "ack", 1)
        a2 = await _until(ra2, "ack", 1)
        # Each client sees exactly its own ack detail; no cross-delivery of
        # the OTHER client's frames between the two reads is hard to prove
        # exhaustively, but each reader must never see an ERROR here.
        assert a1["detail"] == "pong"
        assert a2["detail"] == "pong"
        wa.close()
        wa2.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_new_daemon_dial_evicts_the_old() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        rd2, wd2 = await _dial(record)  # reconnect story
        # The old socket observes EOF.
        raw = await asyncio.wait_for(rd.readline(), timeout=5)
        assert raw == b""
        # The new one still works.
        wd2.write(json.dumps({"op": "ping", "req": 3}).encode() + b"\n")
        await wd2.drain()
        await _until(rd2, "ack", 3)
        wd.close()
        wd2.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_attach_cap_evicts_least_recently_seen() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        pairs = [await _dial(record, client="attach") for _ in range(ATTACH_MAX_CLIENTS)]
        assert registrant.attach_clients() == ATTACH_MAX_CLIENTS
        # Touch every attach EXCEPT the first (the LRU victim).
        for i in range(1, ATTACH_MAX_CLIENTS):
            reader, writer = pairs[i]
            writer.write(json.dumps({"op": "ping", "req": i}).encode() + b"\n")
            await writer.drain()
            await _until(reader, "ack", i)
            await asyncio.sleep(0.05)
        # A further dial evicts the untouched first client.
        rn, wn = await _dial(record, client="attach")
        victim_reader = pairs[0][0]
        # The victim's socket still holds the broadcasts queued BEFORE its
        # eviction; drain until EOF (the eviction itself closes it).
        raw = b"x"
        deadline = asyncio.get_running_loop().time() + 5
        while raw != b"" and asyncio.get_running_loop().time() < deadline:
            raw = await asyncio.wait_for(victim_reader.readline(), timeout=5)
        assert raw == b""  # evicted: EOF
        assert registrant.attach_clients() == ATTACH_MAX_CLIENTS
        wn.close()
        for _, w in pairs:
            w.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_attach_client_cannot_rebind_the_session() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        wa.write(json.dumps({"op": "resume_session", "req": 4, "session_id": "x"}).encode() + b"\n")
        await wa.drain()
        err = await _until(ra, "error", 4)
        assert "cannot rebind" in err["message"]
        wa.write(json.dumps({"op": "new_conversation", "req": 5}).encode() + b"\n")
        await wa.drain()
        err = await _until(ra, "error", 5)
        assert "cannot rebind" in err["message"]
        # The daemon keeps both ops.
        rd, wd = await _dial(record)
        wd.write(json.dumps({"op": "new_conversation", "req": 6}).encode() + b"\n")
        await wd.drain()
        ack = await _until(rd, "ack", 6)
        assert "new_conversation ok" in ack["detail"]
        wa.close()
        wd.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_dead_socket_leaves_the_registry() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        assert registrant.attach_clients() == 1
        wa.close()
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if registrant.attach_clients() == 0:
                break
            await asyncio.sleep(0.05)
        assert registrant.attach_clients() == 0
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_watch_unwatch_accounting_and_floor() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        assert registrant.phone_watchers == 0
        assert registrant.watch_supported is False
        rd, wd = await _dial(record)
        wd.write(json.dumps({"op": "unwatch", "req": 1}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 1)
        # The unwatch-before-watch case floors at zero: a daemon restart
        # redials without unwatching and the counter must not go negative.
        assert registrant.phone_watchers == 0
        assert registrant.watch_supported is True
        wd.write(json.dumps({"op": "watch", "req": 2}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 2)
        assert registrant.phone_watchers == 1
        wd.write(json.dumps({"op": "watch", "req": 3}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 3)
        assert registrant.phone_watchers == 2
        wd.write(json.dumps({"op": "unwatch", "req": 4}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 4)
        assert registrant.phone_watchers == 1
        # The latch never resets.
        assert registrant.watch_supported is True
        wd.close()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_absent_client_field_means_daemon() -> None:
    """An OLD daemon dialing a NEW registrant keeps the daemon class."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)  # no client field
        assert registrant.attach_clients() == 0
        # And a second daemon-class dial still evicts it (reconnect).
        rd2, wd2 = await _dial(record)
        raw = await asyncio.wait_for(rd.readline(), timeout=5)
        assert raw == b""
        wd.close()
        wd2.close()
    finally:
        registrant.close()
