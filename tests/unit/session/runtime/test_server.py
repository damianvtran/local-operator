"""Multi-connection session runtime: N authenticated clients on one control socket.

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

from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    TranscriptEntry,
)
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.types import ATTACH_MAX_CLIENTS, PROTOCOL_VERSION


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
        from local_operator.session.frontend_state import (
            FrontendModelSpec,
            FrontendSessionState,
            FrontendStateStore,
        )

        self._frontend = FrontendStateStore(
            FrontendSessionState(
                session_id="s1",
                epoch="fake-owner",
                cwd="/tmp",
                conversation_title="fake",
                selected_model=FrontendModelSpec(
                    provider="test", model_id="model", context_window=1_000_000
                ),
                effective_model=FrontendModelSpec(
                    provider="test", model_id="model", context_window=1_000_000
                ),
                context_window=1_000_000,
            )
        )

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    @property
    def frontend_state_seed(self):  # noqa: ANN202
        return self._frontend.state

    def subscribe_frontend(self, on_update):  # noqa: ANN001, ANN202
        return self._frontend.subscribe(on_update)

    def subscribe_events(self, on_event):  # noqa: ANN001, ANN202
        self._event_handler = on_event
        return lambda: None

    def emit_event(self, event) -> None:  # noqa: ANN001
        # Production Session folds the canonical seed before raw fan-out. This
        # reduced socket handle mirrors that owner ordering explicitly.
        self._frontend._fold_live_event(event)
        if event.type == "agent_start":
            self._frontend.mutate(streaming=True, generation=event.generation)
        elif event.type == "agent_end":
            self._frontend.mutate(streaming=False)
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
    ) -> str:
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

    async def complete_aside(self, turns):  # noqa: ANN001, ANN202
        self.calls.append(("complete_aside", (turns,), {}))
        return "aside answer"

    async def slash_images(self, command, args, images):  # noqa: ANN001, ANN202
        return await self._record("slash", command, args, images)

    async def run_slash_authoritative(self, command, args, images):  # noqa: ANN001, ANN202
        self.calls.append(("run_slash_authoritative", (command, args, images), {}))
        # The owner returns a typed result the invoker renders locally; this
        # reduced owner answers every routed command with a goal-shaped notice.
        return {"kind": "notice", "text": f"owner ran /{command}", "style": "info"}

    async def adopt_aside(self, messages):  # noqa: ANN001, ANN202
        self.calls.append(("adopt_aside", (messages,), {}))
        return "forked aside"

    def cancel_subagents_count(self):  # noqa: ANN202
        self.calls.append(("cancel_subagents_count", (), {}))
        return 2

    async def job_trajectory(self, job_id, offset, limit):  # noqa: ANN001, ANN202
        """Serve a child's retained events the way the owned handle does.

        Attach snapshots omit trajectories (they exceed the socket's line
        limit), so a follower fetches them per job. Reading them back out of
        the canonical store here keeps this double on the same contract as
        production without a second source of job rows.
        """
        self.calls.append(("job_trajectory", (job_id, offset, limit), {}))
        from local_operator.session.frontend_state import _wire_value

        job = next((row for row in self._frontend.state.jobs if row.id == job_id), None)
        # Production reads plain dicts off the live ``AsyncJob``; this double
        # reads the canonical store, whose retained rows are immutable Mapping
        # wrappers that JSON-encode as item pairs unless thawed first — the
        # same boundary conversion the store's own serializer performs.
        rows = [
            _wire_value(row)
            for row in (list(getattr(job, "trajectory", None) or []) if job is not None else [])
        ]
        first = rows[0] if rows else None
        base_seq = first.get("_traj_seq") if isinstance(first, dict) else None
        return {
            "job_id": job_id,
            "rows": rows[offset : offset + limit],
            "offset": offset,
            "total": len(rows),
            "base_seq": base_seq if isinstance(base_seq, int) else None,
            "known": job is not None,
        }

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
    raise AssertionError("runtime never published a live record")


async def _dial(
    record: registry.SessionRecord,
    *,
    client: str | None = None,
    locality: str | None = None,
):
    """Open + auth one connection; consume the welcome projection."""
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    auth: dict[str, object] = {"key": record.control_key}
    if client is not None:
        auth["client"] = client
    if locality is not None:
        auth["locality"] = locality
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
async def test_protocol_version_is_five_and_cap_constant() -> None:
    assert PROTOCOL_VERSION == 5
    assert ATTACH_MAX_CLIENTS == 4


@pytest.mark.asyncio
async def test_pushed_projection_frame_carries_no_subagent_transcript() -> None:
    """The wire frame a runtime pushes must never embed a child transcript.

    Regression guard for the real-time freeze: a full-repaint projection is
    pushed ~30x/s and the daemon's control-socket reader caps a single frame at
    1 MB. When each subagent row carried its (tail-capped) transcript, a deep
    roster overran that cap, every push was dropped as oversized, and the phone
    silently fell back to the stale durable disk fold. Subagent transcripts are
    now fetched lazily from the child-history endpoint, so the pushed frame must
    contain zero subagent transcript entries even after hydration ran. Todos DO
    stay on the wire (small, and the live working line needs them).
    """
    from types import SimpleNamespace

    from local_operator.harness.comms import SubagentComms
    from local_operator.session.session import Session

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")

    # Seed one subagent row through the same fold the runtime pushes, then
    # hydrate it with a transcript far larger than the render tail.
    fold = runtime.fold
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child")
    fold.set_subagent_details(comms)
    heavy = [TranscriptEntry(id=f"row-{i}", kind="assistant", text="x" * 4096) for i in range(200)]
    fold.set_subagent_hydrated_details("child", heavy, [{"text": "verify", "status": "pending"}])

    runtime.start()
    daemon_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)
        # Force a repaint and read the resulting daemon-side broadcast frame.
        runtime._schedule_push()
        frame = await _until(daemon_reader, "projection")
        subagents = frame["data"]["subagents"]
        assert subagents, "expected the seeded child row on the wire"
        assert all(sub["transcript"] == [] for sub in subagents)
        # Todos survive: the live working line renders them without a fetch.
        assert subagents[0]["todos"], "todos must stay on the wire"
    finally:
        if daemon_writer is not None:
            daemon_writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_v4_event_client_gets_seed_and_events_daemon_gets_no_raw_frames() -> None:
    """Raw AgentEvents are opt-in attach frames; phone daemon stays byte-identical."""
    from local_operator.harness.types import AgentStartEvent, NoticeEvent

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    daemon_writer = attach_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)
        attach_reader, attach_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        attach_writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "events": True,
                    "frontend_state": True,
                }
            ).encode()
            + b"\n"
        )
        await attach_writer.drain()
        assert json.loads(await attach_reader.readline())["op"] == "projection"
        seed = json.loads(await attach_reader.readline())
        assert seed["op"] == "frontend_sync"
        assert seed["data"]["snapshot"]["streaming"] is False

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
        runtime.close()


@pytest.mark.asyncio
async def test_pending_gate_uses_canonical_stream_not_projection_overlay() -> None:
    """A TUI gate reaches followers while phone projection bytes stay ordinary."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    daemon_writer = attach_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)
        attach_reader, attach_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port
        )
        attach_writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "events": True,
                    "frontend_state": True,
                }
            ).encode()
            + b"\n"
        )
        await attach_writer.drain()
        follower = json.loads(await attach_reader.readline())
        sync = json.loads(await attach_reader.readline())
        assert follower["data"]["pending"] is None
        assert sync["data"]["snapshot"]["pending_gate"] is None

        handle._frontend.mutate(
            pending_gate=PendingRequest(
                request_id="approval-1", kind="approval", title="bash", detail="echo hi"
            ).to_json()
        )
        update = await _until(attach_reader, "frontend_update")
        assert update["data"]["changes"]["pending_gate"]["request_id"] == "approval-1"
        await runtime._push()
        daemon = json.loads(await daemon_reader.readline())
        assert daemon["data"]["pending"] is None
    finally:
        for writer in (daemon_writer, attach_writer):
            if writer is not None:
                writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_event_seed_covers_events_before_client_is_ready() -> None:
    """A mid-turn join gets open state once in frontend_sync, then later events."""
    from local_operator.harness.types import AgentStartEvent, NoticeEvent

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        handle.emit_event(AgentStartEvent(generation=4))
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "events": True,
                    "frontend_state": True,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["op"] == "projection"
        seed = json.loads(await reader.readline())
        assert seed["op"] == "frontend_sync"
        assert seed["data"]["snapshot"]["streaming"] is True
        assert seed["data"]["snapshot"]["generation"] == 4
        handle.emit_event(NoticeEvent(text="after seed", kind="info"))
        frame = await _until(reader, "event")
        assert frame["data"]["text"] == "after seed"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_recall_steer_dispatches_by_command_id() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_peer_message_dispatches_with_parsed_args() -> None:
    """A `lop send` peer_message reaches the handle with mode/wake/sender parsed."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


class NoPeerHandle(FakeHandle):
    """An owner runtime that predates peer messaging: no receive_peer_message.

    The dispatch probes the capability with getattr, so a handle that simply
    lacks the method must surface the clear "cannot receive" error rather than
    an AttributeError — exactly the optional-capability contract recall_steer
    documents."""

    receive_peer_message = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_peer_message_on_handle_without_capability_errors_cleanly() -> None:
    handle = NoPeerHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_daemon_and_attach_clients_coexist() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)
        ra, wa = await _dial(record, client="attach")
        ra2, wa2 = await _dial(record, client="attach")
        assert runtime.attach_clients() == 2
        # All three stay live across traffic from any of them.
        wa.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await wa.drain()
        await _until(ra, "ack", 1)
        wa2.write(json.dumps({"op": "ping", "req": 2}).encode() + b"\n")
        await wa2.drain()
        await _until(ra2, "ack", 2)
        assert runtime.attach_clients() == 2
        for w in (wd, wa, wa2):
            w.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_in_process_close_joins_delayed_projection_push() -> None:
    """Loop teardown leaves no coalesced repaint task behind."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    await runtime.start_in_process()
    runtime._schedule_push()
    await asyncio.sleep(0)
    task = runtime._push_task
    assert task is not None and not task.done()

    await runtime.aclose()

    assert task.done()
    assert runtime._push_task is None
    assert not runtime._push_scheduled
    assert runtime._heartbeat_task is None
    assert runtime._server is None

    # A second awaited close must join the completed shutdown rather than
    # returning early based only on the cross-thread close latch.
    await runtime.aclose()


@pytest.mark.asyncio
async def test_in_process_sync_close_schedules_cleanup_without_deadlock() -> None:
    """Legacy synchronous hosts may close from inside the owning loop."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    await runtime.start_in_process()
    runtime._schedule_push()
    await asyncio.sleep(0)

    runtime.close()
    runtime.close()
    task = runtime._shutdown_task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=2)

    assert runtime._heartbeat_task is None
    assert runtime._push_task is None
    assert runtime._server is None


@pytest.mark.asyncio
async def test_broadcast_reaches_every_client() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["prompt", "steer"])
async def test_concurrent_producers_ack_once_and_converge(op: str) -> None:
    """Daemon plus two attaches submit together through one transcript owner."""
    handle = ConcurrentHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_high_volume_event_relay_bounds_nonreader_and_preserves_healthy_order(
    monkeypatch,
) -> None:
    """One slow writer has one bounded task; a healthy peer sees ordered events."""
    from local_operator.harness.types import NoticeEvent
    from local_operator.session.runtime import server as server_module

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    await runtime.start_in_process()
    slow_reader = slow_writer = healthy_writer = None
    blocked = asyncio.Event()
    original_send = runtime._send_to
    try:
        record = runtime._record
        slow_reader, slow_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        slow_writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "events": True,
                    "frontend_state": True,
                }
            ).encode()
            + b"\n"
        )
        await slow_writer.drain()
        assert json.loads(await slow_reader.readline())["op"] == "projection"
        assert json.loads(await slow_reader.readline())["op"] == "frontend_sync"
        assert len(runtime._clients) == 1
        slow_conn = next(iter(runtime._clients.values()))

        healthy_reader, healthy_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        healthy_writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "events": True,
                    "frontend_state": True,
                }
            ).encode()
            + b"\n"
        )
        await healthy_writer.drain()
        assert json.loads(await healthy_reader.readline())["op"] == "projection"
        assert json.loads(await healthy_reader.readline())["op"] == "frontend_sync"

        async def block_only_slow(conn, frame):  # noqa: ANN001, ANN202
            if conn is slow_conn and frame.get("op") == "event":
                await blocked.wait()
                return
            await original_send(conn, frame)

        monkeypatch.setattr(runtime, "_send_to", block_only_slow)
        total = server_module._EVENT_QUEUE_MAX * 2
        for index in range(total):
            handle.emit_event(NoticeEvent(text=f"event-{index}", kind="info"))
            # Healthy writer gets scheduling opportunities while the deliberately
            # blocked peer's one writer remains unable to consume its FIFO.
            await asyncio.sleep(0)

        # The slow client is dropped on overflow rather than retaining one task
        # per event. The only remaining event writer belongs to the healthy peer.
        assert id(slow_conn.writer) not in runtime._clients
        assert len(runtime._event_sends) <= 1
        assert slow_conn.event_queue.qsize() <= server_module._EVENT_QUEUE_MAX

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
        await runtime.aclose()


@pytest.mark.asyncio
async def test_nonreading_socket_cannot_block_active_ack_and_projection() -> None:
    """A real authenticated peer that applies backpressure loses only itself."""
    handle = FakeHandle()
    # Each repaint is large enough that a handful fill the non-reader's kernel
    # window, while remaining below the protocol's one-megabyte frame limit.
    handle._projection.conversation_name = "x" * 700_000
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_acks_stay_point_to_point_under_concurrent_ops() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_new_daemon_dial_evicts_the_old() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_attach_cap_evicts_least_recently_seen() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        pairs = [await _dial(record, client="attach") for _ in range(ATTACH_MAX_CLIENTS)]
        assert runtime.attach_clients() == ATTACH_MAX_CLIENTS
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
        assert runtime.attach_clients() == ATTACH_MAX_CLIENTS
        wn.close()
        for _, w in pairs:
            w.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_attach_client_cannot_rebind_the_session() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
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
        runtime.close()


@pytest.mark.asyncio
async def test_dead_socket_leaves_the_registry() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        ra, wa = await _dial(record, client="attach")
        assert runtime.attach_clients() == 1
        wa.close()
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if runtime.attach_clients() == 0:
                break
            await asyncio.sleep(0.05)
        assert runtime.attach_clients() == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_watch_unwatch_accounting_and_floor() -> None:
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        assert runtime.phone_watchers == 0
        assert runtime.watch_supported is False
        rd, wd = await _dial(record)
        wd.write(json.dumps({"op": "unwatch", "req": 1}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 1)
        # The unwatch-before-watch case floors at zero: a daemon restart
        # redials without unwatching and the counter must not go negative.
        assert runtime.phone_watchers == 0
        assert runtime.watch_supported is True
        wd.write(json.dumps({"op": "watch", "req": 2}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 2)
        assert runtime.phone_watchers == 1
        wd.write(json.dumps({"op": "watch", "req": 3}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 3)
        assert runtime.phone_watchers == 2
        wd.write(json.dumps({"op": "unwatch", "req": 4}).encode() + b"\n")
        await wd.drain()
        await _until(rd, "ack", 4)
        assert runtime.phone_watchers == 1
        # The latch never resets.
        assert runtime.watch_supported is True
        wd.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_absent_client_field_means_daemon() -> None:
    """An OLD daemon dialing a NEW runtime keeps the daemon class."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        rd, wd = await _dial(record)  # no client field
        assert runtime.attach_clients() == 0
        # And a second daemon-class dial still evicts it (reconnect).
        rd2, wd2 = await _dial(record)
        raw = await asyncio.wait_for(rd.readline(), timeout=5)
        assert raw == b""
        wd.close()
        wd2.close()
    finally:
        runtime.close()


# --- ProjectionSink: injected, lazily built, never for a fold-free runtime ---


@pytest.mark.asyncio
async def test_attach_only_runtime_builds_no_projection_fold() -> None:
    """A headless runtime serving only follower terminals pays nothing to
    fold: the welcome serializes the seed directly, and no ProjectionFold is
    constructed for the whole lifetime of the connection."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="daemon")
    assert runtime.projection_sink is None
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        # Repaints still flow (the seed is the host's own mutable object).
        handle._projection.conversation_name = "renamed"
        runtime._schedule_push()
        frame = await _until(reader, "projection")
        assert frame["data"]["conversation_name"] == "renamed"
        assert frame["data"]["session_id"] == "s1"
        assert runtime.projection_sink is None
        assert runtime.projection_sinks_built == 0
    finally:
        if writer is not None:
            writer.close()
        runtime.close()
    assert runtime.projection_sinks_built == 0


@pytest.mark.asyncio
async def test_daemon_dial_builds_the_default_fold_once() -> None:
    """The mobile daemon is the projection consumer; its first dial builds
    the fold, a redial reuses it, and its frames carry the folded state."""
    from local_operator.mobile.projection import ProjectionFold

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    first = second = None
    try:
        record = await _wait_record()
        _, first = await _dial(record)
        assert isinstance(runtime.projection_sink, ProjectionFold)
        assert runtime.projection_sinks_built == 1
        sink = runtime.projection_sink
        reader, second = await _dial(record)  # daemon redial evicts the first
        assert runtime.projection_sink is sink
        assert runtime.projection_sinks_built == 1
        # A fold mutation reaches the daemon's frame.
        runtime.set_pending(PendingRequest(request_id="r1", kind="approval", title="write /tmp/x"))
        frame = await _until(reader, "projection")
        assert frame["data"]["pending"]["request_id"] == "r1"
    finally:
        for w in (first, second):
            if w is not None:
                w.close()
        runtime.close()


@pytest.mark.asyncio
async def test_injected_sink_is_used_as_is() -> None:
    """A host that already owns a fold hands it in; the runtime builds none
    of its own and serializes the injected projection."""
    from local_operator.mobile.projection import ProjectionFold

    handle = FakeHandle()
    fold = ProjectionFold(handle.session_projection_seed)
    fold.projection.model_label = "injected/model"
    runtime = RuntimeServer(handle, kind="tui", projection_sink=fold)
    assert runtime.projection_sink is fold
    assert runtime.fold is fold
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        runtime._schedule_push()
        frame = await _until(reader, "projection")
        assert frame["data"]["model_label"] == "injected/model"
        assert runtime.projection_sinks_built == 0
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


def test_fold_property_rejects_a_foreign_sink() -> None:
    class Stub:
        def __init__(self, projection: SessionProjection) -> None:
            self.projection = projection

        def set_pending(self, pending: Any) -> None:
            self.projection.pending = pending

    handle = FakeHandle()
    runtime = RuntimeServer(
        handle, kind="tui", projection_sink=Stub(handle.session_projection_seed)
    )
    runtime.set_pending(None)  # routed to the stub, no fold built
    assert runtime.projection_sinks_built == 0
    with pytest.raises(TypeError):
        _ = runtime.fold


class TestLiveStateReachesTheRecord:
    """`busy` and `detached` must track reality, not sit at their defaults.

    Round 1 (U2) measured a SINGLE tuple `(False, False, None)` across a whole
    turn and across a client attaching and leaving: `set_busy` had no caller
    anywhere in the tree, and `detached` was computed only inside a pending
    transition. The picker's liveness markers were therefore decorative — a
    runtime grinding through a long turn with no terminal open, the exact thing
    this release makes possible, looked identical to an idle one.
    """

    @pytest.mark.asyncio
    async def test_a_new_runtime_reports_itself_detached(self) -> None:
        """The default direction matters: no terminal has ever attached yet."""
        server = RuntimeServer(FakeHandle(), kind="daemon")
        assert server._record.detached is True

    @pytest.mark.asyncio
    async def test_busy_transitions_republish_the_record(self) -> None:
        """A transition must reach the RECORD, and only a transition may.

        Asserted on republish calls rather than on `_record.busy` because an
        unstarted server has no publisher — the record is rewritten through
        `RecordPublisher.heartbeat`, deliberately the one write path.
        """
        server = RuntimeServer(FakeHandle(), kind="daemon")
        publishes: list[bool] = []
        server._republish = lambda: publishes.append(server._busy)  # type: ignore[method-assign]

        server.set_busy(True)
        server.set_busy(True)  # unchanged: must not republish
        server.set_busy(False)

        assert publishes == [True, False]

    @pytest.mark.asyncio
    async def test_detached_is_deduplicated_on_the_boolean(self) -> None:
        """A second terminal changes nothing a reader can see.

        Asserted because the alternative — republishing per connection — puts a
        staged write on every churn of a session with two viewers.
        """
        server = RuntimeServer(FakeHandle(), kind="daemon")
        publishes: list[object] = []
        server._republish = lambda: publishes.append(1)  # type: ignore[method-assign]
        server._detached = False
        server._republish_detached()  # still 0 clients -> True: one publish
        server._republish_detached()  # unchanged: no publish
        assert len(publishes) == 1


@pytest.mark.asyncio
async def test_watching_surfaces_is_derived_from_real_connections() -> None:
    """A relay's presence is not a person. Derived from REAL dials.

    ``"daemon"`` is the default kind for an auth frame with no ``client``
    field, which is exactly what the mobile daemon's ADOPTION dial sends
    (`mobile/daemon.py::_dial`) — for every session on the machine, held open
    permanently. Counting that as "the phone is watching" meant that on any
    machine running ``lop mobile`` a parked approval sent NO notification, the
    gate held ~283 MB for 24 h, and the model was told a human was watching
    (round 3, B1).

    This test dials the server the way production does instead of injecting a
    kind set as a premise. That distinction is the whole point: four committed
    tests asserted ``frozenset({"daemon"}) -> the phone is watching`` and all
    four passed while the product was broken, because they asserted the
    premise rather than deriving it.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    daemon_writer = attach_writer = None
    try:
        record = await _wait_record()

        assert runtime.watching_surfaces() == frozenset()

        # The adoption dial: no `client` field, exactly as the daemon sends.
        _daemon_reader, daemon_writer = await _dial(record)
        assert (
            runtime.watching_surfaces() == frozenset()
        ), "a daemon adoption dial is a transport connection, not a person watching"

        # A PHONE ACTUALLY OPENING THE SESSION — the real `watch` op the
        # daemon pushes on the SSE 0->N transition, sent over the same wire
        # rather than by poking an attribute. Round 3 asserted against a
        # `note_viewer_active()` helper instead, which is why a fix with NO
        # production caller passed this test while the phone was never
        # counted as watching (round 4, R1/Q1).
        daemon_writer.write(json.dumps({"op": "watch", "req": "w1"}).encode() + b"\n")
        await daemon_writer.drain()
        await _until(_daemon_reader, "ack", "w1")
        assert runtime.watching_surfaces() == frozenset({"viewer"})

        # And closing it again: the same op in reverse, not a TTL expiry.
        daemon_writer.write(json.dumps({"op": "unwatch", "req": "w2"}).encode() + b"\n")
        await daemon_writer.drain()
        await _until(_daemon_reader, "ack", "w2")
        assert (
            runtime.watching_surfaces() == frozenset()
        ), "closing the session on the phone must stop counting as watching"

        # A real terminal.
        _attach_reader, attach_writer = await _dial(record, client="attach")
        assert runtime.watching_surfaces() == frozenset({"attach"})
    finally:
        for writer in (daemon_writer, attach_writer):
            if writer is not None:
                writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_daemon_that_dies_without_unwatch_stops_counting_as_watching() -> None:
    """The case three rounds of tests never asked: a daemon connection that
    goes away UNCLEANLY.

    `phone_watchers` is the daemon connection's state held in a server-global
    counter, and the only decrement is an `unwatch` op the daemon sends from
    an SSE generator's `finally` — in the process that just died. So the
    committed tests, which always send a matching `unwatch`, could not see
    that the count outlives its connection: a daemon restart while a phone is
    watching left a permanent +1, and the session reported a viewer nobody
    could see forever after. Every parked approval on it then sent no desktop
    toast and the model was told a human was watching (round 5, R5).

    That is round 3's B1 failure mode reached by a third route, which is why
    this asserts the property (`watching_surfaces()` after an unclean drop)
    rather than the counter: the counter is the mechanism, and the mechanism
    has now been wrong three different ways.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    daemon_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)

        # A phone opens the session: the real op, over the wire.
        daemon_writer.write(json.dumps({"op": "watch", "req": "w1"}).encode() + b"\n")
        await daemon_writer.drain()
        await _until(daemon_reader, "ack", "w1")
        assert runtime.watching_surfaces() == frozenset({"viewer"})

        # The daemon process DIES — no `unwatch`, because a dead process runs
        # no `finally`. This is the whole point of the test.
        daemon_writer.close()
        for _ in range(100):
            await asyncio.sleep(0.05)
            if not runtime._clients:
                break
        assert not runtime._clients, "the dropped connection was never reaped"

        assert runtime.watching_surfaces() == frozenset(), (
            "a phone cannot still be watching through a connection that no longer "
            "exists — the count belongs to the connection"
        )
    finally:
        if daemon_writer is not None:
            daemon_writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_an_evicted_daemons_late_drop_leaves_the_replacements_watchers() -> None:
    """`_drop_client` runs TWICE on one connection, and the second must be a
    no-op for server-global state.

    The contract is the codebase's own: `_send_to` drops a client whose send
    failed, and that connection's reader loop later observes the close and
    drops it again from its `finally` — "a no-op second removal". The round-5
    fix for the `phone_watchers` leak zeroed the counter UNCONDITIONALLY,
    which broke that contract for the one piece of server-global state a
    daemon owns.

    The ordering IS the defect: an evicted daemon parked inside `_on_request`
    unwinds only when its await returns, by which time the replacement has
    dialled, replayed `watch` and owns the counter. So the late drop reached
    across and wiped a LIVE watcher (round 6, R7).

    Worth stating why this is the more dangerous direction. The leak it
    replaced over-counted, which fails SAFE — a phantom viewer suppresses a
    toast. This failed OPEN: a phone genuinely being looked at reported nobody
    watching, so every parked approval toasted a card already on the user's
    screen and the model was told no one could answer. That is round 4's
    R1/Q1 failure mode by a fourth route.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer_a = writer_b = None
    try:
        record = await _wait_record()

        # Daemon A, with a phone watching through it.
        reader_a, writer_a = await _dial(record)
        writer_a.write(json.dumps({"op": "watch", "req": "a1"}).encode() + b"\n")
        await writer_a.drain()
        await _until(reader_a, "ack", "a1")
        assert runtime.watching_surfaces() == frozenset({"viewer"})
        conn_a = next(c for c in runtime._clients.values() if c.kind == "daemon")

        # Daemon A restarts: B's dial evicts A (the first, legitimate drop).
        reader_b, writer_b = await _dial(record)
        for _ in range(100):
            await asyncio.sleep(0.05)
            if len(runtime._clients) == 1:
                break
        assert len(runtime._clients) == 1, "the evicted daemon was never removed"
        assert runtime.phone_watchers == 0, "eviction must release the old daemon's count"

        # B re-announces the session it is watching, as `_reconcile` does.
        writer_b.write(json.dumps({"op": "watch", "req": "b1"}).encode() + b"\n")
        await writer_b.drain()
        await _until(reader_b, "ack", "b1")
        assert runtime.watching_surfaces() == frozenset({"viewer"})

        # A's parked reader loop finally unwinds and drops a connection that
        # is ALREADY out of the registry. This is the second removal.
        runtime._drop_client(conn_a)

        assert runtime.watching_surfaces() == frozenset({"viewer"}), (
            "a late drop of an already-evicted daemon wiped the REPLACEMENT's "
            "live watcher count — a phone that is being looked at now reports "
            "nobody watching"
        )
        assert runtime.phone_watchers == 1
    finally:
        for writer in (writer_a, writer_b):
            if writer is not None:
                writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_watch_frame_buffered_behind_a_parked_op_cannot_move_the_count() -> None:
    """A `watch`/`unwatch` that arrives after its connection is gone is inert.

    Closing a connection does not stop the frames it already sent. The reader
    loop is strictly serial — `readline()` then `await _on_request(...)` — so
    while an op is parked, anything the daemon wrote sits in the socket
    buffer; `_drop_client` closes the WRITER, but the `StreamReader` keeps
    yielding those lines. `_on_request` therefore runs on a connection that is
    no longer in the registry.

    Production produces exactly this ordering: `notify_watch_transition`
    pushes `unwatch` from the SSE generator's `finally` IN THE DAEMON THAT IS
    DYING, while the relaunched daemon dials and replays `watch`. The late
    frame then wiped the replacement's live count (round 7, R8) — the fifth
    instance of this predicate failing OPEN, where a phone genuinely being
    looked at reports nobody watching, so a parked approval toasts a card
    already on screen and the model is told no one can answer.

    The R7 guard closed the `_drop_client` path only; this is the request
    path, which is a different context onto the same server-global counter.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        writer.write(json.dumps({"op": "watch", "req": "w1"}).encode() + b"\n")
        await writer.drain()
        await _until(reader, "ack", "w1")
        assert runtime.phone_watchers == 1
        conn = next(c for c in runtime._clients.values() if c.kind == "daemon")

        # The connection goes away (eviction by a replacement daemon, or any
        # other drop). Its buffered frames have NOT gone away with it.
        runtime._drop_client(conn)
        assert runtime.phone_watchers == 0

        # A replacement daemon dials and replays its watch, so the count is
        # live again and owned by a different connection.
        reader2, writer2 = await _dial(record)
        writer2.write(json.dumps({"op": "watch", "req": "w2"}).encode() + b"\n")
        await writer2.drain()
        await _until(reader2, "ack", "w2")
        assert runtime.watching_surfaces() == frozenset({"viewer"})

        # Now the dead connection's buffered frame is finally processed. This
        # is the delivery the reader loop performs; it must not be able to
        # reach the live count.
        await runtime._on_request({"op": "unwatch", "req": "late"}, conn)

        assert runtime.watching_surfaces() == frozenset({"viewer"}), (
            "a frame buffered behind a parked op moved the counter after its "
            "connection was dropped, wiping the REPLACEMENT daemon's live "
            "watcher count"
        )
        assert runtime.phone_watchers == 1
        writer2.close()
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_an_attach_clients_watch_cannot_leak_into_the_phone_count() -> None:
    """Only the daemon's count is ever released, so only it may be taken.

    `_drop_client` clears the counter for `kind == "daemon"` alone, so a
    `watch` accepted from an `attach` client incremented something no drop
    path could ever clear — a phantom viewer for the lifetime of the runtime
    (round 7, R9). Unreachable today because only `mobile/daemon.py` sends the
    op, but the asymmetry is one refactor away from being live.

    An attached terminal is already represented: `watching_surfaces` derives
    `attach` from the registry, which is the shape this counter should have.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(json.dumps({"op": "watch", "req": "a1"}).encode() + b"\n")
        await writer.drain()
        await _until(reader, "ack", "a1")

        assert runtime.phone_watchers == 0, (
            "an attach client incremented the phone watcher count, which only "
            "a daemon drop can clear"
        )
        # The terminal is still reported, by the registry-derived path.
        assert "attach" in runtime.watching_surfaces()

        # `watch_supported` latches regardless: it is a version signal about
        # the peer speaking the op, not a count.
        assert runtime.watch_supported is True
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


# --- client locality ---------------------------------------------------------
#
# Some operations only make sense where the user physically is: an OAuth grant
# opens a browser tab and writes a credential into THIS machine's auth.db.
# That question cannot be answered from inside the runtime — trying to infer it
# is what made `/mcp reauth` refuse every routed invocation — so the client
# declares it and the runtime passes it to the handler.


class _LocalityHandle(FakeHandle):
    """Records the locality each routed slash command was dispatched with."""

    def __init__(self) -> None:
        super().__init__()
        self.localities: list[str] = []

    async def run_slash_authoritative(
        self, command, args, images, *, locality="local"
    ):  # noqa: ANN001, ANN202
        self.localities.append(locality)
        return {"kind": "notice", "text": f"owner ran /{command}", "style": "info"}


@pytest.mark.asyncio
async def test_a_client_that_declares_nothing_is_local() -> None:
    """Absent means local: every client today dials over loopback.

    The listener binds 127.0.0.1 only, so a client that reached the runtime is
    on its machine by construction. An older client that never heard of the
    field must therefore keep working, not lose its grants.
    """
    handle = _LocalityHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(
            json.dumps(
                {"op": "slash_result", "req": 3, "command": "mcp", "args": "reauth n"}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 3)
        assert handle.localities == ["local"]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_relay_can_declare_its_client_remote() -> None:
    """The seam a future mobile relay needs: forward the phone's position.

    Without this the runtime would have to guess, and the only guess available
    ("a routed command came from elsewhere") is the wrong one for every client
    that exists today.
    """
    handle = _LocalityHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach", locality="remote")
        writer.write(
            json.dumps(
                {"op": "slash_result", "req": 4, "command": "mcp", "args": "reauth n"}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 4)
        assert handle.localities == ["remote"]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_handle_that_does_not_take_locality_still_works() -> None:
    """Back-compat: the parameter is probed, never forced.

    A handle is an injected collaborator, so widening the call unconditionally
    would break every implementation not updated in lockstep — including the
    reduced doubles this suite is built on.
    """
    handle = FakeHandle()  # its run_slash_authoritative takes three positionals
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach", locality="remote")
        writer.write(
            json.dumps({"op": "slash_result", "req": 5, "command": "goal", "args": ""}).encode()
            + b"\n"
        )
        await writer.drain()
        frame = await _until(reader, "result", 5)
        assert frame["data"]["text"] == "owner ran /goal"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()
