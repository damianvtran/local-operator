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
import logging
import statistics
import threading
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, cast

import pytest

from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    TranscriptEntry,
)
from local_operator.session.frontend_state import FrontendSubscription
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
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

    def subscribe_frontend(  # noqa: ANN001, ANN202
        self, on_update, *, display_window=False
    ) -> FrontendSubscription | Awaitable[FrontendSubscription]:
        # `display_window` mirrors the real `Session.subscribe_frontend`, which
        # the server calls with it both at connect time and on the
        # `frontend_sync` RPC. This double accepted only the positional form, so
        # every server path taking the RPC raised TypeError against it and no
        # test could reach the canonical re-snapshot at all. Capturing a durable
        # window needs a transcript this handle does not have, so the flag is
        # accepted and ignored: the sync carries canonical state without a
        # display window, which is exactly the `window is None` branch
        # `_load_frontend_history` already handles.
        #
        # The AWAITABLE half of the return type is the server's own contract, not
        # a loosening for tests: `_serve_frontend_sync` probes the result with
        # `inspect.isawaitable` because a real handle may bind through a hop, and
        # a double that needs to hold the bind open (see `_HeldBindHandle`) uses
        # exactly that shape.
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
            if getattr(event, "error", None):
                outcome = "error"
            elif getattr(event, "aborted", False):
                outcome = "aborted"
            else:
                outcome = "completed"
            self._frontend.mutate(streaming=False, last_turn_outcome=outcome)
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

    async def set_model_effort(self, provider, model_id, effort):  # noqa: ANN001, ANN202
        return await self._record("set_model_effort", provider, model_id, effort)

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return await self._record("set_effort", effort)

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        return await self._record("slash", command, args)

    async def complete_aside(self, turns) -> str:  # noqa: ANN001
        # ``-> str`` rather than leaving it to inference: an inferred
        # ``Literal["aside answer"]`` makes every subclass that returns different
        # text an incompatible override (reportIncompatibleMethodOverride).
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


async def _on_runtime_loop(runtime: RuntimeServer, coro: Coroutine[Any, Any, Any]) -> Any:
    """Drive a runtime-owned coroutine from the runtime's OWN event loop.

    ``RuntimeServer.start()`` hosts the runtime on a dedicated thread with its own
    loop, and the send path owns objects that belong to that loop (``conn.send_lock``
    and the connection's ``StreamWriter``). Awaiting such a coroutine from the test's
    loop is the cross-loop misuse ``_send_to`` now refuses, and its failure mode was
    not an error: with the lock contended, the test loop parks a waiter future of its
    OWN, the owner's ``Lock.release()`` completes it with ``set_result`` from the wrong
    thread, and that callback is scheduled with plain ``call_soon`` — no self-pipe
    write — so a loop already in ``select()`` is never woken and the await never
    returns. Ten CI shard jobs were cancelled on exactly that park. Every production
    caller of a control op or a repaint is already on the owner's loop; this is how a
    test gets there. ``start_in_process`` hosts are unaffected and need no hop.
    """
    loop = runtime._loop
    assert loop is not None, "start() publishes the runtime's loop"
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


#: The ops a WELCOME may arrive as, for assertions that are not about the
#: welcome itself. A connection asking for BOTH the canonical frontend and the
#: raw event stream is welcomed with the identity-only ``welcome`` frame
#: (``RuntimeServer._slim_welcome_frame``) instead of a full capped projection,
#: because that client shape reads the welcome for identity alone; every other
#: shape keeps the projection. A test whose subject IS the welcome asserts the
#: exact op rather than this set.
_WELCOME_OPS = ("projection", "welcome")


async def _dial(
    record: registry.SessionRecord,
    *,
    client: str | None = None,
    locality: str | None = None,
    slash_consumers: list[str] | None = None,
):
    """Open + auth one connection; consume the welcome projection."""
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
    auth: dict[str, object] = {"key": record.control_key}
    if client is not None:
        auth["client"] = client
    if locality is not None:
        auth["locality"] = locality
    if slash_consumers is not None:
        auth["slash_consumers"] = slash_consumers
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
        assert json.loads(await attach_reader.readline())["op"] in _WELCOME_OPS
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
        # The ATTACH welcome is identity-only (``_slim_welcome_frame``): a full-TUI
        # client discards the projection, so what it carries is the identity with
        # EMPTY collections — no gate overlay to render, and no roster to walk.
        # What the follower actually reads is the canonical snapshot below, which
        # is the assertion this test is about; the daemon's projection overlay is
        # checked at the end of the walk.
        assert follower["op"] == "welcome", follower
        assert follower["data"]["pending"] is None, follower
        assert follower["data"]["subagents"] == [], follower
        assert sync["data"]["snapshot"]["pending_gate"] is None

        handle._frontend.mutate(
            pending_gate=PendingRequest(
                request_id="approval-1", kind="approval", title="bash", detail="echo hi"
            ).to_json()
        )
        update = await _until(attach_reader, "frontend_update")
        assert update["data"]["changes"]["pending_gate"]["request_id"] == "approval-1"
        await _on_runtime_loop(runtime, runtime._push())
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
        assert json.loads(await reader.readline())["op"] in _WELCOME_OPS
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
    """A `lop send` peer_message reaches the handle with mode/wake/sender parsed

    — for a session that HAS been engaged. ``set_record_started`` is called
    first because the receive side now refuses an unengaged session outright
    (see the refusal test below); this test is about the frame's arguments.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    runtime.set_record_started(True)
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


class NoEffortHandle(FakeHandle):
    """An owner runtime that predates a chosen reasoning level.

    The dispatch probes ``set_model_effort`` with getattr, so a handle that
    simply lacks the method must still answer the two-argument switch it does
    implement — the same optional-capability contract ``NoPeerHandle`` and
    ``recall_steer`` document. The consequence, stated rather than discovered:
    against such an owner the chosen level is not applied, and the turn runs at
    the model's own default level.
    """

    set_model_effort = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_a_chosen_effort_dispatches_to_the_optional_capability() -> None:
    """The level rides the SAME frame as the pair, and only when one was chosen.

    Two frames, one op: with a level it must reach the effort-carrying method,
    and without one it must stay byte-for-byte the call every owner has always
    received (the frame carries no ``effort`` key at all in that case, which is
    what an older viewer sends).
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        for req, frame in (
            (
                21,
                {
                    "op": "set_model",
                    "req": 21,
                    "provider": "deepseek",
                    "model_id": "deepseek-flash",
                    "effort": "max",
                },
            ),
            (
                22,
                {
                    "op": "set_model",
                    "req": 22,
                    "provider": "deepseek",
                    "model_id": "deepseek-flash",
                },
            ),
        ):
            writer.write(json.dumps(frame).encode() + b"\n")
            await writer.drain()
            await _until(reader, "ack", req)
        assert handle.calls[0][0:2] == (
            "set_model_effort",
            ("deepseek", "deepseek-flash", "max"),
        )
        assert handle.calls[1][0:2] == ("set_model", ("deepseek", "deepseek-flash"))
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_an_owner_without_the_effort_capability_keeps_its_plain_switch() -> None:
    handle = NoEffortHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record)
        writer.write(
            json.dumps(
                {
                    "op": "set_model",
                    "req": 23,
                    "provider": "deepseek",
                    "model_id": "deepseek-flash",
                    "effort": "max",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert (await _until(reader, "ack", 23))["detail"] == "set_model ok"
        assert handle.calls[-1][0:2] == ("set_model", ("deepseek", "deepseek-flash"))
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_peer_message_to_an_unengaged_session_is_refused() -> None:
    """THE NEW RULE, receive side: a runtime whose record says ``started=False``
    (a fresh ``/new`` in the composer) refuses a ``peer_message`` op.

    This is the only layer that holds against a sender on an OLDER build: such
    a sender reads a pre-field record's missing ``started`` key as True, resolves
    it and dials — so the row can be refused here or not at all, and a row
    written here would become the OPENING row of a conversation its owner never
    started. The refusal rides the ordinary error frame, which both send
    surfaces render as ``could not deliver: ...``.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        assert record.started is False
        reader, writer = await _dial(record)
        writer.write(
            json.dumps({"op": "peer_message", "req": 13, "text": "hi", "mode": "mailbox"}).encode()
            + b"\n"
        )
        await writer.drain()
        err = await _until(reader, "error", 13)
        assert "has not been engaged yet" in err["message"]
        assert "no user message has been sent in it" in err["message"]
        assert "no live session" not in err["message"]
        # The label follows the ADDRESS the sender used: a dial arrives at this
        # runtime's pid, so the refusal names the pid rather than a session id
        # the sender never typed (review round 1, F-5).
        assert f"pid {record.pid} has not been engaged yet" in err["message"], err["message"]
        assert handle.calls == [], "nothing may reach the handle for an unengaged session"

        # And the SAME frame is delivered the moment the session runs a turn,
        # with no re-dial logic anywhere: the bit is the whole state.
        runtime.set_record_started(True)
        writer.write(
            json.dumps({"op": "peer_message", "req": 14, "text": "hi", "mode": "mailbox"}).encode()
            + b"\n"
        )
        await writer.drain()
        ack = await _until(reader, "ack", 14)
        assert "mailbox" in ack["detail"]
        assert handle.calls[-1][0] == "receive_peer_message"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


class NoPeerHandle(FakeHandle):
    """An owner runtime that predates peer messaging: no receive_peer_message.

    The dispatch probes the capability with getattr, so a handle that simply
    lacks the method must surface the clear "cannot receive" error rather than
    an AttributeError — exactly the optional-capability contract recall_steer
    documents.
    """

    receive_peer_message = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_peer_message_on_handle_without_capability_errors_cleanly() -> None:
    handle = NoPeerHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    # The capability probe is reached only by an ENGAGED session now, so the
    # engagment gate is satisfied first; the point here is the missing method.
    runtime.set_record_started(True)
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
async def test_thread_mode_close_wakes_the_serve_loop_instead_of_waiting_out_its_poll() -> None:
    """A thread-mode close must not inherit the loop's wait interval.

    This is the same defect the viewer endpoint had: ``close()`` joins the serve
    thread, and that thread decided it had been asked to stop by re-checking the
    latch on a 200 ms ``sleep``. Measured on the poll (n=15, isolated, the close
    landing at an arbitrary phase): median 206 ms, min 192 ms, max 215 ms — of
    which ~150 ms remains when the close is aimed 50 ms into the wait, which is
    the shape below.

    WALL time, deliberately, and not ``time.thread_time()``: the old cost was
    blocked in a sleep, which consumes no CPU, so a CPU-time instrument would
    report ~0 ms for exactly the code this test exists to reject.

    WHAT THIS GUARANTEES, AND WHAT IT DOES NOT. Identical in shape to the viewer
    endpoint's test: same 50 ms phase GUESS, same five fresh servers, same
    MEDIAN aggregate and the same reason for it — a single starved round on the
    polled code returns in ~1 ms and must not carry the verdict, while a single
    slow round on the fixed code must not false-fail it. The reasoning and its
    load measurements are stated ONCE, in
    ``test_viewer_routing.test_close_wakes_the_serve_loop_instead_of_waiting_out_its_poll``;
    this loop's own numbers only: on a loaded dev host a single round reached
    ~101 ms, ABOVE the 100 ms ceiling, so there is no per-round margin here
    either — the aggregate is what carries the test (measured medians 1-7 ms).

    Thread mode specifically, because the in-process path never parks in
    ``_closed_wait``: it schedules its teardown on the owning loop and has no
    poll to remove.
    """
    samples: list[float] = []
    for _ in range(5):
        runtime = RuntimeServer(FakeHandle(), kind="tui")
        try:
            # start() inside the try: a runtime that fails to start must still
            # be closed by this round's ``finally``.
            runtime.start()
            # Tight-poll the record rather than using ``_wait_record()``, which
            # waits in 50 ms steps. The overshoot matters HERE and nowhere else:
            # the polled code's cost is the REMAINDER of the loop's wait
            # interval, so every 50 ms of overshoot before the settle below
            # moves the close toward the end of that interval and shrinks the
            # floor the base arm owes — measured with the 50 ms cadence, the old
            # code came in at 101-103 ms against the 100 ms ceiling, one
            # scheduling nudge away from the false pass this test exists to
            # prevent. Polling at 5 ms keeps the base arm at ~140-190 ms.
            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                found = registry.scan()
                if found and found[0][1] == "live":
                    break
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("runtime never published a live record")
                await asyncio.sleep(0.005)

            # The live record proves a listener was PUBLISHED, not that it
            # ANSWERS: dial the port it advertises, so a round whose listener
            # died after publishing cannot pass while measuring nothing.
            # CONNECTED is what the record exists for; closing without a frame
            # registers no client (the daemon path only runs after a successful
            # auth).
            probe_reader, probe_writer = await asyncio.open_connection(
                "127.0.0.1", found[0][0].control_port, limit=1 << 20
            )
            probe_writer.close()
            await probe_writer.wait_closed()

            # The record proves the listener is PUBLISHED, not that the loop has
            # parked in ``_closed_wait()`` — ``_serve`` publishes first and parks
            # after — so the loop's phase must still be settled before timing or
            # the close can land before it parks: the latch is then already set
            # when the loop gets there, it never waits, and the POLLED code
            # returns in ~1 ms too (the false negative the median above exists
            # to absorb). Parked, the poll owes the rest of a 200 ms
            # interval and cannot meet the ceiling. Awaited, not ``sleep``: the
            # runtime owns its own thread and loop, and the test has no reason to
            # block its own.
            await asyncio.sleep(0.05)

            started = time.monotonic()
            runtime.close()
            samples.append(time.monotonic() - started)
        finally:
            runtime.close()

    aggregate = statistics.median(samples)
    assert aggregate < 0.1, (
        f"the median of {len(samples)} parked closes took {aggregate * 1000:.0f} ms "
        "— close() is waiting out the serve loop's wait instead of waking it. "
        "Samples (ms): "
        f"{', '.join(f'{sample * 1000:.1f}' for sample in samples)}. Parked, the "
        "pre-fix poll measured through this test's own 50 ms settle lands in "
        "139.5-191.2 ms (median ~151 ms, n=50), so a median under the ceiling "
        "needs three of these five rounds woken early — the loop is not waiting "
        "(server.close -> _request_close -> _wake_close_wait -> _closed_wait)."
    )


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
        assert json.loads(await slow_reader.readline())["op"] in _WELCOME_OPS
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
        assert json.loads(await healthy_reader.readline())["op"] in _WELCOME_OPS
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


async def _until_push(
    reader: asyncio.StreamReader,
    want: object,
    *,
    activity: str | None = None,
    schedule: Callable[[], None] | None = None,
    deadline_s: float = 60.0,
) -> dict[str, Any]:
    """Read pushed projections until one carries ``want`` as the band's age.

    A push is a whole repaint and several can be in flight for one change, so
    waiting for the value under test is the only assertion that names the frame
    it means; the failure message carries the last value seen.

    The wait drives its own deadline and RE-ASKS for a repaint (``schedule``)
    rather than trusting one event's delivery, because the push is coalesced
    onto the runtime's own loop: a single scheduling lost its frame on a loaded
    CI shard (PR #1241, ``test (3.12, 1)``), and that is a property of the wait,
    not of the age under test.

    ``activity`` matches the BAND LABEL as well as the age, and it has to: two
    phases in a row both start at a known zero (``thinking`` then
    ``responding``), so "the first frame carrying 0.0" is not the edge a caller
    means — the label is what names it.
    """
    last: object = "<no frame>"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while loop.time() < deadline:
        if schedule is not None:
            schedule()
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=2)
        except TimeoutError:
            continue
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            continue
        frame = json.loads(text)
        if frame.get("op") != "projection":
            continue
        seen = frame["data"].get("activity")
        last = frame["data"].get("activity_started_s")
        if last == want and (activity is None or seen == activity):
            return frame
        last_pair = f"{seen!r}/{last!r}"
        last = f"activity {last_pair}"
    raise AssertionError(
        f"no pushed frame carried activity_started_s={want!r}"
        + (f" for phase {activity!r}" if activity else "")
        + f" (last {last!r})"
    )


@pytest.mark.asyncio
async def test_a_pushed_frame_carries_the_bands_age_from_the_fold_events_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 4, BLOCKER 1 + MINOR 1: the PUSHED age, off the wire the daemon reads.

    Every other band-age assertion in the tree drives the fold directly or reads
    the handle's seed, and the runtime's own push path is where review round 4
    found the blocker: it re-dated through ``self._projection_sink`` — in
    production a SECOND fold the runtime builds over the handle's projection
    object and never feeds — so the empty state of that fold overwrote the live
    age with ``None`` on every frame build, and the phone withheld its clock for
    every phase, watched edges included.

    So this drives the production path end to end: a real ``TuiSessionHandle``
    over a real session shape, a real ``RuntimeServer``, a real daemon-kind dial
    (which is what builds the runtime's own sink), real harness events through
    the handle's stream, and the age read off the frames the daemon receives.
    """
    import local_operator.mobile.projection as projection_module
    from local_operator.harness.types import (
        AgentEndEvent,
        AgentStartEvent,
        Message,
        MessageUpdateEvent,
    )
    from local_operator.mobile.tui_handle import TuiSessionHandle
    from tests.unit.mobile.test_projection import _StubClock
    from tests.unit.tui.test_app_pilot import FakeSession

    class App:
        def __init__(self, session: Any) -> None:
            self._session = session

        def call_from_thread(self, callback: Any) -> None:
            callback()

    clock = _StubClock()
    monkeypatch.setattr(projection_module, "time", clock)
    session = FakeSession()
    session.streaming = True
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="daemon")
        assert runtime.projection_sink is not None, "a daemon dial is what builds the sink"

        session.emit(AgentStartEvent(generation=1))
        session.emit(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
        frame = await _until_push(
            reader, 0.0, activity="responding", schedule=runtime._schedule_push
        )
        assert frame["data"]["activity"] == "responding"
        assert frame["data"]["activity_started_s"] == 0.0, "a watched edge publishes a KNOWN zero"

        # 45 s of prose with no band event in it: the pushed age must be the
        # PHASE's, not the runtime's own empty fold's state and not the last
        # edge's zero.
        clock.advance(45)
        session.emit(MessageUpdateEvent(message=Message.assistant(), delta="more prose "))
        await _until_push(reader, 45.0, activity="responding", schedule=runtime._schedule_push)

        # An instant the fold cannot date still crosses as unknown, never as a zero.
        session.emit(AgentEndEvent(generation=1))
        await _until_push(reader, None, schedule=runtime._schedule_push)
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
    async def test_a_new_runtime_reports_itself_unstarted(self) -> None:
        """A fresh boot (a ``/new`` session in the composer) has run no turn."""
        server = RuntimeServer(FakeHandle(), kind="daemon")
        assert server._record.started is False

    def _handle_over_transcript(self, tmp_path, name: str, rows: list[str]) -> FakeHandle:
        """A FakeHandle whose owned session reads a transcript file on disk.

        The boot-seed path probes ``handle._session`` (the owned-handle shape
        — the daemon child and exec) and derives ``started`` from the
        transcript FILE via the session's declared ``transcript_path``, so the
        fake only needs the path to be real.
        """
        from types import SimpleNamespace

        path = tmp_path / name / "transcript.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row + "\n" for row in rows))
        handle = FakeHandle()
        handle._session = SimpleNamespace(transcript_path=path)  # type: ignore[attr-defined]
        return handle

    def _tui_handle_over_transcript(self, tmp_path, name: str, rows: list[str]) -> Any:
        """A TUI-SHAPED handle: ``_session`` is a METHOD, as ``TuiSessionHandle``'s is.

        The shape the seed used to be blind to (review round 1, F-1):
        ``getattr`` bound the function, ``has_durable_history`` read
        ``transcript_path`` off it, got ``None`` and answered False — for EVERY
        TUI window, including a ``lop --resume <sid>`` boot over a conversation
        with hundreds of rows. The phone must follow a ``/new``/``/resume``
        swap, which is why the TUI reads its session per call instead of
        storing it, so the two shapes are both real and the seed has to answer
        for both.
        """
        from types import SimpleNamespace

        path = tmp_path / name / "transcript.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row + "\n" for row in rows))
        session = SimpleNamespace(transcript_path=path)

        class _TuiShaped(FakeHandle):
            def _session(self) -> Any:  # noqa: ANN401 — the handle's own shape
                return session

        return _TuiShaped()

    @pytest.mark.asyncio
    async def test_a_tui_shaped_handle_seeds_started_from_durable_history(self, tmp_path) -> None:
        """F-1, the direction that was broken: a TUI-shaped handle over a
        conversation that has run turns publishes ``started=True`` from its
        FIRST publish.

        ``lop --resume <sid>`` is exactly this shape — the TUI's first session
        is adopted before any ``rebind`` — so before the fix its record said
        ``started=false`` over a live conversation and the peer gate refused
        sends to it with a sentence that was false about it.
        """
        handle = self._tui_handle_over_transcript(
            tmp_path,
            "tui-resumed",
            [
                '{"id":"m1","ts":1,"type":"custom","payload":{"custom_type":"title"}}',
                '{"id":"m2","ts":2,"type":"message","payload":{"role":"user"}}',
            ],
        )
        server = RuntimeServer(handle, kind="tui")
        assert server._started is True
        assert server._record.started is True

    @pytest.mark.asyncio
    async def test_a_tui_shaped_handle_over_a_fresh_session_boots_unstarted(self, tmp_path) -> None:
        """The other direction on the same shape: a true ``/new`` still boots
        ``started=False``, because the whole peer gate rests on that bit."""
        bookkeeping_only = self._tui_handle_over_transcript(
            tmp_path,
            "tui-fresh",
            ['{"id":"t1","ts":1,"type":"custom","payload":{"custom_type":"title"}}'],
        )
        server = RuntimeServer(bookkeeping_only, kind="tui")
        assert server._started is False
        assert server._record.started is False

    @pytest.mark.asyncio
    async def test_a_tui_shaped_handle_that_cannot_answer_boots_unstarted(self) -> None:
        """``TuiSessionHandle._session()`` RAISES before the app binds one
        (``RuntimeError("session is still starting")``). A boot must not fail
        over a seed, so the probe swallows it and keeps the conservative False
        the first real turn corrects."""

        class _StillStarting(FakeHandle):
            def _session(self) -> Any:  # noqa: ANN401 — the raising shape
                raise RuntimeError("session is still starting")

        server = RuntimeServer(_StillStarting(), kind="tui")
        assert server._started is False

    def _real_tui_handle(self, tmp_path, name: str, rows: list[str]) -> Any:
        """A REAL ``TuiSessionHandle`` over a stub app (review round 2, F-8).

        The pins above use a ``FakeHandle`` subclass with the right SHAPE
        (``_session`` as a method); this one binds the class the shape stands
        in for, which is the shape QA could not reach from outside (round 1,
        Q3). Cheap to build: the constructor reads the fields a projection
        carries and wires the ``started`` publisher only when the session has
        one, so a stub app and a session stub are enough.
        """
        from types import SimpleNamespace

        from local_operator.mobile.tui_handle import TuiSessionHandle

        path = tmp_path / name / "transcript.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row + "\n" for row in rows))
        session = SimpleNamespace(
            session_id=name,
            transcript_path=path,
            conversation_name=name,
            cwd=str(tmp_path),
        )
        # The real constructor's parameter is typed ``OperatorApp``; the stub is
        # deliberate (the handle reads only ``_session`` off the app), so the
        # ignore documents the double rather than widening the class's type.
        return TuiSessionHandle(SimpleNamespace(_session=session))  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_a_real_tui_handle_seeds_started_the_same_way(self, tmp_path) -> None:
        """F-8: the F-1 fix, on the class rather than on a stand-in for it.

        Both directions, because the enclosing gate (and the refusal that
        depends on it) rests on the fresh direction staying False.
        """
        resumed = RuntimeServer(
            self._real_tui_handle(
                tmp_path,
                "real-tui-resumed",
                [
                    '{"id":"t1","ts":1,"type":"custom","payload":{"custom_type":"title"}}',
                    '{"id":"m1","ts":2,"type":"message","payload":{"role":"user"}}',
                ],
            ),
            kind="tui",
        )
        assert resumed._started is True
        assert resumed._record.started is True

        fresh = RuntimeServer(
            self._real_tui_handle(
                tmp_path,
                "real-tui-fresh",
                ['{"id":"t1","ts":1,"type":"custom","payload":{"custom_type":"title"}}'],
            ),
            kind="tui",
        )
        assert fresh._started is False
        assert fresh._record.started is False

    @pytest.mark.asyncio
    async def test_a_handle_answering_with_something_unreadable_boots_unstarted(self) -> None:
        """F-8, second half: the DERIVATION is inside the guard too.

        The reader ends at ``durable_conversation_path``, which catches only
        ``OSError`` — so a session answering with something that is not a path
        used to raise ``TypeError`` out of ``RuntimeServer.__init__``, killing
        the boot over a seed. A record we cannot read is unengaged: the
        conservative False, corrected by the owner's first turn.
        """
        from types import SimpleNamespace

        class _Odd(FakeHandle):
            def _session(self) -> Any:  # noqa: ANN401 — an unreadable shape
                return SimpleNamespace(transcript_path=object())

        server = RuntimeServer(_Odd(), kind="tui")
        assert server._started is False
        assert server._record.started is False

    @pytest.mark.asyncio
    async def test_a_resumed_boot_seeds_started_from_durable_history(self, tmp_path) -> None:
        """QA Q3: the daemon child behind ``lop --resume <sid>`` builds its
        record at boot, and a conversation that already ran turns under an
        earlier process must read ``started=True`` from the FIRST publish —
        before the owner types anything in THIS process. Bookkeeping rows
        (a persisted title) do not count; only a MESSAGE row does, exactly
        as in ``TuiSessionHandle.rebind``'s re-seed."""
        handle = self._handle_over_transcript(
            tmp_path,
            "resumed",
            [
                '{"id":"m1","ts":1,"type":"custom","payload":{"custom_type":"title"}}',
                '{"id":"m2","ts":2,"type":"message","payload":{"role":"user"}}',
            ],
        )
        server = RuntimeServer(handle, kind="daemon")
        assert server._started is True
        assert server._record.started is True

    @pytest.mark.asyncio
    async def test_an_empty_session_directory_boots_unstarted(self, tmp_path) -> None:
        """The composer gate survives the seed: a true ``/new`` — no message
        rows, whether the transcript is empty or holds only bookkeeping —
        must still publish ``started=False`` so peer broadcasts do not drive
        a turn into a session whose owner has not typed yet."""
        empty = self._handle_over_transcript(tmp_path, "empty", [])
        bookkeeping_only = self._handle_over_transcript(
            tmp_path,
            "titled",
            ['{"id":"t1","ts":3,"type":"custom","payload":{"custom_type":"title"}}'],
        )
        for handle in (empty, bookkeeping_only):
            server = RuntimeServer(handle, kind="daemon")
            assert server._started is False
            assert server._record.started is False

    @pytest.mark.asyncio
    async def test_a_peer_note_only_transcript_boots_unstarted(self, tmp_path) -> None:
        """QA Q4: a quiet-dialled peer note persists as a MESSAGE row (kind
        ``custom``, a ``peer_message`` CustomMessage) WITHOUT a turn running,
        so a message row alone cannot seed ``started`` — that marks the
        session started, after which a peer ``--wake`` or a broadcast drives
        an assistant turn into a session the owner never typed in. Only a
        plain ``Message`` row (kind ``message``, or a legacy row that
        predates the marker) counts."""
        peer_note = (
            '{"id":"p1","ts":1,"type":"message","payload":{"kind":"custom",'
            '"custom_type":"peer_message","attribution":"user","details":{"text":"hi"}}}'
        )
        peer_notes_only = self._handle_over_transcript(tmp_path, "noted", [peer_note, peer_note])
        mixed = self._handle_over_transcript(
            tmp_path,
            "mixed",
            [
                peer_note,
                '{"id":"m1","ts":4,"type":"message","payload":{"kind":"message","role":"user"}}',
            ],
        )
        legacy = self._handle_over_transcript(
            tmp_path,
            "legacy",
            ['{"id":"m2","ts":5,"type":"message","payload":{"role":"assistant"}}'],
        )
        server = RuntimeServer(peer_notes_only, kind="daemon")
        assert server._started is False
        assert server._record.started is False
        for handle in (mixed, legacy):
            server = RuntimeServer(handle, kind="daemon")
            assert server._started is True
            assert server._record.started is True

    @pytest.mark.asyncio
    async def test_started_is_one_way_and_deduplicated(self) -> None:
        """The flag flips once and a repeat ``True`` (every turn after the
        first) publishes nothing — and a ``False`` after ``True`` is IGNORED:
        no code path un-runs a turn, so honouring it would let a mistake
        un-publish a working session. Dropping the bit is reserved for a
        session-identity swap (``reset_record_started``)."""
        server = RuntimeServer(FakeHandle(), kind="daemon")
        publishes: list[bool] = []
        server._republish = lambda: publishes.append(server._started)  # type: ignore[method-assign]

        server.set_record_started(True)
        server.set_record_started(True)  # already started: must not republish
        server.set_record_started(False)  # one-way: a started record ignores False

        assert publishes == [True]
        assert server._started is True
        assert server._record.started is True

    @pytest.mark.asyncio
    async def test_reset_record_started_reseeds_for_a_new_identity(self) -> None:
        """The rebind path: a ``/new`` drops the bit (the composer window
        returns) and a ``/resume`` raises it without ever having run a turn
        in THIS process — both legal only because the identity changed."""
        server = RuntimeServer(FakeHandle(), kind="tui")
        publishes: list[bool] = []
        server._republish = lambda: publishes.append(server._started)  # type: ignore[method-assign]

        server.set_record_started(True)  # the old conversation ran turns
        server.reset_record_started(False)  # /new: fresh composer
        assert server._started is False
        assert server._record.started is False
        server.reset_record_started(True)  # /resume of a session with history
        assert server._started is True
        assert server._record.started is True
        # And the reset True is still one-way afterwards.
        server.set_record_started(False)
        assert server._started is True
        assert publishes == [True, False, True]

    @pytest.mark.asyncio
    async def test_the_periodic_heartbeat_carries_started(self, tmp_path, monkeypatch) -> None:
        """The timer heartbeat (the one that refreshes the record's IDENTITY
        fields from the projection seed) must carry ``started`` explicitly.
        Today the publish happens to survive on ``self._record is
        publisher.record`` — an object-identity accident this pins away: a
        rebuilt or copied record must not make a working session
        broadcast-invisible one heartbeat later."""
        import local_operator.session.runtime.server as server_module

        monkeypatch.setattr(server_module, "HEARTBEAT_INTERVAL_S", 0.05)
        # Isolated config dir: ``start_in_process`` constructs a real
        # RecordPublisher before the recorder replaces it, and its writes
        # must not touch the operator's live registry.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        server = RuntimeServer(FakeHandle(), kind="tui")
        timer_calls: list[dict[str, object]] = []

        class _Recorder:
            # Stands in for RecordPublisher on the timer path only; its call
            # is told apart from ``_republish``'s by the identity fields,
            # which only the timer passes. ``close`` because teardown joins
            # the publisher it finds.
            def heartbeat(self, **kwargs: object) -> None:
                if "session_id" in kwargs:
                    timer_calls.append(dict(kwargs))

            def close(self) -> None:
                pass

        # In-process (the mobile child's own mode) so the heartbeat task
        # shares THIS loop and the test can observe its ticks directly.
        await server.start_in_process()
        try:
            # Point the loop at the recorder, flip the bit, and let a tick (or
            # several — the loop is a 50 ms timer) pass.
            server._publisher = _Recorder()  # type: ignore[assignment]
            server.set_record_started(True)
            deadline = asyncio.get_running_loop().time() + 5
            while not timer_calls and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            assert timer_calls, "the periodic heartbeat never fired"
            assert all(call.get("started") is True for call in timer_calls)
        finally:
            server.close()
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_started_survives_the_republish(self, tmp_path, monkeypatch) -> None:
        """A heartbeat rewrite carries ``started`` forward rather than resetting
        it — the field must stay True once set, or a working session would
        become broadcast-invisible again on the next heartbeat."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        server = RuntimeServer(FakeHandle(), kind="tui")
        server.start()
        try:
            await _wait_record()
            server.set_record_started(True)
            # A republish through the heartbeat path must keep the bit set.
            server._republish()
            found = registry.scan()
            assert found and found[0][0].started is True
        finally:
            server.close()


@pytest.mark.asyncio
async def test_the_record_directory_is_fixed_when_the_runtime_starts(tmp_path, monkeypatch) -> None:
    """The record belongs to the config dir the runtime was STARTED in.

    ``start`` only hands the work to a thread, so a runner loaded enough to
    delay that thread past the caller's own teardown had ``_serve`` resolve
    ``config_dir()`` at the worst possible moment, publish its record into the
    NEXT test's directory, and then delete it there on the way out — ``close``
    re-resolved the same way. Records are keyed by pid alone and an xdist
    worker keeps one pid, so that file belonged to a different session. Both
    shapes need the thread to be late, which is why this pins the DIRECTORY
    rather than trying to reproduce the scheduling delay: the answer must be
    settled at ``start`` no matter when the thread eventually runs.
    """

    started_in = tmp_path / "started-in"
    moved_to = tmp_path / "moved-to"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(started_in))
    original_serve = RuntimeServer._serve

    async def serve_after_the_world_moved(server: RuntimeServer) -> None:
        # Stand in for a worker thread the OS did not schedule promptly: by the
        # time it reaches `_serve`, the process is on another config dir.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(moved_to))
        await original_serve(server)

    monkeypatch.setattr(RuntimeServer, "_serve", serve_after_the_world_moved)
    server = RuntimeServer(FakeHandle(), kind="tui")
    server.start()
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        found: list[registry.SessionRecord] = []
        while loop.time() < deadline:
            found = [rec for rec, state in registry.scan(started_in) if state == "live"]
            if found:
                break
            await asyncio.sleep(0.02)
        assert found, "the runtime must publish into the config dir it started in"
        stray = registry.run_dir(moved_to) / f"{server._record.pid}.json"
        assert not stray.exists(), (
            "a thread that runs late must not publish where the config dir has "
            "since moved — that filename belongs to another session"
        )
    finally:
        server.close()


@pytest.mark.asyncio
async def test_the_record_directory_is_fixed_for_an_in_process_runtime(
    tmp_path, monkeypatch
) -> None:
    """The same invariant on the in-process start path.

    Here the window is REAL, not manufactured: ``_serve`` reaches its first
    yield — ``await asyncio.start_server`` — before it builds the publisher, so
    a config dir that moves while ``start_in_process`` is suspended in that
    await is exactly what a dropped pin lets the publisher resolve inside
    ``_serve``. The wrapper below moves it at that point, and the record must
    still land where this runtime started; removing the pin fails this test
    (``assert []``) while the thread-path test still passes, so the two halves
    are separately caught.
    """

    started_in = tmp_path / "started-in"
    moved_to = tmp_path / "moved-to"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(started_in))
    original_serve = RuntimeServer._serve

    async def serve_after_the_world_moved(server: RuntimeServer) -> None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(moved_to))
        await original_serve(server)

    monkeypatch.setattr(RuntimeServer, "_serve", serve_after_the_world_moved)
    server = RuntimeServer(FakeHandle(), kind="tui")
    await server.start_in_process()
    try:
        live = [rec for rec, state in registry.scan(started_in) if state == "live"]
        assert live, "the runtime must publish into the config dir it started in"
        stray = registry.run_dir(moved_to) / f"{server._record.pid}.json"
        assert not stray.exists(), (
            "an in-process runtime must not publish where the config dir has "
            "since moved — that filename belongs to another session"
        )
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_desktop_watch_lease_separates_visibility_and_notification_delivery() -> None:
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.types import DESKTOP_WATCH_LEASE_S

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    terminal = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        assert runtime.watching_surfaces() == frozenset()
        assert runtime.notification_surfaces() == frozenset()
        await desktop.desktop_watch(visible=True, can_notify=True)
        assert runtime.watching_surfaces() == frozenset({"desktop"})
        assert runtime.notification_surfaces() == frozenset({"desktop"})
        assert runtime.attach_clients() == 1
        await desktop.desktop_watch(visible=False, can_notify=True)
        assert runtime.watching_surfaces() == frozenset()
        assert runtime.notification_surfaces() == frozenset({"desktop"})
        # Advance ONLY the lease timestamp, never the process-wide clock used
        # by asyncio. A crashed renderer need not close its proxy's socket.
        conn = next(c for c in runtime._clients.values() if c.surface == "desktop")
        conn.desktop_seen -= DESKTOP_WATCH_LEASE_S + 1
        assert runtime.watching_surfaces() == frozenset()
        assert runtime.notification_surfaces() == frozenset()
        assert runtime.attach_clients() == 0
        await terminal.connect(record, "s1")
        assert runtime.watching_surfaces() == frozenset({"attach"})
        with pytest.raises(RuntimeError, match="desktop visibility"):
            await terminal.desktop_watch(visible=True, can_notify=True)
        await desktop.desktop_watch(visible=True, can_notify=False)
        assert runtime.watching_surfaces() == frozenset({"attach", "desktop"})
        assert runtime.notification_surfaces() == frozenset()
    finally:
        await desktop.detach()
        await terminal.detach()
        runtime.close()


# ---------------------------------------------------------------------------
# ATTACHED vs ATTENDED (docs/design/attached-interface-signal.md §2)
#
# The incident: a desktop app that was open, FOCUSED and VISIBLE, holding a live
# lease on this session's conversation, whose machine-wide record could not NAME
# the conversation it was showing (``session_id: ""``) — so the attention
# predicate denied, the model-facing probe answered False, and at least twenty
# live sessions carried a block telling their model ``nobody is watching a
# screen``. The operator was reading one of them.
#
# Two questions are pinned apart here, and each consumer reads exactly one:
# ATTACHED ("an interface can PRESENT a question") for the model; ATTENDED ("a
# person is looking right now") for notification rung 1, unchanged.
# ---------------------------------------------------------------------------


@contextmanager
def _desktop_presence_claim(
    root, *, session_id: str, focused: bool = True, visible: bool = True
) -> Iterator[None]:
    """Publish the machine-wide delivery record the incident was read from (§1.1).

    Written through the REAL publisher, so the record carries this process's pid
    and a fresh heartbeat — the file's shape is the contract between the app and
    this reader, and a hand-built dict would stop testing it. Closed on exit:
    ``close`` withdraws only the publisher's OWN record (R6) and reaps its beat
    task, so nothing outlives the test.
    """
    from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
    from local_operator.session.runtime import presence

    publisher = DesktopDeliveryPublisher(root)
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete", "error"],
        session_id=session_id,
        window={"exists": True, "focused": focused, "visible": visible, "minimized": False},
    )
    # The reader caches its answer for PRESENCE_CACHE_TTL_S, so a write must
    # never be masked by an answer taken before it.
    presence.reset_cache()
    try:
        yield
    finally:
        publisher.close()
        presence.reset_cache()


def _desktop_connection(runtime: RuntimeServer):
    """The live desktop attach connection in this runtime's table."""
    return next(c for c in runtime._clients.values() if c.surface == "desktop")


@pytest.mark.asyncio
async def test_a_focused_desktop_pane_that_cannot_name_its_conversation_is_still_attached(
    monkeypatch, tmp_path
) -> None:
    """THE INCIDENT REPRODUCTION (§1.1, §2.3).

    A desktop app with a focused, visible window, a live delivery lease and this
    session's pane mounted, whose record cannot name the conversation, must count
    as ATTACHED — and must still count as UNATTENDED, because focus/visibility on
    a machine-wide record is rung 1's business, not the model's.
    """
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    with _desktop_presence_claim(tmp_path, session_id=""):
        runtime.start()
        try:
            record = await _wait_record()
            await desktop.connect(record, "s1")
            await desktop.desktop_watch(visible=False, can_notify=True)
            assert runtime.attached_surfaces() == frozenset({"desktop"})
            # Tier B deliberately does NOT move here: `desktop_visible` is false,
            # so nobody is looking at this session — which is what routing needs.
            assert runtime.watching_surfaces() == frozenset()
        finally:
            await desktop.detach()
            runtime.close()


@pytest.mark.asyncio
async def test_an_unfocused_desktop_pane_is_attached_though_nobody_is_watching(
    monkeypatch, tmp_path
) -> None:
    """§1.3: focus is the wrong question for the model, and it FLAPS.

    A window that is visible but not focused (the operator is reading a terminal
    beside it) reports ``attended=False``, so rung 1 correctly sees no watcher —
    while the pane is mounted and a card painted now would be there when they
    look. Tier A must answer across that difference; Tier B must not.
    """
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    with _desktop_presence_claim(tmp_path, session_id="s1", focused=False):
        runtime.start()
        try:
            record = await _wait_record()
            await desktop.connect(record, "s1")
            await desktop.desktop_watch(visible=True, can_notify=True)
            assert runtime.attached_surfaces() == frozenset({"desktop"})
            assert runtime.watching_surfaces() == frozenset()
        finally:
            await desktop.detach()
            runtime.close()


@pytest.mark.asyncio
async def test_a_multiplexed_terminal_away_from_this_session_is_attached_but_not_attended() -> None:
    """A viewer holding this session in a tab it is not displaying (§1.3).

    Tier A counts the connection, because that process can paint the card the
    moment the operator switches to it; Tier B must keep dropping it, which is
    the whole of the ``viewer_watch`` fix it was written for.
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    terminal = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _wait_record()
        await terminal.connect(record, "s1")
        await terminal.viewer_watch(displaying=False)
        assert runtime.attached_surfaces() == frozenset({"attach"})
        assert runtime.watching_surfaces() == frozenset()
    finally:
        await terminal.detach()
        runtime.close()


@pytest.mark.asyncio
async def test_a_daemon_connection_is_never_attached() -> None:
    """The mobile daemon's ADOPTION dial covers every session on the machine.

    Counting it would let a machine running ``lop mobile`` report an interface on
    every session, including one nobody has ever opened — the same reasoning as
    rung 1's (``server.py::watching_surfaces``), applied to the model-facing
    question because a model told "a question WILL be presented" has to be told
    the truth.
    """
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        _reader, writer = await _dial(record)
        assert runtime.attached_surfaces() == frozenset()
        assert runtime.watching_surfaces() == frozenset()
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_fifty_real_focus_changes_move_neither_tier_a_nor_its_answer() -> None:
    """FIFTY REAL FOCUS CHANGES ON THE REAL WIRE (round 1, MINOR 5 / NIT 8).

    The churn requirement is that raising and lowering the window cannot move the
    block inside the persisted system prompt. The desktop arm used to be
    ``lease and (visible or can_notify)``, and the wire's ``visible`` is already
    the app's ``visibilityState === 'visible' && hasFocus()`` — so on a host with
    no OS-notification channel that arm WAS focus, and fifty flaps flipped the
    model-facing answer fifty times, writing a ``[session-state]`` row each way.

    The churn test in ``test_prompts_api`` cannot see this: it feeds the renderer
    a constant. This one drives the real ``desktop_watch`` handler with both
    notification configurations and both window states, and asserts the answer
    never moves — while the ATTENTION tier, which is supposed to follow the
    window, does.
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        answers: set[frozenset[str]] = set()
        for can_notify in (False, True):
            for index in range(50):
                await desktop.desktop_watch(visible=bool(index % 2), can_notify=can_notify)
                answers.add(runtime.attached_surfaces())
        assert answers == {frozenset({"desktop"})}

        # The two tiers stay separable: the window is now unfocused/hidden, so
        # nothing is being LOOKED AT — while the pane is still the surface a
        # question would appear on.
        await desktop.desktop_watch(visible=False, can_notify=True)
        assert runtime.watching_surfaces() == frozenset()
        assert runtime.attached_surfaces() == frozenset({"desktop"})
    finally:
        await desktop.detach()
        runtime.close()


@pytest.mark.asyncio
async def test_the_reaper_still_sees_no_viewer_without_a_visible_panel() -> None:
    """THE NEGATIVE THAT MUST NOT MOVE: residency was not loosened (§1.5).

    ``attach_clients()`` is the predicate that keeps a runtime resident. A
    desktop connection that is neither visible nor notification-capable is not a
    front end, so it still counts for nothing THERE — a runtime with no panel on
    screen still exits.

    TIER A SPLITS FROM IT HERE, deliberately (round 1, MINOR 5). The LEASE is a
    heartbeat that names this session's own subscription and is withdrawn when
    the pane leaves, so "lease live" IS "a pane holds this conversation" — the
    fact the model-facing block is about — and ``visible``/``can_notify`` are
    attention and reachability, which is why the two predicates are no longer the
    same expression. The churn the old clause caused is the reason it had to
    change: ``desktop_visible`` is the app's ``visible && focused``, so on a host
    with no notification channel the arm collapsed to focus, and raising and
    lowering that window moved the block inside the persisted system prompt.
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=False, can_notify=False)
        assert runtime.attach_clients() == 0
        assert runtime.attached_surfaces() == frozenset({"desktop"})
    finally:
        await desktop.detach()
        runtime.close()


@pytest.mark.asyncio
async def test_a_presence_record_that_cannot_name_a_session_falls_back_to_the_connection(
    monkeypatch, tmp_path
) -> None:
    """ABSENCE OF EVIDENCE IS NOT EVIDENCE AGAINST (§2.3).

    The publisher blanks ``session_id`` whenever it cannot vouch for it, so an
    empty field says nothing about which conversation is on screen. The
    per-connection flag is the per-session answer, and falling through to it is
    the pre-presence behaviour the reader's docstring promises an older app.
    """
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    with _desktop_presence_claim(tmp_path, session_id=""):
        runtime.start()
        try:
            record = await _wait_record()
            await desktop.connect(record, "s1")
            await desktop.desktop_watch(visible=True, can_notify=True)
            assert runtime._desktop_visible(_desktop_connection(runtime)) is True
            assert runtime.watching_surfaces() == frozenset({"desktop"})
        finally:
            await desktop.detach()
            runtime.close()


@pytest.mark.asyncio
async def test_a_presence_record_naming_another_session_still_denies(monkeypatch, tmp_path) -> None:
    """The denied direction is preserved where the record IS evidence.

    An app that names a DIFFERENT conversation must not suppress this session's
    background banner: that is exactly the case the record exists to catch.
    """
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    with _desktop_presence_claim(tmp_path, session_id="some-other-session"):
        runtime.start()
        try:
            record = await _wait_record()
            await desktop.connect(record, "s1")
            await desktop.desktop_watch(visible=True, can_notify=True)
            assert runtime._desktop_visible(_desktop_connection(runtime)) is False
            assert runtime.watching_surfaces() == frozenset()
        finally:
            await desktop.detach()
            runtime.close()


@pytest.mark.asyncio
async def test_a_presence_record_that_cannot_name_a_session_does_not_suppress_an_unfocused_banner(
    monkeypatch, tmp_path
) -> None:
    """The fallback is SCOPED: an unfocused window still banners (§2.3).

    Falling through to ``conn.desktop_visible`` only grants when the pane really
    is visible, so a window behind another app keeps raising the OS banner —
    otherwise the empty field would silence every surface for a conversation
    nobody is looking at, which is the defect the record was built to stop.
    """
    from local_operator.mobile.attach_client import AttachClient

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    with _desktop_presence_claim(tmp_path, session_id=""):
        runtime.start()
        try:
            record = await _wait_record()
            await desktop.connect(record, "s1")
            # THE UNFOCUSED CASE IS EXPRESSED THROUGH ``visible=False``, NOT
            # ``focused=False``, because that is what the app actually SENDS: the
            # desktop host computes the wire ``visible`` as
            # ``visibilityState === 'visible' && hasFocus()``
            # (``local-operator-ui/src/main/desktop-notifier.ts``), so focus is
            # already folded into it. Flipping the RECORD's ``focused=`` would
            # test a different seam (Tier B's presence read), not this one.
            await desktop.desktop_watch(visible=False, can_notify=True)
            assert runtime._desktop_visible(_desktop_connection(runtime)) is False
            assert runtime.watching_surfaces() == frozenset()
            # ...and the app is still reachable for the out-of-band toast.
            assert runtime.notification_surfaces() == frozenset({"desktop"})
            # Attached all the same: the pane is mounted, so a question is
            # presentable the moment the operator returns to it.
            assert runtime.attached_surfaces() == frozenset({"desktop"})
        finally:
            await desktop.detach()
            runtime.close()


# ---------------------------------------------------------------------------
# THE TRANSPORT-BOUND HOLE (round 3): THE ATTACH FACT IS SESSION-SCOPED
#
# Every desktop fact on RuntimeServer was per-CONNECTION, so a drop erased it:
# the app was up, the pane was mounted, and between the socket closing and the
# bridge's re-dial landing the model was told "No interface is attached".
# Live evidence (2026-09-25): "dropped attach client (... surface=desktop)" at
# 09:14:46 and storms at 09:17:10-22 / 09:19:02-39, each coinciding with an
# injected detached block. The fix under test: the last desktop heartbeat is
# remembered per SESSION, for the same 45 s lease window, not cleared by a
# drop, not read by the residency or attention tiers, cleared ONLY by an
# explicit withdrawal, and backed up by the app's own "showing this session"
# record for a successor runtime booted before the re-dial lands.
# ---------------------------------------------------------------------------


async def _wait_for_drop(runtime: RuntimeServer) -> None:
    """Poll until the runtime has actually observed the closed socket.

    The drop is noticed on the runtime's OWN loop (its reader sees EOF), and
    an assertion about the memory surviving the drop means nothing if the
    assertion races the removal: the old per-connection clause would still be
    live and the test would pass for the wrong reason.
    """
    deadline = asyncio.get_running_loop().time() + 5
    while runtime._clients and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert not runtime._clients, "the runtime never observed the closed socket"


@pytest.mark.asyncio
async def test_socket_lost_while_the_pane_is_mounted_does_not_move_the_attachment_answer() -> None:
    """THE INCIDENT'S FIRST SHAPE: a drop with no withdrawal (round 3).

    The bridge re-dials ~500 ms after a drop and the pane never leaves; that
    gap is what the model read. Fails on the pre-fix tree (``frozenset()``
    once the connection is reaped), passes once the heartbeat is remembered
    per session — while the ATTENTION tier stays empty, because a dropped
    socket is not a person looking.  See the round-3 probe table in
    ``docs/design/attached-interface-signal.md``.
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)
        assert runtime.attached_surfaces() == frozenset({"desktop"})

        # The socket dies with NO withdrawal — every drop in the fleet log,
        # where the bridge is re-dialing the same pane a moment later.
        desktop.close()
        await _wait_for_drop(runtime)

        assert runtime.attached_surfaces() == frozenset(
            {"desktop"}
        ), "the attachment died with the socket that carried it"
        # The two tiers stay separate: nothing is being LOOKED AT.
        assert runtime.watching_surfaces() == frozenset()
    finally:
        desktop.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_redial_before_its_re_assert_has_landed_does_not_move_the_answer() -> None:
    """THE INCIDENT'S SECOND SHAPE: the re-dial's re-assert is best-effort.

    ``AttachedSession._dial`` re-asserts the lease only after the welcome and
    under a 5 s bound, so there is a real window where the successor
    connection exists but has asserted nothing. The connection is not the
    fact; the remembered heartbeat is. Fails on the pre-fix tree (a fresh
    connection with no ``desktop_watch`` yet counts for nothing).
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    redialed = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)
        desktop.close()
        await _wait_for_drop(runtime)

        # The re-dial arrives; the re-assert has not landed yet.
        await redialed.connect(record, "s1")
        assert runtime.attached_surfaces() == frozenset({"desktop"})

        # And the re-assert landing later changes nothing — same answer.
        await redialed.desktop_watch(visible=True, can_notify=True)
        assert runtime.attached_surfaces() == frozenset({"desktop"})
    finally:
        desktop.close()
        redialed.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_lapsed_lease_past_the_ttl_reads_detached() -> None:
    """HONESTY AFTER THE WINDOW (round 3): no heartbeat, no attachment.

    The memory is the SAME 45 s window as the per-connection lease (not a
    second constant and not a wider one), so driving both clocks past
    ``DESKTOP_WATCH_LEASE_S`` with no beat behind them must read detached — a
    closed app cannot keep a runtime claiming an interface forever.
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.types import DESKTOP_WATCH_LEASE_S

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)
        conn = _desktop_connection(runtime)
        stale = time.monotonic() - (DESKTOP_WATCH_LEASE_S + 1.0)

        # Both clocks, with the connection still alive: honest again.
        conn.desktop_seen = stale
        runtime._desktop_attach_seen = stale
        assert runtime.attached_surfaces() == frozenset()

        # AND THE MEMORY ALONE MUST LAPSE TOO: with the socket gone, the
        # remembered heartbeat is the only grant, and after its own TTL with
        # no beat behind it the answer is honest again — a closed app cannot
        # keep a runtime claiming an interface forever. (This half fails on
        # the pre-fix tree one line earlier: the dropped pane already reads
        # detached there.)
        await desktop.desktop_watch(visible=True, can_notify=True)
        desktop.close()
        await _wait_for_drop(runtime)
        assert runtime.attached_surfaces() == frozenset({"desktop"})
        runtime._desktop_attach_seen = stale
        assert runtime.attached_surfaces() == frozenset()
    finally:
        await desktop.detach()
        runtime.close()


@pytest.mark.asyncio
async def test_a_heartbeat_re_arms_the_attachment_answer() -> None:
    """THE MIRROR: one accepted beat re-arms both clocks (round 3).

    A renderer that was silent past the TTL and then beats again is a pane
    that came back; the answer must follow it back up without a re-dial.
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.types import DESKTOP_WATCH_LEASE_S

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)
        conn = _desktop_connection(runtime)
        stale = time.monotonic() - (DESKTOP_WATCH_LEASE_S + 1.0)
        conn.desktop_seen = stale
        runtime._desktop_attach_seen = stale
        assert runtime.attached_surfaces() == frozenset()

        await desktop.desktop_watch(visible=False, can_notify=True)
        assert runtime.attached_surfaces() == frozenset({"desktop"})
    finally:
        await desktop.detach()
        runtime.close()


@pytest.mark.asyncio
async def test_a_successor_runtime_reads_the_record_that_names_this_session(
    monkeypatch, tmp_path
) -> None:
    """A SWAP LEAVES NO CONNECTION AT ALL (round 3, C3).

    The bridge re-engages a successor within boot+dial, but until that dial
    lands there is no conn to count — and a turn starting in the window used
    to be told the app was gone. The app's OWN record names this conversation
    (``present ∧ has_window ∧ session_id == mine``), so the successor can
    answer honestly before the bridge arrives. Bounded by the record's own
    TTL, which its reader already reaps.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    with _desktop_presence_claim(tmp_path, session_id="s1"):
        runtime.start()
        try:
            await _wait_record()
            # NO connection at all: this is the boot window, not a dial.
            assert runtime.attached_surfaces() == frozenset({"desktop"})
            assert runtime.watching_surfaces() == frozenset()
        finally:
            runtime.close()


@pytest.mark.asyncio
async def test_an_unnamed_record_does_not_grant(monkeypatch, tmp_path) -> None:
    """THE C3 OVERRULE, PINNED: ``session_id == ""`` is NOT evidence.

    On this machine the record reads ``session_id: ""`` WHILE the operator is
    watching — the UI withdraws the name on every transient stream end — so a
    grant on the empty field would tell every session on the machine "an
    interface is attached". A name for SOMEONE ELSE is not evidence for this
    session either; both stay detached until the app can name a conversation.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        await _wait_record()
        with _desktop_presence_claim(tmp_path, session_id=""):
            assert runtime.attached_surfaces() == frozenset()
        with _desktop_presence_claim(tmp_path, session_id="someone-else"):
            assert runtime.attached_surfaces() == frozenset()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_dropped_then_recent_heartbeat_moves_none_of_the_other_tiers() -> None:
    """THE NO-MOVE PINS: the memory grants the MODEL answer and nothing else.

    A stale memory must never keep a runtime resident and must never be read
    as attention: after a drop-then-recent heartbeat, ``attach_clients()``
    (the reaper's count), ``watching_surfaces()`` and
    ``_visible_attach_surfaces()`` must read exactly what a heartbeat of the
    same shape reads on a live connection — which, for
    ``visible=False, can_notify=False``, is NOTHING on all three, while
    ``attached_surfaces()`` still answers for the mounted pane.
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=False, can_notify=False)
        desktop.close()
        await _wait_for_drop(runtime)

        assert runtime.attached_surfaces() == frozenset({"desktop"})
        # The residency and attention tiers do not read the memory:
        assert runtime.attach_clients() == 0
        assert runtime.watching_surfaces() == frozenset()
        assert runtime._visible_attach_surfaces() == set()

        # A re-dial plus the same heartbeat: still nothing for them.
        redialed = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
        try:
            await redialed.connect(record, "s1")
            await redialed.desktop_watch(visible=False, can_notify=False)
            assert runtime.attach_clients() == 0
            assert runtime.watching_surfaces() == frozenset()
            assert runtime._visible_attach_surfaces() == set()
            assert runtime.attached_surfaces() == frozenset({"desktop"})
        finally:
            redialed.close()
    finally:
        desktop.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_withdrawal_clears_the_memory_and_a_drop_does_not() -> None:
    """THE C2 PIN: withdrawal clears; drop does not.

    An explicit withdrawal (the bridge's "the pane left for real" signal) is
    the ONE thing that clears the session-scoped memory early: it drops the
    model-facing answer immediately, on the connection that is still open.
    A dropped socket, by contrast, leaves the answer standing for the rest of
    the lease window — that asymmetry IS the fix.

    Fails on the pre-fix tree at the first assertion (``desktop_withdraw``
    does not exist to send).
    """
    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)
        assert runtime.attached_surfaces() == frozenset({"desktop"})

        await desktop.desktop_withdraw()
        assert runtime.attached_surfaces() == frozenset()
        assert runtime.attach_clients() == 0

        # The mirror: the SAME state, dropped instead of withdrawn, survives.
        await desktop.desktop_watch(visible=True, can_notify=True)
        assert runtime.attached_surfaces() == frozenset({"desktop"})
        desktop.close()
        await _wait_for_drop(runtime)
        assert runtime.attached_surfaces() == frozenset({"desktop"})
    finally:
        desktop.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_real_drop_does_not_flip_the_interactivity_block() -> None:
    """PROMPT-LEVEL: the churn never reaches the rendered bytes (round 3).

    The block is recomputed per turn from the probe, so "the attachment died
    with the socket" surfaced to the model as a rewritten system prompt — the
    operator's actual complaint. Mirroring
    ``test_prompts_api.py::test_interactivity_costs_the_same_whatever_the_attach_churn``
    at the WIRE level instead of the renderer level: drive the REAL probe
    across a drop, a re-dial and a re-assert, and assert the rendered bytes
    are identical through all of it (the cost property is that same byte
    equality — nothing is journalled because nothing changes).
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.prompts_api import CHANNEL_ASK, build_system_blocks

    def block_for(interactive: bool) -> str:
        return build_system_blocks(
            [], "", "env", "2026-01-01", interactive=interactive, channel=CHANNEL_ASK
        )[-1]

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    redialed = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        await desktop.connect(record, "s1")
        await desktop.desktop_watch(visible=True, can_notify=True)

        rendered = [block_for(bool(runtime.attached_surfaces()))]

        desktop.close()
        await _wait_for_drop(runtime)
        rendered.append(block_for(bool(runtime.attached_surfaces())))

        await redialed.connect(record, "s1")
        rendered.append(block_for(bool(runtime.attached_surfaces())))
        await redialed.desktop_watch(visible=True, can_notify=True)
        rendered.append(block_for(bool(runtime.attached_surfaces())))

        assert len(set(rendered)) == 1, [
            "the block moved across the churn" if len(set(rendered)) > 1 else ""
        ]
        assert "<interactivity>" in rendered[0]
    finally:
        desktop.close()
        redialed.close()
        runtime.close()


@pytest.mark.asyncio
async def test_desktop_attach_refuses_old_runtime_before_becoming_a_false_terminal() -> None:
    from dataclasses import replace

    from local_operator.mobile.attach_client import AttachClient

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    desktop = AttachClient(lambda _projection: None, lambda _reason: None, surface="desktop")
    try:
        record = await _wait_record()
        with pytest.raises(ConnectionError, match="runtime update"):
            await desktop.connect(replace(record, capabilities=[]), "s1")
        assert runtime.attach_clients() == 0
    finally:
        await desktop.detach()
        runtime.close()


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
        # reach the live count. (On the owner's loop, as the reader loop's own call
        # is — the guard in ``_send_to`` refuses it anywhere else.)
        await _on_runtime_loop(runtime, runtime._on_request({"op": "unwatch", "req": "late"}, conn))

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


class _ConsumersHandle(FakeHandle):
    """Records the ``consumers`` set each routed slash was dispatched with."""

    def __init__(self) -> None:
        super().__init__()
        self.consumers: list[Any] = []

    async def run_slash_authoritative(
        self, command, args, images, *, locality="local", consumers=None
    ):  # noqa: ANN001, ANN202
        self.consumers.append(consumers)
        return {"kind": "notice", "text": f"owner ran /{command}", "style": "info"}


@pytest.mark.asyncio
async def test_a_client_that_declares_slash_consumers_reaches_the_handle() -> None:
    """The auth frame's declaration must arrive where the decision is made.

    The runtime completes an action-carrying receipt's request only for a
    client that did NOT declare that type. That decision lives on the handle,
    so the connection's declaration has to be threaded to it — the same way
    ``locality`` is, and for the same reason: it is a property of the
    CONNECTION, not of the frame.
    """
    handle = _ConsumersHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(
            record, client="attach", slash_consumers=["team_attached", "agent_attached"]
        )
        writer.write(
            json.dumps({"op": "slash_result", "req": 7, "command": "team", "args": "x go"}).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 7)
        assert handle.consumers == [frozenset({"team_attached", "agent_attached"})]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_client_that_omits_slash_consumers_reads_as_none() -> None:
    """Absent-means-old, and ``None`` is how the handle recognises it.

    This is the incident client on the wire: a viewer built before the field
    existed sends no declaration at all, and the runtime must read that as
    "will not submit the request itself" rather than as an empty promise it
    could mistake for a declaration.
    """
    handle = _ConsumersHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(
            json.dumps({"op": "slash_result", "req": 8, "command": "team", "args": "x go"}).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 8)
        assert handle.consumers == [None]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_handle_that_does_not_take_consumers_still_works() -> None:
    """Back-compat for the second optional keyword, proven on a real double.

    ``FakeHandle.run_slash_authoritative`` takes three positionals and neither
    keyword. Passing ``consumers`` unconditionally would raise ``TypeError``
    inside the dispatch and turn every routed slash into an error frame for
    any handle not updated in lockstep — including the reduced ones this suite
    is built on, which is precisely the population the probe protects.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach", slash_consumers=["team_attached"])
        writer.write(
            json.dumps({"op": "slash_result", "req": 9, "command": "goal", "args": ""}).encode()
            + b"\n"
        )
        await writer.drain()
        frame = await _until(reader, "result", 9)
        assert frame["data"]["text"] == "owner ran /goal"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_malformed_slash_consumers_value_does_not_refuse_the_client() -> None:
    """Advisory, never a gate: a bad value degrades to ``None``.

    A client that cannot attach is strictly worse than one whose request the
    runtime completes on its behalf, so a malformed declaration must not close
    the connection — it must read as "declared nothing".
    """
    handle = _ConsumersHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        writer.write(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "slash_consumers": "team_attached",  # a string, not a list
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        welcome = await asyncio.wait_for(reader.readline(), timeout=5)
        assert json.loads(welcome)["op"] == "projection", "a bad field must not refuse the attach"
        writer.write(
            json.dumps(
                {"op": "slash_result", "req": 10, "command": "team", "args": "x go"}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 10)
        assert handle.consumers == [None]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_the_phone_daemon_dials_as_remote() -> None:
    """The relay must declare itself, or loopback silently reads as "the user".

    The daemon connects over loopback like everything else, but it is a RELAY:
    the person is holding a phone on the other side of the mobile portal's
    tunnel. Before this it sent `{"key": ...}` alone and was classified local,
    so a phone's `/mcp reauth` would have opened a browser on the desktop and
    rewritten a credential the phone's owner cannot see (review F1).

    Asserted on the FRAME the daemon actually writes rather than on a constant,
    so deleting the field from the dial fails this test.
    """
    import inspect as _inspect

    from local_operator.mobile import daemon as daemon_module

    source = _inspect.getsource(daemon_module._dial)
    assert (
        '"locality": "remote"' in source
    ), "the daemon must declare locality=remote in its auth frame"


@pytest.mark.asyncio
async def test_a_daemon_class_dial_is_refused_a_grant() -> None:
    """End-to-end cover for F1: dial exactly as the daemon does, and be remote."""
    handle = _LocalityHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        # Byte-identical to mobile/daemon.py::_dial — no `client` field, which
        # means daemon, plus the locality it now declares.
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        writer.write(json.dumps({"key": record.control_key, "locality": "remote"}).encode() + b"\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=5)  # welcome
        writer.write(
            json.dumps(
                {"op": "slash_result", "req": 9, "command": "mcp", "args": "reauth n"}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        await _until(reader, "result", 9)
        assert handle.localities == ["remote"]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


# --- the credential op --------------------------------------------------------
#
# The one op whose frame carries a secret. It has a dedicated name so the
# value never rides a general-purpose field; the server gates the STORE verb
# on the client's declared locality the way the `/mcp` grant verbs are gated,
# and validates the frame so a non-string value is refused rather than
# coerced. Every assertion below is on lengths and outcomes — the placeholder
# is never printed.


class _CredentialHandle(FakeHandle):
    """Records each credential verb with the LENGTH of its value, never the value."""

    def __init__(self) -> None:
        super().__init__()
        self.verbs: list[tuple[str, str, int]] = []

    def credential_op(self, action: str, key: str, value: str) -> dict[str, object]:
        self.verbs.append((action, key, len(value)))
        if action == "store":
            return {"ok": True, "key": key, "replaced": False}
        if action == "names":
            return {"ok": True, "names": [key] if key else []}
        return {"ok": True}


async def _reply_to(reader: asyncio.StreamReader, req: int, n: int = 30) -> dict[str, Any]:
    """The first ``result`` or ``error`` frame carrying ``req``, skipping broadcasts."""
    for _ in range(n):
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            continue
        frame = json.loads(text)
        if frame.get("req") == req and frame.get("op") in ("result", "error"):
            return frame
    raise AssertionError(f"no reply to req {req} arrived")


async def _credential(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, req: int, **fields: object
) -> dict[str, Any]:
    writer.write(json.dumps({"op": "credential", "req": req, **fields}).encode() + b"\n")
    await writer.drain()
    return await _until(reader, "result", req)


@pytest.mark.asyncio
async def test_a_local_client_may_store_a_credential() -> None:
    """Loopback attach clients are local by construction; the store lands."""
    handle = _CredentialHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        placeholder = "x" * 24
        frame = await _credential(
            reader, writer, 11, action="store", key="DEMO_TOKEN", value=placeholder
        )
        assert frame["data"] == {"ok": True, "key": "DEMO_TOKEN", "replaced": False}
        assert handle.verbs == [("store", "DEMO_TOKEN", len(placeholder))]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_remote_client_is_refused_the_store_but_may_list() -> None:
    """Review round 1, R4: the phone relay must not push a secret into the
    desktop's environment. The read verbs return key NAMES only and stay open."""
    handle = _CredentialHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach", locality="remote")
        placeholder = "x" * 24
        frame = await _credential(
            reader, writer, 12, action="store", key="DEMO_TOKEN", value=placeholder
        )
        assert frame["data"] == {"ok": False, "reason": "remote-client"}
        assert handle.verbs == [], "the handle must never see a remote store"
        frame = await _credential(reader, writer, 13, action="names")
        assert frame["data"]["ok"] is True
        assert handle.verbs == [("names", "", 0)]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_credential_frame_with_a_non_string_value_is_refused() -> None:
    """Review round 1, N2: validated rather than ``str()``-coerced into a repr."""
    handle = _CredentialHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(
            json.dumps(
                {"op": "credential", "req": 14, "action": "store", "key": "K", "value": 12345}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        # Whichever frame answers req 14: the mutant (validator removed)
        # STORES the coerced repr and answers ``result``, so waiting on
        # ``error`` alone would fail as a 5 s timeout with no message (round
        # 2, N4). Read the reply and assert on its shape.
        frame = await _reply_to(reader, 14)
        assert frame.get("op") == "error", f"a non-string value was accepted: {frame}"
        assert "value must be a string" in frame.get("message", ""), frame
        assert handle.verbs == []
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_push_builds_no_payload_without_projection_recipients() -> None:
    from unittest.mock import patch

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    await runtime.start_in_process()
    writer = daemon_writer = None
    try:
        with patch.object(
            runtime, "_projection_payload", wraps=runtime._projection_payload
        ) as payload:
            for _ in range(20):
                await runtime._push()
            payload.assert_not_called()
            record = runtime.record
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", record.control_port, limit=1 << 20
            )
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
            assert json.loads(await reader.readline())["op"] in _WELCOME_OPS
            assert json.loads(await reader.readline())["op"] == "frontend_sync"
            # A full-TUI attach RENDERS nothing from the projection, so its welcome
            # no longer builds one at all (``_slim_welcome_frame``). The next two
            # assertions are the half that must not regress: the daemon, which
            # does render it, still gets a built payload — and ``_push`` still
            # skips the attach without building anything.
            payload.assert_not_called()
            payload.reset_mock()
            for _ in range(20):
                await runtime._push()
            payload.assert_not_called()
            daemon_reader, daemon_writer = await _dial(record)
            payload.assert_called_once()
            payload.reset_mock()
            await runtime._push()
            assert json.loads(await daemon_reader.readline())["op"] == "projection"
            payload.assert_called_once()
    finally:
        for connection in (writer, daemon_writer):
            if connection is not None:
                connection.close()
                await connection.wait_closed()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_push_skips_full_tui_clients_but_keeps_welcome_and_daemon() -> None:
    """Test 20: ``_push`` skips events+frontend clients; welcome still delivered.

    Daemon and events-only attach clients keep receiving projection repaints
    byte-for-byte. A full-TUI viewer (events AND frontend_state) got the
    welcome as its identity check and then must not be flooded.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    daemon_writer = events_writer = tui_writer = None
    try:
        record = await _wait_record()
        daemon_reader, daemon_writer = await _dial(record)

        events_reader, events_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        events_writer.write(
            json.dumps({"key": record.control_key, "client": "attach", "events": True}).encode()
            + b"\n"
        )
        await events_writer.drain()
        assert json.loads(await events_reader.readline())["op"] == "projection"

        tui_reader, tui_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        tui_writer.write(
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
        await tui_writer.drain()
        assert json.loads(await tui_reader.readline())["op"] in _WELCOME_OPS
        seed = json.loads(await tui_reader.readline())
        assert seed["op"] == "frontend_sync"

        await _on_runtime_loop(runtime, runtime._push())
        daemon_repaint = json.loads(await asyncio.wait_for(daemon_reader.readline(), timeout=2))
        assert daemon_repaint["op"] == "projection"
        events_repaint = json.loads(await asyncio.wait_for(events_reader.readline(), timeout=2))
        assert events_repaint["op"] == "projection"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(tui_reader.readline(), timeout=0.15)
    finally:
        for writer in (daemon_writer, events_writer, tui_writer):
            if writer is not None:
                writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_every_drop_logs_its_reason_once_at_one_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 21: one line per actual removal, naming the reason.

    At the level the reason EARNS: a close the runtime asked for, or one a peer
    asked for by closing first, is part of a client's ordinary life and stays
    INFO. Every other reason means a client was removed without asking to
    leave, and those are WARNING — see `_GRACEFUL_DROP_REASONS` and
    `test_an_unrequested_drop_is_logged_at_warning`. Both used to be INFO
    beside every routine close, which is how "the sidebar went cold" ended up
    with no findable cause in the runtime log.
    """
    import logging

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        _reader, writer = await _dial(record, client="attach")
        conns = list(runtime._clients.values())
        assert len(conns) == 1
        caplog.set_level(logging.INFO, logger="local_operator.session.runtime.server")
        runtime._drop_client(conns[0], reason="reader eof")
        infos = [
            rec
            for rec in caplog.records
            if rec.levelno == logging.INFO and "dropped" in rec.getMessage()
        ]
        assert len(infos) == 1, [rec.getMessage() for rec in infos]
        assert "dropped attach client" in infos[0].getMessage()
        assert "events=" in infos[0].getMessage()
        assert infos[0].getMessage().endswith(": reader eof")
        assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        # Second call (reader-loop finally) must not log again.
        runtime._drop_client(conns[0], reason="reader eof")
        infos_after = [
            rec
            for rec in caplog.records
            if rec.levelno == logging.INFO and "dropped" in rec.getMessage()
        ]
        assert len(infos_after) == 1
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_an_unrequested_drop_is_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Why the client went is the one thing that must not be buried.

    A viewer dropped for a reason IT did not choose — the attach cap evicting
    it, an event queue overflowing, a send timing out — learns what happened
    only as a cold facade. The runtime's own record of the reason is therefore
    the only place the cause exists at all, and at INFO beside every routine
    close it was invisible in practice: the operator's report of "Reconnect
    failed — select again to retry" had nothing to read.
    """
    import logging

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        _reader, writer = await _dial(record, client="attach")
        conns = list(runtime._clients.values())
        assert len(conns) == 1
        caplog.set_level(logging.INFO, logger="local_operator.session.runtime.server")
        runtime._drop_client(conns[0], reason="attach cap")
        warnings = [
            rec
            for rec in caplog.records
            if rec.levelno == logging.WARNING and "dropped" in rec.getMessage()
        ]
        assert len(warnings) == 1, [rec.getMessage() for rec in warnings]
        assert "dropped attach client" in warnings[0].getMessage()
        assert warnings[0].getMessage().endswith(": attach cap")
        # The reason is still ONE line, not a warning plus an info.
        assert not [
            rec
            for rec in caplog.records
            if rec.levelno == logging.INFO and "dropped" in rec.getMessage()
        ]
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_attach_cap_drop_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writers: list[Any] = []
    try:
        record = await _wait_record()
        caplog.set_level(logging.INFO, logger="local_operator.session.runtime.server")
        for _ in range(ATTACH_MAX_CLIENTS + 1):
            _reader, writer = await _dial(record, client="attach")
            writers.append(writer)
        await asyncio.sleep(0.1)
        messages = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno == logging.WARNING and "dropped" in rec.getMessage()
        ]
        assert any(msg.endswith(": attach cap") for msg in messages), messages
    finally:
        for writer in writers:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_tui_send_timeout_is_five_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 22: full-TUI clients use ``_TUI_SEND_TIMEOUT_S``, not the 1 s bound."""
    from local_operator.session.runtime import server as server_mod

    assert server_mod._SEND_TIMEOUT_S == 1.0
    assert server_mod._TUI_SEND_TIMEOUT_S == 5.0

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    seen: list[float] = []
    original = asyncio.wait_for

    async def spy_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        seen.append(float(timeout) if timeout is not None else -1.0)
        return await original(awaitable, timeout=timeout)

    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        _reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
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
        # Welcome + frontend_sync drain through _send_to.
        await asyncio.sleep(0.2)
        conns = [c for c in runtime._clients.values() if c.wants_events and c.wants_frontend]
        assert conns, "full-TUI client never registered"
        seen.clear()
        monkeypatch.setattr(asyncio, "wait_for", spy_wait_for)
        # ON THE RUNTIME'S OWN LOOP. ``_send_to`` owns ``conn.send_lock`` and the
        # connection's writer, and both belong to the loop ``start()`` hosts — so it
        # is driven from there, the way its only production callers are. Awaiting it
        # from this loop is the cross-loop misuse ``_send_to`` now refuses (see
        # ``test_send_to_refuses_a_foreign_loop_instead_of_parking``); it looked like
        # it worked here only because an UNCONTESTED lock takes the fast path.
        runtime_loop = runtime._loop
        assert runtime_loop is not None, "start() publishes the runtime's loop"
        await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(
                runtime._send_to(conns[0], {"op": "ping"}), runtime_loop
            )
        )
        assert 5.0 in seen, f"TUI send bound was not 5.0 s; saw {seen}"
        assert 1.0 not in seen, f"full-TUI client still used the 1 s bound; saw {seen}"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_send_to_refuses_a_foreign_loop_instead_of_parking() -> None:
    """A loop-owned send is REFUSED off its loop, not waited on until forever.

    ``_send_to`` is the single write path for every frame the runtime emits, and
    both things it touches are owned by the runtime's loop: ``conn.writer`` (a
    StreamWriter whose transport and drain waiter were built on that loop) and
    ``conn.send_lock`` (an ``asyncio.Lock``, which binds itself to the first loop
    that CONTENDS it and is then unusable from any other).

    WHY THE GUARD, AND WHY THIS IS NOT A TIMING ASSERTION: the pre-fix failure mode
    was not an exception at all. Awaiting that path from a foreign loop leaves a
    waiter future on the FOREIGN loop, and the owner's ``Lock.release()`` completes
    it from the wrong thread with ``set_result`` — whose callback is scheduled with
    plain ``call_soon``, an append to the other loop's ready deque with no self-pipe
    write. A loop parked in ``select()`` is never woken, so the await never returns
    and the xdist worker running it is wedged until CI's cap cancels the shard job:
    ten such cancels, several on ``main``, are what this guard exists to make
    impossible to reintroduce silently. An uncontended lock is enough to show it —
    the fast path makes the cross-loop call LOOK fine while binding the lock to the
    wrong loop for every later contention — so this needs no timing, no contention
    and no sleeping: it fails on the pre-fix tree as "DID NOT RAISE".
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        _reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        writer.write(json.dumps({"key": record.control_key, "client": "attach"}).encode() + b"\n")
        await writer.drain()
        deadline = asyncio.get_running_loop().time() + 5
        conns: list[Any] = []
        while asyncio.get_running_loop().time() < deadline and not conns:
            conns = list(runtime._clients.values())
            if not conns:
                await asyncio.sleep(0.02)
        assert conns, "the client never registered"
        assert (
            runtime._loop is not asyncio.get_running_loop()
        ), "this test only means anything while the runtime is thread-hosted"

        with pytest.raises(RuntimeError, match="owning event loop"):
            await runtime._send_to(conns[0], {"op": "ping"})
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


# -- compose-frame folding ----------------------------------------------------
#
# ``_compact_event_queue`` is exercised here against a bare ``_ClientConn`` and
# a bare ``RuntimeServer`` rather than a live socket: the property under test is
# a pure function of the FIFO's contents.
#
# What the neighbouring coverage does and does NOT give. The GENERIC overflow
# drop is driven over a real socket by
# ``test_high_volume_event_relay_bounds_nonreader_and_preserves_healthy_order``,
# but that test emits only ``NoticeEvent`` — it never puts a compose frame on
# the wire, so it is not evidence for this path (review R2). The compose path
# itself was driven end to end against a production ``RuntimeServer`` and a real
# ``AttachedSession`` viewer in QA's round-1 cell, which reproduced the reported
# frame on base (viewer dropped by overflow; three composing rows never adopted
# or retired; the turn never completed) and showed this branch adopting and
# retiring all three. That cell needs two processes and a real socket, so it
# lives with the PR evidence; these tests are the fast structural guard for the
# same property.


def _compose_frame(call_id: str, argument_bytes: int, intent: str | None = None) -> dict[str, Any]:
    return {
        "op": "event",
        "data": {
            "type": "tool_call_compose",
            "tool_call_id": call_id,
            "tool_name": "bash",
            "argument_bytes": argument_bytes,
            "intent": intent,
        },
    }


def _reasoning_frame(message_id: str, delta: str) -> dict[str, Any]:
    return {
        "op": "event",
        "data": {"type": "reasoning_delta", "message_id": message_id, "delta": delta},
    }


def _text_frame(message_id: str, delta: str) -> dict[str, Any]:
    return {
        "op": "event",
        "data": {"type": "message_update", "message": {"id": message_id}, "delta": delta},
    }


def _aside_frame(req: int, delta: str) -> dict[str, Any]:
    """One streamed chunk of an aside, exactly as ``_aside_delta_sink`` shapes it.

    ``req`` is the request that asked, and it is what makes two concurrent
    asides distinguishable here: the panel id the desktop renderer knows as
    ``aside_id`` never crosses this wire.
    """
    return {"op": "aside_delta", "req": req, "data": {"delta": delta}}


def _stalled_conn() -> Any:
    """A connection whose reader never drains, i.e. the case the bound is for."""
    from local_operator.session.runtime.server import _EVENT_QUEUE_MAX, _ClientConn

    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    conn.event_queue = asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
    conn.wants_events = True
    conn.wants_frontend = True
    conn.events_ready = True
    return conn


class _NeverDrains(RuntimeServer):
    """Enqueue-path harness: records drops, never consumes the FIFO."""

    def __init__(self) -> None:  # noqa: D107 — deliberately skips RuntimeServer.__init__
        self._clients: dict[int, Any] = {}
        self._event_sends: set[Any] = set()
        self.dropped: list[str] = []

    def _drop_client(  # type: ignore[override]
        self, conn: Any, *, reason: str = "unspecified"
    ) -> None:
        self.dropped.append(reason)

    async def _drain_event_queue(self, conn: Any) -> None:  # type: ignore[override]
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_a_stalled_viewer_survives_a_long_multi_call_dictation() -> None:
    """A dictation must not overflow the FIFO and drop the viewer.

    Compose frames carry a distinct ``tool_call_id`` and a growing
    ``argument_bytes``, so before the fold NOTHING in the queue was compactible
    during a tool-argument dictation: 300 frames across 3 calls dropped the
    client after 65 with ``event queue overflow (64 frames)``. That matters
    because the measured compose rate at three concurrent calls is 16.0
    frames/s — the 64-frame bound is reached in ~4.0 s, BEFORE the 5.0 s
    ``_TUI_SEND_TIMEOUT_S`` raised specifically to protect stalled TUI viewers.
    The dropped viewer re-attaches and its seed re-mounts composing rows that
    nothing adopts, which is the operator-reported "tool cards frozen at an
    identical elapsed time" on calls that in fact ran.

    Frame counts, not seconds: this asserts a structural property of the
    queue, so it cannot flake under host contention.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for index in range(300):
        server._enqueue_client_frame(conn, _compose_frame(f"call_{index % 3}", index * 10))

    assert server.dropped == [], f"stalled viewer was dropped: {server.dropped}"

    # Compaction runs only ON OVERFLOW, so between passes the queue legitimately
    # refills with not-yet-folded frames. The invariant is therefore asserted
    # where it is claimed — after a pass — rather than at an arbitrary depth.
    server._compact_event_queue(cast(Any, conn))

    # The fold's actual guarantee: at most one entry per in-flight call, each
    # carrying that call's NEWEST byte count. Bounded by the number of calls in
    # flight, never by the dictation's length.
    queued = list(conn.event_queue._queue)
    composes = [f for f in queued if (f.get("data") or {}).get("type") == "tool_call_compose"]
    per_call: dict[str, list[int]] = {}
    for frame in composes:
        data = frame["data"]
        per_call.setdefault(str(data["tool_call_id"]), []).append(int(data["argument_bytes"]))
    for call_id, sizes in per_call.items():
        assert len(sizes) == 1, f"{call_id} kept {len(sizes)} compose frames, expected 1"

    # LOSSLESS in the only sense a snapshot can be: the retained frame is the
    # newest the queue ever saw for that call, so the viewer's byte counter
    # jumps forward rather than backward. Without this the test would pass
    # against a fold that kept the OLDEST frame and discarded every update.
    newest = {f"call_{call}": max(i * 10 for i in range(300) if i % 3 == call) for call in range(3)}
    assert {k: v[0] for k, v in per_call.items()} == newest


@pytest.mark.asyncio
async def test_a_stalled_viewer_survives_a_long_reasoning_stream() -> None:
    """Reasoning fragments must fold, and fold LOSSLESSLY, or the viewer drops.

    Reasoning arrives once per token, so it is the largest single frame family a
    long-thinking turn produces: without the fold, a viewer that stalls for the
    ~4 s the 64-frame bound allows is dropped mid-think and its re-attach shows a
    thinking block that starts in the middle of a sentence. The property asserted
    is the one a fold must not trade away: the retained frame carries EVERY
    fragment, in arrival order, so the viewer paints the same text a live one
    would have painted.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    fragments = [f"thought-{index} " for index in range(300)]
    for fragment in fragments:
        server._enqueue_client_frame(conn, _reasoning_frame("m1", fragment))

    assert server.dropped == [], f"stalled viewer was dropped: {server.dropped}"
    server._compact_event_queue(cast(Any, conn))

    queued = [
        frame
        for frame in conn.event_queue._queue
        if (frame.get("data") or {}).get("type") == "reasoning_delta"
    ]
    assert len(queued) == 1, f"kept {len(queued)} reasoning frames, expected 1"
    assert queued[0]["data"]["delta"] == "".join(fragments)
    assert queued[0]["data"]["message_id"] == "m1"


@pytest.mark.asyncio
async def test_reasoning_never_folds_into_the_answer_text() -> None:
    """The two delta families are adjacent and must still not merge.

    A reasoning fragment and an answer fragment for the SAME message arrive
    interleaved, so a key that named only the stream would fold them together and
    the viewer would paint the model's private thinking inside its answer -- the
    transcript corruption ``ReasoningDeltaEvent`` exists to prevent, and
    invisible to every other test because both frames pass the size guard.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    server._enqueue_client_frame(conn, _text_frame("m1", "the answer"))
    server._enqueue_client_frame(conn, _reasoning_frame("m1", "the reasoning"))
    server._enqueue_client_frame(conn, _reasoning_frame("m1", " continues"))
    server._enqueue_client_frame(conn, _text_frame("m1", " in full"))

    assert server.dropped == []
    server._compact_event_queue(cast(Any, conn))

    kinds = [(f["data"]["type"], f["data"]["delta"]) for f in conn.event_queue._queue]
    assert kinds == [
        ("message_update", "the answer"),
        ("reasoning_delta", "the reasoning continues"),
        ("message_update", " in full"),
    ]


@pytest.mark.asyncio
async def test_a_long_reasoning_stream_still_emits_a_readable_frame() -> None:
    """Delta-sized byte accounting has to hold for the reasoning family too.

    The merge measures the DELTA and adds it to the prior frame's own size rather
    than re-serializing the merged frame. A reasoning frame has no accumulated
    ``message`` to re-dump, so the arithmetic is the same and the output must
    still be a line the reader can accept -- asserted against a stream that would
    exceed the limit if the accounting were per-frame rather than per-delta.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for _ in range(400):
        server._enqueue_client_frame(conn, _reasoning_frame("m1", "x" * 4096))

    assert server.dropped == []
    for frame in conn.event_queue._queue:
        size = len(json.dumps(frame).encode()) + 1
        assert size <= _MAX_LINE_BYTES, f"fold emitted an unreadable {size}-byte frame"


@pytest.mark.asyncio
async def test_a_long_aside_stream_still_emits_a_readable_frame() -> None:
    """Delta-sized byte accounting has to hold for the aside family too."""
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for _ in range(400):
        server._enqueue_client_frame(conn, _aside_frame(7, "x" * 4096))

    assert server.dropped == []
    for frame in conn.event_queue._queue:
        size = len(json.dumps(frame).encode()) + 1
        assert size <= _MAX_LINE_BYTES, f"fold emitted an unreadable {size}-byte frame"


@pytest.mark.asyncio
async def test_an_aside_merge_that_would_not_fit_is_refused_not_truncated() -> None:
    """Two legal aside frames can merge into an ILLEGAL one — refuse the merge.

    Each chunk passed the relay guard individually at enqueue; the merge is the
    one operation that makes a frame bigger than anything the guard was shown,
    which is why the size check exists at all. Truncating instead would deliver a
    HALF answer that looks whole: the aside's text is whatever the viewer's last
    frame said, so a cut in the middle of a sentence is not cosmetic, and the
    POST's authoritative ``text`` would disagree with the card it painted.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    # Individually legal (each fits one line), jointly illegal.
    chunk = "x" * (_MAX_LINE_BYTES // 2 + 1024)
    server._enqueue_client_frame(conn, _aside_frame(7, chunk))
    server._enqueue_client_frame(conn, _aside_frame(7, chunk))

    assert server.dropped == []
    server._compact_event_queue(cast(Any, conn))

    queued = list(conn.event_queue._queue)
    assert len(queued) == 2, "the merge must be refused, not applied"
    assert [f["data"]["delta"] for f in queued] == [chunk, chunk]
    assert "".join(f["data"]["delta"] for f in queued) == chunk + chunk
    for frame in queued:
        size = len(json.dumps(frame).encode()) + 1
        assert size <= _MAX_LINE_BYTES, f"emitted an unreadable {size}-byte frame"


@pytest.mark.asyncio
async def test_two_asides_on_one_connection_never_fold_together() -> None:
    """The aside's stream identity is the REQUEST, and the frame must keep it.

    Two asides can be in flight over one connection (two desktop windows, or a
    retry while the first is still streaming). Folding them would splice one
    viewer's answer into another's — and a merge that dropped the ``req`` would
    deliver the text to a pump with no stream to route it to, so the chunk would
    be discarded on arrival while the frame counted as delivered.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    server._enqueue_client_frame(conn, _aside_frame(1, "one "))
    server._enqueue_client_frame(conn, _aside_frame(1, "still one"))
    server._enqueue_client_frame(conn, _aside_frame(2, "two "))
    server._enqueue_client_frame(conn, _aside_frame(2, "still two"))

    assert server.dropped == []
    server._compact_event_queue(cast(Any, conn))

    assert [(f["req"], f["data"]["delta"]) for f in conn.event_queue._queue] == [
        (1, "one still one"),
        (2, "two still two"),
    ]


@pytest.mark.asyncio
async def test_interleaved_asides_keep_their_own_chunks_in_arrival_order() -> None:
    """Two asides streaming at once must not be spliced across each other.

    Compaction folds ADJACENT frames of one stream, so two interleaved answers
    stay interleaved: the second aside's fragment is between the first's, which
    is exactly why the fold cannot be keyed on the family alone. A viewer paints
    its own answer from its own frames, and a fork of one into the other would
    show another viewer's text inside a private panel.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    server._enqueue_client_frame(conn, _aside_frame(1, "one "))
    server._enqueue_client_frame(conn, _aside_frame(2, "two "))
    server._enqueue_client_frame(conn, _aside_frame(1, "still one"))
    server._enqueue_client_frame(conn, _aside_frame(2, "still two"))

    assert server.dropped == []
    server._compact_event_queue(cast(Any, conn))

    assert [(f["req"], f["data"]["delta"]) for f in conn.event_queue._queue] == [
        (1, "one "),
        (2, "two "),
        (1, "still one"),
        (2, "still two"),
    ]


@pytest.mark.asyncio
async def test_a_stalled_viewer_survives_a_long_aside_stream() -> None:
    """An aside streams one fragment per token, so it needs the fold like text.

    Before this family was merged there was nothing in the FIFO a stall could
    compact: 300 fragments of one answer filled the 64-frame bound and the viewer
    was dropped with ``event queue overflow`` — the same failure the reasoning
    fold closed, on the surface whose slowest reader is a phone. The property
    that must survive the fold is losslessness: the retained frame carries EVERY
    fragment, in arrival order, so the viewer paints the answer the model wrote.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    fragments = [f"part-{index} " for index in range(300)]
    for fragment in fragments:
        server._enqueue_client_frame(conn, _aside_frame(7, fragment))

    assert server.dropped == [], f"stalled viewer was dropped: {server.dropped}"
    server._compact_event_queue(cast(Any, conn))

    queued = list(conn.event_queue._queue)
    assert len(queued) == 1, f"kept {len(queued)} aside frames, expected 1"
    assert queued[0]["op"] == "aside_delta"
    assert queued[0]["req"] == 7
    assert queued[0]["data"]["delta"] == "".join(fragments)


@pytest.mark.asyncio
async def test_an_aside_fragment_never_folds_into_a_neighbouring_conversation_event() -> None:
    """An aside is a private question; the conversation is not it.

    The two families are adjacent in the same FIFO and both carry a ``delta``, so
    a key that named only "a delta" would splice an off-record answer into the
    message the user is reading — and the transcript would then differ from the
    provider's own record of the turn. The FAMILY is part of the key, so they are
    never compared.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    server._enqueue_client_frame(conn, _text_frame("m1", "the answer"))
    server._enqueue_client_frame(conn, _aside_frame(7, "the aside"))
    server._enqueue_client_frame(conn, _aside_frame(8, " another aside"))
    server._enqueue_client_frame(conn, _text_frame("m1", " in full"))

    assert server.dropped == []
    server._compact_event_queue(cast(Any, conn))

    kinds = [
        (f["op"], (f.get("data") or {}).get("type"), (f.get("data") or {})["delta"])
        for f in conn.event_queue._queue
    ]
    assert kinds == [
        ("event", "message_update", "the answer"),
        ("aside_delta", None, "the aside"),
        ("aside_delta", None, " another aside"),
        ("event", "message_update", " in full"),
    ]


@pytest.mark.asyncio
async def test_a_folded_compose_frame_never_overtakes_the_start_that_follows_it() -> None:
    """Folding must not move a compose frame past its call's execution start.

    The UI keys rows by ``tool_call_id`` and adopts a composing row when the
    ``tool_execution_start`` arrives. If a later compose frame were folded onto
    a slot AHEAD of that start, the viewer would apply "still composing" after
    "now executing" and re-mount a composing row for a call already running —
    reintroducing the stranded row this fold exists to remove. The fold
    therefore stops for a call the moment any other frame for it goes by.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    start = {
        "op": "event",
        "data": {"type": "tool_execution_start", "tool_call_id": "c1", "tool_name": "bash"},
    }
    end = {
        "op": "event",
        "data": {"type": "tool_execution_end", "tool_call_id": "c1", "tool_name": "bash"},
    }
    # A frame for ANOTHER call sits between the two folded snapshots. That
    # separation is what makes this test able to fail: with the two composes
    # adjacent, replacing in place and deleting-then-appending produce the same
    # order, so an append-based fold would pass unnoticed. Here it moves c1's
    # row behind c2's and the assertion catches it.
    other = _compose_frame("c2", 99)
    for frame in (
        _compose_frame("c1", 10),
        other,
        _compose_frame("c1", 20),
        start,
        _compose_frame("c1", 30),
        end,
    ):
        server._enqueue_client_frame(conn, frame)

    server._compact_event_queue(cast(Any, conn))

    order = [
        (f["data"]["type"], f["data"]["tool_call_id"], f["data"].get("argument_bytes"))
        for f in conn.event_queue._queue
    ]
    # c1's two pre-start snapshots fold to the newest AND KEEP c1'S ORIGINAL
    # SLOT, still ahead of c2. The trailing snapshot is kept separately behind
    # the start rather than folded back in front of it.
    assert order == [
        ("tool_call_compose", "c1", 20),
        ("tool_call_compose", "c2", 99),
        ("tool_execution_start", "c1", None),
        ("tool_call_compose", "c1", 30),
        ("tool_execution_end", "c1", None),
    ], order


@pytest.mark.asyncio
async def test_folding_does_not_carry_a_compose_slot_across_a_turn_boundary() -> None:
    """Placeholder compose keys repeat per turn, so slots must not outlive one.

    ``harness/loop.py`` latches an index-derived ``compose:{index}`` key when a
    provider announces a call's name before its id, so the NEXT turn's first
    call is ``compose:0`` again. Folding across the boundary would move that
    frame back in front of the ``agent_end``/``agent_start`` between them.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for frame in (
        _compose_frame("compose:0", 10),
        {"op": "event", "data": {"type": "agent_end"}},
        {"op": "event", "data": {"type": "agent_start"}},
        _compose_frame("compose:0", 5),
    ):
        server._enqueue_client_frame(conn, frame)

    server._compact_event_queue(cast(Any, conn))

    assert [f["data"]["type"] for f in conn.event_queue._queue] == [
        "tool_call_compose",
        "agent_end",
        "agent_start",
        "tool_call_compose",
    ]


@pytest.mark.asyncio
async def test_a_compose_slot_does_not_survive_a_model_STEP_boundary() -> None:
    """A recycled placeholder key must not fold backwards past a step's calls.

    THE BOUNDARY THAT MATTERS IS THE STEP, NOT THE RUN. ``compose:{index}`` is
    derived from ``tool_states``, which ``harness/loop.py`` scopes to a single
    ``_model_turn`` — and ``_model_turn`` is called once per STEP from
    ``run()``'s ``while has_more_tool_calls or pending:`` loop, while
    ``agent_start``/``agent_end`` bracket the whole run. So ``compose:0``
    recurs on every tool-calling step, and a reset keyed only on the run
    boundary never fires between them (review R1).

    Nor does the ``elif call_id`` abandon-branch save it: under the placeholder
    regime the compose frame is keyed ``compose:0`` while the start and end
    carry the provider's real id, so the pop misses and the slot stays live
    straight through the execution.

    This is the reviewer's reproduction: without the fix, step 2's snapshot
    folds backwards past TWO execution starts and TWO ends, landing in front of
    work that has already finished.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    def _exec(kind: str, call_id: str) -> dict[str, Any]:
        return {"op": "event", "data": {"type": kind, "tool_call_id": call_id}}

    for frame in (
        {"op": "event", "data": {"type": "agent_start"}},
        # --- step 1: dictate compose:0, then run it under its REAL id --------
        {"op": "event", "data": {"type": "turn_start"}},
        _compose_frame("compose:0", 10),
        _compose_frame("compose:0", 20),
        _compose_frame("compose:0", 30),
        _exec("tool_execution_start", "real_0"),
        _exec("tool_execution_end", "real_0"),
        # --- step 2: the SAME placeholder key, a different call --------------
        {"op": "event", "data": {"type": "turn_start"}},
        _compose_frame("compose:0", 110),
        _compose_frame("compose:0", 120),
        _compose_frame("compose:0", 130),
        _exec("tool_execution_start", "real_1"),
        _exec("tool_execution_end", "real_1"),
        {"op": "event", "data": {"type": "agent_end"}},
    ):
        server._enqueue_client_frame(conn, frame)

    server._compact_event_queue(cast(Any, conn))

    order = [(f["data"]["type"], f["data"].get("argument_bytes")) for f in conn.event_queue._queue]
    # Each step keeps its OWN folded snapshot, behind its own boundary and
    # ahead of its own execution. Step 1's snapshot is not overwritten by step
    # 2's, and step 2's has not moved in front of step 1's start/end.
    assert order == [
        ("agent_start", None),
        ("turn_start", None),
        ("tool_call_compose", 30),
        ("tool_execution_start", None),
        ("tool_execution_end", None),
        ("turn_start", None),
        ("tool_call_compose", 130),
        ("tool_execution_start", None),
        ("tool_execution_end", None),
        ("agent_end", None),
    ], order


@pytest.mark.asyncio
async def test_a_compose_count_that_goes_backwards_starts_a_new_slot() -> None:
    """A shrinking ``argument_bytes`` proves the key was reused, so do not fold.

    ``argument_bytes`` is cumulative within one call, so it can only grow. A
    DECREASE therefore cannot come from the call already holding the slot — it
    is a different call that recycled the key, and folding the two together
    would splice one call's dictation progress onto another's row.

    This is the second, independent half of the step-boundary defence. The
    boundary reset needs the boundary frame to be present in the SAME
    compaction batch; this one needs no boundary frame at all, which covers a
    queue holding only compose frames from either side of a step.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    # No boundary frame anywhere: only the byte count reveals the reuse.
    for frame in (
        _compose_frame("compose:0", 100),
        _compose_frame("compose:0", 200),
        _compose_frame("compose:0", 5),
        _compose_frame("compose:0", 15),
    ):
        server._enqueue_client_frame(conn, frame)

    server._compact_event_queue(cast(Any, conn))

    # Two slots: the first call folded to its newest (200), then the reuse
    # opened a fresh slot BEHIND it which folded to its own newest (15).
    assert [f["data"]["argument_bytes"] for f in conn.event_queue._queue] == [200, 15]


@pytest.mark.asyncio
async def test_an_absent_or_malformed_byte_count_still_folds() -> None:
    """The reuse check must not become a back door to the overflow drop.

    Refusing to fold on every frame whose ``argument_bytes`` is missing or
    non-integer would let a malformed producer reopen the very overflow this
    method exists to close. An equal or absent count is what an un-throttled
    repeat of the SAME call looks like, so folding is the safe default.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for _ in range(200):
        server._enqueue_client_frame(
            conn,
            {
                "op": "event",
                "data": {
                    "type": "tool_call_compose",
                    "tool_call_id": "c1",
                    "tool_name": "bash",
                    # No ``argument_bytes`` at all.
                },
            },
        )

    assert server.dropped == [], f"malformed frames reopened the drop: {server.dropped}"
    server._compact_event_queue(cast(Any, conn))
    assert len(conn.event_queue._queue) == 1


@pytest.mark.asyncio
async def test_an_unchanged_byte_count_still_folds() -> None:
    """Only a DECREASE proves reuse; an equal count is the same call repeating.

    The reuse check must be strictly ``<``. At ``<=`` every repeat of an
    unchanged count would open a fresh slot, so a producer emitting equal
    counts fills the FIFO one frame at a time and the overflow drop this method
    exists to close is reopened — the guard turning into the bug. Caught by
    mutation: ``<`` → ``<=`` passed every other test in this block.
    """
    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    for _ in range(200):
        server._enqueue_client_frame(conn, _compose_frame("c1", 500))

    assert server.dropped == [], f"equal counts reopened the drop: {server.dropped}"
    server._compact_event_queue(cast(Any, conn))
    assert [f["data"]["argument_bytes"] for f in conn.event_queue._queue] == [500]


@pytest.mark.asyncio
async def test_the_compose_fold_never_emits_an_unreadable_frame() -> None:
    """The fold REPLACES, so it cannot assemble an oversized frame.

    ``_compact_event_queue``'s size refusal exists because merging
    ``message_update`` deltas is the one operation that makes a frame bigger
    than anything the relay guard was shown. A compose fold discards the older
    snapshot instead of concatenating, so the output is always a frame that
    already passed the guard individually — asserted here rather than assumed,
    because a future fold that started merging would silently defeat that
    refusal.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn

    # Near-limit but individually legal, the shape that killed a real pump.
    for index in range(200):
        server._enqueue_client_frame(
            conn, _compose_frame(f"call_{index % 3}", index, intent="i" * 300_000)
        )

    assert server.dropped == []
    for frame in conn.event_queue._queue:
        size = len(json.dumps(frame).encode()) + 1
        assert size <= _MAX_LINE_BYTES, f"fold emitted an unreadable {size}-byte frame"


# ---------------------------------------------------------------------------
# The wire fit pass: a live frame carrying images is REFERENCED, not degraded.
#
# Measured on the operator's own session bytes (a 317,726-byte page render,
# base64 through the real attachment store): one ``tool_execution_end`` with two
# of them is an 847,600-byte frame, and two such rows in one frame is 1,695,113
# bytes — 1.62x the 1 MiB line the attach reader enforces. The guard answered
# that with a ``notice``: the event was dropped, the live tool card could never
# settle, and the viewer fell back to a full re-sync.
# ---------------------------------------------------------------------------


def _image_b64(size: int = 400_000) -> str:
    """A deterministic inline payload shaped like a page render's base64."""
    import base64

    return base64.b64encode(bytes(range(256)) * (size // 256)).decode("ascii")


def _wire_bytes(frame: dict[str, Any]) -> int:
    """The size the socket will write, re-derived here on purpose.

    The production rule is ONE function (``server._frame_line_bytes``); this is a
    test asserting against it, so it computes the number itself rather than
    calling the code under test — a measurement taken through the same helper
    cannot catch that helper being wrong.
    """
    return len(json.dumps(frame).encode()) + 1


def _image_tool_end(*, images: int, call_id: str = "call_image") -> dict[str, Any]:
    """One ``tool_execution_end`` whose result carries ``images`` page renders."""
    data = _image_b64()
    content: list[dict[str, Any]] = [{"type": "text", "text": "PAGE"}]
    content += [{"type": "image", "data": data, "mime_type": "image/png"} for _ in range(images)]
    return {
        "type": "tool_execution_end",
        "tool_call_id": call_id,
        "tool_name": "screenshot",
        "is_error": False,
        "result": {
            "tool_call_id": call_id,
            "tool_name": "screenshot",
            "content": content,
            "is_error": False,
        },
    }


def _text_tool_end(*, size: int, call_id: str = "call_text") -> dict[str, Any]:
    """One ``tool_execution_end`` whose result is a single oversized TEXT block.

    The shape no reference can help: the payload IS the text, so the only way to
    keep the event (and with it the card that settles) is to bound the text in
    place. That makes it the frame the SHEDDING stage exists for.
    """
    long_text = "x" * size
    return {
        "type": "tool_execution_end",
        "tool_call_id": call_id,
        "tool_name": "read",
        "is_error": False,
        "result": {
            "tool_call_id": call_id,
            "tool_name": "read",
            "content": [{"type": "text", "text": long_text}],
            "is_error": False,
        },
    }


@pytest.mark.asyncio
async def test_a_three_image_tool_end_frame_is_referenced_not_degraded() -> None:
    """The chokepoint must fit an image-bearing event, not shed it.

    Degrading this frame loses the ``tool_execution_end``, and the end is the
    only thing that settles the live tool card — the ``⊘ interrupted`` fallback
    exists for exactly the stranded card that follows. A reference resolves to
    the same bytes on every in-repo frontend, and the metrics are the frame
    size, the surviving identity fields and the round-trip of each digest.
    """
    import re

    from local_operator.session.attachments import AttachmentStore
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    conn = _stalled_conn()
    server._clients[id(conn.writer)] = conn
    frame = {"op": "event", "data": _image_tool_end(images=3)}
    original = _image_b64()
    assert _wire_bytes(frame) > _MAX_LINE_BYTES

    server._enqueue_client_frame(conn, frame)

    assert server.dropped == []
    enqueued = conn.event_queue.get_nowait()
    assert _wire_bytes(enqueued) <= _MAX_LINE_BYTES
    assert enqueued["data"]["type"] == "tool_execution_end"
    assert enqueued["data"]["tool_call_id"] == "call_image"
    assert enqueued["data"]["tool_name"] == "screenshot"
    images = [block for block in enqueued["data"]["result"]["content"] if block["type"] == "image"]
    assert len(images) == 3
    for block in images:
        assert "data" not in block
        assert re.fullmatch(r"[0-9a-f]{32}", block["attachment"]), block
        assert block["mime_type"] == "image/png"
        resolved = AttachmentStore().get(block["attachment"])
        assert resolved is not None, "the reference does not resolve to stored bytes"
        assert resolved[0] == original


@pytest.mark.asyncio
async def test_the_reference_pass_does_not_mutate_the_shared_frame() -> None:
    """One producer frame reaches every recipient; none of them owns it.

    ``_relay_on_loop`` hands the SAME dict to each connection's enqueue, so a
    pass that rewrote in place would let the first connection decide what the
    second sends — and would leave the producer holding a frame it never built.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    server._closed = threading.Event()
    first, second = _stalled_conn(), _stalled_conn()
    server._clients[id(first.writer)] = first
    server._clients[id(second.writer)] = second
    data = _image_tool_end(images=3)
    before = json.dumps(data, sort_keys=True)

    server._relay_on_loop(data)

    assert json.dumps(data, sort_keys=True) == before, "the producer frame was mutated"
    frames = [conn.event_queue.get_nowait() for conn in (first, second)]
    for frame in frames:
        assert _wire_bytes(frame) <= _MAX_LINE_BYTES
        assert "data" not in frame["data"]["result"]["content"][1]
    assert json.dumps(frames[0], sort_keys=True) == json.dumps(frames[1], sort_keys=True)


@pytest.mark.asyncio
async def test_the_shedding_pass_does_not_mutate_the_shared_frame() -> None:
    """The shed stage edits its argument, and its argument is the producer's.

    ``_bound_live_result_in_place`` clips the result dict and its blocks IN
    PLACE, and the result it is handed comes straight out of the frame the relay
    fans out — so without the copy the first recipient's clip would rewrite the
    payload every later recipient measures, and the producer would be left
    holding a frame it never built. This test is what pins the copy: with it
    removed, all 205 tests in the three touched files still passed while the
    producer's own result was already clipped.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    server = _NeverDrains()
    server._closed = threading.Event()
    first, second = _stalled_conn(), _stalled_conn()
    server._clients[id(first.writer)] = first
    server._clients[id(second.writer)] = second
    data = _text_tool_end(size=2 * _MAX_LINE_BYTES)
    before = json.dumps(data, sort_keys=True)

    server._relay_on_loop(data)

    assert json.dumps(data, sort_keys=True) == before, "the producer frame was mutated"
    frames = [conn.event_queue.get_nowait() for conn in (first, second)]
    for frame in frames:
        assert _wire_bytes(frame) <= _MAX_LINE_BYTES
        # The EVENT survived, which is the whole point of shedding instead of
        # degrading: the card it settles.
        assert frame["data"]["type"] == "tool_execution_end"
        assert frame["data"]["tool_call_id"] == "call_text"
    clipped = frames[0]["data"]["result"]["content"][0]["text"]
    assert clipped != "x" * (2 * _MAX_LINE_BYTES), "the shed stage did not run"
    assert clipped.endswith("…")
    assert json.dumps(frames[0], sort_keys=True) == json.dumps(frames[1], sort_keys=True)


@pytest.mark.asyncio
async def test_a_terminal_that_switched_away_stops_counting_as_a_watcher() -> None:
    """A retained attach is a connection, not a person reading this session.

    THE BUG THIS PINS. A multiplexing TUI keeps the outgoing session's
    connection open when the user switches away, so `_visible_attach_surfaces`
    counted a viewer that was showing something else. `_announce_pending`
    suppresses its out-of-band toast whenever a surface is watching, so a gate
    parked behind such a connection waited in silence for the whole unattended
    timeout with its card painted into a viewer nobody was looking at.

    Asserted on `watching_surfaces()` -- the predicate the notification
    routing actually reads -- rather than on the flag, so the test fails if the
    field stops reaching the decision.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")

        # Displaying by default: an older viewer that never sends the op is
        # counted exactly as it was before this field existed.
        assert runtime.watching_surfaces() == frozenset({"attach"})

        writer.write(
            json.dumps({"op": "viewer_watch", "req": 1, "displaying": False}).encode() + b"\n"
        )
        await writer.drain()
        await _until(reader, "ack", 1)

        # The connection is still open -- only the claim to be showing it went.
        assert runtime.attach_clients() == 1
        assert runtime.watching_surfaces() == frozenset()

        writer.write(
            json.dumps({"op": "viewer_watch", "req": 2, "displaying": True}).encode() + b"\n"
        )
        await writer.drain()
        await _until(reader, "ack", 2)
        assert runtime.watching_surfaces() == frozenset({"attach"})
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_switching_away_re_announces_a_parked_gate() -> None:
    """The 1->0 transition must reach `reannounce_pending`.

    The suppression is only half the defect: a gate that opened while somebody
    WAS watching sends no toast by design, and the re-announce on the detached
    edge is what rescues it. A viewer that switches away without closing its
    socket has to produce that edge, or the rescue never runs.
    """
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    announced: list[str] = []
    setattr(handle, "reannounce_pending", lambda: announced.append("reannounced"))
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        runtime.set_record_pending("approval")
        announced.clear()

        writer.write(
            json.dumps({"op": "viewer_watch", "req": 1, "displaying": False}).encode() + b"\n"
        )
        await writer.drain()
        await _until(reader, "ack", 1)

        assert announced == ["reannounced"], "a parked gate was not re-announced on switch-away"
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_viewer_watch_rejects_a_non_boolean_and_leaves_state_intact() -> None:
    """A malformed frame must not silently blank the display claim."""
    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        writer.write(
            json.dumps({"op": "viewer_watch", "req": 1, "displaying": "no"}).encode() + b"\n"
        )
        await writer.drain()
        reply = await _until(reader, "error", 1)
        assert "boolean" in str(reply.get("message", ""))
        assert runtime.watching_surfaces() == frozenset({"attach"})
        writer.close()
    finally:
        runtime.close()


class _HeldBindHandle(FakeHandle):
    """A handle whose canonical frontend bind waits until the test releases it.

    ``FakeHandle.subscribe_frontend`` returns immediately, so the window in
    which a real bind is genuinely IN FLIGHT — the window ``_serve_frontend_sync``
    and the ``_SYNC_PRIORITY_OPS`` admission gate exist for — is zero, and a
    test could say nothing about it. The gate is a ``threading.Event`` rather
    than an ``asyncio.Event`` on purpose: the bind runs on the RUNTIME's own
    loop thread, so releasing it from the test thread through a loop-bound
    future would be exactly the cross-thread wakeup ``_send_to``'s guard exists
    to refuse.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bind_gate = threading.Event()
        self.bind_entered = threading.Event()

    async def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001
        self.bind_entered.set()
        # Bounded, so a test that fails before releasing the gate reports an
        # assertion rather than hanging the shard.
        await asyncio.to_thread(self.bind_gate.wait, 10.0)
        return self._frontend.subscribe(on_update)


class _BoundButUnsentHandle(FakeHandle):
    """Subscribes at once, then holds the sync frame — the delta window.

    The frame-ordering invariant lives in the gap between "the subscription
    exists" and "the frame that seeds it is on the wire". Holding the bind
    itself cannot reach that gap — with no subscription yet, an update is simply
    part of the capture — so this double holds AFTER subscribing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hold_gate = threading.Event()
        self.bound = threading.Event()

    async def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001
        subscription = self._frontend.subscribe(on_update)
        self.bound.set()
        await asyncio.to_thread(self.hold_gate.wait, 10.0)
        return subscription


async def _dial_frontend(
    record: registry.SessionRecord,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open + auth a full-TUI viewer; consume the welcome projection.

    Deliberately does NOT read the ``frontend_sync``: the point of every test
    below is what happens while that frame is still missing.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
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
    assert json.loads(await asyncio.wait_for(reader.readline(), timeout=5))["op"] in _WELCOME_OPS
    return reader, writer


@pytest.mark.asyncio
async def test_a_viewer_can_be_health_checked_and_controlled_while_its_bind_is_in_flight() -> None:
    """The socket is usable BEFORE the sync lands — and for exactly five ops.

    This is the operator's "prioritize health check and interface connection
    requests", as a socket-level assertion. Before it, ``_on_connection`` ran
    the frontend bind inline, so a viewer whose bind was in flight could not be
    spoken to at all: no ``ping``, no ``steer``, no ``stop`` — and because that
    bind takes a cross-thread hop into the app loop, a busy app loop left a live
    idle session unreachable and its record reading ``wedged`` (measured
    2026-09-18: no welcome within 20 s, heartbeat to 45.1 s).

    The OTHER half matters as much as this one: a connection mid-bind is not yet
    authoritative, so the heavy ops stay refused — through the ordinary error
    frame, never by running them — and the refusal lifts the moment the bind
    lands.
    """
    handle = _HeldBindHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bind_entered.wait, 5), "the bind never started"

        # HEALTH: the liveness probe every surface already speaks.
        writer.write(json.dumps({"op": "ping", "req": 1}).encode() + b"\n")
        await writer.drain()
        assert (await _until(reader, "ack", 1))["detail"] == "pong"

        # CONTROL: a steer is how a user corrects the turn they are watching, so
        # it must not queue behind a snapshot.
        writer.write(json.dumps({"op": "steer", "req": 2, "text": "correction"}).encode() + b"\n")
        await writer.drain()
        await _until(reader, "ack", 2)
        assert ("steer", ("correction",), {}) in handle.calls

        # A HEAVY OP IS REFUSED, not run: admitting it here would let a client
        # act on a connection that has not been told what it is acting on.
        writer.write(json.dumps({"op": "prompt", "req": 3, "text": "hello"}).encode() + b"\n")
        await writer.drain()
        refusal = await _until(reader, "error", 3)
        assert "still connecting" in refusal["message"]
        assert [
            call for call in handle.calls if call[0] == "prompt"
        ] == [], "a pre-sync prompt reached the session instead of being refused"

        # AND THE REFUSAL IS NOT PERMANENT: the same op runs once the bind lands.
        handle.bind_gate.set()
        assert (await _until(reader, "frontend_sync"))["op"] == "frontend_sync"
        writer.write(json.dumps({"op": "prompt", "req": 4, "text": "hello"}).encode() + b"\n")
        await writer.drain()
        await _until(reader, "ack", 4)
        assert ("prompt", ("hello",), {}) in handle.calls
    finally:
        handle.bind_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


class _CancelableHeldBindHandle(_HeldBindHandle):
    """A held-bind handle that can also answer the graceful ``cancel`` rung."""

    async def cancel_gracefully(self):  # noqa: ANN202
        return await self._record("cancel_gracefully")


@pytest.mark.asyncio
async def test_a_cancel_is_admitted_pre_sync() -> None:
    """``cancel`` is one of the ops a still-binding connection may run.

    Review round 2, NIT: ``cancel`` joined :data:`_SYNC_PRIORITY_OPS` in round 1
    (review F2) and nothing tested it, so the next person to prune that set would
    have had no signal. It is admitted for the same reason ``abort`` is — it is
    how a supervisor stops a runaway turn, and refusing it pre-sync would deny
    that exactly when the connection is least able to do anything else. Asserted
    over a real socket with the bind held, like its neighbour above.
    """
    handle = _CancelableHeldBindHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bind_entered.wait, 5), "the bind never started"

        writer.write(json.dumps({"op": "cancel", "req": 1}).encode() + b"\n")
        await writer.drain()
        assert (await _until(reader, "ack", 1))["detail"] == "cancel_gracefully ok"
        assert ("cancel_gracefully", (), {}) in handle.calls
    finally:
        handle.bind_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_repaint_cannot_overtake_the_frontend_sync_that_precedes_it() -> None:
    """A delta must never arrive before the snapshot it is a delta against.

    Registration and snapshot capture were atomic on the authoritative loop
    before this change and still are — but moving the sync into a task creates a
    real window between "the subscription exists" and "the seeding frame is on
    the wire", and a canonical update emitted in that window is exactly the
    frame a follower's client refuses as a state gap (``attach_client`` raises
    "frontend state gap" and goes cold). So the update waits in
    ``frontend_pending`` and is flushed BEHIND the sync.
    """
    handle = _BoundButUnsentHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bound.wait, 5), "the bind never subscribed"

        # Subscribed, snapshot captured, sync frame NOT yet on the wire: this is
        # the window under test.
        handle._frontend.mutate(
            pending_gate=PendingRequest(
                request_id="approval-1", kind="approval", title="bash", detail="echo hi"
            ).to_json()
        )
        handle.hold_gate.set()

        first = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
        assert first["op"] == "frontend_sync", "a repaint overtook the sync that seeds it"
        # The waiting delta is not lost, and it arrives after the seed.
        update = await _until(reader, "frontend_update")
        assert update["data"]["changes"]["pending_gate"]["request_id"] == "approval-1"
    finally:
        handle.hold_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


class _TeardownOnStopHandle(FakeHandle):
    """A host whose ``stop`` hook closes its own runtime, like the TUI's does.

    The TUI shape, in three lines: ``TuiSessionHandle.request_stop`` schedules
    the app's teardown, the teardown closes the registrant, and the ack for the
    ``stop`` op is still in flight on that same runtime. The ``sleep(0)`` is the
    hop's own yield point (the TUI's is a worker thread, measured at ~1.0 s of
    app-loop work); it is what gives the shutdown a chance to run BEFORE the ack
    is written, which is the ordering the test below is about. Without it the
    ack would be written in the same synchronous stretch as ``close()`` and the
    test would pass on a tree that has no fence at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self.runtime: RuntimeServer | None = None

    async def request_stop(self) -> str:
        assert self.runtime is not None, "the test wires the runtime in"
        self.runtime.close()
        # The shape of the real hop: the dispatch PARKS on work owned by another
        # thread, so the loop gets turns in which the teardown can run to the
        # point of dropping this connection. A bare ``asyncio.sleep(0)`` is not
        # enough and would make this test pass on a tree with no fence at all —
        # measured: with ``sleep(0)`` the loop ran the dispatch's resumption
        # before the runner's teardown, so the ack went out either way.
        await asyncio.to_thread(time.sleep, 0.05)
        return "stopping"


@pytest.mark.asyncio
async def test_a_reply_the_runtime_admitted_reaches_its_client_across_a_shutdown() -> None:
    """A ``stop`` that tears its own runtime down must still get its ack out.

    The failure this pins, measured 2026-09-19 with a trace of
    ``_send_to``/``_shutdown_impl``/``_drop_client``: the shutdown ran at
    +1.035 s and dropped the connection with reason ``runtime shutdown`` BEFORE
    the ack was written, so the client read
    ``ConnectionError('runtime closed the connection')`` and the stop ladder
    escalated from a graceful stop that had in fact succeeded to the signal
    rung. It only surfaced once the TUI hop stopped blocking the serving loop,
    because that block had been (accidentally) serialising the dispatch ahead of
    the teardown.
    """
    handle = _TeardownOnStopHandle()
    runtime = RuntimeServer(handle, kind="tui")
    handle.runtime = runtime
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="daemon")
        writer.write(json.dumps({"op": "stop", "req": 7}).encode() + b"\n")
        await writer.drain()
        reply = await _until(reader, "ack", 7)
        assert reply["detail"] == "stopping"
        # The hook really did tear the runtime down — the ack above is not
        # evidence of a runtime that was never closed.
        assert runtime._closed.is_set(), "the hook's close() must have run"
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


class _ParkedSteerHandle(FakeHandle):
    """A handle whose ``steer`` parks until the test releases it.

    The shape of a mutation that is legitimately slow: on a TUI host the steer
    is a hop into the app's loop, so it waits exactly as long as the app is busy.
    While it waits, the connection must still answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.steer_entered = threading.Event()
        self.steer_gate = threading.Event()

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        self.calls.append(("steer", (text,), {}))
        self.steer_entered.set()
        # Bounded so a failing test reports an assertion rather than hanging.
        await asyncio.to_thread(self.steer_gate.wait, 10.0)
        return "steering queued"


@pytest.mark.asyncio
async def test_a_health_check_is_not_queued_behind_a_parked_op() -> None:
    """``ping`` answers while another op on the SAME connection is parked.

    Review round 1, UX U3. The reader loop used to ``await`` each request, so a
    connection was strictly serial: one parked mutation made the whole
    connection mute, and measured over a real socket a ``ping`` sent behind a
    parked ``steer`` went unanswered for 8-15 s. New connections were never
    affected (a fresh dial, its ping and its refusals all answered in 0.00 s) —
    which is why the headline held while the health check did not.

    The other half is asserted too: the chain still ORDERS what it orders. A
    second mutation admitted after the parked one must not overtake it, or two
    mutations could interleave on one session.
    """
    handle = _ParkedSteerHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="daemon")

        writer.write(json.dumps({"op": "steer", "req": 1, "text": "first"}).encode() + b"\n")
        await writer.drain()
        assert await asyncio.to_thread(handle.steer_entered.wait, 5), "the steer never parked"

        # HEALTH: answered while the steer is still parked.
        writer.write(json.dumps({"op": "ping", "req": 2}).encode() + b"\n")
        await writer.drain()
        assert (await _until(reader, "ack", 2))["detail"] == "pong"

        # ORDERING: the second steer is admitted but must WAIT for the first.
        writer.write(json.dumps({"op": "steer", "req": 3, "text": "second"}).encode() + b"\n")
        await writer.drain()
        await asyncio.sleep(0.1)
        assert [call[1][0] for call in handle.calls if call[0] == "steer"] == ["first"], (
            "a later mutation overtook one still in flight — two mutations can "
            "now interleave on one session"
        )

        handle.steer_gate.set()
        assert (await _until(reader, "ack", 1))["detail"] == "steering queued"
        assert (await _until(reader, "ack", 3))["detail"] == "steering queued"
        assert [call[1][0] for call in handle.calls if call[0] == "steer"] == ["first", "second"]
    finally:
        handle.steer_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_viewer_that_dies_mid_bind_leaves_no_subscription_behind() -> None:
    """A connection dropped mid-bind releases what the bind registered.

    Review round 1, F3, and it is a window the previous reasoning denied
    existed: ``_drop_client`` cancels the bind task, and its comment argued a
    cancelled bind could not leave a subscription because "a bind that is still
    parked in its handle call has not received a subscription yet, and
    everything between receiving one and recording it is synchronous". The
    premise is false — the subscription is registered INSIDE the awaited handle
    call — so a cancel landing after that registration and before
    ``conn.frontend_unsubscribe`` is written used to leave a subscriber for the
    life of the app (reproduced over a real socket: "store has subscribers AFTER
    the drop: True" with zero registered clients).

    The fix is structural: the bind is shielded so the cancel cannot abort it
    half-registered, and its cancellation path releases whatever did register.
    """
    handle = _BoundButUnsentHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bound.wait, 5), "the bind never subscribed"
        assert handle._frontend.has_subscribers, "the subscription never existed to leak"

        # The viewer dies mid-bind: its socket goes away while the bind is still
        # parked inside the handle call, so the drop lands between
        # "subscribed" and "recorded".
        writer.close()
        writer = None

        # THE ORDER IS THE TEST. The bind is allowed to land only AFTER the drop
        # has actually happened, because that is the case the old argument
        # missed: a bind that lands after its connection is gone still holds a
        # registration nobody recorded. Releasing first would race the drop and
        # let the bind complete normally — the test would then pass on a tree
        # with no fix at all (measured: it did exactly that).
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if not runtime._clients:
                break
            await asyncio.sleep(0.05)
        assert not runtime._clients, "the viewer was never dropped"
        handle.hold_gate.set()

        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            if not handle._frontend.has_subscribers and not runtime._clients:
                break
            await asyncio.sleep(0.05)
        assert not runtime._clients, "the dropped viewer is still registered"
        assert not handle._frontend.has_subscribers, (
            "the dropped viewer's frontend subscription is still registered — "
            "the session keeps pushing canonical state to a socket nobody owns"
        )
    finally:
        handle.hold_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


class _OffLoopCapableHeldBindHandle(_HeldBindHandle):
    """A PARKED on-loop bind plus a working off-loop one — the busy-owner shape.

    ``_HeldBindHandle`` covers the case where the grace expires and there is
    nothing to fall back to, so the wait simply continues (a single-plane handle,
    the TUI kind). This is the daemon/exec handle the grace exists for: the
    on-loop bind stays parked for the whole test, so the ONLY way this connection
    can be given a canonical state is the off-loop path, and the park is then
    released to model the shielded bind landing late.

    ``release_gate`` is a SECOND gate, and deliberately: it holds the handle call
    open AFTER the subscription has been registered, which is the only window in
    which a stale relay callback can be observed at all. Modelling "the bind
    lands late" without that window would test ``_release_when_landed`` and never
    the token.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release_gate = threading.Event()
        self.late_registered = threading.Event()
        self.off_loop_binds = 0

    async def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001
        subscription = await super().subscribe_frontend(on_update, display_window=display_window)
        # REGISTERED, and the call has not returned: the abandoned on-loop bind
        # now holds a live subscription whose relay callback is stamped stale.
        self.late_registered.set()
        await asyncio.to_thread(self.release_gate.wait, 10.0)
        return subscription

    def subscribe_frontend_nowait(self, on_update):  # noqa: ANN001
        self.off_loop_binds += 1
        # The REAL store call the production handle makes, so the ordering these
        # tests assert is the store's own rather than a double's idea of it.
        return self._frontend.subscribe_threadsafe(on_update)


class _DrainingOffLoopCapableHeldBindHandle(_OffLoopCapableHeldBindHandle):
    """The parked-owner double WITH the production drain latch available on it.

    Bound as a class attribute rather than called unbound, which is the pattern
    ``test_serving_drain``'s ``DrainHost`` uses: a double that re-implemented the latch
    would pin nothing about the state a viewer actually meets, and the attributes the
    real method writes are declared here so the cells can read them back.
    """

    begin_drain = ServingSessionHandle.begin_drain
    end_drain = ServingSessionHandle.end_drain

    def __init__(self) -> None:
        super().__init__()
        #: Written by ``begin_drain`` / cleared by ``end_drain``, read by the cells.
        self._draining = False
        self._retiring_cause = ""
        self._retiring_detail = ""
        self._disposing = False


class _CountingBindHandle(FakeHandle):
    """A HEALTHY handle: binds at once, and records if the fallback was used.

    ``FakeHandle.subscribe_frontend`` returns immediately, so this is the owner
    shape ``_ONLOOP_BIND_GRACE_S`` is sized for (a real one answers in 4.7-5.0 ms
    p50). The off-loop entry point exists and is countable, which is the point:
    "the grace never fires on a healthy owner" is otherwise asserted only by the
    absence of a symptom, and the symptom it would have (no display window on
    every viewer) is invisible while the attach still looks fast and green.
    """

    def __init__(self) -> None:
        super().__init__()
        self.off_loop_calls = 0

    def subscribe_frontend_nowait(self, on_update):  # noqa: ANN001
        self.off_loop_calls += 1
        return self._frontend.subscribe_threadsafe(on_update)


async def _frontend_delta_sequences(reader: asyncio.StreamReader, count: int) -> list[int]:
    """Read exactly ``count`` canonical deltas and return their sequences.

    Other frames are skipped rather than assumed absent (a repaint, an event
    frame): what is under test is the ORDER of the deltas, and a helper that
    tripped over an unrelated frame would fail for the wrong reason.
    """
    sequences: list[int] = []
    for _ in range(80):
        if len(sequences) == count:
            return sequences
        raw = await asyncio.wait_for(reader.readline(), timeout=5)
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            continue
        frame = json.loads(text)
        if frame.get("op") == "frontend_update":
            sequences.append(frame["data"]["sequence"])
    raise AssertionError(f"only {sequences} of {count} deltas arrived")


async def _subscribers_become(handle: FakeHandle, want: int) -> None:
    """Wait on an EVENT (the store's own roster), never on a clock."""
    for _ in range(200):
        if len(handle._frontend._subscribers) == want:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"the store holds {len(handle._frontend._subscribers)} subscribers, wanted {want}"
    )


@pytest.mark.asyncio
async def test_a_parked_owner_binds_off_loop_and_its_late_bind_never_double_relays(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The grace, the fallback and the bind token, on a real socket.

    WHAT THIS PINS. The on-loop bind is shielded and cannot be cancelled, so an
    attach that outlives the grace is served off-loop while the parked bind is
    STILL going to land — and when it lands it registers a SECOND subscriber on
    the same store. The assertion is on the sequences that reach the socket, not
    on the subscriber count alone: if the late callback relayed, this connection
    would receive every delta twice, and a client reading an exact-``+1`` stream
    reads a duplicate as a gap and redials.

    The whole test is event-driven (the handle's own gates, the store's own
    roster) because a schedule this subtle held together by sleeps would be a bet
    on the host.
    """
    handle = _OffLoopCapableHeldBindHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bind_entered.wait, 5), "the bind never started"

        with caplog.at_level(logging.INFO, logger="local_operator.session.runtime.server"):
            sync = await _until(reader, "frontend_sync")
        assert handle.off_loop_binds == 1, (
            "the sync arrived off-loop, but the recorded fallback count says the "
            "grace did not hand the bind over"
        )
        # AND THE HAND-OFF IS OBSERVABLE FROM OUTSIDE THE HANDLE. The runtime's
        # own counter and line are what answer the design's rollout question
        # ("is the fallback firing on healthy owners?") in a production log —
        # pinned HERE because this is the only attach in the file that takes the
        # fallback, and a counter nothing asserts is a number nobody can trust.
        assert runtime.frontend_off_loop_binds == 1
        assert "missed the 100 ms on-loop grace" in caplog.text
        assert sync["data"].get("display_history") is None, (
            "the off-loop path has no display window: it reads the loop-owned "
            "transcript, which is the loop this path exists to avoid"
        )
        base = sync["data"]["sequence"]

        # BEFORE the late bind lands, so these ride the off-loop registration.
        handle._frontend.mutate(goal="before the late bind")
        assert await _frontend_delta_sequences(reader, 1) == [base + 1]

        # RELEASE THE PARK: the abandoned on-loop bind registers its own
        # subscriber, whose callback was stamped before the fallback bumped the
        # token. Both subscribers exist NOW, which is the window this test is for.
        handle.bind_gate.set()
        assert await asyncio.to_thread(handle.late_registered.wait, 5), "the late bind never landed"
        await _subscribers_become(handle, 2)
        handle._frontend.mutate(goal="while both are registered")
        assert await _frontend_delta_sequences(reader, 1) == [base + 2]

        # AND THE STALE SUBSCRIBER IS RECLAIMED, not merely muted: the release
        # the cancellation path already uses runs when the bind lands.
        handle.release_gate.set()
        await _subscribers_become(handle, 1)
        handle._frontend.mutate(goal="after the release")
        assert await _frontend_delta_sequences(reader, 1) == [base + 3]

        # NOTHING ELSE IS ON THE WIRE. A duplicate would sit BEHIND the last
        # assertion above, so a stream that ended here is the claim.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.readline(), timeout=0.3)
    finally:
        handle.bind_gate.set()
        handle.release_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_draining_owner_still_lands_the_canonical_sync_inside_the_envelope() -> None:
    """THE OPERATOR'S SYMPTOM, with the drain latched: a viewer still gets its state.

    The incident's configuration, not an invented one. The runtime that could not be
    attached had latched a stale-build drain (``begin_drain``, taken here through the
    PRODUCTION latch rather than a fake that re-implements it) and was still stepping
    three subagent lanes, so its session loop was exactly as busy as the parked owner
    below models — the attach then missed the 15 s envelope and the operator got
    ``RuntimeUnresponsiveError`` while the session was, technically, alive and serving.

    So this is the parked-owner rig re-driven under a drain. Two things are asserted,
    and the second is the one a future change is most likely to break: the canonical
    ``frontend_sync`` LANDS, inside a bound well under the attach envelope, and it
    lands through the OFF-LOOP fallback — a draining runtime still serves a joining
    viewer. Gating the fallback on "not draining" (a plausible-looking reading of "a
    runtime that is leaving should not bind new viewers") is what this cell fails on,
    and the cost of that gate is another attach nobody can complete.

    The latch is asserted while the sync is in flight, because the property is about a
    DRAINING owner: a test that released the drain before dialling would prove nothing
    about the state the operator was in.
    """
    handle = _DrainingOffLoopCapableHeldBindHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        # The PRODUCTION latch over this reduced handle: ``begin_drain`` asks only for
        # ``_disposing`` and an optional ``session.retire_wakes_to_inbox``, so the same
        # method the real handle runs is the one under test here.
        assert handle.begin_drain("stale-build", "0.62.9 -> 0.62.12")
        assert (
            handle._draining is True
        ), "the drain has to be latched for this cell to mean anything"
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        assert await asyncio.to_thread(handle.bind_entered.wait, 5), "the bind never started"

        started = time.monotonic()
        sync = await _until(reader, "frontend_sync")
        landed_after = time.monotonic() - started

        assert sync["data"].get("sequence") is not None, "the sync carried no canonical state"
        assert landed_after < 10.0, (
            "a draining owner's viewer waited "
            f"{landed_after:.1f}s for its state, and the attach envelope is 15 s: this is "
            "the shape that made the runtime unattachable"
        )
        assert handle.off_loop_binds == 1, (
            "the parked on-loop bind carried the sync, so the off-loop fallback a "
            "draining runtime needs was not used"
        )
        assert runtime.frontend_off_loop_binds == 1
        assert handle._draining is True, (
            "the drain was released under the viewer, so this no longer says anything "
            "about a DRAINING owner"
        )
    finally:
        handle.bind_gate.set()
        handle.release_gate.set()
        if writer is not None:
            writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_healthy_owner_never_reaches_the_off_loop_fallback() -> None:
    """The other half of the grace, on a real socket.

    THE GRACE IS A BET THAT IS WORTHLESS IF IT FIRES ON EVERYONE. The off-loop
    path has no display window, so a grace that misfires on healthy owners takes
    the window away from every viewer while the attach still looks fast and every
    other test stays green. This drives the healthy shape end to end: a handle
    that binds on the session loop at once, its off-loop entry point present and
    countable, and the ``frontend_sync`` read off the wire.
    """
    handle = _CountingBindHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = None
    try:
        record = await _wait_record()
        reader, writer = await _dial_frontend(record)
        sync = await _until(reader, "frontend_sync")
        assert sync["data"]["sequence"] >= 0
        assert handle.off_loop_calls == 0, (
            "a handle that answers in microseconds was served off-loop: the "
            "grace is mistuned for healthy owners"
        )
        assert runtime.frontend_off_loop_binds == 0
        await _subscribers_become(handle, 1)
    finally:
        if writer is not None:
            writer.close()
        runtime.close()


def test_the_on_loop_grace_keeps_its_headroom_over_a_healthy_owners_bind() -> None:
    """The constant is pinned to the distribution that sized it, as a RATIO.

    ``_ONLOOP_BIND_GRACE_S`` is justified by measured healthy binds (p50
    4.7-5.0 ms, p95 <= 15 ms for a whole sync) and nothing else in the tree ties
    the two together: a later change that lowered the grace to 20 ms would leave
    every test green while pushing healthy owners onto the fallback, whose only
    trace is the log line this change adds. A ratio rather than a duration, so
    the assertion states the HEADROOM that was chosen and does not pin a
    host-speed number into the suite.
    """
    from local_operator.session.runtime import server as server_module

    healthy_bind_p95_s = 0.015
    assert server_module._ONLOOP_BIND_GRACE_S >= 6 * healthy_bind_p95_s, (
        f"the on-loop grace is {server_module._ONLOOP_BIND_GRACE_S * 1000:.0f} ms, "
        "which is no longer an order of magnitude over a healthy owner's p95 bind "
        "(15 ms) — healthy owners would take the off-loop fallback and lose their "
        "display window"
    )


@pytest.mark.asyncio
async def test_an_attach_is_welcomed_with_identity_alone_while_a_daemon_keeps_its_projection() -> (
    None
):
    """The slim welcome, on the wire, for the client shape it is for.

    A connection asking for the canonical frontend AND the event stream is a full
    terminal or the desktop, and that client DISCARDS the projection
    (``session/attached.py`` builds it with an ``on_projection`` that ignores its
    argument), so the runtime no longer builds and caps a whole projection for it
    inline on the serving loop. The daemon is the client that DOES render a
    projection, and its welcome is untouched — the two halves are asserted
    together so a change that slimmed both would fail here.
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.mobile.types import _projection_from_json

    handle = FakeHandle()
    runtime = RuntimeServer(handle, kind="tui")
    runtime.start()
    writer = daemon_writer = None
    client = None
    try:
        record = await _wait_record()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
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
        welcome = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
        assert welcome["op"] == "welcome", welcome
        data = welcome["data"]
        # THE IDENTITY IS ALL THAT SURVIVES, and it must: the client checks the
        # conversation it landed on against this field before anything else.
        assert data["session_id"] == "s1"
        assert data["conversation_name"] == "fake"
        # THE PAYLOAD IS EMPTY, NOT MERELY SMALL: the collections are present and
        # empty, which is what keeps the frame a valid projection of its own op
        # for a client that rebuilds it field by field — and it is the same object
        # the send ceiling substitutes, so "identity only" has one definition.
        assert data["transcript"] == [], data
        assert data["subagents"] == [], data
        assert data["todos"] == [], data
        assert data["pending"] is None, data
        # THE COMPATIBILITY CLAIM, TESTED RATHER THAN ASSUMED: the client's own
        # parser rebuilds a projection from this payload.
        assert _projection_from_json(data, record).session_id == "s1"

        # ...and a live connection of the real client shape attaches from it.
        client = AttachClient(
            lambda _projection: None,
            lambda _reason: None,
            events=True,
            frontend_state=True,
        )
        await client.connect(record, "s1")

        # THE OTHER HALF: a daemon renders the projection, so it still gets one.
        daemon_reader, daemon_writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        daemon_writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await daemon_writer.drain()
        daemon_welcome = json.loads(await asyncio.wait_for(daemon_reader.readline(), timeout=5))
        assert daemon_welcome["op"] == "projection", daemon_welcome
        assert (
            "transcript" in daemon_welcome["data"]
        ), "the daemon renders the projection; slimming it would blank the phone"
    finally:
        if client is not None:
            await client.detach()
        if writer is not None:
            writer.close()
        if daemon_writer is not None:
            daemon_writer.close()
        runtime.close()


@pytest.mark.asyncio
async def test_a_bind_that_landed_as_the_grace_fired_is_used_rather_than_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same-turn race ``_bind_with_grace``'s second exit exists for.

    FORCED rather than raced. The wrapper below lets the shielded task land and
    THEN reports the grace as expired, which is the schedule in which ``wait_for``
    cancels the shield while the task that shield covered is already done.
    Handing off there would bind the connection twice, release the landed
    subscription a moment later, and mark a HEALTHY owner as window-less — the
    display window being the one thing the on-loop path carries that the off-loop
    path cannot. So the assertion is that the LANDED subscription is returned.
    """
    runtime = RuntimeServer(FakeHandle(), kind="tui")
    real_wait_for = asyncio.wait_for

    async def grace_expires_after_the_bind_landed(fut, timeout, **kwargs):  # noqa: ANN001
        await real_wait_for(fut, timeout, **kwargs)
        raise TimeoutError

    async def bind() -> str:
        return "the on-loop subscription"

    bind_task = asyncio.ensure_future(bind())
    monkeypatch.setattr(asyncio, "wait_for", grace_expires_after_the_bind_landed)
    assert await runtime._bind_with_grace(bind_task) == "the on-loop subscription"
    assert bind_task.done() and not bind_task.cancelled()


@pytest.mark.asyncio
async def test_a_bind_still_parked_past_the_grace_hands_off_and_leaves_the_task_running() -> None:
    """``None`` means HAND OFF, and the shielded task survives to land later.

    The second half is the premise the fallback rests on: if the grace cancelled
    the task there would be nothing for ``_release_when_landed`` to release when
    it lands, and the bind token's job — retiring a relay callback that arrives
    after the fallback has already bound the connection — would have nothing to
    retire.
    """
    runtime = RuntimeServer(FakeHandle(), kind="tui")

    async def bind() -> str:
        await asyncio.sleep(30)
        return "the on-loop subscription"

    bind_task = asyncio.ensure_future(bind())
    try:
        assert await runtime._bind_with_grace(bind_task) is None
        assert not bind_task.done(), "the grace cancelled the task it exists to shield"
        assert not bind_task.cancelled()
    finally:
        bind_task.cancel()
