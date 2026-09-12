"""The wire mute: a parked viewer's owner stops SENDING delta-grade frames.

``tests/unit/tui/test_parked_source_seam.py`` covers the app-side drop (what
the parked controller does with frames that arrive). What THIS module covers is
the layer under it, which the app-side drop cannot see: the delivery itself.

WHY THE LAYER MATTERS. A parked source keeps its subscription so the
conversation stays warm, and every delta it receives is materialised -- socket
read, JSON decode, event deserialization -- and only then discarded. Measured
on the reporting machine: ~0.26 ms per delivered-and-discarded frame, linear in
frames/s (68 ms/s of loop CPU at 12 streaming sessions, 117 ms/s at double the
rate), scaling with the frame's payload because ``message_update`` carries the
ACCUMULATED message. The owner also serialises each frame once per connection.
``event-mute-v1`` moves the drop to the sender: a parked viewer asks its owner
to stop sending the same three delta-grade types (``EVENT_MUTE_DROP_TYPES``)
and resumes them on reveal, where the presentation rebuilds from history plus
the canonical live seed exactly as it already does after any parked gap.

WHAT MUST NOT REGRESS, tested below one by one:

* the suppression itself (a parked viewer receives no delta frame) -- this is
  the pinned regression; it fails against a tree without the server-side
  filter, where the frames arrive and are discarded app-side;
* skew in both directions -- a new viewer against an owner that cannot mute,
  and an old viewer against an owner that can;
* a reconnect re-asserting the mute (the mute is per CONNECTION, and a fresh
  socket starts unmuted);
* reveal freshness -- the muted-then-revealed viewer ends with the same
  settled state as an always-live one.
"""

from __future__ import annotations

import asyncio
import dataclasses
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    AgentToolUpdate,
    Message,
    MessageUpdateEvent,
    SubagentProgressEvent,
    TextContent,
    ToolExecutionUpdateEvent,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import (
    EVENT_MUTE_CAPABILITY,
    EVENT_MUTE_DROP_TYPES,
)
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over

#: How long to let loopback frames round-trip before reading a count. Waiting
#: LONGER is always safe for the suppression assertions (they assert zero) and
#: the arrival assertions only need the frames to have landed; this is a
#: settle window, not a latency bound.
_SETTLE = 0.35


async def _settle() -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)


async def _mute(viewer: AttachedSession, muted: bool) -> None:
    """Toggle the wire mute and wait for the op to be acked.

    ``set_event_mute`` is deliberately synchronous (the park toggle's shape);
    it spawns the send. Awaiting the spawned tasks here is what makes the
    wire-level assertions below deterministic instead of racing the op.
    """
    viewer.set_event_mute(muted)
    for _ in range(10):
        tasks = list(viewer._event_mute_tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)


def _delta_grade(index: int) -> list[object]:
    """One of EACH type in ``EVENT_MUTE_DROP_TYPES``, not just message_update."""
    message = Message.assistant(f"token {index}")
    message.id = f"mute-live-{index}"
    return [
        MessageUpdateEvent(message=message, delta=f"tok{index}"),
        ToolExecutionUpdateEvent(
            tool_call_id=f"mute-call-{index}",
            tool_name="read",
            partial_result=AgentToolUpdate(),
        ),
        SubagentProgressEvent(job_id=f"mute-job-{index}", label="child", progress="working"),
    ]


def _settled_message(index: int, text: str) -> Message:
    return Message(
        id=f"mute-settled-{index}",
        role="assistant",
        content=[TextContent(text=text)],
        stop_reason="stop",
    )


@asynccontextmanager
async def _remote(tmp_path: Path, name: str) -> AsyncIterator[AttachedSession]:
    """A real owner runtime plus a real ``AttachedSession`` viewer over it."""
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    await seed_transcript(directory, [])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    record = server._record
    remote = await AttachedSession.connect(
        record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )
    try:
        yield remote
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


class _WireLog:
    """Counts what the owner enqueues for the viewer's connection, per type."""

    def __init__(self) -> None:
        self.types: list[str] = []


@asynccontextmanager
async def _counting_server(tmp_path: Path, name: str) -> AsyncIterator[tuple[Any, Any, _WireLog]]:
    """Owner + handle + a log of frames ENQUEUED to attach connections.

    Counting at ``_enqueue_client_frame`` is the wire layer the mute gates: it
    cannot be fooled by a subscription that captured an older bound method, and
    it is exactly what the owner would have written to the socket.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    await seed_transcript(directory, [])
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()

    log = _WireLog()
    original = RuntimeServer._enqueue_client_frame

    def counting(self, conn, frame):  # type: ignore[no-untyped-def]
        if self is server and str(conn.kind) == "attach":
            log.types.append(str((frame.get("data") or {}).get("type") or frame.get("op")))
        return original(self, conn, frame)

    RuntimeServer._enqueue_client_frame = counting  # type: ignore[method-assign]
    try:
        yield server, session, log
    finally:
        RuntimeServer._enqueue_client_frame = original  # type: ignore[method-assign]
        server.close()
        await handle.dispose()


def _deltas(log: _WireLog) -> int:
    return sum(1 for kind in log.types if kind in EVENT_MUTE_DROP_TYPES)


def _state_events(log: _WireLog) -> int:
    return sum(1 for kind in log.types if kind in {"agent_start", "agent_end"})


@pytest.mark.asyncio
async def test_muted_viewer_receives_no_delta_frames(tmp_path) -> None:
    """THE PINNED REGRESSION: a parked viewer's owner stops sending deltas.

    Against a tree without the server-side filter this fails: ``events_muted``
    is never consulted, every delta is enqueued, delivered and materialised,
    and the count below is non-zero. The state-event half is the guard in the
    other direction -- muting everything would pass the first assertion while
    silently rotting a parked viewer's turn state.
    """
    async with _counting_server(tmp_path, "mute-a") as (server, owner, log):
        viewer = await AttachedSession.connect(
            server._record,
            "synthetic-mute-a",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        try:
            assert viewer.supports_event_mute, (
                "the owner must advertise the capability or the viewer cannot "
                "negotiate the mute at all"
            )
            await _mute(viewer, True)
            baseline = len(log.types)

            for index in range(6):
                for event in _delta_grade(index):
                    await owner._emit(event)  # type: ignore[attr-defined]
            await owner._emit(AgentStartEvent(generation=7))  # type: ignore[attr-defined]
            await _settle()

            new = log.types[baseline:]
            assert sum(1 for kind in new if kind in EVENT_MUTE_DROP_TYPES) == 0, (
                "a muted connection must receive no delta-grade frames; got "
                f"{[kind for kind in new if kind in EVENT_MUTE_DROP_TYPES]}"
            )
            assert any(kind == "agent_start" for kind in new), (
                "state-bearing events must keep flowing while muted, or a "
                "parked viewer's turn state silently rots"
            )

            # And unmuting resumes the stream: the reveal path depends on it.
            await _mute(viewer, False)
            baseline = len(log.types)
            for event in _delta_grade(99):
                await owner._emit(event)  # type: ignore[attr-defined]
            await _settle()
            resumed = [kind for kind in log.types[baseline:] if kind in EVENT_MUTE_DROP_TYPES]
            assert len(resumed) == len(
                EVENT_MUTE_DROP_TYPES
            ), f"deltas must resume after unmute; got {resumed}"
        finally:
            await viewer.dispose()


@pytest.mark.asyncio
async def test_mute_is_gated_on_the_capability_new_viewer_old_owner(tmp_path) -> None:
    """Skew, new viewer x old owner: no capability, no send, no behaviour change.

    An owner that predates ``event-mute-v1`` never advertised it, so the viewer
    must not send the op. The observable contract is double: the request is
    refused locally (``supports_event_mute`` False, no task spawned) and the
    owner keeps delivering everything -- the pre-mute behaviour, which the
    app-side drop already covers.
    """
    async with _counting_server(tmp_path, "mute-b") as (server, owner, log):
        record = dataclasses.replace(
            server._record,
            capabilities=[
                cap for cap in server._record.capabilities if cap != EVENT_MUTE_CAPABILITY
            ],
        )
        assert EVENT_MUTE_CAPABILITY not in record.capabilities
        viewer = await AttachedSession.connect(
            record,
            "synthetic-mute-b",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        try:
            assert not viewer.supports_event_mute, (
                "the viewer must read the capability from the RECORD: sending "
                "the op blind would raise an error frame against an old owner"
            )
            baseline = len(log.types)
            viewer.set_event_mute(True)  # must be a local no-op, not a send
            assert not viewer._event_mute_tasks, "no op may be sent without the capability"
            for index in range(3):
                for event in _delta_grade(index):
                    await owner._emit(event)  # type: ignore[attr-defined]
            await _settle()
            delivered = [kind for kind in log.types[baseline:] if kind in EVENT_MUTE_DROP_TYPES]
            assert len(delivered) == len(_delta_grade(0)) * 3, (
                "without the capability the owner must keep delivering exactly "
                f"as before; got {delivered}"
            )
        finally:
            await viewer.dispose()


@pytest.mark.asyncio
async def test_old_viewer_against_new_owner_keeps_full_delivery(tmp_path) -> None:
    """Skew, old viewer x new owner: the default is unmuted, deltas flow.

    A viewer that never sends the op (every shipped build at the time of
    writing) must see byte-identical behaviour: ``events_muted`` defaults
    False, the relay sends everything, and the app-side drop remains the only
    filter. The capability string is advertised so NEW viewers can negotiate.
    """
    async with _counting_server(tmp_path, "mute-c") as (server, owner, log):
        assert (
            EVENT_MUTE_CAPABILITY in server._record.capabilities
        ), "the mute must be advertised for clients that know how to ask"
        viewer = await AttachedSession.connect(
            server._record,
            "synthetic-mute-c",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        try:
            baseline = len(log.types)
            for index in range(3):
                for event in _delta_grade(index):
                    await owner._emit(event)  # type: ignore[attr-defined]
            await _settle()
            delivered = [kind for kind in log.types[baseline:] if kind in EVENT_MUTE_DROP_TYPES]
            assert len(delivered) == len(_delta_grade(0)) * 3
        finally:
            await viewer.dispose()


@pytest.mark.asyncio
async def test_reconnect_reasserts_the_mute(tmp_path) -> None:
    """The mute is per CONNECTION: a redial must put it back by itself.

    Without the re-assert inside the dial path, a parked viewer that lost its
    socket resumes paying full delta delivery for the rest of its parking --
    silently, because the app-side drop keeps the UI correct. The state-event
    half proves the NEW connection is live, so "no deltas" cannot pass merely
    because nothing was reconnected.
    """
    async with _counting_server(tmp_path, "mute-d") as (server, owner, log):
        viewer = await AttachedSession.connect(
            server._record,
            "synthetic-mute-d",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        try:
            await _mute(viewer, True)

            # Kill the socket and redial the way recovery does. The desired
            # mute state is remembered on the viewer, not on the dead client.
            client = viewer._client
            assert client is not None
            client.close()
            await _settle()
            pending = await viewer._dial(server._record)
            if pending is not None:
                await pending
            await _settle()

            baseline = len(log.types)
            for index in range(4):
                for event in _delta_grade(index):
                    await owner._emit(event)  # type: ignore[attr-defined]
            await owner._emit(AgentStartEvent(generation=8))  # type: ignore[attr-defined]
            await _settle()
            new = log.types[baseline:]
            assert any(kind == "agent_start" for kind in new), (
                "the redialed connection must be live for this test to mean "
                "anything; no frames on the new socket at all"
            )
            assert sum(1 for kind in new if kind in EVENT_MUTE_DROP_TYPES) == 0, (
                "a parked viewer that reconnected resumed receiving delta "
                f"frames: {[kind for kind in new if kind in EVENT_MUTE_DROP_TYPES]}"
            )
        finally:
            await viewer.dispose()


@pytest.mark.asyncio
async def test_revealed_viewer_ends_with_the_same_settled_state_as_a_live_one(
    tmp_path,
) -> None:
    """Reveal freshness: muted during a turn, equal state at its end.

    The always-live viewer is the control: same owner, same events, never
    muted. The muted one must end with the same settled row -- the full text,
    not the fragment it saw before muting -- because the authoritative text
    travels on the settled row and the reveal rebuilds from history.
    """
    async with _counting_server(tmp_path, "mute-e") as (server, owner, log):
        live = await AttachedSession.connect(
            server._record,
            "synthetic-mute-e",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        muted = await AttachedSession.connect(
            server._record,
            "synthetic-mute-e",
            config_dir=tmp_path / "config",
            takeover_factory=_never_take_over,
            display_window=True,
        )
        try:
            await _mute(muted, True)

            full_text = "The complete answer, settled while the viewer was parked."
            partial = Message.assistant("The complete answ")
            partial.id = "mute-e-live"
            await owner._emit(AgentStartEvent(generation=9))  # type: ignore[attr-defined]
            await owner._emit(  # type: ignore[attr-defined]
                MessageUpdateEvent(message=partial, delta="The complete answ")
            )
            settled = _settled_message(0, full_text)
            from local_operator.harness.types import MessageEndEvent

            await owner._emit(MessageEndEvent(message=settled))  # type: ignore[attr-defined]
            await owner._emit(  # type: ignore[attr-defined]
                AgentEndEvent(aborted=False, error=None)
            )
            await _settle()

            await _mute(muted, False)
            await _settle()

            assert live.history_message_count == muted.history_message_count, (
                "the revealed viewer must hold the same number of durable rows "
                "as the always-live one"
            )
            assert muted.history_message_count >= 1
            texts = [str(getattr(row, "text", "") or "") for row in muted.display_history_window()]
            assert any(full_text in text for text in texts), (
                "the settled row must carry the FULL text after reveal, not the "
                "fragment the muted viewer saw before the mute: got "
                f"{[text[:60] for text in texts]}"
            )
        finally:
            await live.dispose()
            await muted.dispose()


@pytest.mark.asyncio
async def test_controller_park_toggles_the_wire_mute(tmp_path) -> None:
    """The controller is the caller: ``set_parked`` must drive the session API.

    Pins the app-side wiring so deleting the ``set_event_mute`` call from
    ``EventController.set_parked`` cannot pass unnoticed even though the wire
    contract above stays green.
    """
    async with _remote(tmp_path, "mute-f") as viewer:
        calls: list[bool] = []
        original = viewer.set_event_mute

        def spy(muted: bool) -> None:
            calls.append(muted)
            original(muted)

        viewer.set_event_mute = spy  # type: ignore[method-assign]

        from local_operator.tui.events import EventController

        # The real constructor: ``set_parked`` must exercise the production
        # object, and a park toggle is a synchronous call on the app's loop.
        controller = EventController(viewer, None)

        controller.set_parked(True)
        controller.set_parked(False)
        controller.set_parked(False)  # idempotent: no third call

        assert calls == [True, False], (
            "parking must ask the owner to mute and reveal to unmute, and an "
            f"unchanged state must not re-send: got {calls}"
        )
