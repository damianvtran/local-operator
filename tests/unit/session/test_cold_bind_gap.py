"""A cold viewer's FIRST bind paints what the owner wrote after the cold read.

THE DEFECT THIS PINS. ``lop --resume`` and the TUI's ``/resume`` open a
conversation by painting a COLD facade (``AttachedSession.cold``) and binding it
to the live owner behind the paint. Rows the owner writes between the cold read
and the bind were painted by NEITHER route:

* the durable replay (``_replay_durable_suffix``) only ran for a facade that had
  hydrated before and had a ``previous`` window to measure the gap against, and
  a cold facade has neither; and
* the bind folded those rows' ids into ``_history_ids``/``_message_events``, so
  ``_is_duplicate`` swallowed their live ``message_end`` as a replay.

So a turn that finished while the viewer was opening simply never appeared on
screen. ``AttachedSession._replay_cold_gap`` closes it; this file drives the
real facade against a real ``Session`` behind a real ``RuntimeServer`` socket and
counts what the viewer's subscribers receive.

WHAT IS ASSERTED, per the review conditions: every completed gap message is
delivered EXACTLY ONCE (no gap, no duplicate), it arrives in transcript order,
and it arrives BEFORE the live events of the turn still in flight. The busy
case holds the owner mid-turn with a gated model stream, which is the shape the
operator reported (a busy owner) and the one where the duplicate filter bites.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import pytest

from local_operator.harness.types import (
    HistoryDeltaEvent,
    Message,
    MessageEndEvent,
    TextContent,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over

#: Upper bound on any single wait in this file. Every wait is on an EVENT
#: (a turn settling, a delivery arriving); this only turns a hang into a
#: failure with a sentence instead of a stuck worker.
_EVENT_WAIT_S = 20.0


class _GatedStream:
    """A model stream whose Nth call parks until the test releases it.

    ``ScriptedStream`` replays each turn straight through, which cannot hold an
    owner mid-turn; the busy-owner case needs a turn that has STARTED (so the
    owner has live events to relay) and not finished.
    """

    def __init__(self, turns: Sequence[Sequence[Any]], *, hold_call: int) -> None:
        self.turns = [list(turn) for turn in turns]
        self.calls = 0
        self.hold_call = hold_call
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self, request: Any, signal: Any = None) -> AsyncIterator[Any]:
        del request, signal
        index = self.calls
        self.calls += 1
        turn = self.turns[index]

        async def gen() -> AsyncIterator[Any]:
            if index == self.hold_call:
                self.held.set()
                await self.release.wait()
            for event in turn:
                yield event

        return gen()


def _seed_rows(count: int) -> list[Message]:
    rows: list[Message] = []
    for turn in range(count):
        rows.append(
            Message(id=f"seed-user-{turn}", role="user", content=[TextContent(text=f"q {turn}")])
        )
        rows.append(
            Message(
                id=f"seed-assistant-{turn}",
                role="assistant",
                content=[TextContent(text=f"a {turn}")],
                stop_reason="stop",
            )
        )
    return rows


def _delivered_texts(events: list[Any]) -> list[tuple[str, str]]:
    """(route, text) for every completed message the viewer delivered, in order.

    A completed message reaches a frontend by exactly two routes: a
    ``message_end`` from the live relay, or a row inside a ``history_delta``.
    Counting both is what makes "exactly once" a real claim — a row painted by
    the replay AND by a relayed end would appear twice here.
    """
    out: list[tuple[str, str]] = []
    for event in events:
        if isinstance(event, HistoryDeltaEvent):
            for message in event.messages:
                text = "".join(
                    getattr(part, "text", "") for part in (getattr(message, "content", ()) or ())
                )
                out.append(("delta", text))
        elif isinstance(event, MessageEndEvent):
            message = event.message
            text = "".join(
                getattr(part, "text", "") for part in (getattr(message, "content", ()) or ())
            )
            out.append(("end", text))
    return out


async def _serve_owner(directory: Path, session: Any, cwd: Path, monkeypatch) -> tuple[Any, Any]:
    """Stand a REAL owner up for ``session`` and make it the discoverable one.

    Two things a spawned runtime does that an in-process ``RuntimeServer`` does
    not, and both are load-bearing here. The ``.session.pid`` claim marker is
    what ``find_runtime_record`` trusts before any record, so without it the
    viewer finds no owner at all. And the viewer's bind always runs
    ``engage_runtime`` first; unstubbed, that SPAWNS a real runtime child for a
    session whose owner it cannot see, and the viewer then binds to that child
    instead of to the owner under test (measured while writing this file: the
    gap rows arrived from the child's own read and the live turn never did).
    The stub refuses to spawn and proves the owner is discoverable instead.
    """
    import os

    from local_operator.mobile.attach_client import find_runtime_record

    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(cwd))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
    config_dir = directory.parent.parent

    async def no_spawn(session_id, *_args, **_kwargs):  # noqa: ANN001, ANN202
        record, _owner = find_runtime_record(config_dir, session_id)
        assert record is not None, "the owner under test must be discoverable, not spawned"

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", no_spawn)
    return handle, server


def _owner_clients(server: Any) -> int:
    return sum(1 for conn in server._clients.values() if conn.kind == "attach")


async def _until(predicate, what: str) -> None:  # noqa: ANN001
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(poll(), timeout=_EVENT_WAIT_S)
    except TimeoutError:
        raise AssertionError(f"timed out waiting for {what}") from None


@pytest.mark.asyncio
async def test_a_cold_viewer_paints_the_turn_its_owner_finished_before_the_bind(
    tmp_path: Path, monkeypatch
) -> None:
    """Idle owner: a whole turn landed between the cold read and the bind."""
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "coldgap-idle"
    await seed_transcript(directory, _seed_rows(3))
    stream = _GatedStream([text_turn("the gap answer")], hold_call=-1)
    session = build_session(directory, stream, cwd=tmp_path)
    handle, server = await _serve_owner(directory, session, tmp_path, monkeypatch)

    viewer = await AttachedSession.cold(
        directory.name, config_dir=config, cwd=str(tmp_path), takeover_factory=_never_take_over
    )
    events: list[Any] = []
    unsubscribe = viewer.subscribe(events.append)
    try:
        painted_cold = [str(m.id) for m in viewer.history()]
        assert len(painted_cold) == 6, "the cold read painted the seeded transcript"

        # The owner answers a prompt AFTER the cold read: these are the gap rows.
        await session.prompt("the gap question")

        await viewer.bind_runtime()
        assert not viewer.is_cold, "the viewer bound to the live owner"
        assert _owner_clients(server) == 1, "bound to THE owner under test"
        await _until(lambda: _delivered_texts(events), "the gap rows to be delivered")
        await asyncio.sleep(0)

        delivered = _delivered_texts(events)
        texts = [text for _route, text in delivered]
        assert texts.count("the gap question") == 1, delivered
        assert texts.count("the gap answer") == 1, delivered
        assert texts.index("the gap question") < texts.index("the gap answer"), delivered
        # Nothing the cold read already painted is painted a second time.
        assert not any(text in {"q 0", "a 0", "q 2", "a 2"} for text in texts), delivered
    finally:
        unsubscribe()
        await viewer.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_busy_owners_completed_rows_appear_exactly_once_ahead_of_its_live_turn(
    tmp_path: Path, monkeypatch
) -> None:
    """Busy owner: a finished turn in the gap AND a turn still in flight.

    The duplicate-drop hazard in its sharpest form. The gap turn's rows are
    bound into ``_history_ids`` by the attach, so without the replay their
    ``message_end`` is filtered as a duplicate and they are painted by no one;
    the in-flight turn's user row is the same case one step later. After the
    release, the live turn's answer must arrive by the relay, once, AFTER the
    gap rows.
    """
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "coldgap-busy"
    await seed_transcript(directory, _seed_rows(2))
    stream = _GatedStream(
        [text_turn("finished while opening"), text_turn("the live answer")], hold_call=1
    )
    session = build_session(directory, stream, cwd=tmp_path)
    handle, server = await _serve_owner(directory, session, tmp_path, monkeypatch)

    viewer = await AttachedSession.cold(
        directory.name, config_dir=config, cwd=str(tmp_path), takeover_factory=_never_take_over
    )
    events: list[Any] = []
    unsubscribe = viewer.subscribe(events.append)
    live_turn: asyncio.Task[Any] | None = None
    try:
        await session.prompt("asked while opening")
        live_turn = asyncio.create_task(session.prompt("asked and still running"))
        await asyncio.wait_for(stream.held.wait(), timeout=_EVENT_WAIT_S)

        await viewer.bind_runtime()
        assert not viewer.is_cold, "a busy owner still binds"
        assert _owner_clients(server) == 1, "bound to THE owner under test"

        stream.release.set()
        await asyncio.wait_for(live_turn, timeout=_EVENT_WAIT_S)
        await _until(
            lambda: "the live answer" in [t for _r, t in _delivered_texts(events)],
            "the live turn's answer to be relayed",
        )
        await asyncio.sleep(0.05)

        delivered = _delivered_texts(events)
        texts = [text for _route, text in delivered]
        for expected in (
            "asked while opening",
            "finished while opening",
            "asked and still running",
            "the live answer",
        ):
            assert texts.count(expected) == 1, (expected, delivered)
        order = [
            texts.index("asked while opening"),
            texts.index("finished while opening"),
            texts.index("asked and still running"),
            texts.index("the live answer"),
        ]
        assert order == sorted(order), delivered
        assert not any(text.startswith(("q ", "a ")) for text in texts), delivered
    finally:
        stream.release.set()
        if live_turn is not None and not live_turn.done():
            live_turn.cancel()
        unsubscribe()
        await viewer.dispose()
        server.close()
        await handle.dispose()
