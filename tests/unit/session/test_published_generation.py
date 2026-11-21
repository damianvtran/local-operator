"""V15: snapshots, updates and turn events publish ONE generation counter.

``agent_start``/``agent_end`` carry the LOGICAL turn generation: a run that
compacts and continues (``_drain_continuation`` re-enters ``_run_turn``) is still
the one turn the user started, so its end is re-stamped with the first run's
generation (``_flush_held_end``). The frontend snapshot used to publish the raw
per-run counter ``_generation`` instead, which the continuation bumps. A viewer
that (re)subscribed during the continuation therefore read generation N+1 while
that same turn's ``agent_end`` said N, and every guard that compares the two
-- the desktop transcript reducer's "drop an older end", the voice binding's
"same generation keeps the take" -- threw the turn's own answer away (review
round 7, R7-M1).

Driven on the real ``Session``; only the provider stream is scripted and
``_maybe_compact`` is replaced by one that queues a continuation, which is what
a real post-compaction run does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    CompactionEndEvent,
    CompactionStartEvent,
    Message,
)
from tests.unit.session.test_session import (
    ScriptedStream,
    StreamEndEvent,
    StreamTextDelta,
    make_session,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("continuations", [1, 2])
async def test_the_snapshot_publishes_the_turn_events_generation_through_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, continuations: int
) -> None:
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta=f"run {index}"), StreamEndEvent(stop_reason="stop")]
            for index in range(continuations + 1)
        ]
    )
    session = make_session(tmp_path, stream)
    compactions = {"n": 0}

    async def compact_and_continue() -> None:
        compactions["n"] += 1
        if compactions["n"] <= continuations:
            await session._emit(CompactionStartEvent(reason="context-window"))
            await session._emit(CompactionEndEvent(reason="context-window", success=True))
            session._continuation_queue.append(Message.user("continue"))

    monkeypatch.setattr(session, "_maybe_compact", compact_and_continue)

    starts: list[int] = []
    ends: list[int] = []
    # (run index, snapshot.streaming, snapshot.generation) taken by a window that
    # subscribes fresh at the end of every run's assistant message -- including
    # the ones inside a continuation, which is where the counters used to split.
    snapshots: list[tuple[int, bool, int]] = []
    updates: list[int] = []

    def on_event(event: Any) -> None:
        if event.type == "agent_start":
            starts.append(event.generation)
        elif event.type == "agent_end":
            ends.append(event.generation)
        elif event.type == "message_end" and getattr(event.message, "role", "") == "assistant":
            sync = session.subscribe_frontend(lambda _update: None).sync
            snapshots.append(
                (len(snapshots), bool(sync.snapshot.streaming), int(sync.snapshot.generation))
            )

    session.subscribe(on_event)
    live = session.subscribe_frontend(
        lambda update: (
            updates.append(int(update.changes["generation"]))
            if "generation" in getattr(update, "changes", {})
            else None
        )
    )
    assert live.sync.snapshot.generation == 0

    await session.prompt("do the thing")
    try:
        # One logical turn, however many runs it took.
        assert len(starts) == 1 and len(ends) == 1, (starts, ends)
        (turn,) = starts
        assert ends == [turn]
        assert compactions["n"] == continuations + 1
        # A snapshot at EVERY run, the continuation runs included, says the
        # turn is streaming under the same generation its agent_end carries.
        assert len(snapshots) == continuations + 1, snapshots
        assert all(streaming for _, streaming, _ in snapshots), snapshots
        assert [generation for _, _, generation in snapshots] == [turn] * (continuations + 1), (
            snapshots,
            turn,
        )
        # Live frontend updates never announce a generation the turn events do not.
        assert set(updates) <= {turn}, (updates, turn)
        assert session.frontend_state.generation == turn
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_next_turn_after_a_continued_one_still_moves_the_generation_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One counter does not mean a frozen one, and between turns it never runs ahead.

    Between turns the raw per-run counter is published (no turn is live, so there
    is no logical generation to prefer). What every guard needs from it is order:
    the next turn's ``agent_start`` must be ABOVE whatever an idle snapshot said,
    or the reducer's same-generation check would swallow the start.
    """
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta=f"run {index}"), StreamEndEvent(stop_reason="stop")]
            for index in range(3)
        ]
    )
    session = make_session(tmp_path, stream)
    compactions = {"n": 0}

    async def compact_once() -> None:
        compactions["n"] += 1
        if compactions["n"] == 1:
            session._continuation_queue.append(Message.user("continue"))

    monkeypatch.setattr(session, "_maybe_compact", compact_once)
    starts: list[int] = []
    ends: list[int] = []
    during_second: list[int] = []

    def on_event(event: Any) -> None:
        if event.type == "agent_start":
            starts.append(event.generation)
        elif event.type == "agent_end":
            ends.append(event.generation)
        elif event.type == "message_end" and len(starts) == 2:
            during_second.append(
                int(session.subscribe_frontend(lambda _u: None).sync.snapshot.generation)
            )

    session.subscribe(on_event)
    try:
        await session.prompt("first")
        idle = session.frontend_state.generation
        await session.prompt("second")
        assert starts == ends, (starts, ends)
        first, second = starts
        assert first < idle < second or idle == first, (first, idle, second)
        assert second > idle, "the next turn's start must be above the idle snapshot"
        assert during_second and set(during_second) == {second}, (during_second, second)
    finally:
        await session.dispose()
