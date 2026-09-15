"""A turn's in-flight seed is published only while that turn IS in flight.

``frontend.snapshot.live_events`` is the runtime's bounded transient seed for a
frontend that JOINS MID-TURN, and the viewer folds it AFTER its own durable
history page while stamping every row it creates with its own arrival clock. A
seed that describes a turn which has already ENDED therefore paints its rows at
the reader's arrival - below the conversation's final assistant message. That is
the reported defect, and it had two halves, both inside this repository:

1. The fold that clears the seed runs on the event path, and that path is gated
   in ``Session._emit``. On the normal end path ``agent_end`` is HELD and
   flushed from the pipeline's ``finally``, i.e. after ``_run_turn``'s own
   ``finally`` cleared ``_is_streaming`` - so a runtime with no UI and no
   attached client folded NOTHING at the turn end, and the ended turn's
   ``tool_execution_end`` rows stayed in the frontend state until the next
   turn's ``agent_start``.
2. ``refresh_from_session`` republished ``streaming`` from the live session and
   passed ``live_events`` straight through, so the stale seed rode every later
   snapshot - into the state a joining viewer is handed.

Both are pinned here on a REAL session driven through a REAL turn over real
tools, because a store-level fake cannot see the ordering the defect lives in:
the emit gate, and the held end being flushed after ``_is_streaming`` drops.
The renderer's half of the same report (local-operator-ui#215, placing a seeded
row at its own time instead of the reader's) is not this repository's to test.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.frontend_state import FrontendStateStore
from local_operator.session.session import Session
from local_operator.tools.builtin import build_bash_tool
from tests.e2e.harness import ScriptedStream, build_session, text_turn, tool_call_turn

#: The tool whose call stays outstanding for the duration of a test.
PARK_TOOL = "park"

#: How long a test may wait for the runtime to reach a state it then asserts on.
#: Generous relative to the work (a handful of loop turns) so a loaded CI runner
#: reports the assertion rather than a timeout.
SETTLE_TIMEOUT_S = 20.0


def _parking_tool(released: asyncio.Event) -> AgentTool:
    """A tool whose call parks until the test releases it - a ``wait`` in miniature.

    A stub that returns at once cannot answer this file's question: what the seed
    holds WHILE a call is outstanding, so the call has to be genuinely
    outstanding.
    """

    async def execute(
        call_id: str, _args: Any, _signal: Any, _on_update: Any, _context: Any
    ) -> ToolResult:
        await released.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name=PARK_TOOL,
            content=[TextContent(text="released")],
        )

    return AgentTool(
        name=PARK_TOOL,
        label="Park",
        description="Parks until the test releases it.",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _two_call_turn() -> list[StreamEvent]:
    """One model turn asking for a call that FINISHES and one that stays live.

    Both in one turn on purpose: the calls run concurrently, so the seed can be
    observed holding a settled call beside a running one - which is the state a
    viewer that joins mid-turn has to be able to render and settle.
    """
    return [
        StreamTextDelta(delta="Running both."),
        StreamToolCallDelta(
            index=0,
            id="call-bash",
            name="bash",
            argument_delta=json.dumps({"command": "echo mid-turn"}),
        ),
        StreamToolCallDelta(
            index=1,
            id="call-park",
            name=PARK_TOOL,
            argument_delta=json.dumps({}),
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _build(tmp_path: Path, turns: list[list[StreamEvent]], tools: list[AgentTool]) -> Session:
    """A real session over a real transcript directory in ``tmp_path``."""
    directory = tmp_path / "sessions" / "seedsession01"
    directory.mkdir(parents=True)
    return build_session(directory, ScriptedStream(turns), tools=tools, cwd=directory)


async def _wait_for_seed(session: Session, expected: dict[str, str]) -> None:
    """Wait until the live seed is exactly the ``call id -> type`` map given.

    Waiting on the CONDITION rather than on a fixed sleep: the turn is driven by
    a real event loop, and a clock-based wait would pass on a slow machine for
    the wrong reason or fail on a fast one for no reason. Exact rather than
    subset because the point of the assertion is what the seed holds at ONE
    instant - the window in which a viewer joins.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SETTLE_TIMEOUT_S
    seen: list[dict[str, Any]] = []
    while loop.time() < deadline:
        seen = list(session.frontend_state.live_events)
        if {str(row.get("tool_call_id") or ""): str(row.get("type")) for row in seen} == expected:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"the live seed never became {expected}; it holds "
        f"{[{key: row.get(key) for key in ('type', 'tool_call_id')} for row in seen]}"
    )


@pytest.mark.asyncio
async def test_a_completed_turn_leaves_no_seed_for_an_unobserved_runtime(
    tmp_path: Path,
) -> None:
    """The reported case: a real turn on a runtime nobody is attached to.

    The runtime is asserted to be unobserved first - that is the precondition
    the defect needs (no UI, no subscriber), and a fixture that quietly gained
    either would let this pass for the wrong reason. Asserted on
    ``session.frontend_state`` (the canonical state a snapshot is built from)
    and then on the production join path, because those are two different reads
    of the same field and the defect was visible in both.
    """
    session = _build(
        tmp_path,
        [
            tool_call_turn(
                text="Running the tool.",
                tool_name="bash",
                tool_call_id="call-1",
                arguments={"command": "echo no-viewer"},
            ),
            text_turn("All done."),
        ],
        [build_bash_tool()],
    )
    try:
        assert session._has_ui is False
        assert session._frontend_state_store.has_subscribers is False

        await session.prompt("run the tool")

        state = session.frontend_state
        assert state.streaming is False
        assert list(state.live_events) == [], (
            "an ENDED turn's seed is still in the frontend state, so the next "
            "viewer is handed it and folds it after its own history page"
        )

        subscription = session.subscribe_frontend(lambda _update: None)
        try:
            assert list(subscription.sync.snapshot.live_events) == []
        finally:
            subscription.unsubscribe()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_viewer_attaching_mid_turn_is_handed_the_live_seed(tmp_path: Path) -> None:
    """The invariant the fix must NOT break, and the shape it has to keep.

    A joiner attaching while a call is outstanding must still receive the seed,
    with ONE row per call keyed by the id the later events actually carry: the
    settled call retained as its ``tool_execution_end`` (so a card painted live
    can be settled rather than retired ``⊘ interrupted``) and the running call
    as its ``tool_execution_start``. Double counting would show up here as a
    second row for either id.
    """
    released = asyncio.Event()
    session = _build(
        tmp_path,
        [_two_call_turn(), text_turn("Both done.")],
        [build_bash_tool(), _parking_tool(released)],
    )
    try:
        turn = asyncio.create_task(session.prompt("run both"))
        try:
            await _wait_for_seed(
                session,
                {"call-bash": "tool_execution_end", "call-park": "tool_execution_start"},
            )

            subscription = session.subscribe_frontend(lambda _update: None)
            try:
                seed = list(subscription.sync.snapshot.live_events)
            finally:
                subscription.unsubscribe()

            assert len(seed) == 2, f"one row per call, never two: {seed}"
            assert {str(row["tool_call_id"]): str(row["type"]) for row in seed} == {
                "call-bash": "tool_execution_end",
                "call-park": "tool_execution_start",
            }
        finally:
            released.set()
            await asyncio.wait_for(turn, SETTLE_TIMEOUT_S)

        assert list(session.frontend_state.live_events) == []

        after = session.subscribe_frontend(lambda _update: None)
        try:
            assert list(after.sync.snapshot.live_events) == []
        finally:
            after.unsubscribe()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_turn_boundary_reaches_the_store_with_no_viewer_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's other half, isolated from every publish that could mask it.

    The gate exists to keep PER-TOKEN fold work off a session nobody is
    watching, and the normal turn end reaches it with ``is_streaming`` already
    False (the end is held and flushed after ``_run_turn`` clears the flag).
    ``refresh_from_session`` is stubbed to a no-op here so the ONLY thing that
    can empty the seed is the fold at the boundary - which is the one thing
    this clause contributes. The publish path clears it too
    (``test_a_stale_seed_is_cleared_by_the_next_settled_snapshot``), so this
    test states the gate's own half rather than the end-to-end outcome; the
    state it leaves behind is what a raw ``frontend_state`` read serves (the
    seed accessor the serving and mobile handles expose) before any refresh
    happens.
    """
    session = _build(tmp_path, [], [])
    try:
        store = session._frontend_state_store
        store.mutate(live_events=[{"type": "tool_execution_end", "tool_call_id": "call-1"}])
        assert list(store.state.live_events) != [], "the plant did not land"

        monkeypatch.setattr(
            FrontendStateStore,
            "refresh_from_session",
            lambda self, session, *, initial=False: None,
        )

        await session._emit(AgentEndEvent(generation=1, messages=[]))

        assert list(store.state.live_events) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_stale_seed_is_cleared_by_the_next_settled_snapshot(tmp_path: Path) -> None:
    """The gate itself, deliberately isolated from the fold.

    Admitting turn boundaries to the fold gate repairs the normal end path, but
    the invariant is stronger than that one path and has to survive a fold that
    was missed for any reason: NO session with no turn in flight may publish a
    seed. Driven here by planting stale rows directly in the store, so the only
    thing that can clear them is ``refresh_from_session``'s non-streaming gate -
    and asserted on the DELTA as well as on the state, because a follower
    applies deltas and one that is never told keeps rendering what it was
    handed.
    """
    session = _build(tmp_path, [text_turn("nothing to do")], [])
    try:
        store = session._frontend_state_store
        rows = [{"type": "tool_execution_end", "tool_call_id": "stale-1", "tool_name": "bash"}]
        store.mutate(live_events=rows)
        published: list[Any] = []
        store.subscribe(published.append)
        assert list(store.state.live_events) == rows, "the plant did not land"

        store.refresh_from_session(session)

        assert list(store.state.live_events) == []
        assert any(
            update.changes.get("live_events") == [] for update in published
        ), "the clear must be published, not just held locally"
    finally:
        await session.dispose()
