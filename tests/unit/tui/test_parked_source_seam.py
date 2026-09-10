"""The parked-source latch, driven through the REAL app seam.

The controller-level behaviour of ``EventController.set_parked`` is covered in
``tests/unit/tui/test_events.py``. What is covered HERE is the thing that
module structurally cannot see: whether the app actually *drives* the mode at
the two transitions that own it.

Why a separate, heavier file. Mutation testing during review round 1 deleted
the ``set_parked(False)`` call from ``_adopt_session`` and both existing suites
stayed green (26 passed + 62 passed), because the unpark test calls the setter
directly instead of exercising the seam that is supposed to invoke it. That is
the single worst failure this feature can produce -- a conversation the user
clicked into renders its boundaries but paints **no streaming tokens at all**,
silently, for the whole turn -- and nothing observed it. A test that drives the
production prepare/commit path is the only shape that can.

Both directions are asserted, because the mute is a two-sided latch and each
side has its own failure:

* commit must UNPARK the incoming source, or the visible conversation goes
  blank mid-turn (review R2);
* commit must RE-PARK the outgoing source, or the mute is a one-way latch and
  every session the user visits stays exempt for the rest of its
  ``SIDEBAR_IDLE_RELEASE_S`` retention, still paying full per-token delivery
  while hidden (review R1).

The assertions are on DELIVERED DELTAS -- real events pushed through a real
``RemoteSession`` subscription into the real controller -- not on the ``parked``
flag alone. A flag is the API; the delta count is the mechanism, and review
round 1 found the guard density on the mechanism thinner than the test count
suggested (QA Q1). Each assertion below fails if the corresponding
``set_parked`` call is removed from ``app.py``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from local_operator.harness.types import (
    AgentToolUpdate,
    Message,
    MessageStartEvent,
    MessageUpdateEvent,
    SubagentProgressEvent,
    TextContent,
    ToolExecutionUpdateEvent,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import AssistantDelta
from local_operator.tui.session_interaction import SessionInteraction
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over


def _rows(count: int = 3) -> list[Message]:
    return [
        Message(
            id=f"seam-row-{index:04}",
            role="assistant",
            content=[TextContent(text=f"Saved row {index:04}")],
            stop_reason="stop",
        )
        for index in range(count)
    ]


@asynccontextmanager
async def _remote(tmp_path: Path, name: str):
    """A real owner runtime plus a real ``RemoteSession`` viewer over it.

    Mirrors ``test_rendered_history_paging.remote_session``; kept local so this
    module does not import a neighbour's private fixture, and because the
    sessions here need no history paging at all.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    await seed_transcript(directory, _rows())
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await RemoteSession.connect(
        server._record,
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


def _delta_grade(index: int) -> list[object]:
    """One of EACH type in ``_PARKED_DROP_TYPES``, not just ``message_update``.

    QA round 1 (Q2) observed 100% ``message_update`` in its load generator, so
    ``tool_execution_update`` and ``subagent_progress`` were covered by unit
    test only. Emitting all three here puts every member of the drop set
    through the real subscription at the real seam.
    """
    message = Message.assistant(f"token {index}")
    message.id = f"seam-live-{index}"
    return [
        MessageUpdateEvent(message=message, delta=f"tok{index}"),
        ToolExecutionUpdateEvent(
            tool_call_id=f"seam-call-{index}",
            tool_name="read",
            partial_result=AgentToolUpdate(),
        ),
        SubagentProgressEvent(job_id=f"seam-job-{index}", label="child", progress="working"),
    ]


async def _pump(pilot, count: int = 6) -> None:
    for _ in range(count):
        await pilot.pause()


@pytest.mark.asyncio
async def test_commit_unparks_the_incoming_source_and_reparks_the_outgoing(tmp_path) -> None:
    """The two-sided latch, at the seam, measured in delivered deltas.

    Deleting either ``set_parked`` call from ``app.py`` fails this test; the
    round-1 mutation that survived both existing suites is the ``False`` half.
    """
    async with (
        _remote(tmp_path, "home") as home,
        _remote(tmp_path, "first") as first,
        _remote(tmp_path, "second") as second,
    ):
        # The app boots on a REAL owner-backed session: `_commit_sidebar_session`
        # refuses to switch away from anything else, so a `FakeSession` current
        # session cannot reach the seam under test at all.
        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break

            sources: dict[str, SessionInteraction] = {}
            for remote in (first, second):
                source = SessionInteraction(remote)
                sources[remote.session_id] = source
                app._sidebar_sources[remote.session_id] = source

            async def lease(session_id, *, speculative=False):
                # Only external discovery is redirected: the lease body's own
                # park-at-birth is what the first assertion reads, so it is
                # reproduced here exactly as `_lease_sidebar_source` does it
                # rather than being skipped.
                source = sources[session_id]
                source.preparations += 1
                if source.controller is None:
                    from local_operator.tui.events import EventController

                    # `_interactions` keyed by id(session) is what `_adopt_session`
                    # resolves the incoming source through: without it the adopt
                    # builds a FRESH SessionInteraction and unparks that instead,
                    # which is the production wiring `_lease_sidebar_source` does.
                    app._interactions[id(source.session)] = source
                    source.controller = EventController(source.session, app)
                    app._event_sources[source.controller] = source
                    source.controller.set_parked(True)
                    source.controller.subscribe()
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            first_source = sources[first.session_id]
            second_source = sources[second.session_id]

            # --- commit the FIRST session: it must come off the mute --------
            prepared = await app._prepare_sidebar_session(first.session_id)
            ready = app._commit_sidebar_session(
                first.session_id, prepared, app._sidebar_navigation.generation
            )
            await _pump(pilot, 12)
            if ready is not None and not ready.done():
                ready.cancel()

            assert first_source.controller is not None
            assert not first_source.controller.parked, (
                "the committed source is the conversation on screen: leaving it "
                "parked paints no streaming tokens at all for the whole turn"
            )

            # The mechanism, not the flag: a real delta must reach the app.
            controller = first_source.controller
            posted: list[object] = []
            original_post = controller._post
            controller._post = posted.append  # type: ignore[method-assign]
            try:
                for event in _delta_grade(1):
                    controller._on_event(event)
                controller._flush_assistant()
            finally:
                controller._post = original_post  # type: ignore[method-assign]
            assert any(isinstance(message, AssistantDelta) for message in posted), (
                "a committed source delivered no assistant delta: the unpark at "
                "the commit seam is missing and the visible stream is blank"
            )

            # --- switch AWAY: the outgoing source must go back on the mute --
            prepared = await app._prepare_sidebar_session(second.session_id)
            ready = app._commit_sidebar_session(
                second.session_id, prepared, app._sidebar_navigation.generation
            )
            await _pump(pilot, 12)
            if ready is not None and not ready.done():
                ready.cancel()

            assert second_source.controller is not None
            assert not second_source.controller.parked, "the newly visible source must paint"
            assert first_source.controller.parked, (
                "a source the user switched AWAY from stays unparked forever: the "
                "mute is a one-way latch, so every visited session keeps paying "
                "full per-token delivery for the rest of its retention (R1)"
            )

            # And again as delivered traffic, for every member of the drop set.
            controller = first_source.controller
            posted = []
            original_post = controller._post
            controller._post = posted.append  # type: ignore[method-assign]
            try:
                for event in _delta_grade(2):
                    controller._on_event(event)
            finally:
                controller._post = original_post  # type: ignore[method-assign]
            assert posted == [], (
                "the re-parked outgoing source still minted Textual messages for "
                f"delta-grade events: {posted!r}"
            )

            # A re-parked source must remain SUBSCRIBED, or its owner's stream
            # buffers in `_emit_or_buffer` and lands in one drain at the next
            # click -- the failure mode the rejected deferred-subscribe design
            # had, arriving one seam later.
            assert first._handlers, "a re-parked source must stay subscribed"


@pytest.mark.asyncio
async def test_returning_to_a_reparked_source_paints_its_stream_again(tmp_path) -> None:
    """The round trip: park -> visit -> leave -> RETURN must not stay muted.

    Re-parking on switch-away (R1) creates a state the one-way latch never
    could: a source that is parked for the SECOND time. If the unpark did not
    re-run on the return commit, the fix for R1 would itself produce the blank
    stream R2 is about -- so the round trip is asserted, not assumed.
    """
    async with (
        _remote(tmp_path, "origin") as origin,
        _remote(tmp_path, "alpha") as alpha,
        _remote(tmp_path, "beta") as beta,
    ):

        async def factory():
            return origin

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is origin:
                    break

            sources: dict[str, SessionInteraction] = {}
            for remote in (alpha, beta):
                source = SessionInteraction(remote)
                sources[remote.session_id] = source
                app._sidebar_sources[remote.session_id] = source

            async def lease(session_id, *, speculative=False):
                source = sources[session_id]
                source.preparations += 1
                if source.controller is None:
                    from local_operator.tui.events import EventController

                    # `_interactions` keyed by id(session) is what `_adopt_session`
                    # resolves the incoming source through: without it the adopt
                    # builds a FRESH SessionInteraction and unparks that instead,
                    # which is the production wiring `_lease_sidebar_source` does.
                    app._interactions[id(source.session)] = source
                    source.controller = EventController(source.session, app)
                    app._event_sources[source.controller] = source
                    source.controller.set_parked(True)
                    source.controller.subscribe()
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            async def visit(session_id: str) -> None:
                prepared = await app._prepare_sidebar_session(session_id)
                ready = app._commit_sidebar_session(
                    session_id, prepared, app._sidebar_navigation.generation
                )
                await _pump(pilot, 12)
                if ready is not None and not ready.done():
                    ready.cancel()

            await visit(alpha.session_id)
            await visit(beta.session_id)
            await visit(alpha.session_id)

            controller = sources[alpha.session_id].controller
            assert controller is not None
            assert not controller.parked, "the returned-to source must be live again"

            posted: list[object] = []
            original_post = controller._post
            controller._post = posted.append  # type: ignore[method-assign]
            try:
                controller._on_event(MessageStartEvent(message=Message.assistant("")))
                for event in _delta_grade(3):
                    controller._on_event(event)
                controller._flush_assistant()
            finally:
                controller._post = original_post  # type: ignore[method-assign]

            assert any(isinstance(message, AssistantDelta) for message in posted), (
                "a source parked a SECOND time by the switch-away re-park never "
                "came back: returning to it paints no tokens"
            )
