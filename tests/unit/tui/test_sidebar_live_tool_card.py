"""A sidebar switch must not paint a still-running tool call as interrupted.

The operator resumed a session from the sidebar while a long ``wait``
(``wait_ms=1800000``) was parked. The row read ``⊘ interrupted`` immediately,
every time, while the band beside it said "working" — and the wait was not
interrupted at all: it ran to its deadline and returned
``job … still running after 1800000ms`` through ``_wait``'s clean DEADLINE
branch, with no ``interrupted_by`` anywhere in the transcript.

So the tool was fine and the PRESENTATION was wrong, and the mechanism is
structural rather than a race:

* the viewer files each completed live row into ``RemoteSession._live_history``
  (``_remember_live``), and ``display_history_window()`` hands those rows to
  the next prepared replay;
* a conversation sitting in the sidebar therefore replays an assistant message
  whose ``tool_calls`` have NO answer, because the result does not exist yet;
* ``session_presentation.replay_tool_call`` has exactly two outcomes for a call
  with no result — settled, or ``interrupted`` — so it painted ``⊘``;
* ``_mark_pending_tool_rows`` repainted only rows held by a pending GATE, and a
  tool that is executing has no gate open.

Nothing retired the card: ``_retire_live_tool_cards`` never runs on this path
(asserted below, because "a retirement fired" is the other plausible cause and
ruling it out is what makes this test name the right mechanism).

The guard is a PAIR, and both halves matter equally:

* a call the owner is still executing paints live, never ``⊘``;
* a call from a turn that really did die mid-flight still paints ``⊘`` — the
  round-1 Q2 regression referenced throughout ``frontend_state.py`` is exactly
  the opposite mistake, and a fix that simply stopped painting ``interrupted``
  would pass the first assertion and reintroduce that one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    AgentTool,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp, ToolCard
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    text_turn,
    tool_call_turn,
    user_message,
    wait_for_adoption,
)
from tests.unit.harness.test_comms import DEADLOCK_GUARD_S, MAX_PUMP_TURNS

#: Not ``wait``. ``Session._merge_capability_tools`` merges the real ``wait``
#: builtin into every session it builds, and a same-named double is SHADOWED by
#: it — the turn then answers ``unknown job …`` and the test passes vacuously
#: against a settled error row. The bug is about any long-running tool, so a
#: distinct name is both honest and the only shape that actually exercises it.
PARKING_TOOL = "await_job"


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Headless apps must never rename the caller's real multiplexer workspace.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


def _parking_tool(released: asyncio.Event) -> AgentTool:
    """A tool that parks until the test releases it — a `wait` in miniature.

    Real parking rather than a stub returning instantly: the whole question is
    what the screen shows WHILE a call is outstanding, so the call has to be
    genuinely outstanding for the duration of the switch.
    """

    async def execute(
        call_id: str, _args: Any, _signal: Any, _on_update: Any, _context: Any
    ) -> ToolResult:
        await released.wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name=PARKING_TOOL,
            content=[TextContent(text="the job finished")],
        )

    return AgentTool(
        name=PARKING_TOOL,
        label="Await",
        description="Awaits a background job.",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}, "wait_ms": {"type": "integer"}},
            "required": ["job_id"],
        },
        execute=execute,
    )


@asynccontextmanager
async def live_owners(
    config: Path, ids: list[str], tool: AgentTool, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[[str | None], Awaitable[RemoteSession]]]:
    """Real ``RuntimeServer`` owners behind one sidebar, one per id.

    Re-keyed discovery records, for the reason ``test_sidebar_longrun`` states
    at length: ``registry.publish`` keys by PID and every owner here shares this
    test's PID, so without this N servers overwrite ONE file and the catalog
    reports a single live row.
    """
    from local_operator.session.runtime import registry

    def publish(record: Any, root: Path | None = None) -> Path:
        directory = registry.run_dir(root)
        record.heartbeat_at = time.time()
        handle, path = tempfile.mkstemp(dir=directory, prefix=".x.", suffix=".tmp")
        with os.fdopen(handle, "w") as stream:
            json.dump(record.to_json(), stream)
        target = directory / f"{record.pid}-{record.session_id}.json"
        os.replace(path, target)
        return target

    monkeypatch.setattr(registry, "publish", publish)
    monkeypatch.setattr(registry, "unpublish", lambda pid, root=None: None)

    servers: dict[str, RuntimeServer] = {}
    handles: list[OwnedSessionHandle] = []
    try:
        for session_id in ids:
            directory = config / "sessions" / session_id
            await seed_transcript(
                directory,
                [
                    user_message(f"{session_id} question"),
                    assistant_message(f"{session_id} saved answer"),
                ],
            )
            stream = ScriptedStream(
                [
                    tool_call_turn(
                        text="Waiting for the subagent to finish.",
                        tool_name=PARKING_TOOL,
                        tool_call_id=f"call-{session_id}",
                        arguments={"job_id": "7a73c97ffc54", "wait_ms": 1800000},
                    ),
                    text_turn("the child finished"),
                ]
            )
            owner = build_session(directory, stream, tools=[tool], cwd=config)
            handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
            handles.append(handle)
            server = RuntimeServer(handle, kind="daemon")
            await server.start_in_process()
            servers[session_id] = server

        def find(_directory: Path, session_id: str) -> tuple[Any, Any]:
            server = servers.get(session_id)
            return (server._record, server._record.pid) if server else (None, None)

        async def never() -> Any:
            raise AssertionError("view navigation must never take execution ownership")

        async def resume(session_id: str | None) -> RemoteSession:
            assert session_id is not None
            return await RemoteSession.connect(
                servers[session_id]._record,
                session_id,
                config_dir=config,
                takeover_factory=never,
                display_window=True,
            )

        monkeypatch.setattr("local_operator.mobile.attach_client.find_owner_record", find)
        # Exposed on the callable so a test can reach the OWNER session (to run
        # its turn) without the fixture returning a second value every caller
        # would have to unpack.
        resume.servers = servers  # type: ignore[attr-defined]
        yield resume
    finally:
        for server in servers.values():
            await server.aclose()
        for handle in handles:
            await handle.dispose()


async def _switch(app: OperatorApp, session_id: str, pilot: Any) -> None:
    """The production sidebar path: select, then await the connection it opens."""
    await asyncio.wait_for(app._sidebar_navigation.select(session_id), DEADLOCK_GUARD_S)
    connection = app._interaction.connection_task
    if connection is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(connection), DEADLOCK_GUARD_S)
    for _ in range(40):
        await pilot.pause()


def _cards(app: OperatorApp) -> list[ToolCard]:
    return [block for block in app._transcript_view().blocks() if isinstance(block, ToolCard)]


@pytest.mark.asyncio
async def test_switching_to_a_session_parked_in_a_tool_paints_a_live_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported frame, end to end, against real owners over real sockets.

    The sequence is the operator's: visit the conversation while it is idle
    (which is what leaves a CONNECTED hidden source — the state that files the
    unanswered call into ``_live_history``), leave it, let its turn park inside
    a long tool, then click its sidebar row.

    Both ends of the call's life are asserted, because "never paints
    ``interrupted``" is satisfiable by a card that is stranded live forever:
    the row must be live during the wait AND settle to its real result when the
    tool returns.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    released = asyncio.Event()
    ids = ["home00", "busy01"]
    try:
        async with live_owners(config, ids, _parking_tool(released), monkeypatch) as resume:
            retirements: list[int] = []
            original_retire = OperatorApp._retire_live_tool_cards

            def counted_retire(self: OperatorApp) -> int:
                retirements.append(len(self._tool_cards) + len(self._composing_cards))
                return original_retire(self)

            monkeypatch.setattr(OperatorApp, "_retire_live_tool_cards", counted_retire)

            app = OperatorApp(lambda: resume("home00"), resume_factory=resume)
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                app._set_sidebar_open(True)
                if app._sidebar_timer is not None:
                    app._sidebar_timer.pause()  # no background prewarm races

                await _switch(app, "busy01", pilot)
                await _switch(app, "home00", pilot)

                owner = resume.servers["busy01"]._handle._session  # type: ignore[attr-defined]
                turn = asyncio.create_task(owner.prompt("await the fix"))
                for _ in range(MAX_PUMP_TURNS):
                    await pilot.pause()
                    state = owner.frontend_state
                    if state.streaming and any(
                        event.get("type") == "tool_execution_start" for event in state.live_events
                    ):
                        break
                else:
                    raise AssertionError("the owner never entered the parking tool")
                for _ in range(40):
                    await pilot.pause()

                # PRECONDITION: the call really is unanswered in the rows the
                # switch is about to replay. Without this the assertion below
                # could pass because there was nothing to get wrong.
                # `cast`, because the protocol the app types its sources and
                # `_session` against does not declare the viewer-only display
                # surface these preconditions read. The object IS a
                # `RemoteSession`, which is exactly why the bug reaches here.
                viewer = cast(Any, app._sidebar_sources["busy01"].session)
                rows = viewer.display_history_window()
                unanswered = {
                    call.id for row in rows for call in (getattr(row, "tool_calls", None) or [])
                } - {
                    str(getattr(row, "tool_call_id", ""))
                    for row in rows
                    if getattr(row, "role", "") == "tool"
                }
                assert unanswered == {"call-busy01"}, (
                    "the hidden viewer's display window does not carry the in-flight call, "
                    f"so this test is not exercising the replay path: {unanswered}"
                )

                retirements.clear()
                await _switch(app, "busy01", pilot)

                cards = _cards(app)
                assert [card.tool_call_id for card in cards] == ["call-busy01"]
                assert cards[0]._state != "interrupted", (
                    "the sidebar switch painted ⊘ interrupted on a tool the owner is still "
                    "executing; the session reports streaming="
                    f"{cast(Any, app._session).frontend_state.streaming}"
                )
                assert cards[0]._state in ("running", "waiting"), cards[0]._state
                # Mechanism B, ruled out rather than assumed: no retirement runs
                # on this path, so a future change that DOES retire here (and
                # would reintroduce the ⊘ by another route) fails loudly instead
                # of silently relying on the repaint above to undo it.
                assert not retirements, f"a retirement ran during the switch: {retirements}"

                # The other end of the call's life: released, the live row must
                # take its real outcome rather than stay a spinner forever.
                released.set()
                for _ in range(MAX_PUMP_TURNS):
                    await pilot.pause()
                    if _cards(app)[0]._state not in ("running", "waiting"):
                        break
                else:
                    raise AssertionError("the live row never settled after the tool returned")
                assert _cards(app)[0]._state == "success", _cards(app)[0]._state
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(turn, DEADLOCK_GUARD_S)
    finally:
        released.set()


@pytest.mark.asyncio
async def test_a_call_from_a_turn_that_really_stopped_still_paints_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The over-fix guard, and the reason the repaint is gated on liveness.

    Same replay path, same unanswered call — but the turn has ENDED. A session
    killed mid-batch leaves exactly this on disk, and ``⊘ interrupted`` is the
    honest reading of it: the tool stopped and never reported. A fix that
    stopped painting ``interrupted`` for an absent result, rather than asking
    whether the call is still executing, passes the test above and silently
    reopens the round-1 Q2 defect this one pins shut.

    Driven through ``_mark_pending_tool_rows`` against a session double whose
    two scans answer for a settled turn, because that is the seam the liveness
    question is asked at; the pilot test above already covers the assembled
    path.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()

        card = ToolCard("dead-call", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
        card.restore(state="interrupted")
        assert card._state == "interrupted"

        class Settled:
            """A session whose turn is over: nothing is pending, nothing runs."""

            def pending_display_tool_ids(self) -> set[str]:
                return set()

            def executing_display_tool_ids(self) -> set[str]:
                return set()

        app._mark_pending_tool_rows([card], Settled())
        assert card._state == "interrupted", (
            "a call whose turn already ended was repainted live; interrupted must remain "
            "the outcome for a tool that genuinely stopped mid-flight"
        )

        # And the positive control on the same seam, so the guard above cannot
        # pass merely because the repaint is broken for everyone.
        class Running(Settled):
            def executing_display_tool_ids(self) -> set[str]:
                return {"dead-call"}

        app._mark_pending_tool_rows([card], Running())
        assert card._state == "running"


@pytest.mark.asyncio
async def test_a_repainted_row_is_owned_and_settles_when_the_turn_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third case: the repaint happens, and THEN liveness ends.

    The pair above covers the two endings the call itself can have — the tool
    returns, or the turn was already over before the switch. Neither covers the
    ending that arrives from outside: the owner dies with the call outstanding,
    *after* the row has been repainted live.

    That case is the exact inverse of the bug this file exists for, and it is
    reachable from ordinary causes rather than only from a killed runtime — a
    dropped socket deliberately leaves ``_streaming`` True, so a switch during
    recovery paints ``running`` correctly, and a recovery that then fails ends
    the turn with the row already painted. The liveness predicate is only ever
    asked at switch time; nothing re-asks it when the answer changes.

    So the row's settle path cannot be the predicate. It is OWNERSHIP: a card
    made live must be in the registry every turn-death path iterates, which is
    what this pins. Asserted at the seam because that is where the invariant
    lives — the pilot test above already proves the assembled path reaches it.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()

        card = ToolCard("live-call", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
        card.restore(state="interrupted")

        class Running:
            """A session whose turn is executing this call right now."""

            def pending_display_tool_ids(self) -> set[str]:
                return set()

            def executing_display_tool_ids(self) -> set[str]:
                return {"live-call"}

        registry: dict[str, ToolCard] = {}
        app._mark_pending_tool_rows([card], Running(), registry)
        assert card._state == "running"
        # The ownership claim itself, stated separately from its consequence:
        # a repaint that stopped registering would fail HERE, naming the cause,
        # rather than only at the stranded-row assertion below.
        assert registry == {"live-call": card}, (
            "the repainted row was not entered into the live registry, so no "
            "turn-death path can reach it"
        )

        # MAJOR-2, on the same row: it must actually WEAR the live styling it
        # claims, and must not still wear the replay's interrupted class — which
        # `mark_done` does not remove, so it would otherwise survive into the
        # settled row.
        assert "tool-running" in card.classes
        assert "tool-interrupted" not in card.classes
        # MINOR-1: a row claiming to execute is not a settled row.
        assert card.settled_rows() == 0

        # And the consequence: the turn dies with the call outstanding. The
        # app's own retirement is the path every turn-death site takes.
        app._tool_cards = registry
        retired = app._retire_live_tool_cards()
        assert retired == 1, "the death path could not see the repainted row"
        assert card._state == "interrupted", (
            "a row repainted live is stranded ⋯ running after its owner died; "
            "the frame shows a call executing beside a notice that says it stopped"
        )
        assert "tool-running" not in card.classes
        assert card.settled_rows() == 1


def test_a_row_adopted_mid_execution_settles_with_the_measured_interval() -> None:
    """The settle leg of a clockless row, against #858's measured interval.

    A row repainted live by ``_mark_pending_tool_rows`` has no ``_started`` —
    deliberately, because the page painted it after the tool began — so it
    cannot time itself. #858 (``77f1cc75f``) made the executor's own
    ``duration_s`` survive into the persisted payload, which means a REPLAY of
    such a call renders ``✓ 4.2s``. Before this, the same call settled LIVE to a
    blank column: one call, two different receipts, decided only by whether the
    viewer happened to be watching.

    Pinned as the agreement between the two paths rather than as a literal, and
    pinned as a FALLBACK: a card that timed its own execution must keep its own
    reading, or this would silently replace every native row's clock with the
    executor's.
    """
    measured = 4.25

    adopted = ToolCard("adopted", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
    adopted.restore(state="interrupted")
    adopted.restore(state="running")
    assert adopted._started is None, "the adopted row must not invent a start time"
    adopted.mark_done("done", None, measured_s=measured)
    assert adopted._duration == measured

    # The replay of the same call, through the path #858 fixed: same receipt.
    replayed = ToolCard("replayed", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
    replayed.restore(state="success", result_text="done", duration_s=measured)
    assert replayed._duration == adopted._duration

    # A row with its OWN clock ignores the fallback — no double stamp.
    native = ToolCard("native", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
    native._started = time.monotonic() - 30.0
    native.mark_done("done", None, measured_s=measured)
    assert native._duration is not None and native._duration > 29.0

    # And a malformed interval off the wire degrades to the blank column
    # instead of printing nonsense, the way every external duration must.
    for bad in (float("nan"), -1.0, "4.2", True, None):
        card = ToolCard(f"bad-{bad!r}", PARKING_TOOL, {})
        card.restore(state="interrupted")
        card.restore(state="running")
        card.mark_done("done", None, measured_s=bad)  # type: ignore[arg-type]
        assert card._duration is None, bad


def test_a_clockless_running_row_reserves_the_settled_spine() -> None:
    """D1/D3: the live row says it is alive, in the cells the outcome will use.

    Two facts, and they are one fact: a replayed live row has no ``_started``
    and must not invent one, so it has no clock — but "no clock" must not
    become "no state", because this is the one row on screen actually consuming
    time and the operator switched to it to ask exactly that. And the label's
    WIDTH is what keeps the row still: the status column competes with the
    summary for the same budget, so a label narrower than the settled spine
    lets the summary grow while running and lose characters the instant the
    tool returns — the text moving under the reader at narrow widths.

    Pinned as an equality against the spine rather than against the literal
    ``"running"``, so a future relabel that breaks the geometry fails here
    instead of on a screenshot nobody re-captures.
    """
    from local_operator.tui.widgets.tool_card import (
        DURATION_COL,
        ICON_SUCCESS,
        RUNNING_LABEL,
        cell_len,
    )

    card = ToolCard("clockless", PARKING_TOOL, {"job_id": "7a73c97ffc54"})
    card.restore(state="running")
    assert card._started is None
    assert [text for text, _style in card._status_runs()] == [RUNNING_LABEL]
    assert cell_len(RUNNING_LABEL) == cell_len(f"{ICON_SUCCESS} ") + DURATION_COL

    # Shed whole below the width that holds it, never truncated and never
    # replaced by an outcome glyph that would claim the tool completed — the
    # same ladder `waiting` uses one arm up.
    assert [text for text, _style in card._status_runs(cap=len(RUNNING_LABEL))] == [RUNNING_LABEL]
    assert [text for text, _style in card._status_runs(cap=len(RUNNING_LABEL) - 1)] == [""]


# --- the resume-onto-a-running-turn duplicate --------------------------------
#
# The tests above cover the SIDEBAR: a conversation parked in a tool, switched
# to, repainted live. The resume variant is one step further on and was a
# separate defect: `/resume` (and `--resume`, and the owner-reconnect gap) run
# the SAME projection against a session whose turn is in flight IN THIS
# PROCESS. The replay then painted a second row for a call the running turn's
# own events were already painting — two rows for one call, a "running N
# tools" count that included the replayed ghost, and an `interrupted` stamp on
# a call that had not stopped. The seam is `replay_tool_call`, so these pin it
# there directly; the assembled path is covered by the frames on the MR.


def _call(call_id: str, name: str = PARKING_TOOL) -> Any:
    """One transcript tool-call record, the shape ``replay_tool_call`` reads."""

    return SimpleNamespace(id=call_id, name=name, arguments={"job_id": "7a73c97ffc54"})


@pytest.mark.asyncio
async def test_a_replayed_live_call_is_not_doubled() -> None:
    """A call live in this process mounts ONE row, and the band counts ONE.

    The replay skips its settled row for an in-flight call so the live path
    owns it; where no live path will paint (a local resume: the turn's
    ToolStarted fired before this process subscribed), the projection's owner
    paints the single row afterwards. Either way the call has exactly one
    visible row and the working line's count reflects the one real call.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        app._session = session

        call = _call("call-live")
        app._projection_live_call_ids = {"call-live"}
        app._projection_skipped_live = []
        app._replay_tool_call(call, {})
        # The replay mounted nothing for the live call, and recorded it for
        # the owner to paint.
        assert app._projection_skipped_live == [call]
        assert not [
            b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)
        ], "the replay mounted a settled row beside the live one"

        # The owner paints the one row, clock withheld, counted once.
        app._paint_skipped_live_tool_rows(
            app._transcript_view(), app._tool_cards, app._projection_skipped_live
        )
        cards = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 1, "a live call must have exactly one visible row"
        assert cards[0]._state == "running"
        assert cards[0]._started is None, "the painted row must not invent a start time"
        assert len(app._tool_cards) == 1, "the working line must count one live tool"


@pytest.mark.asyncio
async def test_a_cold_resume_of_the_same_history_still_marks_interrupted() -> None:
    """Nothing is live: the same unanswered call still renders interrupted.

    The guard must not become "never paint interrupted" — a turn that really
    died mid-flight is exactly the case the settled projection exists to
    report, and an empty executing set is the cold-resume answer.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()

        app._projection_live_call_ids = set()  # cold: no live turn
        app._projection_skipped_live = []
        app._replay_tool_call(_call("call-dead"), {})
        cards = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 1
        assert cards[0]._state == "interrupted", (
            "a call from a turn that genuinely stopped was repainted live; "
            "interrupted must remain its outcome"
        )
        assert app._projection_skipped_live == []


@pytest.mark.asyncio
async def test_a_replayed_running_row_never_starts_its_clock() -> None:
    """A painted live row withholds its clock exactly as restore does.

    The transcript carries no true start time for an in-flight call, so the
    one row the owner paints must refuse to count from when it was painted —
    the same guarantee `restore(state="running")` makes, pinned here on the
    painter the projection path uses. Driven through the app's own view so
    the mount is real.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()

        view = app._transcript_view()
        OperatorApp._paint_skipped_live_tool_rows(view, app._tool_cards, [_call("call-clock")])
        card = app._tool_cards["call-clock"]
        assert card._state == "running"
        assert card._started is None
        # One row per call: painting the same skipped call again is a no-op.
        OperatorApp._paint_skipped_live_tool_rows(view, app._tool_cards, [_call("call-clock")])
        assert len(app._tool_cards) == 1
        assert len([b for b in view.blocks() if isinstance(b, ToolCard)]) == 1


def _history_with_one_call(call_id: str, tool_name: str = PARKING_TOOL) -> list[Message]:
    """A tail whose last assistant message asks for exactly one call.

    The shape every live-projection test needs: the call has no result yet,
    so whether the replay settles it, skips it, or the pending scan re-marks
    it is exactly what each test asserts on.
    """
    return [
        user_message("run the thing"),
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[ToolCall(id=call_id, name=tool_name, arguments={"job_id": "j1"})],
            stop_reason="toolUse",
        ),
    ]


@pytest.mark.asyncio
async def test_a_gate_parked_call_keeps_its_waiting_row_on_a_live_projection() -> None:
    """Round 1 MAJOR-1: the projection seed subtracts the pending set.

    A turn parked at an approval gate is ALSO a streaming one, so both
    live-id accessors answer with exactly the call the gate holds. Seeding
    the skip set with the un-subtracted answer skipped that call out of the
    replay, and the painter then showed it `running` while it waited on the
    user — the documented "waiting wins" rule of `_mark_pending_tool_rows`
    could not fire, because no replayed row existed to re-mark. The seed is
    the gate-free set, so the held call takes the mount → `mark_waiting`
    route that predates this PR.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    class Gated(FakeSession):
        """Both accessors answer with the held call: a gated turn is streaming."""

        def pending_display_tool_ids(self) -> set[str]:
            return {"call-gated"}

        def executing_display_tool_ids(self) -> set[str]:
            return {"call-gated"}

    session = Gated()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        app._session = session
        app._project_settled_rows(_history_with_one_call("call-gated"))
        cards = [b for b in app._transcript_view().blocks() if isinstance(b, ToolCard)]
        assert len(cards) == 1, "one row for the held call, mounted by the replay"
        assert cards[0]._state == "waiting"
        assert "tool-running" not in cards[0].classes
        # A waiting row owns no clock either: nothing started executing.
        assert cards[0]._started is None


@pytest.mark.asyncio
async def test_a_live_waiting_card_is_not_doubled_by_the_gated_projection() -> None:
    """Round 2 Q-R2-1: the fold's mount consults the already-painted registry.

    The MAJOR-1 test above drives a FakeSession whose gate answers WITHOUT any
    live card ever mounted, so the fold's append is the only row and a double
    mount cannot show. On the REAL path — ``/resume`` onto a gated turn, or a
    sidebar switch back to one — the turn's own ``ToolStarted`` mounted a
    clocked card during adoption, and the gate-free seed subtraction correctly
    declines to skip the held call: the fold then appended a SECOND ``waiting``
    row beside the live one, and ``_paint_skipped_live_tool_rows`` had nothing
    to reconcile because the skip list was empty. The one-row rule the live-skip
    arm enforces now covers this arm too: a call id with a card already on
    screen mounts nothing, and the pending scan re-marks the EXISTING row.
    """
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    class Gated(FakeSession):
        """Both accessors answer with the held call: a gated turn is streaming."""

        def pending_display_tool_ids(self) -> set[str]:
            return {"call-gated"}

        def executing_display_tool_ids(self) -> set[str]:
            return {"call-gated"}

    session = Gated()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        app._session = session

        # The live turn mounted its card when ToolStarted fired: clocked and
        # registered, exactly as adoption leaves it when the projection runs.
        live_card = ToolCard("call-gated", PARKING_TOOL, {"job_id": "j1"})
        app._append_block(live_card)
        app._tool_cards["call-gated"] = live_card

        app._project_settled_rows(_history_with_one_call("call-gated"))
        cards = [
            b
            for b in app._transcript_view().blocks()
            if isinstance(b, ToolCard) and b.tool_call_id == "call-gated"
        ]
        assert len(cards) == 1, (
            "the projection mounted a second row beside the live card — "
            "one call has exactly one visible row on the gated path too"
        )
        assert cards[0] is live_card, "the surviving row must be the live one"
        assert (
            cards[0]._state == "waiting"
        ), "the pending scan still owns the state correction: waiting, not running"


@pytest.mark.asyncio
async def test_a_painted_skipped_row_settles_success_through_the_controller() -> None:
    """Round 1 Q-1: the adopt path registers the row it paints.

    The skipped call's ToolStarted predates this process's subscription, so
    the controller never knew the call: the real ToolEnded buffered as an
    orphan and turn-end retirement stamped a genuinely successful call
    `interrupted`. The projection now registers the painted ids with the
    controller — the same seam the presentation-commit path uses — so the
    end pairs with the painted card and the receipt reads success.
    """
    from local_operator.harness.types import ToolExecutionEndEvent, ToolResult
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    class Live(FakeSession):
        def pending_display_tool_ids(self) -> set[str]:
            return set()

        def executing_display_tool_ids(self) -> set[str]:
            return {"call-live"}

    session = Live()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        app._session = session
        app._project_settled_rows(_history_with_one_call("call-live"))
        card = app._tool_cards["call-live"]
        assert card._state == "running"

        # The end takes the route a real session's end takes: through the
        # controller, not straight to the card. Registered at paint time, it
        # pairs here instead of buffering behind a start nobody saw.
        end = ToolExecutionEndEvent(
            tool_call_id="call-live",
            tool_name=PARKING_TOOL,
            result=ToolResult(
                tool_call_id="call-live",
                tool_name=PARKING_TOOL,
                content=[TextContent(text="The job finished.")],
                duration_s=2.4,
            ),
            duration_s=2.4,
        )
        controller = app._controller
        assert controller is not None
        controller._on_event(end)
        await pilot.pause()
        await pilot.pause()

        assert card._state == "success", "a real success must not read interrupted"
        assert card._duration == 2.4, "the executor's measured interval is the receipt"
        # Settled through the ordinary path: the registry released the card,
        # so turn death has nothing left to retire as interrupted.
        assert "call-live" not in app._tool_cards
        assert app._retire_live_tool_cards() == 0


@pytest.mark.asyncio
async def test_a_re_delivered_start_never_rearms_the_withheld_clock() -> None:
    """Round 1 U1/U2: re-entry preserves the withheld clock, settle stays honest.

    `/resume` onto a parked turn, and a switch away and back, both re-deliver
    the still-in-flight ToolStarted to the card the projection painted.
    `begin_running` used to restart `_started` at that instant: `0s` on
    arrival, a clock ticking from the RESUME, and a receipt settled to the
    time since the command. A restored card cannot date itself, so the
    re-entry must not arm a clock; the settle falls back to the executor's
    measured interval — the only number that is about the call.
    """
    from local_operator.harness.types import (
        ToolExecutionEndEvent,
        ToolExecutionStartEvent,
        ToolResult,
    )
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    class Live(FakeSession):
        def pending_display_tool_ids(self) -> set[str]:
            return set()

        def executing_display_tool_ids(self) -> set[str]:
            return {"call-live"}

    session = Live()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        app._session = session
        app._project_settled_rows(_history_with_one_call("call-live"))
        card = app._tool_cards["call-live"]
        assert card._state == "running"
        assert card._started is None

        # The re-delivered start — the exact re-entry `/resume` and a return
        # visit produce when the owner's live seed replays through the
        # controller.
        controller = app._controller
        assert controller is not None
        controller._on_event(
            ToolExecutionStartEvent(
                tool_call_id="call-live",
                tool_name=PARKING_TOOL,
                args={"job_id": "j1"},
            )
        )
        await pilot.pause()
        await pilot.pause()

        assert card._started is None, "re-entry must not date a card that cannot know"
        assert card._elapsed() is None, "no elapsed time exists to paint or settle from"

        end = ToolExecutionEndEvent(
            tool_call_id="call-live",
            tool_name=PARKING_TOOL,
            result=ToolResult(tool_call_id="call-live", tool_name=PARKING_TOOL, duration_s=6.0),
            duration_s=6.0,
        )
        controller._on_event(end)
        await pilot.pause()
        await pilot.pause()

        assert card._state == "success"
        assert card._duration == 6.0, "the measured interval, not time since re-delivery"
