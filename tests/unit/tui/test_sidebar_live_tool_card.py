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
from typing import Any, cast

import pytest

from local_operator.harness.types import AgentTool, TextContent, ToolResult
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
