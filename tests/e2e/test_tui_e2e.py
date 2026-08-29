"""End-to-end: does the real TUI actually work, start to finish?

Why this stage exists
---------------------

PR #401 fixed a total event-loop freeze. ``_oauth_refresh_lock`` took an
unbounded blocking ``fcntl.flock`` on a worker thread and then, during
cancellation, called ``os.close(fd)`` from the event-loop thread — and on
macOS/BSD ``close()`` blocks while a sibling thread is parked in ``flock()`` on
that descriptor. Typing ``/resume`` disposes the MCP manager, which cancels
in-flight connects, which is exactly that sequence. The banner and composer
slid to the bottom of the screen and the app stopped responding to anything,
forever.

**The whole unit suite — 8000+ tests — was green through all of it.** Every
piece was individually correct; nothing drove the assembled application, so
nothing noticed that it had stopped moving. That gap is what this file closes,
and it is the reason the assertions here are about LIVENESS rather than about
values: a frozen app has perfectly correct state, it just never paints again.

What is real and what is scripted
---------------------------------

The provider is scripted; everything else is production code. ``OperatorApp``,
``Session``, the agent loop, the real ``write`` tool, ``Transcript`` on real
files, ``McpManager``, and — critically — the real ``flock``-based OAuth
refresh lock. The model is the one thing replaced, for a deliberate reason:

* **The regression is not model-shaped.** A frozen event loop and a missing
  file are caught identically by a scripted turn and a live one, and the
  scripted version catches them in three seconds with no API key.
* **It has to run on fork PRs.** ``cli-sanity`` and ``server-sanity`` are gated
  behind ``head.repo.full_name == github.repository`` because fork PRs get no
  secrets. A live-model e2e stage would inherit that gate, and the ``/resume``
  liveness assertion is precisely the check that must never be skipped — a
  contributor's PR reintroducing the deadlock has to go red on their own PR.
* **A hang test cannot be flaky.** This stage's failure signal is "something
  did not finish in time". Putting a live network call inside that bound would
  make provider latency indistinguishable from the deadlock it exists to
  catch, and a hang test that cries wolf is one people start ignoring.

The live-model path is not lost: ``cli-sanity`` already drives a real
OpenRouter turn through the real agent against a real file write on every
non-fork PR. This stage adds the assembled-application coverage that job
cannot give, and the two are complementary rather than redundant.

Bounding
--------

Every step that can hang runs inside :func:`tests.e2e.watchdog.bounded`. Read
that module before changing a timeout: the failure mode here defeats
``asyncio.wait_for``, thread watchdogs and signal-based timeouts alike, and the
C-level ``faulthandler`` timer is the only bound that survives it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.session.session import Session
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from tests.e2e import mcp_lock
from tests.e2e.harness import (
    LoopLiveness,
    ScriptedStream,
    assistant_message,
    build_session,
    dispose_quietly,
    drain,
    seed_transcript,
    text_turn,
    tool_call_turn,
    transcript_text,
    user_message,
    wait_for_adoption,
)
from tests.e2e.watchdog import bounded

#: Terminal geometry for every test here. Wide enough that the assertions are
#: about content rather than about truncation, and identical across tests so a
#: failure is never a layout difference between them.
SCREEN = (100, 30)

#: Bounds, in wall-clock seconds. Measured healthy costs on a dev machine are
#: 1.7 s (boot), 2.5 s (turn) and 6.6 s (resume, of which 4 s is the deliberate
#: liveness watch below), so these carry roughly an order of magnitude of
#: headroom. That ratio is the point: these exist to catch a HANG — an infinite
#: wait — not to police performance on a loaded shared runner. A tight bound
#: here would produce flaky reds, and a flaky hang-detector is one people learn
#: to rerun, which is exactly how a real freeze gets waved through.
#:
#: Verified against the pre-fix code: the deadlock trips RESUME_BOUND_S and the
#: watchdog dumps both parked threads, so the red arrives in about a minute
#: rather than at the job's own timeout with no diagnostic.
BOOT_BOUND_S = 45.0
TURN_BOUND_S = 45.0
RESUME_BOUND_S = 60.0

#: How long the loop is watched after ``/resume``, and what counts as alive.
#: The pre-fix code scores ZERO resumptions here (the process is deadlocked in
#: a syscall and the watchdog fires instead), so the floor only has to exclude
#: "the loop came back a handful of times and stopped". At 20 ms per pump, 4 s
#: of a healthy loop yields ~180 resumptions; 40 is a floor a loaded runner
#: cannot dip below while still being unmistakably alive.
LIVENESS_WATCH_S = 4.0
LIVENESS_MIN_RESUMPTIONS = 40
LIVENESS_MAX_GAP_S = 1.5


@pytest.mark.asyncio
async def test_the_app_boots_paints_a_frame_and_becomes_interactive(
    headless_tui_env: Path,
) -> None:
    """Startup: the app reaches an interactive state and has painted something.

    The floor the rest of the file stands on. "Interactive" is asserted as the
    composer holding focus and accepting typed text that lands in its buffer —
    a mounted widget tree alone proves only that ``compose`` ran, and the boot
    failure this guards against (a session factory that never returns) leaves a
    perfectly composed screen that answers no keys.
    """
    from local_operator.tui.widgets.editor import Editor

    session = build_session(
        headless_tui_env / "sessions" / "boot",
        ScriptedStream([]),
        cwd=headless_tui_env,
    )

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    try:
        with bounded(BOOT_BOUND_S, "TUI boot to an interactive frame"):
            async with app.run_test(size=SCREEN) as pilot:
                await wait_for_adoption(app, pilot)
                await drain(pilot)

                # It painted: the compositor has real rows, not an empty buffer.
                strips = app.screen._compositor.render_strips()
                assert strips, "the app mounted but painted no frame at all"
                painted = "\n".join(strip.text for strip in strips)
                assert painted.strip(), "the app painted a frame with nothing on it"

                # It is interactive: keystrokes reach the focused composer.
                editor = app.query_one(Editor)
                assert editor.has_focus, "the composer never took focus, so nothing can be typed"
                await pilot.press(*"hello")
                await drain(pilot, cycles=5)
                assert "hello" in editor.text, (
                    "typed keys never reached the composer buffer: the app painted "
                    "a frame but is not servicing input"
                )
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_turn_writes_a_real_file_and_leaves_its_trace_on_screen(
    headless_tui_env: Path,
    workspace: Path,
) -> None:
    """A driven turn: real tool, real file on disk, real card, real transcript.

    Four artifacts, deliberately, because each one can fail while the others
    look fine: the file proves the tool RAN, the card proves the UI was told,
    the transcript records prove it was PERSISTED (so a resume can replay it),
    and the assistant prose proves the streamed text reached the screen. A test
    asserting only the first would pass on an app that executed the turn
    invisibly.
    """
    from local_operator.tools.builtin import build_write_tool

    target = workspace / "e2e-artifact.txt"
    content = "written by the tui e2e stage"

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Writing that file now.",
                tool_name="write",
                tool_call_id="e2e-write-1",
                arguments={"path": str(target), "content": content},
            ),
            text_turn("Done: the file is on disk."),
        ]
    )
    session_dir = headless_tui_env / "sessions" / "turn"
    session = build_session(session_dir, stream, tools=[build_write_tool()], cwd=workspace)

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    try:
        with bounded(TURN_BOUND_S, "a full turn through the write tool"):
            async with app.run_test(size=SCREEN) as pilot:
                await wait_for_adoption(app, pilot)
                await drain(pilot)

                await session.prompt(f"write {content!r} to {target}")
                await drain(pilot, cycles=40)

                # 1. The durable artifact: the tool actually wrote the file.
                assert target.is_file(), (
                    f"the write tool left no file at {target}: "
                    f"workspace holds {sorted(p.name for p in workspace.iterdir())}"
                )
                assert target.read_text(encoding="utf-8") == content

                # 2. The trace on screen: a settled card for THAT call.
                cards = list(app.query(ToolCard))
                assert cards, (
                    "the turn executed a tool but no tool card reached the transcript; "
                    f"transcript reads: {transcript_text(app)!r}"
                )
                assert any(card.tool_name == "write" for card in cards), (
                    "a tool card was mounted but none of them is the write call: "
                    f"{[card.tool_name for card in cards]}"
                )

                # 3. The streamed prose reached the screen too, not just the card.
                on_screen = transcript_text(app)
                assert (
                    "Writing that file now." in on_screen
                ), f"the assistant's streamed text never painted; transcript: {on_screen!r}"

                # 4. Persistence: the call AND its result are on disk, which is
                # what a later resume replays. A turn that renders but does not
                # persist looks identical until you resume it.
                records = _transcript_records(session_dir)
                assert _has_tool_call(records, "write"), (
                    "the write call was never journalled to the transcript, so a "
                    "resume of this session would replay a turn with no tool in it"
                )
                assert _has_tool_result(records, "write"), (
                    "the write call was journalled without its result, which is the "
                    "shape a session killed mid-turn leaves behind"
                )
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_resume_replays_the_transcript_and_the_loop_keeps_painting(
    headless_tui_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test for #401: ``/resume`` must not freeze the app.

    The setup reproduces the reported conditions rather than approximating
    them. The live session owns a real ``McpManager`` with a real OAuth connect
    parked inside the real ``flock``-based refresh lock, held by a foreign
    process — the ordinary state of a TUI whose MCP servers are still coming up
    on a machine running several sessions. ``/resume`` then disposes that
    manager, cancelling the parked connect, which is the exact sequence that
    deadlocked: worker thread inside ``flock()``, event-loop thread inside
    ``os.close()`` on the same descriptor.

    Two assertions, and the ORDER of importance is the reverse of the obvious
    one:

    1. The prior transcript is replayed onto the screen — the resume did its
       job.
    2. **The loop keeps scheduling afterward.** This is the assertion that
       catches #401. A frozen app can satisfy (1) perfectly: the replay is
       synchronous and happens before the freeze, so the last frame painted
       shows the whole restored conversation. Asserting "the transcript
       appeared" alone passes against the broken code. What distinguishes a
       working app from a wedged one is only visible over TIME.

    On the pre-fix code this test does not fail by assertion — the process is
    deadlocked in a syscall, so no Python runs to raise. It fails by watchdog,
    which dumps every thread's stack and exits. See :mod:`tests.e2e.watchdog`.
    """
    from local_operator.session_factory import attach_mcp_dispose

    prior_user = "PRIOR-TURN-QUESTION"
    prior_reply = "PRIOR-TURN-ANSWER"

    resume_dir = headless_tui_env / "sessions" / "resume-target"
    await seed_transcript(
        resume_dir,
        [user_message(prior_user), assistant_message(prior_reply)],
    )

    live_dir = headless_tui_env / "sessions" / "live"
    live_session = build_session(live_dir, ScriptedStream([]), cwd=headless_tui_env)
    resumed_session = build_session(resume_dir, ScriptedStream([]), cwd=headless_tui_env)

    # The real manager, with a real connect parked in the real lock. Attached
    # through the production helper so the dispose path under test is the one
    # the factory actually wires, not a hand-rolled equivalent.
    with mcp_lock.foreign_lock_holder(headless_tui_env):
        manager = await mcp_lock.parked_mcp_manager(headless_tui_env, headless_tui_env, monkeypatch)
        attach_mcp_dispose(live_session, manager)

        async def factory() -> Session:
            return live_session

        async def resume_factory(_resume_id: str | None) -> Session:
            return resumed_session

        app = OperatorApp(factory, resume_factory=resume_factory)
        try:
            async with app.run_test(size=SCREEN) as pilot:
                with bounded(BOOT_BOUND_S, "boot before /resume"):
                    await wait_for_adoption(app, pilot)
                    await drain(pilot)

                # The connect really is parked in the lock: the manager deferred
                # it past the startup gate and holds a live continuation for it.
                # Without this the test could pass while exercising nothing.
                assert manager._pending_continuations, (
                    "no MCP connect is pending, so /resume has nothing to cancel "
                    "and this test would not exercise the deadlock at all"
                )

                with bounded(RESUME_BOUND_S, "/resume and the liveness watch after it"):
                    # The real command path, the same one the picker calls.
                    app._resume_session("resume-target", lambda *_a, **_k: None)
                    await drain(pilot, cycles=40)

                    # (1) The resume did its job: the prior conversation is on
                    # the screen, not merely in the session's memory.
                    on_screen = transcript_text(app)
                    assert prior_user in on_screen and prior_reply in on_screen, (
                        "/resume did not replay the prior transcript onto the screen; "
                        f"transcript reads: {on_screen!r}"
                    )

                    # (2) The assertion that catches #401. Everything above is
                    # also true of a frozen app.
                    liveness = LoopLiveness()
                    await liveness.observe(pilot, LIVENESS_WATCH_S)
                    liveness.assert_alive(
                        minimum=LIVENESS_MIN_RESUMPTIONS,
                        ceiling_s=LIVENESS_MAX_GAP_S,
                        context="/resume disposed an MCP manager with a connect in the OAuth lock",
                    )

                    # Still interactive, not merely still ticking: a loop can
                    # schedule timers while input handling is wedged behind a
                    # blocked worker.
                    from local_operator.tui.widgets.editor import Editor

                    editor = app.query_one(Editor)
                    editor.focus()
                    await pilot.press(*"post")
                    await drain(pilot, cycles=5)
                    assert "post" in editor.text, (
                        "the loop kept ticking but keystrokes no longer reach the "
                        "composer: the app is live but not usable"
                    )
        finally:
            await dispose_quietly(live_session, resumed_session)


# --- transcript inspection ---------------------------------------------------
#
# Read back from the JSONL on disk rather than from the session's in-memory
# history: "would a resume of this session see the turn" is a question about
# the FILE, and an in-memory assertion answers a different one.


def _transcript_records(directory: Path) -> list[dict[str, Any]]:
    import json

    path = directory / "transcript.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:  # a partially-flushed final line is not a record
            continue
    return records


def _payloads(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [rec.get("payload") or {} for rec in records]


def _has_tool_call(records: list[dict[str, Any]], tool_name: str) -> bool:
    """True when an assistant message journalled a call to ``tool_name``."""
    for payload in _payloads(records):
        for call in payload.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name") == tool_name:
                return True
    return False


def _has_tool_result(records: list[dict[str, Any]], tool_name: str) -> bool:
    """True when a result for ``tool_name`` was journalled beside its call."""
    for payload in _payloads(records):
        for result in payload.get("tool_results") or []:
            if isinstance(result, dict) and result.get("tool_name") == tool_name:
                return True
        if payload.get("tool_name") == tool_name and payload.get("role") == "tool":
            return True
    return False
