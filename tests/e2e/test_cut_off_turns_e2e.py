"""A turn cut off by anything but a deliberate stop is an ERROR, with a reason.

The operator's report, verbatim: "occasionally, sessions like this will just get
interrupted after running for a while, I think it has something to do with
updates ... results in some sessions being forgotten about if you're running
many at once for long runtimes", and "make sure that errors are properly called
out in all situations so at least we see it in active sessions as errored".

Before this change a runtime killed mid-turn left an ``attention_started`` with
no outcome, and the next boot published ``interrupted`` for it — the same shape
a user's own ``/stop`` produces. These tests reproduce that shape with REAL
processes: the production ``process.py`` is booted in a subprocess, a real turn
is parked in the real ``bash`` tool (the mock provider's ``[bash:N]`` marker),
and the runtime is killed. The successor boot then has to classify it.

Isolation: the ``headless_tui_env`` fixture redirects the config dir and the
root conftest redirects ``HOME``. The child's environment is rebuilt here with
EVERY ``CMUX_*`` removed regardless, because a runtime that inherited a
workspace id could address the operator's live window (#648). No provider key is
needed: everything runs on ``hosting: test`` / ``model_name: mock``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import COLD_FALLBACK_S
from local_operator.session.runtime import registry
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, text_turn
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: Long enough that the kill lands well inside the sleep, short enough that a
#: leaked child cannot outlive the test's cleanup budget by much.
PARK_S = 30

#: The inherited environment families a spawned runtime must never see, named
#: once because two different mechanisms enforce it here: ``_child_env``
#: rebuilds a child's environment, and ``_strip_child_env`` removes them from
#: THIS process so a runtime the code under test spawns itself (with
#: ``dict(os.environ)``) also gets a clean one.
CHILD_ENV_FAMILIES = ("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_")


def _seed(config_dir: Path, session_id: str) -> Path:
    """A session with one durable row, on the mock provider.

    ``tool_approval_mode: auto`` because the parked turn calls the REAL ``bash``
    tool: with the gate armed the runtime would park on an approval prompt
    instead of in the sleep, which is a different (and covered) shape.
    """
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    return directory


def _child_env(config_dir: Path, session_id: str) -> dict[str, str]:
    # THREE FAMILIES, not just the cmux one. ``LOP_RUNTIME_ADOPT_SESSION`` and
    # the ``LOP_MOBILE_CHILD_*`` pair pin a spawned runtime's session and
    # provider, so a suite run from inside a harness that exports them would
    # spawn a child that ADOPTS the operator's own session — the same class of
    # hazard the ``CMUX_*`` strip exists for (#648). The values this cell needs
    # are set explicitly below, so nothing legitimate is lost.
    env = {k: v for k, v in os.environ.items() if not k.startswith(CHILD_ENV_FAMILIES)}
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "LOP_MOBILE_CHILD_CWD": str(config_dir),
            "LOP_MOBILE_CHILD_RESUME": session_id,
            # Never the quiet exit: these tests are about a DEATH mid-turn, and
            # a grace that expired under the test would be a different cause.
            "LOP_SESSION_GRACE_S": "600",
        }
    )
    return env


def _spawn(config_dir: Path, session_id: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=_child_env(config_dir, session_id),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 30.0) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record for {session_id} within {timeout}s")


async def _never_take_over() -> Any:
    raise AssertionError("a viewer never takes over a session")


async def _attach(config: Path, session_id: str, app: Any = None) -> Any:
    """The production viewer for ``session_id`` (no takeover).

    The real client is what makes "the turn is genuinely under way" observable:
    ``prompt`` returns on durable admission and the viewer's facade reports
    ``streaming`` once the owner is running the turn.
    """
    from local_operator.session.attached import AttachedSession

    record = await _wait_for_record(config, session_id)
    return await AttachedSession.connect(
        record, session_id, config_dir=config, takeover_factory=_never_take_over
    )


async def _park_a_turn(viewer: Any, directory: Path, seconds: int = PARK_S) -> None:
    """Prompt a real turn that holds the real ``bash`` tool open.

    Readiness is taken from the VIEWER's streaming flag rather than from the
    transcript: the assistant row carrying the tool call is persisted at the
    message boundary, so a transcript-only probe can miss a turn that is
    already parked. ``streaming`` is set when the owner starts the turn and
    cleared when it ends, which is exactly the window the kill must land in.
    """
    await viewer.prompt(f"please [bash:{seconds}]")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = getattr(viewer, "frontend_state", None)
        if state is not None and getattr(state, "streaming", False):
            # Past the point where the model answered and the tool is running.
            await asyncio.sleep(1.0)
            return
        await asyncio.sleep(0.2)
    raise AssertionError(
        f"the turn never started; transcript:\n"
        f"{(directory / 'transcript.jsonl').read_text(encoding='utf-8')[-2000:]}"
    )


async def _successor_boot(directory: Path) -> Any:
    """Boot a REAL successor session over the same directory — the restore path.

    This is the production classification seam: ``Session.__init__`` calls
    ``bootstrap_transcript`` and ``async_init`` journals the restored cut-off.
    """
    session = build_session(directory, stream=ScriptedStream([text_turn("ok")]))
    await session.async_init()
    return session


def _rendered_history_text(session: Any) -> str:
    """The next turn's model-visible text, through the production converter."""
    from local_operator.session.session import _default_convert_to_llm

    rendered = _default_convert_to_llm(list(session._context.messages))
    parts: list[str] = []
    for message in rendered:
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _incidents(directory: Path) -> list[Any]:
    """Every ``session_incident`` row, however it was written.

    Filtered on ``payload.custom_type`` rather than on the row TYPE: a
    ``CustomMessage`` is persisted as an ordinary ``message`` row carrying
    ``custom_type`` in its payload, while ``append_custom`` writes a ``custom``
    row. Matching the payload is what the product's own ``latest_custom`` does,
    and it is the only filter that sees both writers.
    """
    return [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]


def _row_text(row: Any) -> str:
    """The text of one projection row, whichever shape the frame serialized it.

    ``_projection_frame`` caps and serializes rows through the projection
    layer, so a row arrives as either the ``to_json`` string or the plain dict;
    reading only one of the two makes an assertion silently about the other.
    """
    if isinstance(row, dict):
        return str(row.get("text") or "")
    return str(row)


def _reap(child: subprocess.Popen[bytes], config_dir: Path) -> None:
    try:
        child.kill()
    except ProcessLookupError:
        pass
    child.wait(timeout=10)
    for record, _state in registry.scan(config_dir):
        try:
            os.kill(record.pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_a_turn_cut_off_by_a_killed_runtime_reports_an_error_with_a_reason(
    headless_tui_env: Path,
) -> None:
    """Cell A — the case-study shape, end to end.

    SIGKILL a runtime parked in a real tool, then boot the successor and assert
    every surface the operator asked about: the durable kind is ``error`` with a
    non-empty reason, exactly ONE ``[session incident]`` reaches the transcript,
    the NEXT turn's history carries it, the sidebar says "Unseen error" with the
    cause, and the phone projection's notice names it too.
    """
    from local_operator.mobile.daemon import _durable_projection, _projection_frame
    from local_operator.session.attention import AttentionStore, conversation_identity
    from local_operator.session.catalog import load_catalog

    config = headless_tui_env
    session_id = "cutoffkill01"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(120, "cut-off: killed runtime"):
            viewer = await _attach(config, session_id)
            await _park_a_turn(viewer, directory)
            killed_pid = None
            for candidate, _state in registry.scan(config):
                if candidate.session_id == session_id:
                    killed_pid = int(candidate.pid)
            assert killed_pid == child.pid, (killed_pid, child.pid)

            # SIGKILL: no exception path, no disposal, no marker — the case
            # study's death shape.
            child.kill()
            child.wait(timeout=10)
            await asyncio.sleep(0.2)
            assert "from the mock provider" not in (directory / "transcript.jsonl").read_text(
                encoding="utf-8"
            ), "the turn completed, so nothing was cut off"

            # The successor boot is what classifies and narrates the orphaned run.
            session = await _successor_boot(directory)
            try:
                state = AttentionStore().state(conversation_identity(directory))
                assert state["kind"] == "error", state
                assert state["cause"] == "runtime-killed", state
                assert state["reason"], "a cut-off must name a reason"
                assert state["anchor_id"].startswith("completion-")

                incidents = _incidents(directory)
                assert len(incidents) == 1, f"expected one incident, got {len(incidents)}"
                assert incidents[0].payload["details"]["token"]

                rendered = _rendered_history_text(session)
                assert "[session incident]" in rendered
                assert "cut-off" in rendered

                # A SECOND boot must not narrate the same run again.
                second = await _successor_boot(directory)
                try:
                    assert len(_incidents(directory)) == 1, "re-opening re-narrated the cut-off"
                finally:
                    await second.dispose()

                # Reap the killed owner's record first: the catalog reports a
                # row with a live-looking owner as "Working", which is true of
                # the RECORD and false of the session, and would mask the
                # outcome spelling this assertion is about.
                registry.scan(config)
                rows = {entry.id: entry for entry in load_catalog(config)}
                sidebar = rows[session_id].status
                assert sidebar.startswith("Unseen error"), sidebar
                assert sidebar.split(" — ", 1)[-1] in state["reason"], sidebar

                projection = _durable_projection(session_id)
                assert projection is not None, "the phone projection vanished"
                frame = _projection_frame(projection)
                notice = [
                    text
                    for text in (_row_text(row) for row in frame["transcript"])
                    if "Stopped with an error" in text
                ]
                assert notice, [_row_text(row) for row in frame["transcript"]][-3:]
                assert state["reason"] in notice[0]
                assert f"pid {killed_pid}" in state["reason"] or (
                    "could not be determined" in state["reason"]
                )
            finally:
                await session.dispose()
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)


@pytest.mark.asyncio
async def test_a_deliberate_stop_still_reports_interrupted(
    headless_tui_env: Path,
) -> None:
    """Cell C — the fix must not turn a user's own cancel into an error.

    The taxonomy flips the DEFAULT for a cut-off; a stop that was actually asked
    for has to keep saying so, with the cause recorded, and must NOT journal an
    incident (there is nothing to explain to the model).
    """
    from local_operator.session.attention import AttentionStore, conversation_identity
    from local_operator.session.runtime.control import stop_session

    config = headless_tui_env
    session_id = "cutoffstop01"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(120, "cut-off: deliberate stop"):
            viewer = await _attach(config, session_id)
            # A park the abort can actually drain inside the graceful rung: the
            # deliberate marker is written by the turn's own finally, so a turn
            # that outlives the ladder's timeout would be killed before it could
            # say it was stopped on purpose.
            await _park_a_turn(viewer, directory, seconds=3)
            record = await _wait_for_record(config, session_id)
            # A REAL `lop stop`: the graceful socket rung first, signals only
            # if it does not answer.
            outcome = await stop_session(record, timeout_s=10, _root=config)
            assert outcome.method == "socket", outcome
            child.wait(timeout=15)

            session = await _successor_boot(directory)
            try:
                state = AttentionStore().state(conversation_identity(directory))
                assert state["kind"] == "interrupted", state
                assert state["cause"] == "user-stop", repr(state)
                assert _incidents(directory) == [], "a deliberate stop must not journal an incident"
            finally:
                await session.dispose()
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        if child.poll() is None:
            _reap(child, config)


@pytest.mark.asyncio
async def test_a_provider_refusal_is_reported_exactly_as_before(
    headless_tui_env: Path,
) -> None:
    """The LIVE path is untouched for a provider error.

    The mock's ``[refuse]`` ends the stream as a refusal, which the loop turns
    into an error end event. The classification must not touch it: the reason is
    the provider's own message, and there is no harness cause — a cut-off marker
    overwriting a provider diagnosis would throw away the more specific fault.
    """
    config = headless_tui_env
    session_id = "cutoffrefuse1"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        from local_operator.session.attention import (
            AttentionStore,
            conversation_identity,
        )

        with bounded(120, "cut-off: provider refusal"):
            viewer = await _attach(config, session_id)
            await viewer.prompt("please [refuse]")
            identity = conversation_identity(directory)
            deadline = time.monotonic() + 30
            state: dict[str, Any] = {}
            while time.monotonic() < deadline:
                state = AttentionStore().state(identity)
                if state["kind"]:
                    break
                await asyncio.sleep(0.2)
            assert state["kind"] == "error", state
            assert "can't help" in state["reason"] or "refus" in state["reason"], state
            assert state["cause"] == "", "a provider error is not a harness cut-off"
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        _reap(child, config)


@pytest.mark.asyncio
async def test_the_next_turn_after_a_restore_carries_the_incident(
    headless_tui_env: Path,
) -> None:
    """The model must be told, not only the human.

    A restored cut-off is replayed into the NEXT turn's provider history: the
    successor runs a real turn over the scripted stream and the request it made
    contains the ``[session incident]`` text. Asserted on the request the
    production loop actually sent, not on the transcript.
    """
    config = headless_tui_env
    session_id = "cutoffnext01"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(120, "cut-off: next turn carries the incident"):
            viewer = await _attach(config, session_id)
            await _park_a_turn(viewer, directory)
            child.kill()
            child.wait(timeout=10)

            stream = ScriptedStream([text_turn("done")])
            session = build_session(directory, stream=stream)
            await session.async_init()
            try:
                await session.prompt("what happened?")
                assert stream.requests, "the successor never called the provider"
                sent = "\n".join(
                    message.text
                    for message in stream.requests[-1].messages
                    if getattr(message, "text", "")
                )
                assert "[session incident]" in sent
                assert "cut-off" in sent
                assert "do not assume the request completed" in sent
            finally:
                await session.dispose()
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        if child.poll() is None:
            _reap(child, config)


@pytest.mark.asyncio
async def test_a_tui_owned_stop_through_the_dispose_route_reports_interrupted(
    headless_tui_env: Path,
) -> None:
    """Cell D — the OTHER rung of the kill switch: in-process, no control op.

    Cell C drives ``lop stop`` over the socket, which is the one rung that has
    recorded the deliberate verdict since the taxonomy landed. The route the
    user's own bare ``/stop`` takes in a TUI-owned session is this one — the app
    disposes the session in place — and it published ``kind=error,
    cause=disposed``: "the session was disposed while this turn was running" on
    a cancel the user asked for, on the default path (review round 1,
    BLOCKER-1). Asserted on the durable store, because a live-only notice would
    have hidden it.

    The turn is parked in the REAL ``bash`` tool (an in-process session with a
    scripted tool call), so it is genuinely mid-flight when the stop lands —
    the same shape the socket-rung cell uses, without a spawned runtime.
    """
    from local_operator.session.attention import AttentionStore, conversation_identity
    from local_operator.tools.builtin import build_bash_tool
    from local_operator.tui.app import OperatorApp
    from tests.e2e.harness import tool_call_turn, wait_for_adoption

    config = headless_tui_env
    session_id = "cutoffdispose1"
    directory = _seed(config, session_id)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="holding the turn open",
                tool_name="bash",
                tool_call_id="call-parked",
                arguments={"command": "sleep 30"},
            ),
            text_turn("the stop landed before this"),
        ]
    )
    session = build_session(directory, stream, tools=[build_bash_tool()], cwd=config)

    async def factory() -> Any:
        # The app awaits its factory: production's boots or attaches and is a
        # coroutine, so handing it a session directly fails adoption with
        # "'Session' object can't be awaited" rather than adopting it.
        return session

    app = OperatorApp(factory)
    with bounded(90, "cut-off: dispose-route stop"):
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            # NOT awaited: an in-process session's ``prompt`` returns when the
            # turn ends, so awaiting it here would block until the parked tool
            # had finished and left nothing to stop.
            task = asyncio.create_task(session.prompt("hold this turn open"))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if getattr(session, "is_streaming", False):
                    await asyncio.sleep(0.5)
                    break
                await asyncio.sleep(0.05)
            assert session.is_streaming, "the turn never started"
            assert app._session is session, "the app must own the session for this route"

            await app._stop_local_session()
            await pilot.pause()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=30)

        state = AttentionStore().state(conversation_identity(directory))
        assert state["kind"] == "interrupted", state
        assert state["cause"] == "user-stop", state
        assert _incidents(directory) == [], "a stop the user asked for is not an incident"


@pytest.mark.asyncio
async def test_a_watched_runtime_killed_mid_turn_ends_with_a_named_cut_off(
    headless_tui_env: Path,
) -> None:
    """The operator's reported flow: the session dies while you are WATCHING it.

    The successor-boot cells above classify a death after the fact. This one is
    the screen in front of the user when it happens, and it used to be the worst
    outcome of the change: a viewer built through ``connect()`` has
    ``_can_go_cold`` False, so the recovery loop took its legacy give-up arm and
    synthesised the BARE abort a user's Esc produces — ``aborted=True,
    error=None`` — while the branch that named a cause was unreachable. Measured
    at the time: card ``interrupted ⊘ 11s``, no notice, no reason, durable state
    still ``kind=None`` at t≈98 s (QA round 1, Q-1; UX U1).
    """
    from local_operator.harness.types import AgentEndEvent
    from local_operator.session.attached import COLD_FALLBACK_S

    config = headless_tui_env
    session_id = "cutoffwatched1"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    ends: list[AgentEndEvent] = []
    try:
        with bounded(120, "cut-off: watched kill"):
            viewer = await _attach(config, session_id)
            assert viewer._can_go_cold is False, "this cell is about the legacy arm"
            viewer.subscribe(
                lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
            )
            await _park_a_turn(viewer, directory)
            started = time.monotonic()
            child.kill()
            child.wait(timeout=10)

            deadline = time.monotonic() + COLD_FALLBACK_S + 15
            while time.monotonic() < deadline and not ends:
                await asyncio.sleep(0.05)
            elapsed = time.monotonic() - started
            assert len(ends) == 1, f"expected one synthesised end, got {len(ends)}"
            (end,) = ends
            # The taxonomy shape, not a bare abort: this is what lets every
            # existing surface paint a failure with a reason.
            assert end.aborted is False, end
            assert end.error, "a cut-off must carry the notice"
            assert end.cut_off_cause == "owner-lost", end
            # The END has to arrive on the cold bound, not on some longer one:
            # this assertion used to be against a second, 90 s deadline that
            # bounded only some of the arms (review round 1, U2).
            assert (
                elapsed < COLD_FALLBACK_S + 5
            ), f"the cut-off waited {elapsed:.1f}s, which is not the cold bound"
            assert "cut off" in str(end.error)
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)


# --- UX round 2, U7 and U6: what the operator's flow does AFTER the verdict ----


@pytest.mark.asyncio
async def test_the_message_typed_after_a_watched_cut_off_is_served(
    headless_tui_env: Path,
) -> None:
    """Cell E (UX round 2, U7) — the reported flow has to end usable.

    A watched cut-off used to be terminal for the whole process: `lop`'s viewer
    wires a takeover factory that raises BY CONSTRUCTION, so the no-record arm
    retried it forever, `_recovering` stayed latched, and the next message was
    accepted and never served — measured at 242 s with the band spinning, no
    error, no timeout and no advice. `/reload` is a binary relaunch and changed
    nothing.

    This drives the real flow with real processes: park a turn, SIGKILL the
    runtime, let the named verdict land, then TYPE AGAIN and require the fresh
    runtime the give-up releases the viewer to start to actually answer.
    """
    config = headless_tui_env
    session_id = "cutoffserve1"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(240, "cut-off: the next message is served"):
            viewer = await _attach(config, session_id)
            await _park_a_turn(viewer, directory)
            killed_pid = child.pid
            child.kill()
            child.wait(timeout=10)

            # The named verdict (U1) is the LAST thing the user hears before
            # typing, so wait for it before typing.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and viewer._streaming:
                await asyncio.sleep(0.1)
            assert viewer._streaming is False, "the cut-off verdict never landed"
            # No assertion that recovery is still RUNNING here: the verdict and
            # the give-up are the same pass now, which is the fix. On the broken
            # tree this would still be True and stay True forever.
            typed_at = time.monotonic()
            await asyncio.wait_for(viewer.prompt("are you there? [bash:2]"), timeout=90)
            bound = time.monotonic()

            # SERVED means the turn's own output lands, not merely that
            # `prompt` returned: the defect was a message accepted and never
            # served, and durable admission is exactly what was granted anyway.
            text = ""
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                text = (directory / "transcript.jsonl").read_text(encoding="utf-8")
                if "from the mock provider" in text:
                    break
                await asyncio.sleep(0.2)
            assert "from the mock provider" in text, (
                "the message was accepted and never served; "
                f"recovering={viewer._recovering} streaming={viewer._streaming}"
            )
            # A FRESH runtime, not the killed one: the give-up releases the
            # viewer to engage, and engaging must not resurrect the dead pid.
            owners = [record.pid for record, _ in registry.scan(config)]
            assert owners and killed_pid not in owners, (owners, killed_pid)
            assert viewer._recovering is False, "the facade is still latched"
            # The rebind is what makes the release a repair, and it happens on
            # the same bound the release did — the wait must not be another
            # give-up window's worth of silence (UX round 1, U2).
            assert (
                bound - typed_at < COLD_FALLBACK_S + 5
            ), f"the wait took {bound - typed_at:.1f}s, which is not the cold bound"
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)


@pytest.mark.asyncio
async def test_a_watched_cut_off_reads_as_errored_without_being_opened(
    headless_tui_env: Path,
) -> None:
    """Cell F (UX round 2, U6) — the operator's "see it in active sessions".

    Nothing opens this session and no successor boots: the runtime dies under a
    watching viewer, and the SIDEBAR has to say so. It read ``Working`` at t=2 s
    and then ``Recent`` through t=60 s, because the durable outcome was written
    only by an open (``bootstrap_transcript``) and nothing had run it. The
    viewer that delivered the verdict now journals it, and this reads the row
    through the same catalog the sidebar reads — never through the store
    directly, which would pass even if the row's precedence still hid it.
    """
    from local_operator.session.catalog import load_catalog

    config = headless_tui_env
    session_id = "cutoffsidebar1"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(180, "cut-off: errored in active sessions"):
            viewer = await _attach(config, session_id)
            await _park_a_turn(viewer, directory)
            child.kill()
            child.wait(timeout=10)

            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and viewer._streaming:
                await asyncio.sleep(0.1)
            assert viewer._streaming is False, "the cut-off verdict never landed"

            # The journal lands on a worker thread and the row also has to stop
            # describing the (dead) record, so wait on the ROW, never the clock.
            status = ""
            rows: dict[str, Any] = {}
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                registry.scan(config)  # reaps the killed owner's record
                rows = {entry.id: entry for entry in load_catalog(config)}
                status = rows[session_id].status
                if status.startswith("Unseen error"):
                    break
                await asyncio.sleep(0.2)
            assert status.startswith(
                "Unseen error"
            ), f"a session nobody opened reads {status!r}, not errored"
            # The reason, not only the verdict: the row names the cause.
            assert status.split(" — ", 1)[-1], status
            # ...and the same row is what ACTIVE membership uses, so it is in
            # the section the operator scans rather than buried in Recent.
            assert rows[session_id].active, rows[session_id]
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)


# --- UX round 1, U1: the frame the PHONE renders has to carry the end ----------


#: The daemon's own password for this cell. Only its cookie signature matters:
#: the requests below are built as ASGI scopes and handed straight to the real
#: endpoint functions, so no socket and no tunnel is involved.
PHONE_PASSWORD = "e2e-phone-pw"


def _phone_request(
    path: str,
    *,
    method: str = "GET",
    session_id: str = "",
    body: dict[str, Any] | None = None,
) -> Any:
    """An authenticated ASGI request for one of the daemon's own routes.

    The daemon's SSE and command endpoints are driven through their REAL
    registered endpoints (``build_app``), not through a re-implementation: the
    frame under test is the one ``_projection_frame`` serialises for a socket.
    """
    from starlette.requests import Request

    from local_operator.mobile.auth import COOKIE_NAME, sign_cookie

    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    headers = [
        (b"host", b"fixture"),
        (b"cookie", f"{COOKIE_NAME}={sign_cookie(PHONE_PASSWORD)}".encode()),
        (b"content-length", str(len(payload)).encode()),
    ]
    if body is not None:
        headers.append((b"content-type", b"application/json"))
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "path": path,
        "path_params": {"session_id": session_id} if session_id else {},
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("fixture", 80),
        "client": ("127.0.0.1", 1),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(scope, receive)


def _endpoint(app: Any, path: str) -> Any:
    return next(route.endpoint for route in app.routes if route.path == path)


async def _phone_json(app: Any, path: str, body: dict[str, Any], session_id: str = "") -> Any:
    response = await _endpoint(app, path)(
        _phone_request(
            path.replace("{session_id:str}", session_id),
            method="POST",
            session_id=session_id,
            body=body,
        )
    )
    return json.loads(bytes(response.body))


async def _phone_frames(app: Any, session_id: str) -> Any:
    """Open the phone's own session stream and return its frame iterator."""
    path = "/api/sessions/{session_id:str}/events"
    response = await _endpoint(app, path)(
        _phone_request(f"/api/sessions/{session_id}/events", session_id=session_id)
    )
    return response.body_iterator


async def _frames_until(iterator: Any, predicate: Any, *, timeout: float) -> list[dict[str, Any]]:
    """Every projection frame until ``predicate(frame)`` holds, then stop.

    Frames are the daemon's real SSE payloads (``event: projection``), so this
    reads exactly what the phone's ``EventSource`` would parse.
    """
    seen: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
        except (StopAsyncIteration, TimeoutError):
            break
        for line in str(chunk).splitlines():
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[len("data: ") :])
            seen.append(frame)
            if predicate(frame):
                return seen
    return seen


def _strip_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every inherited family a spawned runtime could read.

    LOAD-BEARING, not hygiene: the daemon spawns with ``dict(os.environ)``, so a
    test process that inherited ``LOP_RUNTIME_ADOPT_SESSION`` or
    ``CMUX_WORKSPACE_ID`` from the harness it runs inside would spawn a child
    that adopts the OPERATOR'S session. ``_child_env`` enforces the same rule for
    the cells that build a child environment by hand; this one is for a child
    the CODE UNDER TEST spawns.
    """
    for name in list(os.environ):
        if name.startswith(CHILD_ENV_FAMILIES):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_the_phone_frame_carries_the_end_the_daemon_saw(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U1: ``stop_reason``/``cut_off`` have to ARRIVE, or D7's word is invisible.

    ``ProjectionFold`` sets those fields only from a folded ``AgentEndEvent``,
    and a runtime that stops mid-turn never emits one — the follower's socket
    just closes. Measured on the real phone path before this: ``stop_reason=''
    cut_off=False`` at every sample to t+40 s for a daemon-owned session killed
    under the phone, while the danger notice and the list mark both arrived. The
    COMPONENT test passed because it hand-fed the payload; ``composer.tsx`` gates
    its whole resume affordance on ``stop_reason === "aborted"``, so on a real
    phone a cut-off (and a deliberate stop issued from it) had no way back in.

    This drives one daemon-owned session twice: killed while the phone watches
    (a cut-off), and then stopped from the phone itself (a deliberate stop). The
    assertion is on the FRAME the phone receives, and on the rule that matters —
    the field arrives without the notice that accompanies it being lost.
    """
    import signal

    from local_operator.mobile.daemon import MobileDaemon, build_app

    _strip_child_env(monkeypatch)
    config = headless_tui_env
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    daemon = MobileDaemon(password=PHONE_PASSWORD)
    app = build_app(daemon)
    scanner = asyncio.create_task(daemon.scan_loop())
    child_pids: list[int] = []

    async def start_session() -> tuple[str, int]:
        reply = await _phone_json(app, "/api/sessions/start", {"cwd": str(Path.home())})
        assert reply.get("ok"), reply
        child_pids.append(int(reply["pid"]))
        return str(reply["session_id"]), int(reply["pid"])

    async def park_a_turn(session_id: str) -> None:
        """A real turn parked in the real ``bash`` tool, prompted AS the phone.

        ``command_id`` is mandatory on the HTTP boundary, so the phone's own
        continuation identity rides the request exactly as the web composer
        sends it.
        """
        import uuid as _uuid

        reply = await _phone_json(
            app,
            "/api/sessions/{session_id:str}/command",
            {
                "op": "prompt",
                "text": f"please [bash:{PARK_S}]",
                "command_id": str(_uuid.uuid4()),
                "images": [],
            },
            session_id=session_id,
        )
        assert reply.get("ok"), reply

    try:
        with bounded(300, "phone: the frame carries the cut-off end"):
            # --- a CUT-OFF, watched from the phone -----------------------
            session_id, pid = await start_session()
            stream = await _phone_frames(app, session_id)
            await park_a_turn(session_id)
            await _frames_until(stream, lambda f: bool(f.get("streaming")), timeout=60)
            started = time.monotonic()
            os.kill(pid, signal.SIGKILL)
            frames = await _frames_until(stream, lambda f: bool(f.get("stop_reason")), timeout=60)
            cut_off_frame = frames[-1] if frames else {}
            assert cut_off_frame.get("stop_reason") == "aborted", (
                "the phone never received the cut-off's end: "
                f"stop_reason={cut_off_frame.get('stop_reason')!r} "
                f"cut_off={cut_off_frame.get('cut_off')!r} "
                f"streaming={cut_off_frame.get('streaming')!r}"
            )
            assert cut_off_frame.get("cut_off") is True, cut_off_frame.get("cut_off")
            assert cut_off_frame.get("attention", {}).get("kind") == "error", cut_off_frame
            cut_off_elapsed = time.monotonic() - started
            # The notice and the button come from ONE record, so a frame that
            # carries the end must carry the sentence it words.
            #
            # WHICH SENTENCE MOVED, and why this is not a relaxation. Until
            # round 2 this shape landed the no-evidence arm — the daemon's
            # discovery classified only AFTER its own sweep had deleted the dead
            # record — so the row read "the turn was cut off and the cause could
            # not be determined", and "cut off" was what this cell could look
            # for (reviewer MINOR-1). With the record handed to the
            # classification the same death NAMES the runtime it found dead,
            # pid included, which is the stronger claim: the row is asserted
            # against the cause token and the killed pid rather than against a
            # wording that was an artefact of the loss.
            notices = " ".join(
                str(row.get("text") or "") for row in cut_off_frame.get("transcript") or []
            )
            assert "Stopped with an error" in notices, cut_off_frame.get("transcript")
            assert f"pid {pid}" in notices, notices
            assert (
                cut_off_frame.get("attention", {}).get("cause") == "runtime-killed"
            ), cut_off_frame.get("attention")

            # --- a DELIBERATE STOP, issued from the phone ----------------
            session_id, pid = await start_session()
            stream = await _phone_frames(app, session_id)
            await park_a_turn(session_id)
            await _frames_until(stream, lambda f: bool(f.get("streaming")), timeout=60)
            stop_reply = await _phone_json(
                app,
                "/api/sessions/{session_id:str}/command",
                {"op": "stop"},
                session_id=session_id,
            )
            assert stop_reply.get("ok"), stop_reply
            frames = await _frames_until(stream, lambda f: bool(f.get("stop_reason")), timeout=60)
            stop_frame = frames[-1] if frames else {}
            assert stop_frame.get("stop_reason") == "aborted", (
                "the same blindness hit a deliberate stop: "
                f"stop_reason={stop_frame.get('stop_reason')!r}"
            )
            # ...and the two acts stay distinguishable in the field the WORD
            # reads, so the button cannot call a stop a cut-off.
            assert stop_frame.get("cut_off") is False, stop_frame.get("cut_off")
            assert stop_frame.get("attention", {}).get("kind") == "interrupted", stop_frame
            # Printed rather than asserted: how long the end takes to ARRIVE is
            # a measurement for the reviewer, not a contract. The bind this cell
            # cares about is that the field arrives at all, and that the two acts
            # stay distinguishable in it.
            print(
                f"[u1] cut-off frame in {cut_off_elapsed:.1f}s; "
                f"deliberate-stop frame in {time.monotonic() - started - cut_off_elapsed:.1f}s"
            )
    finally:
        scanner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scanner
        await daemon.close_phone_views()
        for task in daemon._dial_tasks.values():
            task.cancel()
        await asyncio.gather(*daemon._dial_tasks.values(), return_exceptions=True)
        for child_pid in child_pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
