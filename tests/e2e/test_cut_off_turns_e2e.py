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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import RECOVERY_GIVE_UP_S
from local_operator.session.runtime import registry
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, text_turn
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: Long enough that the kill lands well inside the sleep, short enough that a
#: leaked child cannot outlive the test's cleanup budget by much.
PARK_S = 30


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
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
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
            assert (
                elapsed < RECOVERY_GIVE_UP_S
            ), f"the cut-off waited {elapsed:.1f}s, which is the give-up path"
            assert "cut off" in str(end.error)
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)
