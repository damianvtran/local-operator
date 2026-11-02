"""A session moves ITSELF only when a real successor boots where it asked.

The cwd-move contract has two processes in it: the runtime that ASKS (its tool
call writes a durable intent and returns ``queued``) and the SUCCESSOR that
ACTS, by booting in the destination. These cells are the real ones — the
production ``process.py`` booted as a child, launched exactly as
``launch._spawn_runtime`` launches it — because the two properties they pin are
invisible to a fake:

* THE COMMIT IS THE SUCCESSOR'S, and it arrives at BOOT. Nothing a viewer, a
  frame or a stop can do commits the move early, and nothing about the request
  itself commits it at all.
* THE CONFIRMATION IS THE SESSION'S DIRECTORY, not this process's. A spawned
  runtime inherits the engaging host's working directory (``launch._spawn_runtime``
  passes no ``cwd=``) and carries the directory it really uses in
  ``LOP_MOBILE_CHILD_CWD``, so a boot reconcile that read ``os.getcwd()`` would
  compare the host's directory, never match, and leave every move ``queued``
  forever. That is what the second assertion in each cell exists to catch — the
  successor's own record names the destination while the intent is still
  ``queued``.

Isolation: ``headless_tui_env`` redirects the config dir and the root conftest
redirects ``HOME``; the child's environment is rebuilt with every ``CMUX_*``/
``LOP_MOBILE_CHILD_*``/``LOP_RUNTIME_*`` removed (a child that inherited a
workspace id could address the operator's live window, #648) and then given the
two names this contract actually uses. ``PYTHONPATH`` points at THIS checkout:
locally the venv's editable install resolves to the primary tree, and a child
that imported that instead would exercise code no PR here changes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime import registry
from tests.e2e.harness import NO_NOTIFY_ENV

pytestmark = pytest.mark.e2e

#: The session this cell moves. Fixed rather than generated: the marker, the
#: transcript and the child's ``--resume`` all key off it, and a readable id is
#: what a failing run's log needs.
SESSION_ID = "ownmove000001"

#: The inherited families a spawned runtime must never see. The same list the
#: cut-off e2e suite strips, for the same reason; ``LOP_BUILD_PREFIX`` is a bare
#: name whose consumers test it with ``startswith``.
_CHILD_ENV_FAMILIES = ("CMUX_", "LOP_MOBILE_CHILD_", "LOP_RUNTIME_", "LOP_BUILD_PREFIX")

#: A real boot takes ~20-25 s on this host under fleet load and the commit lands
#: while ``_serve`` binds, so the bound is generous on purpose: a shorter one
#: would fail for scheduling reasons and read as a broken feature.
BOOT_COMMIT_TIMEOUT_S = 120.0

#: The checkout under test, handed to the child so it imports THIS tree (see the
#: module docstring). ``parents[2]`` is the repo root for ``tests/e2e/<file>``.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_T = TypeVar("_T")


def _seed(config_dir: Path, session_id: str) -> Path:
    """A resumable session on the mock provider, with approvals OFF.

    ``tool_approval_mode: auto`` is the setting this contract is exercised under:
    the approval decision happens at the TOOL call (``move`` is a write-tier op),
    and these cells are about what happens after an intent exists. A gated
    session would park on an approval card nobody is present to answer.
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


def _child_env(config_dir: Path, session_cwd: str) -> dict[str, str]:
    """The environment a cold child gets, with this contract's two names set."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_FAMILIES)}
    env.update(NO_NOTIFY_ENV)
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            # THE DIRECTORY THE SESSION USES — the spawn contract, not the
            # process's cwd. See ``launch._spawn_runtime``: it sets this name and
            # passes no ``cwd=``, so the child inherits the engaging host's
            # directory. Passing the destination here is exactly what a viewer
            # that adopted the retiring frame does.
            "LOP_MOBILE_CHILD_CWD": session_cwd,
            "LOP_MOBILE_CHILD_RESUME": SESSION_ID,
            # Never the quiet exit: these cells kill one child on purpose.
            "LOP_SESSION_GRACE_S": "600",
            "PYTHONPATH": _REPO_ROOT,
        }
    )
    return env


def _spawn_successor(
    config_dir: Path, session_cwd: str, *, inherit_cwd: Path
) -> subprocess.Popen[bytes]:
    """Boot the REAL runtime the way a viewer's re-engage boots it.

    ``cwd=inherit_cwd`` is this cell's stand-in for "the directory the engaging
    host stood in" — deliberately NOT the session's: reproducing the production
    split is the whole point of the assertion that follows.
    """
    log = config_dir / "logs" / f"successor-{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as sink:
        return subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-m", "local_operator.session.runtime.process"],
            env=_child_env(config_dir, session_cwd),
            cwd=str(inherit_cwd),
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _reap(child: subprocess.Popen[bytes]) -> None:
    """Kill the child's whole process group by exact pid and wait for it.

    A real runtime owns a lease and a journal; leaving one running would hold the
    session against the NEXT cell in this file.
    """
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover — a wedged child is a failure
        pytest.fail(f"successor {child.pid} survived SIGKILL")


def _wait_until(predicate: Callable[[], _T | None], *, timeout: float, what: str) -> _T:
    """Poll ``predicate`` until it answers something truthy, or fail by name."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = predicate()
        if found is not None:
            return found
        time.sleep(0.5)
    pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")


def _record_cwd(config_dir: Path, pid: int) -> str | None:
    """The cwd the runtime ``pid`` PUBLISHED, or ``None`` while it has none.

    Keyed by PID, not by session: a killed predecessor leaves its record behind
    until something sweeps it, and a cell that asked "is any record for this
    session in the destination" would read the DEAD one and call the successor
    wrong (measured — the first run of this cell failed exactly that way).
    """
    for record, _state in registry.scan(config_dir):
        if getattr(record, "pid", None) == pid:
            return str(getattr(record, "cwd", "") or "") or None
    return None


def _stage(config_dir: Path) -> str:
    intent = cwd_move.read_move_intent(config_dir, SESSION_ID)
    return intent.stage if intent is not None else "missing"


def _hold(config_dir: Path) -> str:
    """Why the consumer is holding the intent, or ``""`` (agent review B2)."""
    intent = cwd_move.read_move_intent(config_dir, SESSION_ID)
    return intent.blocked_reason if intent is not None else ""


def test_a_real_successor_boot_commits_the_move_and_nothing_else_does(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """The request is not the commit; the SUCCESSOR BOOTING is.

    Ordering is the contract. Asking writes the intent and changes nothing —
    including that the session keeps running where it is — so the first third of
    this cell watches that stage sit at ``queued`` with no successor alive at
    all. The commit then has to arrive from a real boot, and it has to arrive
    with the successor's own record naming the destination: that pair is what
    separates "the session confirmed" from "somebody's cwd happened to match".
    """
    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, host_cwd):
        directory.mkdir()

    intent = cwd_move.enqueue_self_move(
        config_dir,
        SESSION_ID,
        requested_cwd=str(destination),
        effective_cwd=str(source),
        owner_pid=os.getpid(),
    )
    assert intent.stage == cwd_move.STAGE_QUEUED
    # NOTHING ELSE COMMITS IT: three seconds of a real clock with no successor
    # anywhere, and the intent is still exactly where the request left it.
    time.sleep(3)
    assert _stage(config_dir) == cwd_move.STAGE_QUEUED
    assert not any(
        getattr(record, "session_id", "") == SESSION_ID
        for record, _state in registry.scan(config_dir)
    ), "no runtime for this session was started"

    child = _spawn_successor(config_dir, intent.destination_cwd, inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_COMMITTED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor's boot to commit the move",
        )
        assert (
            _record_cwd(config_dir, child.pid) == intent.destination_cwd
        ), "the commit has to be the session that is really in the destination"
        stored = cwd_move.read_move_intent(config_dir, SESSION_ID)
        assert stored is not None
        assert stored.request_id == intent.request_id
        assert stored.source_cwd == intent.source_cwd
        assert stored.destination_cwd == intent.destination_cwd
    finally:
        _reap(child)


def test_a_crash_between_the_request_and_the_boot_is_settled_by_where_it_really_is(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """A lost confirmation is reconciled, and a wrong directory claims nothing.

    The intent outlives the process that wrote it — that is why it is a document
    rather than memory — so the first half kills its writer (SIGKILL, no clean
    exit) and asserts the request survived. The second half is the direction that
    must NOT be inferred: a successor that boots somewhere else leaves the intent
    ``queued``, because "not there yet" is not "failed", and a runtime that
    marked it failed would discard a request the user made.
    """
    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    elsewhere = tmp_path / "elsewhere"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, elsewhere, host_cwd):
        directory.mkdir()

    # 1. THE CRASH: a real process writes the intent and is killed uncleanly.
    writer = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            "-c",
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "from local_operator.session import cwd_move\n"
            "cwd_move.enqueue_self_move(Path(sys.argv[1]), sys.argv[2],"
            " requested_cwd=sys.argv[3], effective_cwd=sys.argv[4], owner_pid=os.getpid())\n"
            "print('written', flush=True)\n"
            "time.sleep(300)\n",
            str(config_dir),
            SESSION_ID,
            str(destination),
            str(source),
        ],
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == b"written"
    finally:
        _reap(writer)
    crashed = cwd_move.read_move_intent(config_dir, SESSION_ID)
    assert crashed is not None, "the request survived the process that made it"
    assert crashed.stage == cwd_move.STAGE_QUEUED

    # 2. A SUCCESSOR IN THE WRONG DIRECTORY claims nothing. It boots elsewhere, which
    #    a QUEUED request reads as "not there yet" rather than "failed" — so the stage
    #    is not ``committed`` because of it. What the runtime DOES do is hand the move
    #    on: with nobody attached to follow, it writes the destination as the session's
    #    recorded truth and stops, so the request is carried by the record instead of
    #    being re-consumed by every later boot (the B2 loop a hold used to cause).
    wrong = _spawn_successor(config_dir, str(elsewhere), inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_HANDOFF or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in a third directory to hand the move on",
        )
        handed = cwd_move.read_move_intent(config_dir, SESSION_ID)
        assert handed is not None
        assert handed.destination_cwd == crashed.destination_cwd, "the request is unchanged"
        assert handed.blocked_reason == "", "a handoff is not a hold"
        _wait_until(
            lambda: True if wrong.poll() is not None else None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the handing-off runtime to stop",
        )
    finally:
        _reap(wrong)

    # 3. THE SUCCESSOR THAT REALLY IS IN THE DESTINATION settles it.
    right = _spawn_successor(config_dir, crashed.destination_cwd, inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_COMMITTED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in the destination to commit the reconciled intent",
        )
        _wait_until(
            lambda: _record_cwd(config_dir, right.pid) == crashed.destination_cwd or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in the destination to publish its record there",
        )
    finally:
        _reap(right)


@pytest.mark.asyncio
async def test_the_real_lop_topology_moves_end_to_end(
    headless_tui_env: Path, tmp_path: Path, operator_cap: bytes
) -> None:
    """The plain-`lop` shape, driven by a REAL tool call inside a REAL turn.

    Regression guard for UX round 2, U9. A `lop` terminal never owns its session —
    ``cli.viewer_factory`` only returns an ``AttachedSession`` — so the topology to
    pin is a runtime over ``ServingSessionHandle`` (which carries ``may_refresh``
    and ``begin_retire``) and a viewer that declares ``cwd_adoption``. The agent
    calls ``sessions(op='move')`` mid-turn; the result must promise the idle point
    (the host CAN act), and when the turn ends the runtime must retire with the
    destination on the frame and the viewer must stage it for its successor's
    spawn. The spawn-and-commit half is the two subprocess cells above.
    """
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.tools.builtin import build_sessions_tool
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        seed_transcript,
        text_turn,
        tool_call_turn,
        user_message,
    )
    from tests.e2e.test_fork_e2e import _never_take_over
    from tests.e2e.watchdog import bounded

    config = headless_tui_env
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    directory = config / "sessions" / SESSION_ID
    await seed_transcript(directory, [user_message("seed")])
    # A REAL context, not a stand-in: the builder's gate is "a launcher exists",
    # and a launcher that is never called keeps the cell about the move alone.
    sessions_tool = build_sessions_tool(
        ToolContext(cwd=str(source), subagent_launcher=lambda label, prompt, **kwargs: "")
    )
    assert sessions_tool is not None
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Moving",
                tool_name="sessions",
                tool_call_id="move-1",
                arguments={"op": "move", "cwd": "../destination"},
            ),
            text_turn("Queued the move"),
        ]
    )
    owner = build_session(directory, stream, tools=[sessions_tool], cwd=source)
    handle = ServingSessionHandle(owner, asyncio.get_running_loop(), cwd=str(source))
    server = RuntimeServer(handle, kind="daemon", operator_cap=operator_cap)
    await server.start_in_process()
    viewer = None
    try:
        with bounded(60, "a real-topology cwd move retires toward the destination"):
            viewer = await AttachedSession.connect(
                server._record,
                owner.session_id,
                config_dir=config,
                takeover_factory=_never_take_over,
            )
            assert server._cwd_move_predicted_hold() == "", "the viewer can follow"
            await owner.prompt("move this session to ../destination")
            for _ in range(200):
                if server._queued_move_stage == "retiring" and viewer.pending_move_cwd:
                    break
                await asyncio.sleep(0.1)
            # The result the MODEL read, from the transcript the session wrote.
            results = [
                str(getattr(block, "text", "") or "")
                for message in owner.history()
                if getattr(message, "tool_call_id", None) == "move-1"
                for block in getattr(message, "content", [])
            ]
            assert results, "the tool ran"
            assert results[0].startswith("queued: "), results[0]
            assert "next idle point" in results[0], "the host can act, so this is true"
            assert server._queued_move_stage == "retiring"
            assert viewer.pending_move_cwd == str(destination.resolve())
            stored = cwd_move.read_move_intent(config, SESSION_ID)
            assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
            assert stored.blocked_reason == "", "never held"
    finally:
        if viewer is not None:
            await viewer.dispose()
        await server.aclose()
        await owner.dispose()


@pytest.mark.asyncio
async def test_an_embedded_hosts_hold_notice_is_written_on_the_apps_loop(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """QA round 3, Q5: the hold announcement never writes from the registrant thread.

    THE REAL EMBEDDED SHAPE, not a mock: an ``OperatorApp`` whose factory returns
    a ``Session`` it OWNS adopts it and builds the production ``TuiSessionHandle``
    plus ``RuntimeServer(kind="tui")`` on the ``lop-mobile-registrant`` thread.
    That handle has no retire latch, so a move holds as ``host_cannot_restart``
    and is announced. The handle publishes no ``session_loop`` either, so the
    runtime's generic hop used to run ``journal_incident`` INLINE on the
    registrant's loop, where it took the transcript's app-loop lock and — 1 run in
    5 — failed the session's own turn persistence ("bound to a different event
    loop"). The spy calls through and records WHERE each call ran; the second turn
    proves the transcript still persists after the announcement.
    """
    import asyncio
    import threading

    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import build_sessions_tool
    from local_operator.tui.app import OperatorApp
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        seed_transcript,
        text_turn,
        user_message,
        wait_for_adoption,
    )
    from tests.e2e.watchdog import bounded

    config = headless_tui_env
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    directory = config / "sessions" / SESSION_ID
    await seed_transcript(directory, [user_message("seed")])
    # The tool is still installed: this cell must exercise the REAL session, and
    # a session without the move op would not be the shape being protected.
    sessions_tool = build_sessions_tool(
        ToolContext(cwd=str(source), subagent_launcher=lambda label, prompt, **kwargs: "")
    )
    assert sessions_tool is not None
    stream = ScriptedStream(
        [
            text_turn("first turn persisted"),
            text_turn("second turn persisted"),
        ]
    )
    session = build_session(directory, stream, tools=[sessions_tool], cwd=source)
    where: list[tuple[str, str, bool]] = []
    app_loop = asyncio.get_running_loop()
    for name in ("queue_notice", "journal_incident"):
        real = getattr(session, name)

        def spy(*args: object, _real: object = real, _name: str = name, **kwargs: object) -> object:
            try:
                same = asyncio.get_running_loop() is app_loop
            except RuntimeError:
                same = False
            where.append((_name, threading.current_thread().name, same))
            return _real(*args, **kwargs)  # type: ignore[operator]

        setattr(session, name, spy)

    async def factory() -> Any:
        return session

    app = OperatorApp(factory)
    with bounded(90, "an embedded host's cwd-move hold is announced on the app loop"):
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            registrant = app._mobile_registrant
            assert registrant is not None, "the owned session must start its registrant"
            assert type(app._mobile_handle).__name__ == "TuiSessionHandle"
            # THE REQUEST IS WRITTEN DIRECTLY, because the TOOL now refuses an
            # incapable host at the CALL (a move a `TuiSessionHandle` can never
            # perform is no longer recorded and held). The regression this cell
            # guards is the thread the ANNOUNCEMENT runs on, and a marker written
            # by any writer reaches that path through the real consumer.
            cwd_move.enqueue_self_move(
                config,
                SESSION_ID,
                requested_cwd=str(destination),
                effective_cwd=str(source),
                owner_pid=os.getpid(),
            )
            from local_operator.session.runtime.server import notify_move_queued

            assert notify_move_queued(SESSION_ID, config) == "host_cannot_restart"
            await session.prompt("move this session to ../destination")
            for _ in range(200):
                if any(name == "journal_incident" for name, _, _ in where):
                    break
                await pilot.pause()
                await asyncio.sleep(0.05)
            stored = cwd_move.read_move_intent(config, SESSION_ID)
            assert stored is not None and stored.blocked_reason == "host_cannot_restart"
            assert {name for name, _, _ in where} == {"queue_notice", "journal_incident"}, where
            off_loop = [entry for entry in where if not entry[2]]
            assert not off_loop, f"a session write ran off the app's loop: {off_loop}"
            # The turn after the announcement still persists end to end.
            await session.prompt("and again")
            text = (directory / "transcript.jsonl").read_text(encoding="utf-8")
            assert "second turn persisted" in text
            assert "the program hosting this session cannot restart it" in text
            await pilot.pause()


def test_a_viewerless_move_hands_off_and_the_next_engage_boots_in_the_destination(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """THE DEFECT THIS CHANGE FIXES, driven end to end with real processes.

    Before: a session with nobody attached could never move itself. The consumer
    refused to retire without a viewer that would adopt the destination
    (``no_capable_viewer``), the request sat on disk, and after thirty minutes it
    lapsed with the cwd unchanged.

    Now the request is carried by the RECORD: the runtime writes the destination,
    stops, and the next engage of any kind starts the session there. This cell
    pins the whole round trip — a real boot consumes the request and hands it on,
    the production engage guard resolves the caller's OLD directory to the new
    one, and a second real boot in that directory commits the move.

    The one seam not exercised here is ``engage_runtime``'s own spawn loop: this
    cell calls the guard it calls (``_engage_cwd_for_recorded_move``) and then
    boots the same child the loop boots. ``tests/unit/session/runtime/
    test_launch_cwd_handoff.py`` drives the loop far enough to assert the
    directory the fork is handed.
    """
    from local_operator.session.runtime.launch import _engage_cwd_for_recorded_move

    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, host_cwd):
        directory.mkdir()

    # THE REQUEST IS WRITTEN BEFORE ANY RUNTIME BOOTS, which is when the boot
    # watcher would see it: a child in another process cannot be woken by the
    # in-process tool path.
    cwd_move.enqueue_self_move(
        config_dir,
        SESSION_ID,
        requested_cwd=str(destination),
        effective_cwd=str(source),
        owner_pid=os.getpid(),
    )
    first = _spawn_successor(config_dir, str(source), inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_HANDOFF or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the viewer-less runtime to record the handoff",
        )
        _wait_until(
            lambda: True if first.poll() is not None else None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the handing-off runtime to stop",
        )
        assert _stage(config_dir) == cwd_move.STAGE_HANDOFF, "recorded, and NOT yet observed"
    finally:
        _reap(first)

    # THE NEXT ENGAGE NAMES THE OLD DIRECTORY — a wake, a message, a shell still
    # standing there — and the guard resolves it to the destination.
    resolved = _engage_cwd_for_recorded_move(config_dir, SESSION_ID, str(source))
    assert resolved == str(destination.resolve()), resolved

    successor = _spawn_successor(config_dir, resolved, inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_COMMITTED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor in the destination to commit the handoff",
        )
        _wait_until(
            lambda: _record_cwd(config_dir, successor.pid) == str(destination.resolve()) or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the successor to publish its record in the destination",
        )
        # AND A LATER BOOT DOES NOT RE-CONSUME ANYTHING: the move is settled, so
        # nothing retires and nothing loops.
        assert successor.poll() is None, "a committed move leaves a serving runtime"
    finally:
        _reap(successor)


def test_a_third_directory_supersedes_a_recorded_handoff_exactly_once(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """AN EXPLICIT CHOICE WINS, and the boot that lands there settles it.

    After a handoff is recorded, a caller naming a directory that is neither the
    record's source nor its destination is not silently redirected: the engage
    guard keeps the caller's directory, and the runtime that boots there
    reconciles the record to ``superseded``. Once is the point — the superseded
    request must not be re-consumed by the next boot (that re-arm is the B2 loop).
    """
    from local_operator.session.runtime.launch import _engage_cwd_for_recorded_move

    config_dir = headless_tui_env
    _seed(config_dir, SESSION_ID)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    third = tmp_path / "third"
    host_cwd = tmp_path / "host"
    for directory in (source, destination, third, host_cwd):
        directory.mkdir()

    cwd_move.enqueue_self_move(
        config_dir,
        SESSION_ID,
        requested_cwd=str(destination),
        effective_cwd=str(source),
        owner_pid=os.getpid(),
    )
    first = _spawn_successor(config_dir, str(source), inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_HANDOFF or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the viewer-less runtime to record the handoff",
        )
    finally:
        _reap(first)

    assert _engage_cwd_for_recorded_move(config_dir, SESSION_ID, str(third)) == str(third)

    other = _spawn_successor(config_dir, str(third), inherit_cwd=host_cwd)
    try:
        _wait_until(
            lambda: _stage(config_dir) == cwd_move.STAGE_SUPERSEDED or None,
            timeout=BOOT_COMMIT_TIMEOUT_S,
            what="the boot in the third directory to supersede the handoff",
        )
        settled = cwd_move.read_move_intent(config_dir, SESSION_ID)
        assert settled is not None and settled.destination_cwd == str(destination.resolve())
        assert other.poll() is None, "a superseded request leaves a serving runtime"
        # NO SECOND RETIRE: the stage stays settled while the runtime keeps serving.
        time.sleep(2.0)
        assert _stage(config_dir) == cwd_move.STAGE_SUPERSEDED
        assert other.poll() is None
    finally:
        _reap(other)


@pytest.mark.asyncio
async def test_cold_peer_send_commits_a_recorded_handoff_in_the_real_runtime(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1: a delivered/acked wake must not secretly cancel the session's move.

    Do not replace engage, publication, or reconcile with a double: those seams
    all passed while the peer producer supplied its implicit HOME as an override.
    The spawn observer only retains the actual child handles for exact-group
    cleanup; arguments and results travel through unchanged.
    """
    from local_operator.mobile.peer_send import deliver_peer_message_outcome
    from local_operator.session.runtime import launch

    config = headless_tui_env
    _seed(config, SESSION_ID)
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        config, SESSION_ID, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    assert cwd_move.accept_handoff(config, SESSION_ID, intent) is not None
    for key, value in NO_NOTIFY_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PYTHONPATH", _REPO_ROOT)
    children: list[subprocess.Popen[bytes]] = []
    real_spawn = launch._spawn_runtime

    def observe(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = real_spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(launch, "_spawn_runtime", observe)
    try:
        outcome = await deliver_peer_message_outcome(
            None,
            session_id=SESSION_ID,
            text="Reply briefly: cold peer handoff regression",
            mode="mailbox",
            wake=True,
            sender={"session_id": "synthetic0001", "name": "handoff regression"},
        )
        assert outcome.state == "delivered" and outcome.wake == "acked", outcome
        assert children, "the target was genuinely cold"
        assert _record_cwd(config, children[0].pid) == str(destination.resolve())
        settled = cwd_move.read_move_intent(config, SESSION_ID)
        assert settled is not None and settled.stage == cwd_move.STAGE_COMMITTED
    finally:
        for child in children:
            _reap(child)
