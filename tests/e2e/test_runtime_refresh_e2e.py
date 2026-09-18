"""A stale idle runtime refreshes itself; the viewer never sees a chore.

The operator's requirement, verbatim: "runtimes that are inactive
automatically refresh and update / bring down the runtime on inactive
sessions so that resuming would do the update. On resume we should never see
the [/stop, then send again] message. The user should never need to run
/stop to refresh or update a runtime."

These tests boot the PRODUCTION ``process.py`` in a subprocess against a
fake install prefix (``LOP_BUILD_PREFIX`` → a temp dir carrying a
``.lop-source`` marker), attach the production ``AttachedSession`` under the
real ``OperatorApp``, and then FLIP the marker the way ``lop-update`` does.
Asserted on the things a user would notice: the old pid is gone, the viewer
is bound to a NEW pid, and nothing on screen says ``/stop``, ``interrupted``
or ``stopped``.

Isolation: the ``headless_tui_env`` fixture redirects the config dir; the
root conftest redirects ``HOME`` and scrubs the cmux/herdr pane variables.
The child's environment is rebuilt here with EVERY ``CMUX_*`` removed
regardless, because a runtime that inherited a workspace id could address
the operator's live window (#648). Timings that are constants in production
(settle, stagger) are shortened through the test-only env overrides
``process.py`` reads.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime import registry
from local_operator.session.runtime.inbox import SPOOL_RECEIPT_WAKE
from local_operator.tui.app import OperatorApp
from local_operator.update import BuildStamp
from tests.e2e.harness import NO_NOTIFY_ENV, transcript_text, wait_for_adoption
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: The words the operator must never read after a refresh. ``stopped`` covers
#: the parked "this session was stopped" screen a mis-read ``stopping`` frame
#: would park the viewer in; ``interrupted`` the synthesised abort a viewer
#: paints for owner death; ``/stop`` the chore itself.
FORBIDDEN = ("/stop", "interrupted", "stopped")

OLD_MARKER = "46a4e9b1234567890abcdef v0.49.8\n"
NEW_MARKER = "f4a70b991234567890abcdef v0.49.9\n"


def _seed(config_dir: Path, session_id: str) -> None:
    """A session with one durable row, on the mock provider."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )


def _child_env(config_dir: Path, prefix: Path, session_id: str, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    # Re-asserted after the strip so a runtime child in these cells cannot
    # announce: the strip removes the pane families only, and every one of
    # these children either runs the mock hosting or settles a real turn.
    env.update(NO_NOTIFY_ENV)
    env.update(
        {
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "LOP_MOBILE_CHILD_CWD": str(config_dir),
            "LOP_MOBILE_CHILD_RESUME": session_id,
            "LOP_BUILD_PREFIX": str(prefix),
            # Fast enough to observe inside the watchdog, slow enough that a
            # marker written mid-test is not acted on before it is whole.
            "LOP_BUILD_SETTLE_S": "0.5",
            "LOP_BUILD_STAGGER_S": "0.5",
            # A long grace so the QUIET exit can never be the thing that
            # retires the runtime in these tests; only the refresh may.
            "LOP_SESSION_GRACE_S": "120",
        }
    )
    env.update(extra)
    return env


def _spawn(
    config_dir: Path, prefix: Path, session_id: str, **extra: str
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=_child_env(config_dir, prefix, session_id, **extra),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _record_for(config_dir: Path, session_id: str) -> Any:
    for record, _state in registry.scan(config_dir):
        if getattr(record, "session_id", "") == session_id:
            return record
    return None


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 30.0) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        record = _record_for(config_dir, session_id)
        if record is not None:
            return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record for {session_id} within {timeout}s")


def _alive(child: subprocess.Popen[bytes]) -> bool:
    """Whether OUR child is still running.

    ``os.kill(pid, 0)`` is the wrong probe for a process this test spawned:
    an exited child stays a ZOMBIE — kill(0) succeeds — until it is reaped,
    so the runtime looked resident for the whole 30 s budget after it had
    already retired. ``poll()`` reaps and answers.
    """
    return child.poll() is None


async def _never_take_over() -> Any:
    raise AssertionError("a viewer never takes over a session")


def _flip(prefix: Path) -> None:
    """What ``lop-update`` does last: rewrite the marker to the new build."""
    (prefix / ".lop-source").write_text(NEW_MARKER, encoding="utf-8")


def _stale_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every runtime the VIEWER spawns (``engage_runtime`` → ``sys.executable
    -m process``) inherits this process's environment, so the successor must
    see the same fake prefix and stay on it — otherwise it would boot on the
    real install's stamp, and the viewer would paint owner skew against a
    build this test does not control."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_a_watched_idle_runtime_refreshes_and_the_viewer_rebinds_silently(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design test 13: the resume-then-update case, end to end.

    Boot a runtime on OLD, attach a real viewer under the real app, flip the
    marker to NEW. Within the settle + stagger the old pid must be gone, the
    record must name a NEW pid, the viewer must be bound (not cold), and the
    ledger must contain none of :data:`FORBIDDEN`.
    """
    from local_operator.session.attached import AttachedSession

    _stale_child_env(monkeypatch)
    config = headless_tui_env
    session_id = "refreshsess1"
    _seed(config, session_id)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / ".lop-source").write_text(OLD_MARKER, encoding="utf-8")
    # The viewer's spawn path and its own skew check read the same prefix.
    monkeypatch.setenv("LOP_BUILD_PREFIX", str(prefix))
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.5")
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "120")

    child = _spawn(config, prefix, session_id)
    viewer = None
    app = None
    try:
        record = await _wait_for_record(config, session_id)
        old_pid = int(record.pid)
        assert old_pid == child.pid
        assert record.source_ref == OLD_MARKER.split()[0]

        viewer = await AttachedSession.connect(
            record, session_id, config_dir=config, takeover_factory=_never_take_over
        )
        assert not viewer.is_cold

        async def factory() -> Any:
            return viewer

        app = OperatorApp(factory)
        with bounded(90, "runtime refresh: watched idle runtime"):
            async with app.run_test(size=(100, 30)) as pilot:
                await wait_for_adoption(app, pilot)
                # A matching stamp on both sides: no owner-skew notice at
                # adopt. (The window's own stamp is the real install's, which
                # this test does not control, so the check is on the copy.)
                app._loaded_build = BuildStamp(version=record.version, source_ref=record.source_ref)
                await pilot.pause()

                _flip(prefix)

                # Wait on the EVENT — the old pid exiting — never on a clock.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and _alive(child):
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert not _alive(child), "the stale idle runtime never retired"
                assert child.returncode == 0, f"the runtime exited {child.returncode}"

                # The viewer re-engages eagerly: a new record, a new pid, bound.
                deadline = time.monotonic() + 30
                new_record = None
                while time.monotonic() < deadline:
                    new_record = _record_for(config, session_id)
                    if (
                        new_record is not None
                        and int(new_record.pid) != old_pid
                        and not viewer.is_cold
                    ):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert (
                    new_record is not None and int(new_record.pid) != old_pid
                ), "no successor runtime was engaged after the refresh"
                assert not viewer.is_cold, "the viewer must be bound to the successor"
                assert (
                    new_record.source_ref == NEW_MARKER.split()[0]
                ), "the successor must run the NEW build"
                await pilot.pause()
                text = transcript_text(app)
                for word in FORBIDDEN:
                    assert word not in text, f"{word!r} reached the ledger:\n{text}"
                successor_pid = int(new_record.pid)
        # Leaving the app disposes the viewer; the successor is retired by
        # the offer-back or the drain. Kill it outright below either way.
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)
    assert successor_pid != old_pid


@pytest.mark.asyncio
async def test_a_busy_runtime_waits_and_refreshes_when_its_turn_ends(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design test 14: never mid-turn.

    The mock provider's ``[bash:N]`` marker calls the REAL ``bash`` tool
    with ``sleep N`` — a runtime that is genuinely busy (a tool slot held, the
    turn lock taken) for a known duration, on a config that auto-approves so
    the gate never parks it. The refresh must not fire while that sleeps;
    once the turn ends the old pid retires and a successor is engaged — with
    none of :data:`FORBIDDEN` on screen at any point, and the turn's reply
    persisted (the update did not cost the work).
    """
    from local_operator.session.attached import AttachedSession

    _stale_child_env(monkeypatch)
    config = headless_tui_env
    session_id = "refreshbusy1"
    _seed(config, session_id)
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / ".lop-source").write_text(OLD_MARKER, encoding="utf-8")
    monkeypatch.setenv("LOP_BUILD_PREFIX", str(prefix))
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.5")
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "120")

    child = _spawn(config, prefix, session_id)
    viewer = None
    try:
        record = await _wait_for_record(config, session_id)
        old_pid = int(record.pid)
        viewer = await AttachedSession.connect(
            record, session_id, config_dir=config, takeover_factory=_never_take_over
        )

        async def factory() -> Any:
            return viewer

        app = OperatorApp(factory)
        with bounded(120, "runtime refresh: busy runtime"):
            async with app.run_test(size=(100, 30)) as pilot:
                await wait_for_adoption(app, pilot)
                app._loaded_build = BuildStamp(version=record.version, source_ref=record.source_ref)
                await pilot.pause()
                # A turn that holds the bash tool for ~6 s: BUSY by every
                # measure (turn lock, streaming, a running tool).
                await viewer.prompt("please [bash:6]")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = getattr(viewer, "frontend_state", None)
                    if state is not None and getattr(state, "streaming", False):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("the runtime never started the turn")

                _flip(prefix)
                # Long enough for settle + stagger + several checks to have
                # passed had the runtime wrongly considered itself idle, and
                # short enough that the sleep is still running.
                held_until = time.monotonic() + 3.0
                while time.monotonic() < held_until:
                    assert _alive(child), "a BUSY runtime retired mid-turn"
                    await pilot.pause()
                    await asyncio.sleep(0.1)

                # The sleep ends, the turn completes; the runtime is now idle
                # AND stale, and must retire on its own.
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and _alive(child):
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert not _alive(child), "the runtime never retired after its turn ended"

                deadline = time.monotonic() + 30
                new_record = None
                while time.monotonic() < deadline:
                    new_record = _record_for(config, session_id)
                    if (
                        new_record is not None
                        and int(new_record.pid) != old_pid
                        and not viewer.is_cold
                    ):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert new_record is not None and int(new_record.pid) != old_pid
                assert not viewer.is_cold
                await pilot.pause()
                text = transcript_text(app)
                for word in FORBIDDEN:
                    assert word not in text, f"{word!r} reached the ledger:\n{text}"
                assert (
                    "Hello from the mock provider!" in text
                ), "the turn that was live during the update must have completed"
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)


def test_an_unwatched_idle_runtime_retires_and_spawns_nothing(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design test 15: no viewer, no successor.

    An unwatched stale runtime retires; NOTHING re-spawns it (only a viewer
    triggers an eager successor). The next engage — here the CLI's
    ``lop send`` path, which is what a peer or a wake would use — then boots
    a runtime from the new stamp.
    """
    _stale_child_env(monkeypatch)
    config = headless_tui_env
    session_id = "refreshcold1"
    _seed(config, session_id)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / ".lop-source").write_text(OLD_MARKER, encoding="utf-8")
    monkeypatch.setenv("LOP_BUILD_PREFIX", str(prefix))
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.5")
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "120")

    child = _spawn(config, prefix, session_id)
    try:
        deadline = time.monotonic() + 30
        record = None
        while time.monotonic() < deadline:
            record = _record_for(config, session_id)
            if record is not None:
                break
            time.sleep(0.05)
        assert record is not None
        old_pid = int(record.pid)
        assert record.source_ref == OLD_MARKER.split()[0]

        _flip(prefix)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and _alive(child):
            time.sleep(0.05)
        assert not _alive(child), "the unwatched stale runtime never retired"
        assert child.returncode == 0

        # NO successor: nothing was watching, nothing is owed.
        time.sleep(1.0)
        assert _record_for(config, session_id) is None, "an unwatched refresh must spawn nothing"

        # The next engage runs the new build. ``lop send`` is the peer/wake
        # path: it engages a runtime for the target when none is live.
        env = _child_env(config, prefix, session_id)
        env.pop("LOP_MOBILE_CHILD_RESUME")
        env.pop("LOP_MOBILE_CHILD_CWD")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from local_operator.cli import main; sys.exit(main())",
                "send",
                "--session",
                session_id,
                "--wake",
                "hello",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        deadline = time.monotonic() + 30
        new_record = None
        while time.monotonic() < deadline:
            new_record = _record_for(config, session_id)
            if new_record is not None and int(new_record.pid) != old_pid:
                break
            time.sleep(0.05)
        assert (
            new_record is not None and int(new_record.pid) != old_pid
        ), "`lop send` must engage a fresh runtime for the retired session"
        assert new_record.source_ref == NEW_MARKER.split()[0], "…from the NEW build"
    finally:
        for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)


def _runtime_log_lines() -> list[str]:
    """Every line this ISOLATED config dir's runtimes have written.

    The runtimes share one ``runtime.log`` (``paths.runtime_log_path``), so a
    reader attributes a line by the pid it names — which is why the drain and
    exit records carry one.
    """
    from local_operator.paths import log_dir

    try:
        raw = (log_dir() / "runtime.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return raw.splitlines()


def _lines_for_pid(pid: int) -> list[str]:
    return [line for line in _runtime_log_lines() if f"pid {pid}" in line]


@pytest.mark.asyncio
async def test_a_busy_runtime_drains_at_the_bound_without_losing_its_turn(
    headless_tui_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound, end to end, against a runtime that never becomes idle.

    A turn that holds the bash tool for longer than the bound (three checks at
    ``BUILD_CHECK_S``), a marker flipped under it, and a viewer watching. This
    runtime has a live turn AND an attached viewer — either alone used to keep a
    stale runtime resident forever, which is how the reporting host ended up
    with eight runtimes executing an install tree that was gone.

    What must hold, in order:
    (i) admissions stop WHILE THE TURN IS STILL RUNNING — a prompt sent then is
        refused with the retiring sentence rather than queued;
    (ii) nothing in flight is aborted: the turn's own reply lands, and nothing
        on screen says ``interrupted``/``stopped``;
    (iii) a peer message sent mid-drain is SPOOLED for the successor, not
        refused and not run against the build that is leaving;
    (iv) the runtime leaves for the build on disk, and the viewer's eager
        re-engage boots a successor from the NEW stamp.
    """
    from local_operator.session.attached import AttachedSession

    _stale_child_env(monkeypatch)
    config = headless_tui_env
    session_id = "refreshdrain1"
    _seed(config, session_id)
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / ".lop-source").write_text(OLD_MARKER, encoding="utf-8")
    monkeypatch.setenv("LOP_BUILD_PREFIX", str(prefix))
    monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.5")
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "120")

    new_ref = NEW_MARKER.split()[0][:7]
    child = _spawn(config, prefix, session_id)
    viewer = None
    try:
        record = await _wait_for_record(config, session_id)
        old_pid = int(record.pid)
        viewer = await AttachedSession.connect(
            record, session_id, config_dir=config, takeover_factory=_never_take_over
        )

        async def factory() -> Any:
            return viewer

        app = OperatorApp(factory)
        with bounded(240, "runtime drain: a busy runtime at the bound"):
            async with app.run_test(size=(100, 30)) as pilot:
                await wait_for_adoption(app, pilot)
                app._loaded_build = BuildStamp(version=record.version, source_ref=record.source_ref)
                await pilot.pause()
                # A turn LONGER than the bound, so the drain has to latch mid-turn.
                await viewer.prompt("please [bash:30]")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = getattr(viewer, "frontend_state", None)
                    if state is not None and getattr(state, "streaming", False):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("the runtime never started the turn")

                _flip(prefix)  # lop-update ran, mid-turn

                # The drain announces itself in the runtime's own log, and it
                # must do so while the turn is still live.
                deadline = time.monotonic() + 120
                drain_lines: list[str] = []
                while time.monotonic() < deadline:
                    drain_lines = [
                        line
                        for line in _lines_for_pid(old_pid)
                        if "no new work will be admitted" in line
                    ]
                    if drain_lines:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.1)
                assert drain_lines, f"never drained:\n{chr(10).join(_runtime_log_lines()[-30:])}"
                assert new_ref in drain_lines[0], drain_lines[0]
                assert _alive(child), "the runtime left while its turn was running"
                assert getattr(
                    viewer.frontend_state, "streaming", False
                ), "the drain must have latched while the turn was STILL running"

                # (i) a caller asking to run a NEW turn is refused, not queued.
                # ``prompt_and_wait`` is the non-streaming prompt op (a loop's
                # idiom): an interactive viewer's text becomes a steer while a
                # turn is live, and a steer rides the turn already running —
                # which is in-flight work, not an admission.
                with pytest.raises(RuntimeError) as caught:
                    await asyncio.wait_for(
                        viewer.prompt_and_wait("a follow-up sent mid-drain"), timeout=60
                    )
                # The refusal is the typed admission category now, and its
                # sentence deliberately no longer contains the word "retiring":
                # it named an internal token and the machinery rather than the
                # session the operator is in (design round 1, D2). What the
                # caller can act on is still named, and the category is what
                # lets the viewer tell this refusal from any other failure —
                # the same pair of pins `test_retiring_refusal.py` carries.
                assert isinstance(caught.value, RuntimeRetiring), caught.value
                assert "send it again" in str(caught.value), str(caught.value)

                # (iii) a peer message is spooled for the successor instead.
                sent = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; from local_operator.cli import main; sys.exit(main())",
                        "send",
                        "--session",
                        session_id,
                        "--wake",
                        "hello from a peer",
                    ],
                    env=_child_env(config, prefix, session_id),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert sent.returncode == 0, sent.stdout + sent.stderr
                assert SPOOL_RECEIPT_WAKE in (sent.stdout + sent.stderr), sent.stdout + sent.stderr

                # (ii) the turn finishes rather than being aborted, and the
                # runtime leaves once it has.
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline and _alive(child):
                    await pilot.pause()
                    await asyncio.sleep(0.1)
                assert not _alive(child), "the runtime never left after its turn ended"
                assert child.returncode == 0, f"the runtime exited {child.returncode}"
                exit_lines = [
                    line
                    for line in _lines_for_pid(old_pid)
                    if "session runtime: exiting (retiring" in line
                ]
                assert exit_lines, "\n".join(_lines_for_pid(old_pid)[-10:])
                assert new_ref in exit_lines[-1], exit_lines[-1]

                # (iv) the viewer re-engages: a new pid, bound, on the NEW build.
                deadline = time.monotonic() + 60
                new_record = None
                while time.monotonic() < deadline:
                    new_record = _record_for(config, session_id)
                    if (
                        new_record is not None
                        and int(new_record.pid) != old_pid
                        and not viewer.is_cold
                    ):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert (
                    new_record is not None and int(new_record.pid) != old_pid
                ), "no successor runtime was engaged after the drain"
                assert not viewer.is_cold
                assert new_record.source_ref == NEW_MARKER.split()[0]
                # The spooled peer message is read by that successor, before
                # its socket even listens — so the work in flight was not the
                # only thing the drain preserved. Read from the TRANSCRIPT, the
                # durable copy: this is the receipt a later resume reads.
                from local_operator.session.transcript import Transcript

                durable = ""
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                    sessions_dir = config / "sessions" / session_id
                    durable = "\n".join(
                        str(entry.payload) for entry in Transcript(sessions_dir).entries()
                    )
                    if "hello from a peer" in durable:
                        break
                assert "hello from a peer" in durable, "the spooled peer message was lost"
                text = transcript_text(app)
                assert "Hello from the mock provider!" in text, "the live turn never completed"
                for word in FORBIDDEN:
                    assert word not in text, f"{word!r} reached the ledger:\n{text}"
                successor_pid = int(new_record.pid)
                # QUIT THE APP FROM INSIDE ITS OWN FRAME — the idiom the other
                # stages of this suite use (``test_viewer_attach_e2e``,
                # ``test_fork_checkpoint_e2e``). Falling out of the block
                # instead runs ``App.run_test``'s teardown from the TEST's
                # frame (``run_test`` awaits ``App._shutdown`` after awaiting
                # the app's own message loop), and Textual stops a widget's
                # timers while it awaits that widget's pump: a timer that comes
                # due in that window calls ``Timer._tick``, which reads the
                # ``active_app`` contextvar the app has already released, so the
                # teardown dies with ``LookupError: active_app`` rather than
                # stopping the timer. That is how this stage passed on
                # macos-latest and failed on ubuntu-latest (run 34824475709) —
                # a slow runner is simply more likely to have a timer due at
                # that instant. Exiting here lets the app finish its own
                # shutdown with its context still live, and the pump above it
                # lets the last messages land first.
                #
                # QUIESCE FIRST, and this is the part that actually removes the
                # race: the teardown stops every timer in the tree, and a timer
                # whose task was created by a callback that came back from a
                # THREAD cannot resolve ``active_app`` (``call_from_thread``
                # schedules on the loop with the calling thread's EMPTY context,
                # and Textual's own ``_stop_all`` then awaits a task whose
                # ``_tick`` raises ``LookupError``). Measured while
                # remediating this round: with every pre-exit timer still clean
                # (diagnosed in-process), the stage still died in teardown when
                # it left a stream and an in-flight adoption behind — ubuntu
                # run 34824475709 is the same failure on a slower runner. So the
                # stage now waits for the app's workers and the session's turn
                # to finish before it exits: there is then nothing in flight to
                # hand a timer back from a thread.
                await app.workers.wait_for_complete()
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and getattr(
                    viewer.frontend_state, "streaming", False
                ):
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                #
                # What this does NOT do (review round 2, NIT 2): ``run_test``'s
                # ``finally`` still calls ``_shutdown`` from the TEST's frame,
                # unconditionally and without an idempotency guard, so the
                # teardown is made BENIGN rather than avoided — the app's own
                # loop has already stopped its timers under the live
                # ``active_app``, and by then there are none left to tick in the
                # window where the contextvar is gone. Do not read this as "the
                # hazard is gone": a stage that skips ``app.exit()`` still has
                # it.
                await pilot.pause()
                app.exit()
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        for pid in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        child.wait(timeout=10)
    assert successor_pid != old_pid


# -- the ARMED probe (QA round 1, Q-4) ------------------------------------------
#
# Every stage above runs on the repo's own interpreter, which is an editable
# venv: ``install_kind()`` answers EDITABLE, ``_tree_is_replaceable()`` is
# False, and the files-gone probe is DISARMED. That is exactly where QA's
# BLOCKER lived — the armed path had no end-to-end coverage anywhere in the
# repo, so the interaction between the drain and the idle predicate shipped
# unmeasured (Q-1). This cell builds a real non-editable install of the tree
# under test and removes the loaded package while a turn is running.


def _noneditable_install(tmp_path: Path, *, stamp: str) -> tuple[Path, Path]:
    """A throwaway uv-tool-shaped, non-editable install. ``(prefix, tree)``.

    No network and no build backend: the package is HARDLINKED out of the tree
    under test (cheap for ~2 000 files, and REAL paths — a symlink would resolve
    back to the worktree, so removing the copy would remove nothing), a
    hand-written ``dist-info`` is what makes ``importlib.metadata`` see a real
    non-editable distribution, and the dependencies are reached through one
    ``.pth`` line pointing at the interpreter that runs this test. That line is
    a bare PATH on purpose: ``site`` appends it WITHOUT processing the ``.pth``
    files inside it, so the test venv's editable-install finder is never
    installed in the child — which is what lets the child import the copy this
    test is about to delete.
    """
    import importlib.metadata
    import json
    import sysconfig

    package = Path(local_operator.__file__).resolve().parent
    prefix = tmp_path / "uv" / "tools" / "local-operator"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(prefix)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    site = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site.mkdir(parents=True, exist_ok=True)
    tree = site / "local_operator"
    shutil.copytree(package, tree, copy_function=os.link)

    version = importlib.metadata.version("local-operator")
    dist = site / f"local_operator-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
    )
    (dist / "INSTALLER").write_text("uv\n", encoding="utf-8")
    # The deps path below makes the test venv's site-packages visible to the
    # child, and THAT holds the real editable install's ``direct_url.json``.
    # ``update._direct_url_payload`` deliberately scans EVERY distribution of
    # this name and takes the first that publishes the marker, so without this
    # the child would answer EDITABLE and the probe would stay disarmed. This is
    # what a non-editable directory install writes for itself (PEP 610): a URL
    # and an empty ``dir_info``, editable absent.
    (dist / "direct_url.json").write_text(
        json.dumps({"url": prefix.as_uri(), "dir_info": {}}), encoding="utf-8"
    )
    (site / "_test_deps.pth").write_text(
        str(Path(sysconfig.get_paths()["purelib"])) + "\n", encoding="utf-8"
    )
    (prefix / ".lop-source").write_text(stamp, encoding="utf-8")
    return prefix, tree


def _completion_attention(config_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """The turn-outcome rows this session's transcript carries, in order.

    The durable equivalent of the viewer's cut-off vocabulary: a drain that
    aborted a turn writes an ``error`` row, a turn that finished writes
    ``complete``. Read from the JSONL rather than the screen because the words
    appear in payloads that have nothing to do with this turn.
    """
    durable = (config_dir / "sessions" / session_id / "transcript.jsonl").read_text(
        encoding="utf-8", errors="replace"
    )
    # A FINAL LINE CAN BE HALF-WRITTEN: this reader is polled while the child is
    # still appending (the warm-up wait below needs the row the moment it
    # lands), so an unparsable line is a read that lost the race, not a fact.
    rows = []
    for line in durable.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [
        row["payload"]["details"]
        for row in rows
        if row.get("type") == "custom"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("custom_type") == "completion_attention"
    ]


def _completion_attention_report(config_dir: Path, session_id: str) -> str:
    """Which of the two ways a missing latch happened, in words.

    "The latch must land while the turn is STILL running" is false in two
    opposite situations, and they are different diagnoses: the drain was LATE
    (the bug this stage exists for), or the turn DIED of the deletion — its
    turn's prelude imports a package module from disk, and this stage has just
    removed the tree that module lives in. The completion rows tell them apart,
    so a red cell names its cause instead of reporting both as one
    (review round 3, MINOR 1).
    """
    rows = _completion_attention(config_dir, session_id)
    killed = [
        row
        for row in rows
        if row.get("kind") == "error" and "No module named" in str(row.get("reason", ""))
    ]
    if killed:
        return (
            "THE DELETED TREE KILLED THE TURN, NOT A LATE DRAIN — the child's own "
            f"site-packages is what a function-local import in its turn prelude "
            f"reaches for, and this cell removed it: {killed!r}"
        )
    return f"the turn's completion rows so far: {rows!r}"


def _install_probe(prefix: Path) -> list[str]:
    """What the CHILD's own interpreter says about the install it will run from.

    Measured in the child rather than assumed from the fixture: if this answers
    anything but ``UV_TOOL`` / ``True`` / a path inside the prefix, the cell
    proves nothing — and a fixture that silently disarmed the probe is how Q-1
    reached a green board in the first place.
    """
    probe = subprocess.run(
        [
            str(prefix / "bin" / "python"),
            "-c",
            "import local_operator; print(local_operator.__file__)\n"
            "import local_operator.update as u; print(u.install_kind())\n"
            "import local_operator.session.runtime.process as p; print(p._tree_is_replaceable())",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if not k.startswith("CMUX_")},
    )
    return (probe.stdout + probe.stderr).strip().splitlines()


def _armed_child_env(config_dir: Path, session_id: str) -> dict[str, str]:
    """``_child_env`` WITHOUT ``LOP_BUILD_PREFIX``.

    This child runs from its own non-editable prefix, so ``build_prefix()`` is
    None and ``installed_build`` reads the marker in its own ``sys.prefix`` —
    the production shape rather than the e2e stage's fake marker. The shortened
    settle/stagger and the stripped ``CMUX_*`` are inherited from the shared
    helper; the grace stays long so only the probe can retire this runtime.
    """
    env = _child_env(config_dir, config_dir, session_id)
    env.pop("LOP_BUILD_PREFIX", None)
    return env


@pytest.mark.asyncio
async def test_the_armed_probe_drains_and_exits_when_the_loaded_tree_vanishes(
    headless_tui_env: Path, tmp_path: Path
) -> None:
    """Q-1's pin on the ARMED path: the drain must COMPLETE, not merely latch.

    ``uv tool install --force`` removes the old distribution before writing the
    new one, so a window in production exists in which a runtime's loaded tree
    is absent. This cell reproduces that window for real — a non-editable
    install of this tree, booted from its own interpreter, with the package
    directory deleted mid-turn — and asserts what a user can see: the runtime
    announces the handover while its turn is STILL running, the turn finishes
    (nothing aborted, no cut-off vocabulary), and the process EXITS.

    That last assertion is the pin. On the head QA measured, the drain latched
    and the process never exited: ``may_refresh`` reached its warm-window term
    through a function-local import of ``session.runtime.process`` — which this
    child runs as ``__main__``, so the import was answered from DISK, gone, and
    raised ``ImportError``; ``process._idle_for_refresh`` reads a failing
    predicate as "not idle", so ``begin_retire``/``_clean_exit`` were unreachable
    for as long as the tree stayed gone and the session was refused forever
    while its lease kept any successor from booting.
    """
    from local_operator.session.attached import AttachedSession

    config = headless_tui_env
    session_id = "refresharmed1"
    _seed(config, session_id)
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )
    prefix, tree = _noneditable_install(tmp_path, stamp=OLD_MARKER)
    probe = _install_probe(prefix)
    assert probe, "the install probe printed nothing"
    assert str(prefix) in probe[0], f"the child imports something else: {probe}"
    assert "UV_TOOL" in probe[1], f"install_kind is not UV_TOOL: {probe}"
    assert probe[2] == "True", f"the probe is not armed: {probe}"

    child = subprocess.Popen(
        [str(prefix / "bin" / "python"), "-m", "local_operator.session.runtime.process"],
        env=_armed_child_env(config, session_id),
        # cwd set to a directory with no package in it, and that matters more
        # than it looks: ``-m`` puts the cwd first on ``sys.path``, so running
        # this child from the checkout would import the WORKTREE's
        # ``local_operator`` instead of the install under test — and the tree
        # this cell removes could never be the loaded one.
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    viewer = None
    try:
        record = await _wait_for_record(config, session_id)
        pid = int(record.pid)
        viewer = await AttachedSession.connect(
            record, session_id, config_dir=config, takeover_factory=_never_take_over
        )
        await viewer.prompt("warm the turn's module path")
        # WARM THE PRELUDE FIRST, then delete. A turn's prelude imports its
        # classifier's module from DISK on the first provider request
        # (a function-local import in ``configure.py``), while
        # ``frontend_state`` says ``streaming`` from the moment the turn is
        # ADMITTED — before that request. Deleting in that gap kills the turn
        # with ``No module named 'local_operator.model.effort_classifier'``,
        # which has nothing to do with the drain: QA round 3 measured the margin
        # at 0.03 s and reproduced the failure deterministically 30 ms earlier,
        # which is why this cell went red once on a loaded runner. One completed
        # turn puts that module in the child's ``sys.modules``, so the tree this
        # cell removes is no longer on the path a live turn needs. Readiness,
        # not a retry (QA round 3; review round 3, MINOR 1) — and the wait is on
        # the turn's own durable COMPLETION rather than on ``streaming``, which
        # is a reading the submit has not yet flipped when this line runs.
        # ... and the wait is on the turn's own durable COMPLETION row, not on
        # the reply text and not on ``streaming``: the reply landing does not
        # mean the turn is over, and the first CI run of this cell failed
        # exactly there — the prompt below arrived while the warm-up turn was
        # still live, was taken as a STEER into it, and no second turn ever
        # opened. A completion row is the turn's own record that it is over.
        warm_deadline = time.monotonic() + 120
        while time.monotonic() < warm_deadline:
            if any(
                row.get("kind") == "complete" for row in _completion_attention(config, session_id)
            ):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the warm-up turn never completed")

        await viewer.prompt("please [bash:12]")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = getattr(viewer, "frontend_state", None)
            if state is not None and getattr(state, "streaming", False):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the runtime never started the turn")

        shutil.rmtree(tree)  # what `uv tool install --force` does first

        # (a) It announces the handover, while the turn is still live.
        deadline = time.monotonic() + 60
        drain_lines: list[str] = []
        while time.monotonic() < deadline:
            drain_lines = [
                line for line in _lines_for_pid(pid) if "no new work will be admitted" in line
            ]
            if drain_lines:
                break
            await asyncio.sleep(0.1)
        assert drain_lines, f"never drained:\n{chr(10).join(_runtime_log_lines()[-30:])}"
        assert "loaded module tree is gone" in drain_lines[0], drain_lines[0]
        assert getattr(
            viewer.frontend_state, "streaming", False
        ), "the latch must land while the turn is STILL running — " + _completion_attention_report(
            config, session_id
        )

        # (b) It EXITS — the Q-1 pin. On the wedged head this loop timed out with
        # the process still resident, holding the session's lease.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and _alive(child):
            await asyncio.sleep(0.2)
        assert not _alive(child), "the drain latched and the runtime NEVER exited:\n" + chr(
            10
        ).join(_lines_for_pid(pid)[-10:])
        assert child.returncode == 0, f"the runtime exited {child.returncode}, not cleanly"

        # (c) Nothing in flight was aborted: the turn's own reply is durable.
        durable = (config / "sessions" / session_id / "transcript.jsonl").read_text(
            encoding="utf-8", errors="replace"
        )
        assert "Hello from the mock provider!" in durable, "the live turn never completed"
        # ... and the session recorded that turn as COMPLETE, which is the
        # durable equivalent of the viewer's cut-off vocabulary. A drain that
        # aborted the turn writes an ERROR attention row for it, so the kind is
        # the assertion. Both turns are covered: the warm-up one above finished
        # before the deletion, and an error row from EITHER is the drain cutting
        # a turn off.
        attention = _completion_attention(config, session_id)
        assert attention, f"no completion was recorded at all:\n{durable[-2000:]}"
        assert [
            row for row in attention if row.get("kind") == "error"
        ] == [], "the drain cut the turn off:\n" + json.dumps(attention, indent=2)
        assert any(row.get("kind") == "complete" for row in attention), attention
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001
                pass
        for other in {child.pid, *(int(r.pid) for r, _ in registry.scan(config))}:
            try:
                os.kill(other, 9)
            except ProcessLookupError:
                pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — the kill above reaps it
            pass
