"""A stale idle runtime refreshes itself; the viewer never sees a chore.

The operator's requirement, verbatim: "runtimes that are inactive
automatically refresh and update / bring down the runtime on inactive
sessions so that resuming would do the update. On resume we should never see
the [/stop, then send again] message. The user should never need to run
/stop to refresh or update a runtime."

These tests boot the PRODUCTION ``process.py`` in a subprocess against a
fake install prefix (``LOP_BUILD_PREFIX`` → a temp dir carrying a
``.lop-source`` marker), attach the production ``RemoteSession`` under the
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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry
from local_operator.tui.app import OperatorApp
from local_operator.update import BuildStamp
from tests.e2e.harness import transcript_text, wait_for_adoption
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
    from local_operator.session.remote import RemoteSession

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

        viewer = await RemoteSession.connect(
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
    from local_operator.session.remote import RemoteSession

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
        viewer = await RemoteSession.connect(
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
