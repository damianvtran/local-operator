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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime import registry
from local_operator.session.runtime.inbox import (
    SPOOL_RECEIPT_PROMPT,
    SPOOL_RECEIPT_WAKE,
)
from local_operator.tui.app import OperatorApp
from local_operator.update import BuildStamp
from tests.e2e import armed_probe_diagnostics
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


#: Backstop for a runtime child to publish its registry record. The wait returns
#: the moment the record exists, so this only bounds a child that never boots.
#: 30 s was not enough on a saturated box: the armed cell's non-editable child
#: (a cold interpreter importing the whole package) took longer than that to
#: reach its first record in 4 of 12 local runs under 12 busy loops on a
#: 14-core host (load average ~60), old head and new alike, failing with
#: ``no record for refresharmed1 within 30.0s`` -- a boot-speed wager, not a
#: product signal. An idle box needs ~2 s.
RECORD_BACKSTOP_S = 120.0


async def _wait_for_record(
    config_dir: Path, session_id: str, timeout: float = RECORD_BACKSTOP_S
) -> Any:
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


#: How long the armed-probe cell's turn parks in the mock's ``[bash:N]`` tool.
#: It is the mock's CAP (``_mock_bash_sleep`` bounds N at 60) on purpose: the
#: turn is not meant to END on this clock. The cell releases it itself, with
#: :func:`_release_parked_sleep`, once it has OBSERVED the latch; the number only
#: has to outlast the latch (5-10 s on an idle box: a 5 s probe tick, a settle,
#: one more tick) with room for a starved runner, and a bigger one would just
#: be a longer hang if the release never came.
ARMED_HOLD_S = 60


def _descendant_commands(root_pid: int) -> list[tuple[int, str, bool]]:
    """``(pid, argv, is_leaf)`` for every live descendant of ``root_pid``.

    ``ps`` rather than psutil: it is not a dependency here, and the one fact
    wanted -- "is the parked ``sleep`` a live descendant of THIS runtime" -- is
    an ancestry question on a table that is short on a test box. ``-A -o`` with
    empty headers parses the same on macOS and the Linux runner.
    """
    table = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,args="], capture_output=True, text=True, timeout=30
    ).stdout
    children: dict[int, list[int]] = {}
    commands: dict[int, str] = {}
    for line in table.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 2 or not (fields[0].isdigit() and fields[1].isdigit()):
            continue
        pid, parent = int(fields[0]), int(fields[1])
        commands[pid] = fields[2] if len(fields) > 2 else ""
        children.setdefault(parent, []).append(pid)
    found: list[tuple[int, str, bool]] = []
    queue = list(children.get(root_pid, []))
    while queue:
        pid = queue.pop()
        found.append((pid, commands.get(pid, ""), pid not in children))
        queue.extend(children.get(pid, []))
    return found


def _parked_sleep_pids(root_pid: int, seconds: int) -> list[int]:
    """The ``sleep <seconds>`` leaf processes under the runtime: the bash tool, RUNNING.

    A leaf, because ``bash -c "sleep N"`` may or may not exec its last command
    and a wrapper's argv repeats the command line; the leaf is the one that is
    actually sleeping, and the one safe to signal.
    """
    wanted = f"sleep {seconds}"
    return [
        pid
        for pid, args, leaf in _descendant_commands(root_pid)
        if leaf and args.strip().split("/")[-1].endswith(wanted)
    ]


def _release_parked_sleep(root_pid: int, seconds: int) -> int:
    """End the parked tool call now (SIGTERM to its exact pid); how many were signalled.

    The turn then completes the way any bash call that exited non-zero does --
    the mock answers the tool row with text -- so the cell's later "the turn's
    reply is durable" and "no error attention row" assertions still read a turn
    that finished rather than one that was aborted.
    """
    signalled = 0
    for pid in _parked_sleep_pids(root_pid, seconds):
        try:
            os.kill(pid, 15)
            signalled += 1
        except ProcessLookupError:
            pass
    return signalled


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


# --- Successor-engage diagnostics (TEMPORARY; test-side only) -------------
#
# WHY THIS EXISTS: ``test_a_watched_idle_runtime_refreshes_and_the_viewer_
# rebinds_silently`` failed on main CI (run 42780150549, job 129710348660, and
# job 129293345900 before it, both ``tui-e2e (macos-latest, 1)``) with "no
# successor runtime was engaged after the refresh": the old runtime retired
# cleanly and no successor record appeared for the full 30 s. Three serial
# local runs passed (successor latency n=3, median 5.42 s, max 6.91 s), so it
# is UNREPRODUCED, and CI captured no log lines to say whether the spawn path
# was never taken (a quiet early return) or was taken and failed. Candidates:
#   1. ``AttachedSession._ensure_bound`` returning early on ``_recovering`` /
#      ``not _can_go_cold`` / ``_disposed`` / ``not is_cold``;
#   2. ``OperatorApp._start_runtime_engage`` returning early on the
#      ``_warm_engage_started`` latch, a non-cold viewer, or
#      ``_runtime_can_start()`` False;
#   3. ``AttachedSession._on_disconnected`` taking the owner-loss path (no
#      RETIRING reason, or ``_recovering`` already set), which cannot produce a
#      successor under ``_never_take_over``;
#   4. an environment leak (``LOP_WAKE_SUPERVISOR_*`` written into os.environ
#      by ``wakes/supervisor.py`` and copied by ``launch._runtime_env``).
#
# CONTRACT: observation only. The spies below call straight through and only
# append to a list; the dump runs on the TIMEOUT path alone, before the
# unchanged assertion, and every read is individually guarded so nothing here
# can raise, wait, or change the outcome. Output goes to stderr, which pytest
# prints under "Captured stderr call" in the CI job log on failure.
#
# REMOVE once a CI occurrence has named the mechanism (and the fix has landed).

_DIAG_TAIL_BYTES = 2048
_DIAG_MAX_CAPTURES = 3
_DIAG_MAX_LISTING = 60


def _diag_read(label: str, read: Any) -> str:
    try:
        return f"{label}={read()!r}"
    except BaseException as error:  # noqa: BLE001 — a diagnostic never raises
        return f"{label}=<unreadable: {type(error).__name__}: {error}>"


def _install_successor_spies(app: Any, viewer: Any, events: list[str], t0: list[float]) -> None:
    """Record which re-engage hops ran, without changing any of them.

    Each target is looked up dynamically by its caller (``_go_cold`` reads
    ``self._refresh_callback`` / ``self._went_cold_callback`` at call time; the
    app calls ``self._start_runtime_engage`` and ``getattr(session,
    "_ensure_bound")``; the attach client's ``on_disconnected`` closure calls
    ``self._on_disconnected(reason)`` and that method schedules
    ``self._recover_runtime()``), so an instance attribute wrapping the
    original sees every call. ``t0`` is filled at the flip so offsets read
    from there; install events land before it and read from their own clock.

    Every spy does its pre-call reads inside a ``try`` so a raising property
    can never skip the real call, and preserves the original's sync/async
    kind, arguments, return value and exception exactly. Each successful
    install records one ``installed`` event, so a hop that never shows up can
    be told apart from a spy that was overwritten or never installed.
    """

    def note(text: Any) -> None:
        # ``text`` may be a zero-arg callable so a post-call message (which
        # formats the error or result) is built inside this guard too: after
        # the real call nothing may replace its return value or exception.
        try:
            base = t0[0] if t0 else time.monotonic()
            events.append(f"+{time.monotonic() - base:.2f}s {text() if callable(text) else text}")
        except BaseException:  # noqa: BLE001
            pass

    def peek(obj: Any, name: str) -> Any:
        # ``getattr``'s default covers AttributeError only; a property that
        # raises anything else must not turn an observed call into a failure.
        try:
            return getattr(obj, name, "?")
        except BaseException as error:  # noqa: BLE001
            return f"<unreadable: {type(error).__name__}>"

    def wrap_callback(name: str) -> None:
        try:
            original = getattr(viewer, name, None)
            if not callable(original):
                note(f"{name} not registered ({original!r})")
                return

            def spy(*args: Any, **kwargs: Any) -> Any:
                try:
                    note(f"viewer.{name} fired; is_cold={peek(viewer, 'is_cold')}")
                except BaseException:  # noqa: BLE001
                    pass
                return original(*args, **kwargs)

            setattr(viewer, name, spy)
            note(f"installed spy on viewer.{name}")
        except BaseException as error:  # noqa: BLE001
            note(f"spy on {name} not installed: {error!r}")

    wrap_callback("_refresh_callback")
    wrap_callback("_went_cold_callback")

    try:
        engage = app._start_runtime_engage

        def engage_spy(*args: Any, **kwargs: Any) -> Any:
            try:
                note(
                    f"app._start_runtime_engage({kwargs.get('reason', args)!r}) entered; "
                    f"latch={peek(app, '_warm_engage_started')} "
                    f"is_cold={peek(viewer, 'is_cold')}"
                )
            except BaseException:  # noqa: BLE001
                pass
            return engage(*args, **kwargs)

        app._start_runtime_engage = engage_spy
        note("installed spy on app._start_runtime_engage")
    except BaseException as error:  # noqa: BLE001
        note(f"spy on _start_runtime_engage not installed: {error!r}")

    try:
        ensure = viewer._ensure_bound

        async def ensure_spy(*args: Any, **kwargs: Any) -> Any:
            try:
                note(
                    f"viewer._ensure_bound({kwargs!r}) entered; recovering="
                    f"{peek(viewer, '_recovering')} can_go_cold="
                    f"{peek(viewer, '_can_go_cold')} disposed="
                    f"{peek(viewer, '_disposed')} is_cold={peek(viewer, 'is_cold')}"
                )
            except BaseException:  # noqa: BLE001
                pass
            try:
                result = await ensure(*args, **kwargs)
            except BaseException as error:
                note(lambda e=error: f"viewer._ensure_bound raised {type(e).__name__}: {e}")
                raise
            note(lambda: f"viewer._ensure_bound returned; is_cold={peek(viewer, 'is_cold')}")
            return result

        viewer._ensure_bound = ensure_spy
        note("installed spy on viewer._ensure_bound")
    except BaseException as error:  # noqa: BLE001
        note(f"spy on _ensure_bound not installed: {error!r}")

    # Candidate 3: which ``_on_disconnected`` branch ran is decided by the
    # reason string alone, and the product keeps none of it, so record the
    # RECEIVED reason and how it compares with each branch constant.
    try:
        disconnected = viewer._on_disconnected

        def disconnected_spy(reason: str) -> Any:
            try:
                from local_operator.mobile.attach_client import (
                    MOVED_REASON,
                    RETIRING_REASON,
                    STOPPED_REASON,
                )

                note(
                    f"viewer._on_disconnected(reason={reason!r}) entered; "
                    f"== RETIRING_REASON={reason == RETIRING_REASON} "
                    f"(RETIRING_REASON={RETIRING_REASON!r}) "
                    f"== MOVED_REASON={reason == MOVED_REASON} "
                    f"== STOPPED_REASON={reason == STOPPED_REASON} "
                    f"disposed={peek(viewer, '_disposed')} "
                    f"recovering={peek(viewer, '_recovering')} "
                    f"deliberate_stop={peek(viewer, '_deliberate_stop')}"
                )
            except BaseException:  # noqa: BLE001
                pass
            return disconnected(reason)

        viewer._on_disconnected = disconnected_spy
        note("installed spy on viewer._on_disconnected")
    except BaseException as error:  # noqa: BLE001
        note(f"spy on _on_disconnected not installed: {error!r}")

    # The owner-lost branch's outcome: ``_recover_runtime`` is a coroutine
    # function wrapped in ``asyncio.create_task``, so the spy stays ``async``
    # and records cancellation (a BaseException) as well as errors.
    try:
        recover = viewer._recover_runtime

        async def recover_spy(*args: Any, **kwargs: Any) -> Any:
            try:
                note(
                    f"viewer._recover_runtime entered; deliberate_stop="
                    f"{peek(viewer, '_deliberate_stop')} "
                    f"is_cold={peek(viewer, 'is_cold')}"
                )
            except BaseException:  # noqa: BLE001
                pass
            try:
                result = await recover(*args, **kwargs)
            except BaseException as error:
                note(lambda e=error: f"viewer._recover_runtime raised {type(e).__name__}: {e}")
                raise
            note(
                lambda: f"viewer._recover_runtime returned {result!r}; "
                f"is_cold={peek(viewer, 'is_cold')} "
                f"runtime_pid={peek(viewer, '_runtime_pid')}"
            )
            return result

        viewer._recover_runtime = recover_spy
        note("installed spy on viewer._recover_runtime")
    except BaseException as error:  # noqa: BLE001
        note(f"spy on _recover_runtime not installed: {error!r}")


def _dump_successor_diagnostics(
    *,
    app: Any,
    viewer: Any,
    config: Path,
    session_id: str,
    old_pid: int,
    child: subprocess.Popen[bytes],
    events: list[str],
    t0: list[float],
) -> None:
    """Print the state every candidate gate reads; never raises."""
    try:
        lines: list[str] = ["===== RUNTIME-REFRESH-DIAG: successor poll timed out ====="]
        flip = t0[0] if t0 else time.monotonic()
        lines.append(f"since_flip={time.monotonic() - flip:.2f}s old_pid={old_pid}")
        lines.append(_diag_read("old_child.returncode", lambda: child.returncode))
        # Gates 1 and 3: the viewer's own flags.
        for label, read in (
            ("viewer.is_cold", lambda: viewer.is_cold),
            ("viewer._recovering", lambda: viewer._recovering),
            ("viewer._disposed", lambda: viewer._disposed),
            ("viewer._can_go_cold", lambda: viewer._can_go_cold),
            ("viewer._deliberate_stop", lambda: viewer._deliberate_stop),
            ("viewer._moved_to", lambda: viewer._moved_to),
            ("viewer._client", lambda: viewer._client),
            ("viewer._client.connected", lambda: getattr(viewer._client, "connected", None)),
            ("viewer._ready_for_events", lambda: viewer._ready_for_events),
            ("viewer._runtime_pid", lambda: viewer._runtime_pid),
            ("viewer._runtime_ready.is_set", lambda: viewer._runtime_ready.is_set()),
            ("viewer._recovery_task", lambda: viewer._recovery_task),
            ("viewer._pending_frame_cwd", lambda: viewer._pending_frame_cwd),
        ):
            lines.append(_diag_read(label, read))
        # The disconnect reason is not retained by AttachedSession; the
        # ``_on_disconnected`` spy records the reason it RECEIVED (and the
        # ``_recover_runtime`` spy the owner-lost branch's outcome). No
        # EngageOutcome is retained either (it is local to ``_ensure_bound``);
        # its spy records return/raise.
        lines.append(
            "last_disconnect_reason=<not retained by product code; see _on_disconnected spy>"
        )
        lines.append("last_engage_outcome=<not retained by product code; see _ensure_bound spy>")
        # Gate 2: the app's latch and startability.
        for label, read in (
            ("app._warm_engage_started", lambda: app._warm_engage_started),
            ("app._runtime_can_start()", lambda: app._runtime_can_start()),
            ("app._refreshed_from", lambda: app._refreshed_from),
            ("app._session is viewer", lambda: app._session is viewer),
        ):
            lines.append(_diag_read(label, read))
        lines.append(f"spy_events ({len(events)}):")
        lines.extend(f"  {event}" for event in list(events)[-40:])
        # The registry as this test sees it.
        lines.append(_diag_read("_record_for", lambda: _record_for(config, session_id)))
        lines.append(
            _diag_read(
                "registry.scan",
                lambda: [
                    (getattr(r, "session_id", "?"), getattr(r, "pid", "?"), s)
                    for r, s in registry.scan(config, reap=False)
                ],
            )
        )
        roots = (("run_tree", config / "run"), ("session_dir", config / "sessions" / session_id))
        for label, root in roots:
            try:
                entries = sorted(root.rglob("*")) if root.exists() else []
                lines.append(f"{label} {root} ({len(entries)} entries):")
                for entry in entries[:_DIAG_MAX_LISTING]:
                    try:
                        size = entry.stat().st_size if entry.is_file() else "dir"
                    except OSError as error:
                        size = f"<{error}>"
                    lines.append(f"  {entry.relative_to(root)} {size}")
            except BaseException as error:  # noqa: BLE001
                lines.append(f"{label}=<unreadable: {error!r}>")
        # The session's small state files (``runtime-exit.json``,
        # ``update-window.json``) say how the old runtime left; bounded.
        try:
            session_root = config / "sessions" / session_id
            for state in sorted(session_root.glob("*.json"))[:_DIAG_MAX_LISTING]:
                try:
                    body = state.read_bytes()[:_DIAG_TAIL_BYTES].decode("utf-8", "replace")
                    lines.append(f"--- {state.name} ---\n{body}")
                except BaseException as error:  # noqa: BLE001
                    lines.append(f"--- {state.name} unreadable: {error!r}")
        except BaseException as error:  # noqa: BLE001
            lines.append(f"session state files=<unreadable: {error!r}>")
        # Successor spawn captures (``launch`` writes ``lop-runtime-*.log`` via
        # mkstemp and unlinks it on success), so a surviving recent one is the
        # output of a spawn that did not hand over.
        try:
            captures = sorted(
                (
                    p
                    for p in Path(tempfile.gettempdir()).glob("lop-runtime-*.log")
                    if p.stat().st_mtime >= time.time() - (time.monotonic() - flip) - 5
                ),
                key=lambda p: p.stat().st_mtime,
            )[-_DIAG_MAX_CAPTURES:]
            lines.append(f"recent spawn captures: {[str(p) for p in captures]}")
            for capture in captures:
                try:
                    tail = capture.read_bytes()[-_DIAG_TAIL_BYTES:].decode("utf-8", "replace")
                    lines.append(f"--- tail {capture} ---\n{tail}")
                except BaseException as error:  # noqa: BLE001
                    lines.append(f"--- {capture} unreadable: {error!r}")
        except BaseException as error:  # noqa: BLE001
            lines.append(f"spawn captures=<unreadable: {error!r}>")
        # Gate 4: what a spawned successor would inherit.
        lines.append(
            _diag_read(
                "os.environ LOP_WAKE_SUPERVISOR_*",
                lambda: {
                    k: v for k, v in os.environ.items() if k.startswith("LOP_WAKE_SUPERVISOR")
                },
            )
        )
        lines.append("===== /RUNTIME-REFRESH-DIAG =====")
        print("\n".join(lines), file=sys.stderr, flush=True)
    except BaseException as error:  # noqa: BLE001 — the assertion below must still fire
        try:
            print(f"RUNTIME-REFRESH-DIAG failed: {error!r}", file=sys.stderr, flush=True)
        except BaseException:  # noqa: BLE001
            pass


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

                # Diagnostics only (see _install_successor_spies).
                diag_events: list[str] = []
                diag_t0: list[float] = []
                _install_successor_spies(app, viewer, diag_events, diag_t0)
                diag_t0.append(time.monotonic())
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
                else:
                    # Timeout path only: the loop ran out without a bound
                    # successor. Diagnostics, then the unchanged assertions.
                    _dump_successor_diagnostics(
                        app=app,
                        viewer=viewer,
                        config=config,
                        session_id=session_id,
                        old_pid=old_pid,
                        child=child,
                        events=diag_events,
                        t0=diag_t0,
                    )
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
    (i) admissions stop WHILE THE TURN IS STILL RUNNING — a caller asking for a
        NEW TURN (``prompt_and_wait``, a loop's idiom) is told this runtime will
        not run it and that a newer build is starting, while a prompt handed to
        the runtime as the owner's own message is QUEUED for that successor and
        answered with the spool receipt;
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

                # (i) a caller asking for a NEW TURN is told the truth, and the
                # truth now includes where the message went.
                # ``prompt_and_wait`` is the non-streaming prompt op (a loop's
                # idiom): an interactive viewer's text becomes a steer while a
                # turn is live, and a steer rides the turn already running —
                # which is in-flight work, not an admission, and stays admitted.
                with pytest.raises(RuntimeError) as caught:
                    await asyncio.wait_for(
                        viewer.prompt_and_wait("a follow-up sent mid-drain"), timeout=60
                    )
                # The refusal is the typed admission category, and its sentence
                # deliberately no longer contains the word "retiring": it named
                # an internal token and the machinery rather than the session the
                # operator is in (design round 1, D2). What it ALSO no longer
                # says is "send it again" — this caller cannot watch a turn that
                # another process will run, but the message it handed over IS
                # queued for exactly that process (memo §4.2 piece 3), so the
                # sentence is the DEFERRAL, named as a deferral, rather than
                # asking for the operation the refusal just performed (agent
                # review round 1, R2; QA round 1, Q-2).
                assert isinstance(caught.value, RuntimeRetiring), caught.value
                assert caught.value.TAIL == RuntimeRetiring.TAIL_QUEUED, caught.value.TAIL
                assert "queued" in str(caught.value), str(caught.value)
                assert "send it again" not in str(caught.value), str(caught.value)

                # (i, continued) THE OWNER'S OWN MESSAGE IS QUEUED, NOT REFUSED.
                # Sent as the raw prompt op a composer's submit issues (the
                # interactive path routes to a steer while a turn streams, which
                # is a different op and deliberately unchanged), so what this
                # exercises is the admission itself: the runtime writes it to
                # ``inbox.jsonl`` for the successor and answers with a receipt of
                # its own rather than the durable-admission ACK — the weaker fact
                # has to be visibly weaker.
                queued_command = str(uuid.uuid4())
                client = viewer._client
                assert client is not None, "the viewer is bound, so it has a client"
                queued_receipt = await client.prompt(
                    "run the report while the build moves",
                    command_id=queued_command,
                )
                assert queued_receipt == SPOOL_RECEIPT_PROMPT, queued_receipt
                # AND A RETRY UNDER THE SAME ID IS NOT A SECOND TURN. The
                # admission identity is append-only, so there are now TWO spool
                # rows carrying one command id and the assertion further down
                # (`[entry.id for entry in ours] == [queued_command]`) is the
                # real-index proof that the successor ran it exactly once — the
                # same claim the unit cells make, but against the production
                # path rather than a double (agent review round 1, R4).
                retry_receipt = await client.prompt(
                    "run the report while the build moves",
                    command_id=queued_command,
                )
                assert retry_receipt == SPOOL_RECEIPT_PROMPT, retry_receipt

                # (i, continued 2) AND A COLD VIEWER CAN STILL JOIN THIS SESSION
                # WHILE IT DRAINS. This is the axis the whole change exists for,
                # and the one the first attempt broke: the engage answers a
                # draining record with the leaving detail rather than a raise, so
                # the joiner's bind REACHES the runtime that holds the transcript
                # instead of reporting a live session as "reconnecting" without
                # dialling it at all (agent review round 1, R1 — measured there
                # as 0 dials on the broken head, 1 on main). A fresh facade is the
                # joiner a second TUI, a `lop` mount or the desktop proxy is.
                joiner = await asyncio.wait_for(
                    AttachedSession.connect(
                        record, session_id, config_dir=config, takeover_factory=_never_take_over
                    ),
                    timeout=30,
                )
                try:
                    assert not joiner.is_cold, "a live, draining session was unreachable"
                    assert joiner._client is not None, "the joiner holds no control connection"
                    # ANSWERABLE too: its own message is queued for the successor
                    # rather than refused, which is the receipt the incident's
                    # users never got.
                    joiner_command = str(uuid.uuid4())
                    joiner_receipt = await joiner._client.prompt(
                        "and resend that to the team",
                        command_id=joiner_command,
                    )
                    assert joiner_receipt == SPOOL_RECEIPT_PROMPT, joiner_receipt
                finally:
                    await joiner.dispose()

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
                # AND THE OWNER'S MESSAGE RAN IN THE SUCCESSOR, as their own
                # turn: the row carries the command id this viewer sent, so the
                # message that was queued IS the message that ran — once (the
                # durable index answers a twice-spooled row), and as a plain user
                # message rather than a peer card (`inbox.SOURCE_USER`).
                ours = [
                    entry
                    for entry in Transcript(config / "sessions" / session_id).entries()
                    if "run the report while the build moves" in str(entry.payload)
                ]
                assert ours, "the queued message never ran in the successor"
                assert [entry.id for entry in ours] == [queued_command], [
                    entry.id for entry in ours
                ]
                assert ours[0].payload.get("kind") == "message", ours[0].payload
                assert ours[0].payload.get("role") == "user", ours[0].payload
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


def _link_or_copy(source: str, destination: str, *, follow_symlinks: bool = True) -> str:
    """Hardlink the install's files when the filesystem allows it, copy otherwise.

    A HARD LINK IS THE POINT ON A NORMAL HOST: ~2 000 files, cheap, and the copy
    is REAL paths (a symlink would resolve back to the worktree, so removing the
    copy would remove nothing). A CROSS-DEVICE TEMP ROOT IS THE ONE CASE IT
    CANNOT SERVE: the GitHub ubuntu runner keeps ``/tmp`` on a different
    filesystem from the checkout, so ``os.link`` answers ``EXDEV``. Fall back to
    ``shutil.copy2`` there; the install needs the FILES, not the link, and this
    test asserts a deleted package tree, not link counts.
    """
    try:
        os.link(source, destination, follow_symlinks=follow_symlinks)
    except OSError:
        return shutil.copy2(source, destination, follow_symlinks=follow_symlinks)
    return destination


def _noneditable_install(tmp_path: Path, *, stamp: str) -> tuple[Path, Path]:
    """A throwaway uv-tool-shaped, non-editable install. ``(prefix, tree)``.

    No network and no build backend: the package is HARDLINKED out of the tree
    under test where the filesystem allows it (cheap for ~2 000 files, and REAL
    paths — a symlink would resolve back to the worktree, so removing the copy
    would remove nothing), copied on a cross-device temp root, a hand-written
    ``dist-info`` is what makes ``importlib.metadata`` see a real non-editable
    distribution, and the dependencies are reached through one
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
    shutil.copytree(package, tree, copy_function=_link_or_copy)

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


def _completion_attention_report(
    config_dir: Path, session_id: str, stderr_path: Path | None = None
) -> str:
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
    # Appended to BOTH branches: the completion rows alone read "complete,
    # complete" on the CI failure while the turn's bash result was a synthetic
    # abort — see ``tests/e2e/armed_probe_diagnostics.py`` for why and when to
    # remove it. Its only caller is the armed-probe cell.
    extra = armed_probe_diagnostics.evidence(
        config_dir / "sessions" / session_id / "transcript.jsonl", stderr_path
    )
    killed = [
        row
        for row in rows
        if row.get("kind") == "error" and "No module named" in str(row.get("reason", ""))
    ]
    if killed:
        return (
            "THE DELETED TREE KILLED THE TURN, NOT A LATE DRAIN — the child's own "
            f"site-packages is what a function-local import in its turn prelude "
            f"reaches for, and this cell removed it: "
            f"{armed_probe_diagnostics.redacted_rows(killed)!r}" + extra
        )
    return (
        f"the turn's completion rows so far: {armed_probe_diagnostics.redacted_rows(rows)!r}"
        + extra
    )


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

    # The child's stderr goes to a test-owned FILE instead of DEVNULL so a red
    # cell can quote its tail (the CI failure left nothing to read). Neither is
    # a tty, so the child sees the same kind of stream; the parent's handle is
    # closed right after the spawn and only read back in a failure message.
    stderr_path = tmp_path / "armed-child-stderr.log"
    stderr_file = stderr_path.open("wb")
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
        stderr=stderr_file,
        start_new_session=True,
    )
    stderr_file.close()
    transcript = config / "sessions" / session_id / "transcript.jsonl"
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

        await viewer.prompt(f"please [bash:{ARMED_HOLD_S}]")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = getattr(viewer, "frontend_state", None)
            if state is not None and getattr(state, "streaming", False):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("the runtime never started the turn")

        # WAIT UNTIL THE BASH TOOL IS RUNNING, THEN DELETE -- ``streaming`` is
        # not that moment. It flips when the turn is ADMITTED; the tool call
        # that this cell parks the turn in is dispatched later (a provider
        # request, then the guardrail check, then the spawn), and each of those
        # touches modules on disk. Delete in that gap and the dispatch fails in
        # microseconds -- "aborted", or "BLOCKED by guardrail ... the safety check
        # failed" -- the turn completes with no tool ever having run, and the
        # latch assertion below reports "completion rows: complete, complete".
        # That was CI's failure (runs 43082470360, 43135142485): the gap is
        # ~ms on an idle box and stretches with runner load, while the old
        # cell's rmtree was timed against ``streaming`` alone. A live
        # ``sleep`` under THIS runtime is the observable that the dispatch is
        # behind us, so the deletion now lands on a turn that is genuinely
        # parked in its tool, which is the situation the cell claims to test.
        # (``AttachedSession`` publishes no per-tool state to wait on instead.)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _parked_sleep_pids(pid, ARMED_HOLD_S):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(
                "the turn never reached its bash tool: "
                + _completion_attention_report(config, session_id, stderr_path)
            )

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
        assert drain_lines, f"never drained:\n{chr(10).join(_runtime_log_lines()[-30:])}" + (
            armed_probe_diagnostics.evidence(transcript, stderr_path)
        )
        assert "loaded module tree is gone" in drain_lines[0], drain_lines[0]
        assert getattr(
            viewer.frontend_state, "streaming", False
        ), "the latch must land while the turn is STILL running — " + _completion_attention_report(
            config, session_id, stderr_path
        )

        # THE TURN IS HELD OPEN UNTIL HERE, NOT FOR A FIXED TIME. The tool parks
        # for ARMED_HOLD_S (the mock's cap), far past the latch, so "the latch
        # landed while the turn was still running" cannot be lost to a slow
        # probe tick; the cell ends the park itself now that the latch is
        # observed and the assertion above has passed. Without it a loaded
        # runner could finish a fixed 12 s turn before a 5 s probe fired twice.
        assert _release_parked_sleep(pid, ARMED_HOLD_S), (
            "the parked bash tool vanished before the cell could release it: "
            + _completion_attention_report(config, session_id, stderr_path)
        )

        # (b) It EXITS — the Q-1 pin. On the wedged head this loop timed out with
        # the process still resident, holding the session's lease.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and _alive(child):
            await asyncio.sleep(0.2)
        assert not _alive(child), (
            "the drain latched and the runtime NEVER exited:\n"
            + chr(10).join(_lines_for_pid(pid)[-10:])
            + armed_probe_diagnostics.evidence(transcript, stderr_path)
        )
        assert child.returncode == 0, f"the runtime exited {child.returncode}, not cleanly" + (
            armed_probe_diagnostics.evidence(transcript, stderr_path)
        )

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
        assert [row for row in attention if row.get("kind") == "error"] == [], (
            "the drain cut the turn off:\n"
            + json.dumps(armed_probe_diagnostics.redacted_rows(attention), indent=2)
            + armed_probe_diagnostics.evidence(transcript, stderr_path)
        )
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
