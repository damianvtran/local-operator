"""Closing the terminal must not cancel a session's work — on a REAL terminal.

The operator's report, verbatim: "I noticed an issue in the TUI where, when I
closed ghostty a bunch of sessions got cancelled at once ... it's unexpected
since all session runtimes are supposed to be detached from interfaces and they
should be able to run in the background when the parent process is quit
(ghostty in this case)".

This is the cross-process form of that report, in the shape a terminal window
actually produces. A REAL ``lop`` TUI is hosted in a **pty** (its own session,
the slave as its controlling terminal, exactly as a terminal emulator gives
it), it engages a REAL detached runtime, a REAL turn is in flight in the real
``bash`` tool, and the terminal then dies in one of the three ways a closed
window does:

* ``ctrl-d`` — EOF on the interface's stdin, the polite way out;
* ``pty-close`` — the master is closed, which the kernel turns into a **real
  ``SIGHUP`` to the pty's foreground process group** (measured with a 15-line
  probe: a handler in the pty child records the HUP, so this cell is the
  ghostty shape rather than a simulation of it);
* ``sigkill-group`` — ``SIGKILL`` to the interface's whole process group.

WHAT THIS PINS, AND WHAT IT DOES NOT. It pins the architecture end to end: an
interface's death cannot reach a session's work, because the runtime is its own
session and group (``launch._spawn_runtime``'s ``start_new_session``) and the
TUI is a viewer whose takeover factory raises by construction. It is therefore a
REGRESSION GUARD, not the cell that fails without this change's handler: the
runtime is spawned detached, so none of these three shapes delivers a ``SIGHUP``
to *it*, and today's tree already survives all three. The fail-today pin for the
handler is the unit cell in
``tests/unit/session/runtime/test_runtime_detachment.py``, which sends the HUP
directly. This file would go red the day a spawn loses its detachment, a
viewer's death starts cascading into the runtime, or a stop-ladder rung is
reached for an interface going away — which is precisely the invariant the
report was about.

Isolation: ``headless_tui_env`` redirects the config dir and the root conftest
redirects ``HOME``; the child's environment is rebuilt here with EVERY
``CMUX_*``/``LOP_*``/``HERDR_*`` variable stripped, because a runtime or TUI that
inherited a workspace id has addressed the operator's live window before (#648),
and an inherited ``LOP_MOBILE_CHILD_*`` makes a spawn adopt a session that is not
this test's. Nothing here signals anything it did not spawn. The per-config-root
secrets broker daemon the CLI's own boot starts idles out after
``IDLE_SHUTDOWN_S`` (30 min) exactly as it does for a user; it is keyed to this
test's config root and cannot evict the operator's.
"""

from __future__ import annotations

import contextlib
import os
import pty
import select
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

#: Imported rather than spelled: the window a viewer allows an owner before it
#: concludes the owner is gone is the yardstick for "the runtime outlived the
#: interface's death". A copy of the number could drift from the product's.
from local_operator.session.attached import COLD_FALLBACK_S
from tests.e2e.test_cut_off_turns_e2e import CHILD_ENV_FAMILIES, _seed
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: The parked tool call, in seconds. Long enough that the terminal death lands
#: well inside it, short enough that the turn completes inside this test's
#: budget instead of being cut off by the runtime's own disposal at teardown.
PARK_S = 8

#: The three shapes a terminal window's death takes. See the module docstring.
DEATHS = ("ctrl-d", "pty-close", "sigkill-group")

#: ``attached.COLD_FALLBACK_S`` is the window in which a viewer concludes its
#: owner is gone. The runtime surviving past TWICE that is what says the
#: interface's death did not become the session's — a viewer-side conclusion
#: would have landed by then on every arm of the recovery ladder.
COLD_FALLBACK_WINDOWS = 2

#: Backstops, not performance assertions: the boot is ~5 s and the turn lands
#: a second or two after the 8 s sleep. They exist so a genuine hang fails the
#: run instead of blocking the worker forever (this suite has no
#: ``pytest-timeout``; ``bounded`` is what turns a hang into a stack dump).
BOOT_S = 90.0
LANDING_S = 90.0

#: How long the ``ctrl-d`` arm keeps asking the interface to quit. MEASURED, and
#: the reason this is a retry rather than a single keystroke: a Ctrl-D written
#: the instant the runtime first reports ``busy`` did NOT quit the app, while the
#: same keystroke 8 s later into the same turn did. The runtime's flag flips
#: inside the turn's own admission path, which is earlier than the composer has
#: finished repainting the submit — so the key is dropped. A person presses it
#: again; so does this, and the interface's death is asserted separately either
#: way, so a retry cannot make the runtime-side assertions pass vacuously.
QUIT_S = 20.0


def _tui_env(config_dir: Path) -> dict[str, str]:
    """The environment a real ``lop`` gets from this test's terminal.

    Built from a strip of the three inherited families (see the module
    docstring) rather than ``dict(os.environ)``, then given the values this cell
    means to set. ``LOP_SESSION_GRACE_S`` is pinned high so the runtime this
    test kills the interface on top of is still resident when the assertions
    run: the default 3 s drain is a residency policy, and a runtime that left
    on its own would make every "it survived" assertion a statement about the
    reaper.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(CHILD_ENV_FAMILIES)}
    env.update(
        {
            "HOME": os.environ.get("HOME", str(config_dir.parent)),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "TERM": "xterm-256color",
            "LOP_SESSION_GRACE_S": "600",
            # Defence in depth for the surface the operator is actually using:
            # a notice or a terminal-title write from a test is attention
            # stolen from a machine running fifty sessions.
            "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
            "LOCAL_OPERATOR_NO_TERMINAL_TITLE": "1",
        }
    )
    env.pop("NO_COLOR", None)
    return env


def _launch_tui(config_dir: Path, session_id: str) -> tuple[int, int]:
    """The real TUI, in a pty, resuming ``session_id``.

    ``pty.fork`` makes the child a session leader whose controlling terminal is
    the slave — the object a terminal emulator gives the process it hosts, and
    the thing closing a ghostty window tears down. The console script is the
    product's own entry point (``.venv/bin/local-operator``), so what runs here
    is the checkout under test rather than a re-implementation of it.
    """
    script = Path(sys.executable).with_name("local-operator")
    assert script.exists(), f"the venv console script is missing: {script}"
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover — the forked child execs immediately
        # A process started from a non-interactive shell inherits SIGINT as
        # SIG_IGN and exec preserves it, so without this the interface would
        # ignore the very signals this test delivers to it. The sibling
        # precedent is tests/unit/providers/test_login_cancel_cli.py.
        for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        os.execve(str(script), [str(script), "--resume", session_id], _tui_env(config_dir))
        os._exit(127)  # pragma: no cover
    return pid, fd


class _Pty:
    """A drained pty master.

    Draining is not optional: the TUI paints continuously, and a full pty
    buffer would block the very process under test — a wedged interface would
    then be indistinguishable from a dead one, which is the shape this test is
    about. Every read is non-blocking with a timeout, and a master that has gone
    away (the death shape) reads as closed rather than raising.
    """

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd
        self.output = bytearray()

    def drain(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._fd is None:
                return
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except OSError:
                self._fd = None
                return
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 65536)
            except OSError:
                self._fd = None
                return
            if not chunk:
                return
            self.output.extend(chunk)

    def write(self, data: bytes) -> None:
        # EIO here means the slave side is gone — i.e. the interface this test is
        # driving has just died, which is the state every caller is working
        # towards. Suppressed so a keystroke racing the death does not turn the
        # cell into an unrelated OSError; the death itself is asserted on the
        # process, not on this write.
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.write(self._fd, data)

    def close(self) -> None:
        """Close the master — the pty's hangup, in this file's own vocabulary."""
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def tail(self, limit: int = 1200) -> str:
        return bytes(self.output[-limit:]).decode("utf-8", errors="replace")


def _records(config_dir: Path, session_id: str) -> list[Any]:
    from local_operator.session.runtime import registry

    return [rec for rec, _state in registry.scan(config_dir) if rec.session_id == session_id]


def _wait_for_runtime(config_dir: Path, session_id: str, terminal: _Pty, *, timeout: float) -> Any:
    """The detached runtime the TUI engaged, once it has published its record.

    The record is the sound readiness gate for the *interface* too: it exists
    only after the runtime it spawned has bound its control socket, so it
    cannot be satisfied by a TUI that has merely painted.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        terminal.drain(0.2)
        found = _records(config_dir, session_id)
        if found:
            return found[0]
    raise AssertionError(
        f"the TUI never engaged a runtime for {session_id} in {timeout}s; "
        f"pty tail:\n{terminal.tail(4000)}"
    )


def _interface_settled(terminal: _Pty) -> None:
    """Wait until the TUI has stopped painting, i.e. has mounted the session.

    A precondition for typing, not an assertion: keystrokes written into a
    composer that does not exist yet are dropped, and the failure would look
    like "the turn never started". The turn itself is asserted on the runtime's
    own ``busy`` flag below, which is an event rather than a clock — so a
    dropped keystroke fails loudly and says so instead of being papered over.
    """
    quiet = 0.0
    deadline = time.monotonic() + 30.0
    while quiet < 2.0 and time.monotonic() < deadline:
        before = len(terminal.output)
        terminal.drain(0.3)
        quiet = quiet + 0.3 if len(terminal.output) == before else 0.0


def _turn_is_running(config_dir: Path, session_id: str, terminal: _Pty, *, timeout: float) -> None:
    """Block until the runtime reports the turn as busy — the turn gate.

    ``busy`` is published by the runtime over the very record this test already
    reads, so it is the product telling us the prompt was accepted and is being
    served. Waiting on it (rather than sleeping "long enough") is what makes a
    keystroke that never landed a clear failure instead of a slow one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        terminal.drain(0.2)
        for rec in _records(config_dir, session_id):
            if rec.busy:
                return
    raise AssertionError(
        f"the typed prompt never started a turn for {session_id} in {timeout}s; "
        f"pty tail:\n{terminal.tail(4000)}"
    )


def _transcript(directory: Path) -> str:
    path = directory / "transcript.jsonl"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _wait_for_landing(directory: Path, *, timeout: float) -> str:
    """The finished turn's text, or the last transcript read.

    The parked ``[bash:N]`` marker makes the mock's follow-up a known string, so
    "the turn landed" is the transcript carrying it — a durable fact written by
    the runtime after its tool returned, which is the only thing that proves the
    work outlived the interface.
    """
    deadline = time.monotonic() + timeout
    while True:
        text = _transcript(directory)
        if "from the mock provider" in text:
            return text
        if time.monotonic() >= deadline:
            return text
        time.sleep(0.25)


def _runtime_log(config_dir: Path) -> str:
    try:
        return (config_dir / "logs" / "mobile.log").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _interface_is_gone(pid: int, *, timeout: float) -> bool:
    """Whether the pty child has actually died — the death shape's precondition.

    Without this, "the runtime survived" would be consistent with "nothing died
    at all", which is the vacuous pass this assertion exists to prevent. A child
    already reaped by the runner raises ``ChildProcessError``, which is also a
    death; the pty's own EOF is the other half of the same fact and is read by
    the caller's drain.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            done, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True
        if done:
            return True
        time.sleep(0.2)
    return False


def _quit_via_key(terminal: _Pty, pid: int) -> None:
    """Ask the real TUI to quit with its own key, re-asking until it does.

    ``ctrl+d`` is the documented quit (``keymap.RESERVED_KEYS``: "ctrl+d quits
    — it cannot be bound"), so this is the interface's own exit rather than a
    signal: the difference matters because a graceful quit has its own
    teardown path, and THAT is the path the report suspected of cancelling
    sessions. See ``QUIT_S`` for why the key is re-sent.
    """
    deadline = time.monotonic() + QUIT_S
    while time.monotonic() < deadline:
        terminal.write(b"\x04")
        terminal.drain(0.5)
        if _interface_is_gone(pid, timeout=0.1):
            return
    # Not gone: the caller's own death assertion reports it with the pty tail.


def _reap_interface(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def _reap_runtime(pid: int, *, timeout: float = 20.0) -> None:
    """Stop the runtime THIS test spawned, and only that one.

    A synthetic session id plus the pid read from its own record: nothing else
    on the machine can match. SIGTERM first (the clean rung, so the test leaves
    the same logged exit a real stop produces), SIGKILL only if it lingers.
    """
    if not _alive(pid):
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("death", DEATHS)
def test_a_closed_terminal_does_not_cancel_the_turn(headless_tui_env: Path, death: str) -> None:
    """The report's shape, three ways, with a real turn parked in a real tool.

    Asserts, in order: the interface really died; the runtime was still alive
    when it did and is still alive a full recovery window later; the parked
    turn's result landed in the transcript; no completion for that turn reads
    as a cut-off; and the runtime never wrote an exit line — i.e. nothing about
    the interface's death ended its work.
    """
    from local_operator.session.attention import AttentionStore, conversation_identity

    config = headless_tui_env
    session_id = f"termclose{DEATHS.index(death)}1"
    directory = _seed(config, session_id)

    terminal: _Pty | None = None
    interface_pid: int | None = None
    runtime_pid: int | None = None
    try:
        with bounded(300, f"terminal close ({death})"):
            interface_pid, fd = _launch_tui(config, session_id)
            terminal = _Pty(fd)

            record = _wait_for_runtime(config, session_id, terminal, timeout=BOOT_S)
            runtime_pid = int(record.pid)
            # THE DETACHMENT CONTRACT, on the real process the real TUI spawned.
            # (`registry`'s ``detached`` field is about whether a front end is
            # ATTACHED — the opposite question, and here it is correctly False:
            # an interface was watching. What makes a teardown unable to reach
            # the work is the runtime's own session and group.)
            assert os.getsid(runtime_pid) == runtime_pid, (
                f"the runtime {runtime_pid} is not its own session leader "
                f"(sid={os.getsid(runtime_pid)}), so an interface's teardown could "
                "signal it"
            )
            assert os.getpgid(runtime_pid) == runtime_pid, (
                f"the runtime {runtime_pid} is not its own group leader "
                f"(pgid={os.getpgid(runtime_pid)})"
            )

            # Type the prompt into the real composer and wait for the runtime to
            # report the turn under way. The parked call is the real `bash`
            # tool: what is in flight when the terminal dies is actual work.
            _interface_settled(terminal)
            terminal.write(f"please [bash:{PARK_S}]\r".encode())
            _turn_is_running(config, session_id, terminal, timeout=LANDING_S)

            # THE TERMINAL DIES.
            if death == "ctrl-d":
                _quit_via_key(terminal, interface_pid)
            elif death == "pty-close":
                terminal.close()
            else:
                os.killpg(interface_pid, signal.SIGKILL)
            assert _interface_is_gone(interface_pid, timeout=30.0), (
                f"the interface ({death}) never died, so the runtime surviving proves "
                f"nothing; pty tail:\n{terminal.tail()}"
            )

            # (1) Still alive when the interface died...
            assert _alive(runtime_pid), (
                f"the runtime died with the interface ({death}); its own log tail:\n"
                f"{_runtime_log(config)[-2000:]}"
            )
            # ...and still alive a full recovery window later. A viewer-side
            # "owner lost" verdict lands at COLD_FALLBACK_S, so surviving twice
            # that says nothing on the interface's side concluded the session.
            settle = COLD_FALLBACK_WINDOWS * COLD_FALLBACK_S
            deadline = time.monotonic() + settle
            while time.monotonic() < deadline:
                terminal.drain(0.2)
                assert _alive(
                    runtime_pid
                ), f"the runtime died {settle:.0f}s after the interface ({death})"

            # (2) The work lands. This is the assertion the report is really
            # about: a detached runtime keeps its turn.
            text = _wait_for_landing(directory, timeout=LANDING_S)
            assert "from the mock provider" in text, (
                f"the parked turn never completed after the interface died ({death}); "
                f"transcript tail:\n{text[-2000:]}"
            )

            # (3) Nothing classified it as a cut-off. Read through the product's
            # own vocabulary rather than a list of strings copied here: if a
            # shutdown cause is ever added, this assertion covers it.
            from local_operator.incidents import is_cut_off_cause

            state = AttentionStore().state(conversation_identity(directory))
            assert not (
                state["kind"] == "error" and is_cut_off_cause(state["cause"])
            ), f"the turn was recorded as a cut-off ({death}): {state!r}"

            # (4) And the runtime never said it was leaving. The exit line is
            # what the report's mass cancellation would have produced; its
            # absence is the difference between "the work finished" and "the
            # work finished because nothing ended it".
            log = _runtime_log(config)
            assert f"exiting (SIGTERM, pid {runtime_pid}" not in log, log[-2000:]
            assert not any(
                line.startswith("session runtime: exiting (")
                for line in log.splitlines()
                if f"pid {runtime_pid}" in line
            ), f"the runtime exited during the test:\n{log[-2000:]}"
    finally:
        if terminal is not None:
            terminal.close()
        if interface_pid is not None:
            _reap_interface(interface_pid)
            # The pty child may still be finishing its own exit; nothing else
            # here waits on it, so a second non-blocking reap keeps the worker
            # free of a zombie it did not ask for.
            with contextlib.suppress(ChildProcessError):
                os.waitpid(interface_pid, os.WNOHANG)
        if runtime_pid is not None:
            _reap_runtime(runtime_pid)
