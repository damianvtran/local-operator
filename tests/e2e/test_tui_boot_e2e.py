"""The assembled ``lop`` must finish loading its resources and STAY ALIVE on a real terminal.

The failure this exists to catch
--------------------------------

``0.54.37`` shipped to production and died a few seconds into an ordinary
session: the operator's log shows the MCP servers connecting, the session
runtime coming up, and then

``UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 4: invalid start byte``

raised from ``textual/drivers/linux_driver.py``'s input thread (the strict
incremental UTF-8 decoder it feeds ``os.read`` through) and the app gone. The
trigger is a terminal sending LEGACY X10 mouse reports —
``ESC [ M <button+32> <x+32> <y+32>``, three RAW bytes — in which any
coordinate at or past column 96 is ``>= 0x80`` and therefore not valid UTF-8.
A strict decoder raises on the driver's input thread, and the app dies with
the resources it had just finished loading.

The whole unit suite was green through it, and so was ``test_tui_e2e.py``:
that file drives ``OperatorApp`` through Textual's ``run_test()`` pilot, which
paints into an in-memory compositor and **never reads a terminal**. Nothing in
the suite exercised the one path that broke — real bytes arriving on a real
stdin — so nothing could notice. This file is the other end of that funnel:
not a widget under test, the product's own console script, in a real pty.

What is asserted, and why each assertion is here
------------------------------------------------

1. **Every resource the boot path pulls in finishes loading.** Two of them,
   asserted on evidence the product itself emits: the session RUNTIME
   (a ``live`` record in the runtime registry — the API ``/resume`` and the
   sidebar read) and the MCP SERVERS (tools in the runtime's own
   ``mcp_cache.db``, which the manager writes only after a connect completes
   and ``tools/list`` returns, plus the spawned peer still alive and a runtime
   log free of ``failed to connect`` — a failure logs a WARNING and leaves no
   process, measured).
2. **The app is interactive before the failure class is exercised.** A marker
   typed into the pty must appear in the pty's output: that requires the
   composer to be mounted, focused and servicing the input thread.
3. **The legacy X10 byte sequence is delivered to that live input path**,
   byte for byte as a terminal sends it. This is the failure this class of
   test missed; a boot test that never touches the input path cannot catch it.
4. **The app is still alive and running after a settling period.** The
   process has not exited, the runtime and the MCP peer are still up, the log
   holds no traceback and no decode error, and a keystroke still reaches the
   app and still makes it repaint — which is the assertion that the input
   thread specifically (the thread that died in production) is alive.
   Deliberately NOT a composer echo at that point: the legacy report is a
   mouse event, and where a delivered click leaves focus is a fact about the
   pointer's position rather than about liveness, so the probe is an
   App-level binding that fires whichever widget is focused.
   Liveness over time is the point: a frozen or half-dead app has perfectly
   correct state, it just never answers again.

On today's pre-fix tree this test fails on its own evidence: the injected
bytes kill the app within a second, the pty closes, and the log carries the
exact ``UnicodeDecodeError`` above. That is recorded in the PR body.

What this deliberately does NOT cover
-------------------------------------

The physical terminal emulator in the operator's report could not be
identified from the logs, so this test does not drive one. It delivers the
exact byte sequence such a terminal sends — the operator's own probes
established it, and it reproduces the identical log line against the shipped
build — rather than claiming emulator-level coverage it does not have.

Bounding and isolation
----------------------

Every step that can hang runs inside :func:`tests.e2e.watchdog.bounded`; read
that module before changing a timeout, because the failure mode this stage
guards (a thread parked in a syscall) defeats ``asyncio.wait_for``, thread
watchdogs and signal-based timeouts alike, and the C-level ``faulthandler``
timer is the only bound that survives it. The stage runs ``-n0`` for the same
reason: a fired watchdog exits the process, which under xdist would kill a
worker carrying unrelated tests.

The child gets an isolated ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR``, and the
``CMUX_*``/``LOP_*``/``HERDR_*`` families are stripped from its environment:
an inherited workspace id has addressed the operator's live window before, and
an inherited ``LOP_MOBILE_CHILD_*``/``LOP_RUNTIME_*`` makes a child adopt a
session that is not this test's. Nothing here signals a process it did not
spawn, and the config dir it writes is the test's own ``tmp_path``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

from tests.e2e.harness import NO_NOTIFY_ENV
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: Terminal geometry for the pty, in (columns, rows). Wide enough that a legacy
#: X10 coordinate in the second half of the screen — where the encoder runs out
#: of ASCII, which is the entire failure — is an ordinary pointer position, and
#: fixed here rather than inherited from whatever terminal pytest was started
#: from. 200x50 matches the operator's capture probes.
SCREEN = (200, 50)

#: The byte sequence a terminal in legacy X10 mouse mode (DECSET 1000, no 1006)
#: sends for two pointer events. RAW bytes, not a text protocol: each report is
#: ``ESC [ M`` then three bytes, and every field is ``value + 32`` — so button 0
#: is ``0x20``, x=96 is ``0x80``, y=44 is ``0x4C``, and a second event at x=215
#: is ``0xF7``. 0x80 and 0xF7 are not valid UTF-8, which is the crash.
X10_REPORTS = b"\x1b[M\x20\x80\x4c" + b"\x1b[M\x20\xf7\x4c"

#: The MCP peer: a real stdio server, the same fixture the desktop e2e tests
#: own through ``McpManager``. Copied into this test's config dir before use so
#: the spawned command line carries THIS test's path — a stray process from a
#: sibling session can then never satisfy the assertion below.
MCP_FIXTURE_SOURCE = Path(__file__).with_name("desktop_mcp_fixture.py")
MCP_FIXTURE_COPY_NAME = "boot_e2e_mcp_fixture.py"
MCP_SERVER_NAME = "boot-e2e-fixture"

#: Environment families a spawned TUI (and the runtime it spawns with
#: ``dict(os.environ)``) must never see. ``CMUX_*`` has renamed the operator's
#: real cmux workspaces before; ``LOP_*`` makes a child adopt a provider,
#: model, session or deferral that is not this test's; ``HERDR_*`` is the same
#: hazard for the other terminal integration.
STRIPPED_ENV_PREFIXES = ("CMUX_", "LOP_", "HERDR_")

#: Bounds, in wall-clock seconds, and why they are this loose. Measured healthy
#: costs: the frame paints in ~4 s, the runtime publishes its record ~2 s after
#: it starts, and the MCP peer's tools land in the cache ~2 s after the peer
#: spawns — an order of magnitude below these. They exist to catch a HANG, not to
#: police performance on a shared runner that is already busy: the same boot was
#: observed at ~11 s with a load average of 166, and a bound that tight would
#: report that as a product failure. The watchdog below is what turns a genuine
#: hang into a stack dump instead of a worker that sits forever.
BOOT_BOUND_S = 120.0
POST_INJECTION_BOUND_S = 60.0

#: How long the app is left running after the legacy bytes are delivered, and
#: how long a typed marker is given to appear. The pre-fix app dies within
#: ~0.5 s of the injection, so the settle window is about giving the fixed path
#: long enough that "still alive" cannot be an artefact of not having looked.
SETTLE_S = 6.0
ECHO_TIMEOUT_S = 20.0

#: Marker text typed into the composer. Short and alphanumeric so it cannot be
#: broken across line wrapping in the painted frame, and distinctive enough
#: that a match in the pty stream is this test's typing rather than chrome.
MARKER_PRE = "zqmarkpre"

#: The keystroke used to prove the input path still SERVICING input once the
#: legacy report has been delivered. ``ctrl+b`` is the sidebar toggle — an
#: App-level binding, so it fires whichever widget holds focus, which is the
#: property that makes it a liveness probe: see the assertion that uses it.
INPUT_PROBE = b"\x02"

#: How long the frame is watched for movement of its own, and then for the
#: probe's response, and how much response counts as "the app serviced it". The
#: floor is two orders of magnitude below the measured repaint (~100 kB) and
#: two orders above the measured idle movement (0 bytes), so it separates the
#: two without being a tuning knob.
IDLE_BASELINE_S = 2.0
INPUT_RESPONSE_S = 2.0
INPUT_RESPONSE_MIN_BYTES = 4096


class _Terminal:
    """A drained pty master.

    Draining is not optional: the TUI repaints continuously and a full pty
    buffer would block the process under test, which would make a wedged app
    look identical to a dead one. Every read is non-blocking with a timeout,
    and a master whose slave has gone reads as closed rather than raising.
    """

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd
        self.output = bytearray()

    def pump(self, seconds: float) -> None:
        """Read whatever arrives for ``seconds``."""
        self.pump_for(seconds)

    def pump_for(self, seconds: float) -> int:
        """Read for ``seconds`` and return how many bytes arrived.

        The count is what turns "the app is still alive" into "the app still
        SERVICING input": a keystroke it handles repaints, and a repaint is
        bytes on the wire, while a dead input path leaves the frame exactly as
        it was.
        """
        before = len(self.output)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._fd is None:
                return len(self.output) - before
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
            except OSError:
                self._fd = None
                return len(self.output) - before
            if not ready:
                continue
            self.read_ready()
        return len(self.output) - before

    def read_ready(self) -> None:
        """One non-blocking read of whatever is pending."""
        if self._fd is None:
            return
        try:
            chunk = os.read(self._fd, 65536)
        except OSError:
            self._fd = None
            return
        if not chunk:
            self._fd = None
            return
        self.output.extend(chunk)

    def write(self, data: bytes) -> None:
        """Send bytes as the terminal would — keystrokes or an escape sequence."""
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.write(self._fd, data)

    def close(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    def tail(self, limit: int = 1500) -> str:
        return bytes(self.output[-limit:]).decode("utf-8", errors="replace")

    def type_until_echoed(self, marker: bytes, timeout: float) -> bool:
        """Type ``marker`` until it appears, retrying while the app comes up.

        A keystroke written before the composer has taken focus is DROPPED — a
        settled precondition, not the behaviour under test — so this retries
        rather than failing on a first attempt that landed too early. The
        distinction that matters is preserved: a genuinely dead input path never
        echoes, exhausts the budget, and fails.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if marker in self.output:
                return True
            self.write(marker)
            self.pump(0.5)
        return marker in self.output


def _child_env(config_dir: Path, home: Path) -> dict[str, str]:
    """The environment a real ``lop`` gets from this test's terminal.

    Rebuilt by stripping the families in :data:`STRIPPED_ENV_PREFIXES` rather
    than handed ``dict(os.environ)`` (see the module docstring), then given the
    values this cell means to set. ``LOCAL_OPERATOR_DEBUG`` is dropped so the
    app takes the ordinary user path, not the verbose one.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(STRIPPED_ENV_PREFIXES)}
    env.update(
        {
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "TERM": "xterm-256color",
            "COLUMNS": str(SCREEN[0]),
            "LINES": str(SCREEN[1]),
            # The same two side channels the headless stage pins: a notice or a
            # terminal-title write from a test is attention stolen from a
            # machine running dozens of concurrent sessions. Both live on the
            # shared harness constant so this file and the other child builders
            # cannot disagree about which switches they are.
            **NO_NOTIFY_ENV,
            "LOCAL_OPERATOR_NO_TERMINAL_TITLE": "1",
        }
    )
    env.pop("NO_COLOR", None)
    env.pop("LOCAL_OPERATOR_DEBUG", None)
    return env


def _stage_config(config_dir: Path, home: Path) -> Path:
    """Write the config the app boots on, and return the MCP fixture's path.

    A provider and a model must be configured or the viewer never engages a
    runtime (``OperatorApp._runtime_can_start``) and the "resources finished
    loading" half of this test has nothing to load. The base URL is a loopback
    port nothing listens on, deliberately: the boot path does not call the
    provider (measured — no request, no error in the log), so standing a stub
    HTTP server up would add a fixture to babysit without buying coverage.
    What the value has to do is resolve a provider/model pair.

    The MCP config is a REAL stdio peer, because MCP servers coming up is what
    the operator's crash log shows the app doing when it died. It is copied
    into ``config_dir`` first, so the spawned command line carries this test's
    own path (see :data:`MCP_FIXTURE_SOURCE`).
    """
    from local_operator.config import ConfigManager

    home.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir).update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "boot-e2e-fixture",
            "providers": {
                "openai-compatible": {
                    "base_url": "http://127.0.0.1:9/v1",
                    "models": {
                        "boot-e2e-fixture": {
                            "context_window": 100000,
                            "supports_tools": True,
                        }
                    },
                }
            },
        }
    )
    fixture = config_dir / MCP_FIXTURE_COPY_NAME
    shutil.copyfile(MCP_FIXTURE_SOURCE, fixture)
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    MCP_SERVER_NAME: {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(fixture)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return fixture


def _spawn_tui(config_dir: Path, home: Path) -> tuple[int, int]:
    """The real TUI, in a real pty, launched the way a user launches it.

    ``pty.fork`` makes the child a session leader whose controlling terminal is
    the slave — the object a terminal emulator gives the process it hosts, and
    the thing that decides whether stdin is a tty at all. The console script is
    the product's own entry point resolved from this venv, so what runs is the
    checkout under test rather than a re-implementation of it.
    """
    script = Path(sys.executable).with_name("local-operator")
    assert script.is_file(), f"the venv console script is missing: {script}"
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover — the forked child execs immediately
        # A process started from a non-interactive shell inherits SIGINT as
        # SIG_IGN and exec preserves it (the precedent is
        # tests/unit/providers/test_login_cancel_cli.py), so without this the
        # app would ignore the signals this test cleans up with.
        for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        os.execve(str(script), [str(script)], _child_env(config_dir, home))
        os._exit(127)  # pragma: no cover
    # Pin the geometry on the pty itself rather than trusting the terminal
    # pytest happens to be running in: a piped or zero-sized parent would
    # otherwise compose a frame at a size these assertions did not choose.
    with contextlib.suppress(OSError):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", SCREEN[1], SCREEN[0], 0, 0))
    os.set_blocking(fd, False)
    return pid, fd


def _child_exit_status(pid: int) -> int | None:
    """The child's exit status, or ``None`` while it is still running."""
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # Already reaped elsewhere; a reaped child is a dead child.
        return 0
    return status if done else None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_runtime(config_dir: Path) -> tuple[str, int] | None:
    """The live runtime record this boot engaged, as ``(session_id, pid)``.

    The registry is the product's own API for "a runtime is up": a record is
    published only after the runtime has bound its control socket, and its
    ``live`` state means the heartbeat is current. That is a stronger readiness
    gate than anything the pty can be asked, because a TUI that has merely
    painted has no record.
    """
    from local_operator.session.runtime import registry

    for record, state in registry.scan(config_dir):
        if state == "live" and _is_alive(record.pid):
            return record.session_id, record.pid
    return None


def _mcp_peer_pids(fixture: Path) -> list[int]:
    """PIDs whose command line names THIS test's MCP fixture copy.

    The path is under the test's ``tmp_path``, so a process answering here was
    spawned by a config only this test wrote — the manager spawns a stdio
    server on connect and leaves nothing behind when the connect fails
    (measured: a bad command logs a WARNING and the process never appears).
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", str(fixture)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


def _logs(config_dir: Path) -> str:
    """Every log this boot wrote, concatenated for the clean-log assertion."""
    directory = config_dir / "logs"
    if not directory.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(directory.glob("*.log")):
        with contextlib.suppress(OSError):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


#: What the failure looks like wherever it lands, in the operator's own words:
#: the raised exception's name and the codec's message. Both are searched so a
#: traceback that rendered the exception without its argument still matches.
_CRASH_MARKERS = ("UnicodeDecodeError", "can't decode byte")


def _decode_failure_evidence(config_dir: Path, terminal: _Terminal) -> str:
    """The first decode-failure line the app left behind, or an empty string.

    Only ever called to build a FAILURE message. The pre-fix app dies so fast
    that the death is the symptom a reader sees first, and the production crash
    is only recognisable by this line — so the assertion that reports the death
    carries the line with it rather than making the reader re-run the test to
    find out which failure it was.
    """
    for where, text in (("log", _logs(config_dir)), ("terminal", terminal.tail(20000))):
        for line in text.splitlines():
            if any(marker in line for marker in _CRASH_MARKERS):
                return f"{where}: {line.strip()[:200]}"
    return ""


def _mcp_tools_cached(config_dir: Path, server: str) -> int:
    """How many tools the runtime's MCP tool cache holds for ``server``, or 0.

    ``<config_dir>/mcp_cache.db`` is the product's own durable record of a
    SUCCESSFUL ``tools/list`` — the manager writes a row for a server only
    after the connect completed and its tools were discovered. That is the
    cross-process signal this test wants for "the MCP servers finished
    loading": unlike the spawned peer (which appears before the handshake) it
    cannot be satisfied by a connect still in flight, and unlike the status
    bar's count it does not depend on the front end having caught up.
    """
    import sqlite3

    path = config_dir / "mcp_cache.db"
    if not path.is_file():
        return 0
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute(
                "SELECT tools_json FROM mcp_tool_cache WHERE server = ?", (server,)
            ).fetchone()
    except sqlite3.Error:
        # Best-effort by contract on the product's side too: a locked or
        # half-written database means "not loaded yet", never a broken test.
        return 0
    if not row:
        return 0
    try:
        return len(json.loads(row[0]))
    except (TypeError, ValueError):
        return 0


def _wait_for_resources(
    pid: int,
    terminal: _Terminal,
    config_dir: Path,
    fixture: Path,
    deadline: float,
) -> tuple[int, int]:
    """Block until the runtime AND the MCP peers have finished loading.

    Waiting on published state rather than sleeping "long enough" is what makes
    a resource that never loaded a clear failure instead of a slow one. Three
    signals, each the product's own:

    * a ``live`` runtime record — the runtime has bound its control socket and
      is heartbeating;
    * tools in the MCP tool cache — the connect round has SETTLED with the
      peer's tools discovered, which the spawned process alone does NOT prove
      (it appears before the handshake finishes);
    * the peer process itself, still alive.

    A run that fails says which signal was missing and prints the terminal and
    log tails, so a reader does not have to re-run it to find out.
    """
    while time.monotonic() < deadline:
        terminal.pump(0.25)
        if _child_exit_status(pid) is not None:
            raise AssertionError(
                "the TUI exited before it finished loading its resources; "
                f"pty tail:\n{terminal.tail()}\nlog tail:\n{_logs(config_dir)[-2000:]}"
            )
        failures = [
            line
            for line in _logs(config_dir).splitlines()
            if f"MCP server {MCP_SERVER_NAME!r} failed to connect" in line
        ]
        if failures:
            raise AssertionError(
                "the MCP peer never loaded: " + failures[0] + f"\npty tail:\n{terminal.tail()}"
            )
        runtime = _live_runtime(config_dir)
        peers = _mcp_peer_pids(fixture)
        if runtime is not None and peers and _mcp_tools_cached(config_dir, MCP_SERVER_NAME):
            return runtime[1], peers[0]
    runtime = _live_runtime(config_dir)
    raise AssertionError(
        "the boot never finished loading its resources: "
        f"live runtime={'yes' if runtime else 'no'}, "
        f"mcp peer pids={_mcp_peer_pids(fixture)}, "
        f"mcp tools cached={_mcp_tools_cached(config_dir, MCP_SERVER_NAME)}; "
        f"pty tail:\n{terminal.tail()}\nlog tail:\n{_logs(config_dir)[-2000:]}"
    )


def _stop_child(pid: int | None, terminal: _Terminal, *, grace_s: float = 5.0) -> None:
    """Stop the TUI this test spawned, politely and then not.

    Only ever called with a pid this test forked: it reaps, so a stale zombie
    cannot outlive the test into pytest's own child handling.
    """
    if pid is None or _child_exit_status(pid) is not None:
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        terminal.pump(0.2)
        if _child_exit_status(pid) is not None:
            return
        time.sleep(0.1)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, 0)


def _stop_other(pid: int | None, *, grace_s: float = 5.0) -> None:
    """Stop a process this test did not fork (the runtime, an MCP peer).

    No ``waitpid``: it is not our child, so only a liveness poll can tell us
    whether the signal landed. Everything signalled here was discovered through
    THIS test's config dir or command line, never by name at large.
    """
    if pid is None or not _is_alive(pid):
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and _is_alive(pid):
        time.sleep(0.1)
    if _is_alive(pid):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_the_assembled_tui_boots_loads_its_resources_and_survives_legacy_terminal_input(
    headless_tui_env: Path,
    tmp_path: Path,
) -> None:
    """Boot, load, take real legacy mouse bytes on stdin, and still be running.

    See the module docstring for what each phase asserts and why. The one line
    to keep in mind while reading: the app that shipped in 0.54.37 satisfied
    every state assertion in this file right up to the moment those three raw
    bytes arrived.
    """
    config_dir = headless_tui_env
    home = tmp_path / "home"
    fixture = _stage_config(config_dir, home)

    pid: int | None = None
    terminal: _Terminal | None = None
    runtime_pid: int | None = None
    try:
        pid, fd = _spawn_tui(config_dir, home)
        terminal = _Terminal(fd)

        # ---- Where the funnel comes up: frame, runtime, MCP peers ----------
        with bounded(BOOT_BOUND_S, "TUI boot and resource load"):
            runtime_pid, _ = _wait_for_resources(
                pid,
                terminal,
                config_dir,
                fixture,
                time.monotonic() + BOOT_BOUND_S - 5.0,
            )

            # ---- Interactive before the input path is stressed -------------
            assert terminal.type_until_echoed(MARKER_PRE.encode(), ECHO_TIMEOUT_S), (
                "typed keys never reached the composer, so the app painted a frame "
                f"but is not servicing input; pty tail:\n{terminal.tail()}"
            )

        # ---- The failure class: legacy X10 reports on the live input path ---
        with bounded(POST_INJECTION_BOUND_S, "legacy X10 input and post-injection liveness"):
            terminal.write(X10_REPORTS)
            terminal.pump(SETTLE_S)

            exit_status = _child_exit_status(pid)
            assert exit_status is None, (
                "the app died after receiving a legacy X10 mouse report "
                f"(status {exit_status}); decode failure on record: "
                f"{_decode_failure_evidence(config_dir, terminal) or 'none found'}; "
                f"pty tail:\n{terminal.tail()}"
            )
            assert _is_alive(runtime_pid), (
                "the session runtime did not survive the terminal input; "
                f"pty tail:\n{terminal.tail()}"
            )
            assert _mcp_peer_pids(fixture), (
                "the MCP peer is gone after the terminal input, so the resources "
                f"that finished loading did not stay loaded; pty tail:\n{terminal.tail()}"
            )

            # ---- The input path is still SERVICING input --------------------
            #
            # Not a composer echo, deliberately. The legacy report is a mouse
            # event at (96, 44): on a terminal where it is handled, a click is a
            # click, and Textual moves focus to whatever is under it — so
            # typing into the composer afterwards would test where the pointer
            # landed, not whether the input thread is alive. What is asserted
            # instead is that a keystroke still reaches the app and the app
            # still repaints in response, measured against how much the frame
            # moves on its own (an idle app is quiet: measured 0 bytes over 2 s,
            # against ~100 kB for the repaint below).
            idle_bytes = terminal.pump_for(IDLE_BASELINE_S)
            terminal.write(INPUT_PROBE)
            response_bytes = terminal.pump_for(INPUT_RESPONSE_S) - idle_bytes
            assert response_bytes >= INPUT_RESPONSE_MIN_BYTES, (
                "the app survived but stopped servicing input after the legacy mouse "
                f"report: a {INPUT_PROBE!r} keystroke repainted {response_bytes} bytes "
                f"over {INPUT_RESPONSE_S:g}s against an idle baseline of "
                f"{idle_bytes} bytes — the input thread is gone; "
                f"pty tail:\n{terminal.tail()}"
            )

            # A crash a surviving process can still hide, in both places it can
            # land: Textual renders the traceback into the frame, and the TUI's
            # own file logging is where the operator read it in production (the
            # sink is fd 2, which the TUI's logging guard points at the log
            # file, so the driver's own panic report lands there).
            for where, text in (
                ("the terminal", bytes(terminal.output)),
                ("the log", _logs(config_dir).encode()),
            ):
                for marker in _CRASH_MARKERS:
                    assert marker.encode() not in text, (
                        f"{where} carries the decode failure that killed 0.54.37 "
                        f"({marker}); "
                        f"tail:\n{text[-3000:].decode('utf-8', errors='replace')}"
                    )
                assert b"Traceback" not in text, (
                    f"{where} carries a traceback; "
                    f"tail:\n{text[-3000:].decode('utf-8', errors='replace')}"
                )
    finally:
        if terminal is not None:
            _stop_child(pid, terminal)
            terminal.close()
        # The runtime idle-exits on its own once its viewer is gone, but it is
        # not this process's child and nothing guarantees the timing: signal
        # it, then its peer, so a leaked process cannot outlive the test.
        _stop_other(runtime_pid)
        for peer in _mcp_peer_pids(fixture):
            _stop_other(peer)
