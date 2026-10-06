"""The stall bound must not be able to park a thread holding the GIL — and it no longer does.

THE FIELD SIGNATURE, verbatim. Two live runtimes on this host (``sample <pid> 3``;
builds 0.62.8 and 0.62.15, different work) show the same thing: the runtime's MAIN
(event-loop) thread parked inside the watchdog's own re-arm —

    task_wakeup_lock_held -> task_step -> ... -> cfunction_call
      -> faulthandler_dump_traceback_later -> cancel_dump_traceback_later
        -> PyThread_acquire_lock_timed -> _pthread_cond_wait -> __psynch_cvwait

— WHILE HOLDING THE GIL, with the thread named ``stall-watchdog-progress`` unable
to take it (``lock_PyThread_acquire_lock`` -> ``_PyParkingLot_Park`` ->
``_PyThreadState_Attach`` -> ``take_gil``). The session wedges, the leg that exists
to report the stall is stopped by the arming path, and the C timer can never finish
what it was doing, so no dump reaches disk.

WHY THE CALL CANNOT RETURN (CPython 3.14.3, ``Modules/faulthandler.c``):
``dump_traceback_later`` calls ``cancel_dump_traceback_later()`` first (``:806``),
which releases ``cancel_event`` and blocks on
``PyThread_acquire_lock(thread.running, 1)`` until the previous timer thread exits
(``:686``, ``:689``) with ``intr_flag=0`` — no ``PyEval_SaveThread``, so the caller
keeps the GIL. Between that cancel and the new timer being started (``:806`` to
``:821``) the process holds no armed timer at all.

THE FIX THESE CELLS PIN (design memo, option T1): no native timer call remains in
this module's production path. The deadline is recorded by ``_arm_timer``, the fire
is Python (``_fire``), and the dump is
``faulthandler.dump_traceback(file=..., all_threads=True)`` — the same thread walk
through the same writer, touching no timer state — taken outside ``_LOCK``. The
out-of-process leg is a ``SIGUSR1`` registration on the same handle.

WHAT THE CELLS BELOW PROVE, and how each can fail:

* P1 an ownership spy: across boot -> engage -> both planes' beats -> a held fire
  -> disarm, the module makes NO call into the timer API at all.
* P2 a child in which a dump is forcibly unable to finish: the module's arm, beat
  and disarm paths still return, and the process is still running Python afterwards.
  On the base ref the same child wedges (the memo's failing-first evidence).
* P3 a source pin: a parse of the module that fails if a native timer call
  reappears anywhere in its executable code.
* P4 the fire's dump is taken OUTSIDE ``_LOCK``, which is what keeps a
  hundreds-of-milliseconds stop-the-world from becoming the other plane's stall.
"""

from __future__ import annotations

import ast
import faulthandler
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime import stall_watchdog

#: The module under test, for the source pin.
MODULE_PATH = Path(stall_watchdog.__file__)

#: The two C entry points the deadlock runs through. Named once, because the pin and
#: the spy must agree on exactly which calls they are about.
RETIRED_CALLS = ("dump_traceback_later", "cancel_dump_traceback_later")

#: How long a cell waits for a publication from the code under test before it reports
#: a wedge rather than a slow host. A backstop, never the assertion.
PUBLISHED_S = 10.0

#: How long the fire search waits for the child's first dump. The child must BOOT
#: (import this tree, start its threads) before it arms its timer, and the fire is
#: then 0.05 s behind that, so the binding term is the boot, not the timer. Dataset:
#: a replication of the spawn under this fleet's load measured the first fire at
#: t+7.94 s (reviewer round 1), where the whole cell costs 0.26-0.44 s in CI (40
#: junit samples); the previous 10 s bound sat only 1.26x above that measurement.
#: 60 s is ~7.5x the measured worst and ~136x the healthy cell. What it stops
#: catching: a child that never arms its timer surfaces after 60 s rather than
#: 10 s, and one that DIES is named at once by the liveness arm inside the wait.
FIRE_S = 60.0

#: How long the child in P2 gets to reach its sentinel. Generous on purpose: the
#: child's own output is what says whether Python is still running.
CHILD_BOUND_S = 25.0

#: The socket buffer the never-drained dump channel is given. The dump of six
#: parked, 250-frame-deep threads is far larger, so the C thread's write cannot be
#: satisfied in one go — measured: with this buffer and no reader, ZERO bytes reach
#: the other end, i.e. the write does not even partly complete.
CHANNEL_SNDBUF = 1024


@pytest.fixture(autouse=True)
def _no_leaked_arm() -> Iterator[None]:
    """Module-level state is process-global: never let one cell arm the next."""
    stall_watchdog.disarm()
    yield
    stall_watchdog.disarm()


def _child_env(config_dir: Path, **extra: str) -> dict[str, str]:
    """A child environment that can only touch ``config_dir``.

    Every inherited ``CMUX_*``/``LOP_*``/``HERDR_*`` variable is stripped: this suite
    is routinely run from inside an operator session whose own values would otherwise
    be inherited, and ``LOCAL_OPERATOR_CONFIG_DIR`` alone is not enough (AGENTS.md,
    "Isolating a run").
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))}
    env["HOME"] = str(config_dir)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env.update(extra)
    return env


def _wait_for(predicate: Any, timeout: float = PUBLISHED_S) -> bool:
    """Wait (bounded) for a predicate on state the code under test writes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _OwnershipSpy:
    """Records which THREAD calls each C entry point. Never installs a real timer.

    Deliberately does NOT delegate: a real ``dump_traceback_later`` in a pytest
    worker would arm a C timer that ends the worker with ``exit=True``, and this spy
    exists to observe calls, not to survive them. What it proves is therefore
    "the module asked for a timer" (or did not) — the end-to-end proof that a *real*
    in-flight dump cannot wedge the arm path is the child in P2.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []
        self.registered: list[tuple[int, bool]] = []
        self.unregistered: list[int] = []

    def __getattr__(self, name: str) -> Any:
        # Only the two retired calls are interesting; everything else (``register``,
        # ``unregister``, ``dump_traceback``) is answered by a recorder so an unrelated
        # new call cannot silently become an AttributeError in a cell.
        def recorder(*args: Any, **kwargs: Any) -> None:
            if name in RETIRED_CALLS:
                self.calls.append((name, threading.get_ident(), threading.current_thread().name))
            elif name == "register":
                self.registered.append((int(args[0]), bool(kwargs.get("chain"))))
            elif name == "unregister":
                self.unregistered.append(int(args[0]))

        return recorder


class _DumpProbe:
    """A probe that answers, so the sampler runs and the fire path is reachable."""

    def __call__(self) -> tuple[object, bool]:
        return (0.0, True)


def test_no_call_into_the_timer_api_is_made_in_an_armed_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: boot, both planes' beats, a held fire and disarm make NO timer-API call.

    THE DECISION THIS PINS is "this module does not touch the C timer", and it is
    asserted as an EMPTY CALLER SET rather than as "not on the loop thread", because
    the field evidence is that the loop thread is where the park hurt while the
    underlying defect is the call itself: the C call parks holding the GIL, so a
    dedicated worker parks holding it too (measured: a child re-arming from a fresh
    thread stops its ticker at the call and never reaches its own ``MAIN-ALIVE``
    print). Empty is the only set that cannot wedge anything.

    WHY A SPY AND NOT A STOPWATCH: a timing bound here would be a bet on the fleet's
    load, and the property is a fact about which calls the module makes. AGENTS.md
    names thread identity and structural spies as the strongest form of "this work
    did not happen on that thread"; this is the degenerate case where the work must
    not happen at all.

    THE LIFETIME IS DRIVEN IN FULL, because a call can hide in any of its phases: the
    boot arm, ``engage`` moving the bound down, a ``beat`` from each plane, the
    executing-loop extension, a fire, the held-fire annotation and re-arm, and
    ``disarm``. The fire is driven on the HELD leg (``busy`` reports work in flight),
    which is the leg that annotates and keeps running — the fatal leg ends the process
    and is exercised by the child in P2.

    MUTATION THIS CELL CATCHES: put a ``faulthandler.dump_traceback_later`` or
    ``cancel_dump_traceback_later`` call back anywhere on this path -> red.
    """
    spy = _OwnershipSpy()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    assert (
        stall_watchdog.arm(
            seconds=1.0,
            directory=tmp_path,
            probe=_DumpProbe(),
            busy=lambda: True,  # work in flight: the fire annotates and stays
        )
        is True
    ), "the cell could not arm, so it drove no phase of the lifetime"

    stall_watchdog.engage()
    stall_watchdog.beat(stall_watchdog.WORKLOAD)
    stall_watchdog.beat(stall_watchdog.SERVING)
    # THE FIRE, on the held leg: the sampler fires at the deadline, dumps every thread
    # into this cell's own dump file, annotates and re-arms.
    fired = _wait_for(
        lambda: stall_watchdog._fired_count(stall_watchdog.dump_path(os.getpid(), tmp_path)) >= 1,
        timeout=PUBLISHED_S,
    )
    stall_watchdog.disarm()

    assert not spy.calls, (
        f"the module called into the C timer API during an armed lifetime: {spy.calls}. "
        f"Those calls park while holding the GIL when a dump is in flight, and the "
        f"second one is the cancel that waits for it — the deadlock both preserved "
        f"specimens are parked in"
    )
    assert fired, (
        "the bound never fired in this cell, so the fire phase of the lifetime was "
        "driven by nothing (the ownership assertion above is only as good as the "
        "phases actually exercised)"
    )
    assert spy.registered, "the out-of-process leg (SIGUSR1) was never registered at arm"
    assert spy.registered[0][1] is False, (
        f"the leg registered with chain={spy.registered[0][1]} (registered={spy.registered}); "
        f"chaining re-raises against the handler faulthandler SAVED, and on a real runtime "
        f"child that was measured as a segfault (rc=-11), so the leg must own its slot"
    )
    assert not spy.unregistered, (
        f"the module unregistered its signal leg ({spy.unregistered}); "
        f"faulthandler.unregister restores the disposition saved at arm time rather than "
        f"the live one, which is the same measured crash by another route"
    )


def test_the_fire_takes_its_dump_outside_the_watchdogs_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P4: the all-thread dump is taken with ``_LOCK`` FREE.

    WHY THIS IS ITS OWN PROPERTY: the gate is what every ``beat`` and every sample
    takes, and an all-thread dump measures in the hundreds of milliseconds on this
    fleet (200 threads x 300 frames, measured: see the PR). Taking it while holding
    the gate would hand the other plane's tick the very stall the artifact exists to
    report — the same class of defect as the deadlock, at a smaller scale.

    THE ASSERTION IS STRUCTURAL: the spy asks the gate whether it is free AT THE
    MOMENT the dump is taken, by trying a non-blocking acquire. A successful acquire
    proves the caller does not hold it; the recorded answer must be exactly that.

    MUTATION THIS CELL CATCHES: move ``_fire`` inside the ``with _LOCK:`` block in the
    sampler, or hold the gate across the dump for any other reason -> red.
    """
    gate_was_free: list[bool] = []
    real_dump = faulthandler.dump_traceback

    def spy_dump_traceback(*args: Any, **kwargs: Any) -> None:
        got = stall_watchdog._LOCK.acquire(blocking=False)
        gate_was_free.append(got)
        if got:
            stall_watchdog._LOCK.release()
        real_dump(*args, **kwargs)

    monkeypatch.setattr(faulthandler, "dump_traceback", spy_dump_traceback)
    assert (
        stall_watchdog.arm(
            seconds=1.0,
            directory=tmp_path,
            probe=_DumpProbe(),
            busy=lambda: True,
        )
        is True
    )
    try:
        assert _wait_for(lambda: bool(gate_was_free), timeout=PUBLISHED_S), (
            "the bound never fired, so this cell proved nothing about the dump's "
            "position relative to the gate"
        )
        assert all(gate_was_free), (
            f"the all-thread dump was taken with the watchdog's gate held "
            f"(free at dump time: {gate_was_free}); the gate is what every beat and "
            f"every sample needs, so holding it across a hundreds-of-milliseconds dump "
            f"is the same defect this module exists to report"
        )
    finally:
        stall_watchdog.disarm()


def test_no_native_timer_call_survives_in_the_modules_executable_code() -> None:
    """P3: a source pin — the retired calls cannot come back silently.

    THE PARSE, not a grep: this module's docstrings quote both call names at length
    (they are the evidence for the change), so a text scan would either fail on its
    own documentation or have to be loosened until it caught nothing. An AST walk
    sees only executable attribute access, which is exactly the set to keep empty.

    It also pins the one *entry point* that must remain: ``faulthandler.dump_traceback``
    is the whole of the new dump path, so a change that removed every reference would
    satisfy this cell while destroying the artifact — hence the second assertion.

    MUTATION THIS CELL CATCHES: any reintroduction of a timer-API call in production
    code, including one added to a new helper -> red.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    offenders: list[tuple[str, int]] = []
    dump_traceback_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr in RETIRED_CALLS:
            offenders.append((func.attr, node.lineno))
        if func.attr == "dump_traceback":
            dump_traceback_calls += 1

    assert not offenders, (
        f"{MODULE_PATH.name} calls the retired timer API at {offenders}. Every one of "
        f"those calls replaces a pending timer, and the replace cancels a dump in "
        f"flight while holding the GIL — see the module docstring for the stacks"
    )
    assert dump_traceback_calls >= 1, (
        "the module no longer takes a dump at all: ``faulthandler.dump_traceback`` is "
        "the whole of the fire's evidence path now"
    )


#: The child for P2. THE PRECONDITION IS FORCED, not waited for: a timer that fires
#: ONCE into a dump channel the parent stops draining the moment it has seen the
#: fire, so the C thread's write cannot complete and that dump stays in flight for
#: good. Only then does the child touch the module's arm path — and the module's own
#: dump goes to its own directory, so nothing here depends on the blocked channel.
#:
#: ONE dump, not a stream, and the difference is load rather than mechanics. The
#: property the rig measures is "one dump in flight while the arm path runs"; a
#: ``repeat=True`` timer added nothing to it (the blocked write parks the C thread
#: on its FIRST fire, and a completing control dump needs no successor) but made
#: the control a perpetual all-thread dump storm — six 250-frame-deep threads
#: walked every 50 ms, the rig's own load on a 4-vCPU CI runner. Measured on CI:
#: the control child died inside ``arm()`` in 5 of the 45 CI executions of this
#: cell (`TICK 1 / TIMER-ARMED control / GO`; 1.5-3.8 s where the healthy cell
#: finishes in 0.26-0.44 s over 40 samples — runs 36361736833, 36364477180,
#: 36367266758, 36389917865, 36466096472). Single-shot keeps the geometry — the
#: write still cannot complete undrained and still completes drained — and
#: removes the storm.
_ARM_PATH_CHILD = """
import faulthandler
import pathlib
import socket
import sys
import threading
import time

from local_operator.session.runtime import stall_watchdog

sink = socket.socket(fileno=int(sys.argv[1]))
mode = sys.argv[2]
go = pathlib.Path(sys.argv[3])
dump_dir = pathlib.Path(sys.argv[4])
go_bound = float(sys.argv[5])

# Parked, DEEP frames: the all-thread dump is far larger than any socket buffer, so
# the C thread's first write blocks the moment nobody is reading.
hold = threading.Event()


def deep(n):
    if n == 0:
        hold.wait()
        return
    return deep(n - 1)


for i in range(6):
    threading.Thread(target=deep, args=(250,), name=f"deep{i}", daemon=True).start()


def ticker():
    i = 0
    while True:
        i += 1
        print(f"TICK {i}", flush=True)
        time.sleep(0.2)


threading.Thread(target=ticker, name="loop-proxy", daemon=True).start()

faulthandler.dump_traceback_later(0.05, repeat=False, file=sink, exit=False)
print(f"TIMER-ARMED {mode}", flush=True)

deadline = time.monotonic() + go_bound
while not go.exists():
    if time.monotonic() > deadline:
        print("NO-GO", flush=True)
        raise SystemExit(0)
    time.sleep(0.01)
print("GO", flush=True)

# THE ARM PATH UNDER TEST: what every runtime's boot, every beat and every clean exit
# does. On the base ref the first of these parks in the cancel of a dump that cannot
# finish, holding the GIL, and the ticker above stops for good.
def probe():
    return (0.0, True)


assert stall_watchdog.arm(seconds=600.0, directory=dump_dir, probe=probe) is True
print("ARMED", flush=True)
stall_watchdog.engage()
stall_watchdog.beat(stall_watchdog.WORKLOAD)
stall_watchdog.beat(stall_watchdog.SERVING)
stall_watchdog.disarm()
print("ARM-PATH-RETURNED", flush=True)
print("LOOP-SURVIVED", flush=True)
"""


class _ArmPathChild:
    """One P2 run, with the dump channel driven from the parent side.

    ``keep_draining`` decides the cell: ``True`` is the CONTROL — the parent keeps
    reading, so the dump completes and nothing can wedge — and ``False`` is the rig,
    where the parent stops reading the moment it has SEEN the fire, so the dump cannot
    finish. The only difference between the two runs is whether the dump completes,
    which is what makes the result a measurement of that rather than of the rig.

    THE OTHER CLASS THIS RIG HAS SEEN is a SIGNAL DEATH in the child under CI
    contention, always after the fire and before ``ARMED``: rc=-11 SIGSEGV in runs
    36522096548 and 36524206106 (both ``test (3.12, 1)``, output ending ``TICK 1 /
    TIMER-ARMED control / GO``), and rc=-7 SIGBUS once in the sibling detachment
    cell. Its cause is not the arm path — ~1.5k local executions of this child
    (plus enlarged walks and throttled drains) never reproduced it — and the rig's
    job is to REPORT a death rather than hide it, so the child runs with CPython's
    crash handler and the message carries rc, the signal's name and the child's own
    dump state.

    THIS PRIMITIVE NEVER RETRIES — one instance is one child, one channel, one
    attempt. The retry lives in the DRIVER that wraps it (:func:`_run_leg`), which
    spawns a fresh instance with fresh, attempt-scoped artifacts per attempt and is
    the only place that decides a signal death deserves another look. Keeping the
    decision there leaves this class a pure, single-run measurement.
    """

    def __init__(self, tmp_path: Path, mode: str, *, keep_draining: bool, attempt: int = 1) -> None:
        self.keep_draining = keep_draining
        self.attempt = attempt
        self.lines: list[str] = []
        # ATTEMPT-SCOPED, and the go file is why: the child takes the arm path only
        # once its go file exists, so a retry that reused attempt 1's would find it
        # already written and fire before its own dump was in flight. See
        # :func:`_leg_artifacts`.
        self.script, self.go, self.dump_dir = _leg_artifacts(tmp_path, mode, attempt)
        self.script.write_text(_ARM_PATH_CHILD, encoding="utf-8")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.read_end, write_end = socket.socketpair()
        write_end.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CHANNEL_SNDBUF)
        os.set_inheritable(write_end.fileno(), True)
        self.process = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            [
                sys.executable,
                str(self.script),
                str(write_end.fileno()),
                mode,
                str(self.go),
                str(self.dump_dir),
                "60",
            ],
            # A SIGNAL DEATH IN THIS CHILD MUST LEAVE A READING: with no crash
            # handler enabled, the 2026-09-29 CI deaths (runs 36522096548,
            # 36524206106) reported only `rc=-11` and the output up to `GO` — the
            # child died silently, so a runner fault could not be told from a real
            # crash. PYTHONFAULTHANDLER makes the child print its own fatal-error
            # report (the thread and the frame it died in) into the captured
            # output through the merged stderr. It only engages on a fatal signal;
            # a death still reds the cell and nothing here retries or loosens.
            env=_child_env(tmp_path, PYTHONFAULTHANDLER="1"),
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(write_end.fileno(),),
        )
        write_end.close()
        self.reader = threading.Thread(target=self._read_stdout, name=f"out-{mode}", daemon=True)
        self.reader.start()

    @property
    def pid(self) -> int:
        """The child's pid — the artifact ``death_report`` reads. Never a bare pgrep."""
        return self.process.pid

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line.rstrip("\n"))

    def wait_for_fire_then_release(self) -> bool:
        """Stop reading once the dump has really started; then say GO.

        The fire is published on the channel itself, so this is an event rather than a
        guess: the child must not touch the arm path until a dump is demonstrably in
        flight, and only the parent can see that. The bound (``FIRE_S``) only decides
        how long "never" is waited out — and a child that EXITED ends the wait at
        once, its reader joined (bounded) so the failure message its caller builds
        carries everything the child wrote (the same discipline as ``wait_for``).
        """
        seen = b""
        deadline = time.monotonic() + FIRE_S
        self.read_end.settimeout(0.5)
        while b"Timeout (" not in seen and time.monotonic() < deadline:
            try:
                chunk = self.read_end.recv(65536)
            except TimeoutError:
                if self.process.poll() is not None:
                    self.reader.join(timeout=2.0)
                    break
                continue
            except OSError:
                break
            if not chunk:
                # EOF: the child's end of the channel is closed, so no fire can
                # arrive however long this waits.
                self.reader.join(timeout=2.0)
                break
            seen += chunk
        fired = b"Timeout (" in seen
        if fired and self.keep_draining:
            threading.Thread(target=self._drain_forever, name="drain", daemon=True).start()
        if fired:
            self.go.write_text("go", encoding="utf-8")
        return fired

    def _drain_forever(self) -> None:
        """Keep the control's channel drained for the child's whole life.

        A quiet half-second is NOT an error, and it is the normal case under CI
        load. ``read_end`` still carries the 0.5 s timeout the fire search set,
        so the first gap longer than that raised ``TimeoutError`` — which the
        previous version caught as ``OSError`` and exited the drain silently:
        from then on the "control" was no longer draining, and the one property
        that makes it a control (the only difference from the rig is the drain)
        was gone without a word. Retry on timeout; only a hard ``OSError`` (the
        socket closed by ``close()``) or EOF ends the drain.
        """
        while True:
            try:
                if not self.read_end.recv(65536):
                    return
            except TimeoutError:
                continue
            except OSError:
                return

    def wait_for(self, marker: str, timeout: float) -> str:
        """Wait up to ``timeout`` for ``marker``; return the child's output either way.

        A child that has EXITED ends the wait at once, and its output is then what
        a failure has to report — so the reader thread is joined (bounded) before
        the output is read back. The child can be reaped before its last lines
        have been consumed, and a message that races its own evidence hides the
        difference between "the child died" (``rc`` names it) and "the child is
        wedged". CI run 36466096472's control failure could show only
        ``TICK 1 / TIMER-ARMED control / GO`` for a child that had exited seconds
        earlier; both distinguishing facts were unavailable to it.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(marker in line for line in self.lines):
                break
            if self.process.poll() is not None:
                # EOF on the pipe arrives with the child's exit; bounded because
                # a grandchild of the child could hold the write end open.
                self.reader.join(timeout=2.0)
                break
            time.sleep(0.05)
        return "\n".join(self.lines)

    def death_report(self) -> str:
        """One diagnostic line for a child whose sentinels are missing.

        Read from artifacts, not from guesses: ``rc`` separates exited from
        killed (negative = the signal number: -9 SIGKILL, -7 SIGBUS, -11
        SIGSEGV) and from still running, and the child's OWN dump file says
        whether ``arm`` got as far as its header write. Both facts survive a
        dead reader, which is exactly what the message this supports could not
        rely on when the control child died on CI.
        """
        rc = self.process.returncode
        if rc is None:
            status = "still running"
        else:
            status = f"rc={rc}"
            if rc < 0:
                try:
                    status += f" ({signal.Signals(-rc).name})"
                except ValueError:  # an unnamed signal number: rc alone still reads
                    pass
        dump = stall_watchdog.dump_path(self.process.pid, self.dump_dir)
        present = "present" if dump.exists() else "absent"
        return f"child: {status}; its own dump file is {present}"

    def close(self) -> None:
        """Kill THIS pid and reap it; close the channel. Never a bare pgrep."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10.0)
        self.read_end.close()


#: How many attempts ONE leg may spend. A child that dies by SIGNAL before its
#: sentinels appear is an environmental class (see the cell docstring), so the leg
#: gets a bounded second look rather than reding main on evidence it never produced;
#: three is the ceiling — a THIRD consecutive signal death is no longer transient,
#: and an unbounded retry would hide a real, deterministic crash behind a green run.
LEG_ATTEMPTS = 3

#: The marker CPython's crash handler prints when ``PYTHONFAULTHANDLER=1`` catches a
#: fatal signal. Our 2026-10-06 occurrence carried NONE (run 37404837806's captured
#: output was ``TICK 1 / TIMER-ARMED control / GO``), so a report is reported when it
#: is there and its absence is never read as evidence about the cause.
_FATAL_MARKER = "Fatal Python error"


def _leg_artifacts(tmp_path: Path, mode: str, attempt: int) -> tuple[Path, Path, Path]:
    """The per-ATTEMPT artifact paths of one leg run: ``(script, go_file, dump_dir)``.

    ATTEMPT-SCOPED, and the GO FILE is what makes that load-bearing rather than tidy.
    The child waits for its go file to exist and only then takes the arm path; a retry
    that reused attempt 1's go file would find it ALREADY THERE — the parent wrote it
    for attempt 1 — and take the arm path before its own fire, measuring nothing this
    cell is about. Fresh paths per attempt make a stale go file unreachable rather
    than merely unlikely. The dump dir is scoped for the same reason: a retried child's
    ``death_report`` asks whether ITS OWN dump exists, and a shared dir would answer
    with the previous child's file.
    """
    return (
        tmp_path / f"arm_path_child_{mode}_{attempt}.py",
        tmp_path / f"go-{mode}-{attempt}",
        tmp_path / f"dumps-{mode}-{attempt}",
    )


class _LegAttempt:
    """One attempt's evidence, kept whether or not the attempt is retried.

    ``output`` is the child's captured output — what a failure message has to carry and
    what the leg's own pins read. ``sentinels_present`` is the leg's verdict on its
    markers. ``exited_by_signal`` is the RETRY GATE, read BEFORE ``close()``: the child
    had EXITED (``poll() is not None``) with a negative returncode. ``returncode`` and
    ``death_report`` are the child's own reading, snapshotted before close — close
    SIGKILLs the child, so a reading taken after it can only ever say ``rc=-9``, the one
    value that cannot tell a parked arm path from a child that died on its own.
    """

    __slots__ = (
        "output",
        "sentinels_present",
        "exited_by_signal",
        "returncode",
        "death_report",
        "pid",
    )

    def __init__(
        self,
        *,
        output: str,
        sentinels_present: bool,
        exited_by_signal: bool,
        returncode: int | None,
        death_report: str,
        pid: int | None,
    ) -> None:
        self.output = output
        self.sentinels_present = sentinels_present
        self.exited_by_signal = exited_by_signal
        self.returncode = returncode
        self.death_report = death_report
        self.pid = pid


def _fatal_report_excerpt(output: str, limit: int = 4) -> str:
    """The first lines of the child's own fatal-error report, or ``""`` when it wrote none.

    ``PYTHONFAULTHANDLER=1`` makes a fatal signal print the dying thread and frame into
    the captured output — but only SOMETIMES: run 37404837806 (our first occurrence after
    #1753) carried no report, and #1753's own crash-report capture did not appear in it
    either. So this reads what is present and reports its absence as absence, never as a
    hint about the cause.
    """
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if _FATAL_MARKER in line:
            return "\n".join(lines[index : index + limit])
    return ""


def _retry_warning(label: str, deaths: "list[_LegAttempt]") -> str:
    """The line a consumed retry leaves in CI's warnings, plus any fatal report.

    A retry exists to keep main green through an environmental death, and that is
    exactly what makes it worth watching: if the class grows from ~4-in-a-window to
    every run, the warning is the only place it is visible, because the cell itself
    stays green. It names each attempt's rc, signal and dump presence, an output tail,
    and the child's own fatal report when it wrote one.
    """
    lines = [
        f"the {label} leg consumed {len(deaths)} of {LEG_ATTEMPTS - 1} retry "
        f"attempt(s): each child EXITED BY SIGNAL before its sentinel appeared, which "
        f"is the environmental class this retry exists for (see this cell's docstring)."
    ]
    for index, attempt in enumerate(deaths, start=1):
        lines.append(
            f"  attempt {index} (pid {attempt.pid}): {attempt.death_report}; output tail "
            f"{attempt.output.splitlines()[-3:]!r}"
        )
        fatal = _fatal_report_excerpt(attempt.output)
        if fatal:
            lines.append(f"  attempt {index} wrote its own fatal report, beginning:\n{fatal}")
    return "\n".join(lines)


def _leg_failure(label: str, note: str, history: "list[_LegAttempt]", *, all_died: bool) -> str:
    """The aggregate a failed leg raises: EVERY attempt's own reading and output.

    A retried leg that finally fails must not lose the attempts before it: the first
    death is often the most informative (it may be the one carrying the crash report),
    and a message that reported only the last child would make the retry look like a
    single run. ``all_died`` separates the exhausted-retry failure from a shape that was
    never retried at all.
    """
    if all_died:
        head = (
            f"the {label} leg's child DIED BY SIGNAL in all {len(history)} attempts, "
            f"each before its sentinel appeared"
        )
    else:
        head = (
            f"the {label} leg failed in a shape that is NOT a signal death, so it was "
            f"not retried: {history[-1].death_report}"
        )
    bodies = [
        f"[attempt {index} pid {attempt.pid}] {attempt.death_report}; output tail "
        f"{attempt.output.splitlines()[-4:]!r}"
        for index, attempt in enumerate(history, start=1)
    ]
    return f"{head}. {note}\n" + "\n".join(bodies)


def _measure_leg(child: Any, *, sentinels: tuple[str, ...]) -> _LegAttempt:
    """Run ONE attempt of a leg on ``child`` and read its outcome — no assertions.

    The fire is a PRECONDITION, not a sentinel: the child only takes the arm path once
    ``wait_for_fire_then_release`` has published a dump in flight, so a leg whose fire
    never arrived cannot have reached its sentinels. The sentinel wait is skipped only
    when the child has already EXITED — it can publish nothing more and its reader has
    already been joined — which keeps a fast death from costing the full bound.
    """
    fired = child.wait_for_fire_then_release()
    bound = CHILD_BOUND_S if (fired or child.process.poll() is None) else 0.0
    output = child.wait_for(sentinels[-1], bound)
    returncode = child.process.returncode
    return _LegAttempt(
        output=output,
        sentinels_present=fired and all(marker in output for marker in sentinels),
        # THE RETRY GATE, read before the caller's close(): EXITED (``poll() is not
        # None``) with a negative returncode — a signal death. Every other shape is a
        # real failure and is never retried: still alive at the bound (the park shape),
        # a clean ``rc >= 0`` exit, or a sentinel that DID appear.
        exited_by_signal=(
            child.process.poll() is not None and returncode is not None and returncode < 0
        ),
        returncode=returncode,
        death_report=child.death_report(),
        pid=child.pid,
    )


def _run_leg(
    *,
    tmp_path: Path,
    label: str,
    mode: str,
    keep_draining: bool,
    sentinels: tuple[str, ...],
    note: str,
    spawn: Any = _ArmPathChild,
    attempts: int = LEG_ATTEMPTS,
) -> str:
    """Run ONE leg, retrying a child that died by signal before its sentinels appeared.

    The retry is deliberately narrow: a FRESH child, fresh artifacts and a fresh
    socketpair per attempt (``_ArmPathChild`` builds them; see :func:`_leg_artifacts`),
    and another attempt ONLY for the observed environmental class. Every other failure
    shape raises on the spot, with the attempts so far aggregated into the message.
    A consumed retry warns rather than passing silently.

    ``spawn`` exists so the semantics above are testable without processes: the cells
    below drive this with a scripted double. Its signature is ``_ArmPathChild``'s.
    """
    history: list[_LegAttempt] = []
    for attempt in range(1, attempts + 1):
        child = spawn(tmp_path, mode, keep_draining=keep_draining, attempt=attempt)
        try:
            evidence = _measure_leg(child, sentinels=sentinels)
        finally:
            child.close()
        history.append(evidence)
        if evidence.sentinels_present:
            if len(history) > 1:
                warnings.warn(_retry_warning(label, history[:-1]), stacklevel=2)
            return evidence.output
        if not evidence.exited_by_signal:
            raise AssertionError(_leg_failure(label, note, history, all_died=False))
    raise AssertionError(_leg_failure(label, note, history, all_died=True))


#: What each leg's failure MEANS, carried into the aggregate so a red CI log still says
#: which of the two legs broke and what that rules out.
_CONTROL_NOTE = (
    "the CONTROL runs the same child with the channel drained, so a control that does "
    "not reach LOOP-SURVIVED means either the rig never produced an in-flight dump at "
    "all (nothing here measures what it claims) or the drain is not the difference this "
    "cell is about"
)
_RIG_NOTE = (
    "with the channel undrained the arm path must still RETURN: a ticker that goes "
    "silent at GO is the field signature — the caller parked inside the C timer call "
    "holding the GIL, so no Python thread in the process could run — and a child that "
    "died on its own is named by rc"
)


def test_the_arm_path_returns_while_a_dump_is_in_flight(tmp_path: Path) -> None:
    """P2: with a dump that cannot finish, the arm path must still return.

    THIS IS THE FAILING-FIRST CELL for the whole change, and it is the one the field
    demanded: the runtime's own boot, its two planes' beats and its clean exit all run
    through the arm path, and on the base ref the first of them parks inside
    ``cancel_dump_traceback_later`` while holding the GIL — every other Python thread
    in the process stops, which is why the child's ``TICK`` line goes silent at
    ``GO`` and neither sentinel appears.

    ASSERTED ON A SENTINEL, never on a stopwatch: ``ARM-PATH-RETURNED`` then
    ``LOOP-SURVIVED``. The bound only decides how long "never" is waited out; it is
    not the evidence.

    THE CONTROL RUNS THE SAME CHILD with the channel drained, and reaches both
    sentinels on any build — so a green result cannot be explained by the rig never
    having produced an in-flight dump at all.

    SIGNAL-DEATH RETRY. The child has also died by an unguarded SIGNAL under CI
    contention, always after the fire and before ``ARMED`` — ``rc=-11`` SIGSEGV, its
    output ending ``TICK 1 / TIMER-ARMED control / GO``. Occurrences inside the
    2026-10-04..06 window: runs 37229558921, 37354166125, 37382485908 and
    37404837806, plus 36522096548 and 36524206106 in September. That is FOUR deaths
    in a ~48 h window against ~1.6k local executions of this child that never
    reproduced it, and one of them still reported the dump file present
    (37404837806) — so it is an environment-dependent signal death, not the guarded
    regression. Each leg therefore gets at most three attempts and ONLY for exactly
    this shape: the child EXITED (``poll() is not None``, read before close) with
    ``rc < 0`` and its sentinel(s) still absent. Every other shape — a park (child
    alive at the bound, ticker silent), a clean ``rc >= 0`` exit, a missing sentinel
    while the child is alive, or a fire that never came while it is alive — fails
    without a retry, which is why the retry CANNOT mask MUTATION THIS CELL CATCHES:
    a re-introduced C timer call parks the child, it does not kill it, and a park is
    never retried. A consumed retry is not silent either: it emits a ``UserWarning``
    naming the attempt, rc, signal, dump presence and an output tail
    (:func:`_retry_warning`), so the class stays visible in CI's warning summary
    while main stays green. Our 2026-10-06 occurrence (37404837806) carried NO
    fatal-error report, so a report present in a retried death is surfaced and its
    absence is never read as evidence about the cause.

    MUTATION THIS CELL CATCHES: any arm path that reaches a C timer call again -> red.
    """
    control_output = _run_leg(
        tmp_path=tmp_path,
        label="control",
        mode="control",
        keep_draining=True,
        sentinels=("LOOP-SURVIVED",),
        note=_CONTROL_NOTE,
    )
    assert "LOOP-SURVIVED" in control_output, (
        "the CONTROL run did not survive with its dump channel drained, so the channel "
        f"is not the difference this cell is about; output was {control_output!r}"
    )

    output = _run_leg(
        tmp_path=tmp_path,
        label="rig",
        mode="wedge",
        keep_draining=False,
        sentinels=("ARM-PATH-RETURNED", "LOOP-SURVIVED"),
        note=_RIG_NOTE,
    )
    assert "ARM-PATH-RETURNED" in output, (
        "the arm path never returned while a dump was in flight. The child stopped "
        f"after: {output.splitlines()[-4:]!r} — a ticker that goes silent at GO is the "
        "field signature: the caller parked inside the C timer call holding the GIL, so "
        "no Python thread in the process could run"
    )
    assert "LOOP-SURVIVED" in output, output


# ---------------------------------------------------------------------------------
# THE RETRY'S OWN SEMANTICS, driven with a scripted double (no processes)
# ---------------------------------------------------------------------------------


class _FakeChildShape:
    """The scripted life of one fake child: what it published, and how it ended.

    ``returncode is None`` means it was STILL ALIVE at the bound — the park shape, which
    must never be retried. Negative is a signal death (the retry class); zero or more is
    a clean exit (also never retried).
    """

    __slots__ = ("fired", "output", "returncode")

    def __init__(self, *, fired: bool, output: str, returncode: int | None) -> None:
        self.fired = fired
        self.output = output
        self.returncode = returncode


class _FakeArmPathChild:
    """A double for :class:`_ArmPathChild` replaying a scripted shape — no process.

    It exposes only the surface the driver reads (fire gate, sentinel wait, ``poll`` /
    ``returncode`` through ``process``, ``death_report``, ``close``, ``pid``) and records
    that it was closed, so a cell can prove the driver closes every attempt. It is
    deliberately NOT a subclass: an attribute the real class gains should make the
    double fail loudly, not inherit. No real processes run in these cells.
    """

    def __init__(
        self, shape: _FakeChildShape, *, mode: str, keep_draining: bool, attempt: int
    ) -> None:
        self.shape = shape
        self.mode = mode
        self.keep_draining = keep_draining
        self.attempt = attempt
        self.closed = False
        self.process = SimpleNamespace(
            pid=10_000 + attempt,
            returncode=shape.returncode,
            poll=lambda: shape.returncode,
        )

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def wait_for_fire_then_release(self) -> bool:
        return self.shape.fired

    def wait_for(self, marker: str, timeout: float) -> str:
        return self.shape.output

    def death_report(self) -> str:
        rc = self.shape.returncode
        if rc is None:
            status = "still running"
        elif rc < 0:
            status = f"rc={rc} ({signal.Signals(-rc).name})"
        else:
            status = f"rc={rc}"
        return f"child: {status}; its own dump file is absent"

    def close(self) -> None:
        self.closed = True


def _scripted_spawn(
    shapes: list[_FakeChildShape],
    calls: list[tuple[str, bool, int]],
    children: list[_FakeArmPathChild],
) -> Any:
    """A spawn double handing out one fake child per attempt, recording each call."""

    def spawn(tmp_path: Path, mode: str, *, keep_draining: bool, attempt: int) -> _FakeArmPathChild:
        calls.append((mode, keep_draining, attempt))
        child = _FakeArmPathChild(
            shapes[attempt - 1], mode=mode, keep_draining=keep_draining, attempt=attempt
        )
        children.append(child)
        return child

    return spawn


_RIG_SENTINELS: tuple[str, ...] = ("ARM-PATH-RETURNED", "LOOP-SURVIVED")
_DEAD_OUTPUT = "TICK 1\nTIMER-ARMED control\nGO"
_GREEN_OUTPUT = "TICK 1\nTIMER-ARMED control\nGO\nARM-PATH-RETURNED\nLOOP-SURVIVED"


def test_the_leg_artifacts_are_attempt_scoped(tmp_path: Path) -> None:
    """A stale go file must be unreachable: every artifact path is per attempt.

    The go file is the load-bearing one — the child takes the arm path only once it
    exists — so a retry that reused it would fire before its own dump was in flight.
    """
    first_script, first_go, first_dumps = _leg_artifacts(tmp_path, "wedge", 1)
    second_script, second_go, second_dumps = _leg_artifacts(tmp_path, "wedge", 2)
    assert first_go == tmp_path / "go-wedge-1"
    assert second_go == tmp_path / "go-wedge-2"
    assert first_go != second_go, "a reused go file would tell the retry child to GO early"
    assert first_script != second_script
    assert first_dumps != second_dumps
    assert second_script.name == "arm_path_child_wedge_2.py"
    assert second_dumps.name == "dumps-wedge-2"


def test_a_signal_death_is_retried_and_the_retry_warns(tmp_path: Path) -> None:
    """The retry class end to end: attempt 1 dies by signal, attempt 2 is green.

    The warning is asserted, not incidental: a consumed retry that passed silently is
    how an environmental class growing into every run would go unnoticed.
    """
    shapes = [
        _FakeChildShape(fired=True, output=_DEAD_OUTPUT, returncode=-11),
        _FakeChildShape(fired=True, output=_GREEN_OUTPUT, returncode=0),
    ]
    calls: list[tuple[str, bool, int]] = []
    children: list[_FakeArmPathChild] = []
    with pytest.warns(UserWarning, match="consumed 1 of 2 retry"):
        output = _run_leg(
            tmp_path=tmp_path,
            label="rig",
            mode="wedge",
            keep_draining=False,
            sentinels=_RIG_SENTINELS,
            note=_RIG_NOTE,
            spawn=_scripted_spawn(shapes, calls, children),
        )
    assert output == _GREEN_OUTPUT
    # A FRESH, attempt-scoped child each time, in order — never a reused instance.
    assert calls == [("wedge", False, 1), ("wedge", False, 2)]
    assert all(child.closed for child in children), "every attempt must be reaped"


def test_a_retried_deaths_fatal_report_reaches_the_warning(tmp_path: Path) -> None:
    """``PYTHONFAULTHANDLER=1`` output, when the child wrote it, is surfaced."""
    fatal = (
        "Fatal Python error: Segmentation fault\n"
        "Current thread 0x0000000123456789 (most recent call first):\n"
        '  File "/tmp/x.py", line 1 in probe'
    )
    shapes = [
        _FakeChildShape(fired=True, output=f"{_DEAD_OUTPUT}\n{fatal}", returncode=-11),
        _FakeChildShape(fired=True, output=_GREEN_OUTPUT, returncode=0),
    ]
    with pytest.warns(UserWarning, match="Fatal Python error: Segmentation fault"):
        _run_leg(
            tmp_path=tmp_path,
            label="rig",
            mode="wedge",
            keep_draining=False,
            sentinels=_RIG_SENTINELS,
            note=_RIG_NOTE,
            spawn=_scripted_spawn(shapes, [], []),
        )


def test_all_attempts_dying_by_signal_fail_with_every_attempt_aggregated(
    tmp_path: Path,
) -> None:
    """An exhausted retry fails, and the message keeps EVERY attempt, not just the last."""
    shapes = [
        _FakeChildShape(fired=True, output=_DEAD_OUTPUT, returncode=-11)
        for _ in range(LEG_ATTEMPTS)
    ]
    calls: list[tuple[str, bool, int]] = []
    with pytest.raises(AssertionError) as excinfo:
        _run_leg(
            tmp_path=tmp_path,
            label="rig",
            mode="wedge",
            keep_draining=False,
            sentinels=_RIG_SENTINELS,
            note=_RIG_NOTE,
            spawn=_scripted_spawn(shapes, calls, []),
        )
    message = str(excinfo.value)
    assert f"DIED BY SIGNAL in all {LEG_ATTEMPTS} attempts" in message
    for index in range(1, LEG_ATTEMPTS + 1):
        assert f"[attempt {index} pid {10_000 + index}]" in message
    assert message.count("rc=-11 (SIGSEGV)") == LEG_ATTEMPTS
    assert len(calls) == LEG_ATTEMPTS


def test_a_park_shape_is_not_retried(tmp_path: Path) -> None:
    """The guarded regression must never be retried: child ALIVE at the bound, no sentinel.

    A park is a live child whose ticker has gone silent — the exact shape the C timer
    call produces — so a retry here would hide the whole change's regression.
    """
    shapes = [_FakeChildShape(fired=True, output=_DEAD_OUTPUT, returncode=None)]
    calls: list[tuple[str, bool, int]] = []
    with pytest.raises(AssertionError) as excinfo:
        _run_leg(
            tmp_path=tmp_path,
            label="rig",
            mode="wedge",
            keep_draining=False,
            sentinels=_RIG_SENTINELS,
            note=_RIG_NOTE,
            spawn=_scripted_spawn(shapes, calls, []),
        )
    assert calls == [("wedge", False, 1)], "a park must fail on the first attempt"
    message = str(excinfo.value)
    assert "NOT a signal death" in message
    assert "still running" in message


def test_a_child_that_exits_cleanly_is_not_retried(tmp_path: Path) -> None:
    """``rc >= 0`` with the sentinel missing is a real failure, not a signal death."""
    shapes = [_FakeChildShape(fired=False, output="NO-GO", returncode=0)]
    calls: list[tuple[str, bool, int]] = []
    with pytest.raises(AssertionError) as excinfo:
        _run_leg(
            tmp_path=tmp_path,
            label="rig",
            mode="wedge",
            keep_draining=False,
            sentinels=_RIG_SENTINELS,
            note=_RIG_NOTE,
            spawn=_scripted_spawn(shapes, calls, []),
        )
    assert calls == [("wedge", False, 1)]
    assert "rc=0" in str(excinfo.value)


def test_the_signal_leg_is_one_registration_and_nothing_else() -> None:
    """Decision 1's pins: one registration, no chain, no unregister, no second handler.

    EACH ASSERTION IS A WAY THE MEASURED SEGFAULT COULD COME BACK, so this cell is the
    regression the crash earns: a chain onto a saved disposition (``chain=True``), an
    ``unregister`` that restores the disposition saved at arm time rather than the live
    one, or the runtime installing its own handler over the leg's slot. The fourth
    asserts the route that replaced the slot — the notification — is what the runtime
    uses, so a later reader cannot re-add `add_signal_handler(debug_stacks, ...)` and
    silently shadow the leg again.
    """
    import inspect

    from local_operator.session.runtime import process as process_module

    source = Path(stall_watchdog.__file__).read_text(encoding="utf-8")
    assert "faulthandler.unregister(" not in source, (
        "the module calls unregister again; that restores the disposition SAVED at arm "
        "time, which is what faulted inside signal handling on a real runtime child"
    )
    assert ", chain=True" not in source, (
        "the module chains onto a saved handler again; faulthandler's chain re-raises "
        "against that saved disposition, measured as a segfault (rc=-11)"
    )
    assert ", chain=False" in source, "the leg must own its slot with no saved-handler path"

    amain = inspect.getsource(process_module.amain)
    assert "add_signal_handler(debug_stacks" not in amain, (
        "the runtime installs its own SIGUSR1 handler again, which shadows the leg on "
        "every live runtime — one signal, one slot, and the leg has to own it"
    )
    assert "notify_on_evidence_signal" in amain, (
        "the runtime's own SIGUSR1 action is no longer reached from the leg: the walk "
        "must be scheduled off the notification, or it is simply gone"
    )


def test_sigusr1_reaches_the_leg_on_a_real_armed_runtime_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 1 end to end: a REAL armed runtime child takes SIGUSR1 and lives.

    THE CRASH THIS CELL IS THE REGRESSION FOR: the re-registering revision died
    ``rc=-11`` on the first SIGUSR1 the suite's own readiness probe sent. So the first
    assertion is that the child is still running, the second that the datum arrived in
    the MODULE'S OWN ``O_APPEND`` HANDLE, the third that it names the process's threads,
    the fourth that the runtime's own walk still happens — one signal, both halves, no
    sigaction fight.

    The child is the production one (``launch._spawn_runtime`` with the detachment
    suite's harness, ``_spawn_interpreter`` pinned to this interpreter so it runs this
    tree). A signal dump writes no ``Timeout (`` line, so the last assertion also pins
    that the leg cannot be read as a bound that fired.
    """
    import signal as signal_module

    from local_operator.session.runtime import launch as launch_module
    from tests.unit.session.runtime.test_runtime_detachment import (
        _SESSION_ID,
        _capture_text,
        _isolate,
        _log_text,
        _reap,
        _seed,
        _wait_for_record,
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed(config_dir)
    _isolate(monkeypatch, config_dir)
    monkeypatch.setattr(launch_module, "_spawn_interpreter", lambda: sys.executable)

    previous_usr1 = signal_module.signal(signal_module.SIGUSR1, signal_module.SIG_IGN)
    child = None
    try:
        child = launch_module._spawn_runtime(
            _SESSION_ID,
            str(config_dir),
            defer_materialise=False,
        )
        pid = child.pid
        _wait_for_record(config_dir, child=child)
        dump = config_dir / "logs" / f"{stall_watchdog.DUMP_PREFIX}-{pid}.log"
        assert dump.is_file(), (
            f"the child never armed its bound:\n{_capture_text(child)}\n"
            f"{_log_text(config_dir)[-800:]}"
        )

        # READINESS by the suite's own probe: SIGUSR1, until the runtime's walk reports.
        deadline = time.monotonic() + 60.0
        kills = 0
        while "state: streaming=" not in _log_text(config_dir):
            assert child.poll() is None, (
                f"the runtime died on SIGUSR1 — rc={child.returncode}, which is the "
                f"regression this cell exists for:\n{_capture_text(child)}\n"
                f"{_log_text(config_dir)[-800:]}"
            )
            assert time.monotonic() < deadline, (
                f"the runtime never performed its own walk after {kills} signals, so the "
                f"notification route is not wired:\n{_log_text(config_dir)[-800:]}"
            )
            os.kill(pid, signal_module.SIGUSR1)
            kills += 1
            time.sleep(0.2)

        # THE LEG: the module's own handle must now carry an all-threads dump.
        deadline = time.monotonic() + 30.0
        text = dump.read_text(encoding="utf-8", errors="replace")
        while "Thread 0x" not in text and time.monotonic() < deadline:
            time.sleep(0.2)
            text = dump.read_text(encoding="utf-8", errors="replace")
        # THE LEG WRITES ONLY THE SIGNAL'S OWN THREAD (``all_threads=False``): the
        # all-threads walk is what segfaulted a booting runtime child, and the thread that
        # interests an operator -- the one parked in the loop -- IS the signal's thread.
        # So the header to look for is the frame, not faulthandler's "Thread 0x" banner,
        # which only the all-threads form writes.
        deadline = time.monotonic() + 30.0
        while "in <module>" not in text and time.monotonic() < deadline:
            time.sleep(0.2)
            text = dump.read_text(encoding="utf-8", errors="replace")
        assert "in <module>" in text, (
            f"the signal never reached the module's handler: {kills} signals sent and the "
            f"dump holds {dump.stat().st_size} bytes\n{_log_text(config_dir)[-800:]}"
        )
        assert child.poll() is None, "the runtime did not survive its own evidence signal"
        assert (
            stall_watchdog.fired_pids(config_dir / "logs") == set()
        ), "a signal dump was read as a bound fire; the leg must leave no fired marker"
        assert stall_watchdog.FIRED_MARKER not in text
    finally:
        signal_module.signal(signal_module.SIGUSR1, previous_usr1)
        if child is not None:
            _reap(child, config_dir)


# ---------------------------------------------------------------------------------
# WHO TAKES THE DEADLINE THE PROGRESS LEG RECORDS
# ---------------------------------------------------------------------------------
#: The window the two in-process cells arm with, in the fake clock's seconds. Short
#: because the window IS the wait: the run has to span it before the predicate can be
#: judged at all, and every step below is one sampler pass.
CELL_WINDOW_S = 2.0

#: How many sampler passes carry a plane stamp. A beat re-arms the deadline and is read
#: before the sampler's own read in the same pass (see ``_SteppingProbe``), so a stamp on
#: every pass would pre-empt the very wake a cell is measuring. Ten passes is one stamp
#: per 1.0 s of the fake clock against a 2.0 s window: frequent enough that neither plane
#: ever looks silent, sparse enough that the sampler's own read lands on passes no beat
#: shares.
STAMP_EVERY_PASSES = 10

#: How many sampler passes a cell waits for before it says the reporter stopped. A
#: pass is the sampler's whole wake, so this is "the leg kept looking", never a
#: stopwatch on a bound.
PASSES_BEFORE_VERDICT = 40


class _SteppingClock:
    """A clock the sampler is driven through, one pass at a time.

    The module reads ``time.monotonic``/``process_time`` through its own module global,
    so the readings the progress predicate is JUDGED on are exactly the two a cell can
    control. Waiting out a real window would make every cell here a bet on host load —
    this fleet runs the suite beside ~25 sessions at a load average of 50-100, where a
    spinning child's own mean measures 0.017-0.021 of a core against the module's 0.05
    floor (measured 2026-09-24, and the reason the child cell below lowers the floor).
    The burn these cells judge is stated rather than measured: 0.05 s of CPU per 0.1 s
    of wall is half a core, which is 2.6x the incident's own 0.19 and 10x the floor.
    """

    def __init__(self, *, wall_per_pass: float = 0.1, cpu_per_pass: float = 0.05) -> None:
        self.wall = 1_000.0
        self.cpu = 5.0
        self.wall_per_pass = wall_per_pass
        self.cpu_per_pass = cpu_per_pass

    def monotonic(self) -> float:
        return self.wall

    def process_time(self) -> float:
        return self.cpu

    def time(self) -> float:
        return 1_700_000_000.0

    def localtime(self, _timestamp: float) -> str:
        # ``_note_quiet_plane`` formats the instant a plane last reported, and ``beat``
        # reaches it; a fixed string is enough, because what a cell asserts is a
        # deadline or a marker, never this rendering.
        return "2026-01-01 00:00:00"

    def strftime(self, _fmt: str, _stamp: object = None) -> str:
        return "2026-01-01 00:00:00"


class _SteppingProbe:
    """The progress probe, stepped: one call is one pass of the window.

    ``_progress_sampler`` reads the probe exactly once per pass, outside its gate, so
    advancing the clock HERE is what makes a pass a step — and it is why the run spans
    its window by construction rather than by the host's willingness to schedule a
    thread. It also STAMPS BOTH PLANES, as a runtime's own loops do: a runtime with
    nothing reporting is the liveness leg's subject, and every cell here is about the
    PROGRESS leg, so a bound left to expire from silence would fire for a reason that is
    not under test.

    THE STAMPS ARE EVERY ``STAMP_EVERY_PASSES`` PASSES RATHER THAN EVERY PASS, and that
    is what the live runtime looks like rather than a convenience. A ``beat`` RE-ARMS
    (:func:`beat` ends in :func:`_rearm`), and this probe is read before the sampler
    reads ``due_at`` in the same pass — so a probe that stamped on every pass would push
    the deadline ahead of that read on every pass, and a cell about "who takes the
    deadline" would be measuring its own rig. The runtime's planes stamp from their own
    loops, unsynchronised with the sampler, so at most one wake per beat can lose the
    race — which is exactly what this cadence models.

    The answers are the arms a real probe can give: a still report with a lane holding a
    step (``in_flight``), and a registry that cannot be read at all (``raises``, which
    the module's own read turns into "no live second leg").
    """

    def __init__(self, clock: _SteppingClock, *, in_flight: bool = False, raises: bool = False):
        self.clock = clock
        self.in_flight = in_flight
        self.raises = raises
        self.passes = 0

    def __call__(self) -> tuple[object, bool]:
        self.passes += 1
        self.clock.wall += self.clock.wall_per_pass
        self.clock.cpu += self.clock.cpu_per_pass
        if self.passes % STAMP_EVERY_PASSES == 0:
            stall_watchdog.beat(stall_watchdog.WORKLOAD)
            stall_watchdog.beat(stall_watchdog.SERVING)
        if self.raises:
            raise RuntimeError("the lane registry could not be read")
        return "still", self.in_flight


def _dump_for_this_process(directory: Path) -> Path:
    return stall_watchdog.dump_path(os.getpid(), directory)


def _fires_in(text: str) -> int:
    return text.count(stall_watchdog.FIRED_MARKER)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _walk_finished_in(text: str) -> bool:
    """Did the dump's all-thread walk run to its own end?

    TWO ENDINGS ARE FINISHED, and both count because they are the same fact seen
    from two sides: the walk reached this process's OLDEST thread -- this test's
    own frame, the strongest reading -- or it reached CPython's own thread cap.
    ``faulthandler.dump_traceback(all_threads=True)`` walks
    ``_Py_DumpTracebackThreads``, which visits newest-to-oldest and stops after
    ``MAX_NTHREADS`` (100) threads with a bare ``...`` line it writes on the way
    out. A pytest worker that has carried a full shard's load holds more live
    threads than that, so the main thread -- the oldest, and the only carrier of
    this file's name -- sits past the cap and is never visited: the CI of PR
    #1695 failed exactly there on three attempts, on three different workers
    (gw2, gw0, gw1), every dump ending ``\\n...\\n`` and naming no frame of the test.

    THE CAP LINE ARRIVES ONLY AT THAT RETURN -- the frame-depth ellipsis carries
    a leading space and nothing else writes a bare ``...`` line -- so "either
    ending is present" reads exactly as "the walk is over": a snapshot that
    caught the walk mid-write carries neither.
    """
    return Path(__file__).name in text or any(line == "..." for line in text.splitlines())


def _armed_in(tmp_path: Path, probe: _SteppingProbe) -> Path:
    """Arm THIS process against ``tmp_path``, with the exit leg held, and return its dump.

    The exit leg is held (``busy`` answers True) because the cell's subject is the
    wake: an arm that is not held ends the process HERE, on the pytest worker, which
    would take the reporter's own thread down with it — that arm is the child cell
    below, where ending a process is the thing it is safe to observe.

    THE SAMPLER IS ALREADY RUNNING WHEN THIS RETURNS (``arm`` starts it), so nothing
    here may assume it has not woken yet: every wait below is on an EVENT — a marker in
    the file, a pass count — never on the sampler being a step behind this thread.
    """
    assert stall_watchdog.arm(
        seconds=CELL_WINDOW_S,
        directory=tmp_path,
        probe=probe,
        busy=lambda: True,
    ), "the bound could not be armed, so this cell would prove nothing about its wake"
    dump = _dump_for_this_process(tmp_path)
    assert dump.is_file(), "arming did not open the dump this process's fire is written to"
    return dump


def test_the_progress_deadline_is_taken_before_the_predicate_can_pre_empt_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE WAKE, both halves: the dump lands AND the sampler carries on firing.

    THE DEFECT THIS CELL IS THE REGRESSION FOR (found by the migration lane on #1517's
    own change). ``_sample`` did ``_fire_progress(armed, now)`` and returned ``not
    armed.held``; with the C timer gone, the MIN_REARM_S deadline that call records is
    taken by the sampler's NEXT pass — and that pass read ``due_at`` AFTER calling
    ``_sample``, so on every pass where the predicate fired again it re-armed over its
    own deadline (``pin``/``deadline`` answer ``now + MIN_REARM_S``) before the read
    could see it. The arm cancelled itself: a loop whose burn never dips got its ``no
    progress`` line and then nothing, for as long as it kept spinning.

    WHY THIS CELL CANNOT PASS ON THE PRE-FIX HEAD, and it is the stepping probe that
    makes that deterministic rather than load-dependent: with the arm held the sampler
    does keep waking, but every one of its passes fires the predicate (a still report
    and a stated half-core burn), so every pass re-arms the deadline it is about to
    read. No dump is written at all, and the wait below times out on a file that holds
    one progress line.

    AND THE OTHER HALF IS THE SPARED DIRECTION: nothing here may end the runtime, which
    is why the exit leg is held and disarm is what stops the leg.
    """
    clock = _SteppingClock()
    monkeypatch.setattr(stall_watchdog, "time", clock)
    probe = _SteppingProbe(clock)
    dump = _armed_in(tmp_path, probe)
    try:
        assert _wait_for(lambda: _fires_in(_read(dump)) >= 1, timeout=120.0), (
            "the progress leg's own deadline was never taken: the run wrote "
            f"{_read(dump).count(stall_watchdog.PROGRESS_MARKER)} progress line(s) over "
            f"{probe.passes} sampler passes and no fire, so the deadline it recorded is "
            f"read after the predicate that pre-empts it:\n{_read(dump)[-600:]!r}"
        )
        # THE SNAPSHOT BELOW WAITS FOR THE WALK TO FINISH, because the marker it just
        # saw is written BEFORE the walk: :func:`_fire` flushes the ``Timeout (``
        # header and then calls ``faulthandler.dump_traceback``, which the module
        # itself measures at "hundreds of milliseconds" under load — a snapshot taken
        # the moment the marker appears can catch the walk half-written. That is not
        # hypothetical: CI run 36495158314 (PR #1712, attempts 1 and 2) read one whose
        # blocks stopped at the relay thread and named no frame of this process, and
        # the same read reproduces on this host under a chunked walk. The recording
        # line is the completion event for the walk — :func:`_record_held_fire`
        # appends it on the pass AFTER :func:`_fire` returned, and the sampler is the
        # only re-armer in this cell — so waiting for it is what makes the three
        # content assertions below read a finished dump rather than a prefix of one.
        assert _wait_for(
            lambda: any(
                line.startswith(stall_watchdog.HELD_MARKER) for line in _read(dump).splitlines()
            ),
            timeout=120.0,
        ), (
            "the dump does not say the fire was held, so a reader cannot tell that this "
            f"runtime survived it:\n{_read(dump)[-600:]!r}"
        )
        text = _read(dump)
        assert (
            stall_watchdog.PROGRESS_MARKER in text
        ), f"the fire does not name the leg that produced it:\n{text[-600:]!r}"
        # AN ALL-THREAD DUMP, and the frames have to be THIS process's: a dump that
        # names no frame of the running test is not the walk the fire promises,
        # unless the walk ended at CPython's own thread cap -- a shard-loaded
        # pytest worker holds 100+ threads, which is where PR #1695's CI went red
        # (see ``_walk_finished_in``).
        assert (
            "Thread 0x" in text or "Current thread" in text
        ), f"the fire wrote no thread dump:\n{text[-600:]!r}"
        assert _walk_finished_in(text), (
            "the dump neither names a frame of this process nor carries CPython's "
            "thread-cap ending, so it is not the finished all-thread walk the "
            f"bound's evidence is:\n{text[-600:]!r}"
        )
        # THE NEXT WAKE: the episode after the fire is re-armed for a whole bound, and a
        # sampler that stopped here would leave that deadline with nobody to take it --
        # one dump and silence, which is the class this fix is about.
        assert _wait_for(lambda: _fires_in(_read(dump)) >= 2, timeout=180.0), (
            "the sampler did not carry on to the next episode: the bound fired once and "
            f"then stopped being re-armed over {probe.passes} passes:\n{_read(dump)[-600:]!r}"
        )
    finally:
        stall_watchdog.disarm()


def test_a_walk_capped_by_cpython_reads_as_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CAP: a worker carrying a shard's worth of threads must not red the cell above.

    THE SHAPE THIS PINS comes from CI rather than from a reading: on PR #1695
    the cell above failed on three attempts (workers gw2, gw0 and gw1) with
    dumps that ended ``\\n...\\n`` and named no frame of this file -- CPython's
    ``_Py_DumpTracebackThreads`` caps its newest-to-oldest walk at 100 threads,
    and a worker that has carried a full shard's load holds more than that, so
    the main thread (the oldest, and the only carrier of this file's name) is
    never reached. This cell builds the capped condition deterministically --
    its own parked threads, newer than main -- and asserts both halves: the cap
    really lands (so this cell cannot pass on an uncapped walk), and
    :func:`_walk_finished_in`, the reading the cell above shares, accepts the
    capped dump as the finished evidence it is.

    THE THREADS ARE OURS AND RELEASED, because a pin to a cap must not become
    the next cell's soup: one Event parks them all and one ``set()`` wakes them.
    """
    gate = threading.Event()
    soup = [
        threading.Thread(target=gate.wait, name=f"cap-soup-{i}", daemon=True) for i in range(110)
    ]
    for thread in soup:
        thread.start()
    try:
        clock = _SteppingClock()
        monkeypatch.setattr(stall_watchdog, "time", clock)
        probe = _SteppingProbe(clock)
        dump = _armed_in(tmp_path, probe)
        try:
            assert _wait_for(lambda: _fires_in(_read(dump)) >= 1, timeout=120.0), (
                "the bound never fired under the cap rig, so this cell proved "
                f"nothing about a capped dump:\n{_read(dump)[-600:]!r}"
            )
            assert _wait_for(
                lambda: any(
                    line.startswith(stall_watchdog.HELD_MARKER) for line in _read(dump).splitlines()
                ),
                timeout=120.0,
            ), f"the fire was never recorded as held:\n{_read(dump)[-600:]!r}"
            text = _read(dump)
            assert any(line == "..." for line in text.splitlines()), (
                "the walk did not hit CPython's thread cap, so this cell proves "
                f"nothing about a capped dump:\n{text[-600:]!r}"
            )
            assert _walk_finished_in(text), (
                "a walk capped by CPython's own thread limit did not read as "
                "finished, so the cell above is red on any worker carrying 100 "
                f"threads:\n{text[-600:]!r}"
            )
        finally:
            stall_watchdog.disarm()
    finally:
        gate.set()
        for thread in soup:
            thread.join(timeout=5)


def test_a_lane_holding_a_step_is_spared_with_no_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SPARED DIRECTION, first arm: work in flight is not a spin.

    A probe answering ``in_flight`` restarts the run on every pass, which is what makes
    the widening safe — a lane mid-batch or parked in a provider call must not be cut by
    the leg that exists for loops going nowhere. What must NOT happen is the thing this
    fix could have broken in the other direction: the sampler stopping (no further look,
    so nothing is ever judged again) or a fire being recorded for a runtime that is
    working. The pass count is the first half of that, the empty dump the second.
    """
    clock = _SteppingClock()
    monkeypatch.setattr(stall_watchdog, "time", clock)
    probe = _SteppingProbe(clock, in_flight=True)
    dump = _armed_in(tmp_path, probe)
    try:
        assert _wait_for(lambda: probe.passes >= PASSES_BEFORE_VERDICT, timeout=120.0), (
            f"the sampler stopped looking after {probe.passes} passes, so a working "
            "runtime is no longer being judged at all"
        )
        text = _read(dump)
        assert (
            stall_watchdog.PROGRESS_MARKER not in text
        ), f"a lane holding a step was read as a spin:\n{text[-600:]!r}"
        assert (
            stall_watchdog.FIRED_MARKER not in text
        ), f"the bound fired over work in flight:\n{text[-600:]!r}"
        assert stall_watchdog.fired_pids(tmp_path) == set(), "a spared arm left a fire on disk"
    finally:
        stall_watchdog.disarm()


def test_an_unreadable_lane_is_spared_with_no_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SPARED DIRECTION, second arm: "I could not read it" is not "it is idle".

    The probe raises on every pass, which the module's own read turns into "no live
    second leg" — the abstention, not a claim that the work has stopped. Fail closed is
    the direction the widened read was asked for, and the sampler must keep looking
    while it abstains: a raise that ended the reporter would turn a corrupt registry
    into a runtime nothing is watching.
    """
    clock = _SteppingClock()
    monkeypatch.setattr(stall_watchdog, "time", clock)
    probe = _SteppingProbe(clock, raises=True)
    dump = _armed_in(tmp_path, probe)
    try:
        assert _wait_for(lambda: probe.passes >= PASSES_BEFORE_VERDICT, timeout=120.0), (
            f"the sampler stopped looking after {probe.passes} passes, so an unreadable "
            "lane leaves the runtime with no reporter at all"
        )
        text = _read(dump)
        assert (
            stall_watchdog.PROGRESS_MARKER not in text
        ), f"an unreadable lane was read as a spin:\n{text[-600:]!r}"
        assert (
            stall_watchdog.FIRED_MARKER not in text
        ), f"the bound fired on a read it could not make:\n{text[-600:]!r}"
        assert stall_watchdog.fired_pids(tmp_path) == set(), "a spared arm left a fire on disk"
    finally:
        stall_watchdog.disarm()


#: The bound the spinning child arms with, in seconds. The old suite's own choice, and
#: it keeps the cell's own wall time to a few bounds.
SPIN_BOUND_S = 2.0

#: How long the child gets to import this tree and boot its runtime before the cell
#: calls it a wedge. GENEROUS ON PURPOSE: importing the session runtime under fleet load
#: takes 40-75 s on this host (AGENTS.md, Environment), and a cell that timed out during
#: the import would be measuring the host. The wait is on the child's own ``armed:``
#: line, so a quick host never pays it.
SPIN_BOOT_S = 300.0

#: How long the cell then lets the child SPIN for. Several bounds, since the progress
#: leg's window is one bound: the loop burns CPU with no transcript, roster or job
#: movement for exactly this long, and nothing about the exit leg is under test here.
SPIN_WINDOW_S = 15.0

#: THE FLOOR IS LOWERED IN THE CHILD, and this is the one number in this cell that is
#: about the HOST rather than the contract. The module's 0.05 of a core is a claim about
#: a quiet machine: measured on this fleet (2026-09-24, load ~100 beside ~25 sessions) a
#: spinning child measures 0.017-0.021 of a core, i.e. BELOW the floor, so on that host
#: the predicate never fires and the cell would report "no dump" for a reason that has
#: nothing to do with the wake it is about. The burn is real either way — only the
#: threshold moves, and the subject here is who TAKES the deadline.
SPIN_FLOOR = 0.0005

#: A REAL runtime child that spins with no progress: a real session, a real
#: ``RuntimeServer``, the PRODUCTION probes (``process._progress_probe`` and
#: ``process._busy_probe``, supplied exactly as ``process.py`` supplies them), and a
#: loop that burns CPU while yielding on every pass so both planes keep their cadence.
#: The rig, not the product: it is the shape the old suite's ``_SPINNING_CHILD`` has,
#: with the floor above.
_SPINNING_CHILD = r"""
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, sys.argv[3])

from local_operator.harness.types import StreamEndEvent
from local_operator.session.runtime import process, server, stall_watchdog
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import make_session


def _stream(request, signal):
    async def gen():
        yield StreamEndEvent(stop_reason="stop")

    return gen()


async def main() -> None:
    root = pathlib.Path(sys.argv[2])
    process.HEARTBEAT_INTERVAL_S = 0.3
    server.HEARTBEAT_INTERVAL_S = 0.3
    stall_watchdog.PROGRESS_CPU_FLOOR = float(sys.argv[4])

    session = make_session(root, _stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"

    # THE PRODUCTION SEAM, not a double: the handle the entry point publishes and the
    # two probes the entry point arms with (``process.amain``).
    process._live_handle = handle
    assert stall_watchdog.arm(
        seconds=float(sys.argv[1]), probe=process._progress_probe, busy=process._busy_probe
    )
    print(f"armed:{os.getpid()}", flush=True)
    print(f"held-at-arm:{stall_watchdog._holds_work(process._busy_probe)}", flush=True)

    stop = asyncio.Event()
    asyncio.create_task(process._beat_stall_watchdog(stop))
    # BOTH PLANES HAVE RUN: the workload ticker stamped its plane and the serving plane
    # published a record. Without this the run says nothing about a process whose loops
    # were ALIVE, which is the only state the progress leg is about.
    await asyncio.sleep(1.0)
    print("both-planes-ran", flush=True)

    while True:
        await asyncio.sleep(0)
        sum(range(200_000))


asyncio.run(main())
"""


def test_a_spinning_runtime_gets_its_dump_and_its_fire(tmp_path: Path) -> None:
    """THE INCIDENT CLASS, on a real child: alive, ticking, and going nowhere.

    THE DEFECT THIS CELL IS THE REGRESSION FOR is the same one, in the arm where it is
    total: ``_sample`` returned ``not armed.held``, so with nothing in flight the
    sampler returned on the first progress pass and NOTHING was left in the process that
    could take the deadline it had just recorded. Measured on the pre-fix head (this rig,
    2026-09-24): one ``no progress`` line in the dump, zero ``Timeout (`` lines, zero
    thread stacks, and a child still burning a core when the cell killed it 15 s later
    — the class was reported by one line and then not bounded or recorded at all.

    WHAT THE FIX HAS TO SHOW HERE is the fire, not a survival: the deadline is taken by
    the sampler's next pass and the dump is written. Which leg the fire then ends on is
    ``_fire``'s own reading and is unchanged by this fix — on this head an arm with
    nothing in flight is the runtime's documented wedge recovery, so the child leaves
    with rc 1 (that is T1's policy on this branch, and #1463's dump-only fold is what
    changes it, not this cell).

    THE CONTINUED WAKE IS THE OTHER CELL'S SUBJECT, deliberately: this arm's runtime
    does not survive its fire on this head, so "keeps waking" cannot be observed here —
    see ``test_the_progress_deadline_is_taken_before_the_predicate_can_pre_empt_it``.
    """
    root = tmp_path / "spin"
    root.mkdir(parents=True, exist_ok=True)
    script = root / "spin_child.py"
    script.write_text(_SPINNING_CHILD, encoding="utf-8")

    child = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
        [
            sys.executable,
            str(script),
            str(SPIN_BOUND_S),
            str(root),
            str(Path(__file__).resolve().parents[4]),
            str(SPIN_FLOOR),
        ],
        env=_child_env(root),
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []

    def _drain_output() -> None:
        """Collect the child's output on its own thread, so no wait can block on it.

        A reader that ran in the cell's own thread would park in ``readline`` on exactly
        the arm this cell exists to red: a child that never fires never prints again, and
        a cell blocked in a read has no way to reach its own verdict.
        """
        assert child.stdout is not None
        for line in child.stdout:
            lines.append(line.rstrip("\n"))

    reader = threading.Thread(target=_drain_output, name="spin-child-output", daemon=True)
    reader.start()
    try:

        def armed_pid() -> int | None:
            for line in list(lines):
                if line.startswith("armed:"):
                    return int(line.split(":", 1)[1].strip())
            return None

        assert _wait_for(
            lambda: armed_pid() is not None or child.poll() is not None, timeout=SPIN_BOOT_S
        ), (
            f"the child neither armed its bound nor exited within {SPIN_BOOT_S}s "
            f"(rc={child.poll()}): {lines[-8:]!r}"
        )
        pid = armed_pid()
        assert (
            pid is not None
        ), f"the child exited before it armed its bound (rc={child.poll()}): {lines[-8:]!r}"
        dump = stall_watchdog.dump_path(pid, root / "logs")

        # THE SPIN: bounded, and the bound is the cell's, not the module's. Waiting on
        # the child's own exit OR the window, never on a timer alone: a child that takes
        # its fire in under the window ends the wait early.
        # The result is deliberately unused: this is the WINDOW, and a child that is
        # still up when it closes is the reading (it is what the pre-fix arm does).
        _wait_for(lambda: child.poll() is not None, timeout=SPIN_WINDOW_S)
        exited = child.poll() is not None
        if not exited:
            child.kill()
        child.wait(timeout=30.0)
        reader.join(timeout=15.0)
        text = _read(dump)

        assert "both-planes-ran" in "\n".join(lines), (
            "the child's planes never ran, so this says nothing about a process whose "
            f"loops were alive: {lines[-8:]!r}"
        )
        if not exited:
            assert stall_watchdog.PROGRESS_MARKER in text, (
                "this rig did not even reach the progress predicate, so it is not "
                f"measuring the wake: {text[-600:]!r}"
            )
        assert stall_watchdog.FIRED_MARKER in text, (
            "the progress leg named the stall and then wrote no fire at all: the loop "
            "that burns a core with no progress is not bounded and its evidence is never "
            f"written. Waited {SPIN_WINDOW_S}s past the arm, dump holds "
            f"{text.count(stall_watchdog.PROGRESS_MARKER)} progress line(s):\n{text[-800:]!r}"
        )
        assert (
            stall_watchdog.PROGRESS_MARKER in text
        ), f"the fire does not name the leg that produced it:\n{text[-800:]!r}"
        assert (
            "Thread 0x" in text or "Current thread" in text
        ), f"the fire wrote no all-thread dump:\n{text[-800:]!r}"
        assert script.name in text, (
            "the dump does not name the spinning loop's own frame, so the all-thread walk "
            f"the fire promises is not in it:\n{text[-800:]!r}"
        )
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30.0)
