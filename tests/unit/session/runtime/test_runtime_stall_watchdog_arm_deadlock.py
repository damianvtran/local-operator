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
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
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
#: repeatedly into a dump channel the parent stops draining the moment it has seen the
#: fire, so the C thread's write cannot complete and that dump is in flight for good.
#: Only then does the child touch the module's arm path — and the module's own dump
#: goes to its own directory, so nothing here depends on the blocked channel.
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

faulthandler.dump_traceback_later(0.05, repeat=True, file=sink, exit=False)
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
    """

    def __init__(self, tmp_path: Path, mode: str, *, keep_draining: bool) -> None:
        self.keep_draining = keep_draining
        self.lines: list[str] = []
        self.script = tmp_path / f"arm_path_child_{mode}.py"
        self.script.write_text(_ARM_PATH_CHILD, encoding="utf-8")
        self.go = tmp_path / f"go-{mode}"
        self.dump_dir = tmp_path / f"dumps-{mode}"
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
            env=_child_env(tmp_path),
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(write_end.fileno(),),
        )
        write_end.close()
        self.reader = threading.Thread(target=self._read_stdout, name=f"out-{mode}", daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line.rstrip("\n"))

    def wait_for_fire_then_release(self) -> bool:
        """Stop reading once the dump has really started; then say GO.

        The fire is published on the channel itself, so this is an event rather than a
        guess: the child must not touch the arm path until a dump is demonstrably in
        flight, and only the parent can see that.
        """
        seen = b""
        deadline = time.monotonic() + PUBLISHED_S
        self.read_end.settimeout(0.5)
        while b"Timeout (" not in seen and time.monotonic() < deadline:
            try:
                seen += self.read_end.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
        fired = b"Timeout (" in seen
        if fired and self.keep_draining:
            threading.Thread(target=self._drain_forever, name="drain", daemon=True).start()
        if fired:
            self.go.write_text("go", encoding="utf-8")
        return fired

    def _drain_forever(self) -> None:
        try:
            while self.read_end.recv(65536):
                pass
        except OSError:
            pass

    def wait_for(self, marker: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(marker in line for line in self.lines):
                break
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        return "\n".join(self.lines)

    def close(self) -> None:
        """Kill THIS pid and reap it; close the channel. Never a bare pgrep."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10.0)
        self.read_end.close()


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

    MUTATION THIS CELL CATCHES: any arm path that reaches a C timer call again -> red.
    """
    control = _ArmPathChild(tmp_path, "control", keep_draining=True)
    try:
        assert control.wait_for_fire_then_release(), (
            "the control child never produced a fired dump on its channel, so this rig "
            f"is not measuring what it claims; output was {control.wait_for('TICK', 1.0)!r}"
        )
        control_output = control.wait_for("LOOP-SURVIVED", CHILD_BOUND_S)
        assert "LOOP-SURVIVED" in control_output, (
            "the CONTROL run did not survive with its dump channel drained, so the "
            f"channel is not the difference this cell is about; output was "
            f"{control_output!r}"
        )
    finally:
        control.close()

    rig = _ArmPathChild(tmp_path, "wedge", keep_draining=False)
    try:
        assert rig.wait_for_fire_then_release(), (
            "no dump was ever seen in flight, so this cell proved nothing about an arm "
            f"path taken while one is; output was {rig.wait_for('TICK', 1.0)!r}"
        )
        output = rig.wait_for("LOOP-SURVIVED", CHILD_BOUND_S)
    finally:
        pid = rig.process.pid
        rig.close()
    assert "ARM-PATH-RETURNED" in output, (
        f"the arm path never returned while a dump was in flight (child pid {pid} was "
        f"killed after {CHILD_BOUND_S}s). The child stopped after: "
        f"{output.splitlines()[-4:]!r} — a ticker that goes silent at GO is the field "
        f"signature: the caller parked inside the C timer call holding the GIL, so no "
        f"Python thread in the process could run"
    )
    assert "LOOP-SURVIVED" in output, output


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
        _wait_for_record(config_dir)
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
