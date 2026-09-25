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
        text = _read(dump)
        assert (
            stall_watchdog.PROGRESS_MARKER in text
        ), f"the fire does not name the leg that produced it:\n{text[-600:]!r}"
        # AN ALL-THREAD DUMP, and the frames have to be THIS process's: a dump that
        # names no frame of the running test is not the walk the fire promises.
        assert (
            "Thread 0x" in text or "Current thread" in text
        ), f"the fire wrote no thread dump:\n{text[-600:]!r}"
        assert Path(__file__).name in text, (
            "the dump does not name a frame of this process, so it is not the all-thread "
            f"walk the bound's evidence is:\n{text[-600:]!r}"
        )
        assert _wait_for(
            lambda: any(
                line.startswith(stall_watchdog.HELD_MARKER) for line in _read(dump).splitlines()
            ),
            timeout=120.0,
        ), (
            "the dump does not say the fire was held, so a reader cannot tell that this "
            f"runtime survived it:\n{text[-600:]!r}"
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
