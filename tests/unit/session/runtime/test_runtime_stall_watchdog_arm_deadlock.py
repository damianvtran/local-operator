"""The stall watchdog's OWN arming call is a wedge risk, and this is its repro.

THE FIELD SIGNATURE, verbatim. Two live runtimes on this host (pids sampled with
``sample <pid> 3``; 0.62.8 and 0.62.15, different builds, different work) show the
SAME thing: the runtime's MAIN (event-loop) thread parks inside the watchdog's own
re-arm -- ``faulthandler_dump_traceback_later`` -> ``cancel_dump_traceback_later``
-> ``PyThread_acquire_lock_timed`` -> ``__psynch_cvwait``, reached from
``task_step`` (an ``asyncio`` task) -- WHILE HOLDING THE GIL, and the thread named
``stall-watchdog-progress`` cannot take it (``lock_PyThread_acquire_lock`` ->
``_PyParkingLot_Park`` -> ``_PyThreadState_Attach`` -> ``take_gil``). The session
wedges, the watchdog's own sampler starves, and the timer can never finish what it
was doing, so no dump is ever written.

WHY THE CALL CANNOT RETURN, measured here rather than assumed. CPython's
``faulthandler_dump_traceback_later`` cancels a previous timer before arming the
next, and that cancel WAITS for an in-flight dump to finish -- with the GIL held,
in whatever thread called it (``PyThread_acquire_lock_timed`` on the trace above).
So an arm/re-arm is only safe while no dump is in flight; if one is, the caller
parks holding the GIL and the whole process starves. Two things follow, and this
file pins both:

* (a) no C-timer call may be MADE on a thread whose parking wedges the session --
  the event-loop thread, the plane thread that beat, or the sampler;
* (b) while such a call is in flight, the watchdog's own gate must stay open, so
  the sampler can still observe and the exit leg can still be re-read.

WHAT IS PROVEN DETERMINISTICALLY HERE, and what is not, stated plainly:

* (a) is proven as a STRUCTURAL FACT -- thread identity, which cannot flake: every
  call into the timer API is recorded with the thread that made it, and the loop's
  own beat is driven through the real seams. This is the same shape
  ``test_store_maintenance_callbacks_run_off_the_event_loop_thread`` uses, and for
  the same reason (see AGENTS.md, "Prefer a structural invariant to a numeric one").
* (b) is proven as a STATE fact in-process: ``_LOCK`` is the single gate every
  sample and every beat passes through, the field trace shows the sampler parked
  exactly there, and a parked timer call holds it -- so the cell asserts the gate
  is free while the call is in flight. The end-to-end form of (b) -- the process's
  Python threads actually stopping -- is proven in the child cell below, against
  the REAL C API, where it is observable at all only from outside the process.
* NOT proven here, and not claimed: WHY a field dump failed to finish. This file
  forces the precondition (a dump that cannot complete) rather than explaining it.
  The forcing is the rig's own substitution -- the dump is written to a channel
  nobody drains -- and the child cell carries a draining control that shows the
  channel is the only difference between "the loop survives" and "the loop wedges".

WHAT IS NOT FIXED HERE. These cells are written to go RED on the pre-fix source
and are left red on purpose: the fix is a design decision about who owns the C
timer (a dedicated arming thread, or a policy that never replaces a pending timer
from a loop thread), and it belongs to the architect's memo on the same defect.
Whoever lands that fix flips these cells green; nothing here merges as a green
suite until then.
"""

from __future__ import annotations

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

#: The name ``_start_sampler`` gives the sampler thread -- the thread the field
#: trace shows unable to take the GIL, and therefore the second thread the timer
#: API must never park.
SAMPLER_NAME = "stall-watchdog-progress"

#: How long a cell waits for a publication from the code under test before it
#: reports a wedge rather than a slow host. A backstop, never the assertion: the
#: assertions below are about WHICH THREAD made a call and WHICH LOCK is free.
PUBLISHED_S = 10.0

#: How long the stand-in timer call in the gate cell parks. Only the teardown
#: waits this long, and it is released earlier by the cell's own ``finally``.
PARK_HOLD_S = 30.0

#: How long the loop thread gets to come back from ONE timer call in the child
#: cell before the cell calls it wedged. Generous on purpose: the child's own
#: ticks are what say whether Python is running at all, so this bound only has to
#: separate "returned" from "never".
CHILD_CALL_BOUND_S = 20.0

#: The socket buffer the never-drained dump channel is given, in bytes. The dump
#: of a single parked thread is already larger than this, so the C thread's write
#: blocks on its own -- measured: with this buffer and no reader, ZERO bytes of
#: the dump reach the other end, i.e. the write does not even partly complete.
CHANNEL_SNDBUF = 1024


@pytest.fixture(autouse=True)
def _no_leaked_arm() -> Iterator[None]:
    """Module-level state is process-global: never let one cell arm the next."""
    stall_watchdog.disarm()
    yield
    stall_watchdog.disarm()


def _child_env(config_dir: Path, **extra: str) -> dict[str, str]:
    """A child environment that can only touch ``config_dir``.

    Every inherited ``CMUX_*``/``LOP_*``/``HERDR_*`` variable is stripped: this
    suite is routinely run from inside an operator session whose own values would
    otherwise be inherited, and ``LOCAL_OPERATOR_CONFIG_DIR`` alone is not enough
    (AGENTS.md, "Isolating a run").
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))}
    env["HOME"] = str(config_dir)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env.update(extra)
    return env


class _ThreadRecordingFaulthandler:
    """Records WHICH THREAD each C-timer call is made on. Touches no real timer.

    The defect is a property of the CALLING THREAD, not of the arguments, so a spy
    that lands on thread identity is the whole instrument. A real timer cannot be
    driven into its blocked replace inside a pytest worker (that is the wedge), so
    the structure is spied here and the real C behaviour is exercised in the child
    cell below.
    """

    def __init__(self) -> None:
        #: ``(kind, thread ident, thread name)`` in call order.
        self.calls: list[tuple[str, int, str]] = []
        self.called = threading.Event()

    def dump_traceback_later(self, seconds: float, **kwargs: Any) -> None:
        self._record("arm")

    def cancel_dump_traceback_later(self) -> None:
        self._record("cancel")

    def _record(self, kind: str) -> None:
        self.calls.append((kind, threading.get_ident(), threading.current_thread().name))
        self.called.set()


def _timer_calls_on(calls: list[tuple[str, int, str]], idents: set[int], names: set[str]):
    """The recorded calls made by any of these threads -- the offenders, if any."""
    return [c for c in calls if c[1] in idents or c[2] in names]


def test_the_c_timer_is_never_armed_from_a_thread_whose_park_wedges_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROPERTY (a): the timer API is only ever called from a dedicated thread.

    THE DEFECT THIS CELL IS RED FOR. ``beat`` and ``arm`` call ``_rearm`` ->
    ``_arm_timer`` -> ``faulthandler.dump_traceback_later`` on the CALLER'S thread,
    and the caller is the workload's own event loop (``process._beat_stall_watchdog``
    is an ``asyncio`` task on it). While a dump is in flight that call parks holding
    the GIL -- which is exactly the ``sample`` trace both preserved specimens carry:
    ``task_step`` -> ``faulthandler_dump_traceback_later`` ->
    ``cancel_dump_traceback_later`` -> ``PyThread_acquire_lock_timed``. So the cell
    drives the real seams from this thread (which is what the runtime's loop is) and
    asserts that none of the calls the module makes lands on it.

    WHY THREAD IDENTITY AND NOT A STOPWATCH: a timing bound here would be a bet on
    the fleet's load, and the property is a fact about where the code ran. This is
    the pattern AGENTS.md names as the strongest form of "this work happened off
    the event loop".

    THE SAMPLER IS NAMED TOO, and it is a deliberate demand rather than a mirror of
    today's code: a design that parks the timer call on ``stall-watchdog-progress``
    would keep the GIL free (so it would pass a loop-only assertion) and still be a
    defect -- that thread is the only Python leg left when everything else is
    parked, and while it sits in the timer API it can neither sample, nor re-read
    the exit leg, nor record a held fire, so the bound it is holding fires on a
    runtime that is working. One dedicated thread for the timer, or none of this.

    MUTATION THIS CELL CATCHES: route the arming back onto the caller's thread (the
    shape on the base ref) -> red. Move it to a thread that is not the loop's ->
    green.
    """
    recorder = _ThreadRecordingFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", recorder)

    # THIS thread plays the runtime's event loop: it arms the process (the entry
    # point's own thread) and beats the workload plane, which is definitionally
    # the loop's thread (see `beat`).
    loop_thread = threading.current_thread()
    tried: list[str] = []

    def a_probe() -> tuple[object, bool]:
        return (len(tried), True)  # motion that changes: a live, answering probe

    armed = stall_watchdog.arm(seconds=30.0, directory=tmp_path, probe=a_probe)
    tried.append("arm")
    if armed:
        stall_watchdog.beat(stall_watchdog.WORKLOAD)
        tried.append("beat")

    # The publication is the spy being called at all: a module that armed nothing
    # is the OTHER defect (an unbounded runtime), and it must fail here rather than
    # pass a "no call was made" assertion vacuously.
    assert recorder.called.wait(PUBLISHED_S), (
        f"no call reached the timer API within {PUBLISHED_S}s of {tried}; the runtime "
        f"has no bound, which this cell treats as a failure rather than as a pass"
    )

    offenders = _timer_calls_on(recorder.calls, {loop_thread.ident or -1}, {SAMPLER_NAME})
    assert not offenders, (
        f"the C timer was called on the session's own thread: {offenders} "
        f"(loop thread is {loop_thread.name!r}/{loop_thread.ident}, calls so far "
        f"{recorder.calls}). A dump in flight makes that call park while holding the "
        f"GIL, which is the wedge both preserved specimens are parked in -- the arming "
        f"call has to be made by a thread whose parking costs the session nothing"
    )
    harness_calls = [c for c in recorder.calls if c[2] == "MainThread"]
    assert harness_calls == [], (
        f"the pytest main thread reached the timer API: {harness_calls}; in a runtime "
        f"that thread is the event loop"
    )


class _ParkingFaulthandler:
    """A stand-in for the one C behaviour that matters: the call does not return.

    ``dump_traceback_later`` parses no timer here -- it parks until the cell
    releases it, which is what a replace does while a dump is in flight. Parking on
    an ``Event`` RELEASES the GIL, deliberately: it isolates the second half of the
    defect (the module's own gate) from the GIL half, which the child cell below
    exercises against the real API.
    """

    def __init__(self, gate: threading.Event, parked: threading.Event) -> None:
        self.gate = gate
        self.parked = parked
        self.cancels = 0

    def dump_traceback_later(self, seconds: float, **kwargs: Any) -> None:
        self.parked.set()
        self.gate.wait(PARK_HOLD_S)

    def cancel_dump_traceback_later(self) -> None:
        self.cancels += 1


def test_the_watchdogs_gate_is_free_while_a_timer_call_is_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROPERTY (b): a parked timer call must not hold the watchdog's own gate.

    THE FIELD TRACE THIS PINS, frame for frame: the thread named
    ``stall-watchdog-progress`` is parked in ``lock_PyThread_acquire_lock`` ->
    ``_PyParkingLot_Park`` -> ``_PyThreadState_Attach`` -> ``take_gil``. It is
    waiting on a PYTHON lock -- ``_LOCK``, which every sample takes (twice per
    wake: around the probe read and around the whole of ``_sample``) -- held by the
    loop thread across its timer call. So even on a hypothetical build where the
    GIL were free, the watchdog could not observe: the leg that reports the stall
    is itself stopped by the arming path.

    WHAT IS ASSERTED, and why it is not a timing claim: the gate is free WHILE the
    call is provably in flight (the stand-in publishes ``parked`` before it parks).
    The only wait is for that publication; there is no elapsed-time comparison on
    the success path.

    MUTATION THIS CELL CATCHES: take ``_LOCK`` across the timer call (the shape on
    the base ref: ``beat`` holds it from the stamp through ``_rearm``) -> red.
    Perform the call outside the gate, or on a thread of its own -> green.
    """
    gate = threading.Event()
    parked = threading.Event()
    stand_in = _ParkingFaulthandler(gate, parked)
    loop: list[threading.Thread] = []

    # The state is armed through a pass-through first, so the cell exercises a
    # RE-ARM (the production shape: `beat` on an armed runtime) rather than the
    # boot arm, and so the sampler exists.
    recorder = _ThreadRecordingFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", recorder)
    assert (
        stall_watchdog.arm(seconds=30.0, directory=tmp_path, probe=lambda: (0.0, True)) is True
    ), "the cell could not arm, so it has nothing to re-arm"
    monkeypatch.setattr(stall_watchdog, "faulthandler", stand_in)

    def beat_from_the_loop() -> None:
        stall_watchdog.beat(stall_watchdog.WORKLOAD)

    loop_thread = threading.Thread(target=beat_from_the_loop, name="loop-proxy", daemon=True)
    loop.append(loop_thread)
    loop_thread.start()
    try:
        assert parked.wait(PUBLISHED_S), (
            "the beat never reached the timer API, so this cell proved nothing about a "
            "call in flight"
        )
        # THE GATE ITSELF, read while the call is in flight: `_LOCK` is what every
        # sample and every beat passes through, so a gate that cannot be taken IS
        # the sampler being unable to observe (its own park, frame for frame).
        acquired = stall_watchdog._LOCK.acquire(timeout=2.0)
        try:
            assert acquired, (
                "the watchdog's gate is held while a timer call is in flight, so no "
                "sample can complete: this is the 'stall-watchdog-progress' half of the "
                "field trace, and it means the leg that exists to report a stall is "
                "itself stopped by the arming path"
            )
        finally:
            if acquired:
                stall_watchdog._LOCK.release()
    finally:
        # Un-park the stand-in BEFORE teardown: `disarm` takes the same gate, and a
        # wedge left here would hang the suite rather than fail a cell.
        gate.set()
        for thread in loop:
            thread.join(timeout=PARK_HOLD_S)


#: The fire the parent watches for on the dump channel: proof, from OUTSIDE the
#: process, that the C timer really fired and is writing into the channel. Nothing
#: inside the child can prove this once it wedges.
FIRE_HEADER = b"Timeout ("

#: The file the parent creates to say "the dump is in flight -- make the calls".
#: A publication rather than a sleep: the child does not guess when the dump landed.
GO_MARKER_S = 60.0

#: The child that drives the REAL C API into the wedge. Its dump goes to a channel
#: the parent stops draining the moment it has seen the fire, which is the
#: substituted precondition: a dump that cannot finish. ``repeat=True`` is what
#: keeps it in that state -- the C thread re-arms itself internally, so the channel
#: stays busy without any Python call.
_TIMER_WEDGE_CHILD = """
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
go_bound = float(sys.argv[4])

# Parked, DEEP frames: the all-thread dump is far larger than any socket buffer,
# so the C thread's first write blocks the moment nobody is reading.
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

# THE PRECONDITION, forced: a timer that fires repeatedly into the dump channel.
# The parent stops draining as soon as it has seen the fire, so the C thread's
# write cannot complete and the dump stays in flight -- nothing in Python has to
# guess that it happened.
faulthandler.dump_traceback_later(0.05, repeat=True, file=sink, exit=False)
print(f"TIMER-ARMED {mode}", flush=True)

deadline = time.monotonic() + go_bound
while not go.exists():
    if time.monotonic() > deadline:
        print("NO-GO", flush=True)
        raise SystemExit(0)
    time.sleep(0.01)
print("GO", flush=True)

# THE CALL UNDER TEST, made on this process's loop thread (its main thread, which
# is where the runtime's event loop runs) through the module's single spelling for
# the C call. Each of these replaces the pending timer -> it must cancel the dump
# in flight first.
for i in range(20):
    stall_watchdog._arm_timer(sink, 30.0, exit_leg=False)
    print(f"CALL {i} RETURNED", flush=True)
    time.sleep(0.05)
print("LOOP-SURVIVED", flush=True)
"""


class _WedgeChildRun:
    """One run of the child, with the dump channel driven from the parent side.

    ``keep_draining`` decides the whole cell: ``True`` is the CONTROL, where the
    parent keeps reading so the dump completes and nothing can wedge; ``False`` is
    the rig, where the parent stops reading the moment it has SEEN the fire, so the
    dump cannot finish. The only difference between the two runs is whether the
    dump completes -- which is what makes the result a measurement of that rather
    than of the rig.
    """

    def __init__(self, tmp_path: Path, mode: str, *, keep_draining: bool) -> None:
        self.mode = mode
        self.keep_draining = keep_draining
        self.lines: list[str] = []
        self.script = tmp_path / f"timer_wedge_child_{mode}.py"
        self.script.write_text(_TIMER_WEDGE_CHILD, encoding="utf-8")
        self.go = tmp_path / f"go-{mode}"
        self.read_end, write_end = socket.socketpair()
        # The dump of one parked, 250-frame-deep thread is already far larger than
        # this buffer, so the C thread's write cannot be satisfied in one go.
        write_end.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CHANNEL_SNDBUF)
        os.set_inheritable(write_end.fileno(), True)
        self.process = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            [
                sys.executable,
                str(self.script),
                str(write_end.fileno()),
                mode,
                str(self.go),
                str(GO_MARKER_S),
            ],
            env=_child_env(tmp_path),
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(write_end.fileno(),),
        )
        write_end.close()
        self.reader = threading.Thread(target=self._read_stdout, name="child-out", daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line.rstrip("\n"))

    def wait_for_fire_then_release(self) -> bool:
        """Stop draining once the dump has really started; then say GO.

        The fire is published on the channel itself, so this is an event rather
        than a guess: the child must not make the calls under test until a dump is
        demonstrably in flight, and only the parent can see that.

        With ``keep_draining`` the reader keeps going instead -- the control -- and
        the go marker is still only sent after the fire is seen, so both runs make
        the calls at the same point in the child's life.
        """
        seen = b""
        deadline = time.monotonic() + PUBLISHED_S
        self.read_end.settimeout(0.5)
        while FIRE_HEADER not in seen and time.monotonic() < deadline:
            try:
                seen += self.read_end.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
        fired = FIRE_HEADER in seen
        if fired and self.keep_draining:
            threading.Thread(target=self._drain_forever, name="dump-drain", daemon=True).start()
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
        """Wait (bounded) for a marker the child prints; return the output either way."""
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


def test_a_timer_call_made_while_a_dump_is_in_flight_cannot_freeze_the_loop(
    tmp_path: Path,
) -> None:
    """PROPERTY (b), end to end and against the REAL C API: the process must live.

    The child arms a timer whose dump can never complete (the channel is never
    drained) and then calls the module's single spelling for the C call twenty
    times from its own loop thread. On the base ref the first call that lands
    after the dump is in flight never returns -- and because it holds the GIL,
    every Python thread in the child stops, which is what ``TICK`` going silent
    shows. The child is the only place this is observable at all: with the GIL
    held there is no Python left inside the process to observe with.

    ASSERTED: the child reached ``LOOP-SURVIVED``. What that means is the whole
    property -- the loop thread came back from every call while a dump was in
    flight, so the watchdog's arming path cannot wedge a runtime.

    THE CONTROL, same child, same bound, channel drained: it reaches
    ``LOOP-SURVIVED`` on any build. Without it a green run would also be explained
    by the rig never having produced an in-flight dump at all.

    MUTATION THIS CELL CATCHES: make the arming call again from the loop thread
    while a dump is in flight (the shape on the base ref) -> red, with the child's
    ``TICK`` line and its last ``CALL n RETURNED`` as the evidence of where it
    stopped.
    """
    control = _WedgeChildRun(tmp_path, "control", keep_draining=True)
    try:
        assert control.wait_for_fire_then_release(), (
            "the control child never produced a fired dump on its channel, so this rig "
            f"is not measuring what it claims; output was {control.wait_for('TICK', 1.0)!r}"
        )
        control_output = control.wait_for("LOOP-SURVIVED", CHILD_CALL_BOUND_S)
        assert "LOOP-SURVIVED" in control_output, (
            "the CONTROL run did not survive with its dump channel drained, so the "
            "channel is not the difference this cell is about; its output was "
            f"{control_output!r}"
        )
    finally:
        control.close()

    rig = _WedgeChildRun(tmp_path, "wedge", keep_draining=False)
    try:
        assert rig.wait_for_fire_then_release(), (
            "no dump was ever seen in flight, so this cell proved nothing about a call "
            f"made while one is; output was {rig.wait_for('TICK', 1.0)!r}"
        )
        output = rig.wait_for("LOOP-SURVIVED", CHILD_CALL_BOUND_S)
    finally:
        pid = rig.process.pid
        rig.close()
    assert "LOOP-SURVIVED" in output, (
        f"the loop thread never came back from the arming call with a dump in flight "
        f"(child pid {pid} was killed after {CHILD_CALL_BOUND_S}s). The child stopped "
        f"after: {output.splitlines()[-4:]!r} -- TICK silent between two CALL lines is "
        f"the field signature: the loop thread parked inside the C timer call holding "
        f"the GIL, so no Python thread in the process can run"
    )
