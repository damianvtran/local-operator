"""The runtime bounds its OWN stall: a C-thread dump, not a departure.

WHY THESE TESTS ARE SHAPED THIS WAY, in one read.

The defect they exist for is a *hang*, not a slow function: five runtimes on
2026-09-20 were parked for 1.5-7.2 h inside a C-level scan, every Python-level
instrument in the process was dead by construction (a thread watchdog needs the
GIL a wedged thread never releases; a signal handler needs bytecodes; an
asyncio timer needs the loop), and nothing named the line. So the mechanism
under test is ``faulthandler.dump_traceback_later`` — armed in a C thread — and
a green test that never occupies the loop would prove nothing about it.

THE REPRODUCTION PARKS THE LOOP ON PURPOSE, and it claims NOTHING about why a
real runtime parks. An earlier revision shipped a fixture that replayed a
manager's child transcripts through the credential-shape pass, on the theory that
``hub resume`` scrubs them on the way back in; review measured that a resume over
a 20 MB credential-shaped store costs 1.27 s of boot against 0.97 s for a 1-row
store, i.e. **resume does not scrub**, and six real runtime passes that burned up
to 60.4 s of turn CPU while ``beat_lag_s`` stayed at 15.00 s never came near the
bound. So the fixture is gone rather than dressed up: what the five frozen
sessions proved is that a loop was parked in a C scan for hours, and what this
file tests is the instrument and the policy for that — a deliberately parked loop
(a GIL-holding C call, and separately the reviewer's busy-but-yielding
``re.search`` loop with a HEALTHY serving plane) — with the trigger left to the
measurement that will name it. No claim in this file, or in the PR, depends on a
caller being identified.

AND A SPY on the C timer (``_FakeFaulthandler``) covers the structure a real
fired timer cannot be asked about in-process: that every beat RE-ARMS the bound,
that the header is on disk BEFORE the timer is armed, and that arming is
idempotent rather than leaking a second handle.

The bound in the child runs is set through the same environment variable an
operator uses (``LOP_RUNTIME_STALL_SECONDS``), so the production path — read the
bound, arm, beat — is the path under test, at 1-2 s instead of 300.

AND THE INTERLOCK, which is the part a mistake here would remove silently: the
``faulthandler`` timer is process-global and shared with the e2e stage's
``bounded`` blocks and the shard watchdog, so this module must never be armed
from a library path a pytest process can reach. That is asserted as a fact about
the SOURCE (exactly one ``arm`` call, inside ``process.__main__``), as a fact
about BEHAVIOUR (calling ``process.main()`` in-process arms nothing), and on a
REAL spawned runtime (which arms, and which disarms on a clean stop) — see
``test_an_in_process_entry_point_arms_nothing``,
``test_the_only_arm_site_is_the_runtime_entry_point`` and
``test_a_real_runtime_child_arms_its_bound_and_disarms_on_a_clean_stop``.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import incidents
from local_operator.harness.types import (
    AgentTool,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.runtime import process as process_module
from local_operator.session.runtime import stall_watchdog
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import ScriptedStream, make_session

#: The worktree root. ``parents[4]`` because this file lives four levels under it
#: (``tests/unit/session/runtime/``).
REPO = Path(__file__).resolve().parents[4]

#: A value that exists ONLY as a Python local in the parked child. It must never
#: appear in a dump: ``faulthandler`` prints frames, never variables, which is
#: the property that lets this file be written into a log directory at all.
LOCAL_SENTINEL = "sentinel-that-must-never-appear-in-a-dump"

#: The bound the child runs use. Long enough that the arming, the file writes and
#: the C timer's own setup cannot race it on a loaded host, short enough that a
#: fired bound is a test that finishes in seconds.
CHILD_BOUND_S = 2

#: The bound the single-stall children run under: one second, which is far below
#: anything an operator may set (the environment floors at ``MIN_BOUND_S``, three
#: heartbeat intervals) and is exactly why these children pass the bound on
#: ``argv`` and arm with ``arm(seconds=...)`` instead of the environment: the
#: mechanism is bound-agnostic, and making every run wait 45 s for the floor would
#: buy no coverage at all.
SHORT_BOUND_S = 1

#: The pid the recycle cell drives BOTH of its lives through. The kernel will not
#: reissue a pid on demand and it does not have to: these files are named by pid
#: alone, so handing the same pid to two children IS the code path a real recycle
#: takes -- which is how QA round 1 drove Q1 too.
SYNTH_PID = 424242


@pytest.fixture(autouse=True)
def _no_leaked_arm() -> Iterator[None]:
    """Module-level state is process-global: never let one cell arm the next."""
    stall_watchdog.disarm()
    yield
    stall_watchdog.disarm()


class _FakeFaulthandler:
    """Spies on the C timer without touching the real one.

    A real timer cannot be fired safely inside a pytest worker — its effect is
    process-global and can disrupt pytest — and asking it whether it is armed is
    not something ``faulthandler`` answers. So the structure is spied here and the FIRING is
    proven in a real child process below. The fake reads the dump file's own
    text at arm time, which is what pins write-before-arm as an ordering fact
    rather than as a comment.
    """

    def __init__(self) -> None:
        self.armed: list[tuple[float, bool, Path]] = []
        self.cancels = 0
        #: The file's contents AT THE MOMENT the timer was armed.
        self.text_at_arm = ""

    def dump_traceback_later(self, seconds: float, **kwargs: Any) -> None:
        handle = kwargs["file"]
        path = Path(handle.name)
        self.text_at_arm = path.read_text(encoding="utf-8")
        self.armed.append((seconds, bool(kwargs.get("exit")), path))

    def cancel_dump_traceback_later(self) -> None:
        self.cancels += 1


def _child_env(config_dir: Path, **extra: str) -> dict[str, str]:
    """A child environment that can only touch ``config_dir``.

    Every inherited ``LOP_*``/``CMUX_*``/``HERDR_*`` variable is stripped first:
    this suite is routinely run from inside an operator session whose own values
    would otherwise be inherited, and ``LOCAL_OPERATOR_CONFIG_DIR`` alone is not
    enough (see AGENTS.md, "Isolating a run").
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))}
    env["HOME"] = str(config_dir)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    env.update(extra)
    return env


def _run_script(
    script: str,
    config_dir: Path,
    args: tuple[str, ...] = (),
    *,
    timeout: float = 90.0,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a self-terminating child as a real file for attributable dump frames.

    A file rather than ``python -c``: under ``-c`` every frame reads
    ``File "<string>"``, which would leave the dump's own evidence — which source
    line was parked — unattributable, and attributing it is the whole point.
    """
    path = config_dir / "parked_child.py"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(path), *args],
        env=_child_env(config_dir, **(env_extra or {})),
        cwd=str(config_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


#: One COMPLETE thread header, exactly as ``faulthandler`` writes it
#: (``Thread 0x000000016c147000 (most recent call first):``). A file cut mid-write
#: ends at the bare ``Thread 0x`` of the next header, which is what a reader keyed on
#: ``FIRED_MARKER`` alone cannot tell from a finished one (QA round 3, Q-2).
_THREAD_HEADER_RE = re.compile(r"^Thread 0x[0-9a-f]+ \(most recent call first\):$", re.MULTILINE)

#: One COMPLETE frame line, as ``faulthandler`` writes it under a thread header:
#: ``  File "/path/to/x.py", line 12 in func``. The anchor at the END of the pattern is
#: what rejects a line the writer had only begun, which is the second shape a reap
#: mid-write leaves (the first being a bare ``Thread 0x``).
_FRAME_RE = re.compile(r'^  File ".+", line \d+ in .+$')


def _write_stopped_on_a_whole_line(lines: list[str]) -> bool:
    """Did the artifact's last writer stop at a line a FINISHED dump can end on?

    TWO WRITERS SHARE THIS FILE, and the condition has to admit both. ``faulthandler``'s
    C thread writes the fire marker and then one block per thread -- a complete header,
    then that thread's frames -- and stops on a FRAME line. This module itself appends
    through the same ``O_APPEND`` handle, and BY DESIGN its lines land BELOW a fire
    rather than above it (``_append_dump_line``; the HELD/OBSERVED aftermath of a fire
    that did not end the process). So a file whose last write was one of those is
    COMPLETE, and a condition that demanded a frame as the final line made the wait
    unsatisfiable on the artifact the product actually writes -- the reap then settled
    on the NEXT fire's stacks instead of the artifact it was built to wait for
    (R4-1: reproduced with an artifact carrying two fires, which the killed child could
    not have written).

    SKIPPING THE MODULE'S OWN LINES IS WHAT KEEPS A RACE OUT rather than letting one
    in: the boundary they are written at is the same ``write()`` boundary the C thread
    uses, so a marker can land between two of its blocks (the interleaving
    ``held_fire`` is explicitly tested against, and the sampler that writes the
    aftermath reads the file itself). Walking back over them lands on the line the
    writer was in the middle of, and only a COMPLETE frame is accepted there -- so a
    marker that landed under a header the writer had not yet put frames under is
    rejected, while the ordinary aftermath line, written under the finished stacks,
    is accepted.
    """
    index = len(lines) - 1
    while index >= 0 and lines[index].startswith(stall_watchdog.ARM_MARKER):
        index -= 1
    return index >= 0 and bool(_FRAME_RE.match(lines[index]))


def _dump_settled(text: str, *, settled: tuple[str, ...] = ()) -> bool:
    """Is this dump FINISHED, rather than caught mid-write?

    The reap below fires as soon as a fire is attributed, and the fire's stacks are
    written by ``faulthandler``'s C thread while this process reads the file -- so the
    two race under load and the reaped file can end in the middle of a header (measured
    on this fleet: the retained artifact ended at ``Thread 0x`` with no address and no
    ``parked_child.py`` frame, and the cell asserting on it went red against a product
    that was fine). This is an ARTIFACT-SHAPE condition rather than a wait on the
    clock: it asks whether the file is where a finished write stops.

    The shape it accepts is :func:`_write_stopped_on_a_whole_line`'s, and the caller's
    own evidence (``settled``) is required on top -- so a fire whose stacks are
    incomplete still holds the wait, whatever the tail looks like. ``repeat`` is False,
    so a settled file stays settled.
    """
    if stall_watchdog.FIRED_MARKER not in text:
        return False
    if not text.endswith("\n"):
        return False
    lines = text.splitlines()
    if not _THREAD_HEADER_RE.search(text):
        return False
    if not _write_stopped_on_a_whole_line(lines):
        return False
    return all(needle in text for needle in settled)


def _run_stalled_script(
    script: str,
    config_dir: Path,
    args: tuple[str, ...] = (),
    *,
    timeout: float = 30.0,
    env_extra: dict[str, str] | None = None,
    dump_pid: int | None = None,
    settled: tuple[str, ...] = (),
) -> SimpleNamespace:
    """Wait for a real native dump, prove the parked child SURVIVED, then reap it.

    ``subprocess.run`` cannot express the expected state for a GIL-held child:
    after a diagnostic fire it remains parked, so waiting for natural completion
    deadlocks the test until its generic timeout. This harness observes the dump
    and child liveness together, then kills only the process group it created.

    IT WAITS FOR A **COMPLETE** DUMP, NOT FOR THE FIRED MARKER (QA round 3, Q-2).
    The child is reaped the instant a fire is attributed, and the stacks are written by
    ``faulthandler``'s own C thread -- so a reap that lands between two of its writes
    leaves a file ending at the bare ``Thread 0x`` of the next header, and the cells
    below then assert on frames that were never flushed. Observed on this fleet under
    load: a retained dump ended mid-write with no ``parked_child.py`` frame in it, so
    the assertion read as a product failure. ``settled`` names the frames the caller
    needs, and :func:`_dump_settled` holds the file to its own shape as well.
    """
    path = config_dir / "parked_child.py"
    path.write_text(script, encoding="utf-8")
    env = _child_env(config_dir, **(env_extra or {}))
    dump = _dump_for(config_dir, dump_pid or 0)
    try:
        previous_mtime_ns = dump.stat().st_mtime_ns
    except OSError:
        previous_mtime_ns = None
    deadline = time.monotonic() + timeout
    observed_fire = False
    timed_out = False
    stdout_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(path), *args],
        env=env,
        cwd=str(config_dir),
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
    )
    dump = _dump_for(config_dir, dump_pid or proc.pid)
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                stat = dump.stat()
                text = dump.read_text(encoding="utf-8")
            except OSError:
                stat = None
                text = ""
            # Synthetic-pid cases deliberately reuse one artifact path across child
            # lifetimes. A prior life's fired marker may still exist before the new
            # child reaches arm(), so require a fresh write before attributing a fire.
            fresh_dump = stat is not None and (
                previous_mtime_ns is None or stat.st_mtime_ns != previous_mtime_ns
            )
            if fresh_dump and _dump_settled(text, settled=settled):
                observed_fire = proc.poll() is None
                break
            time.sleep(0.02)
        else:
            timed_out = True
    finally:
        # A diagnostic fire must not be mistaken for child completion. Reap only
        # the process group this helper created, after proving the timer fired.
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise AssertionError(f"stalled watchdog child {proc.pid} could not be reaped")
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"process group {proc.pid} survived test cleanup")
        stdout_file.flush()
        stderr_file.flush()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        stdout_file.close()
        stderr_file.close()
    if timed_out:
        raise subprocess.TimeoutExpired(proc.args, timeout, output=stdout, stderr=stderr)
    # A PLAIN NAMESPACE RATHER THAN ``CompletedProcess``, and only here: this helper's
    # contract needs a THIRD state — ``returncode is None`` means "the native timer fired
    # while the child was still ALIVE", which ``subprocess.run`` cannot express and every
    # cell below asserts — while ``CompletedProcess.returncode`` is annotated ``int``. The
    # callers all read these fields by attribute, so the sentinel survives exactly as it
    # was and the type gate sees an honest type rather than a suppressed one.
    return SimpleNamespace(
        args=proc.args,
        returncode=None if observed_fire else proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _dump_for(config_dir: Path, pid: int) -> Path:
    return config_dir / "logs" / f"{stall_watchdog.DUMP_PREFIX}-{pid}.log"


def _fired_seconds(text: str) -> float:
    """The value ``faulthandler`` printed on its own fired line, in seconds.

    The one number that says HOW the timer was last armed, and therefore which class of
    fire a reader is holding (see ``HOW_TO_READ_THE_FIRED_VALUE``) — so the cells that
    tell the classes apart read it here rather than matching a formatted string.
    """
    line = next(line for line in text.splitlines() if line.startswith(stall_watchdog.FIRED_MARKER))
    hours, minutes, seconds = line[len(stall_watchdog.FIRED_MARKER) :].split(")")[0].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _deadline_record(directory: Path, pid: int) -> tuple[float, str]:
    """The sibling as a reader of it gets it: ``(epoch deadline, leg that pinned it)``."""
    epoch, leg = stall_watchdog.deadline_path(pid, directory).read_text(encoding="utf-8").split()
    return float(epoch), leg


def _stamp_ages(stdout: str) -> tuple[list[float], list[float]]:
    """The ``ages`` samples a dead-beater child printed, as (workload, serving).

    The dead-beater children below report BOTH planes' stamp ages every 200 ms,
    because the claim they exist for is a claim about a pair: the workload stamp
    stops advancing while the serving plane keeps reporting normally. A child
    that reported only the silent plane could not tell "the tick died" from "the
    whole process wedged".
    """
    samples = [
        (float(workload), float(serving))
        for workload, serving in re.findall(r"ages workload=([\d.]+) serving=([\d.]+)", stdout)
    ]
    return [workload for workload, _ in samples], [serving for _, serving in samples]


# -- the real thing ---------------------------------------------------------

#: The child that wedges, in the shape of the measured failure: a C call that
#: holds the GIL for its whole duration, so every Python thread in the process
#: stops — the stand-in for ``_sre_SRE_Pattern_search``. ``ctypes.PyDLL`` is the
#: binding that does NOT release the GIL (``CDLL`` does), and ``sleep`` is chosen
#: over a spin loop so a fired bound costs no CPU on a host already running ~25
#: sessions.
_PARKED_CHILD = f"""
import ctypes
import os
import pathlib
import sys

from local_operator.session.runtime import stall_watchdog

# A local of THIS frame, on the stack when the dump is taken. faulthandler
# prints frames and never variables, so this must not reach the file.
secret = "{LOCAL_SENTINEL}"
resumed = pathlib.Path(sys.argv[1])
assert stall_watchdog.arm(seconds=float(sys.argv[2])), "the child could not arm the bound"
print(f"armed:{{os.getpid()}}", flush=True)


def park_the_loop_deliberately() -> None:
    lib = ctypes.PyDLL(None)
    lib.sleep.argtypes = [ctypes.c_uint]
    lib.sleep(600)
    resumed.write_text("the parked call returned", encoding="utf-8")


park_the_loop_deliberately()
resumed.write_text("returned normally", encoding="utf-8")
"""


#: ONE LIFE of a synthetic pid, for the recycle cells: ``arm`` as the runtime entry
#: point calls it, an optional single beat, then a bounded GIL-holding C call. The
#: native timer fires while the call is parked; the child returns on its own so the
#: next synthetic pid life can verify the retained artifacts after process survival.
_RECYCLE_CHILD = """
import ctypes
import sys

from local_operator.session.runtime import stall_watchdog

pid = int(sys.argv[1])
bound = float(sys.argv[2])
assert stall_watchdog.arm(seconds=bound, pid=pid), "the life could not arm the bound"
if sys.argv[3] == "beat":
    stall_watchdog.beat(stall_watchdog.SERVING)
print(f"armed:{pid}", flush=True)

lib = ctypes.PyDLL(None)
lib.sleep.argtypes = [ctypes.c_uint]
lib.sleep(600)
print("park returned", flush=True)
"""


def test_a_stalled_loop_is_dumped_and_the_process_survives(tmp_path: Path) -> None:
    """A real GIL-held stall produces a dump while its process is still alive.

    The harness observes the native fire and child liveness, then reaps the test's
    own process group. Production expiry remains dump-only; the test must clean up
    a deliberately parked child without waiting for a call that models a wedge.
    The dump names the parked frame, the header precedes the fire, and a local
    sentinel value never appears in the artifact.
    """
    resumed = tmp_path / "resumed.txt"
    result = _run_stalled_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(resumed), str(CHILD_BOUND_S)),
    )

    assert (
        result.returncode is None
    ), f"the C timer did not fire while the loop was parked: {result.stdout!r} {result.stderr!r}"
    assert not resumed.exists(), "the C call returned before the test reaped the child"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(tmp_path, pid)
    assert dump.is_file(), f"no dump was written; logs: {sorted((tmp_path / 'logs').glob('*'))}"
    text = dump.read_text(encoding="utf-8")

    assert stall_watchdog.FIRED_MARKER in text, text
    assert "park_the_loop_deliberately" in text, text
    assert "parked_child.py" in text, text
    assert text.index(stall_watchdog.ARM_MARKER) < text.index(
        stall_watchdog.FIRED_MARKER
    ), "the dump was written before the header, so a reader cannot tell arm from fire"
    assert LOCAL_SENTINEL not in text, "faulthandler printed local values into the dump"


def test_a_fired_dump_says_the_fire_is_an_observation_not_a_verdict(tmp_path: Path) -> None:
    """The dump states what it knows, so a reader is not left to infer a death.

    THE DEFECT THIS PINS, measured 2026-09-22: a fire landed on the build carrying
    #1419 43 s after install, on a runtime that was MID-TURN AND WORKING, and that
    pid was still alive afterwards, same pid, no successor ever coming up -- while
    the header said a dump below it meant the bound "ENDED this runtime". An
    operator session read the pile of those dumps as a body count, published "31
    bound-kills", and it reached the v0.62.2 release notes. The header is written
    at ARM time, so the statement has to hold for EVERY fire it can precede, and
    the one above it could not: a fire did not end that runtime.

    So this asserts the field's own words in a dump that really FIRED (the state
    the claim is about, not a header-only file), and asserts the verdict's wording
    gone, because that sentence is the reading that shipped.

    AND THE SAME CELL PINS THE TIMER POLICY: every runtime dump is an observation,
    not a verdict, and production uses ``exit=False``. The header is written before
    any fire; it cannot promise process death or that a prior Python sample still
    matches work admission when the native timer expires.

    THE MUTATION THIS CELL EXISTS TO CATCH, and it is the reason a presence assert
    is not enough on its own: drop ``{OBSERVATION_NOT_VERDICT}`` from the header in
    ``arm`` and this cell -- and nothing else in the file -- goes red. A cell that
    cannot fail would be the same class of thing as the header it pins.
    """
    resumed = tmp_path / "resumed.txt"
    result = _run_stalled_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(resumed), str(CHILD_BOUND_S)),
    )
    assert result.returncode is None, (
        f"the native timer did not fire while the loop was parked: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert not resumed.exists(), "the GIL-held call returned before child cleanup"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert "IDLE one is dumped and exited" not in text
    assert "exit=False" in Path(stall_watchdog.__file__).read_text(encoding="utf-8")

    assert stall_watchdog.FIRED_MARKER in text, "this cell is about a dump that fired"
    assert stall_watchdog.OBSERVATION_NOT_VERDICT in text, (
        "a fired dump does not say what a fire IS, so a reader is left to read a watchdog "
        f"timer expiry as a death: {text[:400]!r}"
    )
    assert "ENDED this runtime" not in text, (
        "the header asserts the verdict a fire never computes; a fire can precede a process "
        f"that carries on, which is the incident this wording caused: {text[:400]!r}"
    )
    assert "THIS PROCESS MAY STILL BE ALIVE ON THIS SAME PID" in text, (
        "the runtime dump must state that production expiry is diagnostic-only, "
        f"rather than imply native termination: {text[:400]!r}"
    )
    assert (
        "THIS IS A TIMER OBSERVATION, NOT A VERDICT.\n\n"
        "THIS PROCESS MAY STILL BE ALIVE ON THIS SAME PID. Check the PID; it may still serve.\n\n"
        "PRODUCTION EXPIRY IS DUMP-ONLY: the native timer is armed with exit=False.\n\n"
    ) in text
    assert "THE PID, NOT THIS FILE, IS WHAT SAYS WHETHER THE RUNTIME IS STILL THERE." in text


_BEATING_CHILD = """
import pathlib
import sys
import time

from local_operator.session.runtime import stall_watchdog

# A file the reader can look for: nothing here counts beats, because the fact
# under test is the bound NOT firing, not how many beats landed.
beats = pathlib.Path(sys.argv[1])
assert stall_watchdog.arm(seconds=float(sys.argv[2])), "the child could not arm the bound"
for index in range(30):
    # BOTH planes, because each is tracked separately now: one silent plane is
    # what the bound exists to catch, so a healthy runtime reports on both.
    stall_watchdog.beat(stall_watchdog.WORKLOAD)
    stall_watchdog.beat(stall_watchdog.SERVING)
    beats.write_text(str(index), encoding="utf-8")
    time.sleep(0.2)
print("survived", flush=True)
stall_watchdog.disarm()
"""

_TWO_PLANE_CHILD = r"""
import os
import pathlib
import re
import sys
import threading
import time

from local_operator.session.runtime import stall_watchdog

finished = pathlib.Path(sys.argv[1])
assert stall_watchdog.arm(seconds=float(sys.argv[2])), "the child could not arm the bound"
print(f"armed:{os.getpid()}", flush=True)

#: The serving plane's beat period, and therefore its stamp period. IT IS PART OF
#: THE CONTRACT WITH THE PARENT, not a decoration: a stamp has to land strictly
#: inside the diagnostic window so the parent can prove the serving ticker ran.
SERVING_BEAT_S = 0.2
first_stamp = threading.Event()


def serving_plane() -> None:
    # Healthy, and it SAYS so — on EVERY beat. An earlier revision stamped every
    # 5th beat, which at this beat period put the only stamp at 5 x 0.2 s == 1.0 s:
    # the same instant a 1 s bound (``SHORT_BOUND_S``) fires, so the print raced
    # the exit and lost about half the time — measured 5 pass / 5 fail over 10
    # runs in one session's QA and 2/8 on interleaved A/B runs in another, always
    # with the primary assertions (rc == 1, no ``finished.txt``) green. Four
    # stamps now fall inside the bound instead of one exactly on it, and the
    # Event below orders the FIRST of them before the park.
    count = 0
    while True:
        stall_watchdog.beat(stall_watchdog.SERVING)
        count += 1
        print(f"serving-stamp:{count}", flush=True)
        if count == 1:
            first_stamp.set()
        time.sleep(SERVING_BEAT_S)


threading.Thread(target=serving_plane, daemon=True).start()
# HAPPENS-BEFORE, NOT A RACE. The workload plane does not start parking until the
# serving plane has already put a stamp on the parent's pipe, so the evidence the
# parent asserts on cannot be confused with a ticker that never ran. The wait
# carries no timeout of its own and needs none: the event is its completion
# condition (AGENTS.md, "Wait on the event, never on the clock" — this is that
# rule inside the child, where the event exists).
first_stamp.wait()

# The workload plane is busy in the matcher and does not report progress. Run
# long enough for the native timer to dump, then return under the child's control.
subject = "credential-shaped text \u2603 x" * 20000
pattern = re.compile(r"(SECRET|DSN|Bearer)\s*=\s*\S+")
end = time.monotonic() + max(5.0, float(sys.argv[2]) * 5)
while time.monotonic() < end:
    pattern.search(subject)

finished.write_text("the busy plane returned", encoding="utf-8")
"""


def test_a_parked_workload_plane_trips_the_bound_while_the_serving_plane_is_healthy(
    tmp_path: Path,
) -> None:
    """THE PRODUCT CASE, and the one the pre-fix design could not fire on.

    A ``daemon``/``exec`` runtime serves on its own thread, so the plane that
    stalls is usually NOT the whole process: on 2026-09-20 every sample sat in the
    workload loop's scan while the serving plane's heartbeat had its own thread to
    run on. A bound that any healthy plane's tick can re-arm therefore never fires
    on the measured failure — and is worse than no bound, because it makes the
    fleet look protected.

    RED BEFORE THE FIX, measured on the committed head (``15ec2c63``) with this
    exact rig at ``LOP_RUNTIME_STALL_SECONDS=1``: the process ran 8 s (8x the
    bound) and was killed by an external timeout, with ZERO fired markers and a
    header-only dump. Green after: the timer is re-armed by whichever plane ticks
    next, for the EARLIEST deadline, so the serving plane's own beats shorten it
    toward the silent plane's deadline instead of pushing it out.

    The serving plane's own stamps are asserted, so this cannot pass by having
    both planes frozen — which the single-threaded tests above already cover and
    which would prove nothing about masking. That stamp is ordered rather than
    raced: the child's serving plane stamps on its FIRST beat, and the workload
    plane does not start parking until it has (see ``_TWO_PLANE_CHILD``), so the
    evidence is on the pipe before the timer can take the process down.
    """
    finished = tmp_path / "finished.txt"
    result = _run_stalled_script(
        _TWO_PLANE_CHILD,
        tmp_path,
        args=(str(finished), str(SHORT_BOUND_S)),
    )

    assert result.returncode is None, (
        f"the native timer did not fire on the silent workload plane: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert not finished.exists(), "the GIL-held matcher returned before child cleanup"
    assert "serving-stamp:" in result.stdout, (
        "the serving plane never reported, so this run says nothing about a healthy "
        f"plane masking a silent one: {result.stdout!r} {result.stderr!r}"
    )
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(tmp_path, pid)
    assert dump.is_file(), f"no dump was written; logs: {sorted((tmp_path / 'logs').glob('*'))}"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text
    assert "parked_child.py" in text, f"the dump does not name the parked plane: {text}"


# ============================================================================
# WHAT THE ARTIFACT RECORDS ABOUT A FIRE: which plane, how the timer was armed
# ============================================================================
#
# WHY THIS SECTION EXISTS. A fire's own record carried the arm epoch and nothing else,
# so the first two questions anyone asks of one were unanswerable from it: which loop
# stopped reporting, and when. Measured over 26 retained fires, 6 carried a full
# ``0:05:00``-style value (arm's own bound, no beat ever re-arming it — the
# never-engaged class, when no re-arm-failed line is present) and 19 a small one,
# 0.4-17 s (a beat's recomputed remainder,
# pinned by the OTHER plane's stale stamp). The number was there; nothing said which
# it was; and the stamps that decide it die with the process. The cells below pin the
# two records that fix that — the transition written into the dump, and the deadline
# sibling — and each is proven to fail when its line is removed (see the mutation
# notes in the docstrings; they were run, not imagined).


def test_a_beat_records_the_deadline_and_disarm_removes_the_sibling(tmp_path: Path) -> None:
    """The sibling is the ONE number that turns a wedge from inference into measurement.

    "Last beat 04:26:31 -> deadline 04:31:31, so something pre-empted the bound by 16 s"
    is answerable from it, and so is the reading that says 86 s OVERDUE and silent: the
    same two fields, and the comparison a reader can actually make. It is rewritten by
    every beat, so this drives one and reads what landed — the epoch deadline the timer
    is armed for, and the leg whose stamp pinned it (``workload``, because only the
    serving plane beat and the workload plane's ARM stamp is therefore the oldest).
    ``disarm`` then takes it away beside the dump: a runtime that left on its own terms
    must not leave a deadline behind for a reader to find.

    MUTATION THIS CELL CATCHES: drop ``_record_deadline(armed)`` from ``beat`` — nothing
    writes the sibling, and this is the cell that goes red.
    """
    # A NAMED STARTING STATE, because `_ARMED` is module-global and `arm` returns early
    # when something has left the process armed: without this, a leaked arm would make
    # the call below a no-op and the cell would then fail on a missing FILE rather than
    # on the record it means to check. `disarm` is documented as unconditionally safe.
    stall_watchdog.disarm()
    assert stall_watchdog.arm(seconds=30.0, directory=tmp_path) is True
    try:
        stall_watchdog.beat(stall_watchdog.SERVING)
        epoch, leg = _deadline_record(tmp_path, os.getpid())
        assert leg == stall_watchdog.WORKLOAD, (
            f"the sibling names {leg!r} as the pin; only the serving plane beat, so the "
            f"workload plane's arm stamp is the older one the timer is armed from"
        )
        # THE SIBLING ENCODES A RELATIONSHIP, so this compares it with the bound this
        # process armed for rather than with zero, and the window is one heartbeat either
        # side of it: an epoch is a WALL clock and the line below re-reads one, so a step
        # must not fail the cell — while the two bugs that matter (recording the stamp
        # instead of the deadline, or the monotonic deadline instead of the epoch) put
        # the value near zero, not near the bound.
        remaining = epoch - time.time()
        assert 15.0 < remaining <= 45.0, (
            f"the sibling's deadline is {remaining:.2f}s away, which is not the 30s bound "
            f"this process armed for (nor one heartbeat either side of it)"
        )
    finally:
        stall_watchdog.disarm()

    assert not stall_watchdog.deadline_path(
        os.getpid(), tmp_path
    ).exists(), "a clean exit left the deadline sibling behind"
    assert not stall_watchdog.dump_path(
        os.getpid(), tmp_path
    ).exists(), "a clean exit left the dump behind"


def test_a_runtime_that_never_engaged_fires_at_the_armed_value_with_no_sibling(
    tmp_path: Path,
) -> None:
    """THE OTHER CLASS, told apart by ABSENCE: no beat ever re-armed this timer.

    ``arm`` seeds both planes to the arm instant, so a runtime whose loops never start
    reaches its bound with no stamp of its own behind it, and the value on the fired line
    is the bound arm set rather than a recomputed remainder. ``arm`` CLEARS the sibling and
    never writes it (see the recycle cell below for why clearing is part of it), which is
    what makes a MISSING
    ``runtime-stall-<pid>.deadline`` mean "no beat ever re-armed this timer" rather than
    "the writer was unlucky", and it is the only structural difference between the two
    classes a reader has.

    MUTATION THIS CELL CATCHES: write the sibling in ``arm`` as well — the file now
    advertises a deadline for a process that never beat, and the absence signal is lost.
    """
    resumed = tmp_path / "resumed.txt"
    result = _run_stalled_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(resumed), str(SHORT_BOUND_S)),
    )
    assert (
        result.returncode is None
    ), f"the native timer did not fire at the armed deadline: {result.stdout!r} {result.stderr!r}"
    assert not resumed.exists(), "the parked C call returned before child cleanup"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")

    assert stall_watchdog.FIRED_MARKER in text, text
    assert _fired_seconds(text) == pytest.approx(float(SHORT_BOUND_S)), (
        f"the fired value is not the value arm() set — no beat re-armed this timer, so "
        f"nothing recomputed it: {text[:400]!r}"
    )
    assert (
        stall_watchdog.REARM_MARKER not in text
    ), f"a runtime that never reported was still named as a quiet plane: {text[:400]!r}"
    assert not stall_watchdog.deadline_path(pid, tmp_path / "logs").exists(), (
        "a sibling exists for a runtime that never beat, so its absence cannot mean "
        "'no beat ever re-armed the timer'"
    )


def test_arm_clears_a_deadline_sibling_an_earlier_holder_of_the_pid_left(
    tmp_path: Path,
) -> None:
    """A SIBLING CANNOT OUTLIVE THE LIFE THAT WROTE IT, so presence is about NOW.

    The rule a reader is given (:data:`stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE`) is a
    fact about presence and absence: the sibling is there when a beat of THIS life re-armed
    the timer, and not there when no beat ever did (or when the re-arm FAILED, which writes
    a line of its own). A pid is RECYCLED and these files are
    keyed by pid alone, so a file a previous holder left is the one thing that can make the
    rule say the wrong thing -- and ``arm`` opens the dump with ``"w"`` (fresh per life)
    while leaving the pair's other half alone. Measured (QA round 1, Q1): a run that fired
    the never-engaged signature carried a sibling written 2.5 s BEFORE its own arm epoch,
    naming a plane of a life that was over.

    MUTATION THIS CELL CATCHES: drop the ``inherited.unlink()`` from ``arm`` -- the file
    planted here survives the arming and this cell goes red, which is exactly the state the
    rig caught.
    """
    stall_watchdog.disarm()
    stale = stall_watchdog.deadline_path(os.getpid(), tmp_path)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("1790066886.912 serving\n", encoding="utf-8")
    try:
        assert stall_watchdog.arm(seconds=30.0, directory=tmp_path) is True
        assert not stale.exists(), (
            f"arm left {stale.name} where the previous holder of this pid wrote it, so a "
            f"PRESENT sibling no longer means 'a beat of this life re-armed the timer'"
        )
    finally:
        stall_watchdog.disarm()


def test_a_recycled_pid_fires_its_second_life_with_no_inherited_sibling(
    tmp_path: Path,
) -> None:
    """THE RECYCLE, END TO END: three dump-only fires at one synthetic pid.

    The cell above proves ``arm`` removes a stale file; this one proves each new life
    replaces it with its own current deadline and leg before its dump fires. Each
    deliberately parked child is reaped by the helper after that native fire, leaving
    the dump and sibling artifacts available for the next synthetic life to inspect.

    MUTATION THIS CELL CATCHES: drop the ``inherited.unlink()`` from ``arm``. Life 2 then
    fires the never-engaged value with life 1's sibling still on disk -- a file whose mtime
    precedes life 2's own arm epoch -- and the presence assertion goes red.
    """
    logs = tmp_path / "logs"
    dump = _dump_for(tmp_path, SYNTH_PID)
    sibling = stall_watchdog.deadline_path(SYNTH_PID, logs)

    first = _run_stalled_script(
        _RECYCLE_CHILD,
        tmp_path,
        args=(str(SYNTH_PID), str(CHILD_BOUND_S), "beat"),
        dump_pid=SYNTH_PID,
    )
    assert (
        first.returncode is None
    ), f"life 1 did not fire its native timer: {first.stdout!r} {first.stderr!r}"
    assert "park returned" not in first.stdout
    assert sibling.is_file(), "life 1's fire left no sibling, so this cell tests nothing"
    assert stall_watchdog.FIRED_MARKER in dump.read_text(
        encoding="utf-8"
    ), "life 1's dump is not a fire, so the sibling below is not a fired runtime's"
    leg_1 = _deadline_record(logs, SYNTH_PID)[1]
    assert leg_1 == stall_watchdog.WORKLOAD, (
        f"life 1's sibling names {leg_1!r}, not the plane whose stale stamp pinned the "
        f"deadline, so a stale file could not be told from a correct one here"
    )
    written_1 = sibling.stat().st_mtime

    second = _run_stalled_script(
        _RECYCLE_CHILD,
        tmp_path,
        args=(str(SYNTH_PID), str(CHILD_BOUND_S), "quiet"),
        dump_pid=SYNTH_PID,
    )
    assert (
        second.returncode is None
    ), f"life 2 did not fire its native timer: {second.stdout!r} {second.stderr!r}"
    assert "park returned" not in second.stdout
    text = dump.read_text(encoding="utf-8")
    assert _fired_seconds(text) == pytest.approx(float(CHILD_BOUND_S), abs=0.01), (
        f"life 2 did not fire the never-engaged value, so this run does not show the class "
        f"whose absence signal is the point: {text[:400]!r}"
    )
    arm_2 = float(text.split(" armed for ", 1)[1].split(" at ", 1)[1].split()[0])
    assert not sibling.exists(), (
        f"life 2 fired the never-engaged value with {sibling.read_text(encoding='utf-8')!r} "
        f"still on disk, written at mtime {written_1:.3f} against life 2's own arm at "
        f"{arm_2:.0f} -- a present sibling that proves no beat of the life in front of "
        f"the reader"
    )

    third = _run_stalled_script(
        _RECYCLE_CHILD,
        tmp_path,
        args=(str(SYNTH_PID), str(CHILD_BOUND_S), "beat"),
        dump_pid=SYNTH_PID,
    )
    assert (
        third.returncode is None
    ), f"life 3 did not fire its native timer: {third.stdout!r} {third.stderr!r}"
    assert "park returned" not in third.stdout
    assert sibling.is_file(), (
        "a life that DID beat its own timer left no sibling, so the absence above is not a "
        "signal at all"
    )
    assert sibling.stat().st_mtime > written_1, "life 3's sibling is life 1's file, not its own"


def test_a_beat_whose_rearm_failed_says_the_sibling_is_behind_the_last_beat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ONE CASE WHERE THE SIBLING'S MTIME IS NOT THE LAST BEAT, said out loud.

    A beat whose re-arm raises leaves the previous deadline in force, so
    :func:`stall_watchdog._record_deadline` is not called and the file keeps an earlier
    beat's number AND its mtime -- while the header sends a reader to that mtime for "when
    did this process last report". Measured (QA round 1, Q3): five beats with the re-arm
    raising left the mtime frozen and it jumped on the next healthy beat, i.e. seconds of
    silence that never happened, read as one.

    The sibling cannot carry the correction itself -- its content is the deadline in force,
    which a failed re-arm does not change -- so the beat writes it into the dump, which is
    where the reader holding the sibling is already being sent, and which outlives the
    process. A warning in the log would not: the log is not the post-mortem's artifact.

    MUTATION THIS CELL CATCHES: drop the ``_write_dump_line`` call from that branch -- the
    mtime is frozen after a failed re-arm and nothing in the artifact says why.
    """

    real_faulthandler = stall_watchdog.faulthandler

    class _RefusingFaulthandler:
        """The re-arm RAISES, and the cancel still reaches the real C timer.

        The cancel is delegated rather than stubbed because this cell arms the REAL
        timer before patching this in: a stub would leave a 30 s timer armed
        live past the cell and take a passing pytest worker with it, which is the
        failure mode the whole module is written against.
        """

        def dump_traceback_later(self, *args: object, **kwargs: object) -> None:
            raise OSError("no timer for you")

        def cancel_dump_traceback_later(self) -> None:
            real_faulthandler.cancel_dump_traceback_later()

    stall_watchdog.disarm()
    assert stall_watchdog.arm(seconds=30.0, directory=tmp_path) is True
    try:
        stall_watchdog.beat(stall_watchdog.SERVING)
        sibling = stall_watchdog.deadline_path(os.getpid(), tmp_path)
        written = sibling.read_text(encoding="utf-8")
        beat_mtime = sibling.stat().st_mtime
        time.sleep(0.05)

        monkeypatch.setattr(stall_watchdog, "faulthandler", _RefusingFaulthandler())
        stall_watchdog.beat(stall_watchdog.SERVING)

        assert (
            sibling.read_text(encoding="utf-8") == written
        ), "the failed re-arm rewrote the sibling, so it states a deadline the timer never got"
        assert (
            sibling.stat().st_mtime == beat_mtime
        ), "the sibling's mtime moved, so this cell is not testing the frozen case"
        text = stall_watchdog.dump_path(os.getpid(), tmp_path).read_text(encoding="utf-8")
        assert stall_watchdog.REARM_FAILED_MARKER in text, (
            "the beat's re-arm failed and left the sibling's mtime behind the last beat, and "
            f"the dump says nothing about it: {text[-400:]!r}"
        )
    finally:
        stall_watchdog.disarm()


def test_a_reader_concurrent_with_a_beat_never_sees_a_partial_sibling(
    tmp_path: Path,
) -> None:
    """THE SIBLING IS REPLACED, NOT REWRITTEN, so a reader gets a whole file or none.

    ``Path.write_text`` is open(O_TRUNC) + write + close, and a reader landing inside that
    window observes a TRUNCATED file -- measured (QA round 1, Q4) at 398 beats/s: 2500 of
    28,922 reads returned zero bytes. At the shipping cadence the exposure is one ~205 us
    window per 15 s beat, which is small and not zero: a fire landing inside it leaves a
    ZERO-BYTE sibling for a post-mortem that has no rule for one, which is the class of
    unreadable artifact this module exists to remove.

    ATOMICITY IS THE ASSERTION, NOT A TIMING BOUND: an atomic replace leaves a reader with
    the previous file or the new one, so the shipped shape cannot produce a short read at
    any cadence, and the mutated shape (write in place) produces them by the hundred at
    this one. The read count is asserted too, so a reader that never ran fails as a dead
    instrument instead of passing quietly.

    MUTATION THIS CELL CATCHES: write the destination instead of the temp (replace the
    ``os.replace`` with ``armed.deadline_path.write_text(...)``) -- the zero-byte reads come
    back and ``partial`` is no longer 0.
    """
    stall_watchdog.disarm()
    assert stall_watchdog.arm(seconds=30.0, directory=tmp_path) is True
    path = stall_watchdog.deadline_path(os.getpid(), tmp_path)
    stop = threading.Event()
    seen = {"reads": 0, "partial": 0}

    def read_forever() -> None:
        while not stop.is_set():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                # Not there YET is the first write's window, not a partial read.
                continue
            seen["reads"] += 1
            if len(text.split()) != 2 or not text.endswith("\n"):
                seen["partial"] += 1

    reader = threading.Thread(target=read_forever, daemon=True)
    reader.start()
    try:
        for _ in range(2000):
            stall_watchdog.beat(stall_watchdog.SERVING)
    finally:
        stop.set()
        reader.join(timeout=10.0)
        stall_watchdog.disarm()

    assert not reader.is_alive(), "the reader thread never stopped"
    assert seen["reads"] > 200, f"the concurrent reader is a dead instrument: {seen}"
    assert seen["partial"] == 0, (
        f"{seen['partial']} of {seen['reads']} reads concurrent with a beat saw a partial or "
        f"empty sibling, so a post-mortem can be left holding a file with no meaning"
    )


def test_a_quiet_plane_is_named_in_the_dump_alongside_the_deadline_that_came_due(
    tmp_path: Path,
) -> None:
    """THE SMALL-VALUE CLASS, and the two records that explain it.

    19 of 26 retained fires carried a value far below their bound, and the reason was a
    phase, not a number: a beat had recomputed the deadline from the OTHER plane's stale
    stamp, so the last arming was a remainder. Nothing in the file said so, and nothing
    said WHICH plane. Here the serving plane keeps beating every 0.2 s — asserted, so the
    cell cannot pass on a wholly frozen process — while the workload plane never reports,
    and the records are cross-checked against each other: the dump's quiet line names the
    workload plane, and the sibling's epoch sits one BOUND past the arm (the workload
    plane's stamp plus the bound, i.e. the deadline that came due), while the fired value
    is the small remainder the last beat armed for.

    MUTATION THIS CELL CATCHES: drop ``_note_quiet_plane(armed, now)`` from ``beat`` — the
    dump goes back to saying nothing about which plane went quiet.
    """
    finished = tmp_path / "finished.txt"
    result = _run_stalled_script(
        _TWO_PLANE_CHILD,
        tmp_path,
        args=(str(finished), str(SHORT_BOUND_S)),
    )
    assert result.returncode is None, (
        f"the native timer did not fire on the quiet workload plane: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert not finished.exists(), "the GIL-held matcher returned before child cleanup"
    assert "serving-stamp:" in result.stdout, (
        f"the serving plane never reported, so this run says nothing about one plane's "
        f"stamps carrying the other's deadline: {result.stdout!r}"
    )
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")

    quiet = [line for line in text.splitlines() if line.startswith(stall_watchdog.REARM_MARKER)]
    assert quiet, f"no quiet plane is named in the dump: {text[:800]!r}"
    assert stall_watchdog.WORKLOAD in quiet[0], quiet[0]
    assert text.index(stall_watchdog.REARM_MARKER) < text.index(stall_watchdog.FIRED_MARKER), (
        "the quiet line was written after the fire, so a reader cannot attribute one to "
        f"the other: {text[:800]!r}"
    )

    arm_epoch = float(text.split(" armed for ", 1)[1].split(" at ", 1)[1].split()[0])
    epoch, leg = _deadline_record(tmp_path / "logs", pid)
    assert (
        leg == stall_watchdog.WORKLOAD
    ), f"the deadline was pinned by {leg!r}, not the quiet plane"
    # THE TOLERANCE IS THE HEADER'S OWN RESOLUTION, and 0.5 was the wrong number rather
    # than a tight one: the header records the arm with ``{time.time():.0f}``, which
    # ROUNDS, so the arm instant is only known to within half a second and a deadline
    # derived from it to within a whole one of the bound. Measured as a live flake on
    # 2026-09-22: this cell failed at 0.54s on a loaded host — i.e. on the rounding,
    # with the class it is testing for present in the same run, which is why the value
    # that pins the class is asserted separately below, off the FIRED line.
    assert abs((epoch - arm_epoch) - float(SHORT_BOUND_S)) < 1.0, (
        f"the sibling's deadline is {epoch - arm_epoch:.2f}s past the arm, not the "
        f"{SHORT_BOUND_S}s bound the workload plane's stamp implies, so it is not the "
        f"deadline that came due: {text[:400]!r}"
    )
    assert _fired_seconds(text) < float(SHORT_BOUND_S), (
        f"the last arming was the bound itself, so this run does not show the remainder "
        f"class the small values belong to: {text[:400]!r}"
    )


def test_the_header_says_how_to_read_a_fired_value_and_the_deadline_sibling(
    tmp_path: Path,
) -> None:
    """A fire's numbers are useless to a reader who does not know what armed them.

    This is the clause that makes the two classes readable FROM THE FILE — the bound when
    no beat re-armed the timer, PROVIDED the re-arm succeeded (a runtime that never
    engaged; an engaged runtime whose replacement FAILED can carry the same bound, see
    design review D2), a smaller remainder when a beat recomputed it from the oldest
    plane's stamp — and that points the reader at the
    sibling, whose absence means the first of those and whose mtime is the last beat. It
    is written at ARM time, so it is present whether or not this runtime ever fires.

    MUTATION THIS CELL CATCHES: drop ``{HOW_TO_READ_THE_FIRED_VALUE}`` from the header in
    ``arm`` — the fired value goes back to being a number with no stated meaning. The
    qualifier is asserted by its own parts, so re-stating the never-engaged reading
    unconditionally (design review round 1, D2) also turns this cell red.
    """
    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    try:
        text = stall_watchdog.dump_path(os.getpid(), tmp_path).read_text(encoding="utf-8")
    finally:
        stall_watchdog.disarm()

    assert stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE in text, text[:600]
    # The never-engaged reading is conditional on a SUCCESSFUL re-arm, because an
    # engaged runtime whose timer replacement failed can carry the same armed value.
    assert "PROVIDED THE RE-ARM SUCCEEDED" in stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE
    assert (
        "An ENGAGE whose timer replacement FAILED is the exception"
        in stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE
    )
    assert (
        "that line's absence is what makes the never-engaged reading safe"
        in stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE
    )
    assert (
        f"{stall_watchdog.DUMP_PREFIX}-<pid>{stall_watchdog.DEADLINE_SUFFIX}" in text
    ), f"the header does not name the sibling a reader has to go and open: {text[:600]!r}"


# -- the debug dump's default -------------------------------------------------


def test_the_debug_dump_is_on_by_default_and_can_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DEFAULT is the behaviour, so it is pinned directly (A7).

    This dump was opt-in until 2026-09-20, and that is why it changed: on every
    one of the five runtimes found frozen that day the variable was NOT set, so
    the one instrument that could have named the parked await was unavailable —
    a diagnostic that must be predicted before the freeze it explains is one
    nobody has when they need it. The opt-out survives for a harness that needs
    SIGUSR1's default (fatal) disposition back, and it is pinned as a decision
    rather than through a spawned child because the decision is one line and the
    boot is not what is under test.
    """
    from local_operator.session.runtime import process

    monkeypatch.delenv(process.DEBUG_STACKS_ENV, raising=False)
    assert process.debug_stacks_enabled() is True

    for spelling in ("0", "no", "false", "off", "FALSE", " off "):
        monkeypatch.setenv(process.DEBUG_STACKS_ENV, spelling)
        assert process.debug_stacks_enabled() is False, spelling

    for other in ("1", "yes", "on", "true", ""):
        monkeypatch.setenv(process.DEBUG_STACKS_ENV, other)
        assert process.debug_stacks_enabled() is True, other


def test_the_debug_dump_decision_is_wired_into_the_handler_installation() -> None:
    """A decision nothing consults is not a default, it is a comment.

    Asserted against ``amain``'s own source, and it also pins the shape the
    cross-platform probe reads: the capability probe
    (``debug_stacks is not None`` — SIGUSR1 is POSIX-only) must stay at the call
    site, because the scanner that made the battery's two probe legs fail finds
    its guard by NAME. Moving that probe into the helper would leave an
    unguarded POSIX ``add_signal_handler`` in the tree with a green test here.
    """
    from local_operator.session.runtime import process

    source = inspect.getsource(process.amain)
    assert "debug_stacks_enabled()" in source, "the default is not consulted"
    assert (
        "debug_stacks is not None" in source
    ), "the POSIX capability probe left the call site; the xplat scan reads it there"
    assert "not procstate.is_windows()" in source


#: THE PRODUCT'S TWO PLANES, not a model of them: the real ``RuntimeServer``
#: (``start()`` gives the serving plane its own thread and loop, and its heartbeat
#: loop is what stamps :data:`stall_watchdog.SERVING`) plus the real workload tick
#: (``process._beat_stall_watchdog``, stamping ``WORKLOAD``), with the workload loop
#: then parked in a chain of SHORT C calls — the measured shape, in which the GIL is
#: released every switch interval so the serving plane's thread keeps running.
#:
#: The two cadences are patched to a fraction of a second so the run is seconds
#: rather than minutes. That is a knob, not the rule under test: what is under test
#: is that ONE silent plane trips the bound while the other reports normally, and
#: the cadence only decides how long a test has to wait to see it.
_REAL_TWO_PLANE_CHILD = r"""
import asyncio
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, sys.argv[3])  # the checkout root, for the session factory

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

    session = make_session(root, _stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"
    assert stall_watchdog.arm(seconds=float(sys.argv[1])), "the child could not arm the bound"
    print(f"armed:{os.getpid()}", flush=True)
    print(f"record:{root}/run/mobile/{os.getpid()}.json", flush=True)

    stop = asyncio.Event()
    asyncio.create_task(process._beat_stall_watchdog(stop))
    # Both planes now run on their own loops: the serving plane on the runtime's
    # thread, this ticker on the workload loop.
    await asyncio.sleep(2.0)
    print("both-planes-ran", flush=True)

    subject = "credential-shaped text \u2603 x" * 20000
    pattern = re.compile(r"(SECRET|DSN|Bearer)\s*=\s*\S+")
    end = time.monotonic() + max(10.0, float(sys.argv[1]) * 5)
    while time.monotonic() < end:
        pattern.search(subject)
    print("workload-finished", flush=True)
    os._exit(0)


asyncio.run(main())
"""


def test_the_stalled_harness_only_reads_a_settled_dump(tmp_path: Path) -> None:
    """Q-2: the reap waits for a COMPLETE artifact, not for the fired marker.

    The helper below kills the child it created as soon as it attributes a fire, and
    the stacks are written by ``faulthandler``'s own C thread -- so under load the reap
    can land mid-write and leave a file ending at the bare ``Thread 0x`` of the next
    thread header. A cell asserting on the frames of that file then fails against a
    product that is fine (measured on this fleet: the retained dump held no
    ``parked_child.py`` frame at all). The condition this pins is the artifact's own
    shape, and it is what the helper breaks on instead of the marker.

    AND THE SHAPE IS NOT "ENDS WITH A FRAME" (agent review round 4, R4-1): this module
    appends its own lines BELOW a fire by design, so the artifact a beat leaves can end
    on an aftermath line, and the frame-only condition held the wait open until the
    NEXT fire -- a reap point that depended on a later fire than the one it waited for.
    Both tails are pinned here, and so are the mid-write shapes that accepting the
    module's own lines must not re-admit.
    """
    header = f"{stall_watchdog.ARM_MARKER}policy: dump-only (exit=False)\n"
    complete = (
        f"{header}{stall_watchdog.FIRED_MARKER}0:00:01.5)!\n"
        "Thread 0x000000016c147000 (most recent call first):\n"
        '  File "parked_child.py", line 7 in park_the_loop_deliberately\n'
        '  File "parked_child.py", line 14 in <module>\n'
    )
    # THREE SHAPES A REAP MID-WRITE LEAVES, each caught by a different part of the
    # condition, so removing any one part is visible here (the reviewer's Q-2 artifact
    # was the first).
    at_bare_header = complete[: complete.index("Thread 0x")] + "Thread 0x"
    after_a_header = complete[: complete.index('  File "parked_child.py"')]
    mid_frame = complete[: complete.index('", line 14')] + '", line 1\n'

    assert _dump_settled(complete)
    assert _dump_settled(complete, settled=("parked_child.py",))
    # THE SHAPE THE PRODUCT ACTUALLY WRITES, and the one R4-1 was reported against: a
    # beat's aftermath line, which ``_append_dump_line`` puts BELOW a fire by design. The
    # frame-only tail condition could not see it, so the wait ran on to the next fire's
    # stacks -- a reap point that depended on a LATER fire than the one it waited for.
    aftermath = (
        f"{stall_watchdog.OBSERVED_MARKER}the bound fired at 2026-09-23 21:13:05 and did "
        "NOT end this runtime. No work was in flight when the fire was observed.\n"
    )
    assert _dump_settled(
        complete + aftermath, settled=("parked_child.py",)
    ), "a dump whose last write is this module's own aftermath line is COMPLETE"
    # ...and the two-fire artifact the reviewer retained, where the FIRST fire is what
    # the caller was waiting for and the second is only on the file because the reap did
    # not settle on the first.
    second_fire = (
        f"{stall_watchdog.FIRED_MARKER}0:00:00.172937)!\n"
        "Thread 0x00000001f9ba7f80 (most recent call first):\n"
        '  File "parked_child.py", line 7 in park_the_loop_deliberately\n'
    )
    assert _dump_settled(complete + aftermath + second_fire, settled=("parked_child.py",))
    assert not _dump_settled(
        at_bare_header
    ), "a dump cut inside the next thread header was read as whole"
    assert not _dump_settled(
        after_a_header
    ), "a dump cut between a header and its frames was read as whole"
    assert not _dump_settled(mid_frame), "a dump cut inside a frame line was read as whole"
    # THE INTERLEAVED SHAPE, which accepting module lines could otherwise re-admit: a
    # marker written while the C writer was still inside its pass lands under a header
    # with no frames yet, or under the fire's own marker. Both are mid-write, and the
    # walk back over this module's lines is what tells them from an aftermath line
    # written under finished stacks.
    assert not _dump_settled(
        complete + "Thread 0x00000001f9ba7f80 (most recent call first):\n"
    ), "a dump cut between a LATER header and its frames was read as whole"
    assert not _dump_settled(
        complete + f"{stall_watchdog.FIRED_MARKER}0:00:00.17)!\n"
    ), "a dump that stopped on the next fire's own marker is not yet a finished write"
    assert not _dump_settled(
        mid_frame + aftermath
    ), "a marker under a frame line the writer had only begun was read as whole"
    assert not _dump_settled(
        complete.replace(stall_watchdog.FIRED_MARKER, "no fire")
    ), "an artifact with no fire is not a settled dump"
    assert not _dump_settled(
        complete, settled=("a_frame_that_was_never_flushed",)
    ), "the caller's own evidence is part of what makes the write finished"


def test_the_bound_fires_on_the_real_runtime_while_its_serving_plane_stays_healthy(
    tmp_path: Path,
) -> None:
    """Q2(c): the product's own planes produce diagnostic evidence for silence.

    This is the case the first revision of this change could not fire on, and it
    is the reason that revision was wrong: a ``daemon``/``exec`` runtime serves on
    its own thread, so the plane that stalls is usually NOT the whole process. A
    bound any healthy plane can re-arm therefore never fires on the failure this
    PR exists for — worse than no bound, because it makes the fleet look
    protected. Review measured exactly that on the committed head (8x the bound,
    zero fired markers), and this asserts the fix on the REAL plumbing rather than
    on a rig: the serving plane is ``RuntimeServer.start()``'s own thread and loop,
    and the silent one is ``process._beat_stall_watchdog`` — the runtime's own
    ticker — after it has demonstrably run (``workload-tick-ran``) and then been
    parked by a synchronous scan.

    RED BEFORE THE FIX, structurally: with one shared last-beat the serving plane's
    heartbeat re-armed the full bound every 15 s, so this child never produced a dump
    before the parent had to reap it.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    result = _run_stalled_script(
        _REAL_TWO_PLANE_CHILD,
        config_dir,
        args=(
            str(CHILD_BOUND_S),
            str(config_dir),
            str(Path(__file__).resolve().parents[4]),
        ),
        timeout=60.0,
        # THE FRAME THE ASSERTIONS BELOW NEED, so the helper cannot reap the child
        # between the fire's marker and the frame that names the parked workload
        # (see _dump_settled).
        settled=("parked_child.py",),
    )

    assert result.returncode is None, (
        f"the runtime did not survive to a fired diagnostic dump: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert "workload-finished" not in result.stdout
    assert "both-planes-ran" in result.stdout, (
        f"the workload ticker never ran, so this says nothing about a HEALTHY plane "
        f"going silent: {result.stdout!r} {result.stderr!r}"
    )
    # THE SERVING PLANE REALLY WAS THERE: it published a record, and that record is
    # the heartbeat this runtime's own surfaces read.
    records = sorted((config_dir / "run" / "mobile").glob("*.json"))
    assert records, f"the serving plane never published a record: {sorted(config_dir.rglob('*'))}"

    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(config_dir, pid)
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text
    assert "parked_child.py" in text, f"the dump does not name the parked workload loop: {text}"
    # TWO PLANES IN THE DUMP: faulthandler writes every thread, so a run with the
    # serving plane alive shows its thread as well as the parked one.
    assert text.count("Thread 0x") >= 2, f"only one thread was live at the trip: {text}"


def test_progress_defers_the_bound_in_a_real_process(tmp_path: Path) -> None:
    """A loop that keeps reporting is never killed by its own bound.

    The property the re-arm exists for, on the REAL C timer: the child beats 30
    times over ~6 s against a 2 s bound, so a run that fired would be proof the
    re-arm does nothing. The margin is 10x the beat interval, which is what keeps
    this an assertion about re-arming rather than a stopwatch on a loaded host.
    """
    script = _BEATING_CHILD
    result = _run_script(
        script,
        tmp_path,
        args=(str(tmp_path / "beats.txt"), str(CHILD_BOUND_S)),
    )

    assert result.returncode == 0, f"the bound fired through the beats: {result.stdout!r}"
    assert "survived" in result.stdout, result.stdout
    # A CLEAN EXIT REMOVES ITS FILE, which is what makes a file left behind mean
    # the process died without disarming (see "THE FILE IS THE EVIDENCE" in the
    # module docstring, and the allow-list rows that argue the ``unlink``).
    assert not list(
        (tmp_path / "logs").glob(f"{stall_watchdog.DUMP_PREFIX}-*.log")
    ), "a clean exit left its dump behind, so a file's existence stopped meaning anything"
    assert stall_watchdog.fired_pids(tmp_path / "logs") == set()


_LEAVING_CHILD = """
import os
import sys

from local_operator.session.runtime import stall_watchdog

# argv[1] is the bound: this child writes no sentinel, so it is the first argument.
assert stall_watchdog.arm(seconds=float(sys.argv[1])), "the child could not arm the bound"
print(f"armed:{os.getpid()}", flush=True)
# An exit that never runs the disarm: exactly what a SIGKILL or a crash leaves.
os._exit(0)
"""


def test_a_file_left_by_a_kill_is_not_reported_as_a_fired_bound(tmp_path: Path) -> None:
    """Existence alone is not evidence; the fired marker is.

    The header is written at ARM time (it has to be — faulthandler writes with a
    raw descriptor from a C thread), so a process killed without disarming
    leaves a file behind. Reporting that as a freeze would hand a reader a
    timeout claim for a run that never timed out — the failure mode
    ``tests/e2e/watchdog.py`` names, and the reason this module defines its
    evidence as a marker rather than as a file.
    """
    result = _run_script(_LEAVING_CHILD, tmp_path, args=(str(SHORT_BOUND_S),))
    assert result.returncode == 0, result.stderr
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    leftover = _dump_for(tmp_path, pid)
    assert leftover.is_file(), "the header-only file this test is about was never written"
    text = leftover.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER not in text
    # NEITHER MARKER IS SPELLED IN THE HEADER, and this pair is the regression
    # test for that: the header explains what a fired dump looks like and used to
    # quote faulthandler's own marker to do it, which made every armed file read
    # as a fired bound to any reader testing the marker as a SUBSTRING — including
    # this assertion. The progress marker had the same hazard one line below.
    assert stall_watchdog.PROGRESS_MARKER not in text
    assert (
        stall_watchdog.fired_pids(tmp_path / "logs") == set()
    ), "a header-only file was reported as a fired bound"
    assert (
        stall_watchdog.fired_leg(pid, tmp_path / "logs") is None
    ), "a header-only file was read as an ended runtime"

    fired = tmp_path / "logs" / f"{stall_watchdog.DUMP_PREFIX}-4242.log"
    fired.write_text(
        f"header\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n", encoding="utf-8"
    )
    assert stall_watchdog.fired_pids(tmp_path / "logs") == {4242}
    # ...and the pid is read from the FILE NAME, so a reader holding a record's
    # pid can ask directly instead of globbing for a path: that is the contract
    # ``lop sessions --json``'s ``stall_dump`` relies on.
    assert stall_watchdog.dump_path(4242, tmp_path / "logs") == fired
    # WITH NO PROGRESS LINE THE FIRE WAS THE SILENCE LEG, which is the answer a
    # reader needs to investigate it: a loop that never came back, not one that
    # spun. The class is one token for both, so this is the only discriminator.
    assert stall_watchdog.fired_leg(4242, tmp_path / "logs") == stall_watchdog.LEG_SILENCE

    spinning = tmp_path / "logs" / f"{stall_watchdog.DUMP_PREFIX}-4243.log"
    spinning.write_text(
        f"header\n{stall_watchdog.PROGRESS_MARKER}runtime-stall-bound: 300s of CPU with no "
        f"progress\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n",
        encoding="utf-8",
    )
    assert stall_watchdog.fired_leg(4243, tmp_path / "logs") == stall_watchdog.LEG_PROGRESS
    # A PID WITH NO FILE IS NOT A FIRE, which is the third state a reader has to
    # be able to tell apart: no bound ran at all.
    assert stall_watchdog.fired_leg(999_999, tmp_path / "logs") is None


# -- the bound's own arithmetic ---------------------------------------------


def test_the_bound_is_the_default_unless_the_environment_says_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo costs the default bound, never no bound — except the deliberate 0.

    ``bound_seconds`` runs in the runtime's entry point, where an exception would
    take a session down over a diagnostic; unreadable values therefore fall back
    rather than raise, and ``0``/``off`` is the one spelling that means "do not
    arm" (an operator debugging a wedge, or a test that parks a loop on purpose).
    """
    monkeypatch.delenv(stall_watchdog.ENV_SECONDS, raising=False)
    assert stall_watchdog.bound_seconds() == stall_watchdog.DEFAULT_STALL_S

    monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "92.5")
    assert stall_watchdog.bound_seconds() == 92.5

    # THE FLOOR. A bound tighter than a few heartbeat intervals fires on runtimes
    # that are not stalled at all: review measured this knob at 4 s dumping and
    # killing a HEALTHY runtime at 4.1 s. Anything below three intervals is
    # raised to the floor (and logged) rather than honoured, and the floor is the
    # same 45 s this fleet already uses to call a heartbeat stale.
    floor = stall_watchdog.min_bound_seconds()
    assert floor == 45.0, f"the floor moved to {floor}s; the docstring argues 3 intervals"
    for too_tight in ("4", "12.5", str(floor - 0.001)):
        monkeypatch.setenv(stall_watchdog.ENV_SECONDS, too_tight)
        assert stall_watchdog.bound_seconds() == floor, too_tight

    for spelling in ("0", "off", "no", "false", ""):
        monkeypatch.setenv(stall_watchdog.ENV_SECONDS, spelling)
        assert stall_watchdog.bound_seconds() is None, spelling

    for typo in ("soon", "-1", str(stall_watchdog.MAX_BOUND_S * 2)):
        monkeypatch.setenv(stall_watchdog.ENV_SECONDS, typo)
        assert stall_watchdog.bound_seconds() == stall_watchdog.DEFAULT_STALL_S, typo


def test_a_disabled_bound_arms_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``=0`` must leave the process exactly as it was, file and all."""
    monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "0")
    assert stall_watchdog.arm(directory=tmp_path) is False
    assert stall_watchdog.is_armed() is False
    assert not list(tmp_path.glob(f"{stall_watchdog.DUMP_PREFIX}-*.log"))


def test_an_unusable_dump_directory_disarms_rather_than_failing_the_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic that cannot be armed must not take the runtime down with it.

    ``TMPDIR``-style breakage is the measured shape (see the shard watchdog's own
    guard for the same case): a path that exists but is not a directory, so
    ``mkdir`` raises inside the entry point.
    """
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    assert stall_watchdog.arm(directory=not_a_directory / "logs") is False
    assert stall_watchdog.is_armed() is False
    assert fake.armed == [], "a timer was armed for a dump that cannot be written"


# ============================================================================
# THE BOOT BOUND, AND THE BOUND ENGAGEMENT MOVES IT TO
# ============================================================================
#
# WHY THERE ARE TWO BOUNDS, in one read. ``arm`` runs from the entry point, before
# ``main()``, and it seeds both planes to that instant — but NOTHING IN THE BOOT
# PATH CAN MOVE A STAMP: the workload tick starts after publication and sleeps a
# heartbeat before its first stamp, and the serving beat starts with the serving
# thread. So every second a legitimate boot spends (construction, lease
# arbitration, MCP bring-up, the inbox drain, the socket bind, publication) is spent
# against a clock no boot code can re-arm. The cells below pin the two halves of the
# fix as ONE mechanism: the entry point arms the BOOT bound, and the first
# ENGAGEMENT moves the bound down to the steady one and stamps BOTH planes.
#
# The measured population this is sized for: over 68 retained dumps on the build
# carrying the observation header, 10 of its 17 fires carried the full ``0:05:00``
# value — no beat ever re-armed the timer, the never-engaged class, and an UPPER bound
# on that class rather than an exact count: a failed re-arm is indistinguishable from
# it by the armed value alone (see ``engage``). Those are what
# the boot bound covers and what the engagement reset makes NAMEABLE (a fire that
# re-armed has a deadline sibling; one that never did has none — and a re-arm that
# FAILED writes its own line instead).

#: One child, three arguments: the boot bound, the steady bound, and how long to wait
#: before engaging. The engage line it prints is the evidence the parent asserts on —
#: the bound it moved from and to, and BOTH planes' stamp deltas, so a reset that
#: stamped one plane cannot pass.
#:
#: A NEVER-ENGAGING SHAPE USED TO LIVE HERE and has moved to ``_BOOT_HOLD_CHILD`` below:
#: it armed with no probes, which is not how the entry point arms, so a cell built on it
#: could only ever agree with a claim about the boot window rather than with what a real
#: child does with one (agent review round 1, R1-1). The attribution it was there to pin
#: — the fired value, the absent sibling — is asserted by both of that child's cells.
_ENGAGE_CHILD = """
import os
import pathlib
import sys
import time

from local_operator.session.runtime import stall_watchdog

boot = float(sys.argv[1])
steady = float(sys.argv[2])
delay = float(sys.argv[3])
sentinel = pathlib.Path(sys.argv[4])

assert stall_watchdog.arm(boot_seconds=boot, seconds=steady), "the child could not arm the bound"
print(f"armed:{os.getpid()}", flush=True)
time.sleep(delay)
before = dict(stall_watchdog._ARMED.last_beat)
moved = stall_watchdog.engage()
after = dict(stall_watchdog._ARMED.last_beat)
deltas = " ".join(f"{plane}={after[plane] - before[plane]:.6f}" for plane in sorted(after))
print(
    f"engage:{moved} bound:{stall_watchdog._ARMED.boot_seconds:g}->"
    f"{stall_watchdog._ARMED.seconds:g} stamps:{deltas}",
    flush=True,
)
# Capture the live deadline before intentionally waiting for its diagnostic fire;
# after survival is proved, the bound is expected to be in the past.
epoch, leg = stall_watchdog.deadline_path().read_text(encoding="utf-8").split()
sentinel.with_name("deadline-capture.txt").write_text(
    f"{epoch} {leg} {time.time():.6f}", encoding="utf-8"
)

# This releases the GIL for the native timer, then lets the child observe the dump.
time.sleep(max(steady * 3, 1.0))
text = stall_watchdog.dump_path().read_text(encoding="utf-8")
assert stall_watchdog.FIRED_MARKER in text, "the native timer did not write its dump"
sentinel.write_text("engaged runtime survived dump", encoding="utf-8")
print("survived", flush=True)
"""


def _engage_run(
    tmp_path: Path,
    boot: float,
    steady: float,
    delay: float = 0.0,
) -> tuple[subprocess.CompletedProcess[str], int, Path, float, Path]:
    """Run one life of ``_ENGAGE_CHILD`` and return its dump and survival witness.

    Returns the finished process, its pid, dump path, elapsed time and a sentinel
    written only after the native dump appears while the child remains alive.
    """
    started = time.monotonic()
    sentinel = tmp_path / "engage-survived.txt"
    result = _run_script(
        _ENGAGE_CHILD, tmp_path, args=(str(boot), str(steady), str(delay), str(sentinel))
    )
    elapsed = time.monotonic() - started
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    return result, pid, _dump_for(tmp_path, pid), elapsed, sentinel


def test_the_entry_point_arms_the_boot_bound_and_engagement_moves_it_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BOTH HALVES, in process: 900 armed, 300 after engaging, both planes stamped.

    This is the whole decision as one cell, and each assertion is one of its parts:
    the bound armed is the BOOT one; ``engage`` moves it to the steady one (a RE-ARM,
    not a cancel and a re-arm); BOTH planes' stamps move to the engagement instant,
    which is what makes the next deadline a measurement of what happened AFTER the
    runtime engaged; the deadline sibling is written, so a post-engagement fire is
    distinguishable from the never-engaged class; and a SECOND engage is a no-op,
    which is what idempotence buys a boot path that might reach the line twice.

    MUTATION THIS CELL CATCHES: make ``engage`` a no-op (or let it move the bound
    without stamping a plane) — the armed list stays at one entry and the stamps stay
    at the arm instant, so the boot bound is all this process can ever fire at.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    try:
        assert stall_watchdog.arm(directory=tmp_path) is True
        assert [seconds for seconds, _, _ in fake.armed] == [stall_watchdog.DEFAULT_BOOT_STALL_S]
        armed = stall_watchdog._ARMED
        assert armed is not None
        assert armed.boot_seconds == stall_watchdog.DEFAULT_BOOT_STALL_S
        assert armed.steady_seconds == stall_watchdog.DEFAULT_STALL_S
        seeded = dict(armed.last_beat)

        assert stall_watchdog.engage() is True, (
            "engage reported nothing to do on a process whose armed bound is the boot "
            "one, so the steady bound can only ever be reached by waiting out boot"
        )
        # APPROX, not equality: the re-arm value is ``deadline() - now`` (the same shape
        # ``beat`` uses, so the timer always holds the time left to the earliest plane),
        # and that subtraction carries float error -- 300.00000000000006 on a Linux
        # runner where macOS produced exactly 300.0. The claim is which bound was armed,
        # and a half-second band states that without pinning an arithmetic accident.
        armed_bounds = [seconds for seconds, _, _ in fake.armed]
        assert armed_bounds[:1] == [stall_watchdog.DEFAULT_BOOT_STALL_S], armed_bounds
        assert armed_bounds[1:] == pytest.approx(
            [stall_watchdog.DEFAULT_STALL_S]
        ), f"engagement did not re-arm the timer at the steady bound: {armed_bounds!r}"
        for plane in (stall_watchdog.WORKLOAD, stall_watchdog.SERVING):
            assert armed.last_beat[plane] > seeded[plane], (
                f"the {plane} plane's stamp is still at the arm instant, so a fire after "
                f"engagement would still be measured from the entry point"
            )
        epoch, leg = _deadline_record(tmp_path, os.getpid())
        assert leg in (stall_watchdog.WORKLOAD, stall_watchdog.SERVING), leg
        # The tolerance is not slack: the deadline is computed from ``time.monotonic`` and
        # read back against ``time.time``, so it can sit microseconds over the bound. The
        # DISCRIMINATION is the boot bound (900 s) versus the steady one (300 s), which a
        # half-second band separates with room to spare.
        remaining = epoch - time.time()
        assert 0 < remaining <= stall_watchdog.DEFAULT_STALL_S + 0.5, (
            f"the sibling engagement wrote says {remaining:.1f}s left, which is not the "
            f"steady bound this process moved to"
        )

        assert stall_watchdog.engage() is False, "a second engage moved the bound again"
        assert len(fake.armed) == 2, fake.armed
    finally:
        stall_watchdog.disarm()


def test_engage_never_widens_the_steady_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE INVARIANT: engagement only ever moves the bound DOWN, or does nothing.

    Two shapes, because both are states a real process can be in. An explicit
    ``arm(seconds=...)`` states one bound for the whole arming, so there is no boot
    phase to end; a steady knob set ABOVE the boot bound arms at the larger of the two,
    so the bound is already the steady one by construction. In neither may engagement
    touch the timer — an engage that "moved the bound to the default" would RELAX the
    diagnostic window, which is the one thing this function must never do.

    MUTATION THIS CELL CATCHES: ``armed.seconds = max(armed.seconds, steady_seconds)``,
    or an unconditional ``armed.seconds = DEFAULT_STALL_S`` — the first is caught by the
    explicit-bound arm, the second by both.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    try:
        assert stall_watchdog.arm(seconds=SHORT_BOUND_S, directory=tmp_path) is True
        assert (
            stall_watchdog.engage() is False
        ), "engagement re-armed a process that stated one bound for its whole arming"
        assert [seconds for seconds, _, _ in fake.armed] == [float(SHORT_BOUND_S)], fake.armed

        stall_watchdog.disarm()
        monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "1200")
        assert stall_watchdog.arm(directory=tmp_path) is True
        assert [seconds for seconds, _, _ in fake.armed] == [
            float(SHORT_BOUND_S),
            1200.0,
        ], "the armed bound is not the larger of the two bounds"
        assert (
            stall_watchdog.engage() is False
        ), "engagement extended a bound an operator set ABOVE the boot bound"
        assert [seconds for seconds, _, _ in fake.armed] == [float(SHORT_BOUND_S), 1200.0]
    finally:
        stall_watchdog.disarm()


def test_the_runtime_log_names_both_bounds_when_they_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The operator-facing line says what the dump header says, or it misstates the bound.

    ``announce`` runs from ``main``'s first lines, which is BOOT — so before this split
    its one number was the whole story, and after it a line naming only the boot bound
    would tell a reader that a frozen runtime is judged at 900 s when from publication
    (seconds later) it is judged at 300 s. One bound still reads as one number: the
    pre-split wording is asserted here too, because a cell that only knew the split
    would let the single-bound log line drift away unnoticed.

    MUTATION THIS CELL CATCHES: drop the two-bound branch from ``announce`` — the line
    goes back to naming the boot bound alone and the second half of this cell fails.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    try:
        assert stall_watchdog.arm(directory=tmp_path) is True
        with caplog.at_level(logging.INFO, logger=stall_watchdog.logger.name):
            stall_watchdog.announce()
        line = " ".join(
            record.getMessage()
            for record in caplog.records
            if record.name == stall_watchdog.logger.name
        )
        assert (
            f"{stall_watchdog.DEFAULT_BOOT_STALL_S:.0f}s of no progress during BOOT" in line
        ), line
        assert f"then {stall_watchdog.DEFAULT_STALL_S:.0f}s once it engages" in line, line

        stall_watchdog.disarm()
        caplog.clear()
        assert stall_watchdog.arm(seconds=SHORT_BOUND_S, directory=tmp_path) is True
        with caplog.at_level(logging.INFO, logger=stall_watchdog.logger.name):
            stall_watchdog.announce()
        line = " ".join(
            record.getMessage()
            for record in caplog.records
            if record.name == stall_watchdog.logger.name
        )
        assert f"{SHORT_BOUND_S:.0f}s of no progress dumps every thread" in line, line
        assert "during BOOT" not in line, line
    finally:
        stall_watchdog.disarm()


def test_engagement_keeps_the_native_timer_dump_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every engage-time native re-arm is diagnostic, regardless of sampled state.

    Engagement is a distinct timer site and starts the steady phase; pin it alongside
    the shared wrapper so a future refactor cannot make only this deadline fatal. The
    work probe still informs diagnostic metadata, but cannot authorize native exit.
    """
    captured: list[bool] = []
    monkeypatch.setattr(
        stall_watchdog.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, file, exit: captured.append(exit),
    )
    monkeypatch.setattr(stall_watchdog, "_start_sampler", lambda armed: None)
    try:
        assert stall_watchdog.arm(
            boot_seconds=600.0, seconds=300.0, busy=lambda: True, directory=tmp_path
        )
        assert captured == [False], captured
        assert stall_watchdog.engage() is True
        assert captured == [False, False], captured

        # An idle sample is no authority either: expiry stays diagnostic-only.
        stall_watchdog.disarm()
        captured.clear()
        assert stall_watchdog.arm(
            boot_seconds=600.0, seconds=300.0, busy=lambda: False, directory=tmp_path
        )
        assert stall_watchdog.engage() is True
        assert captured == [False, False], captured
    finally:
        stall_watchdog.disarm()


def test_engage_with_nothing_armed_is_a_no_op(tmp_path: Path) -> None:
    """An in-process host calls it on a boot path that never armed the process timer.

    ``beat`` has the same contract for the same reason (a TUI, a test), and the failure
    it prevents is loud rather than subtle: an engage that raised, or that armed a
    process-global C timer from a library path, would displace the timer the fleet's
    e2e stages own.
    """
    assert not stall_watchdog.is_armed()
    assert stall_watchdog.engage() is False
    assert not list(tmp_path.glob(f"{stall_watchdog.DUMP_PREFIX}-*.log"))


def test_the_boot_knob_resolves_like_the_steady_one_but_means_the_opposite_by_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One resolver, two knobs — and ``off`` reads the same and means something else.

    The steady knob's ``0``/``off`` disables the WATCHDOG; the boot knob's spells "no
    boot phase", which arms straight at the steady bound. Both are asserted here
    because a reader who assumes they behave alike would set one and get the other,
    and the two are one line apart in the same function.
    """
    monkeypatch.delenv(stall_watchdog.ENV_BOOT_SECONDS, raising=False)
    assert stall_watchdog.boot_bound_seconds() == stall_watchdog.DEFAULT_BOOT_STALL_S

    monkeypatch.setenv(stall_watchdog.ENV_BOOT_SECONDS, "600")
    assert stall_watchdog.boot_bound_seconds() == 600.0

    floor = stall_watchdog.min_bound_seconds()
    monkeypatch.setenv(stall_watchdog.ENV_BOOT_SECONDS, "4")
    assert stall_watchdog.boot_bound_seconds() == floor, "the boot bound skipped the floor"

    for typo in ("soon", "-1", str(stall_watchdog.MAX_BOUND_S * 2)):
        monkeypatch.setenv(stall_watchdog.ENV_BOOT_SECONDS, typo)
        assert stall_watchdog.boot_bound_seconds() == stall_watchdog.DEFAULT_BOOT_STALL_S, typo

    for spelling in ("0", "off", "no", "false", ""):
        monkeypatch.setenv(stall_watchdog.ENV_BOOT_SECONDS, spelling)
        assert stall_watchdog.boot_bound_seconds() is None, spelling


def test_a_boot_knob_spelled_off_arms_the_steady_bound_and_only_the_steady_knob_disarms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``off`` on the boot knob is the pre-split behaviour, not an unarmed runtime.

    Worth its own cell because the two knobs share a resolver and would be easy to wire
    to one meaning: a test or an operator that spells the boot bound off must get a
    runtime armed at the steady bound from the first instant (exactly what shipped
    before the split), while ``0`` on the STEADY knob must still leave the process
    completely untouched.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    try:
        monkeypatch.setenv(stall_watchdog.ENV_BOOT_SECONDS, "off")
        assert stall_watchdog.arm(directory=tmp_path) is True
        assert [seconds for seconds, _, _ in fake.armed] == [stall_watchdog.DEFAULT_STALL_S]
        assert stall_watchdog.engage() is False, "there was no boot phase to end"

        stall_watchdog.disarm()
        monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "0")
        assert (
            stall_watchdog.arm(directory=tmp_path) is False
        ), "the boot knob's off spelling disabled the watchdog; only the steady knob may"
        assert stall_watchdog.is_armed() is False
    finally:
        stall_watchdog.disarm()


def test_a_boot_bound_below_the_steady_one_is_announced_not_silently_dropped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """QA round 1, Q-1: the one unusable boot value that used to vanish in silence.

    ``arm`` never arms tighter than the steady bound (``bound = max(boot, steady)``),
    which is intended twice over — the boot stretch must not be judged more tightly than
    the steady one, and ``max`` is what makes ``engage`` a downward move by
    construction — but a boot bound BELOW the steady one therefore leaves no trace at
    all: the header, the announce line and the timer all carry the steady bound while
    the operator's knob said otherwise. Every other unusable spelling (``abc``, ``-5``,
    ``1e12``, below the floor) already warns from ``_bound_from_raw``. Measured before
    this warning: ``LOP_RUNTIME_BOOT_STALL_SECONDS=60`` → ``boot_bound_seconds() ==
    60.0``, ``armed == 300.0``, ``warnings == []``.

    The second half is the inverse, and it is the reason this is a warning rather than
    a line in ``announce``: an operator who configured nothing must not be told about a
    clamp that did not happen. The arithmetic is asserted unchanged in both halves, so
    a future edit cannot satisfy this cell by moving the bound.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    try:
        with caplog.at_level(logging.WARNING, logger=stall_watchdog.logger.name):
            assert stall_watchdog.arm(boot_seconds=60.0, seconds=300.0, directory=tmp_path) is True
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == stall_watchdog.logger.name
        ]
        assert any(
            "boot_seconds=60" in message and "below the steady bound of 300" in message
            for message in messages
        ), f"the dropped boot bound was not announced: {messages}"
        assert [seconds for seconds, _, _ in fake.armed] == [300.0], (
            "the warning must not change the arithmetic: boot is still never armed "
            f"tighter than the steady bound, but the timer took {fake.armed}"
        )

        stall_watchdog.disarm()
        caplog.clear()
        fake.armed.clear()
        with caplog.at_level(logging.WARNING, logger=stall_watchdog.logger.name):
            assert stall_watchdog.arm(directory=tmp_path) is True
        noisy = [
            record.getMessage()
            for record in caplog.records
            if record.name == stall_watchdog.logger.name
        ]
        assert (
            not noisy
        ), f"the shipped default pair warned, which would put a line on every boot: {noisy}"
        assert [seconds for seconds, _, _ in fake.armed] == [
            stall_watchdog.DEFAULT_BOOT_STALL_S
        ], fake.armed
    finally:
        stall_watchdog.disarm()


#: THE PRODUCTION ARM OF THE BOOT WINDOW, as ``process.py``'s ``__main__`` guard
#: makes it: ``arm(probe=process._progress_probe, busy=process._busy_probe)``. The two
#: cells below are this one child with different answers to "is anything in flight",
#: because that answer is what decides whether a boot-phase fire ENDS the process —
#: and the production answer during boot is ``_busy_probe``'s ``True`` (no handle yet),
#: not the ``False`` an arming that supplies no probe produces. A cell that arms
#: without a probe and then asserts a cut agrees with a claim no runtime follows
#: (agent review round 1, R1-1); these cells read the answer off the production
#: callables themselves.
#:
#: The park is a GIL-RELEASING ``CDLL`` sleep rather than ``_ENGAGE_CHILD``'s
#: GIL-holding ``PyDLL`` one, and that is what the held case needs: the process must
#: SURVIVE the fire and still be able to report what it found.
_BOOT_HOLD_CHILD = """
import ctypes
import os
import pathlib
import sys
import time

from local_operator.session.runtime import process, stall_watchdog

probes = sys.argv[1]
boot = float(sys.argv[2])
steady = float(sys.argv[3])
park = float(sys.argv[4])
sentinel = pathlib.Path(sys.argv[5])

if probes == "prod":
    # THE PRE-PUBLICATION BOOT STATE, asserted rather than assumed: this is the window
    # whose exit-leg answer both cells are about, and a rig that drifted out of it
    # would be measuring a different question.
    assert process._live_handle is None, "this child is not in the boot state"
    assert (
        process._busy_probe() is True
    ), "the production busy probe no longer reports work in flight during boot"
    probe, busy = process._progress_probe, process._busy_probe
else:
    # No probe can describe work state; absence never authorizes native termination.
    probe, busy = None, None

if not stall_watchdog.arm(boot_seconds=boot, seconds=steady, probe=probe, busy=busy):
    raise SystemExit("the child could not arm the bound")
armed = stall_watchdog._ARMED
print(
    f"armed:{os.getpid()} probes:{probes} bound:{armed.seconds:g} held:{armed.held}",
    flush=True,
)

lib = ctypes.CDLL(None)
lib.sleep.argtypes = [ctypes.c_uint]
started = time.monotonic()
lib.sleep(int(park))
text = armed.path.read_text(encoding="utf-8")
print(
    f"survived:{time.monotonic() - started:.2f} held:{armed.held} "
    f"fires:{text.count(stall_watchdog.FIRED_MARKER)} "
    f"held_marker:{stall_watchdog.HELD_MARKER in text}",
    flush=True,
)
assert stall_watchdog.FIRED_MARKER in text, "the native timer did not write its dump"
sentinel.write_text("boot survived dump", encoding="utf-8")
"""


def _boot_hold_run(
    tmp_path: Path, probes: str, boot: float, steady: float, park: float
) -> tuple[subprocess.CompletedProcess[str], int, Path, Path]:
    """Run one life of ``_BOOT_HOLD_CHILD`` with a post-fire survival sentinel."""
    sentinel = tmp_path / "boot-survived.txt"
    result = _run_script(
        _BOOT_HOLD_CHILD,
        tmp_path,
        args=(probes, str(boot), str(steady), str(park), str(sentinel)),
    )
    # THE CHILD'S OWN DIAGNOSIS FIRST, because it is the informative one: the child
    # asserts the pre-publication boot state before it arms, so a change to the probe it
    # is about lands in its stderr and not in a parse error here.
    if "armed:" not in result.stdout:
        raise AssertionError(
            f"the child never armed: rc={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    if not sentinel.is_file():
        raise AssertionError(
            f"the child did not survive and observe its dump: rc={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    return result, pid, _dump_for(tmp_path, pid), sentinel


def test_a_hung_boot_with_the_production_probes_is_dumped_and_HELD(
    tmp_path: Path,
) -> None:
    """R1-1: what the ENTRY POINT's arming actually does with a boot that never ends.

    The entry point arms as ``arm(probe=process._progress_probe,
    busy=process._busy_probe)``, and ``_busy_probe`` answers **``True`` while
    ``_live_handle is None``** — the whole pre-publication boot — so ``_Armed.held``
    is True. A hung boot is therefore DUMPED at the boot bound and survives it;
    publication changes the bound to steady but never grants the native timer an exit
    decision. A boot that never publishes remains for operator inspection and stop.

    WHY THIS CELL EXISTS AND WHAT IT REPLACES. The cell below pins the boot bound's
    ATTRIBUTION through an arming with NO busy probe; its unknown work state does not
    reproduce the production entry-point contract (agent review round 1, R1-1).
    This cell arms through the production callables, so a change to either of them
    moves it: ``_busy_probe`` answering False during boot, ``_Armed.held`` no longer
    seeded from the probe, or ``_arm_timer``'s exit leg ceasing to read it all turn
    this cell red on a surviving process.

    MUTATION THIS CELL CATCHES: arm with ``busy=lambda: False`` (or drop the probe)
    and the child no longer records the held-fire evidence below; changing the boot
    probe or its initial sampled state likewise fails the child-level assertions.
    """
    boot, steady = 2 * SHORT_BOUND_S, SHORT_BOUND_S
    result, pid, dump, sentinel = _boot_hold_run(tmp_path, "prod", boot, steady, 4.0)

    line = next((row for row in result.stdout.splitlines() if row.startswith("armed:")), "")
    assert f"probes:prod bound:{boot:g} held:True" in line, (
        f"the production arming no longer holds the boot phase: {line!r} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    summary = next((row for row in result.stdout.splitlines() if row.startswith("survived:")), "")
    assert result.returncode == 0 and summary, (
        f"a boot-phase fire ended a boot the production probe reported in flight: rc="
        f"{result.returncode} {result.stdout!r} {result.stderr!r}"
    )
    assert summary.split()[1] == "held:True", summary
    assert sentinel.read_text(encoding="utf-8") == "boot survived dump"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:400]
    assert _fired_seconds(text) == pytest.approx(float(boot)), (
        f"the fire carried {_fired_seconds(text)}s rather than the {boot:g}s boot bound: "
        f"{text[:400]!r}"
    )
    assert any(
        row.startswith(stall_watchdog.HELD_MARKER) for row in text.splitlines()
    ), f"the dump does not say the boot-phase fire was held: {text[:900]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is True
    boot_note = text.split("THIS IS THE BOOT BOUND", 1)[1].split(
        stall_watchdog.OBSERVATION_NOT_VERDICT, 1
    )[0]
    assert f"When re-arming succeeds, a fired value of {boot:g}s" in boot_note, boot_note
    assert "means the runtime never engaged" in boot_note, boot_note
    assert "If engagement could not re-arm the timer" in boot_note, boot_note
    assert "see the re-arm-failed message in this dump" in boot_note, boot_note
    assert "the timer may still fire at the boot bound" in boot_note, boot_note
    assert not stall_watchdog.deadline_path(pid, tmp_path / "logs").exists(), (
        "a never-engaged held fire left a deadline sibling, so presence no longer "
        "answers 'did anything ever re-arm this timer'"
    )


def test_a_never_engaging_boot_without_a_probe_is_dumped_and_survives(
    tmp_path: Path,
) -> None:
    """An unknown work state still gets a diagnostic-only boot fire.

    No probe can describe work state, but that uncertainty does not authorize native
    termination. The child observes the real timer marker and writes a sentinel; the
    fired value and absent deadline sibling still show that no beat or engagement
    replaced the boot timer.
    """
    boot, steady = 2 * SHORT_BOUND_S, SHORT_BOUND_S
    result, pid, dump, sentinel = _boot_hold_run(tmp_path, "noprobe", boot, steady, 4.0)
    assert result.returncode == 0, (
        f"the native timer ended a boot with no probe: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert "survived:" in result.stdout, result.stdout
    assert sentinel.read_text(encoding="utf-8") == "boot survived dump"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:400]
    assert stall_watchdog.FIRED_MARKER in text, text[:400]
    assert not any(
        row.startswith(stall_watchdog.HELD_MARKER) for row in text.splitlines()
    ), "the no-probe child has no sampler to append the held-fire observation"
    assert _fired_seconds(text) == pytest.approx(float(boot)), (
        f"a runtime that never engaged fired at {_fired_seconds(text)}s rather than its "
        f"{boot:g}s boot bound: {text[:400]!r}"
    )
    assert f"armed for {boot:g}s" in text, (
        "the header does not name the bound the timer was armed for, so the fired value "
        f"cannot be attributed: {text[:400]!r}"
    )
    boot_note = text.split("THIS IS THE BOOT BOUND", 1)[1].split(
        stall_watchdog.OBSERVATION_NOT_VERDICT, 1
    )[0]
    assert f"When re-arming succeeds, a fired value of {boot:g}s" in boot_note, boot_note
    assert "means the runtime never engaged" in boot_note, boot_note
    assert "If engagement could not re-arm the timer" in boot_note, boot_note
    assert "see the re-arm-failed message in this dump" in boot_note, boot_note
    assert "the timer may still fire at the boot bound" in boot_note, boot_note
    assert not stall_watchdog.deadline_path(pid, tmp_path / "logs").exists(), (
        "a never-engaged fire left a deadline sibling, so presence no longer answers "
        "'did anything ever re-arm this timer'"
    )


# This child proves the counterexample to an unconditional reading of the boot value:
# engage moves the in-memory bound, but an unsuccessful timer replacement leaves the
# original C timer in force and records why the boot value may still fire.
_ENGAGE_REARM_FAILURE_CHILD = """
import ctypes
import os
import pathlib
import sys

from local_operator.session.runtime import stall_watchdog

boot = float(sys.argv[1])
steady = float(sys.argv[2])
sentinel = pathlib.Path(sys.argv[3])
assert stall_watchdog.arm(boot_seconds=boot, seconds=steady), "the child could not arm the bound"
armed = stall_watchdog._ARMED
print(f"armed:{os.getpid()} bound:{armed.seconds:g}", flush=True)
real_arm_timer = stall_watchdog._arm_timer
def refuse_rearm(*args, **kwargs):
    raise OSError("timer re-arm refused by test")
stall_watchdog._arm_timer = refuse_rearm
try:
    moved = stall_watchdog.engage()
finally:
    stall_watchdog._arm_timer = real_arm_timer
print(f"engage:{moved} bound:{armed.boot_seconds:g}->{armed.seconds:g}", flush=True)
lib = ctypes.CDLL(None)
lib.sleep.argtypes = [ctypes.c_uint]
lib.sleep(int(boot + 1))
text = armed.path.read_text(encoding="utf-8")
assert stall_watchdog.FIRED_MARKER in text, "the original timer did not fire"
sentinel.write_text("failed re-arm runtime survived dump", encoding="utf-8")
print("survived", flush=True)
"""


def test_engagement_rearm_failure_qualifies_a_boot_bound_fire(
    tmp_path: Path,
) -> None:
    """A failed engage re-arm leaves the original diagnostic deadline in force.

    The child runs the real timer, refuses only the engagement replacement, then
    observes the original fire and writes a sentinel. The qualified boot note and
    failure marker explain why the fire carries the boot value despite engagement.
    """
    boot, steady = 4 * SHORT_BOUND_S, SHORT_BOUND_S
    sentinel = tmp_path / "rearm-failure-survived.txt"
    result = _run_script(
        _ENGAGE_REARM_FAILURE_CHILD,
        tmp_path,
        args=(str(boot), str(steady), str(sentinel)),
    )
    assert result.returncode == 0, (
        f"the original timer ended the child after engage's failed re-arm: "
        f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "survived" in result.stdout, result.stdout
    assert sentinel.read_text(encoding="utf-8") == "failed re-arm runtime survived dump"
    assert f"engage:True bound:{boot:g}->{steady:g}" in result.stdout, result.stdout
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert _fired_seconds(text) == pytest.approx(float(boot)), text[:900]
    assert stall_watchdog.REARM_FAILED_MARKER in text, text[-800:]
    # Assert on the actual C-timer artifact, not only on the source constant: the
    # separate boot note is insufficient if this general reading stays unconditional.
    assert "PROVIDED THE RE-ARM SUCCEEDED" in text, text[:1600]
    assert "An ENGAGE whose timer replacement FAILED is the exception" in text, text[:1600]
    assert "that line's absence is what makes the never-engaged reading safe" in text, text[:1600]
    assert stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE in text, text[:1600]
    assert text.index(stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE) < text.index(
        stall_watchdog.REARM_FAILED_MARKER
    ), text[:1600]
    boot_note = text.split("THIS IS THE BOOT BOUND", 1)[1].split(
        stall_watchdog.OBSERVATION_NOT_VERDICT, 1
    )[0]
    assert f"When re-arming succeeds, a fired value of {boot:g}s" in boot_note, boot_note
    assert "means the runtime never engaged" in boot_note, boot_note
    assert "If engagement could not re-arm the timer" in boot_note, boot_note
    assert "see the re-arm-failed message in this dump" in boot_note, boot_note
    assert "the timer may still fire at the boot bound" in boot_note, boot_note


def test_engagement_moves_the_bound_to_the_steady_one_and_stamps_both_planes(
    tmp_path: Path,
) -> None:
    """After engagement, a real dump carries the STEADY bound and survives.

    A 4 s boot bound and a 1 s steady bound, then an engage and a park: the dump must
    carry the steady bound, not the boot one, and the sibling engage wrote must count
    down from the engagement instant. Both halves matter to the fleet — the boot bound
    buys a slow boot its time, and the steady bound is what #1438's executing-loop
    diagnostics and #1439's in-flight observations use, so an engage that carried the
    boot bound forward for the process's whole life would quietly disable them.

    MUTATION THIS CELL CATCHES: widen the steady bound inside ``engage`` (e.g. set it to
    ``DEFAULT_STALL_S``, or ``max`` the two) — the fired value becomes the wide one and
    this cell, and only this cell, goes red. It also catches a reset that stamps ONE
    plane: the other plane's stamp stays at the arm instant, so the re-armed deadline is
    measured from the arm and the child outlives the window it should not.
    """
    boot, steady = 4 * SHORT_BOUND_S, SHORT_BOUND_S
    result, pid, dump, _elapsed, sentinel = _engage_run(tmp_path, boot, steady)
    line = next((row for row in result.stdout.splitlines() if row.startswith("engage:")), "")
    assert (
        "engage:True" in line
    ), f"the child never engaged, so nothing here is about the reset: {result.stdout!r}"
    assert (
        f"bound:{boot:g}->{steady:g}" in line
    ), f"engagement did not move the bound from the boot bound to the steady one: {line!r}"
    deltas = {
        plane: float(delta)
        for plane, delta in (part.split("=") for part in line.split("stamps:", 1)[1].split())
    }
    assert set(deltas) == {"serving", "workload"}, line
    for plane, delta in deltas.items():
        assert delta > 0.0, (
            f"engagement left the {plane} plane's stamp at the arm instant, so the deadline "
            f"it re-armed is measured from the entry point: {line!r}"
        )

    assert result.returncode == 0, (
        f"the steady-bound timer ended its child: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert "survived" in result.stdout, result.stdout
    assert sentinel.read_text(encoding="utf-8") == "engaged runtime survived dump"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:600]
    # A 50 ms band, not equality: ``_rearm`` derives the remaining time from the stamps
    # this engagement just moved, so it is a few microseconds UNDER the steady bound (the
    # arithmetic runs a moment after the stamp) and the fired value carries that. The
    # discrimination this cell exists for is the boot bound (4 s here) against the steady
    # one (1 s), which a band this width separates many times over.
    assert _fired_seconds(text) == pytest.approx(float(steady), abs=0.05), (
        f"a runtime silent AFTER engaging fired at {_fired_seconds(text)}s rather than the "
        f"{steady:g}s steady bound: {text[:400]!r}"
    )
    epoch_text, leg, captured_text = (
        sentinel.with_name("deadline-capture.txt").read_text(encoding="utf-8").split()
    )
    assert leg in (stall_watchdog.WORKLOAD, stall_watchdog.SERVING), leg
    # Compare against the time captured immediately after engagement, not this
    # parent's post-fire time; the sentinel proves the runtime survived past expiry.
    remaining = float(epoch_text) - float(captured_text)
    assert -0.5 < remaining <= float(steady) + 0.5, (
        f"the sibling counts down from {remaining:.2f}s, which is not the steady bound "
        f"measured from the engagement instant"
    )


def test_the_deadline_resets_on_engagement_not_on_arm(tmp_path: Path) -> None:
    """THE OPERATOR'S ASK, measured: arm, outlive the steady bound, engage, go silent.

    The steady bound is 1 s and the child waits 2 s BEFORE engaging, so a deadline that
    reset on the arm (or an armed bound that was the steady one) would have fired before
    the engage line was printed — and a deadline that did not reset on engagement would
    fire 4 s after the arm carrying 4 on the fired line. Both are asserted: the engage
    line is present, the whole run outlived the steady bound measured from the arm, and
    the child reports the dump and writes a sentinel after the steady fire.

    MUTATION THIS CELL CATCHES: an ``engage`` whose bound move is a no-op (or an
    ``arm`` that arms the steady bound) — the first fires at the boot value, the second
    fires before the child can print.
    """
    boot, steady, delay = 4 * SHORT_BOUND_S, SHORT_BOUND_S, 2.0
    result, _pid, dump, elapsed, sentinel = _engage_run(tmp_path, boot, steady, delay)
    assert "engage:True" in result.stdout, (
        f"the child did not reach its engagement, so a fire before it says nothing about "
        f"the reset: {result.stdout!r} {result.stderr!r}"
    )
    assert elapsed > delay + float(steady) - 0.25, (
        f"the run lasted {elapsed:.2f}s, so it did not outlive the steady bound measured "
        f"from the arm and this cell cannot tell a reset from an arm-time bound"
    )
    assert result.returncode == 0, (
        f"the native timer ended the child: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert "survived" in result.stdout, result.stdout
    assert sentinel.read_text(encoding="utf-8") == "engaged runtime survived dump"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:600]
    # The band is the same one as above and for the same reason (``_rearm`` recomputes
    # from the stamps, so the value lands microseconds under the bound); what it must
    # separate here is the steady bound from the 4 s boot bound, four times away.
    assert _fired_seconds(text) == pytest.approx(float(steady), abs=0.05), (
        f"the fire carries {_fired_seconds(text)}s rather than the steady bound the "
        f"engagement moved to: {text[:400]!r}"
    )


# -- the structure a real firing cannot be asked about -----------------------


def test_the_header_is_written_before_the_timer_is_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write-then-act, as an ordering fact rather than as a promise.

    ``faulthandler`` writes with a raw descriptor from a C thread, so a header
    deferred to fire time would never exist for the reader who arrives after the
    process is gone. The spy reads the file's text at arm time, so a reordering
    that armed first and wrote later fails here.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [(5.0, False)], fake.armed
    assert stall_watchdog.ARM_MARKER in fake.text_at_arm
    assert str(os.getpid()) in fake.text_at_arm
    assert fake.armed[0][2].read_text(encoding="utf-8") == fake.text_at_arm


def test_each_plane_is_tracked_apart_and_a_silent_one_shrinks_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy plane must NOT hide a silent one from the diagnostic bound.

    This is A2's fix as a property of the timer arithmetic, and it is the
    assertion the pre-fix design fails: with one shared "last beat", every tick
    from any plane re-armed the full bound, so a serving plane that kept reporting
    masked a workload plane that had stopped — which is precisely the measured
    failure (the serving plane's heartbeat is on its own thread and stays healthy
    while the workload loop is parked in a scan). Here only SERVING beats after
    the arm, and every one of its ticks must SHORTEN the timer toward the silent
    plane's deadline rather than push it out.

    Also pins the two properties a later edit is most likely to lose: a beat never
    cancels a live timer (the re-arm replaces it, and a cancel-then-fail leaves the
    process with NO bound while the comment claims otherwise — A3), and arming is
    idempotent so a second call cannot leak a handle or displace the file.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    stall_watchdog.beat(stall_watchdog.WORKLOAD)
    assert fake.cancels == 0 and fake.armed == [], "a beat armed a timer no one asked for"

    assert stall_watchdog.arm(seconds=10.0, directory=tmp_path) is True
    assert stall_watchdog.arm(seconds=99.0, directory=tmp_path) is True
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [
        (10.0, False)
    ], "a second arm displaced the first instead of being a no-op"

    stall_watchdog.beat(stall_watchdog.SERVING)
    stall_watchdog.beat(stall_watchdog.SERVING)
    assert fake.cancels == 0, "a beat cancelled a live timer; the re-arm replaces it"
    armed_for = [seconds for seconds, _, _ in fake.armed]
    assert len(armed_for) == 3, armed_for
    assert (
        armed_for[1] < armed_for[0] and armed_for[2] < armed_for[0]
    ), f"the healthy plane re-armed for the FULL bound, so it can mask a silent one: {armed_for}"
    assert not any(
        exit_ for _, exit_, _ in fake.armed
    ), "a diagnostic timer may not terminate the runtime"

    # A typo is not a third plane: it is logged and ignored rather than creating a
    # stamp that nothing would ever refresh (which would fire on a healthy runtime).
    stall_watchdog.beat("servring")
    assert [seconds for seconds, _, _ in fake.armed] == armed_for
    assert stall_watchdog.is_armed() is True

    # A clean disarm removes the file: see the module docstring on why this
    # ``unlink`` is allow-listed, and on what a surviving file then means.
    file = fake.armed[0][2]
    stall_watchdog.disarm()
    assert stall_watchdog.is_armed() is False
    assert fake.cancels == 1, "disarm must cancel the bound it is giving up"
    assert not file.exists(), "a clean disarm left the dump behind, so it no longer means 'fired'"


# -- the interlock with the two pytest-side watchdogs ------------------------


def test_an_in_process_entry_point_arms_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main()`` is callable in-process, and a pytest worker must stay unarmed.

    THIS IS THE INTERLOCK. ``faulthandler``'s timer is process-global, so if the
    runtime's ordinary entry function armed it, then any unit test that drives
    ``process.main()`` in-process (one does, for the log file) would take the
    timer away from the shard watchdog — and any e2e test that boots a runtime
    inside a ``tests.e2e.watchdog.bounded`` block would silently disarm that
    stage's only bound for the rest of the block. Arming therefore lives in the
    ``__main__`` branch, which only ``python -m`` reaches, and this asserts the
    behaviour rather than trusting the layout.
    """
    from local_operator.session.runtime import process

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    async def fake_amain(**_kwargs: object) -> int:
        # ``**kwargs`` because ``main`` passes the operator capability through to
        # the real ``amain`` (issue #1310); a double that pins the signature would
        # fail on a parameter this test is not about. Same shape as
        # ``test_process_reaper``'s double, for that shared reason.
        return 0

    monkeypatch.setattr(process, "amain", fake_amain)
    # ``main`` reconfigures the ROOT logger; the suite's own autouse
    # ``restore_root_logger`` fixture puts that back for the rest of the worker.
    assert process.main() == 0
    assert stall_watchdog.is_armed() is False, (
        "an in-process entry point armed the process-global timer, which would "
        "silence the shard watchdog and the e2e stage's own bound"
    )
    assert not list((tmp_path / "logs").glob(f"{stall_watchdog.DUMP_PREFIX}-*.log"))


def test_the_only_arm_site_is_the_runtime_entry_point() -> None:
    """Arming is a property of the ENTRY POINT, asserted against the source.

    The behavioural twin is above; this catches the shape that would defeat it —
    an ``arm`` call added to ``RuntimeServer`` (reachable via ``start_in_process``,
    which in-process hosts and the whole test suite use) or to any other module.
    Also pins that ``server.py`` touches this module ONLY through ``beat``: a
    ``disarm`` or an ``arm`` from the serving plane would let a library path
    decide the fate of the process-global timer.

    And pins the SAME shape for ``engage``: one call site, on the boot path at the
    publication boundary, and never reachable from the ``__main__`` guard — it moves
    the bound and stamps planes, so a second site (or a library-path one, on an
    in-process host that never armed) would be the same defect this cell exists for.
    """
    arm_sites = [
        str(path)
        for path in sorted((REPO / "local_operator").rglob("*.py"))
        if "stall_watchdog.arm(" in path.read_text(encoding="utf-8")
    ]
    offenders = [site for site in arm_sites if Path(site).name != "process.py"]
    assert offenders == [], f"a library path arms the process-global stall timer: {offenders}"
    assert arm_sites, "no arm site at all: the watchdog is inert"

    process_source = (REPO / "local_operator" / "session" / "runtime" / "process.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(process_source)
    main_guard = [
        node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
    ]
    assert len(main_guard) == 1, "process.py no longer has exactly one __main__ guard"
    guarded = "\n".join(ast.unparse(node) for node in main_guard[0].body)
    assert (
        "stall_watchdog.arm(" in guarded
    ), "the arm call is not inside the __main__ guard, so an in-process caller can reach it"
    assert (
        "stall_watchdog.arm(" not in process_source.split("if __name__")[0]
    ), "an arm call sits outside the entry point"
    # ...and exactly one, so a second arming cannot appear unnoticed.
    assert process_source.count("stall_watchdog.arm(") == 1

    server_source = (REPO / "local_operator" / "session" / "runtime" / "server.py").read_text(
        encoding="utf-8"
    )
    touched = {
        node.attr
        for node in ast.walk(ast.parse(server_source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "stall_watchdog"
    }
    # The serving plane reports progress and NOTHING ELSE: no ``arm``, no
    # ``disarm``. Its plane name travels with the beat, which is why the set is
    # more than one symbol — the point of the pin is that the serving plane cannot
    # create, move or cancel the process-global timer.
    # The serving plane reports progress, and — since its tick can now RECORD its own
    # death (#1425's instrument, which is what lets a reader tell a dead reporter from
    # a silent plane) — that it died. NOTHING ELSE: no ``arm``, no ``disarm``, no
    # deadline arithmetic. Its plane name travels with the beat, which is why the set
    # is more than one symbol — the point of the pin is that the serving plane cannot
    # create, move or cancel the process-global timer, and recording a tick's death
    # does none of those.
    assert "beat" in touched, f"the serving plane never reports progress: {touched}"
    assert touched <= {
        "beat",
        "SERVING",
        "note_tick_death",
    }, f"the serving plane reaches the watchdog for {touched}"

    # ``engage`` IS NOT A SECOND ARM SITE, and it is pinned here beside ``arm`` because it
    # moves the bound and stamps planes: a call added to a library path (or to the
    # entry-point guard, where it would run before the runtime has anything to judge)
    # would be the same class of defect that pin exists for. It belongs on the boot path
    # at the publication boundary — the function that publishes ``_live_handle``, the line
    # the code itself documents as "there is a session to judge".
    engage_sites = [
        str(path)
        for path in sorted((REPO / "local_operator").rglob("*.py"))
        if "stall_watchdog.engage(" in path.read_text(encoding="utf-8")
    ]
    assert [Path(site).name for site in engage_sites] == ["process.py"], (
        f"engagement is reachable from a library path, so a host that never armed the "
        f"timer could move its bound: {engage_sites}"
    )
    assert process_source.count("stall_watchdog.engage(") == 1, (
        "the boot path engages more than once, so a later call could move a bound a beat "
        "has already set"
    )
    assert (
        "stall_watchdog.engage(" not in guarded
    ), "engagement is reachable from the entry point, before a session exists"
    publishers = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "stall_watchdog.engage(" in ast.unparse(node)
        and "_live_handle = handle" in ast.unparse(node)
    ]
    assert publishers == ["amain"], (
        f"engagement has moved off the publication boundary — the window it must cover "
        f"starts where the process publishes the handle the probe judges: {publishers}"
    )


@pytest.mark.slow
def test_a_real_runtime_child_arms_its_bound_and_disarms_on_a_clean_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production wiring, on a process spawned the way a viewer engages one.

    The unit tests above prove the MECHANISM; this proves the ARRIVAL — that the
    ``__main__`` branch is genuinely reached by ``launch._spawn_runtime``'s
    ``-m`` argv (nothing else would tell a reader the installer was wired to
    anything), that the file it names is the one an operator would open, that a
    healthy runtime's header-only file is NOT reported as a fired bound, and that
    a clean stop REMOVES it — so a file that survives means the bound fired.

    READINESS IS GATED, and the gate is not decoration: a SIGTERM that arrives
    before ``amain`` installs its handler block kills the process by default
    disposition, which takes it out through a path that never reaches ``main``'s
    disarm and leaves a header-only file behind. Measured while writing this
    test: the record appears earlier than the handler block (boot instrumentation
    publishes one), so waiting for the record is not readiness. The probe is the
    suite's existing one — a SIGUSR1 dump, armed in the same block, immediately
    after the SIGTERM handler (`test_runtime_detachment` explains the ordering) —
    and SIGUSR1 is set to SIG_IGN in THIS process first so a probe that lands
    early is discarded rather than fatal.

    The spawn harness — a real child, a real config root, reading the child's own
    log — is ``test_runtime_detachment``'s, imported rather than copied: it
    exists for exactly this shape of claim, and a second copy of it would drift
    from the isolation rules it encodes (every inherited ``LOP_*``/``CMUX_*``
    stripped, a scratch root, and the child's group reaped by exact pid).
    """
    import signal as signal_module

    from local_operator.session.runtime import launch as launch_module
    from tests.unit.session.runtime.test_runtime_detachment import (
        _SESSION_ID,
        _await_log,
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

    # THE CHILD MUST BE THE CODE UNDER TEST, and leaving that to the spawn would
    # not give it: `launch._spawn_interpreter()` deliberately returns the CURRENT
    # INSTALL GENERATION's interpreter so a mixed-generation fleet converges, and
    # on a machine with a global `lop` install that is a DIFFERENT BUILD with no
    # `stall_watchdog` in it at all. Measured here: the operator's environment
    # resolves to `~/.local/share/lop/generations/<stamp>/…/bin/python3`, while an
    # isolated HOME (which is what the suite gives every test) falls through to
    # `sys.executable`. A test that let that pointer choose would be asserting
    # about whatever build the host happens to have installed — and could pass
    # while exercising none of this diff. Pinned to this process's interpreter,
    # and the assertion it exists for (the armed dump file) is what proves the
    # child really ran this tree.
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
            "the runtime child is running but never armed its stall bound, so a wedged "
            f"session is as unbounded as before:\n{_capture_text(child)}\n"
            f"{_log_text(config_dir)[-1000:]}"
        )
        text = _await_log(config_dir, "stall watchdog armed", child)
        assert "stall watchdog armed" in text, text[-1000:]
        assert str(dump) in text, "the runtime's own log does not name the dump a reader must open"

        # A HEALTHY RUNTIME'S FILE IS HEADER-ONLY, AND IS NOT EVIDENCE. The header
        # is written at arm time (it has to be), so it is the marker — not the
        # file's existence — that separates "armed and working" from "the bound
        # fired", and this asserts both halves on a real runtime.
        assert stall_watchdog.FIRED_MARKER not in dump.read_text(
            encoding="utf-8"
        ), "a healthy runtime reported a fired bound"
        assert stall_watchdog.fired_pids(config_dir / "logs") == set()

        # HANDLER READINESS, by the child's own hand: the SIGUSR1 task dump is
        # armed in the block directly after the SIGTERM handler, so seeing it is
        # proof the clean-stop path under test exists in the child.
        deadline = time.monotonic() + 60.0
        while "state: streaming=" not in _log_text(config_dir):
            assert child.poll() is None, (
                f"the runtime exited (rc={child.returncode}) before it armed its handlers:\n"
                f"{_capture_text(child)}\n{_log_text(config_dir)[-1000:]}"
            )
            assert time.monotonic() < deadline, (
                "the runtime never armed its signal handlers; the readiness probe is stale:\n"
                f"{_log_text(config_dir)[-1000:]}"
            )
            os.kill(pid, signal_module.SIGUSR1)
            time.sleep(0.2)

        # A CLEAN STOP RECORDS ITSELF AND LEAVES THE FILE. Nothing is in flight,
        # so the runtime leaves at once and ``main``'s finally disarm writes the
        # clean-exit line over the header — the file stays, because existence was
        # never what made it evidence.
        child.terminate()
        deadline = time.monotonic() + 60.0
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert child.poll() is not None, (
            f"the runtime did not leave on SIGTERM:\n{_capture_text(child)}\n"
            f"{_log_text(config_dir)[-1000:]}"
        )
        assert "session runtime: exiting" in _log_text(config_dir), (
            "the runtime did not take its own exit path, so this stopped being a clean stop:\n"
            f"{_log_text(config_dir)[-1000:]}"
        )
        assert (
            not dump.exists()
        ), "a clean exit left the dump behind, so a file no longer means the bound fired"
        assert stall_watchdog.fired_pids(config_dir / "logs") == set()
    finally:
        signal_module.signal(signal_module.SIGUSR1, previous_usr1)
        if child is not None:
            _reap(child, config_dir)


@pytest.mark.slow
def test_a_real_runtime_child_engages_at_publication_and_moves_its_own_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ENGAGEMENT ARRIVES, on a process spawned the way a viewer engages one.

    The cells above prove the mechanism; this proves the ARRIVAL, and it is the only
    evidence that the line is on the boot path at all: a runtime booted through
    ``launch._spawn_runtime`` must show BOTH halves in its own artifacts — a dump header
    that names the BOOT bound (the entry point armed it), and a deadline sibling counting
    down from the STEADY bound (the publication boundary moved it and stamped both
    planes).

    WHY THE SIBLING'S VALUE IS THE WHOLE PROOF. The sibling carries one number: the
    epoch deadline the timer really holds, rewritten by every re-arm. Before this change
    no beat reaches the timer during boot, so a sibling written by the first serving beat
    would count down from the ARM — about 900 s on the shipped defaults. Only an
    engagement at publication makes it \u2248300 s within seconds of the spawn, and only a
    stamp of BOTH planes makes the pin the engagement instant rather than the arm
    instant. So one assertion covers the reset, the bound move and the two-plane stamp at
    once, on a real child, with the number printed on failure.

    MUTATION THIS CELL CATCHES: delete the ``stall_watchdog.engage()`` call from
    ``process.py`` — the sibling then reads \u2248900 s (or, before the first beat, is
    absent), and this cell goes red while every mechanism cell above stays green.
    """
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
    # Same reason as the sibling cell above: the spawn would otherwise hand the child
    # whatever install generation the host has, which is not this tree.
    monkeypatch.setattr(launch_module, "_spawn_interpreter", lambda: sys.executable)

    child = None
    try:
        child = launch_module._spawn_runtime(
            _SESSION_ID,
            str(config_dir),
            defer_materialise=False,
        )
        pid = child.pid
        _wait_for_record(config_dir)

        logs = config_dir / "logs"
        dump = logs / f"{stall_watchdog.DUMP_PREFIX}-{pid}.log"
        deadline = time.monotonic() + 60.0
        while not dump.is_file() and time.monotonic() < deadline:
            assert child.poll() is None, (
                f"the runtime exited (rc={child.returncode}) without arming:\n"
                f"{_capture_text(child)}\n{_log_text(config_dir)[-1000:]}"
            )
            time.sleep(0.05)
        assert dump.is_file(), "the runtime child never armed its stall bound"

        header = dump.read_text(encoding="utf-8")
        assert f"armed for {stall_watchdog.DEFAULT_BOOT_STALL_S:g}s" in header, (
            "the entry point did not arm the BOOT bound, so the boot stretch is still "
            f"covered by the steady one: {header[:600]!r}"
        )
        assert (
            "THIS IS THE BOOT BOUND" in header
        ), f"the header does not name the boot phase: {header[:600]!r}"

        sibling = stall_watchdog.deadline_path(pid, logs)
        deadline = time.monotonic() + 60.0
        while not sibling.is_file() and time.monotonic() < deadline:
            assert child.poll() is None, (
                f"the runtime exited (rc={child.returncode}) before anything re-armed its "
                f"timer:\n{_capture_text(child)}\n{_log_text(config_dir)[-1000:]}"
            )
            time.sleep(0.05)
        assert sibling.is_file(), (
            "nothing re-armed the timer after the child armed it, so the runtime never "
            "engaged (or an engagement that moved the bound did not write the deadline "
            f"sibling):\n{_capture_text(child)}\n{_log_text(config_dir)[-1000:]}"
        )
        epoch, leg = _deadline_record(logs, pid)
        assert leg in (stall_watchdog.WORKLOAD, stall_watchdog.SERVING), leg
        remaining = epoch - time.time()
        assert remaining <= stall_watchdog.DEFAULT_STALL_S + 1.0, (
            f"the child's deadline sits {remaining:.1f}s out, which is the bound it armed "
            f"with at the entry point rather than the steady bound an engagement moves it "
            f"to -- the boot clock was never reset"
        )
        assert remaining > 60.0, (
            f"the child's deadline sits only {remaining:.1f}s out, which is not the steady "
            f"bound measured from a publication boundary seconds ago"
        )
    finally:
        if child is not None:
            _reap(child, config_dir)


# -- the progress leg: spinning without advancing ----------------------------
#
# THE SHAPE THESE CELLS ARE FOR, measured on this fleet on 2026-09-21. Session
# ``14066af01c7a`` (build 0.61.16, i.e. WITH the liveness bound armed) launched a
# four-way subagent batch 16 s after a sibling settled, acknowledged the launch,
# and then produced nothing: no transcript row, no roster movement and no change
# in its four subagent counters across 75 s, +14.3 s of process CPU in that
# window, and a process tree with no build and no command in it. Its serving
# plane's last beat was 18:34:58; the 300 s liveness deadline was 18:39:58; the
# operator's reap landed at 18:39:46, twelve seconds short of it. Both
# instruments were right — the loops were alive and ticking, the work was not
# advancing — and nothing in the design could see the difference.
#
# So these two children are the same runtime, the same real planes and the same
# real probe, differing only in what the workload loop does. The SPIN child is
# the incident; the AWAIT child is the false positive the leg must not take, and
# it is half the evidence rather than a courtesy — a predicate that fires on a
# long wait is the bug this bound exists to prevent wearing the other hat.

_SPINNING_CHILD = r"""
import asyncio
import os
import pathlib
import sys
import time

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

    session = make_session(root, _stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"

    # THE PRODUCTION SEAM, not a double: the handle the entry point publishes and
    # the probe the entry point arms with.
    process._live_handle = handle
    assert stall_watchdog.arm(seconds=float(sys.argv[1]), probe=process._progress_probe)
    print(f"armed:{os.getpid()}", flush=True)

    stop = asyncio.Event()
    asyncio.create_task(process._beat_stall_watchdog(stop))
    # BOTH PLANES HAVE RUN: the workload ticker stamped its plane and the serving
    # plane published a record. This is the state that used to be
    # indistinguishable from health, so the cell refuses to say anything without it.
    await asyncio.sleep(1.0)
    print("both-planes-ran", flush=True)

    # THE INCIDENT: yielding on every pass, so ticker and serving keep their cadence
    # while burning CPU and advancing no work. Bound the child so it can exit after
    # the sampler records its diagnostic fire; the C timer cannot terminate it.
    bound = float(sys.argv[1])
    end = time.monotonic() + max(10.0, bound * 5)
    while time.monotonic() < end:
        await asyncio.sleep(0)
        sum(range(200_000))
    print("spin-finished", flush=True)
    os._exit(0)


asyncio.run(main())
"""

#: The same boot, then a GENUINE long wait: the loop is alive and idle for four
#: bounds, and no leg may fire. Deliberately NOT killed by the test — the child
#: exits on its own and prints how long it waited, so "it survived" is a fact the
#: child reports rather than one the parent infers from a missing dump.
_AWAITING_CHILD = r"""
import asyncio
import os
import pathlib
import sys
import time

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

    session = make_session(root, _stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published()

    process._live_handle = handle
    bound = float(sys.argv[1])
    assert stall_watchdog.arm(seconds=bound, probe=process._progress_probe)
    print(f"armed:{os.getpid()}", flush=True)

    stop = asyncio.Event()
    asyncio.create_task(process._beat_stall_watchdog(stop))
    await asyncio.sleep(0.5)

    started = time.monotonic()
    await asyncio.sleep(bound * 4)
    print(f"still-alive:{time.monotonic() - started:.1f}", flush=True)
    sys.stdout.flush()
    # ``os._exit`` rather than a return: the runtime's own thread is still parked
    # and must not hold the cell open. It is also what leaves the dump file
    # exactly as written, which is the evidence the parent reads — and NOT a
    # ``disarm``, so the file survives to be checked.
    os._exit(0)


asyncio.run(main())
"""


def test_the_progress_leg_fires_on_a_spinning_loop_that_never_advances(tmp_path: Path) -> None:
    """The reproduction, on the REAL plumbing: alive, ticking, and going nowhere.

    Cannot pass on the committed head: the only leg that existed there re-armed
    the timer from both planes, and both planes were healthy in this child — so
    the run produced no dump and had to be killed externally.
    """
    result = _run_script(
        _SPINNING_CHILD,
        tmp_path,
        args=(str(CHILD_BOUND_S), str(tmp_path), str(REPO)),
        timeout=120.0,
    )
    assert (
        result.returncode == 0
    ), f"the dump-only spin child did not exit on its own: {result.stdout!r} {result.stderr!r}"
    assert "both-planes-ran" in result.stdout, (
        f"the planes never ran, so this says nothing about a process whose loops were "
        f"ALIVE: {result.stdout!r} {result.stderr!r}"
    )
    assert "spin-finished" in result.stdout

    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text
    # THE FIRING PATH NAMED ITS CLASS AND ITS LEG. A bound that fires without
    # saying which predicate ended the runtime leaves the next reader unable to
    # tell a fire from a silent non-fire — the defect this class exists for.
    assert stall_watchdog.PROGRESS_MARKER in text, text
    assert incidents.STALL_BOUND_CAUSE in text, text
    assert stall_watchdog.fired_leg(pid, tmp_path / "logs") == stall_watchdog.LEG_PROGRESS
    assert "parked_child.py" in text, f"the dump does not name the spinning loop: {text}"


def test_the_progress_leg_spares_a_genuine_long_await(tmp_path: Path) -> None:
    """THE NEGATIVE, and it is half the evidence: a long wait is not a spin.

    A model call, a tool result or any other await leaves the work motionless for
    as long as it takes, and the liveness leg tolerates it by design. The
    progress leg must not narrow that: it distinguishes WAITING from SPINNING,
    and this child is the waiting one — its loop is alive and idle for four
    bounds, so a run that fired would make every slow provider call fatal.
    """
    result = _run_script(
        _AWAITING_CHILD,
        tmp_path,
        args=(str(CHILD_BOUND_S), str(tmp_path), str(REPO)),
        timeout=120.0,
    )
    assert (
        result.returncode == 0
    ), f"the bound ended a runtime that was merely WAITING: {result.stdout!r} {result.stderr!r}"
    waited = float(result.stdout.split("still-alive:", 1)[1].split()[0])
    assert waited >= CHILD_BOUND_S * 3, (
        f"the child did not actually wait past the bound ({waited}s), so it proves "
        f"nothing: {result.stdout!r}"
    )

    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    leftover = _dump_for(tmp_path, pid)
    assert leftover.is_file(), "the child never armed, so this cell is vacuous"
    text = leftover.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER not in text, text
    assert stall_watchdog.PROGRESS_MARKER not in text, text
    assert stall_watchdog.fired_leg(pid, tmp_path / "logs") is None


# -- the SAME spin, with a child lane in the process --------------------------
#
# THE MANAGER SHAPE, and the reason it needs its own child: a parent driving
# subagents IN PROCESS, its own loop burning CPU in the roster projection walk,
# while a lane holds a step. Nothing in that state moves a plane's stamp and
# nothing closes a STEP — ``_subagent_roster_generation`` bumps on a completed
# assistant message, never on a step being open — so all three progress legs held
# and the process was cut while it was working (``stall_watchdog`` names the shape
# and the O(N^2) walk behind the CPU that makes it reachable).
#
# The child below is ``_SPINNING_CHILD`` plus exactly ONE real lane on the real
# registry, and ``MODE`` is the only thing that varies:
#
#   open_step    the lane's live tail is an assistant message whose tool calls
#                have no answers — a lane parked in a long in-process tool. No
#                counter can express this; it is what the widening adds.
#   parked_call  the lane's OWN forked stream is parked inside ``_record_stream``:
#                a lane waiting on its provider. The shared counter already covers
#                this half, so this arm is a regression guard for it rather than
#                new coverage.
#   unreadable   the lane's context raises, so the per-lane read cannot answer.
#                Fail closed: the process must survive.
#
# The probe's own answer is PRINTED while the lane is in the state under test
# rather than inferred from a missing dump: "no file" says the leg did not fire,
# and the printed tuple is what says the answer it did not fire on was the widened
# one. Each arm gets its own config dir, so each arm has its own dump file.
_LANE_PROBE_CHILD = r"""
import asyncio
import os
import pathlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, sys.argv[4])

from local_operator.harness.types import ChatRequest, Message, ModelSpec, ToolCall
from local_operator.model.configure import SessionStreamFn
from local_operator.session.runtime import process, server, stall_watchdog
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import make_session

MODE = sys.argv[3]


class _UnreadableContext:
    # A lane whose own step cannot be read: the fail-closed branch's input.

    @property
    def messages(self):
        raise RuntimeError("the lane's context cannot be read")


def _completed_tail(lane) -> None:
    # A lane whose last step landed: what a lane parked in a model call looks like.
    lane._context.messages.extend(
        [
            Message.user("do the thing"),
            Message.assistant(tool_calls=[ToolCall(id="done-1", name="bash", arguments={})]),
            Message(role="tool", content=[], tool_call_id="done-1", tool_name="bash"),
            Message.assistant("step done"),
        ]
    )


def _open_step(lane) -> None:
    # A lane mid-batch: the tail is an assistant message with unanswered calls.
    lane._context.messages.extend(
        [
            Message.user("run the long tool"),
            Message.assistant(tool_calls=[ToolCall(id="open-1", name="bash", arguments={})]),
        ]
    )


async def _park_a_provider_call(stream) -> None:
    # Hold a REAL forked child stream inside ``_record_stream``, forever.
    #
    # The production counter path (``SessionStreamFn.fork`` setting
    # ``_counts_as_child_request``) rather than a double: this is what an in-process
    # lane parked in a model call does, and the shared scalar is what the probe's
    # existing counter term reads.
    fork = stream.fork("lane-parked")
    fork._record_usage = MagicMock()

    async def never():
        if False:  # pragma: no cover - makes this an async generator
            yield None
        await asyncio.Event().wait()

    request = ChatRequest(
        model=ModelSpec(provider="test", model_id="m"), messages=[Message.user("go")]
    )
    # ``_record_stream`` is an ASYNC GENERATOR (the product consumes it the same
    # way), and iterating it is what enters the provider call the counter counts.
    async for _event in fork._record_stream(request, never()):  # pragma: no cover
        pass


async def main() -> None:
    root = pathlib.Path(sys.argv[2])
    process.HEARTBEAT_INTERVAL_S = 0.3
    server.HEARTBEAT_INTERVAL_S = 0.3

    # THE MANAGER'S OWN STREAM IS A REAL ``SessionStreamFn``, because a lane's
    # provider request is this process's child work only through a FORK of it.
    stream = SessionStreamFn(MagicMock(), {}, "lane-probe-child")
    stream._record_usage = MagicMock()
    session = make_session(root, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published()

    # A REAL lane on the REAL registry: built by the product's own constructor and
    # attached through the product's own ``record_launch`` / ``attach``.
    lane = make_session(root / "lane", stream)
    _completed_tail(lane)
    comms = session.subagent_comms
    comms.record_launch("job-lane", "lane")
    comms.attach("job-lane", lane, root / "lane")

    # THE STATE IS ESTABLISHED BEFORE THE ARM, so nothing this child does after it
    # can move the work footprint: this cell is about a static process whose only
    # moving part is the CPU its own loop burns.
    if MODE == "open_step":
        _open_step(lane)
    elif MODE == "unreadable":
        lane._context = _UnreadableContext()
    elif MODE == "unreadable_shape":
        # A record whose ``child`` is not a lane this read can vouch for: the flag
        # reads a real False and there is no tail to scan. The direction has to be
        # "hold" — never "judge it idle" — for a shape a future lane class or a
        # double can present (agent review round 1, MINOR 3).
        comms._records["job-lane"].child = SimpleNamespace(_compacting=False)
    elif MODE == "parked_call":
        asyncio.create_task(_park_a_provider_call(stream))
        await asyncio.sleep(0.5)
        assert stream.child_model_requests_in_flight, "the fork never entered its provider call"
    else:
        raise SystemExit(f"unknown MODE {MODE}")

    bound = float(sys.argv[1])
    process._live_handle = handle
    assert stall_watchdog.arm(seconds=bound, probe=process._progress_probe)
    print(f"armed:{os.getpid()}", flush=True)

    stop = asyncio.Event()
    asyncio.create_task(process._beat_stall_watchdog(stop))
    # THE LEG'S OWN ANSWER, taken while the lane holds the state under test.
    print(f"probe:{process._progress_probe()}", flush=True)

    # THE INCIDENT'S LOOP: yielding every pass, so both planes keep their cadence
    # and keep re-arming the timer, while burning CPU and advancing no work.
    started = time.monotonic()
    while time.monotonic() - started < bound * 4:
        await asyncio.sleep(0)
        sum(range(200_000))
    print(f"still-alive:{time.monotonic() - started:.1f}", flush=True)
    os._exit(0)


asyncio.run(main())
"""


def _run_lane_child(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """One lane arm: its own config dir, so its own dump file and its own logs."""
    root.mkdir(parents=True, exist_ok=True)
    return _run_script(
        _LANE_PROBE_CHILD,
        root,
        args=(str(CHILD_BOUND_S), str(root), mode, str(REPO)),
        timeout=120.0,
    )


def _probe_answer(stdout: str) -> tuple[object, bool] | None:
    """The probe tuple the child printed, PARSED rather than substring-matched.

    ``_assert_spared`` asked whether the text ``", True)"`` appeared anywhere in the
    child's stdout (agent review round 1, NIT 2). That is a match on a tuple
    ``repr``: it would start passing for the wrong reason the day ``_work_motion``'s
    own shape carries a ``True``, and it says nothing about WHICH element answered.
    The second element of ``(motion, in_flight)`` is what every arm here is about,
    so the second element is what is asserted — parsed out of the line the child
    prints for exactly this purpose.
    """
    for line in stdout.splitlines():
        if line.startswith("probe:"):
            answer = ast.literal_eval(line[len("probe:") :])
            assert isinstance(answer, tuple) and len(answer) == 2, (
                f"the child printed a probe answer that is not the (motion, in_flight) "
                f"pair, so this arm cannot read it: {line!r}"
            )
            return answer
    return None


def _assert_spared(result: subprocess.CompletedProcess[str], root: Path, mode: str) -> None:
    """The shared half of the arms: the leg answered, and nothing fired."""
    assert result.returncode == 0, (
        f"the progress leg cut a manager whose lane was in '{mode}': "
        f"{result.stdout!r} {result.stderr!r}"
    )
    answer = _probe_answer(result.stdout)
    assert answer is not None, (
        f"the '{mode}' child never printed the probe's own answer, so this arm would pass "
        f"for a different reason than the one it is named for: {result.stdout!r}"
    )
    assert answer[1] is True, (
        f"the probe did not answer 'in flight' for '{mode}', so this arm would pass for "
        f"a different reason than the one it is named for: {result.stdout!r}"
    )
    waited = float(result.stdout.split("still-alive:", 1)[1].split()[0])
    assert waited >= CHILD_BOUND_S * 3, (
        f"the '{mode}' child did not outlast the bound ({waited}s), so it proves nothing: "
        f"{result.stdout!r}"
    )
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(root, pid)
    assert dump.is_file(), f"the '{mode}' child never armed, so this cell is vacuous"
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER not in text, text
    assert stall_watchdog.PROGRESS_MARKER not in text, text
    assert stall_watchdog.fired_leg(pid, root / "logs") is None


def test_the_progress_leg_spares_a_manager_whose_lane_holds_a_step(tmp_path: Path) -> None:
    """THE DISCRIMINATING PAIR: the lane's own open step is the only difference.

    Same child, same real runtime, same real probe, same CPU-burning loop and the
    same static footprint; the WITH-LANE run has one lane mid-batch and the
    WITHOUT-LANE run is ``_SPINNING_CHILD``. The first must survive (its lane holds
    an open step) and the second must still be cut and dumped, which is what keeps
    the widening from being "stop firing at all".

    Cannot pass on the committed head: the lane's open batch is not in flight by
    any measure that build had, so the with-lane child fires exactly like the
    control and this cell reads rc 1.
    """
    with_lane = _run_lane_child(tmp_path / "with-lane", "open_step")
    _assert_spared(with_lane, tmp_path / "with-lane", "open_step")

    control_root = tmp_path / "without-lane"
    control_root.mkdir(parents=True, exist_ok=True)
    control = _run_script(
        _SPINNING_CHILD,
        control_root,
        args=(str(CHILD_BOUND_S), str(control_root), str(REPO)),
        timeout=120.0,
    )
    assert control.returncode == 1, (
        f"the control (no lane at all) was not cut, so this cell cannot tell the "
        f"widening from a leg that stopped working: {control.stdout!r} {control.stderr!r}"
    )
    pid = int(control.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(control_root, pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text
    assert stall_watchdog.PROGRESS_MARKER in text, text
    assert stall_watchdog.fired_leg(pid, control_root / "logs") == (stall_watchdog.LEG_PROGRESS)


def test_the_progress_leg_spares_a_manager_whose_lane_is_parked_in_a_provider_call(
    tmp_path: Path,
) -> None:
    """The model-call half of the lane, which the shared forked-stream counter covers.

    Half of the required pair rather than a courtesy: a manager whose lanes are all
    parked in long model calls is the shape the incident was measured in, and a
    widening that spared its sibling arm (a lane holding a tool batch) while
    letting this one be cut would have moved the false positive rather than closed
    it.
    """
    result = _run_lane_child(tmp_path / "parked-call", "parked_call")
    _assert_spared(result, tmp_path / "parked-call", "parked_call")


def test_an_unreadable_lane_leaves_the_runtime_alive(tmp_path: Path) -> None:
    """FAIL CLOSED: a lane whose step cannot be read preserves the process.

    The direction is the opposite of this session's own tail (``process``
    documents why), and it is the direction the widened read was asked for: the
    lane is known to be there, so "I could not read it" must not be spent as "no
    lane holds work". A run that fired here would make a corrupt registry fatal.
    """
    result = _run_lane_child(tmp_path / "unreadable", "unreadable")
    _assert_spared(result, tmp_path / "unreadable", "unreadable")


def test_a_lane_of_an_unrecognised_shape_holds_the_runtime(tmp_path: Path) -> None:
    """FAIL CLOSED on a child this read cannot vouch for (agent review round 1, MINOR 3).

    The gate is an ``isinstance`` against the real ``Session``, so a
    ``record.child`` that is not a ``Session`` — a double, a lane class built on
    another base — is HELD and the hold is announced. The ``_compacting`` read
    behind it is still plain truthiness, and a ``Session`` SUBCLASS passes the gate
    and is judged by its tail instead: this arm is about the non-``Session`` shape.
    Plain truthiness was the defect: a
    ``MagicMock``'s attribute is truthy for the life of the process, which answered
    "in flight" forever with nothing recording that the read never worked, and the
    mirror case (an attribute reading a real ``False`` on a shape with no tail to
    scan) was spent as "not in flight". ``_assert_spared`` carries the rest: alive
    past four bounds, no fired marker and no progress line in the dump.
    """
    result = _run_lane_child(tmp_path / "unreadable-shape", "unreadable_shape")
    _assert_spared(result, tmp_path / "unreadable-shape", "unreadable_shape")


#: A clock the progress leg can be driven through: the predicate is about the
#: RELATION between three readings, so the readings are what a test has to
#: control. Waiting out a real window would make every cell here a bet on host
#: load, and this host is routinely at a load average of 50-100.
class _FakeClock:
    def __init__(self) -> None:
        self.wall = 1_000.0
        self.cpu = 5.0

    def monotonic(self) -> float:
        return self.wall

    def process_time(self) -> float:
        return self.cpu

    def time(self) -> float:
        return 1_700_000_000.0

    def localtime(self, _timestamp: float) -> str:
        # ``_note_quiet_plane`` formats the instant a plane last reported, and it is
        # reached from ``beat`` — which the cells in the executing-loop section drive
        # directly, so this pair has to exist for them. A fixed string is enough: the
        # assertion those cells make is about the DEADLINE, and what this renders is a
        # human-readable echo of it.
        return "2026-01-01 00:00:00"

    def strftime(self, _fmt: str, _stamp: object = None) -> str:
        return "2026-01-01 00:00:00"


def test_the_progress_leg_needs_all_three_facts_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPINNING IS NOT WAITING, and it is not YIELDING EITHER.

    The predicate is composite because each leg alone is a false positive this
    fleet has already measured. CPU alone fires on every legitimate in-process
    tool (an in-process render or scan burns a core with no transcript movement);
    no-motion alone fires on every long model call, which is the bound this
    module was designed NOT to be; and no-motion-plus-idle misses the incident
    entirely, because the incident burned 19% of a core.

    The three are driven one at a time here, on the real ``_sample`` with a
    controlled clock, because "all three" is the claim and a test that only ever
    shows the firing case cannot state it.
    """
    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    state: dict[str, Any] = {"motion": "still", "in_flight": False, "cpu_per_step": 1.0}

    def probe() -> tuple[object, bool]:
        return state["motion"], bool(state["in_flight"])

    dump = tmp_path / f"{stall_watchdog.DUMP_PREFIX}-4242.log"
    handle = dump.open("w", encoding="utf-8")
    handle.write(f"{stall_watchdog.ARM_MARKER}test header\n")
    handle.flush()
    # Built directly rather than through ``arm``: ``arm`` starts the sampler
    # thread, which would sample this fake clock between the lines below. The
    # predicate is what is under test here; the thread has its own cell.
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4242, probe)

    def step() -> bool:
        fake.wall += 1.0
        fake.cpu += float(state["cpu_per_step"])
        return stall_watchdog._sample(armed)

    # LEG 2 ALONE: a tool batch is executing, so this is work, whatever the CPU says.
    state["in_flight"] = True
    assert [step() for _ in range(8)] == [False] * 8
    assert armed.progress_deadline is None, "a running tool batch did not hold the leg"

    # LEG 1 ALONE: the work is advancing, one step at a time.
    state["in_flight"] = False
    for index in range(8):
        state["motion"] = f"moved-{index}"
        assert step() is False
    assert armed.progress_deadline is None, "a session making progress did not hold the leg"

    # LEG 3 ALONE: the process is WAITING — a model call, a socket, a child
    # process — which burns no CPU in this process at all.
    state["cpu_per_step"] = 0.0
    assert [step() for _ in range(8)] == [False] * 8
    assert armed.progress_deadline is None, "a waiting runtime was read as spinning"

    # ALL THREE: no fire before the retained samples span the window, and a fire
    # once they do.
    state["cpu_per_step"] = 1.0
    # The sample that sees the work STOP is not judged as the first of a run: the
    # run starts there, and a window needs a window's worth of samples.
    state["motion"] = "settled"
    assert step() is False, "a sample that saw movement started a run"
    assert (
        armed.clock.mean_rate(armed.seconds) is None
    ), "a run younger than the window was judged as if it spanned one"
    fired_at = None
    for index in range(1, int(armed.seconds) + 6):
        if step():
            fired_at = index
            break
    assert fired_at is not None, "the progress leg never fired on a sustained spin"
    assert (
        fired_at >= armed.seconds
    ), f"the progress leg fired at sample {fired_at} of a {armed.seconds:g}s window"
    # ...AND THE RETAINED SAMPLES ARE BOUNDED BY THE WINDOW THEY MEASURE, not by a
    # count: this driver looks once a second, so a 4 s window holds five or six.
    assert len(armed.clock.history) <= int(armed.seconds) + 2, len(armed.clock.history)

    # THE STATISTIC, ON THE SCHEDULES BOTH REVIEW ROUNDS MEASURED. Fresh state per
    # case because each is a claim about what ONE run does, and a real window
    # (45 samples of a second against the 45 s window) because the burst cases are
    # about how long a burn stays inside it.
    def fires(cpu_steps: list[float]) -> int | None:
        state["motion"] = "settled"
        state["in_flight"] = False
        spare = stall_watchdog._Armed(dump, handle, 45.0, 4244, probe)
        for index, per_step in enumerate(cpu_steps, start=1):
            fake.wall += 1.0
            fake.cpu += per_step
            if stall_watchdog._sample(spare):
                return index
        return None

    assert fires([0.0] * 60) is None, "an idle runtime was read as spinning"
    # ROUND 1'S SHAPE: one scheduled-out sample no longer discards the run.
    assert fires([1.0] * 20 + [0.0] + [1.0] * 40) is not None
    # ROUND 2'S SHAPE, both directions. A burn buys a run only while it is INSIDE
    # the window: a 4 s burst in a 45 s window fires at sample 46, and a 2 s one
    # never fires however long the run continues. The mean AT that fire is 0.0667,
    # not the 0.089 the burst length over the window suggests: the retained span
    # opens AT the burst's first second, so the CPU counted inside the window is
    # the 3 s after it, over 45 s.
    assert fires([1.0] * 4 + [0.0] * 60) == 46
    assert fires([1.0] * 2 + [0.0] * 60) is None
    # ...and the alternation that never fired at the previous head, because the
    # run kept re-opening on the burn half: a burn/zero 2-cycle is a 50% duty
    # cycle and is judged as one.
    assert fires([1.0, 0.0] * 40) is not None
    # ...and the latency follows the BURN rather than the session's age: after 200
    # idle samples, a spin fires within a window's worth of burn and not after
    # 0.05 x 200 = 10 samples of it, which is what a cumulative mean would need.
    idle_then_spin = fires([0.0] * 200 + [1.0] * 60)
    assert idle_then_spin is not None and idle_then_spin <= 208, idle_then_spin
    # THE FIRE REUSES THE LIVENESS LEG'S EXIT — the same C timer, armed to expire
    # now. The wrapper preserves the diagnostic deadline but never authorizes a
    # native process exit, regardless of the caller's sampled idle state.
    assert spy.armed, "the progress fire never reached the C timer"
    assert spy.armed[-1][1] is False, "a progress fire must remain diagnostic-only"
    assert spy.armed[-1][0] <= stall_watchdog.MIN_REARM_S
    handle.close()
    text = dump.read_text(encoding="utf-8")
    assert stall_watchdog.PROGRESS_MARKER in text
    assert incidents.STALL_BOUND_CAUSE in text


def test_arming_with_a_probe_starts_the_sampler_and_disarming_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg is inert without a probe, and it is a THREAD that has to be stopped.

    ``arm`` is reachable from the child's entry point only (see the interlock
    cells below), and the progress leg is opt-in per call site: an in-process host
    or a test of the liveness leg alone passes no probe and must get exactly the
    bound it had before. A sampler left running after ``disarm`` would be a thread
    judging a runtime that has already gone.
    """
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    assert stall_watchdog.arm(seconds=SHORT_BOUND_S, directory=tmp_path, pid=4242)
    armed = stall_watchdog._ARMED
    assert armed is not None
    assert armed.thread is None, "a probe-less arm must not start a sampler"
    stall_watchdog.disarm()

    assert stall_watchdog.arm(
        seconds=SHORT_BOUND_S,
        directory=tmp_path,
        pid=4243,
        # A probe that can never fire: this cell is about the thread's life, and
        # a sampler that decided to fire mid-assertion would be testing two
        # things at once.
        probe=lambda: ("still", True),
    )
    armed = stall_watchdog._ARMED
    assert armed is not None
    assert armed.thread is not None and armed.thread.is_alive()
    stop = armed.stop
    assert stop is not None and not stop.is_set()
    stall_watchdog.disarm()
    assert stop.is_set(), "the sampler was left running past a clean exit"


#: The cadence cell's child: a sampler whose PROBE CALLS ARE TIMESTAMPED, so the
#: interval it actually looked at can be read off the run instead of inferred from a
#: private local. The probe is a rig clock rather than a production one, and that is
#: the one place in this file where that is the right trade: the cadence is derived
#: from the BOUND, never from what the probe answers, and the bound is what moves. It
#: returns ``((), True)`` — no motion, work in flight — so the progress leg can never
#: fire and the sampler does nothing but wake, which is the whole measurement.
#:
#: ``busy`` answers True for the same reason: the steady bound here is short enough
#: that the timer WILL fire inside the window, and a surviving fire must be recorded
#: before the child can report. Work-state sampling is not what this cell measures;
#: holding it leaves the cadence untouched — a held fire re-arms the TIMER, not the
#: interval the sampler derives from ``armed.seconds``.
_CADENCE_CHILD = """
import sys
import time

from local_operator.session.runtime import stall_watchdog

boot = float(sys.argv[1])
steady = float(sys.argv[2])
engage_after = float(sys.argv[3])
park = float(sys.argv[4])

calls = []


def probe():
    calls.append(time.monotonic())
    return (), True


assert stall_watchdog.arm(
    probe=probe, busy=lambda: True, boot_seconds=boot, seconds=steady
), "could not arm"
started = time.monotonic()
time.sleep(engage_after)
moved = stall_watchdog.engage()
engaged_at = time.monotonic()
time.sleep(park)
stall_watchdog.disarm()
after = [call - started for call in calls if call > engaged_at]
gaps = [second - first for first, second in zip(after, after[1:])]
print(
    f"engage:{moved} bound:{steady:g} calls_after:{len(after)} "
    f"max_gap:{max(gaps) if gaps else -1:.3f}",
    flush=True,
)
"""


def test_the_sampler_cadence_follows_an_engagement_instead_of_the_armed_bound(
    tmp_path: Path,
) -> None:
    """Agent review round 1, R1-3 / QA Q-2: the window moved, so the look has to.

    ``_progress_sampler`` derives its look interval from ``armed.seconds``, and before
    this cell's fix it derived it ONCE, before the loop — while ``_sample`` reads the
    same attribute live for the window it accumulates. This PR is what makes that
    attribute movable after the arm, so the two could disagree: measured with boot 90 /
    steady 45, the probe-call gaps were ``[7.504]`` on BOTH sides of an engagement that
    moved the bound 90 → 45, where ``_sample_interval(45)`` is 3.75 s.

    THE CELL MEASURES IT THE OTHER WAY ROUND, from the bound the sampler was never
    armed for: boot 12 s (``_sample_interval`` 1 s) over a steady 1.2 s (0.1 s), with
    the engagement arriving BEFORE the first wake, so the interval captured at the arm
    can only ever be the wrong one for every sample the window contains. A sampler
    still on the arm's cadence wakes twice in the window; one that re-derives wakes
    once per steady interval, an order of magnitude more often. The assertion is a
    COUNT and a MAXIMUM GAP with wide margins rather than a gap pinned to a float,
    because what is being pinned is WHICH of two cadences is in force, not a
    scheduler's precision under fleet load.

    MUTATION THIS CELL CATCHES: move ``interval = _sample_interval(armed.seconds)``
    back out of the loop (or out of the lock block, past the ``continue``) and
    ``calls_after`` collapses from 16 to 2 and ``max_gap`` rises to the boot interval.
    """
    boot, steady = 12.0, 1.2
    park = 2.5
    result = _run_script(_CADENCE_CHILD, tmp_path, args=(str(boot), str(steady), "0.1", str(park)))
    summary = next((row for row in result.stdout.splitlines() if row.startswith("engage:")), "")
    assert summary.startswith(f"engage:True bound:{steady:g}"), (
        f"the engagement did not happen or did not move the bound: {result.stdout!r} "
        f"{result.stderr!r}"
    )
    calls_after = int(summary.split("calls_after:", 1)[1].split()[0])
    max_gap = float(summary.split("max_gap:", 1)[1])
    assert calls_after >= 6, (
        f"the sampler looked {calls_after}x in {park}s of a {steady:g}s bound, which is "
        f"the {boot:g}s BOOT bound's cadence "
        f"({stall_watchdog._sample_interval(boot):.3f}s) rather than "
        f"the live one ({stall_watchdog._sample_interval(steady):.3f}s): {result.stdout!r} "
        f"{result.stderr!r}"
    )
    assert max_gap <= 0.3, (
        f"the widest gap between samples after the engagement was {max_gap:.3f}s, the "
        f"cadence of the bound the sampler was ARMED with rather than the one it "
        f"measures: {result.stdout!r}"
    )


# -- the in-flight guard and the production wiring ---------------------------
#
# BOTH OF THESE EXIST BECAUSE AGENT REVIEW ROUND 1 MEASURED THEIR ABSENCE, and
# the absence was invisible: the real probe answers "no batch" in every rig that
# only ever spins, so stubbing `_step_in_flight` to `False` left the whole file
# green (18 passed / 1 failed, and the failure was a known flake). That is the
# worst shape a gap can have — the leg that decides whether a legitimate long
# operation is KILLED was reachable by nothing, and the two lines that make the
# predicate live in production were reachable by nothing either. A predicate that
# ships silently disabled with green tests makes the fleet look protected, which
# is the exact failure the #1363 docstring warns about.


@pytest.mark.asyncio
async def test_a_real_tool_batch_holds_the_progress_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-flight leg, on a REAL batch running through the REAL agent loop.

    A tool that blocks on an event is the state the leg exists for: the process
    is burning CPU with no transcript movement — the tool's row lands only when
    the batch returns — and cutting it would destroy work that is going fine.
    So this drives the real loop (``Session.prompt`` over a scripted stream that
    emits a tool call, with a real ``AgentTool`` whose ``execute`` parks), samples
    the REAL ``process._progress_probe`` while the tool is executing, and then
    releases it and shows the same predicate firing once nothing is in flight.

    Both halves in one cell on purpose: a cell that only showed "no fire while a
    tool runs" would pass with the whole leg stubbed out, and a cell that only
    showed the fire would pass with the guard deleted. The mutant is the test.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        started.set()
        await release.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="slow", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="working"),
                StreamToolCallDelta(index=0, id="c1", name="slow", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])
    # A CARRIER, NOT A DOUBLE, and the constraint is measured rather than
    # preferred: ``_step_in_flight`` reads exactly ``handle._session`` (that is
    # its whole contract), while a real ``ServingSessionHandle`` in this harness
    # — no running ``RuntimeServer`` to drive it — leaves the turning settle
    # parked forever: the tool RUNS and the batch holds, and then ``prompt``
    # never returns (measured here, 15 s). Parking on that would make this cell a
    # test of the handle rather than of the guard. The REAL parts the guard is
    # about — the real session, the real agent loop, the real tool, the real
    # ``_progress_probe`` — are all below; only the carrier is synthetic.
    handle = SimpleNamespace(_session=session)
    monkeypatch.setattr(process_module, "_live_handle", handle)

    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    dump = tmp_path / f"{stall_watchdog.DUMP_PREFIX}-5252.log"
    opened = dump.open("w", encoding="utf-8")
    armed = stall_watchdog._Armed(dump, opened, 4.0, 5252, process_module._progress_probe)

    def step() -> bool:
        fake.wall += 1.0
        fake.cpu += 1.0
        return stall_watchdog._sample(armed)

    turn = asyncio.create_task(session.prompt("go"))
    try:
        await asyncio.wait_for(started.wait(), timeout=30.0)
        # THE REAL STATE: the live context ends in unanswered calls, exactly as
        # the loop leaves it for the whole duration of every batch.
        assert process_module._tool_batch_in_flight(session) is True
        assert process_module._step_in_flight(handle) is True
        assert process_module._progress_probe()[1] is True
        # ...AND NOTHING FIRES, however long the CPU leg would otherwise hold.
        assert [step() for _ in range(30)] == [False] * 30, (
            "the bound fired while a REAL tool batch was executing: this is the "
            "false positive that would kill a legitimate in-process tool"
        )
        assert armed.progress_deadline is None
        assert spy.armed == [], "the progress leg armed the C timer during a live tool batch"
    finally:
        release.set()
        await asyncio.wait_for(turn, timeout=30.0)

    # THE SAME PREDICATE, ONCE THE BATCH IS DONE: the tool result is in the
    # context, nothing is executing, and the leg fires on the same spinning loop
    # the cell was already simulating.
    assert process_module._step_in_flight(handle) is False
    assert process_module._progress_probe()[1] is False
    for _ in range(int(armed.seconds) + 4):
        if step():
            break
    else:
        opened.close()
        raise AssertionError(
            "the leg never fired after the tool batch returned, so the first half "
            "of this cell proves nothing about a guard that is doing work"
        )
    assert spy.armed, "the fire never reached the C timer"
    opened.close()
    assert stall_watchdog.PROGRESS_MARKER in dump.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_compaction_in_flight_holds_the_progress_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other in-flight fact: a compaction rewrites history in-process.

    It burns CPU with no transcript movement while it runs (the compaction row
    lands when it finishes), so it is the second shape leg 2 exists for. Set
    through the same attribute the session's own compaction path sets, because
    the point is that this predicate reads THAT attribute — a private one, for
    the reason ``process._step_in_flight``'s docstring gives.
    """
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    assert process_module._step_in_flight(handle) is False
    session._compacting = True
    try:
        assert process_module._step_in_flight(handle) is True
    finally:
        session._compacting = False
    assert process_module._step_in_flight(handle) is False
    await session.dispose()


def test_the_progress_leg_is_wired_at_the_one_arm_site_and_reads_the_live_handle() -> None:
    """The two lines that make the leg LIVE, pinned so neither can vanish quietly.

    Agent review round 1's MAJOR 2: ``_progress_probe`` is wired by exactly two
    statements — the entry point passing ``probe=` and ``amain`` publishing the
    handle it reads — and dropping EITHER left every cell green while the
    predicate was inert for the whole fleet. This asserts them against the source,
    the same way ``test_the_only_arm_site_is_the_runtime_entry_point`` pins the
    arm site's location (a shape no behaviour in this file can observe, because
    an inert leg and a quiet one look identical from outside).
    """
    source = (REPO / "local_operator" / "session" / "runtime" / "process.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    arms = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "arm"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stall_watchdog"
    ]
    assert len(arms) == 1, f"expected exactly one arm call site, found {len(arms)}"
    passed = {kw.arg: kw.value for kw in arms[0].keywords}
    assert "probe" in passed, (
        "the entry point arms without a probe, so the progress leg can never run: "
        "every test would stay green and the fleet would look protected"
    )
    assert isinstance(passed["probe"], ast.Name) and passed["probe"].id == "_progress_probe"
    # THE EXIT LEG'S WIRING IS THE SAME KIND OF LINE, and it is the one R1-1 was
    # about: ``busy=`` is what decides whether a fire may END the process, and dropping
    # it turns a boot that is held today into one that is ended while it is still
    # constructing itself. The two cells above drive both answers through these very
    # callables; this pins that the entry point passes the second one at all, which no
    # behaviour in this file can observe (a child that arms the way the ENTRY POINT
    # does is the only witness, and it is the cell above).
    assert "busy" in passed, (
        "the entry point arms without a busy probe, so a fire during boot is armed "
        "fatally and a runtime still constructing itself would be ended by it"
    )
    assert isinstance(passed["busy"], ast.Name) and passed["busy"].id == "_busy_probe"

    published = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_live_handle" for target in node.targets
        )
    ]
    assert published, (
        "nothing publishes the live handle, so ``_progress_probe`` answers 'judge "
        "nothing' forever and the leg is inert in production"
    )
    assert any(
        isinstance(node.value, ast.Name) and node.value.id == "handle" for node in published
    ), "the handle published is not the runtime's own"

    # ...AND THE ENDS AGREE BY NAME: the probe must read the global the entry
    # point fills, or the two halves are wired to different names.
    probe_source = ast.get_source_segment(source, _function(tree, "_progress_probe"))
    assert (
        probe_source is not None and "_live_handle" in probe_source
    ), "``_progress_probe`` no longer reads ``_live_handle``"
    # The same for the OTHER half of the pair, where the answer is inverted on purpose:
    # no handle yet is "judge nothing" for the progress leg and "in flight" for the exit
    # leg (a runtime constructing itself is not idle in any sense the exit may act on).
    # Lose that read and the boot phase stops being held, silently, with every cell that
    # arms its own probe still green.
    busy_source = ast.get_source_segment(source, _function(tree, "_busy_probe"))
    assert (
        busy_source is not None and "_live_handle" in busy_source
    ), "``_busy_probe`` no longer reads ``_live_handle``, so boot is no longer held"
    # The behavioural twin of the source claim: arming with the real callable
    # installs THE REAL callable, which is what cell above this one drives.
    assert process_module._progress_probe.__module__ == process_module.__name__
    assert process_module._busy_probe.__module__ == process_module.__name__


def _function(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name}")


def test_arming_with_the_real_probe_installs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler is given the runtime's OWN callable, not a stand-in."""
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)
    assert stall_watchdog.arm(
        seconds=SHORT_BOUND_S, directory=tmp_path, pid=5253, probe=process_module._progress_probe
    )
    armed = stall_watchdog._ARMED
    assert armed is not None
    assert armed.probe is process_module._progress_probe
    assert armed.thread is not None and armed.thread.is_alive()
    stall_watchdog.disarm()


#: The ``sitecustomize`` that makes a REAL spawned runtime spin.
#:
#: WHY AN INJECTED THREAD AND NOT A TURN. The shape the progress leg exists for is
#: a process burning CPU with no progress and nothing in flight; producing it
#: through the product would take a model call, a tool and minutes of wall time per
#: run. ``time.process_time()`` is the WHOLE process's CPU, so a daemon thread in
#: the child puts the process in exactly that state — every loop still ticking
#: (a Python thread releases the GIL every switch interval, so the workload ticker
#: and the serving heartbeat keep their cadence), nothing in flight, and a core
#: being burned. It is injected at interpreter start rather than through the spawn
#: API because the child under test must be the real ``-m`` entry point with the
#: real argv: that is the whole point of the cell.
_SPIN_SITECUSTOMIZE = """\
import threading


def _spin() -> None:
    while True:
        sum(range(200_000))


threading.Thread(target=_spin, daemon=True, name="stall-rig-spin").start()
"""

#: The same, plus the one line MAJOR 2's negative removes: ``probe=`` is dropped
#: from the arm inside the child, so the entry point arms without a probe and the
#: predicate can never run. Nothing else differs, which is what makes this a
#: measurement of that argument rather than of the rig.
_DROP_PROBE_SITECUSTOMIZE = _SPIN_SITECUSTOMIZE + """

from local_operator.session.runtime import stall_watchdog as _sw

_arm = _sw.arm


def _arm_without_probe(*args, **kwargs):
    kwargs.pop("probe", None)
    return _arm(*args, **kwargs)


_sw.arm = _arm_without_probe
"""


def _write_sitecustomize(root: Path, body: str) -> Path:
    """A ``PYTHONPATH`` directory holding one ``sitecustomize.py``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sitecustomize.py").write_text(body, encoding="utf-8")
    return root


@pytest.mark.slow
def test_a_real_runtime_child_fires_the_progress_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ACCEPTANCE EVIDENCE: the leg is LIVE in production, on a spawned child.

    Agent review round 2's MAJOR: the source pin above is a complement, not a
    proof — three text-preserving mutants (the publication inside ``if False:``, an
    early ``return (), True`` before the probe reads the handle, and a second
    probe-less arm site in another module) all left it green with the prediction
    inert, and a predicate that ships silently disabled with green tests is worse
    than the gap it closes: it makes the fleet LOOK protected, which is the failure
    #1363's own docstring warns about.

    So this drives the real ``-m`` child through ``launch._spawn_runtime``, with
    the bound set the way an operator sets it, and shows the progress leg FIRING
    there — the fired marker, the progress line, ``fired_leg`` naming that leg,
    and the same runtime still alive until the test reaps it. The negative is the
    same rig with ``probe=`` dropped inside the child: same spin, same bound, and
    nothing fires. Together they are the joint
    proof of BOTH wiring lines, because with no probe and no published handle the
    probe answers "in flight" forever and no progress fire is reachable.

    A ``slow`` cell by construction: the operator-facing bound floors at 45 s
    (``MIN_BOUND_FLOOR_TICKS`` x ``HEARTBEAT_INTERVAL_S``), so the positive needs
    its bound plus a boot and the negative has to outlast it. That floor is the
    reason the bound cannot be shortened here — ``arm`` itself is unfloored, but
    the entry point reads the environment, which is the path under test.
    """
    from local_operator.session.runtime import launch as launch_module
    from tests.unit.session.runtime.test_runtime_detachment import (
        _SESSION_ID,
        _isolate,
        _log_text,
        _reap,
        _seed,
        _wait_for_record,
    )

    monkeypatch.setattr(launch_module, "_spawn_interpreter", lambda: sys.executable)
    bound_s = 45.0
    spawned: list[tuple[Any, Path]] = []

    def spawn_body(body: str, name: str) -> tuple[Any, Path, Path]:
        root = tmp_path / name
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        _seed(config_dir)
        _isolate(monkeypatch, config_dir)
        # AFTER ``_isolate``: it strips every inherited ``LOP_*`` (the child product
        # reads several of them), so a value set before it would be gone.
        monkeypatch.setenv("LOP_RUNTIME_STALL_SECONDS", str(int(bound_s)))
        monkeypatch.setenv("PYTHONPATH", str(_write_sitecustomize(root / "site", body)))
        child = launch_module._spawn_runtime(_SESSION_ID, str(config_dir), defer_materialise=False)
        spawned.append((child, config_dir))
        _wait_for_record(config_dir)
        return (
            child,
            config_dir,
            config_dir / "logs" / f"{stall_watchdog.DUMP_PREFIX}-{child.pid}.log",
        )

    try:
        # -- THE POSITIVE: the production path, firing ----------------------
        child, config_dir, dump = spawn_body(_SPIN_SITECUSTOMIZE, "keep")
        deadline = time.monotonic() + 180.0
        text = ""
        while child.poll() is None and time.monotonic() < deadline:
            try:
                text = dump.read_text(encoding="utf-8")
            except OSError:
                pass
            if stall_watchdog.FIRED_MARKER in text:
                break
            time.sleep(0.5)
        assert child.poll() is None, (
            f"the diagnostic timer terminated the runtime (rc={child.returncode}):\n"
            f"{_log_text(config_dir)[-1500:]}"
        )
        assert (
            stall_watchdog.FIRED_MARKER in text
        ), f"the native timer did not write a dump:\n{_log_text(config_dir)[-1500:]}"
        # The launched runtime remains alive until the test explicitly reaps it.
        assert child.poll() is None
        text = dump.read_text(encoding="utf-8")
        assert stall_watchdog.FIRED_MARKER in text, text[-2000:]
        assert (
            stall_watchdog.PROGRESS_MARKER in text
        ), f"the bound fired, but not the progress leg:\n{text[-2000:]}"
        assert stall_watchdog.fired_leg(child.pid, config_dir / "logs") == (
            stall_watchdog.LEG_PROGRESS
        ), text[-2000:]

        # -- THE NEGATIVE: the same rig, minus `probe=` in the child --------
        child, config_dir, dump = spawn_body(_DROP_PROBE_SITECUSTOMIZE, "drop")
        time.sleep(bound_s * 2)
        assert child.poll() is None, (
            f"the probe-less rig exited (rc={child.returncode}); the negative is not a "
            f"negative:\n{_log_text(config_dir)[-1500:]}"
        )
        assert not dump.exists() or stall_watchdog.FIRED_MARKER not in dump.read_text(
            encoding="utf-8"
        ), "the leg fired with no probe, so this cell does not measure the wiring"
        assert stall_watchdog.fired_leg(child.pid, config_dir / "logs") is None
    finally:
        for child, config_dir in spawned:
            _reap(child, config_dir)


# ============================================================================
# THE TICK ITSELF DYING: reported, recorded, and -- bounded -- undone
# ============================================================================
#
# WHY THIS SECTION EXISTS. Everything above assumes the two ticks RUN. On
# 2026-09-21 the workload tick's task RAISED, and the wiring of the day made that
# invisible three ways at once: nothing read the exception (no done-callback, no
# ``await``, no reader of ``exception()``), nothing re-created the task, and the
# dump -- the artifact an incident reader actually opens -- showed an IDLE,
# HEALTHY process, because ``faulthandler`` dumps THREADS and a dead task has no
# frame at all. So the bound ended a runtime that was working, one deadline later,
# and the file attributed it to "the runtime went silent", which was the one
# explanation that was false. The cells below pin all three halves: the death is
# logged with its exception, recorded where the dump reader will see it, and the
# tick is re-created.
#
# THE CHILD RIG IS RUN TWICE ON PURPOSE. ``bare`` is the wiring being replaced and
# ``supervised`` is the production one; the ``bare`` half is the CONTROL for the
# ``supervised`` half, because a green "it survived" cell on its own cannot
# separate a real fix from a rig that is blind to the failure. See AGENTS.md,
# "Prove the test can still fail" and the dead-instrument section under it.

#: The bound the dead-beater children run under, in seconds. Each child spans
#: FOUR of them, so the run can prove a diagnostic timer fire is independent of
#: the separately controlled child's completion.
DEAD_BEATER_BOUND_S = 2

#: A REAL runtime whose workload tick's FIRST beat raises, in both wirings.
#:
#: ``bare`` reproduces the pre-fix start site verbatim -- a bare task, unreferenced
#: by anything that observes it -- and ``supervised`` starts the production
#: supervisor. NOTHING ELSE DIFFERS between the two runs, so the pair isolates the
#: supervision itself rather than the rig's ability to make a tick die.
#:
#: The serving plane is REAL and stays healthy throughout (``RuntimeServer``'s own
#: heartbeat, on its own thread), because that is the claim being made: the process
#: that dies is not a wedged one. Both planes' stamp ages are printed every 200 ms
#: so the parent can assert the pair -- one stamp frozen past the bound, the other
#: still reporting normally -- rather than infer it from an exit code.
_DEAD_BEATER_CHILD = r"""
import asyncio
import logging
import os
import pathlib
import sys
import time

sys.path.insert(0, sys.argv[3])  # the checkout root, for the session factory

from local_operator.harness.types import StreamEndEvent
from local_operator.session.runtime import process, server, stall_watchdog
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import make_session

# THE RUNTIME'S OWN LOG, onto stdout: the WARNING the supervisor writes is half
# the evidence (the dump's tick-death line is the other half), and a child that
# left the root logger unconfigured would route it through ``logging.lastResort``
# to stderr, unformatted -- a weaker thing to assert on than the record the
# supervision actually writes.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)


def _stream(request, signal):
    async def gen():
        yield StreamEndEvent(stop_reason="stop")

    return gen()


async def main() -> None:
    root = pathlib.Path(sys.argv[2])
    bound = float(sys.argv[1])
    mode = sys.argv[4]
    # Both cadences shortened together: the child's own ticks are what the
    # supervision re-creates, and a bound of second-scale makes the freeze
    # observable in a test that has seconds to spend.
    process.HEARTBEAT_INTERVAL_S = 0.2
    server.HEARTBEAT_INTERVAL_S = 0.2

    session = make_session(root, _stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"
    assert stall_watchdog.arm(seconds=bound), "the child could not arm the bound"
    print(f"armed:{os.getpid()}", flush=True)

    # THE LEVER, and the only thing that differs from a healthy run: the workload
    # tick's FIRST beat raises. Once, not always -- a tick that cannot beat at all
    # is a different scenario (the supervisor gives up on it, which the in-process
    # cell above pins), and what this child is for is the TRANSIENT death that used
    # to be permanent because nothing re-created the task.
    real_beat = stall_watchdog.beat
    raised = {"done": False}

    def rigged_beat(plane):
        if plane == stall_watchdog.WORKLOAD and not raised["done"]:
            raised["done"] = True
            print("beater-raised", flush=True)
            raise RuntimeError("rig: the workload tick's beat raised on its first call")
        return real_beat(plane)

    stall_watchdog.beat = rigged_beat

    stop = asyncio.Event()
    if mode == "bare":
        tick = asyncio.create_task(process._beat_stall_watchdog(stop))
    else:
        tick = asyncio.ensure_future(process._watch_stall_beats(stop))
    assert tick is not None  # the handle outlives the loop below, as in amain

    started = time.monotonic()
    while True:
        await asyncio.sleep(0.2)
        armed = stall_watchdog._ARMED
        now = time.monotonic()
        if armed is None:
            print("disarmed", flush=True)
            break
        print(
            f"ages workload={now - armed.last_beat[stall_watchdog.WORKLOAD]:.2f} "
            f"serving={now - armed.last_beat[stall_watchdog.SERVING]:.2f}",
            flush=True,
        )
        if now - started > bound * 4:
            # ``os._exit`` rather than a return, for two reasons: the serving
            # plane's thread is still live and must not hold the cell open, and
            # this leaves the dump exactly as written. A ``disarm`` would remove
            # it, and the dump's own tick-death line is what the parent reads.
            print("survived", flush=True)
            sys.stdout.flush()
            os._exit(0)


asyncio.run(main())
"""


def test_a_dead_workload_ticker_is_logged_recorded_and_re_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The supervisor's whole policy, with the failure made permanent.

    A tick that dies on EVERY attempt is the case the restart budget exists for,
    and it is the one that separates this fix from an unbounded retry loop: the
    re-creations are counted to the limit, the death is recorded each time, the
    supervisor GIVES UP rather than spinning, and the plane is left unbounded so
    the bound can still fire on it. The last part is the load-bearing one -- a
    supervisor that "kept the plane alive" by suppressing the deadline would be a
    bound that no longer guards anything (see ``stall_watchdog``'s docstring).

    THE DEATHS HERE ARE ALL INSIDE ONE WINDOW, which is what makes this a storm:
    they are ~10 ms apart against ``STALL_BEAT_WINDOW_S``. The cell below is the
    other half of that property -- the same counting, with the deaths spread
    APART, must never disarm the supervision.

    The record is read back through ``tick_deaths`` rather than by grepping the
    file, because the reader is what a future incident will use: an assertion on
    raw text would pass while the fact stayed unreadable.
    """
    from local_operator.session.runtime import process

    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    # ARMED FOR REAL, so there is a real dump file at a real path -- that file is
    # the artifact under test. The timer is the fake's, so nothing can fire.
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)

    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)
    deaths = 0

    async def always_dying_tick(stop: asyncio.Event) -> None:
        nonlocal deaths
        deaths += 1
        raise RuntimeError(f"tick death {deaths}")

    # The one patch that makes the death permanent: the tick dies before its first
    # beat, every time. ``_watch_stall_beats`` looks this up by module attribute, so
    # the patch reaches the supervisor's own re-creation.
    monkeypatch.setattr(process, "_beat_stall_watchdog", always_dying_tick)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(process._watch_stall_beats(asyncio.Event()), timeout=30.0))

    # RE-CREATED TO THE BUDGET AND THEN STOPPED: one more death than re-creations,
    # which is the arming plus every retry. A supervisor that never gave up would
    # hang here instead of counting (the ``wait_for`` above is the hang's backstop).
    assert deaths == process.STALL_BEAT_RESTARTS + 1, (
        f"the supervisor created the tick {deaths} times, not "
        f"{process.STALL_BEAT_RESTARTS + 1}: the restart budget is not what the constant says"
    )
    pid = os.getpid()
    assert stall_watchdog.tick_deaths(pid, tmp_path) == (stall_watchdog.WORKLOAD,) * (
        process.STALL_BEAT_RESTARTS + 1
    ), (
        "the artifact does not name every death, so a dump reader still cannot tell a "
        "dead tick from a silent loop"
    )
    text = stall_watchdog.dump_path(pid, tmp_path).read_text(encoding="utf-8")
    assert text.index(stall_watchdog.ARM_MARKER) < text.index(stall_watchdog.TICK_DEATH_MARKER)
    assert f"RuntimeError: tick death {deaths}" in text, text
    assert (
        stall_watchdog.FIRED_MARKER not in text
    ), "this cell wrote a fired-bound dump; the record has to be readable on its own"

    # THE GIVE-UP IS IN THE RECORD, because the dump is what survives the exit and
    # a reader who opens it must learn the consequence, not just the death.
    assert "GIVES UP here" in text, text
    assert "nothing watches whether the workload plane reports" in text, text

    # THE LOG, which is the half that reaches an operator who never opens the dump.
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == process.STALL_BEAT_RESTARTS + 1, [
        record.getMessage() for record in warnings
    ]
    assert "WORKLOAD tick died" in warnings[0].getMessage()
    assert "RuntimeError: tick death 1" in warnings[0].getMessage()
    assert "re-creating it (death 1 inside" in warnings[0].getMessage()
    assert "is recorded in" in warnings[0].getMessage()
    assert "GIVES UP here" in warnings[-1].getMessage()
    assert "nothing watches whether the workload plane reports" in warnings[-1].getMessage()


def test_deaths_spread_across_the_window_never_disarm_the_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """THE BUDGET IS A RATE, NOT AN ALLOWANCE (agent review round 1, MAJOR 1).

    The defect this closes: with a counter that is only ever incremented, THREE
    unrelated transients spread over a session were enough to spend the budget, and the
    death after them -- hours later, no more related to those three than they were to
    each other -- ended the supervision for the rest of the session, so the WORKLOAD
    stamp could then freeze for good and the bound could end a healthy runtime. (An
    earlier version of this docstring said "the fifth", which is one more than the code
    ends on: the same off-by-one agent review round 3 caught one file over from the
    correction in ``process.py``.)
    That is the very incident this supervision exists to prevent, so the cell that
    forbids it is the one that runs MORE deaths than the budget allows and demands
    that supervision is still there at the end.

    The spacing is not a bet on the host: the fake tick lives for
    ``2 * STALL_BEAT_WINDOW_S`` by construction, so every death is older than the
    window by the time the next one lands, whatever the machine is doing. Nothing
    here asserts a duration -- only how many ticks were created and whether the
    supervisor gave up.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    # The window scaled down so the cell costs ~0.5 s; the RATIO is what the
    # property is about, and it is preserved exactly (the tick outlives one
    # window, then dies).
    window_s = 0.05
    monkeypatch.setattr(process, "STALL_BEAT_WINDOW_S", window_s)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)

    creations: list[float] = []
    wanted = process.STALL_BEAT_RESTARTS + 2

    async def transient_tick(stop: asyncio.Event) -> None:
        """The real tick's shape: it RUNS for a cadence, then a transient kills it."""
        creations.append(time.monotonic())
        await asyncio.sleep(window_s * 2)
        raise RuntimeError(f"transient {len(creations)}")

    monkeypatch.setattr(process, "_beat_stall_watchdog", transient_tick)

    async def scenario() -> None:
        stop = asyncio.Event()
        supervisor = asyncio.ensure_future(process._watch_stall_beats(stop))
        for _ in range(400):
            # WAIT ON THE RECORDED DEATHS, not on the creations: a creation that has not
            # died yet would be swallowed by the ``stop`` below and counted as a death
            # the supervisor never got to report.
            if (
                sum("WORKLOAD tick died" in record.getMessage() for record in caplog.records)
                >= wanted
            ):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError(
                f"only {len(creations)} ticks were created and "
                f"{[r.getMessage() for r in caplog.records]} logged"
            )
        # THE SUPERVISION IS STILL THERE, which is the whole claim: it is ended here by
        # the session ending, not by its own budget.
        assert not supervisor.done(), "the supervision gave up before the session did"
        stop.set()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(scenario(), timeout=30.0))

    messages = [record.getMessage() for record in caplog.records]
    assert len(creations) >= wanted, creations
    assert not any("GIVES UP" in message for message in messages), (
        f"{wanted} deaths spread over more than the window disarmed the supervision, so a "
        f"long session still runs out of budget: {messages}"
    )
    assert sum("WORKLOAD tick died" in message for message in messages) == wanted, messages
    assert (
        stall_watchdog.tick_deaths(os.getpid(), tmp_path) == (stall_watchdog.WORKLOAD,) * wanted
    ), "the record must name every death, in both directions"


def test_the_supervisor_survives_a_fault_in_its_own_recovery_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """NOTHING IN THE SUPERVISOR MAY END IT UNOBSERVED (round 1, MINOR 1).

    The reviewer's own lever: ``stall_watchdog.beat`` raising on EVERY workload
    call, which breaks the tick AND the one stamp the supervisor makes on the
    plane's behalf. Measured against the round-1 head, that raise escaped as an
    unretrieved task exception, the supervisor died, the stamp froze and the bound
    killed a healthy runtime -- the defect this PR fixes, re-created one level up.

    So the cell demands all three halves of the guard: the fault is LOGGED with
    its traceback, it does not stop the supervision, and the death accounting
    still runs to its decision (a guard that swallowed the cycle would leave the
    count short). The record stays readable throughout.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)

    def rigged_beat(plane: str) -> None:
        if plane == stall_watchdog.WORKLOAD:
            raise RuntimeError("rig: the workload stamp never works")

    monkeypatch.setattr(stall_watchdog, "beat", rigged_beat)
    deaths = 0

    async def always_dying_tick(stop: asyncio.Event) -> None:
        nonlocal deaths
        deaths += 1
        raise RuntimeError(f"tick death {deaths}")

    monkeypatch.setattr(process, "_beat_stall_watchdog", always_dying_tick)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(process._watch_stall_beats(asyncio.Event()), timeout=30.0))

    messages = [record.getMessage() for record in caplog.records]
    faults = [m for m in messages if "raised in its own recovery path" in m]
    assert faults, (
        "the supervisor's own stamp is still unguarded: a raise there ends the supervision "
        f"silently, which is this PR's defect one level up: {messages}"
    )
    assert "rig: the workload stamp never works" in caplog.text, caplog.text[-1500:]
    assert len(faults) == process.STALL_BEAT_RESTARTS, (
        f"the guard should absorb one fault per tolerated death ({process.STALL_BEAT_RESTARTS}), "
        f"got {len(faults)}: {messages}"
    )
    assert deaths == process.STALL_BEAT_RESTARTS + 1, (
        f"the death accounting did not run to its decision ({deaths} creations), so the guard "
        f"swallowed the cycle rather than the fault: {messages}"
    )
    assert (
        stall_watchdog.tick_deaths(os.getpid(), tmp_path) == (stall_watchdog.WORKLOAD,) * deaths
    ), "the record must survive the guard"


def test_a_tick_cancelled_while_the_session_is_live_is_recorded_as_a_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A CANCELLATION WITH ``stop`` UNSET IS NOT A SHUTDOWN (round 1, NIT 1).

    ``amain`` sets ``stop`` and then cancels, so the two cases are separable, and
    they must be separated: "the session is ending" and "something else ended the
    ticker" are the pair this module keeps insisting an instrument must not
    confuse. Nothing in the tree cancels this tick today; the cell exists so that
    the day something does, it is in the log and in the dump rather than silent.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)

    async def scenario() -> None:
        supervisor = asyncio.ensure_future(process._watch_stall_beats(asyncio.Event()))
        for _ in range(200):
            # THE REAL TICK, so the cancellation lands on the object production
            # creates; found by coroutine name rather than by reintroducing a
            # handle into ``process`` for a test's convenience.
            ticks = [
                task
                for task in asyncio.all_tasks()
                # ``get_coro()`` is optional, so the name is read through ``getattr``:
                # the alternative is a type error on a helper whose whole job is to
                # find a task by name.
                if getattr(task.get_coro(), "__name__", "") == "_beat_stall_watchdog"
            ]
            if ticks:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("the supervisor never created a tick to cancel")
        ticks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(scenario(), timeout=30.0))

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "cancelled while the session was live" in message for message in messages
    ), f"an externally cancelled tick still reads as a shutdown: {messages}"
    text = stall_watchdog.dump_path(os.getpid(), tmp_path).read_text(encoding="utf-8")
    assert "not a shutdown" in text, text


def test_recording_a_tick_death_with_nothing_armed_is_a_no_op(tmp_path: Path) -> None:
    """The in-process case: no dump file exists, and that must not be an error.

    A TUI host and a test never go through the runtime entry point, so nothing
    arms -- and the tick can still die there. ``note_tick_death`` returning False
    is what lets the supervisor say "this log line is the only trace" instead of
    raising inside the handler that exists to survive a death.

    The read is pointed at ``tmp_path`` rather than left to the default log
    directory: the default resolves from ``HOME``, and a test must never so much
    as read the operator's live store when the case under test does not need it.
    """
    assert stall_watchdog.is_armed() is False
    assert stall_watchdog.note_tick_death(stall_watchdog.WORKLOAD, "RuntimeError: x") is False
    assert stall_watchdog.tick_deaths(os.getpid(), tmp_path) == ()


def test_a_dead_workload_ticker_no_longer_masks_a_healthy_runtime(
    tmp_path: Path,
) -> None:
    """THE ACCEPTANCE CELL: the same rig, bare and supervised, on the real runtime.

    The control run is not decoration. ``bare`` is the same start site without
    the ticker supervisor; the fired diagnostic must leave it alive for the harness
    to reap. The supervised half below then proves recovery.
    What the control shows on the way through is the defect's whole shape:

    * the workload stamp FREEZES and stays frozen past the bound (printed ages),
      while the serving plane keeps reporting every 200 ms -- a HEALTHY runtime;
    * the dump-only timer fires on that frozen stamp, and the bounded harness then
      observes the process alive before reaping its exact child group;
    * the dump carries the fired marker and reads as the SILENCE leg, which is the
      wrong cause;
    * and ``tick_deaths`` is EMPTY, so nothing in the artifact separates this from
      a loop that genuinely parked.

    The supervised run is the same child with the production wiring, and it must
    survive four bounds: the stamp resumes, the death is in the log with its
    exception and in the dump as a named plane. A native timer fire alone cannot
    end this runtime.
    """
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir(parents=True, exist_ok=True)
    bare = _run_stalled_script(
        _DEAD_BEATER_CHILD,
        bare_dir,
        args=(str(DEAD_BEATER_BOUND_S), str(bare_dir), str(REPO), "bare"),
        timeout=60.0,
    )

    assert bare.returncode is None, (
        f"the dump-only control did not fire while its ticker was dead: "
        f"stdout={bare.stdout!r} stderr={bare.stderr!r}"
    )
    assert "beater-raised" in bare.stdout, bare.stdout
    assert "survived" not in bare.stdout, "the control ran through its explicit sleep window"
    bare_workload, bare_serving = _stamp_ages(bare.stdout)
    assert bare_workload, f"the control printed no stamp samples: {bare.stdout!r}"
    assert max(bare_workload) >= DEAD_BEATER_BOUND_S * 0.8, (
        f"the workload stamp never froze ({max(bare_workload):.2f}s of {DEAD_BEATER_BOUND_S}s), "
        f"so the rig is not reproducing the defect: {bare.stdout!r}"
    )
    assert max(bare_serving) < DEAD_BEATER_BOUND_S / 2, (
        f"the serving plane was not healthy ({max(bare_serving):.2f}s), so this run says "
        f"nothing about a runtime that was otherwise working: {bare.stdout!r}"
    )
    bare_pid = int(bare.stdout.split("armed:", 1)[1].split()[0])
    bare_text = _dump_for(bare_dir, bare_pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in bare_text, bare_text[-2000:]
    # ...AND IT NEVER CLAIMS THE LIVE HELD STATE (finding B of the 2026-09-23
    # convergence round). This child arms NO busy probe — there is no way for it to
    # report work in flight — and `held` means "work was in flight and this runtime is
    # still stalled", which a runtime that cannot answer must not have asserted on its
    # behalf. The fire is still on the record, and still read as a survived one by
    # :func:`fire_outcome`, because production expiry ends nothing either way.
    #
    # THE DIRECTION IS WHAT IS PINNED, not the arrival of the sampler's own line: this
    # harness reaps the child the moment the fired marker appears (it is written BEFORE
    # the stacks), so whether the post-fire beat lands inside that window is a race — and
    # a cell that asserted the observation line would fail on the lost side of it for no
    # reason. What must never happen is a held line on a runtime nothing reported work
    # for, and `held_fire` reads exactly that.
    assert (
        stall_watchdog.HELD_MARKER not in bare_text
    ), f"a fire over an unprobeable runtime raised the live held state: {bare_text[-900:]!r}"
    assert stall_watchdog.held_fire(bare_pid, bare_dir / "logs") is False
    assert stall_watchdog.fire_outcome(bare_pid, bare_dir / "logs") == stall_watchdog.FIRE_SURVIVED
    assert stall_watchdog.fired_leg(bare_pid, bare_dir / "logs") == (
        stall_watchdog.LEG_SILENCE
    ), f"the control's artifact no longer reads as the silence leg, which is the lie: {bare_text}"
    assert stall_watchdog.tick_deaths(bare_pid, bare_dir / "logs") == (), (
        "the control recorded a tick death; without the supervisor there is nothing to "
        "record it, so this means the record is being written by something else"
    )

    # -- THE FIX: the same child, the production start site --------------------
    supervised_dir = tmp_path / "supervised"
    supervised_dir.mkdir(parents=True, exist_ok=True)
    supervised = _run_script(
        _DEAD_BEATER_CHILD,
        supervised_dir,
        args=(str(DEAD_BEATER_BOUND_S), str(supervised_dir), str(REPO), "supervised"),
        timeout=60.0,
    )

    assert supervised.returncode == 0, (
        f"the runtime died anyway: rc={supervised.returncode} "
        f"stdout={supervised.stdout!r} stderr={supervised.stderr!r}"
    )
    assert "survived" in supervised.stdout, (
        f"the child never reached four bounds, so survival is not established: "
        f"{supervised.stdout!r} stderr={supervised.stderr!r}"
    )
    workload, serving = _stamp_ages(supervised.stdout)
    assert workload, f"the child printed no stamp samples: {supervised.stdout!r}"
    assert max(workload) < DEAD_BEATER_BOUND_S, (
        f"the workload stamp approached the bound ({max(workload):.2f}s of "
        f"{DEAD_BEATER_BOUND_S}s): the tick was not re-created, or it was re-created too "
        f"late to matter: {supervised.stdout!r}"
    )
    assert max(serving) < DEAD_BEATER_BOUND_S / 2, (
        f"the serving plane stopped reporting, so survival is not the supervision's doing: "
        f"{supervised.stdout!r}"
    )

    # THE LOG: WARNING, with the exception, at the moment of death.
    assert "WARNING" in supervised.stdout, supervised.stdout
    assert "WORKLOAD tick died" in supervised.stdout, supervised.stdout
    assert (
        "RuntimeError: rig: the workload tick's beat raised" in supervised.stdout
    ), supervised.stdout
    assert "re-creating it (death 1 inside" in supervised.stdout, supervised.stdout

    # THE RECORD: beside the plane's own stamp, in the dump, readable back.
    supervised_pid = int(supervised.stdout.split("armed:", 1)[1].split()[0])
    supervised_dump = _dump_for(supervised_dir, supervised_pid)
    assert stall_watchdog.tick_deaths(supervised_pid, supervised_dir / "logs") == (
        stall_watchdog.WORKLOAD,
    ), f"the dump does not name the dead tick: {supervised_dump.read_text(encoding='utf-8')}"
    supervised_text = supervised_dump.read_text(encoding="utf-8")
    assert "RuntimeError: rig: the workload tick's beat raised" in supervised_text
    assert (
        stall_watchdog.FIRED_MARKER not in supervised_text
    ), "a bound fired in the run that is supposed to have survived"
    assert stall_watchdog.fired_leg(supervised_pid, supervised_dir / "logs") is None


def _beater_module_functions(source: str) -> dict[str, ast.AST]:
    """The module-level function bodies of ``process``, by name, as AST nodes.

    AST and not text for the cells below: they pin the WIRING and the SHAPE of the
    supervisor, and a rewritten comment or docstring that happens to mention a name
    must not be able to fail them (agent review round 1, NIT 2 — the earlier
    version asserted the ABSENCE of a substring in ``inspect.getsource``, which is
    a pin on prose as much as on code).
    """
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _task_starters(node: ast.AST) -> list[ast.Call]:
    """Every task-starting call under ``node`` — ``ensure_future``/``create_task``.

    WHOLE-AST, AT ANY DEPTH, and that is the round-2 NIT. The earlier version looked
    only at the MODULE's own function bodies, while the sentence these cells carry
    was about the module: a ticker armed outside any function — a module-level
    expression, a ``lambda``, a class body — would have slipped past the check.
    Either the sentence or the check had to move, and the check moved, because the
    property being pinned is about the module and not about its functions.
    """
    starters: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute):
            called = func.attr
        elif isinstance(func, ast.Name):
            called = func.id
        else:
            continue
        if called in {"ensure_future", "create_task"}:
            starters.append(child)
    return starters


def _started_name(starter: ast.Call) -> str | None:
    """The NAME of the coroutine ``starter`` was handed, or ``None``."""
    if len(starter.args) != 1 or not isinstance(starter.args[0], ast.Call):
        return None
    inner = starter.args[0].func
    if isinstance(inner, ast.Name):
        return inner.id
    if isinstance(inner, ast.Attribute):
        return inner.attr
    return None


def _module_aliases(module: ast.Module) -> dict[str, str]:
    """Module-level ``name = other_name`` bindings, for resolving an ALIASED starter.

    AGENT REVIEW ROUND 3, N1: the round-2 check matched the name written at the call site, so
    ``_orphan = _beat_stall_watchdog`` followed by ``ensure_future(_orphan(stop))`` was a second
    real arm site the check counted as one — the other half of the round-2 NIT. Only single-name
    targets whose value is a single name are followed, and only at module level, which is enough
    for an alias and cannot invent a resolution the language would not make.
    """
    aliases: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = node.value.id
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Name)
            and isinstance(node.target, ast.Name)
        ):
            aliases[node.target.id] = node.value.id
    return aliases


def _resolves_to(name: str, aliases: dict[str, str]) -> set[str]:
    """``name`` and every module-level alias it stands for, transitively."""
    seen = {name}
    while (nxt := aliases.get(name)) is not None and nxt not in seen:
        name = nxt
        seen.add(name)
    return seen


def _starts_coroutine(starter: ast.Call, name: str, aliases: dict[str, str]) -> bool:
    """Whether ``starter`` hands ``name`` — or an alias of it — to a task starter."""
    started = _started_name(starter)
    return started is not None and name in _resolves_to(started, aliases)


def test_a_failing_record_cannot_defeat_the_give_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A TERMINAL STATE A DIAGNOSTIC CAN SWITCH OFF IS NOT TERMINAL (round 2, MINOR 2).

    The give-up decision used to be re-derived AFTER the record write, so anything that
    made that write raise skipped the ``return``: the guard caught it and the loop went
    round again. Reproduced by the reviewer as 130 ticks, 130 guard tracebacks and no
    give-up line in 2 s, with the plane never stamped — a failing REPORT defeating the
    fail-safe, which is this PR's own defect shape wearing a different hat.

    The lever is the module's own record function raising, and the cell demands the
    supervisor still reaches its decision: the tick is created exactly
    ``STALL_BEAT_RESTARTS + 1`` times, the give-up is logged, and the missing record is
    reported as a missing record rather than as a death that did not happen.

    THE ``wait_for`` IS WHAT MAKES THIS CELL RED RATHER THAN WEDGED on the un-protected
    code, where the loop has no exit at all.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)
    deaths = 0

    async def always_dying_tick(stop: asyncio.Event) -> None:
        nonlocal deaths
        deaths += 1
        raise RuntimeError(f"tick death {deaths}")

    def unwritable_record(plane: str, reason: str) -> bool:
        raise OSError("rig: the record cannot be written")

    monkeypatch.setattr(process, "_beat_stall_watchdog", always_dying_tick)
    monkeypatch.setattr(stall_watchdog, "note_tick_death", unwritable_record)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(process._watch_stall_beats(asyncio.Event()), timeout=20.0))

    messages = [record.getMessage() for record in caplog.records]
    assert deaths == process.STALL_BEAT_RESTARTS + 1, (
        f"the supervisor never reached its decision ({deaths} creations), so a record that "
        f"cannot be written defeats the give-up and makes the terminal state optional: {messages}"
    )
    assert any("GIVES UP" in message for message in messages), messages
    assert sum("could not be written" in message for message in messages) == deaths, messages


class _ExplodingLogger:
    """A logger whose calls raise — optionally only for the messages it is told to.

    ``RecursionError`` rather than a synthetic ``RuntimeError``: in this venv
    ``StreamHandler.emit`` re-raises it instead of routing it to ``handleError``, so a runtime
    wedged enough to blow the recursion limit can reach this path for real. That is why a log
    call must not be able to break the path it reports on (agent review round 4, MINOR).

    ``only`` is for a cell that still needs the OTHER lines to arrive — the cancellation cell
    asserts the record's own warning is present, so only its own line explodes there — and
    ``None`` explodes on everything.
    """

    def __init__(self, real: logging.Logger, only: str | None = None) -> None:
        self._real = real
        self._only = only

    def warning(self, message: str, *args: object, exc_info: bool = False) -> None:
        if self._only is None or self._only in message:
            raise RecursionError("rig: maximum recursion depth exceeded while reporting")
        self._real.warning(message, *args, exc_info=exc_info)


def test_a_failing_record_does_not_replace_a_cancellation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same finding's sibling: the CANCELLATION must not depend on the record either.

    An un-totaled ``note_tick_death`` in the cancellation path raised where the shutdown
    path had already committed to re-raising, so the supervisor ended ``OSError`` where it
    was asked for ``cancelled`` — a diagnostic replacing a terminal state, one branch over
    from the give-up that the cell above covers. Round 4 added the LOG LINE beside that
    write to the same cell: it now raises for its own message only, so the assertion below
    about the record's warning still holds while the branch is shown to end ``cancelled``
    with no report possible either.

    The REAL tick is left in place (it sleeps), so the cancellation lands on the object
    production creates and the assertion is about how the supervisor ENDS.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(
        process,
        "logger",
        _ExplodingLogger(process.logger, only="was cancelled while the session was live"),
    )

    def unwritable_record(plane: str, reason: str) -> bool:
        raise OSError("rig: the record cannot be written")

    monkeypatch.setattr(stall_watchdog, "note_tick_death", unwritable_record)

    async def scenario() -> None:
        supervisor = asyncio.ensure_future(process._watch_stall_beats(asyncio.Event()))
        for _ in range(200):
            ticks = [
                task
                for task in asyncio.all_tasks()
                if getattr(task.get_coro(), "__name__", "") == "_beat_stall_watchdog"
            ]
            if ticks:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("the supervisor never created a tick to cancel")
        ticks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(scenario(), timeout=20.0))

    assert any("could not be written" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


def test_a_failing_dump_path_cannot_defeat_the_give_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """THE CLAUSE IS PART OF THE GIVE-UP, NOT JUST THE WRITE (agent review round 3, M2).

    Round 2 made the RECORD total and left the sentence around it building
    ``stall_watchdog.dump_path()`` at the call site — and ``dump_path`` reaches ``log_dir()``,
    so rigging THAT to raise, with the record left total and succeeding, still skipped the
    ``return``: the reviewer measured 121 creations, no ``GIVES UP`` line, and a supervisor
    that never came back — spawning a task on every pass. The fix moved the whole clause inside
    the total call, so there is nothing left between the decision and the return.

    Same shape as the round-2 cell, one line down, and ``wait_for`` is again what makes it red
    rather than wedged on the un-protected code.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)
    deaths = 0

    async def always_dying_tick(stop: asyncio.Event) -> None:
        nonlocal deaths
        deaths += 1
        raise RuntimeError(f"tick death {deaths}")

    def unnameable_dump(pid: int | None = None, directory: Path | None = None) -> Path:
        raise OSError("rig: the dump path cannot be resolved")

    monkeypatch.setattr(process, "_beat_stall_watchdog", always_dying_tick)
    monkeypatch.setattr(stall_watchdog, "dump_path", unnameable_dump)

    with caplog.at_level(logging.WARNING, logger=process.__name__):
        asyncio.run(asyncio.wait_for(process._watch_stall_beats(asyncio.Event()), timeout=20.0))

    messages = [record.getMessage() for record in caplog.records]
    assert deaths == process.STALL_BEAT_RESTARTS + 1, (
        f"the supervisor never reached its decision ({deaths} creations), so a path lookup that "
        f"cannot resolve defeats the give-up — round 2's finding one line down: {messages}"
    )
    assert any("GIVES UP" in message for message in messages), messages
    assert any("dump path could not be resolved" in message for message in messages), messages


def test_a_failing_log_cannot_defeat_the_give_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REPORT MUST NOT BREAK THE PATH IT REPORTS ON (agent review round 4, MINOR).

    Round 2 made the record total and round 3 made its clause total; the give-up's own
    ``logger.warning`` still sat between the decision and the ``return`` — as did the guard's,
    which is the one that keeps the supervision alive at all. Rigged, the give-up never
    happened: the log's exception reached the guard, the guard slept and the loop went round
    again (321 creations in 6 s, against 4 for the control). Every log call in the supervision
    now goes through ``_safe_warning``, so the logger here raises on EVERY message — and the
    recovery stamp raises too, which is what drives each cycle into the guard, so that call is
    exercised rather than assumed.

    The records are the point: with the log impossible, the dump still carries every death.
    A missing log line is a missing diagnostic, never a missing event.
    """
    from local_operator.session.runtime import process

    monkeypatch.setattr(stall_watchdog, "faulthandler", _FakeFaulthandler())
    assert stall_watchdog.arm(seconds=60.0, directory=tmp_path)
    monkeypatch.setattr(process, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(process, "logger", _ExplodingLogger(process.logger))
    deaths = 0

    async def always_dying_tick(stop: asyncio.Event) -> None:
        nonlocal deaths
        deaths += 1
        raise RuntimeError(f"tick death {deaths}")

    def unwritable_stamp(plane: str) -> None:
        raise OSError("rig: the recovery stamp cannot be written")

    monkeypatch.setattr(process, "_beat_stall_watchdog", always_dying_tick)
    monkeypatch.setattr(stall_watchdog, "beat", unwritable_stamp)

    asyncio.run(asyncio.wait_for(process._watch_stall_beats(asyncio.Event()), timeout=20.0))

    assert deaths == process.STALL_BEAT_RESTARTS + 1, (
        f"the supervisor never reached its decision ({deaths} creations) while its logger "
        f"raised, so a report that cannot be made defeats the give-up"
    )
    assert stall_watchdog.tick_deaths(os.getpid(), tmp_path) == (stall_watchdog.WORKLOAD,) * (
        process.STALL_BEAT_RESTARTS + 1
    ), "the dump lost deaths when the log could not be written: the record depends on the report"


def test_the_runtime_entry_point_supervises_the_workload_tick() -> None:
    """The fix is WIRED, and wired in ONE place.

    The cells above drive ``_watch_stall_beats`` directly, so a revert of the one
    line in ``amain`` that selects it would leave all of them green while
    production went back to an unobserved tick — the same shape
    ``test_the_only_arm_site_is_the_runtime_entry_point`` exists for on the other
    side of this module.

    The second assertion is the other half: the tick's coroutine is handed to a task
    starter exactly ONCE in this module — **at any depth, not just inside a module-level
    function, and through an alias if one is used** (agent review round 2, NIT, and round
    3, N1: the check first saw only function bodies, and then only the name written at the
    call site, while its sentence claimed both) — so a second arm site beside the
    supervisor cannot appear unnoticed.
    """
    from local_operator.session.runtime import process

    source = Path(process.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = _beater_module_functions(source)
    aliases = _module_aliases(module)
    assert any(
        _starts_coroutine(starter, "_watch_stall_beats", aliases)
        for starter in _task_starters(functions["amain"])
    ), "amain does not start the supervisor, so a real runtime can run with an unobserved tick"
    ticks = [
        starter
        for starter in _task_starters(module)
        if _starts_coroutine(starter, "_beat_stall_watchdog", aliases)
    ]
    assert len(ticks) == 1, (
        f"the tick's coroutine is handed to a task starter {len(ticks)} times in this module "
        f"(at any depth, aliases resolved): with zero the runtime has no bound ticker at all, "
        f"and with two the second one is unobserved — the defect this PR removes"
    )


def test_the_supervisors_attempts_are_spaced_by_the_tick_itself() -> None:
    """No sleep of the supervisor's own on the normal path.

    WHY THIS IS A STRUCTURAL PIN AND NOT A MEASUREMENT. The spacing that makes a
    hot failure bounded is the tick's own leading sleep — ``_beat_stall_watchdog``
    is wait-then-beat — so an attempt cannot cost less than one heartbeat however
    the supervisor is written. An extra ``await asyncio.sleep(HEARTBEAT_INTERVAL_S)``
    here is exactly what the round-1 draft had, and it is what made its grace
    figure wrong (agent review round 1, MINOR 2); the guard is the ONE place that
    may sleep, for the case where even the tick's own sleep cannot run. A
    wall-clock cell would re-derive this from a loaded host; the structure is the
    fact.
    """
    from local_operator.session.runtime import process

    functions = _beater_module_functions(Path(process.__file__).read_text(encoding="utf-8"))
    supervisor = functions["_watch_stall_beats"]
    in_a_handler: set[int] = set()
    for handler in (node for node in ast.walk(supervisor) if isinstance(node, ast.ExceptHandler)):
        in_a_handler.update(
            line
            for child in ast.walk(handler)
            if isinstance(line := getattr(child, "lineno", None), int)
        )
    sleeps = [
        node.lineno
        for node in ast.walk(supervisor)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "sleep"
    ]
    assert sleeps, "the supervisor never yields to the loop at all"
    outside = [line for line in sleeps if line not in in_a_handler]
    assert not outside, (
        f"the supervisor sleeps on its normal path (lines {outside}), which makes one attempt "
        f"cost two heartbeats: every grace and deadline figure in its docstring would be "
        f"wrong again"
    )


# ============================================================================
# PARKED vs EXECUTING: the liveness leg tells them apart by the loop's FRAME
# ============================================================================
#
# WHY THIS SECTION EXISTS. Measured on this machine's own dump store on 2026-09-22
# (40 files, 37 fired): in 20 of the 37 the workload loop's stack ran through this
# tree's per-event subagent projection path -- `_run_turn -> _emit -> handler ->
# _refresh_state -> set_subagent_details` and the `_publish_busy -> _publish_subagents
# -> subagent_counts -> nodes -> node -> _describe` walk -- i.e. the loop was EXECUTING
# the runtime's own code when the deadline expired, not parked in a call it never
# returned from. The bound cannot see "the loop is running" from a stamp, because the
# stamp is what a loop busy in its own code fails to deliver, so the sampler now
# watches the loop's frame as a second sign of life.
#
# THE CELLS BELOW ARE THE TWO HALVES OF THE CLAIM, and neither can stand alone:
# a FRAME THAT MOVES must not be cut (or the fairness fix is decorative) and a FRAME
# THAT FROZEN must still be cut (or the 2026-09-20 incident the bound was written for
# goes unbounded). The second half is the one a naive reading of the fix breaks, so it
# is tested in BOTH of the shapes it really occurs in: a GIL-releasing park, where the
# sampler keeps looking and must DECLINE to abstain, and a GIL-HOLDING C call, where no
# Python thread runs at all.


class _Frame:
    """Just enough of a frame for the observation: the three fields it reads."""

    def __init__(self, filename: str, lineno: int, function: str) -> None:
        self.f_code = SimpleNamespace(co_filename=filename, co_name=function)
        self.f_lineno = lineno


class _FakeSys:
    """A ``sys`` whose ``_current_frames`` a cell can drive, keyed by thread ident."""

    def __init__(self) -> None:
        self.frames: dict[int, _Frame] = {}

    def _current_frames(self) -> dict[int, _Frame]:
        return dict(self.frames)


def _observed_dump(tmp_path: Path, pid: int) -> tuple[Path, Any]:
    """A dump under the directory the READERS resolve, so ``executing_planes`` sees it."""
    dump = stall_watchdog.dump_path(pid, tmp_path)
    dump.parent.mkdir(parents=True, exist_ok=True)
    handle = dump.open("w", encoding="utf-8")
    handle.write(f"{stall_watchdog.ARM_MARKER}test header\n")
    handle.flush()
    return dump, handle


def _watch_this_thread(armed: Any, plane: str, frames: dict[int, "_Frame"], frame: "_Frame") -> int:
    """Learn THIS thread as a plane's loop, with ``frame`` visible under a fake sys.

    The production learn site is ``beat`` (see ``_Armed.loop_thread``); the cells in
    this section drive the observation directly, so they perform the same two steps
    it does. THE THREAD HAS TO BE REAL AND LIVE: the observation asks ``is_alive()``
    before it reads a frame, so a stand-in object here would be testing the fixture
    rather than the module.
    """
    thread = threading.current_thread()
    armed.loop_thread[plane] = thread
    ident = thread.ident
    assert ident is not None, "the test thread has no ident, so no frame can be read"
    frames[ident] = frame
    return ident


def test_a_moving_frame_is_observed_and_a_still_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observation itself, on a controlled frame: movement, not wedgedness.

    Driven through ``sys._current_frames`` rather than by parking a real thread,
    because the claim is about the DISCRIMINATOR and a real parked thread can only
    ever show one of its two answers. Each assertion is a different reason the frame
    might not read as movement, and each has to fall on the correct side:
    the first look has nothing to compare against, an unchanged frame is a park, a
    changed line or a changed function is execution, and a frame that cannot be read
    is not movement either — inventing a sign of life from an unreadable frame would
    extend the bound on exactly the runtime that cannot be looked at.

    AND THE OBSERVATION IS NOT THE EXTENSION, which is what the closing assertion
    pins: none of this moves ``sign_of_life``. The timestamp the deadline is measured
    from has exactly ONE writer, the granted path in ``_extend_for_execution``, so a
    reader may see movement recorded in the dump on a run that was then cut at the
    bound — the refused hand-off. Letting the observation move the deadline is the
    defect ``test_a_raising_probe_does_not_move_the_deadline_a_sibling_beats_on``
    exists for.
    """
    fake_sys = _FakeSys()
    monkeypatch.setattr(stall_watchdog, "sys", fake_sys)
    dump, handle = _observed_dump(tmp_path, 4243)
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4243, lambda: ("still", True))
    # THE SIBLING IS PARKED FAR OUT, and it has to be stated rather than left to the
    # seed: ``_Armed.__init__`` stamps the SERVING plane with the host's own
    # ``time.monotonic()``, whose ORIGIN differs by machine (seconds since boot on a CI
    # runner, hours on a long-lived host). Leaving it implicit made the closing
    # deadline assertion depend on which of the two stamps was earlier, which is how it
    # passed here and failed on the runner (``assert 502.78 == (507.0 + 4.0)``).
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    ident = _watch_this_thread(
        armed, stall_watchdog.WORKLOAD, fake_sys.frames, _Frame("loop.py", 11, "walk")
    )

    # THE FIRST LOOK: nothing to compare against, so nothing is executing. This is
    # what keeps a moment of silence after a beat from reading as a moving loop.
    assert armed.observe_execution(500.0) == ((), ())
    assert armed.last_move == {}, "the first look claimed movement out of nothing"
    # THE SAME FRAME: a park, in the one shape the bound exists for.
    assert armed.observe_execution(501.0) == ((), ())
    assert (
        armed.sign_of_life(stall_watchdog.WORKLOAD) == 100.0
    ), "an unchanged frame moved the plane's sign of life"
    # A DIFFERENT LINE: executing, and the observation is dated.
    fake_sys.frames[ident] = _Frame("loop.py", 12, "walk")
    assert armed.observe_execution(502.0) == (
        (stall_watchdog.WORKLOAD,),
        (stall_watchdog.WORKLOAD,),
    )
    assert armed.last_move[stall_watchdog.WORKLOAD] == 502.0
    assert (
        armed.sign_of_life(stall_watchdog.WORKLOAD) == 100.0
    ), "the observation alone moved the deadline, so a refused hand-off would too"
    # A DIFFERENT FUNCTION, same line: still executing — the triple, not the line.
    fake_sys.frames[ident] = _Frame("loop.py", 12, "other_walk")
    moved, noted = armed.observe_execution(503.0)
    assert moved == (stall_watchdog.WORKLOAD,)
    assert moved and not noted, (
        "the SAME stretch opened a second record, so the artifact grows per look "
        "rather than per stretch"
    )
    assert armed.last_move[stall_watchdog.WORKLOAD] == 503.0
    # A FRAME THAT CANNOT BE READ is not movement.
    del fake_sys.frames[ident]
    assert armed.observe_execution(504.0) == ((), ())
    assert (
        armed.last_move[stall_watchdog.WORKLOAD] == 503.0
    ), "a frame that could not be read was counted as a sign of life"
    # A FRAME THAT COMES BACK IS A NEW BASELINE, not a movement: there is nothing to
    # compare it against, exactly as on the first look. Inventing a sign of life from
    # a gap would be claiming progress on the strength of being unable to look.
    fake_sys.frames[ident] = _Frame("loop.py", 12, "walk")
    assert armed.observe_execution(505.0) == ((), ())
    assert armed.last_move[stall_watchdog.WORKLOAD] == 503.0
    # ...AND A STRETCH ENDS WHEN THE LOOP STAMPS FOR ITSELF AGAIN, not when a single
    # look happens to catch the same line twice: the top frame alternates between a
    # call and its caller, so a rule keyed on one still look would re-record the same
    # stretch several times a second. This is the stamp, and the next movement after
    # it is therefore a NEW stretch.
    fake_sys.frames[ident] = _Frame("loop.py", 13, "walk")
    moved, noted = armed.observe_execution(506.0)
    assert moved == (stall_watchdog.WORKLOAD,)
    assert not noted, "a gap in readability ended the stretch, but nothing reported"
    armed.last_beat[stall_watchdog.WORKLOAD] = 507.0
    fake_sys.frames[ident] = _Frame("loop.py", 14, "walk")
    moved, noted = armed.observe_execution(508.0)
    assert moved == (stall_watchdog.WORKLOAD,)
    assert noted == (
        stall_watchdog.WORKLOAD,
    ), "a stretch after the loop reported for itself was not recorded as its own"
    # A PLANE WITH NO LEARNED THREAD IS NEVER OBSERVED — the never-engaged class, and
    # why the fix cannot reach it.
    assert stall_watchdog.SERVING not in armed.loop_thread
    # AND NOTHING ABOVE GRANTED AN EXTENSION: the observation is complete and the
    # deadline is still the plane's own stamp plus the bound.
    assert armed.executing_at == {}, "an observation moved the deadline without a grant"
    assert (
        armed.deadline() == armed.last_beat[stall_watchdog.WORKLOAD] + 4.0
    ), "the observation moved the deadline the bound fires on"
    handle.close()


def test_a_recycled_ident_does_not_extend_a_plane_whose_loop_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DEAD PLANE IS CUT EVEN WHEN ITS IDENT IS TAKEN BY A LIVE THREAD.

    The observation used to key on ``threading.get_ident()``, which CPython RECYCLES:
    on this host six sequential short-lived threads collapsed to one distinct ident.
    Once the plane's loop thread had ended, ``sys._current_frames()[ident]`` began
    returning some OTHER live thread's frames, so an executing stranger read as the
    dead plane still working — ``executing_at`` advanced and the liveness leg abstained
    on a plane with no reporter, indefinitely. That is precisely the fail-safe the tick
    supervision documents (``_watch_stall_beats``: a plane whose reporter is gone is
    ended one deadline after its last stamp), so the module was defeating its own
    promise. The old cells only covered the shape where the ident is ABSENT from the
    frame map, which is the wrong shape for a recycled one — hence this cell.

    THE CELL ASSERTS ITS OWN PRECONDITION: the live thread must really hold the dead
    thread's ident, or the recycling it exists to exercise did not happen and the
    rest of the assertions would pass vacuously.
    """
    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    fake_sys = _FakeSys()
    monkeypatch.setattr(stall_watchdog, "sys", fake_sys)
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    dump, handle = _observed_dump(tmp_path, 4251)
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4251, lambda: ("still", True))
    # The learn site itself, not a hand-written field: the plane is learned by a beat
    # ISSUED FROM the short-lived thread, exactly as a runtime learns it.
    monkeypatch.setattr(stall_watchdog, "_ARMED", armed)
    ended = threading.Thread(
        target=lambda: stall_watchdog.beat(stall_watchdog.WORKLOAD), name="dead-loop", daemon=True
    )
    ended.start()
    ended.join(timeout=5)
    assert not ended.is_alive(), "the plane's loop thread did not end, so nothing is dead"
    assert armed.loop_thread[stall_watchdog.WORKLOAD] is ended
    dead_ident = ended.ident
    assert dead_ident is not None

    # A LIVE THREAD TAKES THE SAME IDENT. Spawned one at a time and joined unless it
    # reused the dead thread's ident, because that reuse is the whole mechanism.
    live: threading.Thread | None = None
    for _ in range(64):
        candidate = threading.Thread(target=lambda: time.sleep(30), name="ident-thief", daemon=True)
        candidate.start()
        if candidate.ident == dead_ident:
            live = candidate
            break
        candidate.join(timeout=5)
    assert live is not None, (
        "no short-lived thread reused the dead loop's ident, so this cell cannot say "
        "anything about the recycled-ident shape"
    )
    assert live.is_alive()

    # Its frames, under the SAME ident the dead plane is keyed by, and moving between
    # the two looks below — which is what an ident-keyed lookup would read as the dead
    # plane executing.
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    fake.wall = 101.0
    monkeypatch.setattr(stall_watchdog, "_ARMED", armed)
    fake_sys.frames[dead_ident] = _Frame("stranger.py", 11, "moving")
    try:
        assert armed.observe_execution(500.0) == ((), ())
        fake_sys.frames[dead_ident] = _Frame("stranger.py", 12, "moving")
        assert armed.observe_execution(501.0) == (
            (),
            (),
        ), "a live stranger's frames were read as the dead plane executing"
        # THE ASSERTION THE FINDING ASKS FOR: the plane is CUT, not abstained. Its
        # deadline is its own last stamp plus the bound, untouched by the stranger.
        assert armed.executing_at == {}, "the dead plane was granted an extension"
        assert armed.sign_of_life(stall_watchdog.WORKLOAD) == 100.0
        assert armed.pin() == (stall_watchdog.WORKLOAD, 104.0), (
            "the deadline a sibling's beat would re-arm from moved for a plane whose "
            "loop is gone"
        )
        # ...and the sibling's own beat re-arms the SHARED timer from that same
        # unchanged deadline, which is the path the extension would have travelled.
        spy.armed.clear()
        stall_watchdog.beat(stall_watchdog.SERVING)
        assert spy.armed, "the sibling's beat did not reach the timer"
        assert spy.armed[-1][0] == pytest.approx(
            104.0 - fake.wall, abs=1e-9
        ), "a beat re-armed the shared timer for an extension a dead plane never earned"
    finally:
        if live is not None:
            live.join(timeout=0)
        handle.close()


def test_a_raising_probe_does_not_move_the_deadline_a_sibling_beats_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE REFUSAL IS TOTAL: an unevaluable probe cannot move the shared deadline.

    Property (a) is stated in three places — the module docstring,
    ``_extend_for_execution`` and ``_read_probe``: the extension is refused unless a
    probe was supplied AND answered on that very sample, so a runtime whose progress
    leg is inert or unevaluable keeps today's behaviour rather than becoming
    unbounded. THE REFUSAL USED TO GATE ONLY THE SAMPLER'S OWN RE-ARM: the timestamp
    was written by the OBSERVATION, before any leg decided, and ``pin``/``deadline`` —
    which a ``beat`` on the OTHER plane re-arms the shared C timer from — read it
    unconditionally. So an executing loop with a raising probe survived on a runtime
    whose sibling kept stamping: the deadline never counted down.

    The discriminating assertion is therefore on ``pin()`` and on the duration a
    sibling's ``beat`` actually hands the C timer, not on whether the sampler armed
    anything: ``spy.armed == []`` was already true and stayed true, which is why the
    pre-fix cell was green while the process was unbounded.
    """
    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    fake_sys = _FakeSys()
    monkeypatch.setattr(stall_watchdog, "sys", fake_sys)
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    def broken() -> tuple[object, bool]:
        raise RuntimeError("the probe cannot read live session state right now")

    dump, handle = _observed_dump(tmp_path, 4252)
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4252, broken)
    monkeypatch.setattr(stall_watchdog, "_ARMED", armed)
    ident = _watch_this_thread(
        armed, stall_watchdog.WORKLOAD, fake_sys.frames, _Frame("loop.py", 11, "walk")
    )
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    fake.wall = 101.0

    # Two samples with the loop's frames MOVING, which is the state the extension
    # exists for — and the probe raises for both, so neither earns one.
    assert stall_watchdog._sample(armed) is False
    fake_sys.frames[ident] = _Frame("loop.py", 12, "walk")
    fake.wall += 1.0
    assert stall_watchdog._sample(armed) is False

    assert armed.executing_at == {}, "the refused hand-off still wrote the extension"
    assert armed.pin() == (
        stall_watchdog.WORKLOAD,
        104.0,
    ), "an unevaluable probe moved the deadline, so the runtime is unbounded"
    # THE SIBLING IS THE CARRIER, and this is the number it hands the C timer.
    spy.armed.clear()
    stall_watchdog.beat(stall_watchdog.SERVING)
    assert spy.armed, "the sibling's beat did not reach the timer"
    assert spy.armed[-1][0] == pytest.approx(104.0 - fake.wall, abs=1e-9), (
        f"the sibling's beat re-armed for {spy.armed[-1][0]}s instead of counting down "
        f"from the plane's own stamp plus the bound"
    )
    handle.close()


def test_a_failed_re_arm_leaves_the_timer_on_the_shorter_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ONE BRANCH THAT DECIDES "fire early rather than late", exercised.

    An extension that cannot be handed to ``faulthandler`` leaves the timer on its
    PREVIOUS deadline, which is shorter than the extension would have been — the safe
    direction, and a claim that was only commented before this cell existed. What must
    NOT happen is the alternative reading, that a failed re-arm withdraws the bound:
    the deadline the extension recorded stays in force, so every later beat and the
    progress leg still measure from it.
    """
    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    fake_sys = _FakeSys()
    monkeypatch.setattr(stall_watchdog, "sys", fake_sys)

    class _RaisingFaulthandler(_FakeFaulthandler):
        """A C timer that refuses every arming, which is the branch under test."""

        def __init__(self) -> None:
            super().__init__()
            #: What the extension TRIED to hand the timer, so "did it try" and "did it
            #: land" stay separate facts.
            self.attempts: list[float] = []

        def dump_traceback_later(self, seconds: float, **kwargs: Any) -> None:
            self.attempts.append(seconds)
            raise OSError("the C timer refused this arming")

    spy = _RaisingFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    dump, handle = _observed_dump(tmp_path, 4253)
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4253, lambda: ("still", True))
    ident = _watch_this_thread(
        armed, stall_watchdog.WORKLOAD, fake_sys.frames, _Frame("loop.py", 11, "walk")
    )
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    fake.wall = 101.0

    assert stall_watchdog._sample(armed) is False
    fake_sys.frames[ident] = _Frame("loop.py", 12, "walk")
    fake.wall += 1.0
    assert stall_watchdog._sample(armed) is False, "a failed re-arm fired the bound"
    assert spy.attempts, "the extension never reached the C timer at all"
    # The extension is IN FORCE even though the arming failed: a later beat measures
    # from it, which is what keeps a transient arming failure from being a way for the
    # hand-off to be silently withdrawn.
    assert armed.executing_at == {stall_watchdog.WORKLOAD: fake.wall}
    assert armed.deadline() == fake.wall + 4.0
    handle.close()


def test_the_abstention_records_the_observation_and_re_arms_the_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extension, its record, and the refusal that keeps it bounded.

    Two things have to hold at once and only one of them is visible from the
    deadline: the timer is pushed out (so the process is not cut), and the dump says
    why (so a reader is not left inferring it from an absence). The third assertion
    is the bound's teeth — with NO probe there is no progress leg to hand the runtime
    to, so the observation is recorded and the deadline is NOT pushed out.
    """
    fake = _FakeClock()
    monkeypatch.setattr(stall_watchdog, "time", fake)
    fake_sys = _FakeSys()
    monkeypatch.setattr(stall_watchdog, "sys", fake_sys)
    spy = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)

    dump, handle = _observed_dump(tmp_path, 4244)
    armed = stall_watchdog._Armed(dump, handle, 4.0, 4244, lambda: ("still", True))
    # The OTHER plane kept stamping, so the deadline in play is the executing one's.
    # Without this the serving plane's own silence is the earliest deadline and the
    # sibling names IT, which is correct behaviour and not what this cell is about.
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    ident = _watch_this_thread(
        armed, stall_watchdog.WORKLOAD, fake_sys.frames, _Frame("loop.py", 11, "walk")
    )
    assert stall_watchdog._sample(armed) is False, "the first look fired"
    fake_sys.frames[ident] = _Frame("loop.py", 12, "walk")
    fake.wall += 1.0
    assert stall_watchdog._sample(armed) is False, "an executing loop was cut"
    assert spy.armed, "the abstention never reached the C timer, so the bound still ran out"
    assert stall_watchdog.executing_planes(4244, tmp_path) == (
        stall_watchdog.WORKLOAD,
    ), f"the dump does not say the loop was executing: {dump.read_text(encoding='utf-8')!r}"
    # THE SIBLING NAMES THE DEADLINE THE TIMER NOW HOLDS, so a reader with no dump open
    # can still see which plane's silence the re-arm was measured against.
    _epoch, leg = _deadline_record(tmp_path, 4244)
    assert leg == stall_watchdog.WORKLOAD, leg

    # THE REFUSAL. Same observation, no probe: the record still lands (the file's
    # reader is asking what the loop was doing), and nothing re-arms — which is the
    # only thing standing between a moving frame and an unbounded process.
    bare_dump, bare_handle = _observed_dump(tmp_path, 4245)
    bare = stall_watchdog._Armed(bare_dump, bare_handle, 4.0, 4245, None)
    bare.last_beat[stall_watchdog.WORKLOAD] = 100.0
    bare.last_beat[stall_watchdog.SERVING] = 9_999.0
    bare_ident = _watch_this_thread(
        bare, stall_watchdog.WORKLOAD, fake_sys.frames, _Frame("loop.py", 11, "walk")
    )
    assert stall_watchdog._sample(bare) is False
    spy.armed.clear()
    fake_sys.frames[bare_ident] = _Frame("loop.py", 13, "walk")
    fake.wall += 1.0
    assert stall_watchdog._sample(bare) is False
    assert spy.armed == [], (
        "the bound abstained with no progress leg to hand the runtime to, so a "
        "frames-moving loop would now be unbounded"
    )
    assert bare.executing_at == {}, "the refused hand-off moved the deadline anyway"
    assert bare.deadline() == 104.0
    assert stall_watchdog.executing_planes(4245, tmp_path) == (stall_watchdog.WORKLOAD,)
    handle.close()
    bare_handle.close()


def test_the_observation_sees_calls_rather_than_instruction_level_progress() -> None:
    """WHY THE RIGS LOOK THE WAY THEY DO: the discriminator's reach, measured.

    ``frame.f_lineno`` read off a FOREIGN frame is refreshed at a CALL, not per
    instruction, so a loop of bare arithmetic reads as one frozen line however fast it
    runs while a loop that calls something does not. That is a property of CPython
    rather than of this module, and it decides which shapes the abstention can rescue —
    so it is pinned here rather than left implied by a rig that happens to have calls in
    it. The bounds are deliberately loose (the measurement behind them is 0 of 60
    against 35 of 55): what this asserts is the DIRECTION, and a loaded host must not be
    able to turn that into a flake.

    THE CONSEQUENCE IS PART OF THE CLAIM, and it is not an exemption: a running loop
    that holds one call-free frame is still read as silent and is still cut by the
    liveness leg. See the module docstring — the PROGRESS leg is what bounds that shape,
    because it burns a core with its work standing still.
    """

    def leaf(value: int) -> int:
        return value + 1

    def bare(running: list[bool]) -> None:
        # NO CALL IN THE LOOP AT ALL: `running[0]` is a subscript, and using
        # `running[0]` rather than an Event is the whole point — an `is_set()` in the
        # condition would be a call and would make this the other shape.
        total = 0
        while running[0]:
            total += 1
            total += 2
            total += 3

    def calling(running: list[bool]) -> None:
        total = 0
        while running[0]:
            total += leaf(1)
            total += leaf(2)
            total += leaf(3)

    def changes_seen(worker: Any) -> int:
        """How often the top frame triple differs between two 20 ms looks.

        SIXTY LOOKS AND NOT TWENTY, because the assertion is a COUNT of differences
        and the count is a sampled statistic: measured under fleet load, a 20-look run
        once came back with 2 where the same code returned 3-5 unloaded
        (``assert 2 >= 3``), which is a flake in the CELL and not a thinner
        discriminator — the underlying measurement is 42/65 against 0/53. The extra
        ~0.8 s of runtime is what buys a margin the host's load cannot close.
        """
        running = [True]
        thread = threading.Thread(target=worker, args=(running,), daemon=True)
        thread.start()
        ident = thread.ident
        assert ident is not None, "the worker did not start, so nothing was sampled"
        try:
            time.sleep(0.1)
            seen, previous = 0, None
            for _ in range(60):
                frame = sys._current_frames().get(ident)
                if frame is not None:
                    triple = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                    if previous is not None and triple != previous:
                        seen += 1
                    previous = triple
                time.sleep(0.02)
            return seen
        finally:
            running[0] = False
            thread.join(timeout=5)

    bare_changes = changes_seen(bare)
    call_changes = changes_seen(calling)

    assert call_changes >= 3, (
        f"a call-heavy loop showed only {call_changes} frame changes in 60 looks, so "
        f"the observation cannot see the projection walk it was added for"
    )
    assert bare_changes <= 4, (
        f"a call-free loop showed {bare_changes} frame changes, which contradicts the "
        f"measured property this design (and every rig here) is built on"
    )
    assert call_changes > bare_changes


#: A synchronous stretch, as a child: no ``await`` and no beat FROM THE WORKLOAD LOOP,
#: with a body long enough that the frame this thread is in differs between any two
#: looks. A HEALTHY SERVING PLANE runs beside it, which is the shape the real fires
#: have: a beat on one plane re-arms the timer toward the other's deadline (the two
#: most recent dumps came due at 2.28 s and 12.29 s rather than at the bound, for
#: exactly that reason), so the deadline in play here is the STARVED plane's and the
#: abstention is able to move it. A child with no serving tick would fire on the
#: never-engaged plane instead -- a different case, and a correct fire.
#: ``in-flight`` is the progress leg's own second clause, and the pair of cells below
#: adds it and removes it to separate the two ways this process can be spared.
_EXECUTING_CHILD = """
import os
import pathlib
import sys
import threading
import time

from local_operator.session.runtime import stall_watchdog

bound = float(sys.argv[1])
sentinel = pathlib.Path(sys.argv[2])
in_flight = sys.argv[3] == "in-flight"
serving_started = threading.Event()


def stretch(seconds):
    # A CALL per iteration, not bare arithmetic, and that is load-bearing rather than
    # decorative: `frame.f_lineno` on a FOREIGN frame is refreshed at calls, so a body
    # of pure `total += 1` reads as one frozen line however fast it runs (measured 0 of
    # 60 reads changing) and this child would be testing the park case instead. The
    # projection walk this models is all calls.
    end = time.monotonic() + seconds
    total = 0
    while time.monotonic() < end:
        total += leaf(1)
        total += leaf(2)
        total += leaf(3)
        total += leaf(4)
        total += leaf(5)
        total += leaf(6)
        total += leaf(7)
        total += leaf(8)
    return total


def leaf(value):
    return value + 1


probe = (lambda: ("still", True)) if in_flight else (lambda: ("still", False))


def busy():
    # Motion is not ownership: only the dedicated exit probe may hold a live step.
    return in_flight


assert stall_watchdog.arm(seconds=bound, probe=probe, busy=busy), (
    "the child could not arm the bound"
)
print(f"armed:{os.getpid()}", flush=True)


def serving_tick():
    # A HEALTHY SERVING PLANE: a separate thread, reporting on its own cadence, so the
    # timer is re-armed toward the WORKLOAD plane's deadline rather than never being
    # re-armed at all. The first stamp releases the baseline snapshot below.
    while True:
        time.sleep(bound / 10)
        stall_watchdog.beat(stall_watchdog.SERVING)
        serving_started.set()


# The stamp a starved tick leaves behind, and what teaches the bound WHICH thread's
# frame to watch. Without it the plane has no loop thread and cannot be observed at
# all, which is the behaviour the never-engaged cells already pin.
stall_watchdog.beat(stall_watchdog.WORKLOAD)
threading.Thread(target=serving_tick, name="serving-tick", daemon=True).start()
assert serving_started.wait(), "the serving plane did not publish its first stamp"
# Snapshot the REAL arm deadline after the serving beat has refreshed its plane: the
# older workload stamp must remain the pin until executing frames earn an extension.
# This gives the parent a structural before/after value even when a legitimate held
# C-timer observation is present in the dump.
deadline_path = stall_watchdog.deadline_path(os.getpid())
sentinel.with_name("deadline-before.txt").write_text(
    deadline_path.read_text(encoding="utf-8"), encoding="utf-8"
)
print(f"stretch:{stretch(bound * 3)}", flush=True)
sentinel.write_text("the stretch returned", encoding="utf-8")
sys.stdout.flush()
# Exit WITHOUT disarming, so the artifact survives as the thing this cell reads: the
# bound never fired, so nothing removed it (`os._exit` skips the clean-exit unlink).
os._exit(0)
"""


def test_an_executing_loop_with_a_step_in_flight_is_not_cut(tmp_path: Path) -> None:
    """Prove executing workload frames extend the deadline while the child completes.

    The child returns with rc 0 and publishes its sentinel. It snapshots the
    workload-pinned deadline after the serving plane's first beat, then verifies
    moving WORKLOAD frames advance that deadline and appear in ``executing_planes``.
    This test does not assert that the timer fired; the separate idle-to-admission
    subprocess regression covers real dump-only C-timer expiry and process survival.
    """
    sentinel = tmp_path / "returned.txt"
    result = _run_script(
        _EXECUTING_CHILD,
        tmp_path,
        args=(str(CHILD_BOUND_S), str(sentinel), "in-flight"),
    )

    assert result.returncode == 0, (
        f"the bound cut a loop that was executing with a step in flight: {result.stdout!r} "
        f"{result.stderr!r}"
    )
    assert sentinel.is_file(), "the stretch did not finish"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(tmp_path, pid)
    text = dump.read_text(encoding="utf-8")
    before_epoch_text, before_leg = (
        (tmp_path / "deadline-before.txt").read_text(encoding="utf-8").split()
    )
    before_epoch = float(before_epoch_text)
    after_epoch, after_leg = _deadline_record(tmp_path / "logs", pid)
    assert before_leg == stall_watchdog.WORKLOAD, (
        f"the baseline was not pinned by the initial workload stamp: "
        f"{before_epoch_text} {before_leg}; {text[:2000]!r}"
    )
    assert after_epoch > before_epoch + 0.001, (
        "the executing workload did not advance the real armed deadline: "
        f"before={before_epoch_text} {before_leg}, after={after_epoch:.3f} {after_leg}; "
        f"{text[:2000]!r}"
    )
    assert stall_watchdog.WORKLOAD in stall_watchdog.executing_planes(
        pid, tmp_path / "logs"
    ), f"the dump does not record the workload loop executing: {text!r}"


def test_the_executing_loop_hands_the_runaway_to_the_progress_leg(tmp_path: Path) -> None:
    """THE BOUNDING ARGUMENT, EXERCISED: diagnosis hands over without ending work.

    Same child, one argument different: no step is in flight, so the progress leg has
    a run to judge and fires on it — while the liveness leg abstains. The executing
    and no-progress markers show which observations produced the dump; diagnostic
    expiry still does not end the child.
    """
    sentinel = tmp_path / "returned.txt"
    result = _run_script(
        _EXECUTING_CHILD,
        tmp_path,
        args=(str(CHILD_BOUND_S), str(sentinel), "no-step"),
    )

    assert (
        result.returncode == 0
    ), f"the diagnostic timer ended the workload: {result.stdout!r} {result.stderr!r}"
    assert sentinel.read_text(encoding="utf-8") == "the stretch returned"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert (
        stall_watchdog.PROGRESS_MARKER in text
    ), f"the progress leg did not fire on a frames-moving runaway: {text[:2000]!r}"
    assert stall_watchdog.WORKLOAD in stall_watchdog.executing_planes(
        pid, tmp_path / "logs"
    ), f"the liveness leg did not record the executing loop: {text[:2000]!r}"


#: A park that RELEASES the GIL, so the sampler keeps looking at a frame that never
#: changes. This is the cell that catches an abstention which fires on "we have a
#: frame" rather than on "the frame MOVED" — the difference between the halves of
#: this section.
_FROZEN_WAIT_CHILD = """
import os
import pathlib
import sys
import threading
import time

from local_operator.session.runtime import stall_watchdog

bound = float(sys.argv[1])
assert stall_watchdog.arm(seconds=bound, probe=lambda: ("still", False)), (
    "the child could not arm the bound"
)
print(f"armed:{os.getpid()}", flush=True)


def serving_tick():
    # A HEALTHY SERVING PLANE, so the ONLY silent plane is the frozen WORKLOAD one.
    # Without it the bound would fire on the never-engaged serving plane's deadline
    # and this cell would pass even if the abstention wrongly rescued a frozen frame.
    while True:
        time.sleep(bound / 10)
        stall_watchdog.beat(stall_watchdog.SERVING)


threading.Thread(target=serving_tick, name="serving-tick", daemon=True).start()
stall_watchdog.beat(stall_watchdog.WORKLOAD)

# PARKED, WITH THE SAMPLER STILL ALIVE: `Event.wait` releases the GIL, so the
# sampler keeps sampling while this frame stays still. A real stall does not return
# after a diagnostic dump, so the parent reaps the child once it sees that dump.
threading.Event().wait(600)
pathlib.Path(sys.argv[2]).write_text("wait returned", encoding="utf-8")
"""


def test_a_frozen_frame_is_dumped_while_the_sampler_keeps_sampling(tmp_path: Path) -> None:
    """A frozen frame is dumped without turning diagnosis into retirement.

    ``_PARKED_CHILD`` (the GIL-holding park) covers the case where nothing in the
    process can run; this covers the case where the sampler is running normally and
    must nonetheless DECLINE to abstain, because a frame that has not moved is the
    reading the bound was written for. A plane silent for the bound with a still frame
    is the one state that is indistinguishable from the 2026-09-20 death from inside
    the process, and it still produces diagnostic evidence at the deadline.
    """
    sentinel = tmp_path / "frozen-wait-returned.txt"
    result = _run_stalled_script(
        _FROZEN_WAIT_CHILD, tmp_path, args=(str(CHILD_BOUND_S), str(sentinel))
    )

    assert result.returncode is None, (
        f"the native timer did not fire while the frozen frame was parked: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert not sentinel.exists(), "the frozen frame returned before child cleanup"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:2000]
    assert (
        stall_watchdog.executing_planes(pid, tmp_path / "logs") == ()
    ), f"a frozen frame was recorded as an executing loop: {text[:2000]!r}"


#: The 2026-09-20 shape in mechanism: a C call that holds the GIL for its whole
#: duration, the stand-in for ``_sre_SRE_Pattern_search``. ``ctypes.PyDLL`` is the
#: binding that does NOT release the GIL, so no Python thread in this process --
#: the sampler included -- runs again, and the serving plane cannot report either.
#: That is the shape's own property rather than a choice made here, and it is why
#: this cell is the weaker of the two frozen ones: no implementation could rescue it,
#: so what it guards is the ABSTENTION not lifting a hard park (`_FROZEN_WAIT_CHILD`
#: is the cell that can fail on a wrong implementation).
_FROZEN_MATCHER_CHILD = """
import ctypes
import os
import sys

from local_operator.session.runtime import stall_watchdog

bound = float(sys.argv[1])
assert stall_watchdog.arm(seconds=bound, probe=lambda: ("still", True)), (
    "the child could not arm the bound"
)
print(f"armed:{os.getpid()}", flush=True)
stall_watchdog.beat(stall_watchdog.WORKLOAD)

lib = ctypes.PyDLL(None)
lib.sleep.argtypes = [ctypes.c_uint]
lib.sleep(600)
sys.stdout.write("the parked call returned\\n")
"""


def test_a_frame_frozen_in_a_c_matcher_is_dumped(tmp_path: Path) -> None:
    """THE INCIDENT CELL: the frozen frame still fires with the abstraction in place.

    The 2026-09-20 incident was 100% of the loop inside one C-level matcher for
    1.5-7.2 h. Both legs abstain for different reasons here — no stamp can arrive and
    the frame cannot move, so the liveness leg has nothing to extend; and a step is in
    flight, so the progress leg has nothing to judge — which is exactly why the bound
    has to fire on the stamp alone.
    """
    result = _run_stalled_script(_FROZEN_MATCHER_CHILD, tmp_path, args=(str(SHORT_BOUND_S),))

    assert result.returncode is None, (
        f"the native timer did not fire while the matcher held the GIL: "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert "the parked call returned" not in result.stdout
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, text[:2000]
    assert stall_watchdog.executing_planes(pid, tmp_path / "logs") == ()


#: A loop parked in its own selector, waiting on a child that never reports, with the
#: workload tick running on the same loop — the manager-awaiting-subagents shape the
#: operator asked about. NOT ``_AWAITING_CHILD`` above, which is the pre-existing
#: long-await cell's rig: that one models the loop going quiet, this one models it
#: waiting while the tick keeps running, and they are different questions.
_AWAITING_AND_RUNNING_CHILD = """
import asyncio
import os
import pathlib
import sys

from local_operator.session.runtime import stall_watchdog

bound = float(sys.argv[1])
sentinel = pathlib.Path(sys.argv[2])
assert stall_watchdog.arm(seconds=bound, probe=lambda: ("still", False)), (
    "the child could not arm the bound"
)
print(f"armed:{os.getpid()}", flush=True)


async def beat_loop():
    # The workload plane's tick, driven the way the runtime drives it: an asyncio
    # task on this loop, woken by its own sleep. BOTH planes, because one silent
    # plane is what the bound exists to catch -- a child reporting for the workload
    # alone would be cut for the serving plane's silence and would be testing that
    # instead of this.
    while True:
        await asyncio.sleep(0.15)
        stall_watchdog.beat(stall_watchdog.WORKLOAD)
        stall_watchdog.beat(stall_watchdog.SERVING)


async def waiting_on_a_child():
    # A child that never reports: the loop's own frame is parked in the selector for
    # the whole wait, and this process burns no CPU at all.
    await asyncio.Event().wait()


async def main():
    asyncio.ensure_future(beat_loop())
    asyncio.ensure_future(waiting_on_a_child())
    await asyncio.sleep(bound * 3)


asyncio.run(main())
sentinel.write_text("the wait returned", encoding="utf-8")
print("survived", flush=True)
stall_watchdog.disarm()
"""


def test_a_loop_awaiting_a_child_keeps_beating_and_is_never_cut(tmp_path: Path) -> None:
    """THE OPERATOR'S CASE, answered: waiting on subagents is not a stall.

    A manager's loop parked for three times the bound on a child that never reports,
    burning no CPU, with its own frame still for the whole wait — it returns because
    an ``await`` hands control back, so the workload tick is not starved. This models
    the manager/subagent wait case: lack of activity alone is not a stall.

    A loop that cannot run at all still gets a diagnostic dump (see the frozen-frame
    cases below); neither case authorizes the native timer to end the process.
    """
    sentinel = tmp_path / "returned.txt"
    result = _run_script(
        _AWAITING_AND_RUNNING_CHILD,
        tmp_path,
        args=(str(CHILD_BOUND_S), str(sentinel)),
    )

    assert result.returncode == 0, (
        f"the manager did not finish its wait: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert sentinel.read_text(encoding="utf-8") == "the wait returned"
    assert "survived" in result.stdout, result.stdout
    assert stall_watchdog.fired_pids(tmp_path / "logs") == set()


def test_a_blocked_probe_does_not_hold_the_lock_a_beat_needs(tmp_path: Path) -> None:
    """THE SECONDARY DEFECT'S REGRESSION CELL: the probe must not queue a stamp.

    ``beat`` takes the same module lock the sampler does, and the production probe
    reads LIVE session state from a foreign thread — transcript footprints, job rows,
    a context another thread is appending to — so it can be slow for as long as that
    state is contended. Holding the lock across it makes every stamp wait on a
    diagnostic read, which turns a slow probe into a silent plane and a silent plane
    into a killed runtime. THIS CELL BLOCKS: it drives the real sampler through
    ``arm``, lets it enter a probe that waits on an Event, and requires a ``beat`` from
    another thread to complete while that probe is still blocked. On the pre-fix code
    the beat cannot complete until the probe returns, so the assertion is red there —
    which is the only reason it is worth having.
    """
    spy = _FakeFaulthandler()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(stall_watchdog, "faulthandler", spy)
    entered, released, complete = (threading.Event() for _ in range(3))
    worker: threading.Thread | None = None
    try:

        def probe() -> tuple[object, bool]:
            # The shape the real one has when it is slow: it reached the state it
            # needs and is waiting for it, with the lock NOT held around this.
            #
            # SEVERAL TIMES THE STAMP'S DEADLINE, and that margin is the cell's whole
            # reliability: with both waits at 10 s the probe releases itself at the
            # same instant the stamp's own wait expires, so whichever reaches the wire
            # first decides a coin flip. Measured against the literal pre-fix module:
            # 2 false greens in 8 runs, and 3 of 3 against an equivalent hand-written
            # mutant. The pre-fix shape has the beat queued behind a lock this call
            # holds, so it now loses deterministically instead of sometimes.
            entered.set()
            released.wait(60)
            return "still", True

        assert stall_watchdog.arm(
            seconds=1.0, directory=tmp_path, pid=4246, probe=probe
        ), "the child could not arm the bound"
        assert entered.wait(10), "the sampler never reached the probe"

        def stamp() -> None:
            stall_watchdog.beat(stall_watchdog.WORKLOAD)
            complete.set()

        worker = threading.Thread(target=stamp, name="stamp-while-probing", daemon=True)
        worker.start()
        assert complete.wait(10), (
            "a beat could not complete while the probe was running, so the probe is "
            "holding the lock that stamps a plane"
        )
    finally:
        released.set()
        if worker is not None:
            worker.join(timeout=10)
        stall_watchdog.disarm()
        monkeypatch.undo()


# -- the EXIT LEG: a fire is not a death when work is in flight ----------------
#
# The act this section separates was one act: the timer dumped every thread and
# ``_exit``ed. The operator's rule for both the bound and a build move is that a
# runtime is replaced when its turn is COMPLETE — never on a heuristic of
# inactivity — and the fleet said the two acts were not the same event: of 40
# dumps on this host 37 carry the fired marker, and the event-loop thread's own
# stack in those fires is a runtime DOING WORK (``session._run_turn`` ->
# ``_emit`` -> ``serving._refresh_state``) more often than a runtime that
# stopped. Production always keeps the diagnostic dump and the runtime alive,
# regardless of its sampled work state; the idle and post-clear controls below pin
# that fail-closed contract rather than expecting the old fatal exit.

#: The child that arms with an EXIT LEG the way the runtime's entry point does —
#: a probe injected from outside the module — and then parks in a GIL-RELEASING
#: sleep. The sleep is deliberate and is the opposite choice from the PyDLL park
#: above: a held fire is recorded by the SAMPLER, and a park that holds the GIL
#: would stop the one thread the recording depends on. ``argv``: the sentinel it
#: writes when it stops on its own terms, the bound, a flag file the probe reads
#: so the answer can be driven from the test, the probe variant, and optionally
#: the second at which it clears the flag (the work finishing under it).
_EXIT_LEG_CHILD = """
import os
import pathlib
import sys
import time

from local_operator.session.runtime import stall_watchdog

sentinel = pathlib.Path(sys.argv[1])
bound = float(sys.argv[2])
flag = pathlib.Path(sys.argv[3])
variant = sys.argv[4]
release_after = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
stop_after = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
wait_for = sys.argv[7] if len(sys.argv) > 7 else ""


def busy():
    return flag.exists()


def unreadable():
    # A probe that cannot answer — the state a bug in the predicate would leave.
    # A comment and not a docstring: this child is a string inside the test file, so a
    # nested triple quote would end it early (measured once already).
    raise RuntimeError("the work report could not be read")


if variant == "busy":
    armed = stall_watchdog.arm(seconds=bound, busy=busy)
elif variant == "raises":
    armed = stall_watchdog.arm(seconds=bound, busy=unreadable)
elif variant == "idle":
    armed = stall_watchdog.arm(seconds=bound, busy=lambda: False)
else:
    armed = stall_watchdog.arm(seconds=bound)
assert armed, "the child could not arm the bound"
print(f"armed:{os.getpid()}", flush=True)

started = time.monotonic()
while True:
    time.sleep(0.05)
    age = time.monotonic() - started
    if release_after and age >= release_after:
        flag.unlink()
        release_after = 0.0
    if stop_after and age >= stop_after:
        if wait_for:
            # A BOUNDED WAIT FOR THE READING LINE the parent asserts on: the sampler
            # writes it one wake after the fire, so a child that stopped on a clock
            # alone would race that assertion on a loaded host. Failing loudly is the
            # point — a cell that cannot see the line must not pass by timing.
            reading_deadline = time.monotonic() + 5
            while time.monotonic() < reading_deadline:
                if wait_for in stall_watchdog.dump_path().read_text(encoding="utf-8"):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError(f"the sampler never wrote {wait_for!r}")
        # The clean exit a live runtime takes: it disarms, and the artifact of the
        # fire it survived has to survive that (see ``stall_watchdog.disarm``).
        stall_watchdog.disarm()
        sentinel.write_text("stopped on its own terms", encoding="utf-8")
        sys.exit(0)
"""


_DUMP_ONLY_IDLE_CHILD = """
import os
import pathlib
import sys
import time

from local_operator.session.runtime import stall_watchdog

sentinel = pathlib.Path(sys.argv[1])
flag = pathlib.Path(sys.argv[2])
sampled_idle = pathlib.Path(sys.argv[3])
bound = float(sys.argv[4])
admit = sys.argv[5] == "admit"
calls = 0

def busy():
    global calls
    calls += 1
    is_busy = flag.exists()
    # The first call seeds _Armed; this line proves the sampler itself observed
    # False before the test advances either case to the timer fire.
    if calls > 1 and not is_busy:
        sampled_idle.write_text("sampled idle", encoding="utf-8")
    return is_busy

assert stall_watchdog.arm(seconds=bound, busy=busy), "the child did not arm"
print(f"armed:{os.getpid()}", flush=True)
dump = stall_watchdog.dump_path()
deadline = time.monotonic() + bound * 5
while time.monotonic() < deadline:
    if sampled_idle.exists() and stall_watchdog.FIRED_MARKER in dump.read_text(encoding="utf-8"):
        break
    time.sleep(0.02)
else:
    raise AssertionError("the idle sample or the C-timer dump never arrived")
if admit:
    # Admission deliberately follows both the sampler's False and its native
    # timer fire; that stale observation must not terminate the new work.
    flag.write_text("work admitted", encoding="utf-8")
    held_deadline = time.monotonic() + bound * 5
    while time.monotonic() < held_deadline:
        if stall_watchdog.HELD_MARKER in dump.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("the sampler did not observe post-fire work admission")
    sentinel.write_text("admitted", encoding="utf-8")
else:
    # Wait for the line the sampler appends after the fire, so the parent's
    # assertions read a SETTLED artifact rather than racing the sampler: the marker
    # is written by the Python thread that observed the fire, one wake after it
    # landed.
    reading_deadline = time.monotonic() + bound * 5
    while time.monotonic() < reading_deadline:
        if stall_watchdog.OBSERVED_MARKER in dump.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("the sampler never recorded the idle fire")
    sentinel.write_text("idle survived", encoding="utf-8")
"""


def test_a_dump_only_idle_fire_survives_later_work_admission(tmp_path: Path) -> None:
    """A real idle fire is diagnostic; both idle and later work survive it."""
    for mode, expected in (("idle", "idle survived"), ("admit", "admitted")):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        sentinel = run_dir / "sentinel.txt"
        flag = run_dir / "busy.flag"
        sampled_idle = run_dir / "sampled-idle.txt"
        result = _run_script(
            _DUMP_ONLY_IDLE_CHILD,
            run_dir,
            args=(
                str(sentinel),
                str(flag),
                str(sampled_idle),
                str(SHORT_BOUND_S),
                mode,
            ),
        )
        assert result.returncode == 0, (
            f"the native timer ended the {mode} child: rc={result.returncode} "
            f"{result.stdout!r} {result.stderr!r}"
        )
        assert sampled_idle.read_text(encoding="utf-8") == "sampled idle"
        assert sentinel.read_text(encoding="utf-8") == expected
        pid = int(result.stdout.split("armed:", 1)[1].split()[0])
        text = _dump_for(run_dir, pid).read_text(encoding="utf-8")
        assert stall_watchdog.FIRED_MARKER in text, f"no C-timer dump for {mode}: {text!r}"
        if mode == "idle":
            # NOTHING WAS IN FLIGHT, so the live stalled state must stay OFF while the
            # fire stays ON the record (finding B).
            assert any(
                line.startswith(stall_watchdog.OBSERVED_MARKER) for line in text.splitlines()
            ), f"the idle fire left no observation line: {text[:900]!r}"
            assert not any(
                line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
            ), f"an idle fire raised the live held state: {text[:900]!r}"
            assert stall_watchdog.held_fire(pid, run_dir / "logs") is False
            assert pid not in stall_watchdog.held_pids(run_dir / "logs")
        else:
            # Work was admitted AFTER the fire, and the fire that lands over it — the
            # one the re-arm after the flip is for — carries the held reading.
            assert any(
                line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
            ), f"the fire over admitted work was not recorded as held: {text[:900]!r}"
            assert stall_watchdog.held_fire(pid, run_dir / "logs") is True


def test_a_fire_with_work_in_flight_dumps_and_the_runtime_SURVIVES(tmp_path: Path) -> None:
    """THE CELL THE OPERATOR'S RULE RESTS ON, driven through a real process.

    A child arms with a busy probe answering True, parks in a GIL-releasing sleep,
    and a bound later its bound fires. What must be true afterwards is the whole
    change: the process is ALIVE (so the sleep finished and the sentinel was
    written on its own terms), the dump exists with the fired marker, the dump
    says the fire did NOT end it, and ``held_fire`` — the reader a surface uses —
    answers True for the pid.

    THE ARTIFACT OUTLIVES THE CLEAN EXIT, asserted here rather than in the disarm
    cell alone: the child disarms on its way out, and a dump that vanished there
    would erase the only record of a stall that a person still has to resolve.
    """
    sentinel = tmp_path / "sentinel.txt"
    flag = tmp_path / "busy.flag"
    flag.write_text("work in flight", encoding="utf-8")

    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        args=(str(sentinel), str(SHORT_BOUND_S), str(flag), "busy", "0", "4"),
    )

    assert result.returncode == 0, (
        f"the bound ended a runtime that reported work in flight: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert sentinel.read_text(encoding="utf-8") == "stopped on its own terms"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert text.count(stall_watchdog.FIRED_MARKER) >= 1, (
        f"the bound never fired, so this cell proves nothing about a fire it survived: "
        f"{text[:600]!r}"
    )
    assert any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"the dump does not say the fire was held: {text[:900]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is True


def test_an_UNREADABLE_work_report_still_fires_and_holds(tmp_path: Path) -> None:
    """THE INVARIANT: no path withholds the fire. Only the exit is ever refused.

    The sibling failure this protects against is an abstention — a bound that stops
    bounding, so a genuinely wedged runtime never gets its evidence and never surfaces.
    That is worse than the cut it replaced, and it is the shape a "hold the exit"
    change invites: re-arm the timer away, or skip the fire, when the predicate cannot
    be evaluated.

    So this is the surviving cell's child with a probe that RAISES, and what is
    asserted is that the bound fired anyway: the dump exists with the fired marker, it
    says the fire was held, and the process is alive to have written its sentinel.
    ``_holds_work`` answers True for an unreadable report, and ``process._busy_probe``
    answers the same way for each of its three unknown cases — no handle yet, a handle
    with no probe, a probe that raised — so the decision is always "we cannot prove
    this runtime is idle", and the EVIDENCE is never a function of that answer.
    """
    sentinel = tmp_path / "sentinel.txt"
    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        args=(str(sentinel), str(SHORT_BOUND_S), "unused", "raises", "0", "4"),
    )

    assert result.returncode == 0, (
        f"an unreadable work report ended the runtime, which is the cut this change "
        f"removes: rc={result.returncode} {result.stdout!r} {result.stderr!r}"
    )
    assert sentinel.read_text(encoding="utf-8") == "stopped on its own terms"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert (
        text.count(stall_watchdog.FIRED_MARKER) >= 1
    ), f"the fire was WITHHELD when the predicate could not be read: {text[:600]!r}"
    assert any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"the dump does not say the fire was held: {text[:900]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is True


def test_the_same_child_with_NO_work_in_flight_is_dumped_and_survives(
    tmp_path: Path,
) -> None:
    """An idle-looking fire is diagnostic only; the live pid is not retired."""
    sentinel = tmp_path / "sentinel.txt"
    flag = tmp_path / "busy.flag"  # never created: the probe answers False

    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        args=(str(sentinel), str(SHORT_BOUND_S), str(flag), "idle", "0", "4"),
        timeout=60.0,
    )

    assert result.returncode == 0, (
        f"the native timer ended an apparently idle runtime: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert sentinel.read_text(encoding="utf-8") == "stopped on its own terms"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert stall_watchdog.FIRED_MARKER in text, "the child survived without a real timer dump"
    # THE FIRE WAS OVER NOTHING (finding B): the reading this dump earns is the neutral
    # observation, and the long-standing cell here asserted the HELD marker instead —
    # which is exactly the permanent false "stalled with work in flight" a listing then
    # rendered for a runtime with nothing in flight.
    assert any(
        line.startswith(stall_watchdog.OBSERVED_MARKER) for line in text.splitlines()
    ), f"the idle fire was not recorded as an observation: {text[:900]!r}"
    assert not any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"an idle fire set the live held state: {text[:900]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is False
    assert pid not in stall_watchdog.held_pids(tmp_path / "logs")


def test_the_exit_leg_follows_the_work_from_one_fire_to_the_next(tmp_path: Path) -> None:
    """A cleared sampled hold does not turn a later production fire fatal.

    The child starts with work in flight, survives the first fire, then clears its
    flag. The sampler observes the new state and later fires remain diagnostic-only;
    the child exits on its own after publishing its completion sentinel.
    """
    sentinel = tmp_path / "sentinel.txt"
    flag = tmp_path / "busy.flag"
    flag.write_text("work in flight", encoding="utf-8")

    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        # Work clears after the first fire; the child is still responsible for
        # stopping itself later, since neither timer fire may terminate it.
        args=(
            str(sentinel),
            str(SHORT_BOUND_S),
            str(flag),
            "busy",
            "1.5",
            "4",
            stall_watchdog.OBSERVED_MARKER,
        ),
        timeout=60.0,
    )

    assert result.returncode == 0, (
        f"a later timer fire ended the child after work cleared: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert sentinel.read_text(encoding="utf-8") == "stopped on its own terms"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"no surviving fire was recorded while work was in flight: {text[:900]!r}"
    # ...AND THE LATER ONE, OVER CLEARED WORK, IS AN OBSERVATION (finding B): the two
    # readings have to be tellable apart in the SAME artifact, which is the shape a
    # person meets when a runtime recovers and then wedges again.
    assert any(
        line.startswith(stall_watchdog.OBSERVED_MARKER) for line in text.splitlines()
    ), f"the fire over cleared work was not recorded as an observation: {text[:900]!r}"
    assert (
        text.count(stall_watchdog.FIRED_MARKER) >= 2
    ), f"no later dump was recorded after work cleared: {text[:600]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is True


def test_a_caller_with_no_busy_probe_keeps_dump_only_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent probe does not turn diagnostic expiry into process termination.

    Reduced callers without a probe still receive all-thread dumps; production
    never authorizes native termination from a sampled work state.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [(5.0, False)], fake.armed


def test_a_busy_probe_that_raises_keeps_dump_only_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable answer cannot affect the dump-only timer policy.

    The native timer never consumes sampled work state, because no sample can
    synchronize with every work-admission path.
    """

    def exploding() -> bool:
        raise RuntimeError("the session's busy state could not be read")

    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    assert stall_watchdog.arm(seconds=5.0, busy=exploding, directory=tmp_path) is True
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [(5.0, False)], fake.armed


def test_a_held_fire_is_read_off_the_marker_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``held_fire`` distinguishes the THREE states a fired dump can now be in.

    A reader that answered off the fired line alone would call a runtime that
    survived its own bound "gone", which is the operator's question answered
    backwards. The three files here are the three states, and the fourth case —
    the marker without a fire — must NOT read as held, because a file that says
    "held" without saying "fired" describes a process that never reached its bound
    (the header is written at arm time, which is exactly why no marker of ours is
    spelled there).
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    fired_without_held = logs / f"{stall_watchdog.DUMP_PREFIX}-11.log"
    fired_without_held.write_text(f"{stall_watchdog.FIRED_MARKER}0:00:05)!\n", encoding="utf-8")
    fired_then_held = logs / f"{stall_watchdog.DUMP_PREFIX}-22.log"
    fired_then_held.write_text(
        f"{stall_watchdog.FIRED_MARKER}0:00:05)!\n{stall_watchdog.HELD_MARKER}it did not end\n",
        encoding="utf-8",
    )
    held_without_fire = logs / f"{stall_watchdog.DUMP_PREFIX}-33.log"
    held_without_fire.write_text(f"{stall_watchdog.HELD_MARKER}it did not end\n", encoding="utf-8")

    assert stall_watchdog.held_fire(11, logs) is False
    assert stall_watchdog.held_fire(22, logs) is True
    assert stall_watchdog.held_fire(33, logs) is False
    assert stall_watchdog.held_fire(44, logs) is False, "no dump is not a held fire"


def test_a_beat_records_a_fire_that_landed_while_it_was_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The beat path records a held fire too, through the PUBLIC seam.

    A fire can land between two wakes of the sampler, and the beat that follows is
    an ordinary re-arm — so this is the cell that pins the ordering inside
    ``_rearm``: record first, arm second. Reversed, the re-arm would reset the
    size baseline over the fire it should have annotated, and the runtime would then
    be re-armed from a deadline already in the past.

    The fire is written by hand because the real C timer cannot be asked to fire
    inside this in-process state-machine test; what is exercised here is everything
    that happens AFTER it.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    assert stall_watchdog.arm(seconds=5.0, busy=lambda: True, directory=tmp_path) is True
    pid = os.getpid()
    dump = stall_watchdog.dump_path(pid, tmp_path)

    with dump.open("a", encoding="utf-8") as handle:
        handle.write(f"{stall_watchdog.FIRED_MARKER}0:00:05)!\n  File x, line 1\n")

    stall_watchdog.beat(stall_watchdog.SERVING)

    text = dump.read_text(encoding="utf-8")
    assert any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"the beat re-armed over the fire without recording it: {text[:800]!r}"
    assert text.index(stall_watchdog.FIRED_MARKER) < text.index(
        stall_watchdog.HELD_MARKER
    ), "the marker landed above the fire, so a reader cannot tie the two together"
    assert stall_watchdog.held_fire(pid, tmp_path) is True
    # Both files survive the clean exit that this arm's own teardown performs.
    stall_watchdog.disarm()
    assert dump.exists(), "the artifact of a fire this runtime survived was erased"
    assert stall_watchdog.deadline_path(pid, tmp_path).exists()


def test_held_pids_is_the_subset_of_fired_pids_that_survived(tmp_path: Path) -> None:
    """The third state as a SET, because a listing is where it is read.

    ``fired_pids`` answers "whose bound fired"; ``held_pids`` answers "whose bound
    fired and did NOT end them", and the two are nested by construction. A surface
    that has only the first cannot tell a post-mortem from a session that is still
    holding work and needs a ``lop stop``, which is the operator's question answered
    backwards — so the nesting is asserted rather than assumed.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"{stall_watchdog.DUMP_PREFIX}-11.log").write_text(
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n", encoding="utf-8"
    )
    (logs / f"{stall_watchdog.DUMP_PREFIX}-22.log").write_text(
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n{stall_watchdog.HELD_MARKER}held\n",
        encoding="utf-8",
    )
    # A header-only file is an armed runtime, not a fire, and it must not leak in.
    (logs / f"{stall_watchdog.DUMP_PREFIX}-33.log").write_text(
        f"{stall_watchdog.ARM_MARKER}armed\n{stall_watchdog.HELD_MARKER}held\n", encoding="utf-8"
    )

    assert stall_watchdog.fired_pids(logs) == {11, 22}
    assert stall_watchdog.held_pids(logs) == {22}
    assert stall_watchdog.held_pids(logs) <= stall_watchdog.fired_pids(logs)


def test_the_held_fire_backoff_can_never_shorten_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-1 (round 2): the cap is a CEILING, never a floor.

    ``min(bound * 2**n, MAX)`` with a bound ABOVE the cap returns the cap — so an
    operator who configured a two-hour bound got a "backoff" of one hour, i.e. a bound
    that fires twice as often as they asked for. The doubling is meant to stretch the
    interval from the bound it starts at, so the arithmetic is
    ``max(bound, min(bound * 2**n, MAX))``.

    Built on the REAL ``_Armed`` rather than a stand-in (agent review round 3, and CI's
    ``type-check``): a duck-typed stub is not what this module hands ``_rearm``, and the
    deadline it computes from is part of what the branch reads. ``time.monotonic`` is
    pinned far ahead of the stamps so the interval this branch computes is what lands,
    rather than the ordinary deadline arithmetic.
    """
    captured: list[float] = []
    monkeypatch.setattr(
        stall_watchdog,
        "_arm_timer",
        lambda handle, remaining, *, exit_leg: captured.append(remaining),
    )
    monkeypatch.setattr(stall_watchdog, "_record_held_fire", lambda armed: False)
    monkeypatch.setattr(stall_watchdog, "_dump_size", lambda path: 0)

    dump, handle = _observed_dump(tmp_path, 4246)
    armed = stall_watchdog._Armed(dump, handle, 7200.0, 4246)
    armed.held = True
    armed.held_fires = 0
    monkeypatch.setattr(stall_watchdog.time, "monotonic", lambda: 10_000.0)
    armed.after_fire_at = 10_000.0  # the fire is NOW, so only the interval decides

    stall_watchdog._rearm(armed)
    assert captured, "no timer was armed at all"
    assert (
        captured[-1] >= 7200.0
    ), f"a two-hour bound was re-armed for {captured[-1]}s: the cap shortened it"


def test_the_backoff_counter_is_per_EPISODE_not_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-1 (round 2), second half: the hold clearing ends the episode.

    A lifetime counter meant a runtime that survived a fire, cleared its work and wedged
    AGAIN later re-armed at the backed-off interval — up to twelve times the configured
    bound — and a diagnostic-only arm would delay the next dump after recovery by the
    same factor. ``_apply_exit_leg`` is the flip the sampler makes when the work clears
    with no fire pending, so that is where the episode ends.
    """
    monkeypatch.setattr(stall_watchdog, "_arm_timer", lambda handle, remaining, *, exit_leg: None)
    monkeypatch.setattr(stall_watchdog, "_record_held_fire", lambda armed: False)

    dump, handle = _observed_dump(tmp_path, 4247)
    armed = stall_watchdog._Armed(dump, handle, 300.0, 4247)
    armed.held = True
    armed.held_fires = 3

    stall_watchdog._apply_exit_leg(armed, False)
    assert armed.held is False
    assert armed.held_fires == 0, (
        "the backoff counter survived the episode, so a later wedge re-arms at up to "
        "twelve times the configured bound"
    )


def test_a_marker_that_landed_MID_LINE_still_reads_as_held(tmp_path: Path) -> None:
    """Q-4 (round 2): the marker can interleave with faulthandler's own flush.

    ``faulthandler`` writes the dump from its own thread through a buffered handle while
    the sampler appends our marker to the same ``O_APPEND`` descriptor, so the marker
    lands at the end of whatever faulthandler had written but not flushed — MID-LINE
    (measured on this fleet: one of four real fires, in exactly the shape built below).
    A line-start test read that genuine held dump as "not held", which through the
    product's own readers means no STALLED cell, ``stall_held: false`` beside a set
    ``stall_dump``, and ``death_verdict`` narrating a still-alive runtime as one that
    ended ITSELF. The module's own header states the rule this restores: markers are
    read as SUBSTRINGS, which is why no header quotes one.

    RESTORED BY THE FOLLOW-UP PR, and the reason is disclosure rather than repair. A later
    commit on the #1439 branch dropped this cell and NEITHER that commit NOR the round's
    remediation comment said so, so the reader pair a surface actually calls lost its only
    coverage while the verdict-level case that survived
    (``test_a_fired_stall_bound_is_narrated_as_its_own_class``) kept the product path
    guarded. It passes as written on the merged head, which is what a restored cell should
    do; what it discriminates is the GRANULARITY -- ``held_fire`` and ``held_pids`` over an
    interleaved dump, which is the pair ``journal`` reads -- and it goes red the moment the
    reader is degraded back to a line-start test.
    """
    from local_operator.session.runtime import stall_watchdog

    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pid = 987_654
    dump = logs / f"{stall_watchdog.DUMP_PREFIX}-{pid}.log"
    # The measured interleaving: faulthandler's unflushed tail, then our marker.
    dump.write_text(
        "[stall watchdog] armed for 300s\n"
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n"
        "Thread 0x1 (most recent call first, thread id=1):\n"
        '  File "/tmp/x.py", line 1 in <module>\n'
        f'  File "{stall_watchdog.HELD_MARKER}the bound fired at 1.0 and did NOT end this '
        "runtime\\n",
        encoding="utf-8",
    )

    assert stall_watchdog.held_fire(pid, logs) is True, (
        "a held marker that landed mid-line read as no held fire, so a surviving runtime "
        "is reported as one the bound ended"
    )
    assert pid in stall_watchdog.held_pids(
        logs
    ), "the third state vanished from the listing for an interleaved dump"


def test_the_executing_extension_takes_the_exit_leg_from_the_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executing-loop extension also goes through the dump-only wrapper.

    The spy checks the real faulthandler call so this exceptional deadline-extension
    path cannot bypass production's unconditional ``exit=False`` policy.
    """
    captured: list[bool] = []
    monkeypatch.setattr(
        stall_watchdog.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, file, exit: captured.append(exit),
    )
    dump, handle = _observed_dump(tmp_path, 4248)
    armed = stall_watchdog._Armed(dump, handle, 300.0, 4248, lambda: ("still", True))
    armed.last_beat[stall_watchdog.WORKLOAD] = 100.0
    armed.last_beat[stall_watchdog.SERVING] = 9_999.0
    armed.held = True

    stall_watchdog._extend_for_execution(armed, 500.0, (stall_watchdog.WORKLOAD,))
    assert captured == [
        False
    ], f"the extension armed a native timer that may terminate a runtime: exit={captured}"

    # ...and the other direction: even when the caller labels the arm idle, the
    # single native timer wrapper remains dump-only.
    armed.held = False
    stall_watchdog._extend_for_execution(armed, 501.0, (stall_watchdog.WORKLOAD,))
    assert captured[-1] is False, "an IDLE arm must not authorize a native exit"


# ============================================================================
# THE FOUR FINDINGS OF THE 2026-09-23 CONVERGENCE ROUND, PINNED TOGETHER
# ============================================================================
# Each finding exists because a READER was inferring from an artifact's SILENCE: which
# policy the build that wrote it followed (A), whether work was in flight when the
# bound fired (B), whether this bound still exits a process (C), and which phase the
# process had reached when it fired (D). So every cell below asserts a POSITIVE
# statement — a marker, a header sentence, or an ordering — and each names the false
# reading it replaces, because the false reading is what the stored artifacts say to
# the next person who opens them.

#: A LEGACY (pre-dump-only) header, as the builds on ``main`` wrote it: the sentence
#: that states an idle fire takes a native exit. Written out here so the cells that
#: claim a watchdog-caused death use the artifact that can support one.
LEGACY_FATAL_HEADER = (
    f"{stall_watchdog.ARM_MARKER}pid 1 armed for 300s at 1.0\n"
    f"{stall_watchdog.LEGACY_FATAL_POLICY_PHRASES[1]}\n"
)

#: A current header, as THIS build writes it.
CURRENT_HEADER = (
    f"{stall_watchdog.ARM_MARKER}pid 1 armed for 300s at 1.0\n{stall_watchdog.DUMP_POLICY_MARKER}\n"
)

#: One fire, with a stack line under it: enough for every reader here.
FIRED_BODY = f"{stall_watchdog.FIRED_MARKER}0:05:00)!\n  File x, line 1\n"


def _write_dump(logs: Path, pid: int, text: str) -> Path:
    logs.mkdir(exist_ok=True)
    path = logs / f"{stall_watchdog.DUMP_PREFIX}-{pid}.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_fire_outcome_attributes_only_what_the_artifact_states(tmp_path: Path) -> None:
    """FINDING A: a fire is a watchdog death only where the artifact says it could be.

    The reader this replaces answered "did the bound end this runtime?" from the ABSENCE
    of a post-fire marker, which is true of the legacy fatal builds — an idle fire there
    exited the process before anything could append a line — and false of every
    dump-only one, where a GIL-held fire cannot append anything either. The table below
    is the whole new contract, and the third and fourth rows are the ones the old
    inference got wrong in opposite directions.
    """
    logs = tmp_path / "logs"
    held = f"{stall_watchdog.HELD_MARKER}the bound fired and did NOT end this runtime\n"
    observed = f"{stall_watchdog.OBSERVED_MARKER}the bound fired and did NOT end this runtime\n"
    cases = {
        # A DUMP-ONLY BUILD'S FIRE ENDED NOTHING, marker or no marker.
        11: (CURRENT_HEADER + FIRED_BODY, stall_watchdog.FIRE_SURVIVED),
        12: (CURRENT_HEADER + FIRED_BODY + observed, stall_watchdog.FIRE_SURVIVED),
        13: (CURRENT_HEADER + FIRED_BODY + held, stall_watchdog.FIRE_SURVIVED),
        # A LEGACY BUILD'S UNHELD FIRE IS THE ONE DEATH THIS READER MAY CLAIM.
        14: (LEGACY_FATAL_HEADER + FIRED_BODY, stall_watchdog.FIRE_LEGACY_FATAL),
        15: (
            LEGACY_FATAL_HEADER + FIRED_BODY + held,
            stall_watchdog.FIRE_SURVIVED,
        ),
        # ...AND A FIRE WHOSE ARTIFACT PROVES NEITHER CONVENTION PROVES NOTHING: the
        # absent marker is not evidence, which is the reading this whole cell restores.
        16: (
            f"{stall_watchdog.ARM_MARKER}pid 1 armed for 300s at 1.0\n" + FIRED_BODY,
            stall_watchdog.FIRE_UNKNOWN,
        ),
        # NO FIRE AT ALL — an armed header a SIGKILL left, a header with only the
        # annotation, and a file that does not exist.
        17: (CURRENT_HEADER, stall_watchdog.FIRE_UNKNOWN),
        18: (
            CURRENT_HEADER + f"{stall_watchdog.TEARDOWN_MARKER}{stall_watchdog.TEARDOWN_NOTE}\n",
            stall_watchdog.FIRE_UNKNOWN,
        ),
    }
    for pid, (text, expected) in cases.items():
        _write_dump(logs, pid, text)
        assert stall_watchdog.fire_outcome(pid, logs) == expected, (pid, expected, text)
    assert stall_watchdog.fire_outcome(19, logs) == stall_watchdog.FIRE_UNKNOWN

    # AND THE HELD MARKER STILL WINS ON A LEGACY ARTIFACT, which is what the check
    # order in ``fire_outcome`` is for: row 15 carries the fatal sentence AND the held
    # line, and the held line is the one that describes what happened.
    assert stall_watchdog.held_fire(15, logs) is True
    assert stall_watchdog.fire_outcome(15, logs) != stall_watchdog.FIRE_LEGACY_FATAL


def test_the_reading_a_fire_earns_is_the_only_thing_that_sets_the_held_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """FINDING B: the neutral reading must not raise the live ``stall_held`` state.

    ``held_fire``/``held_pids`` answer "work was in flight and this runtime is still
    stalled" — the state a listing renders as needing a person — and until this split
    the sampler appended ``HELD_MARKER`` for EVERY fire it observed, so an idle fire (or
    one whose work had cleared, or one over a GIL-held loop) left that state set for the
    life of the dump. The fire is still recorded; what changed is which line records it.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    monkeypatch.setattr(stall_watchdog, "_start_sampler", lambda armed: None)
    pid = os.getpid()
    try:
        # (a) NOTHING IN FLIGHT: an observation, and the live state stays OFF.
        assert stall_watchdog.arm(seconds=5.0, busy=lambda: False, directory=tmp_path) is True
        dump = stall_watchdog.dump_path(pid, tmp_path)
        with dump.open("a", encoding="utf-8") as handle:
            handle.write(FIRED_BODY)
        with caplog.at_level(logging.WARNING, logger=stall_watchdog.logger.name):
            stall_watchdog.beat(stall_watchdog.SERVING)
        records = [
            record
            for record in caplog.records
            if record.name == stall_watchdog.logger.name
            and "did NOT end this runtime" in record.getMessage()
        ]
        text = dump.read_text(encoding="utf-8")
        assert any(
            line.startswith(stall_watchdog.OBSERVED_MARKER) for line in text.splitlines()
        ), f"the idle fire left no observation line: {text[:800]!r}"
        assert stall_watchdog.HELD_MARKER not in text, (
            "an idle fire was recorded as held, which is the permanent false "
            f"'stalled with work in flight' this split exists to prevent: {text[:800]!r}"
        )
        assert stall_watchdog.held_fire(pid, tmp_path) is False
        assert pid not in stall_watchdog.held_pids(tmp_path)
        # ...and it is still a fire: the observation is not a suppression.
        assert pid in stall_watchdog.fired_pids(tmp_path)
        assert stall_watchdog.fire_outcome(pid, tmp_path) == stall_watchdog.FIRE_SURVIVED
        assert any("no work in flight" in record.getMessage() for record in records), (
            "the log line beside the listing still says STALLED for a fire over nothing: "
            f"{[r.getMessage() for r in records]}"
        )

        # (b) WORK IN FLIGHT: the same seam raises it, so the split did not simply
        # switch the state off for everyone.
        stall_watchdog.disarm()
        assert stall_watchdog.arm(seconds=5.0, busy=lambda: True, directory=tmp_path) is True
        dump = stall_watchdog.dump_path(pid, tmp_path)
        with dump.open("a", encoding="utf-8") as handle:
            handle.write(FIRED_BODY)
        stall_watchdog.beat(stall_watchdog.SERVING)
        text = dump.read_text(encoding="utf-8")
        assert any(
            line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
        ), f"a fire over work in flight lost its held reading: {text[:800]!r}"
        assert stall_watchdog.held_fire(pid, tmp_path) is True
        assert pid in stall_watchdog.held_pids(tmp_path)
    finally:
        stall_watchdog.disarm()


def test_no_production_timer_site_can_arm_a_fatal_expiry() -> None:
    """FINDING C: the dump-only policy, as a check rather than as a promise.

    Two halves, both against the SOURCE, because the sentence that shipped a fatal
    binding was a literal at a call site: the module's only native call passes
    ``exit=False``, and no production call anywhere passes ``exit_leg=True`` or
    ``exit=True``. The behavioural half — a real child that survives its bound — is
    asserted in the cells that drive ``_PARKED_CHILD`` and ``_EXIT_LEG_CHILD``.
    """
    checked = 0
    for path in sorted((REPO / "local_operator").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "dump_traceback_later" not in source and "_arm_timer(" not in source:
            continue
        checked += 1
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in {"exit", "exit_leg"}:
                    continue
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    raise AssertionError(
                        f"a production timer site arms a fatal expiry: "
                        f"{path}:{node.lineno} {ast.unparse(node)}"
                    )
    assert checked >= 1, "no production timer site found: the bound is inert"

    tree = ast.parse(
        (REPO / "local_operator" / "session" / "runtime" / "stall_watchdog.py").read_text(
            encoding="utf-8"
        )
    )
    native = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dump_traceback_later"
    ]
    assert native, "the module no longer arms a native timer at all"
    for node in native:
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        assert "exit" in keywords and isinstance(keywords["exit"], ast.Constant), ast.unparse(node)
        assert (
            keywords["exit"].value is False
        ), f"the native timer is no longer dump-only: {ast.unparse(node)}"


#: The sentences that SHIPPED as promises this build cannot keep, as substrings rather
#: than the bare words: a message MAY say this bound ends nothing — that is the policy —
#: and a check on the words alone would forbid the correction along with the defect. Each
#: entry is a fragment of a line that really was written into a dump or a log.
STALE_EXIT_PHRASES = (
    "and ends the runtime",
    "holds the exit",
    "ENDED this runtime",
    "is dumped and exited",
    "the exit faulthandler takes",
    "armed with exit=True",
    "_exit(1)",
    "hit its bound and is gone",
)


def _assert_no_stale_exit_claim(name: str, text: str) -> None:
    """Both halves of the stale-claim guard, for ONE surface.

    Shared rather than inlined so the SAME assertion covers every bound shape the
    arming can take (agent review round 3, MINOR-3): the cell used to arm once, with
    the default two-bound split, so the equal-bounds sentence was never observed and a
    stale fragment appended to it left the guard green.
    """
    for phrase in STALE_EXIT_PHRASES:
        assert (
            phrase not in text
        ), f"{name} still asserts an exit this build cannot take: {phrase!r}"
    for legacy in stall_watchdog.LEGACY_FATAL_POLICY_PHRASES:
        assert legacy not in text, (
            f"{name} spells a LEGACY policy sentence, which would make every dump "
            f"this build writes read as a fatal-build artifact: {legacy!r}"
        )


def test_no_runtime_message_claims_the_bound_ends_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """FINDING C, second half: the words the runtime writes must match the policy.

    Three surfaces, because each one is read by somebody who cannot check the code: the
    dump header (an operator opening the file after a freeze), the arm-time log line
    (the first thing they read in ``runtime.log``), and the two header constants that
    are rendered into every dump. The phrases below are the ones that SHIPPED as stale
    promises; the legacy sentences are asserted absent from this build's own output, so
    a future header cannot accidentally make every dump read as a legacy-fatal artifact
    to :func:`fire_outcome`.

    BOTH BOUND SHAPES ARE DRIVEN, because ``arm`` writes a DIFFERENT sentence for each
    and the guard is only as wide as the surfaces it looks at (agent review round 3,
    MINOR-3). ``arm(seconds=…)`` sets the boot and steady bounds to one number and takes
    the single-bound line; the default arming takes the two-bound line. Measured on the
    head this cell was reviewed at: appending the shipped fragment ``"and ends the "
    "runtime"`` to the single-bound sentence left the cell GREEN, because the default
    arming it used never rendered that sentence. Both shapes are asserted now, so the
    same edit goes red.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    shapes = {
        # The default arming: a boot bound that differs from the steady one.
        "two bounds (the default arming)": {"directory": tmp_path},
        # ``seconds`` alone is a statement about this process's WHOLE arming, so it
        # stands as both bounds and the runtime announces the one-bound sentence.
        "one bound (equal boot and steady)": {
            "seconds": stall_watchdog.DEFAULT_STALL_S,
            "directory": tmp_path,
        },
    }
    try:
        for label, kwargs in shapes.items():
            assert stall_watchdog.arm(**kwargs) is True
            try:
                header = stall_watchdog.dump_path(os.getpid(), tmp_path).read_text(encoding="utf-8")
                caplog.clear()
                with caplog.at_level(logging.INFO, logger=stall_watchdog.logger.name):
                    stall_watchdog.announce()
                logged = " ".join(
                    record.getMessage()
                    for record in caplog.records
                    if record.name == stall_watchdog.logger.name
                )
                assert logged, f"{label}: the arming said nothing at all"
                surfaces = {
                    "the dump header": header,
                    "the arm-time log line": logged,
                    "OBSERVATION_NOT_VERDICT": stall_watchdog.OBSERVATION_NOT_VERDICT,
                    "HOW_TO_READ_THE_FIRED_VALUE": stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE,
                    "DUMP_ONLY_STATEMENT": stall_watchdog.DUMP_ONLY_STATEMENT,
                }
                for name, text in surfaces.items():
                    _assert_no_stale_exit_claim(f"{label}: {name}", text)
                assert (
                    stall_watchdog.DUMP_POLICY_MARKER in header
                ), f"the header does not state this build's policy: {header[:400]!r}"
                assert "dumps every thread" in logged, logged
                # ...AND THE LINE SAYS WHAT IT IS: the policy, in the one word both
                # bound-shapes share. The stale-phrase scan above is the guard that
                # matters; this is the positive half, so a line rewritten to say nothing
                # at all cannot pass.
                assert "dump-only" in logged, logged
            finally:
                stall_watchdog.disarm()
    finally:
        stall_watchdog.disarm()


def test_the_teardown_annotation_is_read_off_the_ordering_alone(tmp_path: Path) -> None:
    """FINDING D: the phase is classified by POSITION, and ambiguity stays unknown.

    The annotation is written before ``asyncio.Runner`` cancels tasks and joins the
    default executor, so a fire BELOW it was taken during teardown and one ABOVE it was
    not. Everything else — a marker that is present but not above the fire, a marker that
    landed mid-line because both writers share one buffered descriptor, an artifact with
    no marker at all — is reported as unknown rather than inferred, which is what keeps a
    GIL-held stall (which can never reach the annotation) from being filed as teardown.
    """
    logs = tmp_path / "logs"
    note = f"{stall_watchdog.TEARDOWN_MARKER}{stall_watchdog.TEARDOWN_NOTE}\n"
    cases = {
        11: (CURRENT_HEADER + note + FIRED_BODY, True),
        12: (CURRENT_HEADER + FIRED_BODY + note, None),
        13: (
            CURRENT_HEADER
            + f'  File "{stall_watchdog.TEARDOWN_MARKER}{stall_watchdog.TEARDOWN_NOTE}\n'
            + FIRED_BODY,
            None,
        ),
        14: (CURRENT_HEADER + FIRED_BODY, None),
        15: (CURRENT_HEADER + note, None),
        # SEVERAL FIRES AROUND ONE ANNOTATION, which is the shape a dump-only build can
        # really leave: a fire ends nothing, so a runtime can take one, return from
        # ``amain``, be annotated, and then take the teardown fire in the executor join.
        # The annotation stands ABOVE the fire it belongs to and BELOW an earlier one, so
        # a reader keyed on the FIRST fire cannot claim the class its own comment names
        # (agent review round 3, MINOR-1). Every fire below the annotation answers True,
        # the last one is what the ordering is asked about, and a dump whose fires ALL
        # sit above the annotation is still unknown.
        16: (CURRENT_HEADER + note + FIRED_BODY + FIRED_BODY, True),
        17: (CURRENT_HEADER + FIRED_BODY + note + FIRED_BODY, True),
        18: (CURRENT_HEADER + FIRED_BODY + FIRED_BODY + note, None),
    }
    for pid, (text, expected) in cases.items():
        _write_dump(logs, pid, text)
        assert stall_watchdog.fired_in_runner_teardown(pid, logs) is expected, (pid, text)
    # A pid with NO artifact at all is unknown too, and it is the last case rather
    # than the first because 17 above is now a real multi-fire row.
    assert stall_watchdog.fired_in_runner_teardown(99, logs) is None


def test_the_teardown_annotation_is_a_no_op_when_unarmed_and_idempotent_when_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINDING D, second cell: the annotation changes nothing but the artifact.

    ``note_runner_teardown`` runs on the runtime's shutdown path, so it must be free for
    every caller that never armed a bound (the in-process hosts, the TUI, this whole
    suite) and it must be once-per-arming: it names a PHASE, and a phase is entered once.
    The rest of the cell pins that it did not become a second timer policy — the native
    arms stay ``exit=False``, a fire over work in flight still records the held reading,
    and ``disarm`` still removes an un-fired dump.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)
    monkeypatch.setattr(stall_watchdog, "_start_sampler", lambda armed: None)

    # (a) NOTHING ARMED: no-op, and no artifact is created to annotate.
    assert stall_watchdog.note_runner_teardown() is False
    assert not list(tmp_path.glob(f"{stall_watchdog.DUMP_PREFIX}-*.log"))

    pid = os.getpid()
    try:
        assert stall_watchdog.arm(seconds=5.0, busy=lambda: True, directory=tmp_path) is True
        assert stall_watchdog.note_runner_teardown() is True
        assert stall_watchdog.note_runner_teardown() is False, "the phase was annotated twice"
        assert stall_watchdog._ARMED is not None
        assert stall_watchdog._ARMED.runner_teardown.is_set() is True
        dump = stall_watchdog.dump_path(pid, tmp_path)
        text = dump.read_text(encoding="utf-8")
        assert text.count(stall_watchdog.TEARDOWN_MARKER) == 1, text
        assert [armed[1] for armed in fake.armed] == [False], fake.armed

        # (b) A LATER FIRE SITS BELOW THE ANNOTATION, which is the production shape and
        # the reason the classifier can answer at all.
        with dump.open("a", encoding="utf-8") as handle:
            handle.write(FIRED_BODY)
        stall_watchdog.beat(stall_watchdog.SERVING)
        text = dump.read_text(encoding="utf-8")
        assert stall_watchdog.fired_in_runner_teardown(pid, tmp_path) is True
        assert any(
            line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
        ), f"the annotation changed how a fire is recorded: {text[:900]!r}"
        assert stall_watchdog.held_fire(pid, tmp_path) is True
        # EVERY arm this run took is dump-only, including the re-arm the fire itself
        # performed; the count is not asserted because the re-arm's own interval is the
        # backoff's business, not this cell's.
        assert fake.armed and all(armed[1] is False for armed in fake.armed), fake.armed
        # The artifact of a fire the runtime survived still outlives the clean exit.
        stall_watchdog.disarm()
        assert dump.exists()
    finally:
        stall_watchdog.disarm()

    # (c) ...and with no fire, the clean exit still removes the dump.
    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    quiet = stall_watchdog.dump_path(pid, tmp_path)
    assert quiet.exists()
    assert stall_watchdog.note_runner_teardown() is True
    stall_watchdog.disarm()
    assert not quiet.exists(), "an annotated run that never fired left a body on disk"


_EXECUTOR_PARKED_CHILD = """
import asyncio
import os
import pathlib
import sys
import time

from local_operator.session.runtime import process, stall_watchdog

release = pathlib.Path(sys.argv[1])
parked = pathlib.Path(sys.argv[2])
done = pathlib.Path(sys.argv[3])
bound = float(sys.argv[4])

# ARMED INSIDE THE COROUTINE, so the annotation is not racing the timer: this cell is
# about the ORDER of the annotation and the fire, and an arm taken before the body ran
# could fire before the annotation on a loaded host. The bound still starts before the
# executor join, which is the phase under test.
def worker() -> str:
    parked.write_text("parked in the default executor", encoding="utf-8")
    deadline = time.monotonic() + 300
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return "released"


async def body(operator_cap: bytes | None = None) -> int:
    assert stall_watchdog.arm(seconds=bound, busy=lambda: False), "the child could not arm"
    print(f"armed:{os.getpid()}", flush=True)
    # The future is deliberately dropped: the work is the executor thread itself, and
    # what this cell needs is a runner whose teardown has something to join.
    asyncio.get_running_loop().run_in_executor(None, worker)
    return 0


process.amain = body
rc = asyncio.run(process._run_amain(None))
done.write_text(f"clean exit {rc}", encoding="utf-8")
"""


def test_a_real_child_annotates_the_runner_teardown_before_the_executor_join(
    tmp_path: Path,
) -> None:
    """FINDING D, third cell: the whole mechanism on a real process.

    A child arms a short bound and parks a default-executor worker on a file only the
    parent can release, so ``asyncio.run`` returns and its Runner then blocks in
    ``shutdown_default_executor`` -> ``_do_shutdown`` -> ``Thread.join``. What must be
    true is the whole finding: the bound fires and the process SURVIVES it, the dump
    carries the annotation ABOVE the fire, the parked join is what the stacks show, and
    releasing the worker lets the child finish cleanly on its own terms — which is what
    makes this an annotation of a phase rather than a claim about a death.
    """
    release = tmp_path / "release.txt"
    parked = tmp_path / "parked.txt"
    done = tmp_path / "done.txt"
    script = tmp_path / "parked_child.py"
    script.write_text(_EXECUTOR_PARKED_CHILD, encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(script), str(release), str(parked), str(done), str(SHORT_BOUND_S)],
        env=_child_env(tmp_path),
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        dump = _dump_for(tmp_path, proc.pid)
        deadline = time.monotonic() + 60.0
        fired = False
        settled = False
        while time.monotonic() < deadline:
            assert proc.poll() is None, (
                "the child left before its bound fired, so nothing was parked in the "
                f"executor join: rc={proc.returncode}"
            )
            if parked.exists() and dump.is_file():
                text = dump.read_text(encoding="utf-8")
                if stall_watchdog.FIRED_MARKER in text:
                    fired = True
                    # THE MARKER LINE LANDS BEFORE THE STACKS: ``faulthandler`` writes
                    # the fired value, then walks every thread. Reading at the first
                    # sight of the marker would look at a half-written dump, so this
                    # waits for the frame the annotation exists to explain.
                    if "_do_shutdown" in text or "shutdown_default_executor" in text:
                        settled = True
                        break
            time.sleep(0.02)
        assert fired, (
            "the bound never fired while the executor join was parked: "
            f"parked={parked.exists()} dump={dump.is_file()}"
        )
        assert settled, "the dump never grew the executor-join stacks the fire interrupted"
        text = dump.read_text(encoding="utf-8")
        assert (
            stall_watchdog.TEARDOWN_MARKER in text
        ), f"the runner's own teardown was not annotated: {text[:600]!r}"
        assert text.index(stall_watchdog.TEARDOWN_MARKER) < text.index(
            stall_watchdog.FIRED_MARKER
        ), "the annotation landed below the fire, so it cannot describe that fire"
        assert "_do_shutdown" in text or "shutdown_default_executor" in text, (
            "the stacks do not show the executor join this annotation exists to explain: "
            f"{text[:1200]!r}"
        )
        assert not done.exists(), "the child completed before the worker was released"
        assert stall_watchdog.fired_in_runner_teardown(proc.pid, tmp_path / "logs") is True
        assert (
            stall_watchdog.fire_outcome(proc.pid, tmp_path / "logs") == stall_watchdog.FIRE_SURVIVED
        ), "a dump-only fire was not read as survived by the production reader"

        # RELEASE, and let it finish on its own terms: nothing here kills it, and the
        # exit code is the proof that the fire ended nothing.
        release.write_text("the parent releases the worker", encoding="utf-8")
        assert proc.wait(timeout=120.0) == 0, (
            f"the annotated child did not finish cleanly: rc={proc.returncode} "
            f"{proc.stderr.read() if proc.stderr else ''}"
        )
        assert done.read_text(encoding="utf-8") == "clean exit 0"
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


_GIL_PARKED_RUNNER_CHILD = """
import asyncio
import ctypes
import os
import sys

from local_operator.session.runtime import process, stall_watchdog


def park_the_loop_deliberately() -> None:
    lib = ctypes.PyDLL(None)
    lib.sleep.argtypes = [ctypes.c_uint]
    lib.sleep(600)


async def body(operator_cap: bytes | None = None) -> int:
    assert stall_watchdog.arm(seconds=float(sys.argv[1]), busy=lambda: False), "could not arm"
    print(f"armed:{os.getpid()}", flush=True)
    park_the_loop_deliberately()
    return 0


process.amain = body
asyncio.run(process._run_amain(None))
"""


def _run_parked_child(
    script: str, work_dir: Path, args: tuple[str, ...], settle: str
) -> tuple[str, int]:
    """Spawn a GIL-parked child, wait for its fire AND its stacks, then reap it.

    ``_run_stalled_script`` reaps as soon as ``FIRED_MARKER`` appears, which is while
    ``faulthandler`` is still walking every thread — so a cell that asserts on a STACK
    can read a dump truncated mid-write, and would fail for a reason that has nothing to
    do with the behaviour under test (measured: it did, under fleet load). This waits
    for the frame it is about while the child is still alive, then reaps only the
    process group it created.
    """
    path = work_dir / "parked_child.py"
    path.write_text(script, encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(path), *args],
        env=_child_env(work_dir),
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    dump = _dump_for(work_dir, proc.pid)
    deadline = time.monotonic() + 45.0
    text = ""
    try:
        while time.monotonic() < deadline:
            assert (
                proc.poll() is None
            ), f"the parked child left before its dump settled: rc={proc.returncode}"
            if dump.is_file():
                text = dump.read_text(encoding="utf-8")
                if stall_watchdog.FIRED_MARKER in text and settle in text:
                    break
            time.sleep(0.02)
        else:
            raise AssertionError(f"the dump never carried {settle!r} beside a fire: {text[:800]!r}")
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    return text, proc.pid


def test_a_GIL_parked_child_is_never_annotated_as_teardown(tmp_path: Path) -> None:
    """FINDING D, fourth cell: the phase a GIL-held stall never reaches.

    The annotation is written by the wrapper's ``finally``, so a loop parked inside
    ``amain`` with the GIL held can never get it — the coroutine does not return. The
    fire is still an observation on a dump-only build; what it must NOT be is an
    attributed teardown, because a reader that inferred one from a parked stack would be
    inventing the phase it cannot see.
    """
    text, pid = _run_parked_child(
        _GIL_PARKED_RUNNER_CHILD,
        tmp_path,
        (str(SHORT_BOUND_S),),
        "park_the_loop_deliberately",
    )
    assert stall_watchdog.FIRED_MARKER in text, text[:400]
    assert "park_the_loop_deliberately" in text, text[:800]
    assert stall_watchdog.TEARDOWN_MARKER not in text, (
        "a phase marker was written for a process that never left ``amain``: " f"{text[:800]!r}"
    )
    assert stall_watchdog.fired_in_runner_teardown(pid, tmp_path / "logs") is None
    assert (
        stall_watchdog.fire_outcome(pid, tmp_path / "logs") == stall_watchdog.FIRE_SURVIVED
    ), "the GIL-held fire lost its honest reading while its phase stayed unknown"
