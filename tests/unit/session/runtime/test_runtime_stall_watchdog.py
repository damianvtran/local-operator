"""The runtime bounds its OWN stall: a C-thread dump, then a departure.

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
import inspect
import logging
import os
import re
import subprocess
import sys
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

    A real timer cannot be fired safely inside a pytest worker — firing it
    ``_exit``s the worker — and asking it whether it is armed is not something
    ``faulthandler`` answers. So the structure is spied here and the FIRING is
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
    """Run ``script`` as a REAL file, so the dump names a path a reader can open.

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
#: point calls it, an optional single beat, then a park in a GIL-holding C call so
#: the C timer really fires and the process really leaves. A FIRE IS WHAT LEAVES THE
#: TWO FILES BEHIND (its exit path never disarms), which is what the next life of the
#: pid then has to inherit.
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
"""


def test_a_stalled_loop_is_dumped_and_the_process_leaves(tmp_path: Path) -> None:
    """The reproduction: a parked loop names itself and the runtime leaves.

    What is asserted, and why each half is needed:

    * the process LEFT — rc 1 and the resumed-sentinel absent. rc alone could be
      a crash; the sentinel is what separates "the bound fired" from "the call
      returned and a later assertion failed".
    * the dump NAMES THE PARKED FRAME with its source path — the line nothing in
      the process could report before this (all five frozen runtimes needed
      ``sample`` from outside, and one of them was reaped by hand 6.9 h later).
    * the header PRECEDES the fired marker, which is write-then-act as a fact
      about the file rather than as a promise.
    * the LOCAL SENTINEL is absent — the property that makes this dump safe to
      write into a directory that also holds prompts.
    """
    resumed = tmp_path / "resumed.txt"
    result = _run_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(resumed), str(CHILD_BOUND_S)),
    )

    assert result.returncode == 1, f"the bound did not fire: {result.stdout!r} {result.stderr!r}"
    assert not resumed.exists(), "the parked call resumed; the bound fired too late to matter"
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

    THE MUTATION THIS CELL EXISTS TO CATCH, and it is the reason a presence assert
    is not enough on its own: drop ``{OBSERVATION_NOT_VERDICT}`` from the header in
    ``arm`` and this cell -- and nothing else in the file -- goes red. A cell that
    cannot fail would be the same class of thing as the header it pins.
    """
    resumed = tmp_path / "resumed.txt"
    result = _run_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(resumed), str(CHILD_BOUND_S)),
    )
    assert result.returncode == 1, f"the bound did not fire: {result.stdout!r} {result.stderr!r}"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")

    assert stall_watchdog.FIRED_MARKER in text, "this cell is about a dump that fired"
    assert stall_watchdog.OBSERVATION_NOT_VERDICT in text, (
        "a fired dump does not say what a fire IS, so a reader is left to read a watchdog "
        f"timer expiry as a death: {text[:400]!r}"
    )
    assert "ENDED this runtime" not in text, (
        "the header asserts the verdict a fire never computes; a fire can precede a process "
        f"that carries on, which is the incident this wording caused: {text[:400]!r}"
    )


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
#: THE CONTRACT WITH THE PARENT, not a decoration: a stamp has to land STRICTLY
#: inside the bound, because the bound is what `_exit(1)`s this process.
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
# parent asserts on cannot be scheduled into the exit. The wait carries no timeout
# of its own and needs none: the bound armed above is this child's backstop, so a
# serving thread that somehow never ran ends the process with the very `_exit(1)`
# the parent's first assertion reads anyway (AGENTS.md, "Wait on the event, never
# on the clock" — this is that rule inside the child, where the event exists).
first_stamp.wait()

# The workload plane: busy in the matcher, and it never reports its progress.
subject = "credential-shaped text \u2603 x" * 20000
pattern = re.compile(r"(SECRET|DSN|Bearer)\s*=\s*\S+")
while True:
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
    result = _run_script(
        _TWO_PLANE_CHILD,
        tmp_path,
        args=(str(finished), str(SHORT_BOUND_S)),
    )

    assert result.returncode == 1, (
        f"the workload plane parked and the bound never fired: {result.stdout!r} "
        f"{result.stderr!r}"
    )
    assert not finished.exists(), "the busy workload plane returned; the bound fired too late"
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
# never-engaged class) and 19 a small one, 0.4-17 s (a beat's recomputed remainder,
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
    result = _run_script(
        _PARKED_CHILD,
        tmp_path,
        args=(str(tmp_path / "resumed.txt"), str(SHORT_BOUND_S)),
    )
    assert result.returncode == 1, f"the bound did not fire: {result.stdout!r} {result.stderr!r}"
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
    the timer, and not there when no beat ever did. A pid is RECYCLED and these files are
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
    """THE RECYCLE, END TO END: three real fires at one pid, two of them a life apart.

    The cell above proves ``arm`` removes the file; this one proves the reader's claim
    across process lifetimes, which is where the file count is a fact about a life rather
    than about a directory: a fire leaves the sibling behind (its exit path never disarms,
    for the reasons :func:`stall_watchdog.disarm` documents), the next life of the same pid
    starts with none, and the life after that -- which beats -- gets its own back. That is
    what makes presence and absence answerable about the pid in front of a reader.

    MUTATION THIS CELL CATCHES: drop the ``inherited.unlink()`` from ``arm``. Life 2 then
    fires the never-engaged value with life 1's sibling still on disk -- a file whose mtime
    precedes life 2's own arm epoch -- and the presence assertion goes red.
    """
    logs = tmp_path / "logs"
    dump = _dump_for(tmp_path, SYNTH_PID)
    sibling = stall_watchdog.deadline_path(SYNTH_PID, logs)

    first = _run_script(_RECYCLE_CHILD, tmp_path, args=(str(SYNTH_PID), str(CHILD_BOUND_S), "beat"))
    assert first.returncode == 1, f"life 1 did not fire: {first.stdout!r} {first.stderr!r}"
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

    second = _run_script(
        _RECYCLE_CHILD, tmp_path, args=(str(SYNTH_PID), str(CHILD_BOUND_S), "quiet")
    )
    assert second.returncode == 1, f"life 2 did not fire: {second.stdout!r} {second.stderr!r}"
    text = dump.read_text(encoding="utf-8")
    assert _fired_seconds(text) == pytest.approx(float(CHILD_BOUND_S)), (
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

    third = _run_script(_RECYCLE_CHILD, tmp_path, args=(str(SYNTH_PID), str(CHILD_BOUND_S), "beat"))
    assert third.returncode == 1, f"life 3 did not fire: {third.stdout!r} {third.stderr!r}"
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
        timer before patching this in: a stub would leave a 30 s ``exit=True`` timer
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
    result = _run_script(
        _TWO_PLANE_CHILD,
        tmp_path,
        args=(str(finished), str(SHORT_BOUND_S)),
    )
    assert result.returncode == 1, (
        f"the workload plane parked and the bound never fired: {result.stdout!r} "
        f"{result.stderr!r}"
    )
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
    no beat re-armed the timer (a runtime that never engaged), a smaller remainder when a
    beat recomputed it from the oldest plane's stamp — and that points the reader at the
    sibling, whose absence means the first of those and whose mtime is the last beat. It
    is written at ARM time, so it is present whether or not this runtime ever fires.

    MUTATION THIS CELL CATCHES: drop ``{HOW_TO_READ_THE_FIRED_VALUE}`` from the header in
    ``arm`` — the fired value goes back to being a number with no stated meaning.
    """
    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    try:
        text = stall_watchdog.dump_path(os.getpid(), tmp_path).read_text(encoding="utf-8")
    finally:
        stall_watchdog.disarm()

    assert stall_watchdog.HOW_TO_READ_THE_FIRED_VALUE in text, text[:600]
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
    while True:
        pattern.search(subject)


asyncio.run(main())
"""


def test_the_bound_fires_on_the_real_runtime_while_its_serving_plane_stays_healthy(
    tmp_path: Path,
) -> None:
    """Q2(c): the product's own two planes, one parked, and the bound fires.

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
    heartbeat re-armed the full bound every 15 s, so this child never left and the
    run had to be killed externally.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    result = _run_script(
        _REAL_TWO_PLANE_CHILD,
        config_dir,
        args=(
            str(CHILD_BOUND_S),
            str(config_dir),
            str(Path(__file__).resolve().parents[4]),
        ),
        timeout=120.0,
    )

    assert result.returncode == 1, (
        f"the parked workload plane did not trip the bound while the serving plane "
        f"was healthy: {result.stdout!r} {result.stderr!r}"
    )
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
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [(5.0, True)], fake.armed
    assert stall_watchdog.ARM_MARKER in fake.text_at_arm
    assert str(os.getpid()) in fake.text_at_arm
    assert fake.armed[0][2].read_text(encoding="utf-8") == fake.text_at_arm


def test_each_plane_is_tracked_apart_and_a_silent_one_shrinks_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy plane must NOT be able to keep a parked one alive.

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
        (10.0, True)
    ], "a second arm displaced the first instead of being a no-op"

    stall_watchdog.beat(stall_watchdog.SERVING)
    stall_watchdog.beat(stall_watchdog.SERVING)
    assert fake.cancels == 0, "a beat cancelled a live timer; the re-arm replaces it"
    armed_for = [seconds for seconds, _, _ in fake.armed]
    assert len(armed_for) == 3, armed_for
    assert (
        armed_for[1] < armed_for[0] and armed_for[2] < armed_for[0]
    ), f"the healthy plane re-armed for the FULL bound, so it can mask a silent one: {armed_for}"
    assert all(exit_ for _, exit_, _ in fake.armed), "the timer is not armed to exit"

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
    assert touched <= {"beat", "SERVING", "note_tick_death"}, (
        f"the serving plane reaches the watchdog for {touched}"
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

    # THE INCIDENT: yielding on every pass, so the ticker and the serving plane
    # keep their cadence and keep re-arming the timer, while burning CPU and
    # advancing no work at all. Nothing here is a double: the spin is real CPU.
    while True:
        await asyncio.sleep(0)
        sum(range(200_000))


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
        result.returncode == 1
    ), f"the spinning loop was not ended by the progress leg: {result.stdout!r} {result.stderr!r}"
    assert "both-planes-ran" in result.stdout, (
        f"the planes never ran, so this says nothing about a process whose loops were "
        f"ALIVE: {result.stdout!r} {result.stderr!r}"
    )

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

    def strftime(self, _fmt: str) -> str:
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
    # now, and `exit=True` — so the dump is written and the process leaves.
    assert spy.armed, "the progress fire never reached the C timer"
    assert spy.armed[-1][1] is True, "a progress fire must exit, like the liveness leg"
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
    # The behavioural twin of the source claim: arming with the real callable
    # installs THE REAL callable, which is what cell above this one drives.
    assert process_module._progress_probe.__module__ == process_module.__name__


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
    the bound set the way an operator sets it, and shows the leg FIRING there —
    `rc == 1`, the fired marker, the progress line, and ``fired_leg`` naming the
    progress leg. The negative is the same rig with ``probe=`` dropped inside the
    child: same spin, same bound, and nothing fires. Together they are the joint
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
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        assert child.poll() is not None, (
            f"the real runtime never left, so the leg did not fire in production:\n"
            f"{_log_text(config_dir)[-1500:]}"
        )
        assert child.returncode == 1, (
            f"the runtime left with rc={child.returncode}; the bound exits 1:\n"
            f"{_log_text(config_dir)[-1500:]}"
        )
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
#: FOUR of them, so the surviving run has to prove that a bound which demonstrably
#: ends the control child never ends this one.
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


def test_a_dead_workload_ticker_no_longer_takes_a_healthy_runtime_with_it(
    tmp_path: Path,
) -> None:
    """THE ACCEPTANCE CELL: the same rig, bare and supervised, on the real runtime.

    The control run is not decoration. ``bare`` is the pre-fix start site held
    verbatim, and it must still die -- if it ever stops dying, the supervised half
    below has stopped measuring anything, which is exactly the failure this file's
    "prove the test can still fail" rule is about. What the control shows on the
    way through is the defect's whole shape:

    * the workload stamp FREEZES and stays frozen past the bound (printed ages),
      while the serving plane keeps reporting every 200 ms -- a HEALTHY runtime;
    * the bound then kills the process (rc 1) on that frozen stamp;
    * the dump carries the fired marker and reads as the SILENCE leg, which is the
      wrong cause;
    * and ``tick_deaths`` is EMPTY, so nothing in the artifact separates this from
      a loop that genuinely parked.

    The supervised run is the same child with the production wiring, and it must
    survive four bounds: the stamp resumes, the death is in the log with its
    exception and in the dump as a named plane, and no bound fires at all.
    """
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir(parents=True, exist_ok=True)
    bare = _run_script(
        _DEAD_BEATER_CHILD,
        bare_dir,
        args=(str(DEAD_BEATER_BOUND_S), str(bare_dir), str(REPO), "bare"),
        timeout=180.0,
    )

    assert bare.returncode == 1, (
        f"the CONTROL no longer dies, so this cell cannot tell a fix from a blind rig: "
        f"rc={bare.returncode} stdout={bare.stdout!r} stderr={bare.stderr!r}"
    )
    assert "beater-raised" in bare.stdout, bare.stdout
    assert "survived" not in bare.stdout, "the control has to die for the pair to mean anything"
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
        timeout=180.0,
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


# -- the EXIT LEG: a fire is not a death when work is in flight ----------------
#
# The act this section separates was one act: the timer dumped every thread and
# ``_exit``ed. The operator's rule for both the bound and a build move is that a
# runtime is replaced when its turn is COMPLETE — never on a heuristic of
# inactivity — and the fleet said the two acts were not the same event: of 40
# dumps on this host 37 carry the fired marker, and the event-loop thread's own
# stack in those fires is a runtime DOING WORK (``session._run_turn`` ->
# ``_emit`` -> ``serving._refresh_state``) more often than a runtime that
# stopped. Keeping the dump and deciding the exit per re-arm is the change those
# cells pin; the wedge recovery for an IDLE runtime is asserted beside it, since
# it is the property this must not cost.

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
        # The clean exit a live runtime takes: it disarms, and the artifact of the
        # fire it survived has to survive that (see ``stall_watchdog.disarm``).
        stall_watchdog.disarm()
        sentinel.write_text("stopped on its own terms", encoding="utf-8")
        sys.exit(0)
"""


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
    assert text.count(stall_watchdog.FIRED_MARKER) >= 1, (
        f"the fire was WITHHELD when the predicate could not be read: {text[:600]!r}"
    )
    assert any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), f"the dump does not say the fire was held: {text[:900]!r}"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is True


def test_the_same_child_with_NO_work_in_flight_is_still_ended(tmp_path: Path) -> None:
    """THE WEDGE RECOVERY, and the CONTROL that makes the cell above mean something.

    Same rig, same bound, same park: the only difference is what the exit leg was
    told. An idle runtime keeps today's behaviour — the bound ends it — and a
    control that also survived would show the cell above was measuring the rig
    rather than the decision.
    """
    sentinel = tmp_path / "sentinel.txt"
    flag = tmp_path / "busy.flag"  # never created: the probe answers False

    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        args=(str(sentinel), str(SHORT_BOUND_S), str(flag), "idle"),
        timeout=60.0,
    )

    assert result.returncode == 1, (
        f"an idle runtime was not ended by its own bound: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    assert not sentinel.exists(), "the child reached its own exit; the bound did not end it"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert text.count(stall_watchdog.FIRED_MARKER) >= 1, text[:600]
    assert not any(
        line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()
    ), "a fatal fire wrote the held marker, so a reader would call an ended runtime stalled"
    assert stall_watchdog.held_fire(pid, tmp_path / "logs") is False


def test_the_exit_leg_follows_the_work_from_one_fire_to_the_next(tmp_path: Path) -> None:
    """The exit leg is a STATE, not a decision taken once at arm time.

    The child reports work in flight, survives its bound, and then CLEARS its flag
    — the work finished — with no re-arm of its own. The sampler re-reads the leg,
    and the next fire must be fatal: the runtime that has nothing in flight is
    exactly the runtime the bound exists to reap, so a hold that could not be
    lifted would trade a silent cut for a runtime nothing can reclaim.

    Both readings are asserted on ONE dump, which is why this cell is worth its
    seconds: the held marker is there (the first fire was held) and the process
    still died (the second was not).
    """
    sentinel = tmp_path / "sentinel.txt"
    flag = tmp_path / "busy.flag"
    flag.write_text("work in flight", encoding="utf-8")

    result = _run_script(
        _EXIT_LEG_CHILD,
        tmp_path,
        # Work clears at 2.5s; the bound is 1s, so the first fire is held and the
        # next one — after the leg flips — is not.
        args=(str(sentinel), str(SHORT_BOUND_S), str(flag), "busy", "2.5", "0"),
        timeout=60.0,
    )

    assert result.returncode == 1, (
        f"the runtime was never ended after its work finished: rc={result.returncode} "
        f"{result.stdout!r} {result.stderr!r}"
    )
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    text = _dump_for(tmp_path, pid).read_text(encoding="utf-8")
    assert any(line.startswith(stall_watchdog.HELD_MARKER) for line in text.splitlines()), (
        f"no held fire was recorded before the work cleared, so this cell cannot show a "
        f"flip: {text[:900]!r}"
    )
    assert (
        text.count(stall_watchdog.FIRED_MARKER) >= 2
    ), f"the timer fired once, so there was no second episode to be fatal: {text[:600]!r}"


def test_a_caller_with_no_busy_probe_keeps_the_old_exit_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NO PROBE IS NOT "UNKNOWN": it is a caller with nothing to report.

    The distinction is the whole reason this is a cell. A probe that RAISES holds
    the exit leg (the runtime said it could report and the report failed), while a
    caller that supplied no probe at all keeps the bound's documented behaviour —
    the one production arm site always supplies one, so holding on absence would
    only disarm the bound for the rigs and reduced hosts that cannot speak.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    assert [(seconds, exit_) for seconds, exit_, _ in fake.armed] == [(5.0, True)], fake.armed


def test_a_busy_probe_that_raises_holds_the_exit_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable answer must not authorise a cut.

    The dump is written either way, so the two ways of being wrong are not
    symmetric: holding wrongly costs one ``lop stop``, while cutting wrongly
    destroys a turn that nothing can reconstruct.
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

    The fire is written by hand because the real one runs in a C thread that takes
    the process with it when the leg is fatal; what is exercised here is everything
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
