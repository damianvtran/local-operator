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
import inspect
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator import incidents
from local_operator.session.runtime import stall_watchdog

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


def serving_plane() -> None:
    # Healthy, and it SAYS so: the test asserts these stamps landed, so a green
    # run cannot be explained away by the serving plane having been stuck too.
    count = 0
    while True:
        stall_watchdog.beat(stall_watchdog.SERVING)
        count += 1
        if count % 5 == 0:
            print(f"serving-stamp:{count}", flush=True)
        time.sleep(0.2)


threading.Thread(target=serving_plane, daemon=True).start()

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
    which would prove nothing about masking.
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
    assert "beat" in touched, f"the serving plane never reports progress: {touched}"
    assert touched <= {"beat", "SERVING"}, f"the serving plane reaches the watchdog for {touched}"


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

    # ALL THREE, and the window is the whole of the claim: the first disagreeing
    # sample ends the run, and no shorter run may fire.
    state["cpu_per_step"] = 1.0
    # The sample that sees the work STOP is not the first sample of a run: the
    # run starts on the next one, which is what "a sample that disagreed ends it"
    # costs and why the window is measured from there rather than from here.
    state["motion"] = "settled"
    assert step() is False, "a sample that saw movement started a run"
    assert armed.clock.since is None
    assert step() is False
    started = armed.clock.since
    assert started is not None, "the run never started"
    for _ in range(int(armed.seconds) + 2):
        if step():
            break
    else:
        raise AssertionError("the progress leg never fired on a sustained spin")
    assert (
        fake.wall - started >= armed.seconds
    ), f"the progress leg fired {fake.wall - started}s into a {armed.seconds}s window"
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
