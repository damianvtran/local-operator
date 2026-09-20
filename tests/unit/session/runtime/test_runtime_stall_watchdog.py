"""The runtime bounds its OWN stall: a C-thread dump, then a departure.

WHY THESE TESTS ARE SHAPED THIS WAY, in one read.

The defect they exist for is a *hang*, not a slow function: five runtimes on
2026-09-20 were parked for 1.5-7.2 h inside a C-level scan, every Python-level
instrument in the process was dead by construction (a thread watchdog needs the
GIL a wedged thread never releases; a signal handler needs bytecodes; an
asyncio timer needs the loop), and nothing named the line. So the mechanism
under test is ``faulthandler.dump_traceback_later`` — armed in a C thread — and
a green test that never occupies the loop would prove nothing about it.

THE REPRODUCTION IS THE MEASURED TRIGGER, not an invented one.
``test_a_resume_sized_replay_scan_is_dumped_and_the_process_leaves`` builds a
fixture session store (a manager's child transcripts, synthesized at realistic
sizes in ``tmp_path`` — never the operator's live files, which are 32.6 MB across
22 children for the session that froze twice) and drives the replay that
``hub resume`` performs: the real reader a resumed child is built on
(``Transcript.__init__`` — ``harness/subagent.py`` says that reader IS the whole
resume mechanism) and then the credential-shape pass over its bytes, which the
ops session measured at 49.66 s of CPU over those 22 children. It asserts WHAT
WAS WRITTEN (the file, the innermost frame naming the scan) and WHERE THE
PROCESS WENT (rc 1, and a sentinel the finished replay would have written that
must NOT exist) — never how many milliseconds anything took.

A SECOND, CHEAPER INSTRUMENT STAYS, and it is a different assertion rather than
a duplicate: ``test_a_stalled_loop_is_dumped_and_the_process_leaves`` parks the
loop in a deliberate GIL-holding C call (``ctypes.PyDLL`` does not release the
GIL; ``CDLL`` does), which is the *mechanism* of the freeze — every Python thread
of the process stops — at a fraction of the bytes. The replay test can only fail
if the scan is slow; that one fails if the C timer stops surviving a starved
interpreter, which is the property the whole design rests on.

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
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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
assert stall_watchdog.arm(), "the child could not arm the bound"
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
        args=(str(resumed),),
        env_extra={stall_watchdog.ENV_SECONDS: str(CHILD_BOUND_S)},
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
assert stall_watchdog.arm(), "the child could not arm the bound"
for index in range(30):
    stall_watchdog.beat()
    beats.write_text(str(index), encoding="utf-8")
    time.sleep(0.2)
print("survived", flush=True)
stall_watchdog.disarm()
"""

#: Bytes per synthesized child transcript. The real children of the session that
#: froze twice are 0.6-4.5 MB each (32,621,531 bytes across 22 of them), so these
#: are the SMALL end of the real range and a test-sized slice of the real cost.
REPLAY_CHILD_BYTES = 500_000

#: How many children the fixture store carries. Three keeps the scan's total cost
#: several times the bound below on this host while the fixture stays under 2 MB.
REPLAY_CHILDREN = 3

#: The bound the replay child runs under. One second against a scan the fixture
#: costs ~3.5 s of CPU (measured on this host: 2.30 us/byte with 6270 hits in a
#: 0.86 MB child, i.e. the per-hit full-text scan of that child alone is ~5.4 G
#: byte-scans). The margin is what makes this an assertion about the BOUND rather
#: than a stopwatch: a scan of this size cannot outrun it, however fast the box.
REPLAY_BOUND_S = 1

#: A row body carrying a credential SHAPE, because the shape pass early-returns on
#: text with no anchor (``redaction_shapes.has_shape_anchor``) and a fixture of
#: plain prose would cost nothing to scan — i.e. it would prove nothing. The
#: non-ASCII rune is there because the ops session's own samples landed in the
#: matcher's UCS-2 path (``sre_ucs2_*``), which is chosen by the string being
#: searched, and a fixture that cannot reach it is a fixture that tests less.
_REPLAY_CHUNK = (
    "ran the deploy against the staging cluster\n"
    "MONGO_DSN=mongodb+srv://agent_runtime_model_worker:[redacted]@db.example/app\n"
    "AWS_SECRET_ACCESS_KEY=[redacted]    Authorization: Bearer [redacted]\n"
    "a line with non-ascii \u2603 so the scanner has a wide string to walk\n"
)


def _write_child_transcripts(root: Path) -> int:
    """A manager's child transcripts, at realistic sizes, inside ``tmp_path``.

    Written by the TEST rather than the child so the fixture is inspectable when
    a run fails, and so the child's own timeline is arming -> replay -> (never)
    finished. Nothing here reads the operator's live store: the measured run was
    against 32.6 MB of real transcripts, and reading those at test time is how an
    agent ends up holding a 32 MB string it did not ask for.
    """
    total = 0
    for index in range(REPLAY_CHILDREN):
        directory = root / f"child-{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        written = 0
        while written < REPLAY_CHILD_BYTES:
            body = _REPLAY_CHUNK * 2
            rows.append(
                json.dumps(
                    {
                        "id": f"e{len(rows)}",
                        "ts": 1,
                        "type": "message",
                        "payload": {
                            "kind": "message",
                            "role": "tool",
                            "content": [{"type": "text", "text": body}],
                        },
                    }
                )
            )
            written += len(body)
        (directory / "transcript.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        total += written
    return total


_REPLAY_CHILD = r"""
import os
import pathlib
import sys

from local_operator.redaction_shapes import scrub_shapes
from local_operator.session.runtime import stall_watchdog
from local_operator.session.transcript import Transcript

store = pathlib.Path(sys.argv[1])
finished = pathlib.Path(sys.argv[2])
assert stall_watchdog.arm(), "the child could not arm the bound"
print(f"armed:{os.getpid()}", flush=True)

# THE REPLAY ``hub resume`` PERFORMS: the stopped child's own directory read back
# by the real reader (``harness/subagent.py``: a resumed child is built on the
# stopped child's directory, and that reader is the whole of the mechanism), then
# the credential-shape pass over those bytes.
for child in sorted(store.glob("child-*")):
    replay = Transcript(child)
    scrub_shapes("\n".join(entry.to_json() for entry in replay.entries()))

finished.write_text("the replay finished", encoding="utf-8")
"""


def test_a_resume_sized_replay_scan_is_dumped_and_the_process_leaves(tmp_path: Path) -> None:
    """THE MEASURED TRIGGER: a resume-sized replay scan, named and bounded.

    Session ``e837562a4c28`` froze twice on 2026-09-20, both times ~30 s after
    start and both times immediately after ``hub resume`` of a subagent, at ~0.9
    core with samples inside the shape pass's matcher and the transcript taking
    zero writes. Nothing could say WHERE: the process had no instrument that
    survives a starved interpreter, and the one that would have
    (``LOP_RUNTIME_DEBUG_STACKS``) was not set on the launcher.

    This is that reproduction at test size, and what it asserts is the point of
    the PR: the dump NAMES THE LINE the loop was parked on — the credential-shape
    scan, reached through the real reader a resumed child is built on — and the
    process LEAVES instead of sitting there for hours.

    The innermost frame is a Python caller of a C scan, and that is not a
    shortcoming: ``faulthandler`` prints Python frames, so a C-level ``in`` /
    ``search`` appears as the line that called it — which is still the answer to
    "which line?". The frame this names (``_credential_fragments_survive``'s
    ``if value in text``) is a FULL-TEXT scan per hit, which is why 0.86 MB of
    replayed text with 6270 hits costs ~2 s rather than microseconds; that cost
    model, and its bound, are the next PR's subject rather than this one's.
    """
    store = tmp_path / "store"
    written = _write_child_transcripts(store)
    assert written >= 1_000_000, f"the fixture shrank to {written} bytes; it would prove nothing"
    finished = tmp_path / "finished.txt"

    result = _run_script(
        _REPLAY_CHILD,
        tmp_path,
        args=(str(store), str(finished)),
        env_extra={stall_watchdog.ENV_SECONDS: str(REPLAY_BOUND_S)},
    )

    assert (
        result.returncode == 1
    ), f"the replay scan finished inside the bound: {result.stdout!r} {result.stderr!r}"
    assert (
        "armed:" in result.stdout
    ), f"the child never armed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert not finished.exists(), "the replay completed, so the bound fired too late to matter"
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    dump = _dump_for(tmp_path, pid)
    assert dump.is_file(), f"no dump was written; logs: {sorted((tmp_path / 'logs').glob('*'))}"
    text = dump.read_text(encoding="utf-8")

    assert stall_watchdog.FIRED_MARKER in text, text
    assert "redaction_shapes.py" in text, f"the dump does not name the scan: {text}"
    assert "scrub_shapes" in text, f"the dump does not name the pass: {text}"
    assert text.index(stall_watchdog.ARM_MARKER) < text.index(stall_watchdog.FIRED_MARKER)


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
        args=(str(tmp_path / "beats.txt"),),
        env_extra={stall_watchdog.ENV_SECONDS: str(CHILD_BOUND_S)},
    )

    assert result.returncode == 0, f"the bound fired through the beats: {result.stdout!r}"
    assert "survived" in result.stdout, result.stdout
    assert not list(
        (tmp_path / "logs").glob(f"{stall_watchdog.DUMP_PREFIX}-*.log")
    ), "a clean exit left its dump behind, so a file's existence no longer means the bound fired"


_LEAVING_CHILD = """
import os

from local_operator.session.runtime import stall_watchdog

assert stall_watchdog.arm(), "the child could not arm the bound"
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
    result = _run_script(_LEAVING_CHILD, tmp_path)
    assert result.returncode == 0, result.stderr
    pid = int(result.stdout.split("armed:", 1)[1].split()[0])
    leftover = _dump_for(tmp_path, pid)
    assert leftover.is_file(), "the header-only file this test is about was never written"
    assert stall_watchdog.FIRED_MARKER not in leftover.read_text(encoding="utf-8")
    assert (
        stall_watchdog.fired_dumps(tmp_path / "logs") == []
    ), "a header-only file was reported as a fired bound"

    fired = tmp_path / "logs" / f"{stall_watchdog.DUMP_PREFIX}-4242.log"
    fired.write_text(
        f"header\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n", encoding="utf-8"
    )
    assert stall_watchdog.fired_dumps(tmp_path / "logs") == [fired]


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

    monkeypatch.setenv(stall_watchdog.ENV_SECONDS, "12.5")
    assert stall_watchdog.bound_seconds() == 12.5

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


def test_every_beat_re_arms_the_one_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound measures NO PROGRESS, so each beat restarts it.

    Also pins the two properties a later edit is most likely to lose: arming is
    idempotent (a second call does not leak a second handle or displace the
    file), and a beat with nothing armed — the in-process host, a TUI, a test —
    touches the C timer not at all.
    """
    fake = _FakeFaulthandler()
    monkeypatch.setattr(stall_watchdog, "faulthandler", fake)

    stall_watchdog.beat()
    assert fake.cancels == 0 and fake.armed == [], "a beat armed a timer no one asked for"

    assert stall_watchdog.arm(seconds=5.0, directory=tmp_path) is True
    assert stall_watchdog.arm(seconds=9.0, directory=tmp_path) is True
    assert len(fake.armed) == 1, "a second arm displaced the first instead of being a no-op"

    stall_watchdog.beat()
    stall_watchdog.beat()
    assert fake.cancels == 2
    assert [seconds for seconds, _, _ in fake.armed] == [
        5.0,
        5.0,
        5.0,
    ], "a beat re-armed with a different bound than the one that was armed"
    assert stall_watchdog.is_armed() is True

    file = fake.armed[0][2]
    stall_watchdog.disarm()
    assert stall_watchdog.is_armed() is False
    assert fake.cancels == 3
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

    async def fake_amain() -> int:
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
    assert touched == {"beat"}, f"the serving plane reaches the watchdog for {touched}"


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
        assert stall_watchdog.fired_dumps(config_dir / "logs") == []

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

        # A CLEAN STOP LEAVES NO DUMP. Nothing is in flight, so the runtime leaves
        # at once and ``main``'s finally disarms; a file that survived here would
        # make existence stop meaning "the bound fired".
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
    finally:
        signal_module.signal(signal_module.SIGUSR1, previous_usr1)
        if child is not None:
            _reap(child, config_dir)
