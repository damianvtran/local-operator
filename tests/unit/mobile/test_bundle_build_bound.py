"""The portal build's bound: a bound that leaves a fork bomb running is not one.

WHY THESE EXIST
---------------
``_build_bundle`` used to run each step as ``subprocess.run(..., timeout=600)``,
which bounds the DIRECT CHILD only. When the bound fires, Python kills the leader
and returns, and everything the leader forked keeps running — and, in the shape
that took a machine down on 2026-09-21, keeps SPAWNING. A fresh clone's portal
build reached it through a ``packageManager`` pin the global pnpm did not match:
pnpm resolves such a pin by installing that pnpm WITH pnpm, so the tree grew back
after the first kill (+100 processes and +5 GB every 25 s) until the host had
0.1 GB free and no swap, and was rebooted.

TWO GUARDS, ONE INCIDENT. The step now runs in its own process group and the
GROUP is signalled on every path that can end the wait — the bound, an abort raised
in this process (Ctrl-C), an EXTERNAL stop signal (SIGTERM/SIGHUP/SIGQUIT, which
raises nothing here at all and is why the handlers exist), and the ordinary exit —
and a runner whose reported version is not the pinned one is refused before any
pnpm process starts. All of that is exercised here.

The defect IS a process-tree behaviour, so no amount of mocking would be
evidence: each test plants a grandchild whose command line carries a unique
token (a pnpm stand-in does the forking) and asserts on what is ALIVE afterwards
— the same instrument the field observation used (``pgrep -f``).

WHAT FAILS ON THE PRE-FIX TREE, stated precisely because the distinction matters
when grading the evidence — and, for the bound branch, because the COUNT MOVES
with the revision the question is asked of, so both revisions were measured.
"The pre-fix tree" is the BRANCH BASE, ``fcf56b68``, and the file measured is the
one carrying this docstring: it collects **25** tests there and **23 fail, 2
pass**. SIX of the failures are assertions on the pre-fix BEHAVIOUR — the
self-exiting step leaves its descendants alive, the mismatch runs
``pnpm install`` instead of refusing, the refusal that the two route tests assert
on never comes (twice), the probe runs inside the tree that carries the manifest,
and the invocation list shows no probe at all. The **two bound-branch** tests fail
a step earlier, at the monkeypatched ``_BUILD_STEP_TIMEOUT`` /
``_BUILD_KILL_GRACE``, neither of which the base's ``install.py`` has: they raise
``AttributeError`` at the patch — before ``_build_bundle`` is called at all, so no
process starts — which is why the bound-branch before/after is carried by the
PR's process listings rather than by a test. The abort and external-stop tests
fail before their own subject as well, and not for the defect's sake: the driver
they drive calls ``_run_build_step``,
absent on the base, so the step never plants the descendant they wait for.
Everything else raises ``AttributeError`` on ``_pinned_pnpm`` / ``_pin_mismatch``
— the function under test does not exist there at all. (The two that pass are
``test_an_unreadable_runner_is_not_refused`` — on the base there is no refusal to
be wrong about — and this file's own sweep test, which touches no module state.)
On the branch's own ``8b39b4bb`` — where both rules DO exist — the same file
collects 25 and **5 fail, 20 pass**: of the five tests that assert on the step's
process group, exactly one fails, and it is the external-stop one (``a descendant
([pid]) outlived an external SIGTERM``), which is what the branch's last commit
``92f0d928`` closes. The other four are the three refusal tests, whose assertions
describe the copy this PR ships (one pre-existing and strengthened here, two
added), and ``test_a_dev_engines_pin_is_read_too``, a spelling that arrives with
``92f0d928``.

THIS FILE ALSO CARRIES THE MEMORY-BOUND TESTS (the section at the end, added with
the second bound on 2026-09-21), which is why the paragraph above counts 25 while
the file collects **30** on this branch. Those five have their own before/after
reading and it is NOT a tree comparison, so they add nothing to the count above:
``fcf56b68`` has neither ``local_operator/memory_guard.py`` (``git ls-tree
fcf56b68 local_operator/memory_guard.py`` is empty) nor ``_run_build_step`` in its
``install.py``, so there they fail at the first symbol they touch, before any
process starts. What discriminates the memory bound is a PAIR of readings of the
SAME child in the SAME tree — the ceiling live, and the budget disabled so that
only the clock remains — and it is measured in
``test_a_fast_allocator_is_killed_at_the_ceiling_not_by_the_clock`` and
``test_the_time_bound_alone_lets_the_same_child_run_on``.
``test_a_probe_the_ceiling_stopped_is_still_no_answer`` asserts the other half of
the same claim: the fail-open in ``_pin_mismatch`` may rest only on a bound that
holds, so the bound's half is pinned too.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, Iterator

import pytest

from local_operator import memory_guard
from local_operator.mobile import install

#: These spawn real processes and wait on real signals; the suite marks that
#: shape `slow` (see pyproject.toml's marker list).
pytestmark = pytest.mark.slow


def _token() -> str:
    """A marker that appears ONLY in a planted descendant's command line."""
    return f"bundle-bound-test-{uuid.uuid4().hex}"


def _pids_carrying(token: str) -> list[int]:
    """Every process whose command line carries ``token``, orphans included.

    ``pgrep -f`` is the instrument the incident was observed with, and unlike a
    ``ps --ppid`` walk it sees a process that has been re-parented — which is
    what a leaked descendant becomes.
    """
    result = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    return [int(line) for line in result.stdout.split() if line.strip()]


def _wait_until_gone(token: str, timeout: float = 15.0) -> list[int]:
    """The pids still carrying ``token`` once the tree should be dead."""
    deadline = time.monotonic() + timeout
    remaining = _pids_carrying(token)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _pids_carrying(token)
    return remaining


def _sweep(patterns: Iterable[str]) -> list[int]:
    """SIGKILL every pid matching each pattern, and return the ones still alive.

    Killed by exact PID and never by a program name: this host runs two dozen
    sessions, and a name-scoped ``pgrep`` here is how another session's process
    tree gets killed. Survivors are RETURNED rather than swallowed, because a
    sweep is only as good as its patterns and a sweep that misses the leader is
    the leak this file has already had once (see the fixture).

    One owner for the sweep, so the fixture's teardown and the test that pins it
    cannot drift apart.
    """
    for pattern in patterns:
        for pid in _pids_carrying(pattern):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    survivors: list[int] = []
    for pattern in patterns:
        survivors.extend(_wait_until_gone(pattern, timeout=5.0))
    return survivors


@pytest.fixture
def reap_markers(tmp_path: Path) -> Iterator[list[str]]:
    """Kill anything a failing test leaves behind, by exact pid, and say so.

    A test for a leak must not become one: this host runs two dozen sessions, and
    an orphaned sleeper from a red test would outlive it. Killed by PID, and never
    by a program name.

    TWO patterns, because a token-only sweep misses the LEADER. The planted token
    lives in the GRANDCHILD's argv; the ``stand-in-pnpm`` leader's argv carries the
    script's PATH and no token at all, so a test that failed between the fork and
    the reap left the leader re-parented (``ppid=1``) to its 600 s linger —
    observed in 1 of 3 full-file runs at load ~105, where it had to be killed by
    hand. ``tmp_path`` is unique per test (pytest names the directory after the
    test), so sweeping it can only reach this test's own stand-ins and driver,
    never a sibling test's or another session's.

    The sweep's survivors are ASSERTED on, so a pattern that stops matching fails
    the test that leaked instead of leaking quietly.
    """
    tokens: list[str] = []
    try:
        yield tokens
    finally:
        patterns = [*tokens, str(tmp_path)]
        survivors = _sweep(patterns)
        assert survivors == [], f"the sweep left {survivors} alive for {patterns}"


#: A pnpm stand-in: it records what it was asked to do, answers ``--version``
#: with the version ``config.json`` names, and — when configured to — plants a
#: grandchild carrying the test's token before lingering like a real build does.
#: Forking is the point: the defect is what happens to a step's DESCENDANTS when
#: the bound fires, and a stand-in that only slept would not have any. The
#: grandchild ignores SIGTERM, so only a SIGKILL rung can clear it.
_FAKE_RUNNER = '''#!/usr/bin/env python3
"""Generated by tests/unit/mobile/test_bundle_build_bound.py — see that module."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GRANDCHILD = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"

#: The instrument: one line per invocation, so a test can assert what the
#: builder did NOT run as well as what it did.
with (HERE / "invocations.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            {
                "argv": sys.argv[1:],
                "cwd": os.getcwd(),
                "manifest": (Path(os.getcwd()) / "package.json").exists(),
            }
        )
        + "\\n"
    )

#: Its OWN pid, in a file of its own rather than a line in the log above: a test
#: that wants to WATCH this group (sample its RSS, then find its processes dead)
#: has to know which process is the step's leader, and `pgrep -f` cannot say — the
#: driver's argv carries the stand-in's path too. Written whole rather than
#: appended, because it is read WHILE the step runs and a half-written jsonl line
#: is a parse error instead of a reading.
(HERE / "leader.pid").write_text(str(os.getpid()), encoding="utf-8")


def take(mb):
    """Take ``mb`` of memory, TOUCHING every page of it.

    Touching is the point: a block the kernel has not faulted in is not resident,
    and RSS — the quantity the bound reads — would not see it. Capped by ``mb``, on
    purpose: these tests run on the host the incident took down, and the aim is to
    cross a ceiling of tens of MB quickly, not to build a second incident.
    """
    target = int(mb) * 1024 * 1024
    taken = 0
    held = []
    while taken < target:
        chunk = bytearray(min(8 * 1024 * 1024, target - taken))
        chunk[::4096] = b"\\x01" * (len(chunk) // 4096)
        held.append(chunk)
        taken += len(chunk)
        time.sleep(0.05)
    return held


if CONFIG.get("deaf"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

if "--version" in sys.argv[1:]:
    # A probe that GROWS instead of answering is the shape `_pin_mismatch`'s
    # fail-open is written for: pnpm resolving the pin rather than reporting a
    # version. `probe_grow_mb` is that probe, and it never answers.
    if CONFIG.get("probe_grow_mb"):
        held = take(CONFIG["probe_grow_mb"])
        time.sleep(CONFIG.get("linger", 600))
    print(CONFIG["version"])
    raise SystemExit(0)

if CONFIG.get("dist"):
    dist = Path(os.getcwd()) / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")

# The LEADER's own share of the group's memory, when a test is about the SUM.
if CONFIG.get("leader_grow_mb"):
    held = take(CONFIG["leader_grow_mb"])

if CONFIG.get("fork"):
    subprocess.Popen(
        [sys.executable, "-c", GRANDCHILD, CONFIG["token"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(CONFIG.get("linger", 600))

if CONFIG.get("grow_mb"):
    # A DESCENDANT that grows, because the group is what the bound kills and the
    # field defect was the tree, not the leader. The token rides its argv so the
    # same `pgrep -f` instrument the incident was observed with finds it.
    subprocess.Popen(
        [sys.executable, str(HERE / "grow-child.py"), CONFIG["token"], str(CONFIG["grow_mb"])],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(CONFIG.get("linger", 600))

raise SystemExit(CONFIG.get("exit", 1))
'''

#: The grandchild a ``grow`` stand-in plants, as its own file rather than a
#: ``-c`` one-liner: it is a real loop, and quoting it through the generator above
#: is how a test instrument turns into a puzzle.
_GROW_CHILD = '''#!/usr/bin/env python3
"""Generated by tests/unit/mobile/test_bundle_build_bound.py."""

import os
import signal
import sys
import time
from pathlib import Path

#: Only a SIGKILL rung clears it, which is what makes the group reap observable.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
#: Its pid, for a test that reports WHICH processes the bound ended.
Path(__file__).with_name("child.pid").write_text(str(os.getpid()), encoding="utf-8")

target = int(sys.argv[2]) * 1024 * 1024
taken = 0
held = []
while taken < target:
    chunk = bytearray(min(8 * 1024 * 1024, target - taken))
    chunk[::4096] = b"\\x01" * (len(chunk) // 4096)
    held.append(chunk)
    taken += len(chunk)
    time.sleep(0.05)

time.sleep(600)
'''


class StandIn:
    """A generated pnpm stand-in inside a temp dir, plus its recorded calls."""

    def __init__(self, root: Path, **config: object) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.script = root / "stand-in-pnpm"
        self.script.write_text(_FAKE_RUNNER, encoding="utf-8")
        self.grower = root / "grow-child.py"
        self.grower.write_text(_GROW_CHILD, encoding="utf-8")
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    @property
    def runner(self) -> list[str]:
        """The argv prefix the builder is asked to run.

        Through ``sys.executable`` rather than a shebang: the stand-in must run
        under the interpreter already under test, not under whichever ``python3``
        a PATH lookup finds on this host.
        """
        return [sys.executable, str(self.script)]

    def invocations(self) -> list[dict[str, object]]:
        log = self.root / "invocations.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def argv_seen(self) -> list[list[str]]:
        return [list(entry["argv"]) for entry in self.invocations()]  # type: ignore[arg-type]


#: A driver that runs ONE build step from the tree under test and reports how it
#: ended, so a test can signal THAT process from the outside — SIGINT or SIGTERM —
#: exactly as an operator's Ctrl-C or `kill` would. The step's group is its own
#: session, so the signal never reaches the children and the test cannot reach
#: them through the driver either; that is the point.
#:
#: Streams go to DEVNULL rather than a pipe, because the step's own descendants
#: inherit the driver's descriptors: a pipe here would make the reader wait on
#: holders the driver never owned (the hazard `tests/unit/scripts/
#: test_run_bounded.py::_wrapper` documents), which would couple the assertion to
#: pipe lifetimes instead of to the process group.
_DRIVER = '''#!/usr/bin/env python3
"""Generated by tests/unit/mobile/test_bundle_build_bound.py.

The optional probe delay exists because of a WINDOW, not for convenience: the
step's memory budget is resolved by FORKING host probes (``vm_stat``,
``sysctl -n vm.swapusage``), so the instant between the spawn and the guarded
wait is real, and it is milliseconds long on an idle host and hundreds on a
loaded one. A test that hopes to deliver an abort into that instant is a coin
toss; one that widens it deliberately is a discriminator. The probes still run
for real — only their duration is inflated, which is what a loaded host does to
them on its own.
"""

import os
import sys
import time
from pathlib import Path

from local_operator import memory_guard
from local_operator.mobile import install

_delay_ms = int(os.environ.get("BOUND_DRIVER_PROBE_DELAY_MS", "0"))
if _delay_ms:
    _real_runner = memory_guard._default_runner

    def _slow_runner(argv):
        answer = _real_runner(argv)
        # The budget probes only: the guard's own per-tick ``ps`` read stays cheap,
        # so what is widened is the spawn-to-guard window and nothing else.
        if argv[:1] != ["ps"]:
            time.sleep(_delay_ms / 1000)
        return answer

    memory_guard._default_runner = _slow_runner

runner = sys.argv[1:-1]
cwd = Path(sys.argv[-1])
print(f"driver pid={os.getpid()}", flush=True)
try:
    result = install._run_build_step([*runner, "install", "--frozen-lockfile"], cwd, timeout=120)
except BaseException as exc:  # the abort arm: KeyboardInterrupt and friends
    print(f"driver: {type(exc).__name__} propagated", flush=True)
    raise SystemExit(9)
print(f"driver: rc={result.returncode}", flush=True)
'''

#: What :data:`_DRIVER` exits with once the abort arm has propagated an abort.
_DRIVER_ABORT_STATUS = 9

#: How far the abort rung inflates each host probe, in ms.
#:
#: The window this rung has to sit in is the instant between the spawn and the
#: guarded wait, and in that instant the step's memory budget forks its probes.
#: Without this, the rung is a coin toss: the signal lands in the wait most of the
#: time (green) and inside the window occasionally (red) — which is exactly the
#: flake the reviewer measured at 7/10. Two probes are made per budget, so the
#: window is ~2x this value, and the token-driven wait below notices a planted
#: descendant in ~50-200 ms, comfortably inside it.
_PROBE_WINDOW_MS = 500


def _write_driver(root: Path) -> Path:
    driver = root / "step-driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    return driver


def _start_driver(
    root: Path, fake: "StandIn", web: Path, *, probe_delay_ms: int = 0
) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    if probe_delay_ms:
        env["BOUND_DRIVER_PROBE_DELAY_MS"] = str(probe_delay_ms)
    return subprocess.Popen(
        [sys.executable, str(_write_driver(root)), *fake.runner, str(web)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _kill_if_alive(proc: "subprocess.Popen[Any]") -> None:
    """Reap a driver that did not end on its own, by its exact pid.

    ``Any`` for the child's text mode on purpose: the abort driver's stdout is a
    DEVNULL pipe and the stream test drives a TEXT-mode child, and this helper only
    polls and kills — the mode is not its business.
    """
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def _wait_for_pid_carrying(token: str, timeout: float = 20.0) -> list[int]:
    """Token-carrying pids, once the stand-in's descendant exists to be found.

    Waited for rather than slept past: a signal delivered before the step has
    planted its tree would test the window rather than the path.
    """
    deadline = time.monotonic() + timeout
    found = _pids_carrying(token)
    while not found and time.monotonic() < deadline:
        time.sleep(0.1)
        found = _pids_carrying(token)
    return found


def _web(tmp_path: Path, *, pin: str | None, dev_engines: dict[str, object] | None = None) -> Path:
    """A build tree: a manifest carrying whatever pin the test is about."""
    web = tmp_path / "web"
    web.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"name": "web", "private": True}
    if pin is not None:
        manifest["packageManager"] = pin
    if dev_engines is not None:
        manifest["devEngines"] = dev_engines
    (web / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    return web


def test_the_sweep_matches_the_leader_and_not_only_the_planted_token(tmp_path: Path) -> None:
    """The sweep's PATTERNS, pinned where the leak actually was.

    The planted token is in the grandchild's argv; the leader's carries the
    stand-in's script path instead, so a token-only sweep — what this file had
    until an abort test left a re-parented leader behind in 1 of 3 full-file runs
    — matches the descendant and misses the process that owns the group. Asserted
    on the sweep itself rather than through a failing test, because the sweep is
    what has to hold whatever fails; the fixture's teardown drives the same
    function with the same two patterns.
    """
    token = _token()
    fake = StandIn(tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600)
    leader = subprocess.Popen(
        [*fake.runner, "install", "--frozen-lockfile"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for_pid_carrying(token), "the stand-in never planted its grandchild"
        # The leak, asserted rather than hoped for: the token does NOT name the
        # leader, which is why the tree's path has to be a pattern of its own.
        assert leader.pid not in _pids_carrying(token)

        survivors = _sweep([token, str(tmp_path)])

        assert survivors == [], f"the sweep left {survivors} alive"
        assert leader.poll() is not None, "the leader outlived the sweep"
        assert _pids_carrying(str(tmp_path)) == []
    finally:
        _kill_if_alive(leader)


def test_the_bound_reaps_the_group_not_just_the_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reap_markers: list[str]
) -> None:
    """The field defect: the bound fires, the leader dies, the work does not.

    WHAT THIS FAILS WITH ON THE BASE depends on which base, so name it: against
    the branch base ``fcf56b68`` it cannot run at all — the two constants patched
    below do not exist there, so it raises ``AttributeError`` in 0.5 s — and the
    behaviour it asserts is absent there anyway, because
    ``subprocess.run(timeout=...)`` kills the direct child only and nothing else
    was watching the group. From ``94f2ecde`` on, where both the constants and
    the group bound exist, it passes; the base's behaviour is carried by the PR's
    process listings rather than by this test.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600)
    web = _web(tmp_path, pin=None)
    monkeypatch.setattr(install, "_BUILD_STEP_TIMEOUT", 3.0)
    monkeypatch.setattr(install, "_BUILD_KILL_GRACE", 1.0)

    error = install._build_bundle(web, fake.runner)

    assert error is not None, "a step killed by the bound must not read as success"
    assert "bundle build failed" in error
    survivors = _wait_until_gone(token)
    assert survivors == [], (
        f"a descendant of the bounded step survived it ({survivors}): the bound "
        "reached the leader and nothing reached the group"
    )


def test_the_bound_clears_a_group_that_ignores_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reap_markers: list[str]
) -> None:
    """SIGTERM is rung one; a group that ignores it must still be cleared.

    Both processes ignore SIGTERM here, so this holds only if the bound escalates
    to SIGKILL — the grace window is the one thing a bare ``timeout`` waits out.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(
        tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600, deaf=True
    )
    web = _web(tmp_path, pin=None)
    monkeypatch.setattr(install, "_BUILD_STEP_TIMEOUT", 3.0)
    monkeypatch.setattr(install, "_BUILD_KILL_GRACE", 1.0)

    error = install._build_bundle(web, fake.runner)

    assert error is not None
    survivors = _wait_until_gone(token)
    assert survivors == [], f"a SIGTERM-deaf group ({survivors}) outlived the bound"


def test_a_descendant_does_not_outlive_a_step_that_exits_by_itself(
    tmp_path: Path, reap_markers: list[str]
) -> None:
    """An ordinary exit is not a bound at all, and nothing else reaps the group.

    This is the half ``timeout(1)`` — and a plain ``subprocess.run`` — cannot
    cover by construction: by the time the leader has exited there is nothing
    left to signal, and whatever it forked keeps the machine's memory. This step
    SUCCEEDS, so the assertion is about a green build rather than a killed one,
    and it is the branch the fork bomb regrew on.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(
        tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=0, exit=0, dist=True
    )
    web = _web(tmp_path, pin=None)

    error = install._build_bundle(web, fake.runner)

    assert error is None, f"the stand-in's steps succeed, so this should build: {error}"
    assert (web / "dist" / "index.html").exists()
    survivors = _wait_until_gone(token)
    assert survivors == [], (
        f"a descendant ({survivors}) outlived a step that exited on its own — the "
        "case the sweep exists for, and the one that let the fork bomb regrow "
        "after the first kill"
    )


def test_the_post_exit_sweep_is_posix_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows there is no group to sweep, and the sweep must not pretend otherwise.

    ``taskkill /T`` walks the tree from the LEADER, and on this path the leader is
    already reaped — so the only thing a call could do is aim at a pid that may
    since have been recycled. Stated rather than fixed (the alternative is a
    parent-pid walk, which needs a ``psutil`` dependency this repo deliberately
    does not take), and pinned here so the honesty claim in
    :func:`install._sweep_step_group` cannot drift from the code.
    """
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        install.procstate,
        "terminate_process_tree",
        lambda pid, force=False: calls.append((pid, force)) or True,
    )
    # ``text=True`` because the helper is typed for a text Popen; the stand-in's
    # output is irrelevant here, which is why both streams go to DEVNULL.
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    proc.wait()

    install._sweep_step_group(proc, None)

    assert calls == [], "a reaped leader has no tree left to walk: sweeping by its pid is aimless"


def test_the_abort_rung_reaps_the_group_when_this_process_is_interrupted(
    tmp_path: Path, reap_markers: list[str]
) -> None:
    """The abort rung: an interruption raised IN this process (a real Ctrl-C).

    One of the paths :func:`install._run_build_step` claims, and the one that had
    no test at all (QA round 1, Q-2). Delivered as a real SIGINT to a real driver
    running one step, so what is exercised is the path an operator's Ctrl-C
    takes — not a monkeypatched exception, which would only prove the test can
    raise.

    WHERE THE SIGNAL LANDS, which is the whole of its teeth (agent review round
    1, BLOCKER-1). The spawn-to-wait instant is a real window — the budget forks
    host probes in it — and an abort delivered there leaked the whole step group
    until that review: measured on this host, `guard = _step_memory_guard(pgid)`
    sitting one line ABOVE the `try` that owns the reap arms escaped with the step
    leader still alive and re-parented to 1, its descendant with it. So the rung
    does not hope to land in that window: the driver inflates the probes
    (:data:`_PROBE_WINDOW_MS`) and the signal is sent the moment the step's tree
    is visible, which is the same token-driven trigger that found the leak and
    NOT the fixed-delay shape that missed it by landing after the window.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600)
    web = _web(tmp_path, pin=None)
    proc = _start_driver(tmp_path, fake, web, probe_delay_ms=_PROBE_WINDOW_MS)
    try:
        assert _wait_for_pid_carrying(token), "the step never planted its descendant"
        os.kill(proc.pid, signal.SIGINT)  # exactly what a terminal's Ctrl-C delivers
        assert (
            proc.wait(timeout=30) == _DRIVER_ABORT_STATUS
        ), "the abort must reach the caller, not be swallowed into a result"
    finally:
        _kill_if_alive(proc)
    survivors = _wait_until_gone(token)
    assert survivors == [], (
        f"a descendant ({survivors}) outlived an abort — the arm that reaps on the "
        "way out did not reach the group"
    )


def test_an_external_sigterm_reaps_the_group_before_we_die(
    tmp_path: Path, reap_markers: list[str]
) -> None:
    """The external-stop rung: SIGTERM from OUTSIDE, which raises nothing here.

    The step is its own session, so a signal sent to this process never reaches
    the children, and SIGTERM's default disposition terminates without running a
    line of our code — so an exception arm cannot help (QA round 1, Q-1). This is
    how the incident began: its first install was SIGTERM'd and the tree kept
    growing with nothing left owning it.

    Both halves are asserted: nothing survives, and the process still REPORTS the
    signal (killed by SIGTERM) rather than being swallowed into a quiet exit.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600)
    web = _web(tmp_path, pin=None)
    proc = _start_driver(tmp_path, fake, web)
    try:
        assert _wait_for_pid_carrying(token), "the step never planted its descendant"
        os.kill(proc.pid, signal.SIGTERM)
        assert (
            proc.wait(timeout=30) == -signal.SIGTERM
        ), "the handler must re-deliver the signal after reaping, not swallow it"
    finally:
        _kill_if_alive(proc)
    survivors = _wait_until_gone(token)
    assert survivors == [], (
        f"a descendant ({survivors}) outlived an external SIGTERM: the group is its "
        "own session, so nothing in the driver's exception arms could have run"
    )


def test_a_pnpm_that_is_not_the_pinned_one_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recursion's engine, refused: no build step is ever started.

    ``package.json`` pins pnpm@11.22.0 and the runner reports 10.30.3. Asserted on
    the refusal AND on the absence of any ``install``/``build`` invocation,
    because a refusal that still spawned pnpm would leave the engine running.

    PATH is emptied first, and that is not tidiness: the refusal is only this
    host's answer when NO route can supply the pin, and a resolving Corepack is one
    (it builds through Corepack instead — see
    ``test_pnpm_fetch_guard.py::test_the_corepack_route_builds_a_tree_the_path_pnpm_cannot``),
    so a test that inherited a developer's or a CI runner's PATH would measure a
    decision that depends on the machine rather than on the guard.
    """
    no_tools = tmp_path / "no-tools"
    no_tools.mkdir()
    monkeypatch.setenv("PATH", str(no_tools))
    fake = StandIn(tmp_path / "bin", version="10.30.3")
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, fake.runner)

    assert error is not None
    assert "10.30.3" in error, f"the refusal must name the version on PATH: {error}"
    assert "11.22.0" in error, f"the refusal must name the pinned version: {error}"
    # The CONDITION is what holds on every host; the routes below it are a list of
    # what this host actually has, because no single route is universal (R1-1:
    # `npm install -g` cannot land where another manager owns the `pnpm` on PATH,
    # and Corepack is absent from Node >= 25). Both host shapes are pinned by the
    # two tests below, so a copy assertion that only passes where a given launcher
    # happens to exist cannot come back.
    assert "has to report 11.22.0" in error
    assert "whatever manager installed the pnpm already on PATH" in error
    assert ("npm install -g pnpm@11.22.0" in error) == (install._shim_argv("npm") is not None)
    assert ("corepack enable" in error) == (install._shim_argv("corepack") is not None)
    assert "fans out" in error, "the why has to be in the sentence, not only in the code"
    assert fake.argv_seen() == [
        ["--version"]
    ], f"only the version probe may run: {fake.argv_seen()}"


def test_the_refusal_drops_every_tool_route_that_does_not_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remedy the reader cannot run is a second refusal, not a remedy.

    With nothing on ``PATH`` neither the npm route nor the Corepack one exists on
    this (simulated) host, so neither may be named — while the condition that has
    to hold and the route that is always available (whatever installed the pnpm
    already on PATH) stay in the sentence. That is the shape R1-1 asked for: state
    the requirement, and offer only routes the machine can run.
    """
    fake = StandIn(tmp_path / "bin", version="10.30.3")
    web = _web(tmp_path, pin="pnpm@11.22.0")
    no_tools = tmp_path / "no-tools"
    no_tools.mkdir()
    monkeypatch.setenv("PATH", str(no_tools))

    error = install._build_bundle(web, fake.runner)

    assert error is not None
    assert "11.22.0" in error
    assert error.index("has to report 11.22.0") < error.index("`lop mobile install`")
    assert "npm install -g" not in error, f"this route does not exist on this host: {error}"
    assert "corepack" not in error, f"this route does not exist on this host: {error}"
    assert "whatever manager installed the pnpm already on PATH" in error


def test_the_refusal_keeps_the_corepack_route_where_one_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: where Corepack IS installed, that route stays in the copy.

    Both shapes have to be pinned, not just the one this host happens to be in —
    the routes are a fact about the machine, so a test that only checks their
    absence would go green by deleting them everywhere. The npm route is asserted
    ABSENT here as well, so the two are shown to be independent rather than two
    spellings of one conditional.

    Read straight off :func:`install._pin_mismatch` rather than through
    :func:`install._build_bundle`, because on a host where Corepack resolves that
    route is TAKEN, not offered: ``_runner_or_refusal`` builds through Corepack, so
    the guard's sentence is the only place this copy can be observed. The taken
    route is pinned by
    ``test_pnpm_fetch_guard.py::test_the_corepack_route_builds_a_tree_the_path_pnpm_cannot``.
    """
    fake = StandIn(tmp_path / "bin", version="10.30.3")
    web = _web(tmp_path, pin="pnpm@11.22.0")
    tools = tmp_path / "tools"
    tools.mkdir()
    # The launcher spelling each platform's `shutil.which` looks for, so this
    # asserts the guard on Windows too rather than only where the shim is a
    # bare name.
    shim = tools / ("corepack.cmd" if os.name == "nt" else "corepack")
    shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))

    error = install._pin_mismatch(fake.runner, web)

    assert error is not None
    assert "corepack enable" in error
    assert "npm install -g" not in error, f"npm does not resolve on this PATH: {error}"


def test_the_version_probe_runs_outside_the_pinned_tree(tmp_path: Path) -> None:
    """WHERE the probe runs is load-bearing, not tidy.

    pnpm reads the nearest ``package.json`` above its cwd, so probing
    ``pnpm --version`` INSIDE the pinned tree does not answer — it starts
    resolving the pin, which is the recursion this guard exists to refuse
    (measured: past 120 s and spawning ``pnpm add pnpm@11.22.0`` children, where
    the same command in an empty directory answered in under a second). Asserted
    on the probe's own recorded cwd.
    """
    fake = StandIn(tmp_path / "bin", version="10.30.3")
    web = _web(tmp_path, pin="pnpm@11.22.0")

    install._build_bundle(web, fake.runner)

    probe = fake.invocations()[0]
    assert (
        probe["manifest"] is False
    ), f"the probe ran in a tree that carries a manifest: {probe['cwd']}"
    assert Path(str(probe["cwd"])) != web


def test_a_matching_pin_builds(tmp_path: Path) -> None:
    """The normal path still works: the guard is a wall, not a wall for everyone.

    What this does NOT show: a real pnpm build. It shows the decision function and
    the wiring — the runner is asked for its version, is allowed through, and both
    steps then run in the pinned tree. The PR says so plainly.
    """
    fake = StandIn(tmp_path / "bin", version="11.22.0", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, fake.runner)

    assert error is None
    assert fake.argv_seen() == [["--version"], ["install", "--frozen-lockfile"], ["build"]]
    steps = fake.invocations()[1:]
    assert all(
        entry["manifest"] is True for entry in steps
    ), "the build steps must run in the PINNED tree, not in the probe's temp dir"
    assert all(Path(str(entry["cwd"])) == web for entry in steps)


def test_an_unreadable_runner_is_not_refused(tmp_path: Path) -> None:
    """An inconclusive probe proceeds: the GROUP BOUND is the wall.

    Refusing here would trade a rare hang for a common false refusal on machines
    whose pnpm is fine, and the bound already makes a bad build survivable.
    """
    fake = StandIn(tmp_path / "bin", version="", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, fake.runner)

    assert error is None
    assert ["install", "--frozen-lockfile"] in fake.argv_seen()


def test_corepack_is_the_route_out_and_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corepack RESOLVES the pin, so it is never the mismatch this guard refuses.

    Checked on both spellings Corepack reaches us as: the argv
    :func:`install._package_runner` builds, and a ``pnpm`` shim Corepack itself
    installed on PATH — the route the refusal recommends must not be caught by it.
    """
    assert install._corepack_shaped(["corepack", "pnpm"]) is True

    shim = tmp_path / "pnpm"
    shim.write_text('#!/bin/sh\n# corepack shim\nexec corepack pnpm "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert install._corepack_shaped(["pnpm"]) is True


def test_a_plain_runner_is_not_taken_for_corepack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the check: a global pnpm must still be compared to the pin."""
    shim = tmp_path / "pnpm"
    shim.write_text('#!/bin/sh\nexec node /somewhere/pnpm.cjs "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert install._corepack_shaped(["pnpm"]) is False


def _npx_stand_in(tools: Path) -> StandIn:
    """An ``npx`` on PATH that records its argv and writes the dist on ``build``.

    The SAME generated script as :class:`StandIn` — one instrument for the whole
    file, so a change that breaks the recording breaks both loudly — copied to
    the name ``npx`` and given an ABSOLUTE interpreter shebang. The absolute one
    matters: these tests point ``PATH`` at this directory alone, so
    ``#!/usr/bin/env python3`` would fail to resolve its own interpreter before
    the script ever ran. The exec bit is what ``shutil.which`` (inside
    ``_shim_argv``) requires.
    """
    npx = StandIn(tools, exit=0, dist=True)
    body = npx.script.read_text(encoding="utf-8").replace(
        "#!/usr/bin/env python3", f"#!{sys.executable}", 1
    )
    shim = tools / ("npx.cmd" if os.name == "nt" else "npx")
    shim.write_text(body, encoding="utf-8")
    shim.chmod(0o755)
    return npx


def test_the_npx_arm_is_chosen_only_when_no_other_route_can_supply_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last-resort route: nothing local can satisfy the pin, so npx fetches it.

    The host this was written on is the shape: pnpm 10.30.3 on PATH against a
    ``pnpm@11.22.0`` pin, and Node v26.5.0 — which ships no corepack at all — so
    neither the seeded arm nor the corepack arm resolves and the guard would
    otherwise refuse. ``npx --yes pnpm@11.22.0`` fetches the PIN and runs it,
    without entering pnpm's own ``pnpm add pnpm@<pin>`` self-install (the
    recursion this guard exists to refuse).

    PATH is the stand-in directory ALONE, so the route is measured against this
    test's npx and not against a developer's or a CI runner's corepack; PNPM_HOME
    points at an empty home so no seeded pin short-circuits the arm (the arm order
    is asserted by the tests above it).
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    npx = _npx_stand_in(tools)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("PNPM_HOME", str(tmp_path / "empty-home"))
    fake = StandIn(tmp_path / "bin", version="10.30.3", exit=1)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    error = install._build_bundle(web, fake.runner)

    assert error is None, f"npx can satisfy the pin, so this host must build: {error}"
    assert fake.argv_seen() == [
        ["--version"]
    ], "the runner that cannot satisfy the pin is probed and never asked to build"
    # The pinned version is what npx is asked for, and the two build steps are
    # appended to the npx argv prefix exactly as they are for the other arms.
    assert npx.argv_seen() == [
        ["--yes", "pnpm@11.22.0", "install", "--frozen-lockfile"],
        ["--yes", "pnpm@11.22.0", "build"],
    ], f"npx runs the PIN, once per step: {npx.argv_seen()}"


def test_the_npx_arm_is_not_consulted_when_path_pnpm_is_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No download on a machine that is already fine — the arm's ordering, pinned.

    The routes are consulted only when the guard WOULD refuse, so a matching pnpm
    on PATH must leave npx untouched. A route that ran anyway would fetch a
    package manager for a build that needed nothing, which is the property the
    earlier arms were ordered to preserve and this one must not lose.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    npx = _npx_stand_in(tools)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("PNPM_HOME", str(tmp_path / "empty-home"))
    fake = StandIn(tmp_path / "bin", version="11.22.0", exit=0, dist=True)
    web = _web(tmp_path, pin="pnpm@11.22.0")

    assert install._build_bundle(web, fake.runner) is None

    assert ["build"] in fake.argv_seen(), "PATH's pnpm is the one that builds"
    assert npx.argv_seen() == [], "nothing to fix, so nothing is fetched"


def test_the_npx_arm_is_not_reached_for_a_range_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A range is not a pin, and this arm must not be the first route to resolve one.

    ``_pinned_pnpm`` reports the ``packageManager`` version verbatim, so a tree
    pinning ``pnpm@^11`` reaches this arm with ``^11`` in hand. Fetching
    ``pnpm@^11`` through npx would run *whatever the registry serves today* — a
    wider promise than any existing route makes (pnpm's own switch returns early
    on a range, and ``_package_manager_env`` states at length that a range is not
    a fetch), and an untestable one. The exact-version matcher the disarm gating
    already uses is what excludes it, so the refusal below is today's behaviour.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    npx = _npx_stand_in(tools)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setenv("PNPM_HOME", str(tmp_path / "empty-home"))
    fake = StandIn(tmp_path / "bin", version="10.30.3", exit=1)
    web = _web(tmp_path, pin="pnpm@^11")

    error = install._build_bundle(web, fake.runner)

    assert error is not None, "a range falls through to the refusal, not to a fetch"
    assert "^11" in error
    assert npx.argv_seen() == [], f"nothing may be fetched for a range: {npx.argv_seen()}"


@pytest.mark.parametrize(
    ("pin", "expected"),
    [
        ("pnpm@11.22.0", "11.22.0"),
        ("pnpm@11.22.0+sha512.deadbeef", "11.22.0"),
        ("npm@10.9.0", None),
        (None, None),
    ],
)
def test_the_pin_is_read_from_the_trees_own_manifest(
    tmp_path: Path, pin: str | None, expected: str | None
) -> None:
    """Only a pnpm pin is enforceable, and a Corepack integrity suffix is not a version."""
    web = _web(tmp_path, pin=pin)

    assert install._pinned_pnpm(web) == expected


def test_a_tree_with_no_manifest_has_no_pin_to_enforce(tmp_path: Path) -> None:
    """Older snapshots carry no manifest; refusing there would break the updater."""
    empty = tmp_path / "no-manifest"
    empty.mkdir()

    assert install._pinned_pnpm(empty) is None


def test_a_dev_engines_pin_is_read_too(tmp_path: Path) -> None:
    """pnpm 11's second spelling resolves through the same managed-version path.

    ``devEngines.packageManager`` (pnpm 11.0.0) is the field whose ``onFail``
    decides whether a miss errors, warns or DOWNLOADS, so an exact pin there is
    guarded exactly like ``packageManager`` (QA round 1, Q-5).
    """
    web = _web(
        tmp_path,
        pin=None,
        dev_engines={"packageManager": {"name": "pnpm", "version": "11.22.0"}},
    )

    assert install._pinned_pnpm(web) == "11.22.0"


@pytest.mark.parametrize(
    ("dev_engines", "expected"),
    [
        ({"packageManager": {"name": "pnpm", "version": "^11.5.1"}}, None),
        ({"packageManager": {"name": "npm", "version": "10.9.0"}}, None),
        ({"runtime": {"name": "node", "version": "22.14.0"}}, None),
    ],
)
def test_a_dev_engines_value_equality_cannot_judge_is_not_a_pin(
    tmp_path: Path, dev_engines: dict[str, object], expected: str | None
) -> None:
    """A RANGE (and anything not naming pnpm) is deliberately not enforced.

    ``^11.5.1`` is not a version, so equality cannot judge it and this guard
    protects NOTHING for a range: the host's pnpm is used as-is, matching or not
    (measured — a host pnpm 10.30.3 does not satisfy ``^11.5.1`` and is let
    through; what pnpm then does with the mismatch is pnpm's behaviour, not this
    guard's). Judging one needs a semver implementation, which is not this fix.
    Stated here so the omission is a decision, not a gap.
    """
    web = _web(tmp_path, pin=None, dev_engines=dev_engines)

    assert install._pinned_pnpm(web) == expected


def test_the_package_manager_field_wins_when_both_spellings_are_present(tmp_path: Path) -> None:
    """A manifest with both pins is judged on ``packageManager``.

    It is the field Corepack resolves first and the one these trees carry; a
    manifest whose two pins disagree is a broken manifest rather than a guess this
    guard should make.
    """
    web = _web(
        tmp_path,
        pin="pnpm@11.22.0",
        dev_engines={"packageManager": {"name": "pnpm", "version": "9.1.0"}},
    )

    assert install._pinned_pnpm(web) == "11.22.0"


# ---------------------------------------------------------------------------
# The MEMORY bound (2026-09-21): the wall the time bound cannot be
# ---------------------------------------------------------------------------
#
# THE ARITHMETIC THESE TESTS ARE THE EVIDENCE FOR, in the incident's own numbers.
# `_run_build_step` bounded a build child by TIME — `_BUILD_STEP_TIMEOUT = 600.0` —
# and the tree the incident left behind grew at "+100 processes and +5 GB every
# 25 s", i.e. 0.2 GB/s steady with bursts to 0.4 GB/s (3.2 GB in the first 8 s of
# the reproduction, recorded on `_pin_mismatch`). Do the multiplication:
# 0.2 GB/s x 600 s is on the order of **120 GB** of growth, against a 36 GB host
# and a 32 GB device. The bound is three orders of magnitude too late for the
# failure it was added for — the host is dead long before it fires — and the one
# place the code RELIES on it is `_pin_mismatch`'s deliberate fail-open ("the bound
# protects that case"), which cannot rest on a bound that does not hold at the
# recorded rate.
#
# So the step samples its group's memory and kills the GROUP on breach, reusing
# `memory_guard`'s ceiling and sampler rather than a second budget. Two readings
# discriminate it, on a REAL process group, and they are the same child:
#
#   * GUARDED — the child that grows past the ceiling ends in ~1 s with
#     `StepMemoryExceeded`, and the leader AND its descendant are both gone.
#   * PRE-FIX — with the memory budget disabled (which is `_wait_for_step`'s
#     single blocking `communicate`, byte for byte the code before this change)
#     nothing stops that same child but the clock.
#
# The child is CAPPED at what it takes: it has to cross a ceiling of tens of MB
# quickly, on the host the incident took down, rather than reproduce the incident.

#: The memory driver: ONE ``_run_build_step`` under a pinned ceiling, reporting
#: through a file. Its own driver rather than the abort one above because the
#: ceiling has to be patched in the process that RUNS the step —
#: ``_step_memory_budget`` answers from the real host there, and the real host's
#: ceiling is GB where these tests cross tens of MB — and because a file survives
#: the step's descendants inheriting this process's descriptors.
_MEMORY_DRIVER = '''#!/usr/bin/env python3
"""Generated by tests/unit/mobile/test_bundle_build_bound.py."""

import json
import sys
import time
from pathlib import Path

from local_operator.mobile import install

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ceiling = int(spec["ceiling_mb"])


def _fixture_budget():
    """A pinned ceiling, in the REAL Budget shape (arms and source included).

    ``ceiling_mb=0`` with ``source="disabled"`` is the shape the real computation
    returns for a disabled guard, and it is what makes the PRE-FIX reading exact:
    ``_step_memory_guard`` answers None, so the step falls back to the single
    blocking ``communicate`` the code had before the memory bound existed.
    """
    return install.memory_guard.Budget(
        ceiling_mb=ceiling,
        soft_mb=int(ceiling * 0.8),
        available_mb=4096,
        total_mb=16384,
        reserve_mb=2048,
        source="disabled" if ceiling == 0 else "auto",
        reason="test fixture",
    )


install._step_memory_budget = _fixture_budget

#: Capture the guard the step builds, so the verdict can carry THE NUMBER THE
#: CEILING WAS COMPARED AGAINST rather than a sampler's second opinion of the same
#: group. They are not the same quantity: on macOS the guard refines its reading
#: with the per-process footprint (compressor-inclusive), so it reads ABOVE the raw
#: RSS sum, and under compressor pressure the child's RSS can collapse below what it
#: allocated. See the test that reads this.
_real_step_memory_guard = install._step_memory_guard
_guards = []


def _capturing_step_memory_guard(*args, **kwargs):
    guard = _real_step_memory_guard(*args, **kwargs)
    _guards.append(guard)
    return guard


install._step_memory_guard = _capturing_step_memory_guard

record = {"outcome": None, "message": None, "elapsed": None}
started = time.monotonic()
try:
    result = install._run_build_step(
        [*spec["runner"], "install", "--frozen-lockfile"],
        Path(spec["cwd"]),
        timeout=float(spec["timeout"]),
    )
except install.StepMemoryExceeded as exc:
    record["outcome"] = "memory"
    record["message"] = str(exc)
except BaseException as exc:
    record["outcome"] = type(exc).__name__
    record["message"] = str(exc)
else:
    record["outcome"] = "rc=" + str(result.returncode)
record["elapsed"] = time.monotonic() - started
_guard = _guards[-1] if _guards else None
record["guard_peak_bytes"] = _guard.peak_bytes if _guard is not None else None
record["guard_ceiling_bytes"] = _guard.hard_bytes if _guard is not None else None
Path(spec["out"]).write_text(json.dumps(record), encoding="utf-8")
'''


def _pid_from(path: Path) -> int | None:
    """A pid a child wrote about itself, or None while it is not there yet."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _fixture_budget(ceiling_mb: int) -> memory_guard.Budget:
    """A pinned ceiling in the REAL Budget shape, for tests that run in-process.

    ``ceiling_mb=0`` with ``source="disabled"`` is the shape the real computation
    returns for a disabled guard, and it is what makes the PRE-FIX reading exact.
    """
    return memory_guard.Budget(
        ceiling_mb=ceiling_mb,
        soft_mb=int(ceiling_mb * 0.8),
        available_mb=4096,
        total_mb=16384,
        reserve_mb=2048,
        source="disabled" if ceiling_mb == 0 else "auto",
        reason="test fixture",
    )


def _wait_pids_gone(pids: Sequence[int], timeout: float = 15.0) -> list[int]:
    """Wait for these EXACT pids to leave the table; return the survivors.

    A pid is a stronger reading than a pattern, and a zombie still answers
    ``os.kill(pid, 0)``, so this waits for a real reap rather than for the signal
    to have been sent.
    """
    deadline = time.monotonic() + timeout
    alive: list[int] = []
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            return []
        time.sleep(0.05)
    return alive


def _drive_step(
    tmp_path: Path,
    fake: StandIn,
    web: Path,
    *,
    timeout: float,
    ceiling_mb: int,
    limit: float = 90.0,
) -> tuple[dict[str, Any], int, list[int]]:
    """Run one step through the memory driver: (verdict, group peak, pids).

    The peak is sampled from HERE, against the group of the process the driver's
    step spawned — a reading of the real group rather than a number a child reports
    about itself. The pids come from files the children write about THEMSELVES,
    because ``pgrep -f`` cannot tell the step's leader from the driver: both carry
    the stand-in's path in their argv.
    """
    driver_script = tmp_path / "memory-driver.py"
    driver_script.write_text(_MEMORY_DRIVER, encoding="utf-8")
    spec = tmp_path / "memory-spec.json"
    verdict_file = tmp_path / "verdict.json"
    spec.write_text(
        json.dumps(
            {
                "runner": fake.runner,
                "cwd": str(web),
                "timeout": timeout,
                "ceiling_mb": ceiling_mb,
                "out": str(verdict_file),
            }
        ),
        encoding="utf-8",
    )
    pid_files = (tmp_path / "bin" / "leader.pid", tmp_path / "bin" / "child.pid")
    driver = subprocess.Popen(
        [sys.executable, str(driver_script), str(spec)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    peak = 0
    deadline = time.monotonic() + limit
    try:
        while driver.poll() is None and time.monotonic() < deadline:
            for pid in (_pid_from(path) for path in pid_files):
                if pid is not None and pid not in pids:
                    pids.append(pid)
            if pids:
                rss = memory_guard.group_rss_bytes(pids[0])
                if rss is not None and rss > peak:
                    peak = rss
            time.sleep(0.05)
    finally:
        _kill_if_alive(driver)
    assert verdict_file.exists(), "the memory driver wrote no verdict"
    verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
    return verdict, peak, pids


@pytest.mark.slow
def test_a_fast_allocator_is_killed_at_the_ceiling_not_by_the_clock(
    tmp_path: Path, reap_markers: list[str]
) -> None:
    """The discriminator: the ceiling ends the step in ~1 s where the clock would take 60.

    A group that crosses only in SUM — the leader takes 40 MB, the descendant it
    plants takes 64 MB, against a 110 MB ceiling — so the reading also shows the
    bound is on the GROUP rather than on the leader, which is the same unit the
    kill has to cover.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(
        tmp_path / "bin",
        version="11.22.0",
        token=token,
        linger=600,
        leader_grow_mb=40,
        grow_mb=64,
    )
    web = _web(tmp_path, pin=None)

    verdict, peak, pids = _drive_step(tmp_path, fake, web, timeout=60.0, ceiling_mb=110)

    assert verdict["outcome"] == "memory", verdict
    # Well inside the 60 s this test gave the step, and orders of magnitude inside
    # the 600 s the production constant gives one: the ceiling is what ended it.
    assert verdict["elapsed"] < 10.0, verdict
    # THE GUARD'S OWN PEAK, not this test's ``ps`` reading of the same group: those
    # are different quantities and the ceiling is compared against this one. On
    # macOS the guard refines its reading with the per-process footprint, which
    # includes compressed memory, so it can read well above the raw RSS sum — and
    # under real compressor pressure the child's RSS genuinely COLLAPSES below what
    # it allocated (agent review round 1, MAJOR-2, measured on this host: an RSS
    # sampler that never saw more than 60.6 MB while the guard acted at 124 MB;
    # the same child read 119.2 MB and then 29.0 MB while the guard read 47.9 and
    # 160.4). An RSS floor is therefore a reading of the wrong quantity and goes red
    # for the wrong reason — it did, 7 of 9 whole-file runs at load ≈78, with every
    # behavioural assertion passing.
    guard_peak = verdict["guard_peak_bytes"]
    assert isinstance(guard_peak, int), f"the driver captured no guard reading: {verdict}"
    assert guard_peak >= 110 * 1024 * 1024, (
        f"the guard's own peak ({guard_peak} B) never reached the ceiling it reports "
        "as breached — the kill and the reading would then be different events"
    )
    # The independent sampler is kept, but only as a sanity floor: it has to have
    # seen a real group at all. Its peak is NOT evidence about the ceiling.
    assert peak >= 8 * 1024 * 1024, f"the sampler never saw the group at all: {peak} B"
    message = str(verdict["message"])
    assert message.startswith(install._STEP_MEMORY_HEADER)
    assert "110 MB ceiling" in message
    assert len(pids) == 2, f"expected a leader and a descendant, saw {pids}"
    # The GROUP died: the descendant AND the leader, by exact pid.
    assert _wait_pids_gone(pids) == [], f"a pid from the killed group is still there: {pids}"
    assert _wait_until_gone(token) == []


@pytest.mark.slow
def test_the_time_bound_alone_lets_the_same_child_run_on(
    tmp_path: Path, reap_markers: list[str]
) -> None:
    """The PRE-FIX reading: disable the memory budget and the clock is all there is.

    The same child, the same growth, the same group — with the only change being
    that the budget answers ``disabled``, which makes ``_wait_for_step`` take the
    single blocking ``communicate`` this file had before the memory arm existed.
    Named precisely, because it is easy to read this as a tree comparison: the
    reading is "this branch's ``_run_build_step`` with the memory bound removed",
    which isolates the arm under test; ``fcf56b68`` has no ``_run_build_step`` to
    remove it from, so it cannot be the comparison. The test above asserts
    ``outcome == "memory"`` and an end in ~1 s; THAT assertion fails against this
    reading, which ends when the clock says so and not before. The pair is the
    discriminator — same test, bound removed — rather than two scenarios.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(
        tmp_path / "bin",
        version="11.22.0",
        token=token,
        linger=600,
        leader_grow_mb=40,
        grow_mb=64,
    )
    web = _web(tmp_path, pin=None)

    verdict, peak, pids = _drive_step(tmp_path, fake, web, timeout=3.0, ceiling_mb=0)

    assert verdict["outcome"] == "TimeoutExpired", verdict
    assert verdict["elapsed"] >= 3.0, verdict
    # The child was resident and had taken its whole share while the clock ran:
    # the time bound is not evidence about memory at all.
    assert peak >= 100 * 1024 * 1024, f"group peak while the clock ran: {peak} B"
    assert _wait_pids_gone(pids) == []


@pytest.mark.slow
def test_a_probe_the_ceiling_stopped_is_still_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reap_markers: list[str]
) -> None:
    """The `_pin_mismatch` fail-open rests on the bound, so pin the bound's half.

    `_pin_mismatch` deliberately does NOT refuse an install when the version probe
    cannot answer, on the stated grounds that the bound protects that case. That
    sentence was resting on a 20 s time bound while the probe's own reproduction
    grew at 3.2 GB in 8 s — roughly 8 GB of further growth before the clock could
    fire. This asserts the other half of the fix: a probe stopped by the CEILING
    still reads as "no answer", through `_runner_reports`' own
    `except (OSError, SubprocessError)`, rather than as an exception out of an
    install. Both halves matter: a bound that fails closed here would cost every
    user with an inconclusive probe an install.
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(tmp_path / "bin", version="11.22.0", token=token, linger=600, probe_grow_mb=128)
    monkeypatch.setattr(install, "_step_memory_budget", lambda: _fixture_budget(110))

    started = time.monotonic()
    answer = install._runner_reports(fake.runner)
    elapsed = time.monotonic() - started

    assert answer is None, "a probe the ceiling stopped must read as no answer"
    assert elapsed < 10.0, f"the probe ran {elapsed}s against a {install._PIN_PROBE_TIMEOUT}s bound"
    assert _wait_until_gone(token) == [], "the stopped probe left a descendant behind"


def test_a_sliced_wait_captures_the_same_streams_as_one_blocking_call() -> None:
    """Re-entering ``communicate`` must neither drop nor duplicate the step's output.

    The step's wait is taken in slices now, because nothing can sample a group from
    inside one blocking call, and the streams it collects are what ``_failure_detail``
    reads — a bound that cost the step its own error message would be a worse
    install, not a safer one. CPython documents the half that matters most
    ("retrying communication will not lose any output"); it says nothing about the
    other half, that the bytes already read are not handed over TWICE, and a
    duplicated stream is a garbled failure. Both halves are asserted against the
    same child, so this fails if ``_wait_for_step`` stops being equivalent to the
    call it replaced.
    """
    child = (
        "import sys, time;"
        "sys.stdout.write('A' * 100000); sys.stdout.flush();"
        "sys.stderr.write('E' * 50000); sys.stderr.flush();"
        "time.sleep(0.4);"
        "sys.stdout.write('B' * 100000); sys.stdout.flush()"
    )
    # A ceiling this test cannot reach: the subject here is the STREAMS, and the
    # guard exists only to put the wait on its sliced path.
    budget = memory_guard.Budget(4096, 3276, 4096, 16384, 2048, "auto", "test fixture")
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        guard = memory_guard.Guard(proc.pid, budget, tick_s=0.05)
        out, err, report = install._wait_for_step(proc, guard, 30.0)
    finally:
        _kill_if_alive(proc)

    assert report is None, "the ceiling fired in a test that set it out of reach"
    assert out.count("A") == 100000, "the first write was lost across slices"
    assert out.count("B") == 100000, "the last write was lost across slices"
    assert err.count("E") == 50000


def test_a_memory_kill_is_worded_at_both_package_runner_call_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of the package-manager path WORD a memory kill; neither RAISES one.

    ``corepack enable`` runs through :func:`_run_build_step`, so it can be killed for
    memory, and it is reached from two call sites: :func:`_build_bundle` (whose catch
    this change widened) and :func:`snapshot_bundle` (whose catch it did not — the
    twin that let ``StepMemoryExceeded`` escape ``lop update``'s snapshot step as a
    traceback instead of the sentence that step exists to print). The fault is
    injected at ``_package_runner``, so no process runs and the subject is exactly
    the catch; the KILL itself is covered on real groups by the other tests here.
    """
    web = _web(tmp_path, pin=None)

    def _killed(*_args: object, **_kwargs: object) -> None:
        raise install.StepMemoryExceeded(
            f"{install._STEP_MEMORY_HEADER}: synthetic (no process ran)"
        )

    monkeypatch.setattr(install, "_package_runner", _killed)

    # The twin: named with the step that failed, exactly as the time bound is.
    error = install._build_bundle(web)
    assert error is not None, "a step killed for memory must not read as success"
    assert "bundle build failed" in error
    assert install._STEP_MEMORY_HEADER in error

    # The site agent review round 1 found: `lop update` prints this function's
    # return value, so a raise here is a traceback in the operator's face.
    detail = install.snapshot_bundle(web)
    assert detail == "skipped (pnpm could not be prepared; build at `lop mobile install`)", detail


def test_the_step_budget_is_the_shared_arithmetic_with_the_build_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sharing itself, pinned: a second copy of the arithmetic would pass every other test here.

    ``0.5 x available minus reserve``, the reserve constants and the swap-pressure
    floor have ONE owner (``memory_guard``), and that is why this path is wiring
    rather than a second guard. Nothing behavioural in this file would notice a
    re-implementation downstream, so the call is asserted directly: the shared
    function, with this path's floor, and nothing else.
    """
    seen: dict[str, object] = {}
    original = memory_guard.compute_budget

    def spy(**kwargs: object) -> memory_guard.Budget:
        seen.update(kwargs)
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(install.memory_guard, "compute_budget", spy)
    budget = install._step_memory_budget()

    assert seen.get("floor_mb") == install._STEP_MEMORY_FLOOR_MB
    assert budget.source in {"auto", "disabled"}
    # Measured, not assumed: a package-manager child costs more before it does
    # anything (pnpm --version peaks at ~121 MB here) than the smallest command the
    # bash tool runs, so this path's floor has to sit above that one's.
    assert install._STEP_MEMORY_FLOOR_MB > memory_guard._MIN_CEILING_MB
    assert install._BUILD_MEMORY_KILL_GRACE < install._BUILD_KILL_GRACE
