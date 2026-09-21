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
PR's process listings rather than by a test. The abort and external-stop tests fail before their own subject as well,
and not for the defect's sake: the driver they drive calls ``_run_build_step``,
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
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import pytest

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

if CONFIG.get("deaf"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

if "--version" in sys.argv[1:]:
    print(CONFIG["version"])
    raise SystemExit(0)

if CONFIG.get("dist"):
    dist = Path(os.getcwd()) / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")

if CONFIG.get("fork"):
    subprocess.Popen(
        [sys.executable, "-c", GRANDCHILD, CONFIG["token"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(CONFIG.get("linger", 600))

raise SystemExit(CONFIG.get("exit", 1))
'''


class StandIn:
    """A generated pnpm stand-in inside a temp dir, plus its recorded calls."""

    def __init__(self, root: Path, **config: object) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.script = root / "stand-in-pnpm"
        self.script.write_text(_FAKE_RUNNER, encoding="utf-8")
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
"""Generated by tests/unit/mobile/test_bundle_build_bound.py."""

import os
import sys
from pathlib import Path

from local_operator.mobile import install

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


def _write_driver(root: Path) -> Path:
    driver = root / "step-driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    return driver


def _start_driver(root: Path, fake: "StandIn", web: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(_write_driver(root)), *fake.runner, str(web)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_if_alive(proc: "subprocess.Popen[bytes]") -> None:
    """Reap a driver that did not end on its own, by its exact pid."""
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
    """
    token = _token()
    reap_markers.append(token)
    fake = StandIn(tmp_path / "bin", version="11.22.0", fork=True, token=token, linger=600)
    web = _web(tmp_path, pin=None)
    proc = _start_driver(tmp_path, fake, web)
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
    tmp_path: Path,
) -> None:
    """The recursion's engine, refused: no build step is ever started.

    ``package.json`` pins pnpm@11.22.0 and the runner reports 10.30.3. Asserted on
    the refusal AND on the absence of any ``install``/``build`` invocation,
    because a refusal that still spawned pnpm would leave the engine running.
    """
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

    error = install._build_bundle(web, fake.runner)

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
