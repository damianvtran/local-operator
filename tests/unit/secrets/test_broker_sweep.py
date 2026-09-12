"""The suite-wide broker sweep: what it must reap, and what it must never touch.

These are not unit tests of a fixture but of the behaviour the fixture hides.
The sweep in ``tests/conftest.py`` is the only thing standing between a test run
and a key-holding daemon left alive on the operator's machine, and it silently
stopped finding its candidates when it began running after pytest had already
reclaimed ``tmp_path`` — measured: one run of ``tests/unit/secrets/test_cli.py``
left 28 live brokers, one per store-touching test, each holding a master key in
memory.

They start REAL brokers, because what broke was the path arithmetic and not a
mock's return value: `socket_path` is a pure function of the config dir NAME, so
the sweep can still reach a broker whose directory pytest has deleted, and only
a real daemon proves that.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import cast

import pytest

from local_operator.secrets import client
from local_operator.secrets.keys import secrets_dir
from local_operator.secrets.protocol import _runtime_fallback_dir, socket_path
from local_operator.session.runtime.registry import pid_alive
from tests.conftest import _SWEEP_ROOT_KEY, _secret_config_dirs, _stop_brokers_in


def _start(base: Path) -> int:
    """Start a real broker for ``base`` and return its pid."""
    assert client.ensure_broker(base), "the broker never came up"
    status = client.broker_status(base) or {}
    pid = status.get("pid")
    assert isinstance(pid, int), status
    return pid


def _kill(base: Path) -> None:
    """Stop a broker this test started, so the test never leaks one itself.

    Deliberately the client's own status/pid route rather than `_stop_brokers_in`,
    except where a test is *about* the sweep: a test that used the code under
    test to clean up could not tell a broken sweep from a leaked broker.
    """
    status = client.broker_status(base) or {}
    pid = status.get("pid")
    if isinstance(pid, int):
        with suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def _uses_the_fallback_socket(base: Path) -> bool:
    """Which layout `socket_path` chose for ``base`` — asked, not assumed.

    A deep config dir cannot fit its socket under ``sun_path`` (104 bytes) and
    gets the ``$TMPDIR/lop-secrets-<uid>-<digest>`` runtime dir; a shallow one
    keeps the socket INSIDE ``<base>/secrets``. Which one a test sees is a
    property of the machine's tmp depth — ``TMPDIR=/tmp`` on Linux CI against the
    long ``/var/folders`` path here — so a test asserting either layout is
    asserting the box it runs on. Ask the code what it did, and assert that.
    """
    return socket_path(base).parent == _runtime_fallback_dir(secrets_dir(base))


def _deep_config_dir(tmp_path: Path) -> Path:
    """A config dir whose socket MUST take the ``$TMPDIR`` fallback layout.

    For the tests whose SUBJECT is that layout: constructing it makes them valid
    on every runner instead of only where the tmp depth happens to be enough,
    which is what made an earlier version of this file pass locally and fail on
    CI. The assert is the construction's own guard — without it, a later change
    to the nesting could quietly hand the test the other layout instead.
    """
    deep = tmp_path / ("d" * 60) / ("e" * 60) / "config"
    deep.mkdir(parents=True)
    assert _uses_the_fallback_socket(
        deep
    ), "the construction failed: this config dir took the in-directory layout"
    return deep


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    """Wait for a signalled daemon to stop being a process at all.

    `_stop_brokers_in` waits for the daemon to stop ANSWERING, which happens as
    it closes its listener — a step before it exits. The zombie probe is what
    tells those apart: signal-0 alone reports a zombie as alive (the trap
    `registry.pid_alive` documents at length), and a daemon this process started
    stays a zombie until this process reaps it. Asserting the kill without it
    would flap on the scheduling of an exit already under way.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid, check_zombie=True):
            return True
        time.sleep(0.05)
    return False


def _kill_pid(pid: int) -> None:
    with suppress(OSError):
        os.kill(pid, signal.SIGTERM)


def test_the_sweep_reaps_a_broker_under_a_candidate(config_root: Path) -> None:
    """The happy path, against a real daemon and a real socket.

    ``config_root`` is ``tmp_path/config`` with HOME redirected — the exact shape
    ``test_cli.py`` uses — so this is the configuration whose broker the sweep
    used to leave behind.
    """
    _start(config_root)
    try:
        assert client.is_running(config_root), "the broker is not reachable before the sweep"
        _stop_brokers_in([config_root])
        assert not client.is_running(config_root), "the sweep left the broker running"
        # What the sweep owes on the FILESYSTEM depends on the layout it was
        # handed: with the socket in the runtime dir, that dir is the sweep's to
        # remove; with the socket inside the config dir, the file lives in a
        # directory that is not the sweep's to unlink (pytest's tmp_path fixture
        # removes it, and the daemon's death is the contract asserted above).
        if _uses_the_fallback_socket(config_root):
            assert not socket_path(config_root).exists(), "the socket outlived the broker"
            assert not _runtime_fallback_dir(
                secrets_dir(config_root)
            ).exists(), "the runtime directory outlived the broker"
    finally:
        _kill(config_root)


def test_the_sweep_leaves_a_broker_it_was_not_asked_about_alone(
    tmp_path: Path, config_root: Path
) -> None:
    """The safety property: the sweep is scoped to the candidates it is given.

    A broker under a config dir the candidate list does not name stands in for
    the two processes the sweep must never kill — another agent's live session,
    and the operator's own store under their real ``~/.local-operator``. Both are
    reachable by name and neither is a candidate, which is the whole reason the
    sweep takes a list instead of walking the process table. Driven against a
    real daemon because that is the only way to catch a "reap" that goes looking
    beyond its list.
    """
    other = tmp_path / "another-live-session"
    other.mkdir()
    _start(config_root)
    _start(other)
    try:
        _stop_brokers_in([config_root])
        assert not client.is_running(config_root), "the sweep missed its own candidate"
        assert client.is_running(other), (
            "the sweep killed a broker outside its candidate list — a live session's "
            "store, or the operator's, is not this suite's to reap"
        )
    finally:
        _kill(config_root)
        _kill(other)


def test_a_broker_survives_the_removal_of_its_config_dir(tmp_path: Path) -> None:
    """Why the sweep records PATHS: a fallback socket does not live in the dir.

    For the layout where a sweep running at teardown CAN still work: pytest has
    just reclaimed the config directory, and the socket is in the runtime dir
    derived from that directory's NAME rather than from anything on disk, so
    naming a deleted directory is still enough to find and stop its broker.

    The deep directory is CONSTRUCTED rather than inherited from ``tmp_path``:
    on a runner whose tmp path is short enough (Linux CI runs ``TMPDIR=/tmp``)
    the socket stays inside the config dir, dies with it, and this test's premise
    is simply false there — which is what took CI red on this file. The other
    layout is pinned by
    `test_an_in_directory_socket_is_unreachable_once_its_config_dir_is_gone`.
    """
    base = _deep_config_dir(tmp_path)
    pid = _start(base)
    try:
        shutil.rmtree(base)
        assert not base.exists(), "the config dir was expected to be gone"
        assert socket_path(
            base
        ).exists(), "the fallback socket must outlive the config dir it was derived from"
        assert (client.broker_status(base) or {}).get("pid") == pid
        _stop_brokers_in([base])
        assert _wait_gone(pid), "a broker whose config dir is gone must still be reapable by name"
    finally:
        _kill(base)
        _kill_pid(pid)


def test_an_in_directory_socket_is_unreachable_once_its_config_dir_is_gone() -> None:
    """The measurement behind the call-phase reap, pinned in-repo.

    A SHORT config dir needs no fallback, so its socket lives inside its own
    ``secrets`` directory. pytest's ``tmp_path`` fixture removes that directory at
    its teardown, the socket file goes with it, and from then on no path-derived
    lookup can reach the daemon — which is why a sweep running only at teardown
    left 31 live key-holding brokers from one run of ``test_cli.py`` on exactly
    this layout. That is why the reap happens while the socket still exists.

    ``/tmp`` deliberately, not ``tmp_path``: the subject is a path short enough to
    stay under ``sun_path`` (104 bytes), and a pytest tmp_path is not by
    construction. The daemon is stopped by pid here — nothing in the suite can
    reach it once the directory is gone, which is the point.
    """
    scratch = Path(tempfile.mkdtemp(prefix="lop-sweep-short-", dir="/tmp"))
    try:
        base = scratch / "config"
        base.mkdir()
        assert not _uses_the_fallback_socket(
            base
        ), "the test needs the in-directory layout, and /tmp is short enough for it"
        pid = _start(base)
        try:
            shutil.rmtree(base)
            assert not socket_path(
                base
            ).exists(), "the test's premise: the socket lived inside the config dir"
            _stop_brokers_in([base])
            assert pid_alive(pid, check_zombie=True), (
                "a sweep can no longer reach this daemon, so the reap has to happen "
                "while its socket exists — if this ever fails, the sweep got smarter "
                "and the call-phase reap can be reconsidered"
            )
        finally:
            _kill_pid(pid)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_the_sweep_does_nothing_where_the_broker_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The platform guard, driven by its effect.

    The broker is a POSIX daemon — `local_operator.secrets.client` imports
    `fcntl` at module scope — and the sweep now runs for every test from the call
    phase as well as from teardown, so an unguarded call fails an entire Windows
    run rather than one test: that is how the `filesystem-boundaries-windows` job
    caught the first version of the call-phase reap, which had the guard only in
    the fixture.

    Pinned through the named seam rather than by mutating `os.name`, which would
    have `pathlib` hand out `WindowsPath` objects on this host; a REAL broker under
    a named candidate is what proves the guard: it survives, which it could not if
    the import were reached (it raises there) or the kill were.
    """
    base = tmp_path / "config"
    base.mkdir()
    pid = _start(base)
    try:
        monkeypatch.setattr("tests.conftest._broker_daemon_is_available", lambda: False)
        _stop_brokers_in([base])
        assert client.is_running(base), "the sweep acted on a platform where it must not"
    finally:
        _kill(base)
        _kill_pid(pid)


def test_the_sweep_never_signals_its_own_pid_or_a_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusals that keep the reap from killing the run it is part of.

    Driven through a spy on `os.kill` with the two things the sweep reads faked,
    because the failure this pins is process death: `test_broker.py`'s `broker`
    fixture serves the broker IN-PROCESS, so its status reports the test
    process's own pid, and the first version of the call-phase reap signalled it
    — `worker 'gw0' crashed while running ...`, 17 failed on the 3.12 shards, and
    a whole `-n0` run dead with it. A test that actually delivered those signals
    could only report its own disappearance, so the spy does the asserting.

    `0` and `-1` are the other half: `kill(0, ...)` signals this process's entire
    GROUP and `kill(-1, ...)` every process this user may signal, so a record
    carrying one is never a broker to reap. The positive control (a pid far
    outside any real range) proves the spy sees a signal when there is one to
    send, so the three refusals are not passing by inertness.
    """
    base = tmp_path / "config"
    base.mkdir()
    decoy = tmp_path / "broker.sock"
    decoy.touch()
    sent: list[tuple[int, int]] = []
    real_kill = os.kill

    def spy(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if pid == 999_999_999:  # positive control: nothing to signal, don't try
            raise ProcessLookupError
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", spy)
    monkeypatch.setattr("local_operator.secrets.protocol.socket_path", lambda base=None: decoy)
    for pid in (os.getpid(), 0, -1):
        monkeypatch.setattr(
            "local_operator.secrets.client.broker_status",
            lambda base=None, _pid=pid: {"ok": True, "pid": _pid},
        )
        sent.clear()
        _stop_brokers_in([base])
        assert sent == [], (
            f"the sweep signalled {sent} for pid {pid} — its own process, or a whole "
            "process group, instead of a separate broker"
        )

    monkeypatch.setattr(
        "local_operator.secrets.client.broker_status",
        lambda base=None: {"ok": True, "pid": 999_999_999},
    )
    sent.clear()
    _stop_brokers_in([base])
    assert sent == [
        (999_999_999, signal.SIGTERM)
    ], "the positive control sent nothing, so the refusals above prove nothing"


class _StubNode:
    """The two attributes `_secret_config_dirs` reads off a pytest item."""

    def __init__(self, funcargs: dict[str, object], stash: pytest.Stash) -> None:
        self.funcargs = funcargs
        self.stash = stash


class _StubRequest:
    """A stand-in for the `pytest.FixtureRequest` the sweep is handed.

    `_secret_config_dirs` reads only `request.node.funcargs` and `request.node.stash`,
    so a stub is what lets this test put the sweep in the exact teardown-time state
    (recorded paths, directories gone) without running a nested pytest session.
    """

    def __init__(self, node: _StubNode) -> None:
        self.node = node


def test_the_candidates_keep_paths_whose_directories_are_already_gone(
    tmp_path: Path, config_root: Path
) -> None:
    """The regression guard for the leak, at the level the leak actually happened.

    A teardown-time walk of ``tmp_path`` finds nothing once pytest has reclaimed
    it — which is what made the sweep a no-op for every test that redirects its
    config dir into ``tmp_path``. The recorded paths are what carry the sweep
    over that window, so the candidate list must contain a directory that no
    longer exists.
    """
    gone = tmp_path / "test_something0"
    gone.mkdir()
    (gone / "config").mkdir()
    candidate = gone / "config"

    stash: pytest.Stash = pytest.Stash()
    stash[_SWEEP_ROOT_KEY] = (gone, candidate)
    # `funcargs` still names the path, as pytest's does, but the directory it
    # points at is gone — exactly the teardown-time state.
    node = _StubNode({"tmp_path": gone}, stash)
    shutil.rmtree(gone)

    candidates = _secret_config_dirs(
        cast(pytest.FixtureRequest, _StubRequest(node)), tmp_path / "home"
    )

    assert candidate in candidates, (
        "a candidate recorded while the test ran is missing from the list, so its "
        "broker would never be reaped"
    )
    assert not gone.exists(), "the test's premise: the directories are gone by teardown"


def _run_nested_pytest(
    tmp_path: Path, body: str, *, extra_argv: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Run one test file under a REAL nested pytest that loads this repo's conftest.

    The end-to-end tests below cannot be expressed in-process: what they pin is
    pytest's own FINALISATION ORDER — ``tmp_path`` is torn down before an autouse
    fixture declared in ``tests/conftest.py``, because that fixture is set up
    first and finalisers run in reverse — and a test cannot observe its own
    teardown. A nested run can: the inner test records what it started, the outer
    one reads the record after the inner session has finished tearing down.

    ``pytest_plugins = ["tests.conftest"]`` rather than a copy of the fixture:
    the subject is the SHIPPED conftest, so anything that reproduces it here
    would pass while the real one leaked — which is precisely the failure issue
    #958 describes. ``-n0`` keeps the inner session single-process so the pid the
    inner test writes is the broker's parent-visible one, and ``-p
    no:cacheprovider`` keeps the inner run from writing a cache into the repo.

    **``tmp_path_retention_policy = "failed"`` is load-bearing, not inherited.**
    A nested run given its own ``-c`` ini does NOT pick up the root
    ``pyproject.toml``, and that setting is the entire precondition for the bug:
    under it, `tmp_path`'s own finaliser REMOVES the directory when the test
    passed, before an autouse fixture declared in ``tests/conftest.py`` is
    finalised — so a teardown-time walk finds nothing. Verified both ways here:
    with the default policy the directory still exists at teardown and a
    teardown-only sweep would have worked, which would have made this test pass
    against the very code it is meant to catch.
    """
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "conftest.py").write_text('pytest_plugins = ["tests.conftest"]\n')
    (inner / "test_inner.py").write_text(body)
    # Mirrors the root `pyproject.toml`'s `[tool.pytest.ini_options]` for the two
    # settings this reproduction depends on. `-c` below points the inner run at
    # it, which also stops it inheriting the root `addopts` (`-n auto`).
    (inner / "pytest.ini").write_text(
        "[pytest]\ntmp_path_retention_policy = failed\ntmp_path_retention_count = 3\n"
    )
    repo_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    # The inner interpreter must import BOTH `local_operator` and `tests.conftest`
    # from this worktree — see AGENTS.md on a subprocess resolving the root
    # checkout's copy when it is launched with the wrong path.
    environment["PYTHONPATH"] = str(repo_root)
    environment["LO958_RECORD"] = str(tmp_path / "record.txt")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(inner / "test_inner.py"),
            "-q",
            "-n0",
            "-p",
            "no:cacheprovider",
            "-c",
            str(inner / "pytest.ini"),
            f"--basetemp={tmp_path / 'inner-basetemp'}",
            *extra_argv,
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


#: The inner test for the end-to-end case: start a real broker under the config
#: dir the test itself uses, and record its pid for the outer test to check.
_INNER_STARTS_A_BROKER = '''
import os
from pathlib import Path

from local_operator.secrets import client


def test_starts_a_broker(tmp_path, monkeypatch):
    """Stands in for every test that touches the store: it leaves a broker up."""
    base = tmp_path / "config"
    base.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    assert client.ensure_broker(base), "the broker never came up"
    pid = (client.broker_status(base) or {}).get("pid")
    assert isinstance(pid, int), "no pid to hand to the outer test"
    Path(os.environ["LO958_RECORD"]).write_text(f"{pid}\\n{base}")
'''


def test_a_broker_a_test_started_is_dead_once_that_test_has_torn_down(tmp_path: Path) -> None:
    """The end-to-end regression guard for issue #958, at the level that broke.

    Every other test in this file drives `_stop_brokers_in` or the candidate list
    DIRECTLY, which is why the leak survived them all: the helper was correct and
    the sweep still reaped nothing, because the candidate list it was given at
    teardown had already collapsed to the default-config-dir entry. The bug lived
    in the ORDERING, so only a test that submits to the real ordering can catch a
    regression of it.

    Fails against the pre-#985 conftest (candidate discovery at teardown only)
    and passes with the call-phase reap; that A/B is in the PR for #958.

    Asserted through `_wait_gone` rather than a bare `pid_alive` for the reason
    that helper documents, and it matters MORE here: the sweep waits for the
    daemon to stop ANSWERING, which happens as it closes its listener — a step
    before it exits — so the broker is reliably still a process for a moment
    after the inner pytest returns (measured on this machine: alive at t+0s,
    gone by t+0.25s). A bare probe would flap on that scheduling. The bounded
    wait still separates the two outcomes the test is about, because a LEAKED
    broker does not exit at all: it idles for 30 minutes.
    """
    result = _run_nested_pytest(tmp_path, _INNER_STARTS_A_BROKER)
    assert result.returncode == 0, f"the inner run failed:\n{result.stdout}\n{result.stderr}"

    record = (tmp_path / "record.txt").read_text().splitlines()
    pid, base = int(record[0]), Path(record[1])
    try:
        assert _wait_gone(pid), (
            f"broker {pid} for {base} outlived the test that started it — the sweep "
            "saw no candidate naming this test's config dir (issue #958)"
        )
    finally:
        _kill_pid(pid)


#: The inner test for the session-end net: a MODULE-SCOPED fixture starts a
#: broker in its teardown, under a directory from ``tmp_path_factory``. That
#: directory is in no test's ``tmp_path``, so no per-test record can name it and
#: the module finaliser runs after the last test's function-scoped sweep — the
#: one shape neither per-test mechanism can reach. Verified to leak without the
#: net: the broker was still alive a second after the inner run returned.
_INNER_STARTS_A_BROKER_AT_TEARDOWN = """
import os
from pathlib import Path

import pytest

from local_operator.secrets import client


@pytest.fixture(scope="module")
def module_scoped_store(tmp_path_factory):
    base = tmp_path_factory.mktemp("modcfg")
    yield base
    # Module teardown: after the last test's function-scoped autouse sweep, in a
    # directory that was never any test's tmp_path. Only the session-end net
    # sees this one.
    assert client.ensure_broker(base)
    pid = (client.broker_status(base) or {}).get("pid")
    Path(os.environ["LO958_RECORD"]).write_text(f"{pid}\\n{base}")


def test_touches_the_store(module_scoped_store, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(module_scoped_store))
"""


def test_the_session_end_net_reaps_a_broker_started_after_the_per_test_reap(
    tmp_path: Path,
) -> None:
    """The case neither per-test mechanism can see, and the count the net reports.

    A module-scoped fixture's directory comes from ``tmp_path_factory`` and so
    belongs to no test's ``tmp_path``: no per-test record can name it, and its
    finaliser runs after the last function-scoped sweep. That is a structural
    gap rather than a bug in the per-test reap, and it is what
    `pytest_sessionfinish` covers.

    Issue #958 asked for the net to REPORT what it reclaimed, so a leak past the
    per-test reap stays visible instead of being silently absorbed; the reported
    count is asserted for that reason rather than as decoration.
    """
    result = _run_nested_pytest(tmp_path, _INNER_STARTS_A_BROKER_AT_TEARDOWN)
    assert result.returncode == 0, f"the inner run failed:\n{result.stdout}\n{result.stderr}"

    record = (tmp_path / "record.txt").read_text().splitlines()
    pid, base = int(record[0]), Path(record[1])
    try:
        assert _wait_gone(pid), (
            f"broker {pid} for {base} survived the session it was started in — the "
            "session-end net did not reach it"
        )
        assert "[broker-sweep] session-end net reclaimed 1 live broker" in result.stderr, (
            "the net reaped the broker but reported nothing, so a suite leaking "
            f"past the per-test reap would look clean:\n{result.stderr}"
        )
    finally:
        _kill_pid(pid)


def test_a_spawned_broker_names_itself_in_the_process_listing(config_root: Path) -> None:
    """Issue #958's open question: the branded child in a teardown listing.

    Every pre-#954 CI teardown listed a bare ``Local Operator`` child and nothing
    said what it was. It was this: `_spawn_broker` launches ``[sys.executable,
    "-m", ...]``, and in any process that has been through
    `procname.reexec_branded` — every real ``lop`` launch — ``sys.executable`` IS
    the branded hardlink, so the broker inherited the product name alone.

    Asserted on ``argv``, deliberately, because that is the axis a teardown dump
    reads and the only one that survives: Linux ``comm`` truncates at 15 bytes,
    so ``Local Operator`` and every labelled form are indistinguishable there.
    The argv is read back from the REAL spawned process rather than from
    `_broker_argv0`, so a label that never reaches the child fails this.

    The branded-``sys.executable`` precondition is macOS-only (a branded image is
    only planted there — `procname.branded_link_path` returns None elsewhere), so
    this test pins the part that holds EVERYWHERE: the label reaches the child's
    argv whatever the image is. The branded-parent capture is in the PR for #958.
    """
    pid = _start(config_root)
    try:
        listing = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        assert "[secret broker]" in listing, (
            f"a spawned broker is unidentifiable in `ps`: {listing!r} — this is the "
            "unexplained branded child of issue #958"
        )
        # The digest ties the row to the socket directory on disk, which is what
        # makes the label actionable rather than merely decorative; and it is a
        # digest and not the path because argv is world-readable.
        digest = _runtime_fallback_dir(secrets_dir(config_root)).name.rsplit("-", 1)[-1]
        assert f"store={digest}" in listing, (
            f"the label does not name this broker's store: {listing!r}; expected the "
            f"digest {digest} that also names its runtime directory"
        )
        assert "local_operator.secrets.brokerd" in listing, (
            "the module must stay in argv: it is how the issue's reopen condition "
            f"('argv is not -m local_operator.secrets.brokerd') is checked: {listing!r}"
        )
    finally:
        _kill(config_root)
        _kill_pid(pid)
