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
