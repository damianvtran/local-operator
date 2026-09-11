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
from contextlib import suppress
from pathlib import Path
from typing import cast

import pytest

from local_operator.secrets import client
from local_operator.secrets.keys import secrets_dir
from local_operator.secrets.protocol import _runtime_fallback_dir, socket_path
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


def test_a_broker_survives_the_removal_of_its_config_dir(config_root: Path) -> None:
    """Why the sweep captures PATHS: the socket does not live in the config dir.

    pytest reclaims ``tmp_path`` before the autouse sweep is finalised, so the
    sweep runs with the directory gone. The socket it needs is in the fallback
    runtime dir — a config dir deep enough to need one, which a pytest tmp_path
    always is — and the fallback path is derived from the config dir's NAME, not
    from anything on disk. So naming a deleted directory is still enough to find
    and stop its broker, which is the property the call-phase capture relies on
    and the one this test pins.
    """
    pid = _start(config_root)
    try:
        shutil.rmtree(config_root)
        assert not config_root.exists(), "the config dir was expected to be gone"
        assert socket_path(
            config_root
        ).exists(), "the fallback socket must outlive the config dir it was derived from"
        assert (client.broker_status(config_root) or {}).get("pid") == pid
        _stop_brokers_in([config_root])
        assert not client.is_running(
            config_root
        ), "a broker whose config dir is gone must still be reapable by name"
    finally:
        _kill(config_root)


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
