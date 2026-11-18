"""A process may stop only the broker IT started, and only while it is alone.

WHY THIS FILE EXISTS. The broker is detached on purpose -- it must outlive the
session that lazily started it -- so nothing ever stopped one. That is right
for an operator's store and wrong for an evaluation episode's SCRATCH store:
every episode that resolved a secret left one ``secrets.brokerd`` behind
(measured on v0.107.2: one orphan per OSWorld episode, each holding its socket
until the 30-minute idle timer). ``client.stop_spawned_broker`` is the narrow
primitive the episode host uses to take that lifetime back, and these tests
pin the half that matters more than the stop itself: every way it must REFUSE.

The broker here is a FAKE (a pid, a liveness flag and a status dict behind the
client's own seams), so both branches are provable without a daemon: the stop
is observed as the exact ``os.kill(pid, SIGTERM)`` the client issued.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import pytest

from local_operator.secrets import client

FAKE_BROKER_PID = 424242

# The autouse ``refuse_real_secret_broker_spawn`` guard replaces
# ``ensure_broker`` wholesale, and ``ensure_broker`` recording its spawn is the
# subject here. Opting in is safe because no cell can start a real daemon:
# every test builds a ``FakeBroker`` first, which replaces the ``_spawn_broker``
# seam (and ``os.kill``) before anything calls ``ensure_broker``.
pytestmark = pytest.mark.allow_secret_broker


class FakeBroker:
    """The broker as the client's seams see it: spawnable, pingable, killable."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, running: bool = False) -> None:
        self.running = running
        self.pid = FAKE_BROKER_PID
        self.sessions: list[int] = []
        self.spawns = 0
        self.kills: list[tuple[int, int]] = []
        #: Whether SIGTERM actually ends it (False models a wedged daemon).
        self.dies_on_term = True
        monkeypatch.setattr(client, "_spawn_broker", self._spawn)
        monkeypatch.setattr(client, "is_running", lambda base=None: self.running)
        monkeypatch.setattr(client, "broker_status", self._status)
        monkeypatch.setattr(client.os, "kill", self._kill)
        # The registry is process-global by design; a test must start empty.
        monkeypatch.setattr(client, "_SPAWNED_BROKERS", {})

    def _spawn(self, base: Path | None) -> int:
        del base
        self.spawns += 1
        self.running = True
        return self.pid

    def _status(self, base: Path | None = None) -> dict[str, Any] | None:
        del base
        if not self.running:
            return None
        return {"ok": True, "pid": self.pid, "locked": False, "sessions": list(self.sessions)}

    def _kill(self, pid: int, signum: int) -> None:
        self.kills.append((pid, signum))
        if self.dies_on_term:
            self.running = False


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return config


def test_a_broker_this_process_started_is_stopped_by_its_exact_pid(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = FakeBroker(monkeypatch)
    assert client.ensure_broker(base) is True
    assert broker.spawns == 1
    [spawned] = client.spawned_brokers().values()
    assert spawned.pid == FAKE_BROKER_PID

    assert client.stop_spawned_broker(base) == "stopped"
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]
    # Forgotten once settled: a second call can never signal a recycled pid.
    assert client.spawned_brokers() == {}
    assert client.stop_spawned_broker(base) == "not-spawned-here"
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]


def test_a_broker_merely_attached_to_is_never_touched(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's own broker: already listening, so nothing was spawned."""

    broker = FakeBroker(monkeypatch, running=True)
    assert client.ensure_broker(base) is True
    assert broker.spawns == 0
    assert client.spawned_brokers() == {}

    assert client.stop_spawned_broker(base) == "not-spawned-here"
    assert broker.kills == []
    assert broker.running is True


def test_a_replacement_broker_on_the_same_socket_is_never_touched(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ours exited and someone else's now holds the socket: not ours to stop."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    broker.pid = FAKE_BROKER_PID + 1

    assert client.stop_spawned_broker(base) == "replaced"
    assert broker.kills == []
    assert client.spawned_brokers() == {}


def test_a_broker_other_sessions_registered_with_is_left_running(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started here, then adopted: a session that is not us depends on it."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    broker.sessions = [os.getpid(), 99991]

    assert client.stop_spawned_broker(base) == "in-use"
    assert broker.kills == []
    assert broker.running is True


def test_our_own_registered_session_does_not_keep_it_alive(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    broker.sessions = [os.getpid()]

    assert client.stop_spawned_broker(base) == "stopped"
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]


def test_a_broker_that_already_exited_is_reported_gone_without_a_signal(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    broker.running = False  # idle timer, crash, or the operator stopped it

    assert client.stop_spawned_broker(base) == "already-gone"
    assert broker.kills == []


def test_a_forked_child_does_not_inherit_the_spawn(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry is copied by ``fork``; ownership must not be."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    monkeypatch.setattr(client.os, "getpid", lambda: 31337)

    assert client.spawned_brokers() == {}
    assert client.stop_spawned_broker(base) == "not-spawned-here"
    assert broker.kills == []


def test_a_passphrase_tier_broker_is_never_stopped(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its memory is the only copy of the operator's unlock; nothing recreates it."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    monkeypatch.setattr("local_operator.secrets.keys.key_mode", lambda base=None: "passphrase")

    assert client.stop_spawned_broker(base) == "holds-unlock"
    assert broker.kills == []


def test_a_wedged_broker_is_reported_not_claimed_stopped(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = FakeBroker(monkeypatch)
    broker.dies_on_term = False
    client.ensure_broker(base)

    assert client.stop_spawned_broker(base, timeout=0.2) == "timeout"
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]
