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


def test_a_slow_first_spawn_that_wins_the_socket_is_still_owned(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #4236 QA Q1, on real daemons: ``ensure#1 -> False [83910]``,
    ``ensure#2 -> False [83911]``, socket then held by 83910, outcome
    ``replaced`` -- the FIRST spawn left listening with ppid 1.

    A daemon slower to listen than ``ensure_broker`` waits makes the next call
    spawn again, and the record kept only the latest pid. The first one then
    bound the socket, so the owner compared the live pid with a daemon that had
    already exited and disowned the one it leaked.
    """

    broker = FakeBroker(monkeypatch)
    spawned = iter((FAKE_BROKER_PID, FAKE_BROKER_PID + 1))

    def slow_spawn(base: Path | None) -> int:
        del base
        broker.spawns += 1
        return next(spawned)  # started, but not listening inside the wait

    monkeypatch.setattr(client, "_spawn_broker", slow_spawn)
    assert client.ensure_broker(base, timeout=0.01) is False
    assert client.ensure_broker(base, timeout=0.01) is False
    assert broker.spawns == 2
    [entry] = client.spawned_brokers().values()
    assert entry.pids == (FAKE_BROKER_PID, FAKE_BROKER_PID + 1)
    assert entry.pid == FAKE_BROKER_PID + 1

    # The FIRST daemon binds; the second exits on the bind.
    broker.running = True
    broker.pid = FAKE_BROKER_PID

    stop = client.settle_spawned_broker(base)
    assert stop == ("stopped", FAKE_BROKER_PID)
    # Exactly the pid the socket's holder reported -- never the remembered latest.
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]
    assert client.spawned_brokers() == {}


def test_a_pid_this_process_never_spawned_is_still_a_replacement(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening the record to every spawn must not widen it to anyone's."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    client._record_spawn(base, FAKE_BROKER_PID + 1)
    broker.pid = FAKE_BROKER_PID + 2

    stop = client.settle_spawned_broker(base)
    # No pid is reported as "ours": the one answering is foreign.
    assert stop == ("replaced", None)
    assert broker.kills == []


@pytest.mark.parametrize("answer", ["missing", "not-a-list"])
def test_a_broker_that_cannot_say_who_registered_is_left_running(
    base: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """PR #4236 review m1. A version-skewed daemon answers ``status`` with a
    synthesised ``{"pid", "incompatible": True}`` and NO ``sessions``; reading
    that as an empty list SIGTERMed a broker without knowing who depended on
    it. Unknown is not nobody."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)
    status: dict[str, Any] = {"pid": FAKE_BROKER_PID, "incompatible": True}
    if answer == "not-a-list":
        status["sessions"] = None
    monkeypatch.setattr(client, "broker_status", lambda base=None: status)

    assert client.settle_spawned_broker(base) == ("in-use", FAKE_BROKER_PID)
    assert broker.kills == []
    assert broker.running is True
    # Still on record: a later, readable answer may let its owner stop it.
    assert len(client.spawned_brokers()) == 1


def test_a_narrowed_claim_leaves_an_outer_owners_spawn_alone(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scope that opened AFTER an earlier spawn for the same store owns only
    its own: the daemon holding the socket is this process's, not this
    caller's, so it is neither signalled nor dropped from the record."""

    broker = FakeBroker(monkeypatch)
    client.ensure_broker(base)  # the outer owner's, and the one listening
    client._record_spawn(base, FAKE_BROKER_PID + 1)  # the scope's own, lost the bind

    stop = client.settle_spawned_broker(base, owned_pids=(FAKE_BROKER_PID + 1,))
    assert stop == ("not-spawned-here", FAKE_BROKER_PID)
    assert broker.kills == []
    assert len(client.spawned_brokers()) == 1

    # The same daemon IS stopped for a caller that claims it.
    assert client.settle_spawned_broker(base, owned_pids=(FAKE_BROKER_PID,)) == (
        "stopped",
        FAKE_BROKER_PID,
    )


def test_the_episode_scope_stops_a_slow_first_spawn_through_the_real_client(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same Q1 race end to end: the episode host's scope over the REAL
    client (only the daemon is fake). Before the fix this read
    ``[(<last pid>, "replaced")]`` with no signal sent."""

    from local_operator.evaluation.runner import host_secrets

    broker = FakeBroker(monkeypatch)
    spawned = iter((FAKE_BROKER_PID, FAKE_BROKER_PID + 1))
    monkeypatch.setattr(client, "_spawn_broker", lambda base: next(spawned))

    with host_secrets.owned_broker_scope() as scope:
        assert client.ensure_broker(base, timeout=0.01) is False
        assert client.ensure_broker(base, timeout=0.01) is False
        broker.running = True
        broker.pid = FAKE_BROKER_PID  # the first spawn bound the socket

    assert scope.outcomes == [(FAKE_BROKER_PID, "stopped")]
    assert broker.kills == [(FAKE_BROKER_PID, signal.SIGTERM)]
