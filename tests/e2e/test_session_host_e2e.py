"""A runtime forked by the session host: whose child is it, and does a host's death touch it?

Design-surface-independence rules R1 (parent identity) and R3 (a host restart does
not kill runtimes), exercised with every actor real: a real ``host_service``
process on its real socket, a real ``engage_runtime`` through the real
``_spawn_runtime`` seam, a real session runtime process (the mock provider), the
real discovery record. Nothing about the spawn is stubbed.

WHAT IT PINS
* **Parentage, not only survival.** The runtime's ``ppid`` is the HOST, and its
  session and group are its own. A runtime the surface forked would have the
  surface's pid there; ``test_the_parent_assertion_discriminates`` runs the local
  route and requires exactly that, so a spawn that silently reverted to
  surface-parented fails the first test rather than passing it (design §8: "assert
  parent identity, not only survival").
* **R3, scripted.** SIGTERM the host: it exits 0 and the runtime is alive and
  still ``live`` in the store. A SUCCESSOR host is started; a fresh engage of the
  same session finds the runtime by its record (``spawned=False``) and the
  successor forks nothing for it — adoption by record, not ownership by birth —
  while a NEW session engaged through the successor does get its own fork.

WHAT IT DOES NOT PIN. The launchd wiring (it addresses the operator's real
``gui/<uid>`` domain, which a test must never do) and Gate B (the real app
force-quit, interactive and recorded; design §8). Here the host is an ordinary
subprocess and its parent is the pytest process: the property asserted is "the
runtime's parent is the HOST", which is what makes it launchd's child in
production, and that the host's death leaves it alive.

Teardown reaps by EXACT pid with SIGKILL (the runtime's own stop ladder is not
under test, and SIGKILL is the one signal the victim-side gate never refuses).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator import procstate
from local_operator.session.runtime import launch, registry, session_host
from local_operator.session.runtime.types import SessionRecord
from tests.e2e.harness import NO_NOTIFY_ENV


def _ppid_of(pid: int) -> int:
    out = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return int(out) if out else -1


def _pgid_sid(pid: int) -> tuple[int, int]:
    return os.getpgid(pid), os.getsid(pid)


def _record(root: Path, session_id: str) -> tuple[SessionRecord, str] | None:
    for record, state in registry.scan(root):
        if record.session_id == session_id:
            return record, state
    return None


def _published(root: Path, session_id: str) -> tuple[SessionRecord, str]:
    """The session's record, failing loudly when it was never published."""
    found = _record(root, session_id)
    assert found is not None, f"no record published for {session_id}"
    return found


@pytest.fixture
def rig(headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator["_Rig"]:
    root = headless_tui_env
    (root / "config.yml").write_text("values:\n  hosting: test\n  model_name: mock\n")
    for key in [k for k in os.environ if k.startswith(("LOP_", "CMUX_"))]:
        monkeypatch.delenv(key, raising=False)
    for key, value in NO_NOTIFY_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    state = _Rig(root)
    try:
        yield state
    finally:
        state.teardown()


class _Rig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.hosts: list[subprocess.Popen[bytes]] = []
        self.runtime_pids: list[int] = []

    def start_host(self) -> subprocess.Popen[bytes]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LOP_", "CMUX_"))}
        env["LOCAL_OPERATOR_CONFIG_DIR"] = str(self.root)
        log = (self.root / f"host{len(self.hosts)}.log").open("wb")
        host = subprocess.Popen(
            [sys.executable, "-m", "local_operator.session.runtime.host_service"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.hosts.append(host)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if session_host.ping(self.root):
                return host
            if host.poll() is not None:
                raise AssertionError(f"the host exited {host.returncode} before serving")
            time.sleep(0.1)
        raise AssertionError("the host never answered")

    def engage(
        self, session_id: str, *, hosted: bool, monkeypatch: pytest.MonkeyPatch
    ) -> launch.EngageOutcome:
        if hosted:
            monkeypatch.setenv(session_host.SPAWN_AUTHORITY_ENV, "1")
        else:
            monkeypatch.delenv(session_host.SPAWN_AUTHORITY_ENV, raising=False)
        outcome = asyncio.run(
            launch.engage_runtime(
                session_id, str(self.root), launch.WarmErrand(), config_dir=self.root, deadline_s=90
            )
        )
        found = _record(self.root, session_id)
        assert found is not None, "the runtime never published a record"
        self.runtime_pids.append(found[0].pid)
        return outcome

    def teardown(self) -> None:
        for pid in self.runtime_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for host in self.hosts:
            if host.poll() is None:
                host.send_signal(signal.SIGTERM)
                try:
                    host.wait(10)
                except subprocess.TimeoutExpired:
                    host.kill()
                    host.wait(10)
        # A runtime whose parent was the (now dead) host is reparented to init and
        # reaped there; one that was OUR child (the local-route control) is ours.
        for pid in self.runtime_pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def test_a_host_forked_runtime_is_the_hosts_child_in_its_own_session(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = rig.start_host()
    outcome = rig.engage("hostedsess001", hosted=True, monkeypatch=monkeypatch)
    assert outcome.spawned is True
    record, state = _published(rig.root, "hostedsess001")
    assert state == "live"
    pid = record.pid
    assert _ppid_of(pid) == host.pid, "the runtime must descend from the host, not from the surface"
    assert _ppid_of(pid) != os.getpid()
    pgid, sid = _pgid_sid(pid)
    assert (
        pgid == pid and sid == pid
    ), "own session and group: no signal to the host's group reaches it"
    assert (pgid, sid) != _pgid_sid(host.pid)


def test_the_parent_assertion_discriminates(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The NEGATIVE CONTROL: the local route yields a surface-parented runtime, and the
    parent check above would have failed on it."""
    outcome = rig.engage("localsess0001", hosted=False, monkeypatch=monkeypatch)
    assert outcome.spawned is True
    record, _state = _published(rig.root, "localsess0001")
    assert _ppid_of(record.pid) == os.getpid()


def test_sigterm_to_the_host_leaves_its_runtime_alive_and_a_successor_adopts_it_by_record(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = rig.start_host()
    rig.engage("survivor00001", hosted=True, monkeypatch=monkeypatch)
    record, _ = _published(rig.root, "survivor00001")
    pid = record.pid
    assert _ppid_of(pid) == first.pid

    first.send_signal(signal.SIGTERM)
    assert (
        first.wait(15) == 0
    ), "a deliberate stop is a clean exit (KeepAlive{SuccessfulExit:false})"
    assert not session_host.ping(rig.root), "the first host is really gone"

    # THE ASSERTION R3 EXISTS FOR: the host's death did not take the runtime.
    time.sleep(1.0)
    assert procstate.pid_alive(pid) and not procstate.is_zombie(pid)
    found = _published(rig.root, "survivor00001")
    assert found[1] == "live"
    assert found[0].pid == pid
    assert (
        _ppid_of(pid) in (1, -1) or _ppid_of(pid) != first.pid
    ), "reparented away from the dead host"

    # The SUCCESSOR adopts by record: engaging the same session dials the existing
    # runtime and forks nothing; a new session gets its own fork from the successor.
    second = rig.start_host()
    again = rig.engage("survivor00001", hosted=True, monkeypatch=monkeypatch)
    assert again.spawned is False, "adoption by record: the runtime already exists"
    assert _published(rig.root, "survivor00001")[0].pid == pid
    stats = _stats(rig.root)
    assert stats["forks"] == 0 and stats["pid"] == second.pid

    rig.engage("successor0001", hosted=True, monkeypatch=monkeypatch)
    new_record, _ = _published(rig.root, "successor0001")
    assert _ppid_of(new_record.pid) == second.pid
    assert _stats(rig.root)["forks"] == 1


def _stats(root: Path) -> dict[str, object]:
    import socket

    from local_operator.session.runtime.standby import _recv, _send

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect(str(session_host.socket_path(root)))
    try:
        _send(sock, {"op": "stats", "v": session_host.PROTOCOL_VERSION})
        reply = _recv(sock)
    finally:
        sock.close()
    assert reply["ok"], reply
    return reply
