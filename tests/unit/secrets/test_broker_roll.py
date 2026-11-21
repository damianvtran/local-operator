"""Rolling the secret broker without a credential gap (``secrets.roll``).

**What these pin, and why with real processes.** The roll's whole contract is
about which PROCESS answers the socket at which moment and which process the
kernel says is asking, so the old broker, the successor and every descendant
here are real processes: the old broker is a real ``brokerd`` spawned on a tmp
config dir, the successor is spawned by ``roll_broker`` itself, the registered
session is THIS test process (a real ``SessionRegistration``), and each
"agent's ``$(lop secret get)``" is a real child of it. A mocked pid would prove
nothing about the boundary (the same reasoning ``test_broker.py`` gives).

Every child that retrieves replaces ``access.open_store`` with a tripwire, so
the one outcome these tests exist to forbid — the UNNOTIFIED local decrypt that
``access.retrieve_secret`` falls back to in the keyfile tier — prints
``FALLBACK`` instead of quietly serving the value.

Isolation: ``config_root`` redirects HOME and the config dir into ``tmp_path``;
every broker started here is reaped by exact pid in the ``reaper`` fixture. The
operator's live broker is never touched.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import local_operator
from local_operator.secrets import client
from local_operator.secrets import session as session_module
from local_operator.secrets.broker import SecretBroker
from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.keys import key_path, load_master_key, wrap_master_key, write_private_file
from local_operator.secrets.peer import ProcessIdentity, process_info
from local_operator.secrets.protocol import (
    PROTOCOL_VERSION,
    encode_bytes,
    lock_path,
    recv_frame,
    rolling_path,
    send_frame,
    socket_path,
    successor_socket_path,
)
from local_operator.secrets.store import SecretStore

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
pytestmark = [
    pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux"),
    pytest.mark.allow_secret_broker,
]

SECRET_NAME = "ROLL_CANARY"
SECRET_VALUE = "roll-canary-5e0d9a7c31"
PASSPHRASE = "correct horse battery staple"
TREE = str(Path(local_operator.__file__).resolve().parent.parent)


# --- fixtures -----------------------------------------------------------------


class Reaper:
    """Every broker pid a test started, stopped by EXACT pid at teardown."""

    def __init__(self) -> None:
        self.pids: set[int] = set()

    def add(self, pid: int | None) -> None:
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            self.pids.add(pid)

    def reap(self) -> None:
        for pid in self.pids:
            _stop_pid(pid)


def _stop_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
        except ChildProcessError:
            if process_info(pid) is None:
                return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (OSError, ChildProcessError):
        pass


@pytest.fixture
def reaper() -> Iterator[Reaper]:
    instance = Reaper()
    try:
        yield instance
    finally:
        instance.reap()


@pytest.fixture
def keyfile_store(config_root: Path) -> bytes:
    """A keyfile-tier store holding the canary; returns its master key."""
    key = generate_master_key()
    store = SecretStore(key, base=config_root)
    store.initialize()
    store.set(SECRET_NAME, SECRET_VALUE.encode(), description="roll canary")
    write_private_file(key_path(config_root), key)
    return key


def _start_old_broker(config_root: Path, reaper: Reaper) -> int:
    """A REAL detached broker on the tmp store — the one a roll replaces."""
    pid = client._spawn_broker(config_root)
    reaper.add(pid)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.broker_status(config_root)
        if status and status.get("pid") == pid:
            return pid
        time.sleep(0.05)
    raise AssertionError("the old broker never came up")


def _status(config_root: Path) -> dict:
    return client.broker_status(config_root) or {}


def _await(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _retrieve_script(config_root: Path) -> str:
    return textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from local_operator.secrets import access

        def _tripwire(*args, **kwargs):
            print("FALLBACK", flush=True)
            raise SystemExit(3)

        access.open_store = _tripwire
        started = time.monotonic()
        try:
            value = access.retrieve_secret({SECRET_NAME!r}, Path({str(config_root)!r}))
        except SystemExit:
            raise
        except Exception as error:
            print(f"ERROR {{type(error).__name__}}: {{error}}")
            print(f"ELAPSED {{time.monotonic() - started:.2f}}")
            raise SystemExit(2)
        print("VALUE " + value.decode())
        print(f"ELAPSED {{time.monotonic() - started:.2f}}")
        """)


def _child_retrieve(config_root: Path) -> subprocess.Popen[str]:
    """A real DESCENDANT of this (registered) process running ``$(lop secret get)``."""
    return subprocess.Popen(
        [sys.executable, "-c", _retrieve_script(config_root)],
        env={**os.environ, "PYTHONPATH": TREE},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(child: subprocess.Popen[str]) -> tuple[str, float]:
    out, err = child.communicate(timeout=60)
    elapsed = 0.0
    for line in out.splitlines():
        if line.startswith("ELAPSED "):
            elapsed = float(line.split()[1])
    return out + err, elapsed


def _roll_rows(key: bytes, config_root: Path) -> list:
    return [e for e in SecretStore(key, base=config_root).audit_entries() if e.event == "roll"]


def _register_without_reroll(config_root: Path) -> socket.socket:
    """A registration from a PRE-FIX client: no ``caps``, never re-registers."""
    from local_operator.secrets.keys import registration_ticket

    channel = client._connect(socket_path(config_root))
    send_frame(
        channel,
        {
            "version": PROTOCOL_VERSION,
            "op": "register",
            "session_id": None,
            "ticket": encode_bytes(registration_ticket(config_root)),
        },
    )
    reply = recv_frame(channel)
    assert reply.get("ok"), reply
    channel.settimeout(None)
    return channel


# --- the roll itself -----------------------------------------------------------


def test_roll_swaps_the_broker_and_the_session_is_notified_again(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """The headline: new pid, session re-registered, descendant NOTIFIED, one row.

    Also carries constraints (a), (b), (d): no ``key``/``retrieve`` reached the
    old broker during the roll; nothing but the temp socket was created on
    disk and no plaintext landed anywhere; the old broker answered the
    canonical path right up to the rename.
    """
    old_pid = _start_old_broker(config_root, reaper)
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    try:
        secrets_dir = key_path(config_root).parent
        runtime_dir = socket_path(config_root).parent
        before_files = {p.name for p in secrets_dir.iterdir()} | {p.name for p in runtime_dir.iterdir()}
        audit_before = len(SecretStore(keyfile_store, base=config_root).audit_entries())
        answered_before_switch: list[object] = []

        def before_switch() -> None:
            answered_before_switch.append(client.request("ping", config_root).get("pid"))
            assert successor_socket_path(config_root).exists()

        from local_operator.secrets.roll import roll_broker

        result = roll_broker(config_root, generation_stale=lambda pid: True, before_switch=before_switch)
        reaper.add(result.new_pid)

        assert result.status == "rolled", result
        assert result.old_pid == old_pid and result.new_pid not in (None, old_pid)
        assert answered_before_switch == [old_pid], "the old broker stopped serving before the rename"
        assert process_info(old_pid) is None, "the old broker was not told to exit"
        assert _status(config_root).get("pid") == result.new_pid
        assert not successor_socket_path(config_root).exists()
        assert not rolling_path(config_root).exists()
        assert _await(lambda: os.getpid() in (_status(config_root).get("sessions") or ())), (
            "the session did not re-register with the successor"
        )

        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, out
        assert f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen, "the descendant was served without the session being notified"

        assert _await(lambda: len(_roll_rows(keyfile_store, config_root)) == 1)
        rows = _roll_rows(keyfile_store, config_root)
        assert len(rows) == 1 and rows[0].outcome.startswith("allow:"), rows
        assert str(old_pid) in rows[0].outcome and str(result.new_pid) in rows[0].outcome
        assert SECRET_NAME not in rows[0].outcome and SECRET_VALUE not in rows[0].outcome

        # (a) the successor took nothing from the old broker: no key/retrieve
        # rows between the roll's start and the descendant's retrieval.
        during = SecretStore(keyfile_store, base=config_root).audit_entries()[audit_before:]
        assert not [e for e in during if e.event == "key"], during
        # (b) nothing new on disk but the database's own files; no plaintext.
        after_files = {p.name for p in secrets_dir.iterdir()} | {p.name for p in runtime_dir.iterdir()}
        # `broker.lock` is the roller's empty flock file (the old broker here was
        # spawned directly, not through the lazy start that would create it).
        allowed = {"broker.lock", "store.db-wal", "store.db-shm", "secrets.db-wal", "secrets.db-shm"}
        assert after_files - before_files <= allowed, (
            after_files - before_files
        )
        for directory in {secrets_dir, runtime_dir}:
            for path in directory.iterdir():
                if path.is_file() and path.name != key_path(config_root).name:
                    assert SECRET_VALUE.encode() not in path.read_bytes(), path
    finally:
        registration.close()


def test_a_descendant_during_reregistration_is_held_then_notified(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(i) part 1 / R1: between the rename and re-registration the child WAITS.

    The session is slowed to re-register ~2.5 s after the old broker exits; a
    descendant asking in that window must be held by the successor and then
    served WITH the notice — never denied into the local fallback.
    """
    _start_old_broker(config_root, reaper)
    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", (2.5, 0.6, 1.2))
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    try:
        from local_operator.secrets.roll import roll_broker

        result = roll_broker(config_root, force=True)
        reaper.add(result.new_pid)
        assert result.status == "rolled", result
        child = _child_retrieve(config_root)
        out, elapsed = _finish(child)
        assert "FALLBACK" not in out, out
        assert f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen
        assert elapsed >= 1.0, f"the child was not held (answered in {elapsed:.2f}s)"
    finally:
        registration.close()


def test_a_session_that_never_reregisters_gets_refusals_not_the_fallback(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """(i) part 2 / R1-R2: past the grace the answer is a refusal, never a decrypt.

    The registration declares ``reroll`` (so the roll proceeds) but is built
    without a known broker pid, so it never re-registers — the worst case.
    """
    _start_old_broker(config_root, reaper)
    registered = client.register_session_channel(config_root)
    assert registered is not None
    registration = session_module.SessionRegistration(registered[0], lambda *_: None)
    try:
        from local_operator.secrets.roll import roll_broker

        result = roll_broker(config_root, force=True)
        reaper.add(result.new_pid)
        assert result.status == "rolled", result
        out, elapsed = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, out
        assert SECRET_VALUE not in out, out
        assert "ERROR SecretStoreError" in out and "re-registered" in out, out
        assert elapsed <= 10.0
    finally:
        registration.close()


def test_readiness_mismatch_aborts_and_leaves_the_old_broker(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, tmp_path: Path
) -> None:
    """(c) / R5: a successor answering with another build is killed before the rename."""
    old_pid = _start_old_broker(config_root, reaper)
    fake = tmp_path / "fake_successor.py"
    fake.write_text(
        textwrap.dedent("""
            import json, os, socket, struct, sys
            path = sys.argv[sys.argv.index("--bind") + 1]
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path); os.chmod(path, 0o600); server.listen(4)
            while True:
                conn, _ = server.accept()
                size = struct.unpack(">I", conn.recv(4))[0]
                request = json.loads(conn.recv(size))
                reply = {"ok": True, "pid": os.getpid(), "protocol": 1, "locked": False,
                         "sessions": [], "version": "0.0.0-not-this-build", "prefix": sys.prefix}
                body = json.dumps(reply).encode()
                conn.sendall(struct.pack(">I", len(body)) + body)
                conn.close()
            """)
    )

    def spawn(_base: Path | None, args: tuple[str, ...]) -> int:
        process = subprocess.Popen([sys.executable, str(fake), *args])
        reaper.add(process.pid)
        return process.pid

    from local_operator.secrets.roll import roll_broker

    result = roll_broker(config_root, force=True, spawn=spawn)
    assert result.status == "failed" and result.code == "readiness", result
    assert "version" in result.reason
    assert _status(config_root).get("pid") == old_pid, "the old broker did not survive"
    assert not successor_socket_path(config_root).exists()
    assert not rolling_path(config_root).exists()
    assert process_info(result.new_pid) is None, "the failed successor was left running"
    rows = _roll_rows(keyfile_store, config_root)
    assert len(rows) == 1 and rows[0].outcome.startswith("fail:"), rows


def test_a_successor_that_never_answers_is_reaped(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """(c): no answer within the readiness bound ⇒ failed, old broker untouched."""
    old_pid = _start_old_broker(config_root, reaper)

    def spawn(_base: Path | None, _args: tuple[str, ...]) -> int:
        process = subprocess.Popen(["sleep", "30"])
        reaper.add(process.pid)
        return process.pid

    from local_operator.secrets.roll import roll_broker

    result = roll_broker(config_root, force=True, spawn=spawn, ready_timeout=1.0)
    assert result.status == "failed" and result.code == "readiness", result
    assert _status(config_root).get("pid") == old_pid
    assert process_info(result.new_pid) is None


def test_stranded_sessions_defer_and_name_the_pids(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """(f) / 79(a) / R7: a pre-fix registrant blocks the roll, visibly."""
    old_pid = _start_old_broker(config_root, reaper)
    channel = _register_without_reroll(config_root)
    try:
        from local_operator.secrets.roll import roll_broker

        def no_spawn(*_args: object) -> int:
            raise AssertionError("a deferred roll must not spawn")

        result = roll_broker(config_root, generation_stale=lambda pid: pid == old_pid or None, spawn=no_spawn)
        assert result.status == "deferred" and result.code == "sessions", result
        assert result.stranded == (os.getpid(),)
        assert str(os.getpid()) in result.reason
        assert "will roll once these sessions restart" in result.reason
        assert "restart`" not in result.reason, "R7: the deferral must not offer a restart route"
        assert _status(config_root).get("pid") == old_pid
    finally:
        channel.close()


def test_a_held_lock_defers(config_root: Path, keyfile_store: bytes, reaper: Reaper) -> None:
    """(h) / R4: ``broker.lock`` serialises rolls; a second one defers."""
    old_pid = _start_old_broker(config_root, reaper)
    handle = os.open(lock_path(config_root), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        from local_operator.secrets.roll import roll_broker

        result = roll_broker(config_root, force=True)
        assert result.status == "deferred" and result.code == "busy", result
        assert _status(config_root).get("pid") == old_pid
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def test_tier_rule(config_root: Path, reaper: Reaper) -> None:
    """(e): unlocked passphrase ⇒ deferred; locked ⇒ rolled; planted key ⇒ deferred."""
    from local_operator.secrets.roll import roll_broker

    key = load_master_key(config_root, create=True)
    SecretStore(key, base=config_root).initialize()
    wrap_master_key(config_root, key, PASSPHRASE)

    def no_spawn(*_args: object) -> int:
        raise AssertionError("a deferred roll must not spawn")

    # Locked passphrase broker: holds no key, nothing to lose ⇒ rolls.
    _start_old_broker(config_root, reaper)
    result = roll_broker(config_root, force=True)
    reaper.add(result.new_pid)
    assert result.status == "rolled", result
    assert _status(config_root).get("locked") is True

    # Unlocked: holds the only unwrapped key ⇒ deferred, nothing spawned.
    client.unlock(PASSPHRASE, config_root)
    assert _status(config_root).get("locked") is False
    unlocked = roll_broker(config_root, force=True, spawn=no_spawn)
    assert unlocked.status == "deferred" and unlocked.code == "passphrase", unlocked
    _stop_pid(result.new_pid)

    # A planted master.key beside a hardened store: the VALIDATING predicate
    # says hardened, so the roll fails closed.
    key_path(config_root).write_bytes(generate_master_key())
    _start_old_broker(config_root, reaper)
    planted = roll_broker(config_root, force=True, spawn=no_spawn)
    assert planted.status == "deferred" and planted.code == "tier-unknown", planted


def test_stop_still_means_stop(config_root: Path, keyfile_store: bytes, reaper: Reaper) -> None:
    """(g): a session whose broker was STOPPED does not resurrect one."""
    from local_operator.secrets.handlers import _stop_broker

    _start_old_broker(config_root, reaper)
    registration = session_module.register_session(lambda *_: None, base=config_root)
    assert registration is not None
    try:
        assert _stop_broker(client) is True
        time.sleep(3.0)  # past every re-registration attempt
        assert client.broker_status(config_root) is None, "a stopped broker was respawned"
    finally:
        registration.close()


# --- `lop secret broker restart` (79's addition (b)) ---------------------------


def _restart(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    from local_operator.secrets import handlers

    code = handlers._broker(argparse.Namespace(broker_command="restart", json=False))
    return code, capsys.readouterr().err


def test_restart_never_causes_the_unnotified_fallback(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE REGRESSION: after a bare ``restart``, a descendant is notified or refused.

    Pre-fix ``restart`` was stop-then-start: the session's channel closed, it
    never re-registered, the new broker denied its descendant, and the keyfile
    tier served that descendant through the unnotified local decrypt.
    """
    _start_old_broker(config_root, reaper)
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    try:
        code, err = _restart(capsys)
        reaper.add(_status(config_root).get("pid"))
        assert code == 0, err
        _await(lambda: os.getpid() in (_status(config_root).get("sessions") or ()))
        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, f"restart left the descendant on the unnotified fallback: {out}"
        if f"VALUE {SECRET_VALUE}" in out:
            assert SECRET_VALUE in seen, "served without the session's redaction notice"
        else:
            assert "ERROR" in out, out
    finally:
        registration.close()


def test_restart_refuses_while_sessions_cannot_reregister(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, capsys: pytest.CaptureFixture[str]
) -> None:
    """79(b) / R7: the keyfile tier with stranded sessions REFUSES, old broker kept."""
    old_pid = _start_old_broker(config_root, reaper)
    channel = _register_without_reroll(config_root)
    try:
        code, err = _restart(capsys)
        reaper.add(_status(config_root).get("pid"))
        assert code == 2, err
        assert str(os.getpid()) in err and "will roll once these sessions restart" in err, err
        assert _status(config_root).get("pid") == old_pid
    finally:
        channel.close()


def test_passphrase_restart_never_serves_unnotified(
    config_root: Path, reaper: Reaper, capsys: pytest.CaptureFixture[str]
) -> None:
    """79(b): the kept stop+start for an UNLOCKED passphrase broker cannot leak.

    With no plaintext key on disk the local fallback has nothing to decrypt
    with, and ``retrieve_secret`` re-raises broker refusals in that tier.
    """
    key = load_master_key(config_root, create=True)
    store = SecretStore(key, base=config_root)
    store.initialize()
    store.set(SECRET_NAME, SECRET_VALUE.encode(), description="roll canary")
    wrap_master_key(config_root, key, PASSPHRASE)
    _start_old_broker(config_root, reaper)
    client.unlock(PASSPHRASE, config_root)
    registration = session_module.register_session(lambda *_: None, base=config_root)
    assert registration is not None
    try:
        code, err = _restart(capsys)
        reaper.add(_status(config_root).get("pid"))
        assert code == 0, err
        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out and SECRET_VALUE not in out, out
        assert "ERROR" in out, out
    finally:
        registration.close()


# --- protocol-level pins (no daemons) -------------------------------------------


def test_no_wire_operation_starts_a_roll(config_root: Path, keyfile_store: bytes) -> None:
    """R6: a roll is local-only; the broker has no verb that triggers one."""
    broker = SecretBroker(config_root, key_provider=lambda: keyfile_store, idle_shutdown_s=0)
    broker.start()
    try:
        for operation in ("roll", "handover", "successor"):
            with pytest.raises(Exception, match="unauthorized|unknown operation|descend|no lop session is registered"):
                client.request(operation, config_root)
    finally:
        broker.stop(timeout=5)


def test_expected_lineage_is_keyed_by_start_time_not_pid(config_root: Path) -> None:
    """R2: a recycled pid (same number, other start time) is never held for."""
    me = process_info(os.getpid())
    assert me is not None
    impostor = ProcessIdentity(pid=me.pid, start_time=me.start_time + 1)
    broker = SecretBroker(config_root, expected_sessions={me.pid: impostor})
    assert broker._expected_unregistered_ancestor(me) is None
    genuine = SecretBroker(config_root, expected_sessions={me.pid: me})
    assert genuine._expected_unregistered_ancestor(me) == me.pid


def test_temp_socket_must_be_private(tmp_path: Path) -> None:
    """R4: a non-0600 temp socket is refused before any rename."""
    from local_operator.secrets.roll import _check_private_socket

    directory = tmp_path / "d"
    directory.mkdir(mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / "s"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        os.chmod(path, 0o644)
        assert "0600" in (_check_private_socket(path) or "")
        os.chmod(path, 0o600)
        assert _check_private_socket(path) is None
        os.chmod(directory, 0o755)
        assert "0700" in (_check_private_socket(path) or "")
    finally:
        server.close()


def test_a_symlinked_canonical_socket_is_not_rolled(
    config_root: Path, keyfile_store: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: the rename target must still be the expected SOCKET, never a link."""
    from local_operator.secrets import roll

    canonical = socket_path(config_root)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(tmp_path / "elsewhere")
    monkeypatch.setattr(client, "request", lambda op, base=None, **_: {"ok": True, "pid": os.getpid()})
    result = roll.roll_broker(config_root, force=True)
    assert result.status == "deferred" and "not a socket" in result.reason, result
    assert canonical.is_symlink()


def test_the_client_waits_out_a_roll_instead_of_falling_back(
    config_root: Path, keyfile_store: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: with the sentinel present a denial is retried, then fails CLOSED.

    ``BrokerRolling`` is not a ``BrokerDenied``/``BrokerUnavailable``, so
    ``access.retrieve_secret`` propagates it instead of decrypting locally.
    """
    from local_operator.secrets import access
    from local_operator.secrets.roll import _write_sentinel

    def deny(*_args: object, **_kwargs: object) -> dict:
        raise client.BrokerDenied("no registered session ancestor")

    monkeypatch.setattr(client, "request", deny)
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "ROLL_WAIT_S", 0.3)

    def tripwire(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the unnotified local decrypt was reached during a roll")

    monkeypatch.setattr(access, "open_store", tripwire)
    socket_path(config_root).parent.mkdir(parents=True, exist_ok=True)
    _write_sentinel(rolling_path(config_root), 1)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret(SECRET_NAME, config_root)
    rolling_path(config_root).unlink()
    # Outside a roll the documented keyfile fallback is unchanged.
    with pytest.raises(AssertionError, match="unnotified local decrypt"):
        access.retrieve_secret(SECRET_NAME, config_root)


def test_an_expired_sentinel_is_ignored(config_root: Path) -> None:
    """A roller that died must not hold retrievals hostage."""
    path = rolling_path(config_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"old_pid": 1, "deadline": 1}')
    assert not client.rolling_in_progress(config_root)


# --- the update wiring -----------------------------------------------------------


def test_both_platform_tables_carry_the_broker_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator import update

    assert "secret broker" in [name for name, _, _ in update._refresh_steps()]
    assert "secret broker" in [name for name, _, _ in update._linux_refresh_steps()]


def test_roll_results_map_to_report_statuses() -> None:
    from local_operator import update
    from local_operator.secrets.roll import RollResult, stranded_sessions_reason

    reason = stranded_sessions_reason((4242,))
    deferred = update._broker_roll_refresh(RollResult("deferred", reason, code="sessions"))
    assert update._plist_status(deferred.kind) == "pending" and deferred.detail == reason
    rolled = update._broker_roll_refresh(RollResult("rolled", old_pid=1, new_pid=2))
    assert update._plist_status(rolled.kind) == "refreshed"
    assert update._plist_status(update._broker_roll_refresh(RollResult("already")).kind) == "already"
    failed = update._broker_roll_refresh(RollResult("failed", "boom"))
    assert update._plist_status(failed.kind) == "failed"


def test_reconcile_keeps_the_deferral_reason() -> None:
    from local_operator import update
    from local_operator.secrets.roll import stranded_sessions_reason
    from local_operator.services import StaleDaemon

    reason = stranded_sessions_reason((4242, 4343))
    statuses = (update.DaemonStatus("secret broker", "pending", reason),)
    inventory = lambda: [StaleDaemon("secret broker", 99, "gen-old", "generic")]  # noqa: E731
    out = update._reconcile_with_processes(statuses, inventory=inventory)
    assert out == (update.DaemonStatus("secret broker", "pending", reason),)
    assert "4242, 4343" in out[0].reason and "will roll once these sessions restart" in out[0].reason


def test_the_successor_socket_fits_wherever_the_canonical_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``broker.sock.next`` overflowed sun_path beside a 103-byte socket.

    Measured in ``test_cli.py``'s tmp dir: the canonical path fit at 103 bytes,
    the temp one was 108, so the successor could never bind and every restart
    was refused. Pinned at the exact boundary.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.secrets.protocol import MAX_SOCKET_PATH

    # config/secrets/broker.sock: pad the config dir so the canonical is 103 bytes.
    suffix = len("/secrets/broker.sock")
    stem = str(tmp_path / "c")
    root = Path(stem + "x" * (MAX_SOCKET_PATH - suffix - len(stem)))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    canonical = socket_path(root)
    assert canonical.parent == root / "secrets" and len(str(canonical)) == MAX_SOCKET_PATH
    assert len(str(successor_socket_path(root))) <= MAX_SOCKET_PATH
    assert successor_socket_path(root).parent == canonical.parent
