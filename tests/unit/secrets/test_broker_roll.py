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
from typing import Any

import pytest

import local_operator
from local_operator.secrets import client
from local_operator.secrets.audit import AuditEntry
from local_operator.secrets import session as session_module
from local_operator.secrets.broker import SecretBroker
from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.keys import (
    key_path,
    load_master_key,
    wrap_master_key,
    write_private_file,
)
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


def _status(config_root: Path) -> dict[str, Any]:
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


def _roll_rows(key: bytes, config_root: Path) -> list[AuditEntry]:
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
        before_files = {p.name for p in secrets_dir.iterdir()} | {
            p.name for p in runtime_dir.iterdir()
        }
        audit_before = len(SecretStore(keyfile_store, base=config_root).audit_entries())
        answered_before_switch: list[object] = []

        def before_switch() -> None:
            answered_before_switch.append(client.request("ping", config_root).get("pid"))
            assert successor_socket_path(config_root).exists()

        from local_operator.secrets.roll import roll_broker

        result = roll_broker(
            config_root, generation_stale=lambda pid: True, before_switch=before_switch
        )
        reaper.add(result.new_pid)

        assert result.status == "rolled", result
        assert result.old_pid == old_pid and result.new_pid not in (None, old_pid)
        assert answered_before_switch == [
            old_pid
        ], "the old broker stopped serving before the rename"
        assert process_info(old_pid) is None, "the old broker was not told to exit"
        assert _status(config_root).get("pid") == result.new_pid
        assert not successor_socket_path(config_root).exists()
        # S1: the sentinel outlives the old broker by the successor's hold.
        assert client.rolling_in_progress(config_root)
        assert _await(
            lambda: os.getpid() in (_status(config_root).get("sessions") or ())
        ), "the session did not re-register with the successor"

        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, out
        assert f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen, "the descendant was served without the session being notified"

        assert _await(lambda: len(_roll_rows(keyfile_store, config_root)) == 1)
        rows = _roll_rows(keyfile_store, config_root)
        assert len(rows) == 1 and rows[0].outcome.startswith("allow:"), rows
        # S3: both builds are named, the old one from the predecessor's status.
        import re

        assert re.match(r"allow:old pid \d+ \(.+\) -> new pid \d+ \(.+\)$", rows[0].outcome), rows
        assert str(old_pid) in rows[0].outcome and str(result.new_pid) in rows[0].outcome
        assert SECRET_NAME not in rows[0].outcome and SECRET_VALUE not in rows[0].outcome

        # (a) the successor took nothing from the old broker: no key/retrieve
        # rows between the roll's start and the descendant's retrieval.
        during = SecretStore(keyfile_store, base=config_root).audit_entries()[audit_before:]
        assert not [e for e in during if e.event == "key"], during
        # (b) nothing new on disk but the database's own files; no plaintext.
        after_files = {p.name for p in secrets_dir.iterdir()} | {
            p.name for p in runtime_dir.iterdir()
        }
        # `broker.lock` is the roller's empty flock file (the old broker here was
        # spawned directly, not through the lazy start that would create it).
        allowed = {
            "broker.lock",
            "broker.rolling",
            "store.db-wal",
            "store.db-shm",
            "secrets.db-wal",
            "secrets.db-shm",
        }
        assert after_files - before_files <= allowed, after_files - before_files
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
    fake.write_text(textwrap.dedent("""
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
            """))

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
    assert result.new_pid is not None, "the roll never spawned a successor"
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
    assert result.new_pid is not None, "the roll never spawned a successor"
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

        result = roll_broker(
            config_root, generation_stale=lambda pid: pid == old_pid or None, spawn=no_spawn
        )
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
    assert result.new_pid is not None, "the locked roll reported no successor pid"
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
        assert "FALLBACK" not in out, f"restart left the descendant on the fallback: {out}"
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


def _dispatched_operations() -> set[str]:
    """Every op string the REAL dispatcher (``_handle_requests``) compares against.

    Read from its AST rather than guessed (review CR7): ``operation == "x"``
    and ``operation in ("x", ...)`` are the only two shapes it uses, so a new
    verb cannot be added without appearing here.
    """
    import ast
    import inspect

    from local_operator.secrets import broker as broker_module

    source = inspect.getsource(broker_module.SecretBroker._handle_requests)
    tree = ast.parse(textwrap.dedent(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "operation"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                found.add(comparator.value)
            elif isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                found.update(
                    element.value
                    for element in comparator.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
    return found


def test_no_wire_operation_starts_a_roll(config_root: Path, keyfile_store: bytes) -> None:
    """R6 / CR7: a roll is local-only; the dispatcher's op set has no roll verb at all."""
    operations = _dispatched_operations()
    assert operations == {"register", "ping", "status", "unlock", "key", "retrieve"}, operations
    # And anything outside that set is refused as unknown (after the gate).
    broker = SecretBroker(config_root, key_provider=lambda: keyfile_store, idle_shutdown_s=0)
    broker.start()
    try:
        refused = "unauthorized|unknown operation|descend|no lop session is registered"
        with pytest.raises(Exception, match=refused):
            client.request("roll", config_root)
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
    monkeypatch.setattr(
        client, "request", lambda op, base=None, **_: {"ok": True, "pid": os.getpid()}
    )
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

    def deny(*_args: object, **_kwargs: object) -> dict[str, Any]:
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
    already = update._broker_roll_refresh(RollResult("already"))
    assert update._plist_status(already.kind) == "already"
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
    assert "4242, 4343" in out[0].reason
    assert "will roll once these sessions restart" in out[0].reason


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


# --- round-1 remediation (ledger 4389-r1) ---------------------------------------


def test_a_session_registered_mid_roll_aborts_the_switch(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """S1 (re-read half): a session that registers with the OLD broker after the
    successor's one-time snapshot is unknown to the successor. The roller
    re-reads the old broker right before the rename and aborts; the old broker
    keeps serving and the session's descendant is notified, never the fallback.
    """
    old_pid = _start_old_broker(config_root, reaper)
    seen: list[str] = []
    late: list[object] = []

    def before_switch() -> None:
        late.append(
            session_module.register_session(
                lambda _name, value: seen.append(value.decode()), base=config_root
            )
        )

    from local_operator.secrets.roll import roll_broker

    try:
        result = roll_broker(config_root, force=True, before_switch=before_switch)
        reaper.add(result.new_pid)
        assert late and late[0] is not None
        assert result.status == "failed" and result.code == "switch", result
        assert str(os.getpid()) in result.reason
        assert _status(config_root).get("pid") == old_pid
        assert not client.rolling_in_progress(config_root), "a failed roll must clear its sentinel"
        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out and f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen
    finally:
        for registration in late:
            if registration is not None:
                registration.close()  # type: ignore[attr-defined]


def test_a_late_session_is_covered_by_the_sentinel_after_the_rename(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1 (sentinel half), R1 WITHOUT #4388: even if a late registration slips
    past the re-read (simulated by disabling it), the sentinel lives on for the
    successor's hold after the rename, so the client retries the successor's
    denial until the session has re-registered, and is then NOTIFIED. Deleting
    the sentinel as soon as the old broker exited ended that retry early and
    the denial reached the unnotified local decrypt.
    """
    from local_operator.secrets import roll as roll_module

    _start_old_broker(config_root, reaper)
    monkeypatch.setattr(roll_module, "_unknown_to_successor", lambda *_args: [])
    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", (1.5, 0.6, 1.2))
    seen: list[str] = []
    late: list[object] = []

    def before_switch() -> None:
        late.append(
            session_module.register_session(
                lambda _name, value: seen.append(value.decode()), base=config_root
            )
        )

    try:
        result = roll_module.roll_broker(config_root, force=True, before_switch=before_switch)
        reaper.add(result.new_pid)
        assert result.status == "rolled", result
        out, elapsed = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, out
        assert f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen, "served without the late session being notified"
    finally:
        for registration in late:
            if registration is not None:
                registration.close()  # type: ignore[attr-defined]


def test_hold_plus_notice_fits_the_client_timeout() -> None:
    """CR1 / S5: the successor's hold plus the notice (plus a margin for
    write-lock queueing) must finish inside the client's request timeout, the
    client's single roll budget must cover hold + notice, and the hold must
    still cover the session's own re-registration schedule (+1 s watcher tick).
    """
    from local_operator.secrets import broker as broker_module

    hold = broker_module.ROLL_GRACE_S
    notice = broker_module.NOTIFY_ACK_TIMEOUT_S
    margin = broker_module.ROLL_HOLD_MARGIN_S
    # #4389 R3-2: the client clips a roll read to ROLL_WAIT_S, not the request
    # timeout, so the hold + notice must fit inside THAT with a real margin
    # (6 + 2 == 8 used to leave none).
    assert margin >= 1.0
    assert hold + notice + margin <= client.ROLL_WAIT_S <= client.REQUEST_TIMEOUT_S
    assert hold >= sum(session_module._REREGISTER_DELAYS_S) + 1.0


def test_stop_drains_an_in_flight_retrieve_before_dropping_sessions(
    config_root: Path, keyfile_store: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR2 / S4: a retrieve already authorized when ``stop()`` begins must still
    find its session and notify it. Clearing the registry first made it find
    none, skip the notice and send the value unscrubbed.
    """
    import threading

    broker = SecretBroker(config_root, key_provider=lambda: keyfile_store, idle_shutdown_s=0)
    broker.start()
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    original = broker._master_key
    entered = threading.Event()

    def slow_key() -> bytes:
        entered.set()
        time.sleep(1.0)
        return original()

    monkeypatch.setattr(broker, "_master_key", slow_key)
    stopper: threading.Thread | None = None
    try:
        child = _child_retrieve(config_root)
        assert entered.wait(15), "the child's retrieve never reached the broker"
        stopper = threading.Thread(target=broker.stop, kwargs={"timeout": 5})
        stopper.start()
        out, _ = _finish(child)
        assert "FALLBACK" not in out, out
        if f"VALUE {SECRET_VALUE}" in out:
            assert SECRET_VALUE in seen, "a draining broker served the value with no notice"
        assert f"VALUE {SECRET_VALUE}" in out, out
    finally:
        if stopper is not None:
            stopper.join(10)
        registration.close()
        broker.stop(timeout=5)


def test_a_stopping_broker_refuses_a_sessionless_read(
    config_root: Path, keyfile_store: bytes
) -> None:
    """CR2 belt: past the drain, a stopping broker cannot tell \"no session\"
    from \"session just dropped\", so it refuses as ``unredactable``."""
    broker = SecretBroker(config_root, key_provider=lambda: keyfile_store, idle_shutdown_s=0)
    sent: list[tuple[str, str]] = []

    def capture(_conn: object, code: str, message: str, **_: object) -> None:
        sent.append((code, message))

    broker._reply_error = capture  # type: ignore[method-assign]
    broker._stopping.set()
    me = process_info(os.getpid())
    assert me is not None
    broker._handle_retrieve(None, me, {"name": SECRET_NAME})  # type: ignore[arg-type]
    assert sent and sent[0][0] == "unredactable", sent


def test_a_second_roll_while_sessions_reregister_defers(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1 (QA repro): slow re-registration, restart twice, one child read. The
    second roll used to see no sessions (status listed only registered ones),
    so its successor held nothing for the session and the child's read met the
    unnotified fallback. Now the second roll defers while the first successor
    is still holding, the child is notified, and a roll after settling works.
    """
    from local_operator.secrets.roll import roll_broker

    _start_old_broker(config_root, reaper)
    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", (2.5, 0.6, 1.2))
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    try:
        first = roll_broker(config_root, force=True)
        reaper.add(first.new_pid)
        assert first.status == "rolled", first
        second = roll_broker(config_root, force=True)
        reaper.add(second.new_pid)
        assert second.status == "deferred" and second.code == "settling", second
        assert _status(config_root).get("pid") == first.new_pid
        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out and f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen
        assert _await(lambda: os.getpid() in (_status(config_root).get("sessions") or ()))
        third = roll_broker(config_root, force=True)
        reaper.add(third.new_pid)
        assert third.status == "rolled", third
    finally:
        registration.close()


def test_status_reports_held_sessions_and_a_stacked_successor_keeps_them(
    config_root: Path, keyfile_store: bytes
) -> None:
    """Q1, the merge half: a broker still holding a predecessor's session lists
    it in ``expected_lineage`` (with its caps) and ``settling``; a successor
    snapshotting that broker carries the session into its own expected set."""
    from local_operator.secrets.broker import _successor_inputs

    me = process_info(os.getpid())
    assert me is not None
    broker = SecretBroker(
        config_root,
        key_provider=lambda: keyfile_store,
        idle_shutdown_s=0,
        expected_sessions={me.pid: me},
        expected_caps={me.pid: ("reroll",)},
    )
    broker.start()
    try:
        status = _status(config_root)
        assert status.get("settling") is True
        assert status.get("expected_lineage") == [
            {"pid": me.pid, "start_time": me.start_time, "caps": ["reroll"]}
        ]
        inputs = _successor_inputs(config_root, os.getpid())
        assert inputs is not None and me.pid in inputs.expected
        assert inputs.caps[me.pid] == ("reroll",)
    finally:
        broker.stop(timeout=5)


_DYING_ROLLER = """
import os, sys
from pathlib import Path
from local_operator.secrets import roll

real = roll._write_sentinel

def dying(path, old_pid, *, ttl=roll.SENTINEL_TTL_S, signalled=False):
    if signalled:
        os._exit(9)  # the roller dies between the rename and its SIGTERM
    real(path, old_pid, ttl=ttl, signalled=signalled)

roll._write_sentinel = dying
roll.roll_broker(Path(sys.argv[1]), force=True)
os._exit(1)
"""


def test_the_successor_retires_its_predecessor_when_the_roller_dies(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """CR3: a roller that dies after the rename but before its SIGTERM used to
    leave the old broker alive forever (it never idles out while sessions are
    registered). The successor now signals its verified predecessor itself."""
    old_pid = _start_old_broker(config_root, reaper)
    roller = subprocess.run(
        [sys.executable, "-c", _DYING_ROLLER, str(config_root)],
        env={**os.environ, "PYTHONPATH": TREE},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert roller.returncode == 9, (roller.returncode, roller.stderr[-2000:])
    new_pid = _status(config_root).get("pid")
    reaper.add(new_pid)
    assert isinstance(new_pid, int) and new_pid != old_pid

    def old_gone() -> bool:
        try:
            return os.waitpid(old_pid, os.WNOHANG)[0] == old_pid
        except ChildProcessError:
            return process_info(old_pid) is None

    assert _await(old_gone, timeout=10), "the orphaned predecessor was never retired"


def test_a_swapped_canonical_socket_aborts_the_switch(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """CR6 (rename-time inode re-check): a different socket at the canonical path
    when the roller is about to commit aborts the roll; nothing is renamed."""
    old_pid = _start_old_broker(config_root, reaper)
    canonical = socket_path(config_root)
    aside = canonical.with_name("broker.aside")
    impostor = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def before_switch() -> None:
        os.rename(canonical, aside)
        impostor.bind(str(canonical))
        os.chmod(canonical, 0o600)

    from local_operator.secrets.roll import roll_broker

    try:
        result = roll_broker(config_root, force=True, before_switch=before_switch)
        reaper.add(result.new_pid)
        assert result.status == "failed" and result.code == "switch", result
        assert "changed during the roll" in result.reason
        assert process_info(old_pid) is not None
        assert not successor_socket_path(config_root).exists()
    finally:
        impostor.close()
        canonical.unlink(missing_ok=True)
        if aside.exists():
            os.rename(aside, canonical)


def test_a_different_pid_at_the_canonical_socket_aborts_the_switch(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR6 (rename-time pid re-check): same inode, but the answering pid is not
    the broker the roll started from ⇒ abort, old broker untouched."""
    old_pid = _start_old_broker(config_root, reaper)
    real_request = client.request

    def before_switch() -> None:
        def lying(operation: str, base: Path | None = None, **fields: object) -> dict[str, Any]:
            reply = real_request(operation, base, **fields)  # type: ignore[arg-type]
            if operation == "status" and fields.get("connection") is None:
                reply = {**reply, "pid": 1}
            return reply

        monkeypatch.setattr(client, "request", lying)

    from local_operator.secrets.roll import roll_broker

    result = roll_broker(config_root, force=True, before_switch=before_switch)
    reaper.add(result.new_pid)
    assert result.status == "failed" and result.code == "switch", result
    assert "different broker" in result.reason
    assert process_info(old_pid) is not None


def test_a_recycled_old_pid_is_never_signalled(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR6 (pid-reuse SIGTERM guard): if the old pid's start time no longer
    matches after the rename, the roller sends nothing and says so."""
    from local_operator.secrets import peer

    old_pid = _start_old_broker(config_root, reaper)
    real_info = peer.process_info

    def before_switch() -> None:
        def recycled(pid: int) -> ProcessIdentity | None:
            identity = real_info(pid)
            if identity is not None and pid == old_pid:
                return ProcessIdentity(pid=pid, start_time=identity.start_time + 1)
            return identity

        monkeypatch.setattr(peer, "process_info", recycled)

    from local_operator.secrets.roll import roll_broker

    result = roll_broker(config_root, force=True, before_switch=before_switch)
    reaper.add(result.new_pid)
    assert result.status == "rolled", result
    assert "no longer the old broker" in result.warning
    assert real_info(old_pid) is not None, "a recycled pid was signalled"


def test_channel_only_registration_does_not_advertise_reroll(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """S2 / CR5: ``client.register_session`` returns a bare channel nothing
    re-registers, so it must not claim ``reroll``; the roll defers and names it."""
    old_pid = _start_old_broker(config_root, reaper)
    channel = client.register_session(config_root)
    assert channel is not None
    try:
        lineage = _status(config_root).get("session_lineage") or []
        assert [entry["caps"] for entry in lineage if entry["pid"] == os.getpid()] == [[]]
        from local_operator.secrets.roll import roll_broker

        result = roll_broker(config_root, force=True)
        assert result.status == "deferred" and result.code == "sessions", result
        assert os.getpid() in result.stranded
        assert _status(config_root).get("pid") == old_pid
    finally:
        channel.close()


def test_a_successor_that_exits_early_is_reported_as_exited(
    config_root: Path, keyfile_store: bytes, reaper: Reaper
) -> None:
    """QA minor: an early exit is reported as such, not as \"not ready within 5s\"."""
    old_pid = _start_old_broker(config_root, reaper)

    def spawn(_base: Path | None, _args: tuple[str, ...]) -> int:
        return subprocess.Popen(["true"]).pid

    from local_operator.secrets.roll import roll_broker

    result = roll_broker(config_root, force=True, spawn=spawn, ready_timeout=5.0)
    assert result.status == "failed" and result.code == "readiness", result
    assert "exited before answering" in result.reason and "within" not in result.reason
    assert _status(config_root).get("pid") == old_pid


def test_a_protocol_mismatch_deferral_names_a_working_remedy() -> None:
    """CR4: ``restart`` refuses a broker speaking another protocol, so neither
    the refusal nor the broker's own version error may point at ``restart``."""
    import inspect

    from local_operator.secrets import broker as broker_module
    from local_operator.secrets import roll as roll_module

    assert "lop secret broker stop" in inspect.getsource(roll_module._roll_locked)
    handler = inspect.getsource(broker_module.SecretBroker._handle_requests)
    assert "lop secret broker stop" in handler and "lop secret broker restart" not in handler


def test_an_exited_session_does_not_abort_the_switch() -> None:
    """S1 re-read, no false abort: a session the old broker still lists only
    because its watcher has not ticked since it exited is not "late"."""
    from local_operator.secrets.roll import _unknown_to_successor

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    me = process_info(os.getpid())
    assert me is not None
    # dict[str, object]: dict is invariant, so the literal must carry the
    # parameter's declared type rather than its inferred nested shape.
    old: dict[str, object] = {
        "session_lineage": [
            {"pid": gone.pid, "start_time": 1.0, "caps": ["reroll"]},
            {"pid": me.pid, "start_time": me.start_time, "caps": ["reroll"]},
        ]
    }
    none_held: dict[str, object] = {"session_lineage": [], "expected_lineage": []}
    assert _unknown_to_successor(old, none_held) == [me.pid]
    held: dict[str, object] = {"expected_lineage": [{"pid": me.pid, "start_time": me.start_time}]}
    assert _unknown_to_successor(old, held) == []


# --- round-2 convergence (S6, S7, S8/CR8/CR9, Q3) ----------------------------------


def _silent_broker(config_root: Path) -> socket.socket:
    """Accepts and never answers: a held/wedged broker as the client sees it."""
    path = socket_path(config_root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(64)
    return listener


def test_a_held_first_attempt_that_outlives_the_sentinel_never_falls_open(
    config_root: Path, keyfile_store: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8/CR8/CR9: the FIRST attempt is inside the roll budget too.

    The sentinel is live when the read starts and expires while the broker
    holds it. Before, the first attempt ran unclipped with ``seen_roll`` unset,
    timed out after the sentinel was gone, and its ``BrokerUnavailable`` went
    to the fallback; now ``seen_roll`` is seeded from the sentinel present at
    the start, the attempt is clipped to ``ROLL_WAIT_S``, and it fails CLOSED.
    """
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.setattr(client, "ROLL_WAIT_S", 0.8)
    monkeypatch.setattr(client, "REQUEST_TIMEOUT_S", 2.5)
    listener = _silent_broker(config_root)
    try:
        _write_sentinel(rolling_path(config_root), 1, ttl=0.3)
        started = time.monotonic()
        with pytest.raises(client.BrokerRolling):
            client.retrieve(SECRET_NAME, config_root)
        elapsed = time.monotonic() - started
    finally:
        listener.close()
    # CR9: bounded by the roll budget, not by a full request timeout.
    assert elapsed < client.ROLL_WAIT_S + 0.6, f"took {elapsed:.2f}s"


def test_a_legacy_broker_session_registered_mid_roll_is_still_found() -> None:
    """S7: a broker from before the roll reports bare ``sessions`` pids only.

    The first roll after an update always starts from such a broker; reading
    only ``session_lineage`` made the pre-rename re-read a no-op there.
    """
    from local_operator.secrets.roll import _unknown_to_successor

    me = process_info(os.getpid())
    assert me is not None
    legacy: dict[str, object] = {"pid": 1, "sessions": [me.pid]}
    successor: dict[str, object] = {"session_lineage": [], "expected_lineage": []}
    assert _unknown_to_successor(legacy, successor) == [me.pid]
    known: dict[str, object] = {"session_lineage": [{"pid": me.pid, "start_time": me.start_time}]}
    assert _unknown_to_successor(legacy, known) == []
    # An exited legacy session is not "late" either.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    exited: dict[str, object] = {"sessions": [gone.pid]}
    assert _unknown_to_successor(exited, successor) == []


def test_the_sentinel_counts_from_the_old_brokers_exit(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S6: a late registrant re-registers only after the old broker's drain, so
    the client's retry window (the sentinel) must run ROLL_GRACE_S from the
    observed EXIT, not from the rename. A slow exit is simulated by delaying
    the SIGTERM itself; before the fix the sentinel's deadline stayed pinned
    to the pre-signal write and expired that much earlier.
    """
    import json

    from local_operator.secrets import broker as broker_module
    from local_operator.secrets import roll as roll_module

    old_pid = _start_old_broker(config_root, reaper)
    real_kill = os.kill
    delay_s = 1.5

    def slow_term(pid: int, sig: int) -> None:
        if pid == old_pid and sig == signal.SIGTERM:
            time.sleep(delay_s)
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", slow_term)
    result = roll_module.roll_broker(config_root, force=True)
    finished = time.time()
    monkeypatch.setattr(os, "kill", real_kill)
    reaper.add(result.new_pid)
    assert result.status == "rolled", result
    sentinel = json.loads(rolling_path(config_root).read_text(encoding="utf-8"))
    assert sentinel["deadline"] >= finished + broker_module.ROLL_GRACE_S - 0.5, (
        sentinel["deadline"] - finished
    )


def test_a_legacy_predecessor_version_is_read_off_its_process(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q3: a predecessor whose ``status`` has no ``version`` is labelled from its image."""
    from local_operator.secrets import broker as broker_module

    me = os.getpid()
    monkeypatch.setattr(client, "request", lambda *_a, **_k: {"pid": me, "sessions": []})
    monkeypatch.setattr(
        broker_module, "_predecessor_build_version", lambda pid: "0.110.1", raising=False
    )
    inputs = broker_module._successor_inputs(config_root, me)
    assert inputs is not None and inputs.version == "0.110.1"


def test_an_unreadable_predecessor_version_says_why(
    config_root: Path, keyfile_store: bytes
) -> None:
    """Q3: the audit row names the cause instead of "(unknown)"."""
    from local_operator.secrets import broker as broker_module

    successor = SecretBroker(
        config_root,
        key_provider=lambda: keyfile_store,
        idle_shutdown_s=0,
        predecessor_version="",
    )
    successor._audit_roll(4242)
    rows = _roll_rows(keyfile_store, config_root)
    assert len(rows) == 1, rows
    assert "(unknown)" not in rows[0].outcome
    assert f"({broker_module.PREDECESSOR_VERSION_UNREPORTED})" in rows[0].outcome


# --- #4389 round 3: one deadline through every phase (S5/CR9) ------------------


class _Clock:
    """A deterministic ``time.monotonic``: it moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _SplitReplySocket:
    """A connection whose reply arrives in two receives; the first spends the budget."""

    def __init__(self, clock: _Clock, spend: float) -> None:
        import json
        import struct

        body = json.dumps({"ok": True, "pid": 1}).encode()
        self._chunks = [struct.pack("!I", len(body)), body]
        self._clock = clock
        self._spend = spend
        self.recvs = 0
        self._timeout: float | None = None

    def settimeout(self, value: float | None) -> None:
        self._timeout = value

    def gettimeout(self) -> float | None:
        return self._timeout

    def sendall(self, _data: bytes) -> None:
        pass

    def recv(self, _count: int) -> bytes:
        self.recvs += 1
        chunk = self._chunks.pop(0)
        self._clock.now += self._spend
        return chunk

    def close(self) -> None:
        pass


def test_a_reply_split_across_receives_cannot_outlive_the_deadline(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5/CR9: the remaining budget is recomputed before EACH receive.

    The header arrives as the budget runs out; the body would arrive on a second
    receive. Before, one timeout computed up front was reused for every phase,
    so the second receive ran past the deadline and the call succeeded late.
    """
    from local_operator.secrets import protocol

    assert protocol._HEADER.format in ("!I", ">I")
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    link = _SplitReplySocket(clock, spend=2.0)
    monkeypatch.setattr(client, "_connect", lambda path, timeout=0: link)
    with pytest.raises(client.BrokerUnavailable):
        client._bounded_request("ping", config_root, clock.now + 1.0)
    assert link.recvs == 1, "a receive started after the caller's deadline"


def test_an_exhausted_budget_makes_no_connection(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5/CR9: no work after expiry — the old 100 ms floor connected anyway."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)

    def connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("connected after the deadline")

    monkeypatch.setattr(client, "_connect", connect)
    with pytest.raises(client.BrokerUnavailable):
        client._bounded_request("ping", config_root, clock.now - 0.01)


def test_no_spawn_after_the_first_ping_spent_the_deadline(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5/CR9: a silent broker's first ping eats the budget; nothing is spawned after."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    deadline = clock.now + 1.0
    pings: list[float] = []

    def silent_ping(_base: object, _deadline: object) -> bool:
        pings.append(clock.now)
        clock.now = max(clock.now, deadline)  # the ping waits until the budget is gone
        return False

    spawned: list[object] = []
    monkeypatch.setattr(client, "_alive", silent_ping)
    monkeypatch.setattr(client, "_spawn_broker", lambda base=None: spawned.append(base) or -1)
    monkeypatch.setattr(client, "_record_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "PEER_AUTHENTICATION_SUPPORTED", True)
    assert client.ensure_broker(config_root, deadline=deadline) is False
    assert spawned == [], "a broker was spawned after the caller's deadline"
    assert len(pings) == 1, f"pinged again after the deadline: {pings}"


def test_a_real_roll_keeps_the_lineage_marker_through_reregistration(
    config_root: Path, keyfile_store: bytes, reaper: Reaper, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-3 + S6 tail, on a real roll with AUTOMATIC re-registration left on.

    R3-3: the session's lineage marker survives the automatic re-registration
    with the successor, its child is served through it (with the notice), and
    the marker is gone once the session closes.

    S6 tail: the sentinel written just before the old broker is signalled must
    outlive the whole bounded exit wait (``OLD_EXIT_TIMEOUT_S``), then be
    re-armed for the successor's hold from the exit.
    """
    from local_operator.secrets import roll as roll_module
    from local_operator.secrets.broker import ROLL_GRACE_S
    from local_operator.secrets.session import lineage_dir

    writes: list[tuple[float, bool]] = []
    real_write = roll_module._write_sentinel

    def spy(path, old_pid, *, ttl=roll_module.SENTINEL_TTL_S, signalled=False):
        writes.append((ttl, signalled))
        return real_write(path, old_pid, ttl=ttl, signalled=signalled)

    monkeypatch.setattr(roll_module, "_write_sentinel", spy)
    _start_old_broker(config_root, reaper)
    seen: list[str] = []
    registration = session_module.register_session(
        lambda _name, value: seen.append(value.decode()), base=config_root
    )
    assert registration is not None
    mine = f"{os.getpid()}-"

    def markers() -> list[str]:
        directory = lineage_dir(config_root)
        if not directory.is_dir():
            return []
        return [p.name for p in directory.iterdir() if p.name.startswith(mine)]

    try:
        assert len(markers()) == 1, "no lineage marker after the first registration"
        before = markers()
        result = roll_module.roll_broker(config_root, force=True)
        reaper.add(result.new_pid)
        assert result.status == "rolled", result
        assert _await(
            lambda: os.getpid() in (_status(config_root).get("sessions") or ())
        ), "the session did not re-register with the successor on its own"
        assert markers() == before, "the marker changed across the automatic re-registration"
        out, _ = _finish(_child_retrieve(config_root))
        assert "FALLBACK" not in out, out
        assert f"VALUE {SECRET_VALUE}" in out, out
        assert SECRET_VALUE in seen
    finally:
        registration.close()
    assert markers() == [], "the marker outlived the session's close"

    signalled = [ttl for ttl, flagged in writes if flagged]
    assert signalled, writes
    assert signalled[0] >= roll_module.OLD_EXIT_TIMEOUT_S + ROLL_GRACE_S, (
        f"the pre-signal sentinel ({signalled[0]}s) lapses inside the "
        f"{roll_module.OLD_EXIT_TIMEOUT_S}s exit wait"
    )
    assert signalled[-1] == ROLL_GRACE_S, "not re-armed for the hold after the exit"
