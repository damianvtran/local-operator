"""The broker daemon, end to end, with real processes where it matters.

Design reference: ``docs/design/secret-store.md`` §2.1, §6, §13.

The security-relevant cases here deliberately spawn REAL subprocesses rather
than faking a peer identity. A mocked pid proves nothing about a boundary whose
entire mechanism is what the kernel reports about the process on the other end
of a socket — the two cases from spike 9 (an agent's genuine ``bash``
grandchild ALLOWED, a detached ``setsid`` script DENIED) are only meaningful
when the processes are real.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from local_operator.secrets import client
from local_operator.secrets.broker import SecretBroker
from local_operator.secrets.errors import BrokerUnavailable
from local_operator.secrets.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    recv_frame,
    send_frame,
    socket_path,
)
from local_operator.secrets.store import SecretStore

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="the broker is implemented for macOS and Linux"
)

SECRET_VALUE = b"s3cr3t-token-value"


@pytest.fixture
def broker_store(config_root: Path, master_key: bytes) -> SecretStore:
    store = SecretStore(master_key, base=config_root)
    store.initialize()
    store.set("DEMO_TOKEN", SECRET_VALUE, description="broker test")
    return store


@pytest.fixture
def broker(config_root: Path, master_key: bytes, broker_store: SecretStore):
    """A running broker with the key already in memory."""
    instance = SecretBroker(config_root, key_provider=lambda: master_key, idle_shutdown_s=0)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop(timeout=5)


@pytest.fixture
def session(broker: SecretBroker, config_root: Path):
    """A registered session that answers §6 notifications, as a real one does."""
    channel = client.register_session(config_root, session_id="test-session")
    assert channel is not None, "the session could not register with the broker"
    received: list[tuple[str, bytes]] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            notice = client.read_notification(channel)
            if notice is None:
                return
            received.append(notice)
            client.acknowledge(channel)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield received
    finally:
        stop.set()
        channel.close()
        thread.join(timeout=2)


def _retrieve_script(config_root: Path) -> str:
    """A client that retrieves DEMO_TOKEN and prints the outcome."""
    return textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        from pathlib import Path
        from local_operator.secrets import client
        try:
            value = client.retrieve("DEMO_TOKEN", Path({str(config_root)!r}))
            print(f"OK {{value.decode()}}")
        except Exception as exc:
            print(f"DENIED {{type(exc).__name__}}: {{exc}}")
        """)


# --- socket surface ---------------------------------------------------------


def test_socket_is_private(broker: SecretBroker) -> None:
    """0600 in a 0700 directory. macOS enforces both at connect() (spike 7)."""
    path = broker.path
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(path.parent.stat().st_mode & 0o777) == "0o700"


@pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason=(
        "root bypasses file mode bits (CAP_DAC_OVERRIDE), so this asserts nothing when "
        "the suite runs as root — measured: uid 0 connects to a mode-000 socket, uid 1000 "
        "gets EACCES. CI's Linux container runs as root; the macOS leg does not."
    ),
)
def test_socket_mode_is_enforced_by_the_kernel(broker: SecretBroker) -> None:
    """The mode bits are a real barrier, not advisory metadata.

    Verified on macOS (spike 7: ``chmod 000`` gave even the owner ``EACCES``)
    and re-verified on Linux as a non-root user. This is what excludes OTHER
    uids outright; peer authentication handles the same-uid case that mode bits
    cannot express.
    """
    path = broker.path
    os.chmod(path, 0o000)
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2)
        with pytest.raises(OSError):
            probe.connect(str(path))
        probe.close()
    finally:
        os.chmod(path, 0o600)


def test_a_stale_socket_from_a_dead_broker_is_reaped(
    config_root: Path, master_key: bytes, broker_store: SecretStore
) -> None:
    """A killed broker leaves an inode; the next one must not be wedged by it."""
    path = socket_path(config_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()  # a corpse, with nothing listening
    instance = SecretBroker(config_root, key_provider=lambda: master_key, idle_shutdown_s=0)
    instance.start()
    try:
        assert client.is_running(config_root)
    finally:
        instance.stop(timeout=5)


# --- ancestry authentication (the spike 9 pair) -----------------------------


def test_agent_bash_grandchild_is_allowed(
    broker: SecretBroker, session: list[tuple[str, bytes]], config_root: Path, tmp_path: Path
) -> None:
    """(a) The real agent shape: session -> bash -> python. MUST be served."""
    script = tmp_path / "kin.py"
    script.write_text(_retrieve_script(config_root))
    result = subprocess.run(
        ["bash", "-c", f"{sys.executable} {script}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "OK" in result.stdout, f"{result.stdout}{result.stderr}"
    assert SECRET_VALUE.decode() in result.stdout


def test_detached_setsid_script_is_denied(
    broker: SecretBroker, session: list[tuple[str, bytes]], config_root: Path, tmp_path: Path
) -> None:
    """(b) The threat-model script: double-forked, setsid, reparented to launchd.

    This is the case the whole ancestry design exists to stop, and the shape a
    script dropped by a bad link actually has.
    """
    output = tmp_path / "attacker.out"
    script = tmp_path / "attacker.py"
    script.write_text(textwrap.dedent(f"""
            import os, sys, time
            if os.fork():
                os._exit(0)
            os.setsid()
            time.sleep(0.8)
            sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
            from pathlib import Path
            from local_operator.secrets import client
            try:
                value = client.retrieve("DEMO_TOKEN", Path({str(config_root)!r}))
                line = f"LEAKED {{value.decode()}} ppid={{os.getppid()}}"
            except Exception as exc:
                line = f"DENIED {{type(exc).__name__}} ppid={{os.getppid()}}"
            open({str(output)!r}, "w").write(line)
            """))
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not output.exists():
        time.sleep(0.1)
    assert output.exists(), "the detached attacker never reported back"
    text = output.read_text()
    assert "DENIED" in text, text
    assert SECRET_VALUE.decode() not in text
    assert "ppid=1" in text, f"the attacker was not actually detached: {text}"


def test_nothing_is_served_without_a_registered_session(
    broker: SecretBroker, config_root: Path, tmp_path: Path
) -> None:
    """No sessions registered means no descendants, so the broker serves nobody."""
    script = tmp_path / "orphan.py"
    script.write_text(_retrieve_script(config_root))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert "DENIED" in result.stdout, result.stdout
    assert SECRET_VALUE.decode() not in result.stdout


def test_a_dead_sessions_descendants_stop_being_authorized(
    broker: SecretBroker, config_root: Path
) -> None:
    """Registration is revoked when the session dies (design §2.1)."""
    channel = client.register_session(config_root, session_id="short-lived")
    assert channel is not None
    status = client.broker_status(config_root) or {}
    assert os.getpid() in (status.get("sessions") or [])

    channel.close()  # the session exits
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = client.broker_status(config_root) or {}
        if os.getpid() not in (status.get("sessions") or []):
            break
        time.sleep(0.1)
    assert os.getpid() not in (status.get("sessions") or []), "a dead session stayed registered"


def test_a_peer_cannot_register_a_pid_it_does_not_own(
    broker: SecretBroker, config_root: Path
) -> None:
    """The pid comes from the kernel, never from the request body.

    A self-declared pid would let any process register an arbitrary victim and
    then authorize its own descendants through it.
    """
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    connection.connect(str(broker.path))
    try:
        send_frame(
            connection,
            {"version": PROTOCOL_VERSION, "op": "register", "pid": 1, "session_id": "forged"},
        )
        reply = recv_frame(connection)
        assert reply.get("ok")
        # Registered as the CONNECTING process, not the pid it asked for.
        assert reply.get("registered") == os.getpid()
    finally:
        connection.close()


# --- §6 redaction ordering --------------------------------------------------


def test_session_is_notified_before_the_child_is_served(
    broker: SecretBroker, config_root: Path, master_key: bytes, broker_store: SecretStore
) -> None:
    """The §6 invariant, proven by making the ack SLOW.

    A fast ack passes even if the broker replies first, because the
    notification wins the race anyway. Delaying the ack separates the two
    designs: if the ordering is a real invariant the retrieval cannot return
    until the ack completes, so its elapsed time is bounded below by the delay.
    """
    ack_delay = 0.75
    channel = client.register_session(config_root, session_id="slow-session")
    assert channel is not None
    notified: list[tuple[str, bytes]] = []

    def loop() -> None:
        notice = client.read_notification(channel)
        if notice is None:
            return
        notified.append(notice)
        time.sleep(ack_delay)  # stands in for registering with the redactor
        client.acknowledge(channel)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        value = client.retrieve("DEMO_TOKEN", config_root)
        elapsed = time.monotonic() - started
    finally:
        thread.join(timeout=5)
        channel.close()

    assert value == SECRET_VALUE
    assert notified == [("DEMO_TOKEN", SECRET_VALUE)]
    assert elapsed >= ack_delay, (
        f"the retrieval returned in {elapsed:.3f}s despite a {ack_delay}s ack — "
        "the broker replied before the session acknowledged, so the ordering is a "
        "race rather than an invariant"
    )


def test_ordering_holds_for_concurrent_retrievals(
    broker: SecretBroker, session: list[tuple[str, bytes]], config_root: Path
) -> None:
    """~10 concurrent sessions is the real workload (design §4)."""
    errors: list[str] = []

    def fetch() -> None:
        try:
            if client.retrieve("DEMO_TOKEN", config_root) != SECRET_VALUE:
                errors.append("wrong value")
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=fetch) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(session) == 10, f"expected 10 notifications, saw {len(session)}"


def test_a_session_that_never_acks_does_not_block_retrieval_forever(
    config_root: Path, master_key: bytes, broker_store: SecretStore
) -> None:
    """A wedged session degrades to a bounded delay, not a hang.

    Denying the retrieval instead would convert a cosmetic risk (a secret in a
    transcript) into a functional failure (the agent's command fails), which is
    the worse trade — see ``SecretBroker._notify_session``.
    """
    instance = SecretBroker(
        config_root, key_provider=lambda: master_key, idle_shutdown_s=0, notify_ack_timeout_s=0.5
    )
    instance.start()
    channel = client.register_session(config_root, session_id="wedged")
    assert channel is not None
    try:
        started = time.monotonic()
        assert client.retrieve("DEMO_TOKEN", config_root) == SECRET_VALUE
        elapsed = time.monotonic() - started
        assert elapsed < 8, f"a silent session stalled the retrieval for {elapsed:.1f}s"
    finally:
        channel.close()
        instance.stop(timeout=5)


# --- failure modes (§13) ----------------------------------------------------


def test_retrieval_fails_fast_when_no_broker_is_running(config_root: Path) -> None:
    """A clear error, never a hang (design §13)."""
    started = time.monotonic()
    with pytest.raises(BrokerUnavailable):
        client.retrieve("DEMO_TOKEN", config_root)
    assert time.monotonic() - started < 10


def test_broker_death_serves_no_value(
    broker: SecretBroker, session: list[tuple[str, bytes]], config_root: Path
) -> None:
    """Once the broker is gone nothing is served — never a stale value."""
    assert client.retrieve("DEMO_TOKEN", config_root) == SECRET_VALUE
    broker.stop(timeout=5)
    with pytest.raises(BrokerUnavailable):
        client.retrieve("DEMO_TOKEN", config_root)


def test_a_version_mismatch_is_reported_not_guessed(broker: SecretBroker) -> None:
    """Skew is routine: the operator updates the runtime under live sessions."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    connection.connect(str(broker.path))
    try:
        send_frame(connection, {"version": PROTOCOL_VERSION + 99, "op": "ping"})
        reply = recv_frame(connection)
        assert not reply.get("ok")
        assert reply.get("code") == "version"
    finally:
        connection.close()


def test_an_oversized_frame_is_refused(broker: SecretBroker) -> None:
    """The length prefix arrives from an UNAUTHENTICATED peer."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    connection.connect(str(broker.path))
    try:
        connection.sendall((2**31).to_bytes(4, "big") + b"{}")
        with pytest.raises((ProtocolError, OSError)):
            recv_frame(connection)
    finally:
        connection.close()


# --- lazy start (#401) ------------------------------------------------------


def test_concurrent_lazy_starts_produce_one_broker(
    config_root: Path, broker_store: SecretStore
) -> None:
    """Ten sessions racing must produce ONE broker, none blocking on the lock.

    ``AGENTS.md`` #401: a blocking ``flock`` deadlocked the event loop, and
    this runs on the session's thread. The lock is ``LOCK_NB``; losers poll for
    the winner's socket instead of waiting.
    """
    results: list[bool] = []
    durations: list[float] = []
    lock = threading.Lock()

    def race() -> None:
        started = time.monotonic()
        ok = client.ensure_broker(config_root)
        with lock:
            results.append(ok)
            durations.append(time.monotonic() - started)

    threads = [threading.Thread(target=race) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    try:
        assert all(results) and len(results) == 10, results
        assert max(durations) < 20, f"a racer blocked for {max(durations):.1f}s"
        status = client.broker_status(config_root)
        assert status is not None
    finally:
        status = client.broker_status(config_root) or {}
        pid = status.get("pid")
        if isinstance(pid, int):
            os.kill(pid, signal.SIGTERM)


# --- socket path length (sun_path) ------------------------------------------


def test_a_deep_config_dir_still_gets_a_bindable_socket(config_root: Path) -> None:
    """``sun_path`` is 104 bytes on macOS and ``bind()`` fails past it.

    The default config dir yields a 49-byte path, but
    ``LOCAL_OPERATOR_CONFIG_DIR`` is operator-controlled and pytest's
    ``tmp_path`` is routinely ~145 bytes — which made every broker test here
    fail with a bare length error before the fallback existed. A store must not
    become unusable because its config dir is deep.
    """
    from local_operator.secrets.protocol import MAX_SOCKET_PATH, socket_path

    deep = config_root / ("d" * 80) / ("e" * 80) / "config"
    deep.mkdir(parents=True)
    path = socket_path(deep)
    assert len(str(path)) <= MAX_SOCKET_PATH, f"{path} is {len(str(path))} bytes"

    # And it must be bindable in fact, not merely short enough in principle.
    from local_operator.secrets.protocol import ensure_runtime_dir

    ensure_runtime_dir(path.parent)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(path))
        assert path.exists()
    finally:
        probe.close()
        path.unlink(missing_ok=True)


def test_two_stores_never_share_a_fallback_socket(config_root: Path) -> None:
    """Colliding would hand a caller a broker holding a DIFFERENT store's key."""
    from local_operator.secrets.protocol import socket_path

    first = config_root / ("a" * 90) / "one"
    second = config_root / ("a" * 90) / "two"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    assert socket_path(first) != socket_path(second)


def test_the_lock_lives_beside_the_socket(config_root: Path) -> None:
    """A lock in another directory would serialise the wrong rendezvous point."""
    from local_operator.secrets.protocol import lock_path, socket_path

    deep = config_root / ("f" * 80) / ("g" * 80) / "config"
    deep.mkdir(parents=True)
    assert lock_path(deep).parent == socket_path(deep).parent


def test_a_runtime_dir_owned_by_another_user_is_refused(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a shared ``/tmp`` another account could pre-create the directory.

    Adopting it would place a secret-broker socket inside a directory somebody
    else controls, so ownership is verified rather than assumed.
    """
    from local_operator.secrets import protocol

    directory = config_root / "runtime"
    directory.mkdir(mode=0o700)
    # Capture the real uid FIRST: patching os.getuid with a lambda that calls
    # os.getuid() patches the function it is calling and recurses.
    impostor = os.getuid() + 1
    monkeypatch.setattr(protocol.os, "getuid", lambda: impostor)
    with pytest.raises(ProtocolError, match="owned by uid"):
        protocol.ensure_runtime_dir(directory)
