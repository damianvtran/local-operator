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
from contextlib import suppress
from dataclasses import replace
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


def test_register_without_a_ticket_is_refused(broker: SecretBroker, config_root: Path) -> None:
    """`register` is AUTHENTICATED; an unticketed caller gets nothing (review R1).

    This test previously asserted ``ok`` on this exact frame, documenting the
    hole as intended behaviour — which is why 141 green tests missed a complete
    authentication bypass. ``register`` used to be dispatched before the
    authorization gate, and because the ancestry walk yields the peer as the
    first element of its own chain, any process that registered itself became
    its own authorizing ancestor and could then ask for the master key.

    Both halves are asserted here: the registration is refused, and no session
    appears in the broker's table as a result of trying.
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
        assert not reply.get("ok"), reply
        assert reply.get("code") == "unauthorized", reply
        assert "ticket" in str(reply.get("error", "")).lower(), reply
    finally:
        connection.close()

    status = client.broker_status(config_root) or {}
    assert os.getpid() not in (status.get("sessions") or []), "the refusal still registered a pid"


def test_a_peer_cannot_register_a_pid_it_does_not_own(
    broker: SecretBroker, config_root: Path
) -> None:
    """The pid comes from the kernel, never from the request body.

    A self-declared pid would let any process register an arbitrary victim and
    then authorize its own descendants through it. Ticketed here, since
    registration is authenticated (R1) — the point under test is that a VALID
    registrant still cannot choose which pid it registers.
    """
    from local_operator.secrets.keys import registration_ticket
    from local_operator.secrets.protocol import encode_bytes

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    connection.connect(str(broker.path))
    try:
        send_frame(
            connection,
            {
                "version": PROTOCOL_VERSION,
                "op": "register",
                "pid": 1,
                "session_id": "forged",
                "ticket": encode_bytes(registration_ticket(config_root)),
            },
        )
        reply = recv_frame(connection)
        assert reply.get("ok"), reply
        # Registered as the CONNECTING process, not the pid it asked for.
        assert reply.get("registered") == os.getpid()
    finally:
        connection.close()


# --- §6 redaction ordering --------------------------------------------------


def test_session_is_notified_before_the_child_is_served(
    broker: SecretBroker,
    config_root: Path,
    master_key: bytes,
    broker_store: SecretStore,
    tmp_path: Path,
) -> None:
    """The §6 invariant, proven by making the ack SLOW.

    A fast ack passes even if the broker replies first, because the
    notification wins the race anyway. Delaying the ack separates the two
    designs: if the ordering is a real invariant the retrieval cannot return
    until the ack completes, so its elapsed time is bounded below by the delay.

    The retrieval runs in a CHILD process, which is what §6 case (2) actually
    describes: the session process itself never triggers a notice, because a
    value it fetches lands in its own memory where it registers the redaction
    directly — and demanding an ack from the very process blocked on the reply
    would deadlock it.
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
    script = tmp_path / "child.py"
    script.write_text(_retrieve_script(config_root))
    try:
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=60
        )
        elapsed = time.monotonic() - started
    finally:
        thread.join(timeout=5)
        channel.close()

    assert "OK" in result.stdout, f"{result.stdout}{result.stderr}"
    assert SECRET_VALUE.decode() in result.stdout
    assert notified == [("DEMO_TOKEN", SECRET_VALUE)]
    assert elapsed >= ack_delay, (
        f"the retrieval returned in {elapsed:.3f}s despite a {ack_delay}s ack — "
        "the broker replied before the session acknowledged, so the ordering is a "
        "race rather than an invariant"
    )


def test_ordering_holds_for_concurrent_retrievals(
    broker: SecretBroker, session: list[tuple[str, bytes]], config_root: Path, tmp_path: Path
) -> None:
    """~10 concurrent sessions is the real workload (design §4).

    Retrievals run as CHILD processes: only a descendant produces a §6 notice,
    and this test is about those notices not interleaving or losing each
    other's acks on one session's stream.
    """
    errors: list[str] = []
    script = tmp_path / "concurrent.py"
    script.write_text(_retrieve_script(config_root))

    def fetch() -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True, timeout=60
            )
            if SECRET_VALUE.decode() not in result.stdout:
                errors.append(f"wrong value: {result.stdout}{result.stderr}")
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=fetch) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert len(session) == 10, f"expected 10 notifications, saw {len(session)}"


def test_a_session_that_never_acks_denies_the_child_within_the_timeout(
    config_root: Path, master_key: bytes, broker_store: SecretStore, tmp_path: Path
) -> None:
    """A wedged session DENIES the descendant, bounded, rather than leaking (R3).

    This test used to assert the opposite — that the value was served anyway
    after the timeout — which made §6's "invariant" language false: a session
    that never acks is precisely the case where nothing downstream can scrub
    the value, because a `$( )` retrieval never passes through any filter. An
    attacker who controls whether the session acks could choose that outcome
    deliberately.

    What must still hold is that it is BOUNDED: the child gets a clear refusal
    quickly, never a hang.
    """
    instance = SecretBroker(
        config_root, key_provider=lambda: master_key, idle_shutdown_s=0, notify_ack_timeout_s=0.5
    )
    instance.start()
    channel = client.register_session(config_root, session_id="wedged")
    assert channel is not None
    # Never read from `channel`, so no ack is ever sent.
    script = tmp_path / "child.py"
    script.write_text(_retrieve_script(config_root))
    try:
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=60
        )
        elapsed = time.monotonic() - started
        assert "DENIED" in result.stdout, f"{result.stdout}{result.stderr}"
        assert SECRET_VALUE.decode() not in result.stdout, "an unscrubbable value was served"
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


# --- the R1 bypass, as real processes (review round 1) ----------------------


@pytest.fixture
def hardened_broker(config_root: Path, master_key: bytes, broker_store: SecretStore):
    """An UNLOCKED passphrase-tier broker: the tier where ancestry is load-bearing.

    Built by actually hardening the store — `wrap_master_key` removes the
    plaintext key file — so `key_mode` reports `passphrase` and the broker's
    tier-dependent checks take the same branch they take in production. A
    fixture that merely withheld the key provider would leave a `master.key` on
    disk and test the wrong tier.
    """
    from local_operator.secrets.keys import wrap_master_key

    wrap_master_key(config_root, master_key, "test-passphrase")
    instance = SecretBroker(config_root, idle_shutdown_s=0)
    instance.start()
    instance.unlock_with_key(master_key)  # as `lop secret unlock` would
    try:
        yield instance
    finally:
        instance.stop(timeout=5)


def _self_register_exploit(config_root: Path, socket_file: Path, output: Path) -> str:
    """The reviewer's exploit: double-fork + setsid, then talk raw frames.

    Reparented to launchd (ppid 1) — the exact shape §8's table calls a
    "Detached script" — and it speaks the protocol directly rather than going
    through the client, because the client is ours and an attacker's would not
    be. It presents the registration ticket too: an attacker that can reach the
    socket can read that file, so a fix that relied on the ticket alone in this
    tier would be no fix at all.
    """
    from local_operator.secrets.keys import registration_ticket
    from local_operator.secrets.protocol import encode_bytes

    ticket = encode_bytes(registration_ticket(config_root))
    return textwrap.dedent(f"""
        import os, sys, socket, time
        if os.fork():
            os._exit(0)
        os.setsid()
        time.sleep(0.5)
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        from local_operator.secrets.protocol import (
            PROTOCOL_VERSION, recv_frame, send_frame, decode_bytes,
        )
        lines = [f"ppid={{os.getppid()}}"]

        def call(**fields):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect({str(socket_file)!r})
            send_frame(s, {{"version": PROTOCOL_VERSION, "ticket": {ticket!r}, **fields}})
            return s, recv_frame(s)

        held, reply = call(op="register", session_id="impostor")
        lines.append(f"register_ok={{bool(reply.get('ok'))}}")
        for op, field in (("retrieve", "value"), ("key", "key")):
            s, reply = call(op=op, name="DEMO_TOKEN")
            if reply.get("ok"):
                lines.append(f"LEAKED {{op}} {{decode_bytes(reply[field])!r}}")
            else:
                lines.append(f"denied {{op}}: {{reply.get('code')}}")
            s.close()
        held.close()
        open({str(output)!r}, "w").write("\\n".join(lines))
        """)


def test_self_registration_cannot_buy_the_master_key(
    hardened_broker: SecretBroker, config_root: Path, tmp_path: Path
) -> None:
    """R1: a detached process may not make itself a session and read everything.

    Reproduces the reviewer's exploit verbatim against an unlocked
    passphrase-tier broker. Before the fix this printed the canary value and a
    32-byte master key from one 60-byte JSON frame; the ancestry walk yielded
    the peer as the first element of its own chain, so registering was enough
    to become one's own authorizing ancestor.

    Asserts BOTH halves: the registration is refused, and neither the value nor
    the key is ever served.
    """
    output = tmp_path / "exploit.out"
    script = tmp_path / "exploit.py"
    script.write_text(_self_register_exploit(config_root, hardened_broker.path, output))
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not output.exists():
        time.sleep(0.1)
    assert output.exists(), "the detached attacker never reported back"
    text = output.read_text()

    assert "ppid=1" in text, f"the attacker was not actually detached: {text}"
    assert "register_ok=False" in text, f"self-registration was allowed: {text}"
    assert "LEAKED" not in text, text
    assert SECRET_VALUE.decode() not in text, text


def test_the_operator_terminal_works_after_unlock_but_a_detached_script_does_not(
    hardened_broker: SecretBroker, config_root: Path, tmp_path: Path
) -> None:
    """The hardened tier must be USABLE, and only by the one who unlocked it.

    Two properties that have to hold together, which is why they are asserted
    in one test: unlocking grants the operator's own shell standing for this
    boot (without it, `harden` bricks the store — QA Q2), and that grant must
    not extend to the detached attacker running as the same uid (without that,
    it is R1 by another door).
    """
    root = str(Path(__file__).resolve().parents[3])
    # Unlock and then retrieve, from ONE shell: the grant is given to the
    # parent of the unlocking process, so the retrieval has to descend from
    # that same shell to prove the grant works the way an operator uses it.
    unlock = tmp_path / "unlock.py"
    unlock.write_text(textwrap.dedent(f"""
        import sys, socket
        sys.path.insert(0, {root!r})
        from local_operator.secrets.protocol import PROTOCOL_VERSION, recv_frame, send_frame
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(20)
        s.connect({str(hardened_broker.path)!r})
        send_frame(s, {{"version": PROTOCOL_VERSION, "op": "unlock",
                        "passphrase": "test-passphrase"}})
        print(recv_frame(s).get("ok"))
        """))
    fetch = tmp_path / "terminal_get.py"
    fetch.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {root!r})
        from pathlib import Path
        from local_operator.secrets import client
        value = client.retrieve("DEMO_TOKEN", Path({str(config_root)!r}))
        print("TERMINAL", value.decode())
        """))
    shell = subprocess.run(
        ["bash", "-c", f"{sys.executable} {unlock} && {sys.executable} {fetch}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "TERMINAL" in shell.stdout, f"{shell.stdout}{shell.stderr}"
    assert SECRET_VALUE.decode() in shell.stdout, "the unlocking terminal was denied its own store"

    # Same broker, now unlocked: the detached attacker still gets nothing.
    output = tmp_path / "after-unlock.out"
    script = tmp_path / "after-unlock.py"
    script.write_text(_self_register_exploit(config_root, hardened_broker.path, output))
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not output.exists():
        time.sleep(0.1)
    assert output.exists(), "the detached attacker never reported back"
    text = output.read_text()
    assert "ppid=1" in text, text
    assert "register_ok=False" in text, f"self-registration was allowed post-unlock: {text}"
    assert "LEAKED" not in text, text
    assert SECRET_VALUE.decode() not in text, text


def test_a_session_registered_on_a_terminal_grant_dies_with_that_terminal(
    hardened_broker: SecretBroker, config_root: Path, tmp_path: Path
) -> None:
    """QA Q8: the unlock grant must be bounded by the terminal's LIFETIME.

    §13 says the grant lasts as long as the shell the operator unlocked in, and
    for a plain process that held: kill the shell and its children are denied.
    But a process inside that terminal may `register` ITSELF as a session —
    correctly permitted, since it descends from an unlocked terminal — and a
    registered session is an INDEPENDENT authorizing entry. QA measured the
    escape end to end: with the granting shell SIGKILLed and the squatter
    reparented to launchd, it was still served the secret, while a control
    process was denied at the same moment. The grant was bounded by the
    BROKER's lifetime, not the terminal's, which is not what the doc promises.

    Driven with real processes because the whole mechanism is what the kernel
    reports about a lineage; a faked pid would prove nothing about it. The
    squatter is a grandchild that registers, then reports whether it can still
    retrieve after its granting shell is gone.
    """
    root = str(Path(__file__).resolve().parents[3])
    started = tmp_path / "squatter-registered"
    verdict = tmp_path / "squatter-verdict"
    squatter = tmp_path / "squat.py"
    squatter.write_text(textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {root!r})
        from pathlib import Path
        from local_operator.secrets import client
        base = Path({str(config_root)!r})
        # Register while the granting terminal is still alive: this is the
        # escalation under test, and it is ALLOWED at this moment.
        channel = client.register_session(base, session_id="squatter")
        open({str(started)!r}, "w").write("registered" if channel else "refused")
        # Wait for the granting shell to be killed, then ask again.
        while not os.path.exists({str(tmp_path / "shell-is-dead")!r}):
            time.sleep(0.1)
        try:
            value = client.retrieve("DEMO_TOKEN", base)
            open({str(verdict)!r}, "w").write(f"SERVED {{value.decode()}}")
        except Exception as exc:
            open({str(verdict)!r}, "w").write(f"DENIED {{type(exc).__name__}}: {{exc}}")
        """))
    # One shell: it unlocks (taking the grant on ITSELF as the unlocker's
    # parent), then launches the squatter as its own descendant, then reports
    # its pid so the test can kill exactly that shell.
    unlock = tmp_path / "unlock.py"
    unlock.write_text(textwrap.dedent(f"""
        import sys, socket
        sys.path.insert(0, {root!r})
        from local_operator.secrets.protocol import PROTOCOL_VERSION, recv_frame, send_frame
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(20)
        s.connect({str(hardened_broker.path)!r})
        send_frame(s, {{"version": PROTOCOL_VERSION, "op": "unlock",
                        "passphrase": "test-passphrase"}})
        assert recv_frame(s).get("ok")
        """))
    # `A && B &` would make bash fork a SUBSHELL to hold the `&&` list, and the
    # grant would land on that subshell rather than on the shell this test can
    # kill — measured: killing the outer bash then left a live grant holder and
    # the assertion below failed against a working fix. Sequencing with `;`
    # keeps the unlock in the shell itself, which is also the shape a real
    # prompt has.
    shell = subprocess.Popen(
        [
            "bash",
            "-c",
            f"{sys.executable} {unlock}; "
            f"nohup {sys.executable} {squatter} >/dev/null 2>&1 & "
            "sleep 300",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not started.exists():
            time.sleep(0.1)
        assert started.exists(), "the squatter never reported back"
        assert started.read_text() == "registered", (
            "the squatter could not register inside the granted terminal, so this test "
            "is no longer exercising the escape it exists for"
        )

        # Kill the granting terminal. Its grant, and everything that borrowed
        # standing from it, must stop authorizing.
        shell.kill()
        shell.wait(timeout=30)
    finally:
        with suppress(OSError):
            shell.kill()
    (tmp_path / "shell-is-dead").write_text("dead")

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not verdict.exists():
        time.sleep(0.1)
    assert verdict.exists(), "the squatter never answered after the terminal died"
    text = verdict.read_text()
    assert text.startswith(
        "DENIED"
    ), f"a session registered on a dead terminal's grant is still served: {text}"
    assert SECRET_VALUE.decode() not in text, text


def test_registrations_are_bounded(broker: SecretBroker, config_root: Path) -> None:
    """R4: registrations must not exhaust the broker for legitimate callers.

    Registration is authenticated now, so this is no longer a free
    unauthenticated flood — but a runaway or compromised caller that holds the
    ticket must still not be able to wedge the daemon, and the session watchers
    must draw from their own budget rather than the request workers'.
    """
    from local_operator.secrets.broker import MAX_SESSIONS, MAX_WORKERS
    from local_operator.secrets.keys import registration_ticket
    from local_operator.secrets.protocol import encode_bytes

    ticket = encode_bytes(registration_ticket(config_root))
    held: list[socket.socket] = []
    try:
        # Registrations are keyed by the KERNEL-reported pid, and every
        # connection here comes from this one process, so these all collapse
        # onto one slot: re-registration replaces rather than accumulates,
        # which is itself the first half of the bound.
        # A fixed, small flood: enough to prove one pid cannot accumulate
        # slots, and independent of MAX_SESSIONS so raising that constant
        # cannot turn this test into an fd-exhaustion error instead of an
        # assertion failure.
        for _ in range(40):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(5)
            connection.connect(str(broker.path))
            send_frame(
                connection,
                {"version": PROTOCOL_VERSION, "op": "register", "ticket": ticket},
            )
            recv_frame(connection)
            held.append(connection)

        status = client.broker_status(config_root) or {}
        sessions = status.get("sessions") or []
        assert len(sessions) == 1, f"one pid took {len(sessions)} session slots: {sessions}"

        # The cap must be real rather than incidental to that collapsing, and
        # it must be budgeted apart from the request workers — the R4 failure
        # was 64 registrations starving every legitimate caller.
        assert MAX_SESSIONS < MAX_WORKERS, (
            f"MAX_SESSIONS ({MAX_SESSIONS}) must stay below MAX_WORKERS ({MAX_WORKERS}) so a "
            "full session table cannot consume the capacity requests need"
        )
        # And the broker must still answer a fresh caller after the flood.
        assert client.is_running(config_root), "the broker stopped answering after a flood"
    finally:
        for connection in held:
            with suppress(OSError):
                connection.close()


def test_register_is_audited(broker: SecretBroker, config_root: Path, master_key: bytes) -> None:
    """R5: the verb that grants standing to ask must appear in the chain (§12).

    Both outcomes: without an audit record of a denial, the R1 bypass would
    have left no trace at all.
    """
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(5)
    connection.connect(str(broker.path))
    try:
        send_frame(connection, {"version": PROTOCOL_VERSION, "op": "register"})
        assert not recv_frame(connection).get("ok")
    finally:
        connection.close()

    channel = client.register_session(config_root, session_id="audited")
    assert channel is not None
    try:
        events = [
            entry.event
            for entry in SecretStore(master_key, base=config_root).audit_entries(limit=50)
        ]
    finally:
        channel.close()
    assert any("register" in event for event in events), events
    assert any("deny:register" in event for event in events), events


def test_a_peer_is_not_its_own_authorizing_ancestor() -> None:
    """R1, second half: `_walk` yields the peer first, and that must not authorize.

    Unit-level on purpose. The end-to-end tests above prove the DOOR is shut
    (`register` is authenticated), so they stay green even with this half
    reverted — which is exactly why it needs its own guard: if a later change
    ever reopens registration, self-as-own-ancestor must not silently become a
    second bypass again. The two halves are independent, and the review asked
    for both to hold alone.

    Asserts the distinction that matters: a peer registered as a session is
    authorized AS ITSELF (its own pin verified), while an UNREGISTERED peer
    gets nothing from appearing at the head of its own ancestry chain.
    """
    from local_operator.secrets.peer import authorize, process_info

    me = process_info(os.getpid())
    assert me is not None

    # Not registered: appearing first in one's own walk must not authorize.
    allowed, reason = authorize(me, {})
    assert not allowed, reason

    # A registered session speaking for itself is allowed, and the reason says
    # so explicitly rather than claiming a descendant relationship.
    allowed, reason = authorize(me, {me.pid: me})
    assert allowed, reason
    assert "acting for itself" in reason, reason

    # The pid pin still governs: a same-pid impostor with a different identity
    # is refused even though the pid is registered.
    impostor = replace(me, unique_id=(me.unique_id or 0) + 1, start_time=me.start_time + 1)
    allowed, reason = authorize(me, {me.pid: impostor})
    assert not allowed, reason
