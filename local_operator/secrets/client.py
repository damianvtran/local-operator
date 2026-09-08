"""Talking to the broker from a session, a CLI run, or an agent's child.

Design reference: ``docs/design/secret-store.md`` §2.2, §6, §13.

**Every client connects for itself, and this is a security property rather
than a style choice.** Spikes 6 and 7 measured what happens when a connection
is inherited: ``LOCAL_PEERPID`` reports the pid of whoever called ``connect()``,
and that identity outlives the connector, so a process handed a live socket on
fd 3 authenticates as a process that has already exited. An inherited
connection therefore LAUNDERS identity. Nothing in this module caches a
connection across a fork, and nothing passes one to a child.

**The #401 rule is the hard constraint here.** ``AGENTS.md`` records a blocking
``flock`` in the MCP OAuth refresh path deadlocking the TUI's event loop, and
this module runs inside that same process. So: every socket operation has a
hard timeout, the lazy-start lock is ``LOCK_NB`` with a bounded retry and is
never awaited, and no call here can block indefinitely on any outcome —
including the broker being wedged rather than absent, which a naive
"connect and wait" would hang on forever.

**Failure is honest.** When the broker cannot be reached, callers get
:class:`BrokerUnavailable` and the CLI falls back to reading the key file in
``keyfile`` mode. What never happens is a stale or wrong value: there is no
cache to serve from, so "broker down" degrades to "slower path" or to a clear
error in ``passphrase`` mode, never to a plausible wrong answer (§13).
"""

from __future__ import annotations

import fcntl
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from local_operator.secrets.errors import BrokerUnavailable, SecretStoreError
from local_operator.secrets.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_bytes,
    ensure_runtime_dir,
    lock_path,
    recv_frame,
    send_frame,
    socket_path,
)

#: Hard ceiling on reaching the broker. §13 requires a retrieval attempted
#: while the broker is down to return a clear error and NEVER to hang.
CONNECT_TIMEOUT_S = 2.0

#: Hard ceiling on a request/response round trip once connected. Separate from
#: the connect timeout because the failures differ: connect fails fast when
#: nothing listens, while a wedged broker accepts and then never answers, and
#: only this bound covers that.
REQUEST_TIMEOUT_S = 10.0

#: How long to wait for a just-spawned broker to start listening.
STARTUP_TIMEOUT_S = 5.0

#: Non-blocking lock retry cadence during lazy start. Ten sessions racing to
#: start a broker must produce ONE broker, and the losers must poll for the
#: winner's socket rather than block on the lock (#401).
_LOCK_POLL_S = 0.05


def _connect(path: Path, timeout: float = CONNECT_TIMEOUT_S) -> socket.socket:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(str(path))
    except OSError as exc:
        connection.close()
        raise BrokerUnavailable(f"could not reach the secret broker at {path}: {exc}") from exc
    connection.settimeout(REQUEST_TIMEOUT_S)
    return connection


def request(
    operation: str,
    base: Path | None = None,
    *,
    connection: socket.socket | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Send one request and return the reply, raising on a refusal.

    Opens its own connection unless one is supplied, because peer
    authentication happens per connection and a shared one would attribute
    every caller to whoever opened it.
    """
    owned = connection is None
    link = connection if connection is not None else _connect(socket_path(base))
    try:
        send_frame(link, {"version": PROTOCOL_VERSION, "op": operation, **fields})
        reply = recv_frame(link)
    except (OSError, ProtocolError, socket.timeout) as exc:
        raise BrokerUnavailable(f"the secret broker did not answer: {exc}") from exc
    finally:
        if owned:
            try:
                link.close()
            except OSError:  # pragma: no cover
                pass
    if not reply.get("ok"):
        message = str(reply.get("error") or "the secret broker refused the request")
        code = reply.get("code")
        if code in ("unauthorized", "unauthenticated"):
            raise BrokerDenied(message)
        if code == "locked":
            raise BrokerLocked(message)
        raise SecretStoreError(message)
    return reply


class BrokerDenied(SecretStoreError):
    """The broker authenticated the peer and refused it.

    Distinct from :class:`BrokerUnavailable` on purpose: "you are not a
    descendant of a live lop session" is a policy decision the operator may
    need to act on, while an unavailable broker is an operational condition
    that falls back to the key file. Collapsing the two would hide a denial
    behind a silent fallback, which is the one thing a security boundary must
    not do.
    """


class BrokerLocked(SecretStoreError):
    """The store is hardened and this boot has not been unlocked yet."""


def is_running(base: Path | None = None) -> bool:
    """Is a broker listening right now? Never raises, never blocks long."""
    try:
        request("ping", base)
    except SecretStoreError:
        return False
    return True


def broker_status(base: Path | None = None) -> dict[str, Any] | None:
    """The broker's own view of itself, or ``None`` when it is not running."""
    try:
        return request("status", base)
    except SecretStoreError:
        return None


def ensure_broker(base: Path | None = None, *, timeout: float = STARTUP_TIMEOUT_S) -> bool:
    """Start a broker if none is listening. Returns True when one is up.

    **The lock is non-blocking, deliberately (#401).** Ten sessions may reach
    this at once and exactly one must spawn a daemon. A blocking ``flock``
    would do that too — and is precisely the construct that froze the TUI in
    #401, because this runs on the session's thread. So the winner takes
    ``LOCK_EX | LOCK_NB``; every loser skips the spawn entirely and polls for
    the winner's socket, bounded by ``timeout``. No caller ever waits on the
    lock itself, so a holder that wedges costs a bounded delay rather than a
    deadlock.
    """
    if is_running(base):
        return True

    from local_operator.secrets.keys import ensure_secrets_dir, secrets_dir

    ensure_secrets_dir(base)
    lock_file = lock_path(base)
    # The lock follows the socket into the short fallback directory when the
    # config path is too deep for sun_path; that directory must exist and be
    # ours before a lock file is opened inside it.
    if lock_file.parent != secrets_dir(base):
        ensure_runtime_dir(lock_file.parent)
    deadline = time.monotonic() + timeout
    handle = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Someone else is starting one. Poll for their socket instead of
            # waiting on the lock.
            while time.monotonic() < deadline:
                if is_running(base):
                    return True
                time.sleep(_LOCK_POLL_S)
            return False
        # Won the race. Re-check: the previous holder may have started one
        # between our ping and our lock acquisition.
        if is_running(base):
            return True
        _spawn_broker(base)
        while time.monotonic() < deadline:
            if is_running(base):
                return True
            time.sleep(_LOCK_POLL_S)
        return False
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def _spawn_broker(base: Path | None) -> None:
    """Launch a detached broker process.

    ``start_new_session`` detaches it from this process group so the daemon
    does not die with the session that happened to start it, and does not take
    the session's SIGINT. Its streams go to ``devnull``: a daemon inheriting
    the TUI's stdout would paint over the interface, and inheriting its stdin
    would let it steal keystrokes.

    Note the asymmetry with the ancestry rule this feature is built on: the
    broker being detached is correct (it must outlive its starter), while a
    CLIENT being detached is exactly the attacker shape spike 9 denies. They
    are different processes with different requirements.
    """
    environment = os.environ.copy()
    if base is not None:
        from local_operator.paths import CONFIG_DIR_ENV

        environment[CONFIG_DIR_ENV] = str(base)
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "wb") as devnull_out:
        subprocess.Popen(
            [sys.executable, "-m", "local_operator.secrets.brokerd"],
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True,
            env=environment,
            close_fds=True,
        )


def fetch_master_key(base: Path | None = None) -> bytes:
    """Ask the broker for the master key, for a caller it has authorized."""
    reply = request("key", base)
    key = reply.get("key")
    if not isinstance(key, str):
        raise SecretStoreError("the broker returned no key material")
    return decode_bytes(key)


def retrieve(name: str, base: Path | None = None) -> bytes:
    """Fetch one secret's value through the broker.

    Routed through the broker rather than decrypted locally so the §6
    notification fires: the broker tells the owning session to redact this
    value and waits for its ack before this call returns. That ordering is why
    a value fetched inside ``$( )`` cannot reach the transcript ahead of the
    filter that scrubs it.
    """
    reply = request("retrieve", base, name=name)
    value = reply.get("value")
    if not isinstance(value, str):
        raise SecretStoreError("the broker returned no value")
    return decode_bytes(value)


def unlock(passphrase: str, base: Path | None = None) -> None:
    """Unwrap a hardened store's key into the broker's memory for this boot."""
    request("unlock", base, passphrase=passphrase)


def register_session(
    base: Path | None = None, *, session_id: str | None = None
) -> socket.socket | None:
    """Register this process as a live lop session; returns the channel.

    The returned socket is the session's notification channel and MUST be kept
    open: closing it deregisters the session, at which point its descendants
    stop being authorized. The caller reads :func:`read_notification` from it.

    Returns ``None`` when no broker could be started, so a session boots
    normally without a store rather than failing — the store is an optional
    capability, not a boot dependency (§13).
    """
    if not ensure_broker(base):
        return None
    try:
        connection = _connect(socket_path(base))
    except SecretStoreError:
        return None
    try:
        send_frame(
            connection,
            {
                "version": PROTOCOL_VERSION,
                "op": "register",
                "session_id": session_id,
            },
        )
        reply = recv_frame(connection)
    except (OSError, ProtocolError, socket.timeout):
        connection.close()
        return None
    if not reply.get("ok"):
        connection.close()
        return None
    # No timeout on the channel afterwards: it is long-lived and mostly idle,
    # and a timeout here would tear down a healthy session's registration.
    connection.settimeout(None)
    return connection


def read_notification(connection: socket.socket) -> tuple[str, bytes] | None:
    """Block for one ``retrieved`` notice; ``None`` when the channel closes.

    Deliberately does NOT acknowledge. The broker is blocked on that ack and
    will not reply to the retrieving child until it arrives, so the caller must
    register the value with its redactor FIRST and only then call
    :func:`acknowledge`. Ack-on-read would return the ordering to a race — the
    guarantee §6 asks for is only real when the ack means "I am already
    scrubbing this", not "I have heard of it".
    """
    try:
        frame = recv_frame(connection)
    except (OSError, ProtocolError):
        return None
    if frame.get("event") != "retrieved":
        return None
    name = str(frame.get("name") or "")
    value = decode_bytes(str(frame.get("value") or ""))
    return name, value


def acknowledge(connection: socket.socket) -> None:
    """Tell the broker the value is registered and it may reply to the child."""
    try:
        send_frame(connection, {"ack": True})
    except (OSError, ProtocolError):  # pragma: no cover - session tearing down
        pass


__all__ = [
    "BrokerDenied",
    "BrokerLocked",
    "BrokerUnavailable",
    "acknowledge",
    "broker_status",
    "ensure_broker",
    "fetch_master_key",
    "is_running",
    "read_notification",
    "register_session",
    "retrieve",
    "unlock",
]
