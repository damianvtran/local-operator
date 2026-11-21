"""Registering a live lop session with the broker (design §2.1, §6).

**Why this module exists at all.** The broker authorizes a caller by asking
whether it descends from a REGISTERED session, and until this landed nothing in
shipping code ever registered one — only tests did (QA finding Q2). The
consequences were not cosmetic: with an empty session table the broker denied
everyone, so `lop secret harden` produced a store whose own `unlock` was
refused by the ancestry gate and whose secrets no shipped command could reach.
Authenticating `register` (review R1) without also wiring up the legitimate
registrant would have converted that bypass into a permanent lockout, so the
two land together.

**What registration does and does not buy.** Holding a registration means the
session's descendants — an agent's `bash`, the eval worker — are authorized,
and it is what makes the §6 redaction notice deliverable. It does not make the
session process itself more trusted than the tier allows: in `keyfile` mode the
master key is on disk beside the ticket, so this is observability rather than
access control, and only in `passphrase` mode is the boundary load-bearing.
§8's table states that per tier.

**The channel is the registration.** The broker deregisters a session when its
socket closes, so this connection has to stay open for the session's whole
life; closing it is how a dead session promptly stops authorizing anything.
That is also why the reader thread here is not optional: the broker now WAITS
for an ack before serving a descendant's retrieval and denies it when none
comes (review R3), so a session that registered but never answered would break
every `$(lop secret get …)` its own agent ran.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SessionRegistration:
    """A live registration plus the thread that answers redaction notices.

    Constructed through :func:`register_session`, which returns ``None`` when
    no broker is reachable — the store is an optional capability and must never
    be a boot dependency (§13).
    """

    def __init__(
        self,
        channel: socket.socket,
        on_secret: Callable[[str, bytes], None],
        *,
        lineage_marker: Path | None = None,
    ) -> None:
        self._channel = channel
        self._on_secret = on_secret
        #: This registration's entry in :func:`lineage_dir`; withdrawn on
        #: :meth:`close`. ``None`` when it could not be published.
        self._lineage_marker = lineage_marker
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="secret-session-notices",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        """Register each notified value with the redactor, THEN acknowledge.

        The ordering is the whole point (§6): the ack means "I am already
        scrubbing this", not "I have heard of it". The broker holds the
        child's reply until this returns, so a value cannot reach the child —
        and therefore cannot reach the transcript — before the filter knows it.
        """
        from local_operator.secrets import client

        while not self._stopping.is_set():
            try:
                notice = client.read_notification(self._channel)
            except OSError:
                return
            if notice is None:
                return
            name, value = notice
            try:
                self._on_secret(name, value)
            except Exception:  # noqa: BLE001 — a redactor fault must not wedge the broker
                # Deliberately NOT acked: the broker fails closed on a missing
                # ack, so the child is denied rather than served a value this
                # session could not arrange to scrub. Continuing to the next
                # notice keeps one bad value from killing the channel.
                logger.warning("could not register secret %r for redaction", name, exc_info=True)
                continue
            try:
                client.acknowledge(self._channel)
            except OSError:
                return

    def close(self) -> None:
        """Deregister this session and stop answering notices.

        **``shutdown`` before ``close``, and that ordering is the whole of this
        method's correctness.** The reader thread is parked in ``recv`` on this
        socket, and ``close()`` alone does not send the peer its EOF while that
        read holds the file description — measured on CI (Linux): the broker's
        session table still held the pid for the whole 5 s the new guards polled
        for, twice, while the same code deregistered in under a second on
        macOS, whose ``close`` does revoke the descriptor. ``shutdown`` acts on
        the socket rather than the descriptor, so it both makes the broker's
        liveness peek see ``b""`` immediately and wakes the parked read; the
        ``close`` afterwards only releases the descriptor.
        """
        self._stopping.set()
        # Withdrawn FIRST: a session closing on purpose has stopped redacting,
        # so its descendants should get the out-of-session behaviour from now
        # on rather than a refusal that names a session which is going away.
        _withdraw_lineage_marker(self._lineage_marker)
        self._lineage_marker = None
        try:
            self._channel.shutdown(socket.SHUT_RDWR)
        except OSError:  # pragma: no cover - already gone
            pass
        try:
            self._channel.close()
        except OSError:  # pragma: no cover - already gone
            pass
        self._thread.join(timeout=2)


def register_session(
    on_secret: Callable[[str, bytes], None],
    *,
    session_id: str | None = None,
    base: Path | None = None,
) -> SessionRegistration | None:
    """Register this process as a live lop session, or return ``None``.

    ``None`` covers every "no store this boot" case — no broker could be
    started, the registration ticket is unreadable, the broker refused — and is
    never an error the caller has to handle: a session boots and runs normally
    without a secret store.

    **A REFUSAL is logged before returning ``None`` (review round 1, MAJOR-2).**
    "Nothing to register" and "the broker said no" both end here, but only the
    second is a limit the operator should be able to see: in the hardened tier
    a registrant with no lineage — a daemon-spawned runtime, §13 — is refused,
    and that refusal used to be silent at every level because nothing raised.
    """
    try:
        from local_operator.secrets import client

        channel = client.register_session(base, session_id=session_id)
    except Exception:  # noqa: BLE001 — the store is optional; boot must not fail on it
        logger.debug("secret broker registration failed", exc_info=True)
        return None
    if channel is None:
        return None
    return SessionRegistration(
        channel, on_secret, lineage_marker=_publish_lineage_marker(base, session_id)
    )


# --- lineage markers ---------------------------------------------------------
#
# **Why a marker on disk at all.** `access.retrieve_secret` must refuse the
# unnotified local decrypt for a descendant of a session whose redactor is
# registered, and the case that matters is the broker being GONE (crashed,
# killed, rolling) — so the broker's in-memory session table, the authority
# while it runs, is exactly what is unavailable when the question is asked. A
# freshly respawned daemon has an empty table and denies the descendant, which
# is indistinguishable on the wire from "no session ever registered". So the
# registering session leaves its own record, and the client answers the
# question the way the broker does: by walking its ancestry.
#
# **Why ancestry and not ``LOCAL_OPERATOR_SESSION_ID``.** That variable is set
# for every session's tool children whether or not the session ever reached a
# broker (a session booted before the store existed, or one whose registration
# was refused, has no redactor to notify), and it is attribution only. The
# marker is published only after the broker ACCEPTED the registration, so it
# means precisely "a redactor is listening for this lineage".
#
# **Identity, not pid.** Each name carries the process start time
# (``peer.ProcessIdentity``), so a session that died without closing leaves a
# marker that matches nothing once its pid is recycled; stale entries are
# pruned on the next publish. A random suffix keeps two registrations from one
# process (one runtime hosting several sessions) from withdrawing each other.
#
# **Not a security boundary, and it does not need to be.** In the keyfile tier
# a same-uid process can read the key file directly (§8); the marker only
# decides whether an ORDINARY retrieval may silently degrade. Deleting it
# restores today's fallback, which is no more than reading the key already
# allows. Forging one only refuses the forger.


def lineage_dir(base: Path | None = None) -> Path:
    """Where live registrations leave their lineage markers (inside the 0700 secrets dir)."""
    from local_operator.secrets.keys import secrets_dir

    return secrets_dir(base) / "live-sessions"


def _marker_key(pid: int, start_time: int) -> str:
    return f"{pid}-{start_time}"


def _publish_lineage_marker(base: Path | None, session_id: str | None) -> Path | None:
    """Record this process as a registered session. Best-effort: ``None`` on failure.

    A failed publish keeps today's behaviour for this session's descendants
    (the fallback), which is the §13 direction; it must never fail a boot.
    """
    import secrets as _random

    try:
        from local_operator.secrets.peer import process_info

        identity = process_info(os.getpid())
        if identity is None:
            return None
        directory = lineage_dir(base)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _prune_stale_markers(directory)
        path = (
            directory / f"{_marker_key(identity.pid, identity.start_time)}-{_random.token_hex(4)}"
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(session_id or "")
        return path
    except Exception:  # noqa: BLE001 — the store is optional; boot must not fail on it
        logger.debug("could not publish the secret session lineage marker", exc_info=True)
        return None


def _withdraw_lineage_marker(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _prune_stale_markers(directory: Path) -> None:
    """Drop markers whose process is gone (a session that crashed without closing)."""
    from local_operator.secrets.peer import process_info

    for entry in directory.iterdir():
        pid_text, _, rest = entry.name.partition("-")
        start_text = rest.partition("-")[0]
        if not (pid_text.isdigit() and start_text.isdigit()):
            continue
        identity = process_info(int(pid_text))
        if identity is None or identity.start_time != int(start_text):
            _withdraw_lineage_marker(entry)


def registered_session_ancestor(base: Path | None = None) -> tuple[int, str] | None:
    """The registered session this process descends from, as ``(pid, session_id)``.

    Walks from the PARENT, not from this process: a process that hosts a
    session itself (the runtime, whose own in-process fetch the broker serves
    without a notice) is not a tool child of it, and refusing its fallback would
    break the host's own reads after a broker crash for no transcript benefit.
    ``None`` when no ancestor holds a live registration, or the directory is
    absent — the overwhelmingly common case outside a session, answered by one
    ``stat``.
    """
    directory = lineage_dir(base)
    try:
        names = [entry.name for entry in directory.iterdir()]
    except OSError:
        return None
    if not names:
        return None
    markers: dict[str, str] = {}
    for name in names:
        pid_text, _, rest = name.partition("-")
        markers.setdefault(f"{pid_text}-{rest.partition('-')[0]}", name)
    from local_operator.secrets.peer import _walk, parent_pid

    parent = parent_pid(os.getpid())
    if parent is None:
        return None
    for ancestor in _walk(parent):
        name = markers.get(_marker_key(ancestor.pid, ancestor.start_time))
        if name is None:
            continue
        try:
            session_id = (directory / name).read_text(encoding="utf-8").strip()
        except OSError:
            session_id = ""
        return ancestor.pid, session_id
    return None


def register_variable_store_session(
    session: Any,
    *,
    session_id: str | None = None,
    base: Path | None = None,
) -> SessionRegistration | None:
    """Register the process that HOLDS ``session``'s ``VariableStore`` (§6).

    **Why the store's process and not the front end a human is typing into.**
    The §6 notice asks the notified session to add a child's
    ``$(lop secret get X)`` value to the filter that scrubs its own streamed
    output, and to the value set the bash/eval redactors re-read per chunk.
    Both live in the ``VariableStore`` of the process that RUNS the session
    (``Session.variables``). In the attached architecture that process is the
    session runtime, and the TUI holds only an ``AttachedSession`` facade —
    which has no store at all — so a registration made by the viewer could
    only refuse, and the broker (correctly) denied the descendant. Registering
    from the runtime's handle is what makes the notice land on the process that
    can actually honour it.

    The sink resolves the store per notice rather than capturing it, and RAISES
    when there is none: ``SessionRegistration._read_loop`` deliberately does not
    ack a failed redaction, so the broker fails closed and denies the child
    instead of serving a value nothing can scrub. That is the required
    direction — do not soften it into a log-and-continue.

    ``None`` on the same terms as :func:`register_session` — no broker, no
    readable ticket, refused — plus one more: a runtime that boots before a
    secret store exists has nothing any descendant could retrieve, and skipping
    the registration there keeps a session that never touches the store from
    starting a broker daemon of its own. A store created later is picked up on
    the next session boot.

    **The base is the caller's, on purpose (MINOR-3, NIT-2).** ``base`` is the
    root this session was built from, passed by the caller
    (``ServingSessionHandle``), never re-read from ``config_dir()`` here: the
    store-existence check below and the registration itself must agree about
    WHICH store this session owns, or a handle rooted at a different store
    could check one and register against another.

    **A session with no ``variables`` still registers, and that is not an
    oversight (NIT-2).** Skipping it would leave a descendant with no
    registered ancestor at all, and the broker's ancestry refusal is not the end
    of that path: ``access.retrieve_secret`` catches ``BrokerDenied`` and, in
    the keyfile tier, falls through to an UNNOTIFIED local decrypt, so the
    child is SERVED the value with nothing registered to scrub it — the exact
    §6 leak. Registering anyway keeps the broker's fail-closed denial, and the
    sink raises per notice so the child is denied rather than served raw. That
    deny path is pinned by
    ``tests/unit/secrets/test_runtime_session_registration.py``.
    """
    from local_operator.secrets.keys import store_path

    if not store_path(base).is_file():
        return None

    def on_secret(_name: str, value: bytes) -> None:
        store = getattr(session, "variables", None)
        if store is None:
            raise RuntimeError("the session has no variable store to redact through")
        store.register_redaction(value.decode("utf-8", errors="replace"))

    return register_session(on_secret, session_id=session_id, base=base)
