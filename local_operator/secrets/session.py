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

#: Back-off before each re-registration attempt after the channel closes: three
#: attempts within ~2 s. A roll renames the successor into place BEFORE the old
#: broker is told to stop, so the first attempt normally finds it; the bound
#: keeps a broker that was stopped on purpose from being polled for long, and
#: stays well inside the successor's ``broker.ROLL_GRACE_S`` hold.
_REREGISTER_DELAYS_S = (0.2, 0.6, 1.2)


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
        base: Path | None = None,
        session_id: str | None = None,
        broker_pid: int | None = None,
        lineage_marker: Path | None = None,
    ) -> None:
        """``broker_pid`` enables re-registration; ``None`` keeps the old one-shot life.

        Only :func:`register_session` knows which broker answered, so only a
        registration made there re-registers; a channel handed in by a caller
        that cannot say which broker it belongs to keeps today's behaviour.
        """
        self._channel = channel
        self._on_secret = on_secret
        self._base = base
        self._session_id = session_id
        self._broker_pid = broker_pid
        #: Guards the channel SWAP against ``close()``: a re-registration that
        #: completes after close began must not leave a live channel behind.
        self._swap_lock = threading.Lock()
        #: This registration's entry in :func:`lineage_dir`; withdrawn on
        #: :meth:`close`. ``None`` when it could not be published. Published
        #: ONCE per session by :func:`register_session` and kept across every
        #: re-registration: the session (and so its redactor) is unchanged by
        #: a broker roll, only its channel moves.
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
                notice = None
            if notice is None:
                # The channel closed. That used to end the registration for
                # the session's whole life: no reconnect, so after any broker
                # restart this session's descendants were denied and, in the
                # keyfile tier, served by the UNNOTIFIED local decrypt. A roll
                # (`secrets.roll`) now relies on this re-registering.
                if self._stopping.is_set() or not self._reregister():
                    return
                continue
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

    def _reregister(self) -> bool:
        """Register afresh with a SUCCESSOR broker, if one took over. True on success.

        **A fresh registration, never a resumed one (security R2).** The new
        broker authenticates this process from the kernel's peer credentials
        and the ticket exactly as on first boot; nothing carries over from the
        old channel.

        **Never starts a broker** (``ensure=False``): a channel closed by
        ``lop secret broker stop`` must stay closed, so no answer means give up.
        **Never fights a revocation**: if the SAME broker pid still answers, it
        closed this channel on purpose (a granting terminal died, §Q8) and
        re-registering would undo that decision.

        Only reached after the old channel reported EOF, i.e. after the old
        broker finished every in-flight notify/ack it had (its ``stop()`` joins
        request workers before closing session channels), so the ack ordering
        of :meth:`_read_loop` is never cut mid-notice (security R3).
        """
        if self._broker_pid is None:
            return False
        from local_operator.secrets import client
        from local_operator.secrets.errors import SecretStoreError

        for delay in _REREGISTER_DELAYS_S:
            if self._stopping.wait(delay):
                return False
            try:
                answering = client.request("ping", self._base).get("pid")
            except SecretStoreError:
                continue
            if answering == self._broker_pid:
                return False
            registered = _open_registration(self._base, self._session_id, ensure=False)
            if registered is None:
                return False
            channel, pid = registered
            with self._swap_lock:
                if self._stopping.is_set():
                    channel.close()
                    return False
                old, self._channel = self._channel, channel
                self._broker_pid = pid if pid is not None else answering
            try:
                old.close()
            except OSError:  # pragma: no cover - already gone
                pass
            logger.info("re-registered this session with secret broker pid %s", self._broker_pid)
            return True
        return False

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
        with self._swap_lock:
            self._stopping.set()
            channel = self._channel
        # Withdrawn FIRST: a session closing on purpose has stopped redacting,
        # so its descendants should get the out-of-session behaviour from now
        # on rather than a refusal that names a session which is going away.
        _withdraw_lineage_marker(self._lineage_marker)
        self._lineage_marker = None
        try:
            channel.shutdown(socket.SHUT_RDWR)
        except OSError:  # pragma: no cover - already gone
            pass
        try:
            channel.close()
        except OSError:  # pragma: no cover - already gone
            pass
        self._thread.join(timeout=2)


def _open_registration(
    base: Path | None, session_id: str | None, *, ensure: bool
) -> tuple[socket.socket, int | None] | None:
    """THE one place this module opens a broker channel, first or re-registration.

    Both :func:`register_session` and :meth:`SessionRegistration._reregister`
    go through here, so the wire contract of a registration (ticket, declared
    ``reroll`` capability) is identical either way. Process-level side effects
    of BECOMING a session — the live-session lineage marker
    (:func:`_publish_lineage_marker`, which the fail-closed read path in
    ``access`` consults) — belong in :func:`register_session` and deliberately
    NOT here: a re-registration is the same session moving to a new broker, and
    the marker published at boot stays valid across it (it is keyed on this
    process, not on the broker), so republishing per roll would only leave
    untracked markers that :meth:`SessionRegistration.close` never withdraws.

    Only this module's registrations advertise ``reroll``, because only
    :class:`SessionRegistration` actually re-registers (security S2).
    """
    from local_operator.secrets import client

    return client.register_session_channel(
        base, session_id=session_id, ensure=ensure, caps=client.REGISTRATION_CAPS
    )


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

        registered = _open_registration(base, session_id, ensure=True)
    except Exception:  # noqa: BLE001 — the store is optional; boot must not fail on it
        logger.debug("secret broker registration failed", exc_info=True)
        return None
    if registered is None:
        return None
    channel, broker_pid = registered
    if broker_pid is None:
        # A broker too old to name itself on the register reply: ask once, so a
        # later roll can still be told apart from a revocation.
        try:
            broker_pid = client.request("ping", base).get("pid")
        except Exception:  # noqa: BLE001 — re-registration is best effort
            broker_pid = None
    return SessionRegistration(
        channel,
        on_secret,
        base=base,
        session_id=session_id,
        broker_pid=broker_pid,
        lineage_marker=_publish_lineage_marker(base, session_id),
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
            logger.warning(_MARKER_LOST_WARNING, "this process's identity could not be read")
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
    except Exception as exc:  # noqa: BLE001 — the store is optional; boot must not fail on it
        # WARNING, not debug (#4388 R4/S3): without a marker this session's
        # tool children silently regain the unnotified keyfile fallback after
        # a broker crash, i.e. lose the fail-closed property. Boot still goes
        # on (§13); the operator just gets to see what was lost.
        logger.warning(_MARKER_LOST_WARNING, exc, exc_info=True)
        return None


#: Logged once per registration whose lineage marker could not be published.
_MARKER_LOST_WARNING = (
    "could not publish this session's secret lineage marker (%s); if the secret "
    "broker stops mid-session, this session's commands will read secrets without "
    "the redaction notice instead of being refused"
)


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
        if identity is not None:
            if identity.start_time != int(start_text):
                _withdraw_lineage_marker(entry)  # the pid was recycled: definitively stale
            continue
        # `None` is "could not identify", not "gone" (#4388 S3): a transient
        # lookup failure used to delete ANOTHER live session's marker, silently
        # reopening its children's fallback. Prune only on proof of absence.
        if _pid_definitively_gone(int(pid_text)):
            _withdraw_lineage_marker(entry)


def _pid_definitively_gone(pid: int) -> bool:
    """True only when the kernel says no process has ``pid`` (``ESRCH``)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:  # EPERM: it exists, just not ours to signal
        return False
    return False


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
