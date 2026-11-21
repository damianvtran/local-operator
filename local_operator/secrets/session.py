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
    """THE one place this module opens a registration, first or re-registration.

    Both :func:`register_session` and :meth:`SessionRegistration._reregister`
    go through here, so anything that must accompany EVERY registration (e.g.
    the live-session lineage marker the fail-closed access PR #4388 publishes
    from ``register_session``) has a single home and a re-registration after a
    broker roll can never skip it.
    """
    from local_operator.secrets import client

    return client.register_session_channel(base, session_id=session_id, ensure=ensure)


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
        channel, on_secret, base=base, session_id=session_id, broker_pid=broker_pid
    )


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
