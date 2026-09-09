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
from typing import Callable

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
    ) -> None:
        self._channel = channel
        self._on_secret = on_secret
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
        """Deregister this session and stop answering notices."""
        self._stopping.set()
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
    """
    try:
        from local_operator.secrets import client

        channel = client.register_session(base, session_id=session_id)
    except Exception:  # noqa: BLE001 — the store is optional; boot must not fail on it
        logger.debug("secret broker registration failed", exc_info=True)
        return None
    if channel is None:
        return None
    return SessionRegistration(channel, on_secret)
