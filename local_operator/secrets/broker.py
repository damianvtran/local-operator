"""The secret broker daemon: holds the master key in RAM, serves it to kin.

Design reference: ``docs/design/secret-store.md`` §2.1, §2.3, §6, §12, §13.

**What the daemon is for, precisely, because it differs by tier.** In the
default ``keyfile`` tier the master key is on disk and a broker adds no
at-rest secrecy — a process that reads the key file still wins, exactly as
§8's table says. What it adds there is the §6 redaction ordering and an audit
trail that records the peer. In the opt-in ``passphrase`` tier the key exists
on disk only scrypt-wrapped, the unwrapped copy lives solely in this process's
memory, and that memory is protected by the one real kernel boundary available
(``task_for_pid`` denied, ``lldb`` blocked behind a ``SecurityAgent`` prompt —
spikes 3 and 5). Only in that tier does the broker convert "a script reads the
key file" into "a script must impersonate a lop session or raise a password
dialog on the operator's screen". Both claims are stated in the CLI's
``status`` output; neither may be widened.

**Threads, not asyncio, and why that is not the #401 hazard.** ``AGENTS.md``
records a blocking ``flock`` deadlocking the session's event loop. That lesson
binds the CLIENT, which runs inside a session process and therefore uses hard
timeouts and a non-blocking lock (see :mod:`local_operator.secrets.client`).
This daemon is a SEPARATE process with no event loop to block, so a thread per
connection is the simple correct shape. The rule preserved here is the one
that generalises: no lock is ever held across an operation that can block for
an unbounded time. The registry lock guards dictionary mutation only and is
never held while doing socket I/O.

**In-flight requests survive shutdown.** ``stop()`` closes the listening socket
so no new work is accepted, then joins the worker threads, so a retrieval that
has already been authorized completes rather than being severed halfway — a
severed retrieval is the case that would leave a caller unable to tell "denied"
from "died", which §13 requires never to be ambiguous.
"""

from __future__ import annotations

import errno
import hmac
import os
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.peer import (
    PeerAuthenticationUnavailable,
    ProcessIdentity,
    authorize_by,
    parent_pid,
    peer_identity,
    process_info,
)
from local_operator.secrets.protocol import (
    MAX_SOCKET_PATH,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_bytes,
    encode_bytes,
    ensure_runtime_dir,
    lock_path,
    recv_frame,
    send_frame,
    socket_path,
)

#: How long the broker waits for a session to acknowledge a redaction notice
#: before serving the value anyway. See :meth:`SecretBroker._notify_session`
#: for why serving anyway is the correct failure mode rather than denying.
NOTIFY_ACK_TIMEOUT_S = 2.0

#: How long a connection may sit without sending a complete request. Bounds a
#: peer that connects and then stalls — accidentally (a wedged agent) or
#: deliberately — from occupying a worker thread forever.
REQUEST_TIMEOUT_S = 10.0

#: Idle shutdown. The broker exits when no session has been registered and
#: nothing has asked it for anything for this long, so a machine that ran one
#: ``lop secret get`` an hour ago is not left with a daemon holding a key.
#: Deliberately NOT applied while any session is registered: in passphrase mode
#: an exit costs the operator a re-unlock, and §2.3 promises once per boot.
IDLE_SHUTDOWN_S = 30 * 60.0

#: Ceiling on concurrently served connections. The real workload is ~10
#: sessions (§4); this bounds a runaway or hostile caller from spawning
#: unbounded threads while leaving generous headroom.
MAX_WORKERS = 64

#: Ceiling on registered sessions, budgeted SEPARATELY from ``MAX_WORKERS``
#: (review R4). A session watcher is long-lived while a request worker is
#: short-lived, so drawing both from one pool let 64 registrations starve every
#: legitimate caller — measured, and a denial of the operator's entire
#: credential set in passphrase mode. The operator runs ~10 sessions (§4); 32
#: is generous headroom for that and still far below the worker budget, so a
#: full session table can never consume the capacity requests need.
MAX_SESSIONS = 32

#: Unlock backoff after a wrong passphrase: first delay, and the ceiling.
#: `unlock` is reachable without ancestry (it authenticates by passphrase), so
#: it is an online oracle unless a wrong guess costs increasing time (R6).
UNLOCK_BACKOFF_BASE_S = 0.25
UNLOCK_BACKOFF_MAX_S = 8.0


class BrokerError(SecretStoreError):
    """The broker refused a request, with a reason meant for the operator."""


@dataclass
class _Session:
    """A registered live lop session and the channel back to it.

    ``identity`` is pinned at registration and never re-resolved: that is what
    makes a dead session's descendants stop being authorized even if the pid
    comes back around on a different process.
    """

    identity: ProcessIdentity
    connection: socket.socket
    session_id: str | None
    #: The unlocked TERMINAL this session's standing rests on, when it had no
    #: standing of its own. ``None`` for a session that registered under the
    #: keyfile tier's ticket-only rule or that descended from another
    #: registered session. A session holding this is revoked when that terminal
    #: dies, which is what keeps the unlock grant bounded by the terminal's
    #: lifetime rather than the broker's (QA Q8).
    granting_terminal: int | None = None
    #: Serialises writes to this session's connection. Two concurrent
    #: retrievals attributed to the same session would otherwise interleave
    #: their notify frames on one stream and each read the other's ack.
    write_lock: threading.Lock = field(default_factory=threading.Lock)


class SecretBroker:
    """The daemon. Construct, :meth:`start`, and :meth:`stop` when done.

    ``key_provider`` returns the master key and is called lazily so a locked
    passphrase-mode broker can start, listen, and answer ``status`` and
    ``unlock`` without holding a key at all.
    """

    def __init__(
        self,
        base: Path | None = None,
        *,
        key_provider: Callable[[], bytes] | None = None,
        idle_shutdown_s: float = IDLE_SHUTDOWN_S,
        notify_ack_timeout_s: float = NOTIFY_ACK_TIMEOUT_S,
    ) -> None:
        self._base = base
        self._path = socket_path(base)
        self._key_provider = key_provider
        self._idle_shutdown_s = idle_shutdown_s
        self._notify_ack_timeout_s = notify_ack_timeout_s

        self._key: bytes | None = None
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._workers: set[threading.Thread] = set()
        #: Long-lived session watchers, budgeted apart from ``_workers`` (R4).
        #: One per registered session for that session's whole life, so
        #: counting them against the request budget let registrations starve
        #: retrievals; they are bounded by ``MAX_SESSIONS`` instead.
        self._watchers: set[threading.Thread] = set()
        self._stopping = threading.Event()

        #: Guards ``_sessions``, ``_workers`` and ``_watchers`` only. Never
        #: held across I/O.
        self._lock = threading.Lock()
        self._sessions: dict[int, _Session] = {}
        self._last_activity = time.monotonic()
        #: Inode of the socket THIS broker bound, so shutdown never unlinks a
        #: successor's socket at the same path. See :meth:`start`.
        self._bound_inode: int | None = None
        #: Consecutive wrong passphrases, reset by a successful unlock (R6).
        self._unlock_failures = 0
        #: Terminals that proved knowledge of the passphrase by unlocking this
        #: broker, pinned by identity. Authorizing ancestors for this boot in
        #: the hardened tier only — see :meth:`_grant_terminal`.
        self._terminals: dict[int, ProcessIdentity] = {}

    # --- lifecycle ----------------------------------------------------------

    @property
    def path(self) -> Path:
        """The socket this broker listens on."""
        return self._path

    def start(self) -> None:
        """Bind, secure and begin accepting. Raises :class:`BrokerError`.

        The socket is created inside the 0700 secrets directory and chmod'd
        0600 immediately. Both halves matter and macOS enforces them at
        ``connect()`` (spike 7), so this is a genuine barrier against other
        uids rather than advisory metadata.
        """
        from local_operator.secrets.keys import ensure_secrets_dir, secrets_dir

        ensure_secrets_dir(self._base)
        # socket_path() already relocates to a short private directory when the
        # natural path would exceed sun_path; that directory has to exist and be
        # ours before anything binds inside it.
        if self._path.parent != secrets_dir(self._base):
            ensure_runtime_dir(self._path.parent)
        if len(str(self._path)) > MAX_SOCKET_PATH:
            # Unreachable via socket_path(), but a caller may pass a path in.
            # bind() would otherwise fail with a bare EINVAL, which reads as a
            # mystery rather than as "that path is too long".
            raise BrokerError(
                f"the broker socket path is {len(str(self._path))} bytes, over the "
                f"{MAX_SOCKET_PATH}-byte limit a unix socket allows: {self._path}"
            )
        self._reap_stale_socket()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self._path))
        except OSError as exc:
            server.close()
            if exc.errno == errno.EADDRINUSE:
                raise BrokerError(f"another broker is already listening on {self._path}") from exc
            raise BrokerError(f"could not bind the broker socket {self._path}: {exc}") from exc
        os.chmod(self._path, 0o600)
        # Remember WHICH inode we bound. `stop()` unlinks by path, and a path
        # is not an identity: a broker that is still draining in-flight
        # requests while its replacement has already bound a fresh socket at
        # the same path would otherwise delete the SUCCESSOR's socket on its
        # way out, leaving a live daemon nothing can connect to. Found by
        # `broker restart` (QA Q4), which makes that overlap routine rather
        # than rare.
        try:
            self._bound_inode = self._path.stat().st_ino
        except OSError:  # pragma: no cover - the bind just succeeded
            self._bound_inode = None
        server.listen(MAX_WORKERS)
        # A timeout on accept() is what lets the accept loop notice _stopping
        # and the idle deadline; a blocking accept would only wake on a
        # connection, so a quiet broker could never shut itself down.
        server.settimeout(0.5)
        self._server = server
        self._stopping.clear()
        self._last_activity = time.monotonic()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="secret-broker-accept", daemon=True
        )
        self._accept_thread.start()

    def _reap_stale_socket(self) -> None:
        """Remove a socket file left behind by a broker that died.

        A crashed daemon leaves the inode; ``bind()`` then fails with
        ``EADDRINUSE`` forever and the operator has a store that cannot start.
        Probed by CONNECTING first: if something answers, a live broker owns
        it and this one must not unlink it out from under a running daemon.
        """
        if not self._path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(self._path))
        except OSError:
            # Nothing listening (ECONNREFUSED) or unreachable: the file is a
            # corpse and removing it is the only way forward.
            self._path.unlink(missing_ok=True)
        else:
            raise BrokerError(f"another broker is already listening on {self._path}")
        finally:
            probe.close()

    def _unlink_own_socket(self) -> None:
        """Remove the socket file only if it is still the one we bound.

        Compared by inode rather than by path: see :meth:`start`. A successor
        broker that has already rebound the path owns a DIFFERENT inode, and
        removing it would strand a live daemon behind a missing rendezvous
        point.
        """
        if self._bound_inode is not None:
            try:
                if self._path.stat().st_ino != self._bound_inode:
                    return  # a successor owns this path now; leave it alone
            except OSError:
                return  # already gone
        # One unlink, deliberately: the tree guard counts call sites of this
        # shape, and a single reviewable site is easier to keep honest than two.
        self._path.unlink(missing_ok=True)
        self._bound_inode = None

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting, let in-flight requests finish, then clean up."""
        self._stopping.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:  # pragma: no cover - close rarely fails
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=timeout)
            self._accept_thread = None
        with self._lock:
            workers = list(self._workers) + list(self._watchers)
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for worker in workers:
            worker.join(timeout=timeout)
        for session in sessions:
            try:
                session.connection.close()
            except OSError:  # pragma: no cover
                pass
        self._unlink_own_socket()
        # Drop the key reference on the way out. Python offers no guarantee the
        # bytes are scrubbed from the heap, so this is hygiene, not erasure,
        # and is not claimed as more than that.
        self._key = None

    def serve_forever(self) -> None:
        """Block until the broker stops (idle timeout, or :meth:`stop`)."""
        while not self._stopping.is_set():
            self._stopping.wait(0.5)
            if self._should_idle_out():
                break
        self.stop()

    def _should_idle_out(self) -> bool:
        if self._idle_shutdown_s <= 0:
            return False
        with self._lock:
            if self._sessions:
                return False
            idle_for = time.monotonic() - self._last_activity
        return idle_for >= self._idle_shutdown_s

    # --- key handling -------------------------------------------------------

    @property
    def locked(self) -> bool:
        """True when the broker holds no usable key (passphrase mode, pre-unlock)."""
        return self._key is None and self._key_provider is None

    def unlock_with_key(self, key: bytes) -> None:
        """Install an unwrapped master key into memory (used by ``unlock``)."""
        self._key = key

    def _master_key(self) -> bytes:
        """The key this store is CURRENTLY sealed under, never a stale cache.

        **The cache is validated against the database on every use, and that is
        the point of this method (sibling of QA Q10).** The broker holds the key
        in memory for a whole boot while ``rotate`` runs in a different process
        and re-seals the store under a new one. Nothing invalidated this copy,
        so after any rotation the broker kept serving the superseded key: the
        blind index is derived from the master key, so every lookup came back
        "No secret named X" \u2014 the store intact and reported empty. Measured
        directly against a live broker, and the CLI only escaped it because
        `access.py` falls back to reading the key file when the broker's answer
        does not work out.

        The check is a fingerprint comparison, not a re-read of the key: the
        fingerprint is a public column of the store readable WITHOUT a key
        (that is why it exists), so this costs one small SQLite read and never
        needs the plaintext key file \u2014 which in the hardened tier does not
        exist. On a mismatch the cached key is dropped and the tier decides
        what happens next: ``keyfile`` re-reads the new key from disk through
        the provider, while a hardened broker has no way to unwrap the new key
        and must say so, because the passphrase is not held here.
        """
        if self._key is not None and not self._key_opens_store(self._key):
            self._key = None
            if self._key_provider is None:
                raise BrokerError(
                    "the secret store's master key was rotated after this broker was "
                    "unlocked, so the key it holds no longer opens the store. Run "
                    "`lop secret unlock` again to unlock it with the new key."
                )
        if self._key is None:
            if self._key_provider is None:
                raise BrokerError(
                    "the secret store is hardened with a passphrase and this broker is "
                    "locked. Run `lop secret unlock` once to unlock it for this boot."
                )
            self._key = self._key_provider()
        return self._key

    def _key_opens_store(self, key: bytes) -> bool:
        """Does ``key`` match the fingerprint the store records for itself?

        ``True`` whenever the question cannot be answered — no store yet, no
        fingerprint row, an unreadable database — for the same "cannot tell, do
        not guess" reason :func:`recorded_key_fingerprint` returns ``None``
        there. Guessing the other way would make the broker refuse to serve a
        perfectly good key because the database was momentarily unreadable.
        """
        from local_operator.secrets.crypto import key_fingerprint
        from local_operator.secrets.store import recorded_key_fingerprint

        try:
            recorded = recorded_key_fingerprint(self._base)
        except OSError:  # pragma: no cover - unreadable dir; cannot tell
            return True
        return recorded is None or recorded == key_fingerprint(key)

    # --- accept / dispatch --------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stopping.is_set():
            server = self._server
            if server is None:
                return
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                # The listening socket was closed under us by stop().
                return
            with self._lock:
                if len(self._workers) >= MAX_WORKERS:
                    over_capacity = True
                else:
                    over_capacity = False
            if over_capacity:
                self._close_quietly(connection)
                continue
            worker = threading.Thread(
                target=self._serve_connection,
                args=(connection,),
                name="secret-broker-conn",
                daemon=True,
            )
            with self._lock:
                self._workers.add(worker)
            worker.start()

    @staticmethod
    def _close_quietly(connection: socket.socket) -> None:
        try:
            connection.close()
        except OSError:  # pragma: no cover
            pass

    def _serve_connection(self, connection: socket.socket) -> None:
        """Authenticate the peer, then serve it until it goes away."""
        keep_open = False
        try:
            connection.settimeout(REQUEST_TIMEOUT_S)
            try:
                identity = peer_identity(connection)
            except PeerAuthenticationUnavailable as exc:
                # Fail CLOSED and say why. A platform where the peer cannot be
                # identified serves nobody rather than serving everybody.
                self._reply_error(connection, "unauthenticated", str(exc))
                return
            keep_open = self._handle_requests(connection, identity)
        except (ProtocolError, OSError, socket.timeout):
            # A peer that hangs up, stalls or sends junk is ordinary. It is not
            # worth a traceback in a daemon with no console.
            pass
        finally:
            if not keep_open:
                self._close_quietly(connection)
            with self._lock:
                self._workers.discard(threading.current_thread())

    def _handle_requests(self, connection: socket.socket, identity: ProcessIdentity) -> bool:
        """Serve one request. Returns True when the connection must stay open.

        A ``register`` request converts this connection into the session's
        long-lived notification channel, which is why the thread hands
        ownership over rather than closing it.
        """
        request = recv_frame(connection)
        with self._lock:
            self._last_activity = time.monotonic()

        version = request.get("version")
        if version != PROTOCOL_VERSION:
            # **The pid rides on the refusal, and it is load-bearing (Q4).**
            # The advice is "restart the broker", and restarting means stopping
            # THIS daemon — but a version-skewed client cannot ask `status` for
            # the pid, because `status` fails this very gate first. Without the
            # pid here the only actionable message the operator can be given is
            # one they cannot act on, so it is answered as part of the refusal.
            self._reply_error(
                connection,
                "version",
                f"this broker speaks protocol {PROTOCOL_VERSION}, the caller speaks "
                f"{version!r}. Restart the broker after updating: lop secret broker restart",
                pid=os.getpid(),
                protocol=PROTOCOL_VERSION,
            )
            return False

        operation = request.get("op")
        if operation == "register":
            return self._handle_register(connection, identity, request)
        if operation == "ping":
            send_frame(connection, {"ok": True, "pid": os.getpid()})
            return False
        if operation == "status":
            self._handle_status(connection)
            return False
        if operation == "unlock":
            # **Before the ancestry gate, deliberately (QA Q2).** `unlock`
            # presents the passphrase, and the passphrase IS its
            # authentication — a stronger one than ancestry, since it is never
            # on disk in any form while the ticket and the key file are. Behind
            # the gate this verb was unreachable: unlocking required descending
            # from a registered session, and in a freshly-booted hardened store
            # there are no sessions yet, so `harden` bricked the store outright.
            # Ordering it here is what makes the passphrase tier enterable.
            self._handle_unlock(connection, identity, request)
            return False

        # **No ticket shortcut on data operations, in either tier.** A ticket
        # is a FILE, and the detached attacker reads files owned by this uid as
        # easily as the operator does, so accepting one in place of ancestry
        # here would serve exactly the script §8 promises to stop — measured,
        # as a regression, when an earlier revision of this fix did that. The
        # operator's own terminal is served instead by proving knowledge of the
        # passphrase (`_grant_terminal`) in the hardened tier, and by
        # `access.py`'s key-file fallback in the keyfile tier, where §8 already
        # concedes the broker is not the boundary.
        allowed, reason = self._authorize(identity)
        if not allowed:
            self._audit_denial(identity, operation, reason)
            self._reply_error(connection, "unauthorized", reason)
            return False

        if operation == "key":
            self._handle_key(connection, identity)
        elif operation == "retrieve":
            self._handle_retrieve(connection, identity, request)
        else:
            self._reply_error(connection, "protocol", f"unknown operation {operation!r}")
        return False

    def _authorize(self, identity: ProcessIdentity) -> tuple[bool, str]:
        """Ancestry check against the registered sessions."""
        allowed, reason, _ = self._authorize_by(identity)
        return allowed, reason

    def _authorize_by(self, identity: ProcessIdentity) -> tuple[bool, str, int | None]:
        """:meth:`_authorize`, also reporting WHICH entry admitted the caller.

        The registry is COPIED under the lock and the walk runs outside it: the
        walk makes syscalls per hop, and holding the lock across them would let
        one slow walk stall every other session's registration.
        """
        terminals = self._live_terminals()
        with self._lock:
            sessions = {pid: session.identity for pid, session in self._sessions.items()}
            # A terminal that unlocked this broker authorizes its descendants
            # exactly as a registered session does (see `_grant_terminal`).
            # Merged here rather than kept in a second walk so there is ONE
            # authorization path to reason about and audit.
            sessions.update(terminals)
        allowed, reason, by = authorize_by(identity, sessions)
        if not allowed or by is None:
            return allowed, reason, None
        return allowed, reason, self._grant_behind(by, terminals)

    def _grant_behind(self, authorizer: int, terminals: dict[int, ProcessIdentity]) -> int | None:
        """Which granted TERMINAL an authorization ultimately rests on, if any.

        **Standing is inherited, or the bound is one hop deep (QA Q8).** A
        squatter admitted by the unlocked terminal registers as a session; its
        child then registers as a *descendant of that session*, whose authorizer
        is the squatter rather than the terminal. Without this, the child's
        ``granting_terminal`` would be ``None`` and it would survive the sweep
        that removes its parent — the same escape one level down. So a session
        admitted by another session inherits whatever that session's standing
        rested on, and one sweep revokes the whole chain.

        ``None`` means the authorization stands on its own: a session that
        registered under the keyfile tier's ticket-only rule, or one whose chain
        reaches a session with no borrowed standing.
        """
        if authorizer in terminals:
            return authorizer
        with self._lock:
            session = self._sessions.get(authorizer)
        return session.granting_terminal if session is not None else None

    def _live_terminals(self) -> dict[int, ProcessIdentity]:
        """Granted terminals whose process is still the one that unlocked, evicting the rest.

        **Why entries are dropped rather than merely skipped (review R10).** A
        dead grant authorizes nobody either way, but :func:`authorize` STOPS the
        walk on a registered-pid/identity mismatch rather than continuing past
        it. So a stale entry whose pid was recycled onto an unrelated process
        that happens to sit in a legitimate caller's ancestry denies that caller
        with a confusing "pid was reused" message — it fails toward locking the
        operator out of their own credentials. Reaping here also bounds
        ``_terminals``, which was otherwise the one unbounded dict on the
        security path, asymmetric with the ``MAX_SESSIONS`` cap this PR added
        for R4.

        Called on the authorization path rather than on a timer: it is one
        ``process_info`` per grant (typically one), and a grant that just died
        must stop authorizing on the NEXT request, not at the next tick.
        """
        with self._lock:
            snapshot = dict(self._terminals)
        live = {
            pid: identity
            for pid, identity in snapshot.items()
            if (current := process_info(pid)) is not None and identity.same_process_as(current)
        }
        dead = set(snapshot) - set(live)
        if dead:
            with self._lock:
                for pid in dead:
                    # Only if it is still the same grant: an unlock between the
                    # snapshot and here may have re-granted this pid.
                    if self._terminals.get(pid) is snapshot[pid]:
                        del self._terminals[pid]
            # **The grant is bounded by the terminal's LIFETIME, and this is
            # what makes that true (QA Q8).** A process inside a granted
            # terminal may register itself as a session, and a registered
            # session is an independent authorizing entry: killing the terminal
            # left the self-registered squatter serving secrets to its own
            # descendants for the broker's whole life, while a control shell in
            # the same terminal was correctly denied. Measured, not theorised.
            # Standing that was borrowed from a terminal ends with it, so the
            # sessions admitted on that basis are revoked here.
            self._revoke_sessions_granted_by(dead)
        return live

    def _revoke_sessions_granted_by(self, terminals: set[int]) -> None:
        """Deregister sessions whose only standing was a now-dead granted terminal.

        Their channels are closed as well as dropped: closing is what the
        session's own client observes as revocation, and a session left holding
        an open channel it believes is live would keep receiving §6 notify
        frames for retrievals it is no longer entitled to see.
        """
        with self._lock:
            revoked = [
                session
                for session in self._sessions.values()
                if session.granting_terminal in terminals
            ]
            for session in revoked:
                del self._sessions[session.identity.pid]
        for session in revoked:
            self._audit(
                session.identity,
                "register",
                f"revoked:the terminal that authorized this session ({session.granting_terminal}) "
                "has exited",
            )
            self._close_quietly(session.connection)

    # --- operations ---------------------------------------------------------

    def _handle_register(
        self, connection: socket.socket, identity: ProcessIdentity, request: dict[str, Any]
    ) -> bool:
        """Register the CONNECTING process as a live session, if entitled.

        A session may only register ITSELF. The pid is taken from the kernel's
        view of the connection, never from the request body — a self-declared
        pid would let any process register an arbitrary victim pid and then
        authorize its own descendants through it.

        **Entitlement is a ticket, because no process-shape check is sound
        (review R1).** Registration used to be unauthenticated, and since
        :func:`~local_operator.secrets.peer._walk` yielded the peer as the first
        element of its own ancestry, any process that registered itself became
        its own authorizing ancestor — one frame from a detached script to the
        master key. The repair cannot be "check that the peer looks like a
        session": measured here, a same-uid attacker forges ppid 1, session
        leadership and even a controlling tty (``pty.fork``), and code
        signatures are unreadable (§2.1). So the registrant presents a secret
        from the 0700 secrets directory instead — see
        :func:`~local_operator.secrets.keys.registration_ticket` for why that is
        honest per tier rather than circular.

        Compared in constant time: the ticket is a fixed-length secret and a
        length-independent early-exit comparison would leak its bytes to a peer
        allowed to retry.
        """
        if not self._ticket_is_valid(request.get("ticket")):
            reason = "the caller did not present a valid session registration ticket"
            self._audit_denial(identity, "register", reason)
            self._reply_error(connection, "unauthorized", reason)
            return False
        # **In the hardened tier the ticket alone is not enough, and this is
        # the finding my own adversarial test caught.** A detached `setsid`
        # attacker reads the 0600 ticket exactly as it reads any file owned by
        # this uid, so a ticket-only gate let it register itself against an
        # UNLOCKED passphrase-tier broker and retrieve the canary — R1
        # reopened by a different door. Nothing on disk can separate that
        # attacker from the operator, so in this tier a registrant must also
        # descend from something that proved knowledge of the passphrase: a
        # granted terminal (`_grant_terminal`) or an already-registered
        # session. `keyfile` mode keeps the ticket-only path, where §8 already
        # concedes that a caller able to read the ticket could read the master
        # key beside it.
        granting_terminal: int | None = None
        if not self._is_keyfile_tier():
            allowed, why, granting_terminal = self._authorize_by(identity)
            if not allowed:
                reason = (
                    "this store is hardened, so registering a session also requires descending "
                    f"from an unlocked terminal or a registered session ({why})"
                )
                self._audit_denial(identity, "register", reason)
                self._reply_error(connection, "unauthorized", reason)
                return False

        session = _Session(
            identity=identity,
            connection=connection,
            session_id=str(request.get("session_id") or "") or None,
            # Recorded so this session dies with the terminal it borrowed its
            # standing from (QA Q8). `None` when the session had standing of
            # its own — it descended from another registered session, or the
            # keyfile tier admitted it on the ticket alone — in which case no
            # terminal's death should revoke it.
            granting_terminal=granting_terminal,
        )
        with self._lock:
            previous = self._sessions.get(identity.pid)
            # Re-registration by the same pid replaces rather than adds, so a
            # session that reconnects cannot consume a second slot.
            if previous is None and len(self._sessions) >= MAX_SESSIONS:
                over_capacity = True
            else:
                over_capacity = False
                self._sessions[identity.pid] = session
                self._last_activity = time.monotonic()
        if over_capacity:
            # R4: registrations used to be free AND unbounded, so 64 of them
            # exhausted the worker budget and wedged the broker for every
            # legitimate caller — a denial of the operator's whole credential
            # set. Refuse loudly and audit it instead of closing the connection
            # silently, so the cause is in the chain rather than invisible.
            reason = f"the broker already holds {MAX_SESSIONS} registered sessions"
            self._audit_denial(identity, "register", reason)
            self._reply_error(connection, "unavailable", reason)
            return False
        if previous is not None:
            self._close_quietly(previous.connection)
        self._audit(identity, "register", "allow")
        send_frame(connection, {"ok": True, "registered": identity.pid})

        watcher = threading.Thread(
            target=self._watch_session,
            args=(session,),
            name=f"secret-broker-session-{identity.pid}",
            daemon=True,
        )
        with self._lock:
            self._watchers.add(watcher)
        watcher.start()
        return True

    def _watch_session(self, session: _Session) -> None:
        """Deregister a session when its process dies or its socket closes.

        Two independent signals, because either alone has a hole: the socket
        closing covers a clean exit, and the liveness poll covers a session
        killed with SIGKILL whose socket lingers in the kernel. A session that
        is gone MUST stop authorizing its descendants promptly, which is the
        property §2.1 rests on.

        There is a THIRD way this loop ends and it is not an error: the broker
        itself may close the channel out from under the watcher when a grant is
        revoked (:meth:`_revoke_sessions_granted_by`). Every operation on the
        connection is therefore inside the `try`, including the `settimeout`
        that used to sit above it — a revocation racing this loop otherwise
        raised `EBADF` on a daemon thread, which surfaces as an unhandled
        thread exception rather than the orderly teardown it actually is.
        """
        try:
            while not self._stopping.is_set():
                current = process_info(session.identity.pid)
                if current is None or not current.same_process_as(session.identity):
                    break
                try:
                    session.connection.settimeout(1.0)
                    # The session sends nothing on this channel except acks,
                    # which _notify_session consumes under the write lock. A
                    # read here that returns b"" means the far end closed.
                    with session.write_lock:
                        session.connection.setblocking(False)
                        try:
                            if session.connection.recv(1, socket.MSG_PEEK) == b"":
                                break
                        except BlockingIOError:
                            pass
                        except OSError:
                            break
                        finally:
                            # Restoring blocking mode is best-effort for the
                            # same reason: a revoked channel is already closed
                            # and the restore is meaningless, not a failure.
                            with suppress(OSError):
                                session.connection.setblocking(True)
                except OSError:
                    break
                self._stopping.wait(1.0)
        finally:
            with self._lock:
                if self._sessions.get(session.identity.pid) is session:
                    del self._sessions[session.identity.pid]
            self._close_quietly(session.connection)
            with self._lock:
                self._watchers.discard(threading.current_thread())

    def _handle_status(self, connection: socket.socket) -> None:
        with self._lock:
            sessions = sorted(self._sessions)
        send_frame(
            connection,
            {
                "ok": True,
                "pid": os.getpid(),
                "locked": self._key is None and self._key_provider is None,
                "sessions": sessions,
                "protocol": PROTOCOL_VERSION,
            },
        )

    def _handle_key(self, connection: socket.socket, identity: ProcessIdentity) -> None:
        """Serve the master key to an authorized peer.

        This is the seam :mod:`local_operator.secrets.access` documents: the
        peer then talks to SQLite directly, so the database's concurrency story
        stays in WAL (§4, measured at 10 concurrent readers) instead of being
        re-implemented inside a single daemon.

        It also means an authorized peer holds the key, which is stated plainly
        rather than hidden: authorization is ancestry, so anything a lop session
        can spawn can obtain the key. The broker's boundary is against
        processes OUTSIDE that lineage, not within it (§9.1).
        """
        try:
            key = self._master_key()
        except BrokerError as exc:
            self._reply_error(connection, "locked", str(exc))
            return
        self._audit(identity, "key", "allow")
        send_frame(connection, {"ok": True, "key": encode_bytes(key)})

    def _ticket_is_valid(self, offered: Any) -> bool:
        """Does ``offered`` match this store's registration ticket?

        Compared with :func:`hmac.compare_digest` so a peer that may retry
        cannot walk the ticket out byte by byte off the timing of an early
        exit. A missing or unreadable ticket file denies rather than opening
        registration: failing closed is the only safe direction for the verb
        that grants standing to ask for the key.
        """
        from local_operator.secrets.keys import registration_ticket

        if not isinstance(offered, str) or not offered:
            return False
        try:
            expected = registration_ticket(self._base)
            candidate = decode_bytes(offered)
        except (SecretStoreError, OSError, ValueError):
            return False
        return hmac.compare_digest(candidate, expected)

    def _is_keyfile_tier(self) -> bool:
        """May a registrant be admitted on the ticket alone, with no lineage?

        The tier decides how much a ticket is worth. In ``keyfile`` mode the
        ticket sits beside ``master.key``, so a caller able to read one could
        read the other and decrypt the store with no broker at all — §8 records
        that this tier stops nobody at the socket. In ``passphrase`` mode the
        only unwrapped copy is in this process's memory, so a file secret
        proves nothing and lineage from the passphrase is required instead.

        **Asks :func:`~local_operator.secrets.keys.key_of_record_is_plaintext`,
        NOT :func:`~local_operator.secrets.keys.key_mode` (review R4-1).** This
        is an authorization decision, and `key_mode` answers on the mere
        presence of `master.key` — a path any process running as this uid can
        create. Consuming the display predicate here meant an attacker could
        write 32 random bytes to it, flip this gate to the ticket-only path,
        register, and be served the unwrapped master key out of this process's
        memory; measured 3/3 from a detached `setsid` process at ppid=1 against
        a genuinely hardened, unlocked store. The planted key was junk, so it
        was never key theft — it was a lie told to a predicate that only
        observed. The validating predicate checks the installed key against the
        fingerprint the database records, which an attacker cannot forge
        without the key it names.

        Fails closed (treats the store as hardened) if the answer cannot be
        established, because the hardened path is the one with the stricter
        check and a wrong denial in the keyfile tier is absorbed by
        `access.py`'s documented key-file fallback.
        """
        from local_operator.secrets.keys import key_of_record_is_plaintext

        try:
            return key_of_record_is_plaintext(self._base)
        except OSError:  # pragma: no cover - unreadable dir fails closed
            return False

    def _handle_unlock(
        self, connection: socket.socket, identity: ProcessIdentity, request: dict[str, Any]
    ) -> None:
        """Unwrap the passphrase-wrapped key and cache it for this boot.

        Reached BEFORE the ancestry gate (see :meth:`_dispatch`), so the
        passphrase is doing the authenticating here. Every attempt is audited
        and a wrong one costs an increasing delay: without that this is an
        online guessing oracle bounded only by scrypt's ~180 ms (review R6).
        The backoff is applied before the reply rather than by refusing, so a
        legitimate operator who mistypes is slowed rather than locked out of
        their own store — the failure mode that matters when the alternative is
        an unreachable credential set.
        """
        # The recovery-aware variant: it unwraps the key of record and, when
        # that key no longer opens the store, finishes a rotation of a hardened
        # store that was interrupted between its COMMIT and its install. This
        # is the only moment the passphrase exists, so it is the only place a
        # STAGED WRAPPED key can be tested — the keyfile tier's equivalent
        # repair runs unattended in `resolve_master_key`, which can fingerprint
        # plaintext staged keys without one.
        from local_operator.secrets.keys import unwrap_master_key_matching

        passphrase = request.get("passphrase")
        if not isinstance(passphrase, str) or not passphrase:
            self._reply_error(connection, "protocol", "unlock requires a passphrase")
            return
        with self._lock:
            failures = self._unlock_failures
        if failures:
            # Geometric, capped: 0.25s, 0.5s … 8s. Enough to make sustained
            # guessing pointless against scrypt's cost, short enough that a
            # human retry never looks hung.
            time.sleep(min(UNLOCK_BACKOFF_BASE_S * (2 ** (failures - 1)), UNLOCK_BACKOFF_MAX_S))
        try:
            key = unwrap_master_key_matching(self._base, passphrase)
        except SecretStoreError as exc:
            with self._lock:
                self._unlock_failures += 1
            self._audit_denial(identity, "unlock", "wrong passphrase")
            self._reply_error(connection, "passphrase", str(exc))
            return
        self._key = key
        with self._lock:
            self._unlock_failures = 0
        self._grant_terminal(identity)
        self._audit(identity, "unlock", "allow")
        send_frame(connection, {"ok": True})

    def _grant_terminal(self, identity: ProcessIdentity) -> None:
        """Let the terminal that just unlocked keep using the store this boot.

        **The problem this solves, and why nothing simpler works.** In
        ``passphrase`` mode the operator's own ``lop secret get`` is denied by
        construction (§13): typed at a shell prompt it has no lop session among
        its ancestors. Every disk-based credential fails to fix this, because
        the detached `setsid` attacker runs as the same uid and reads any file
        the operator can — measured: honouring the registration ticket here
        served that attacker the unwrapped key, which is R1 all over again.

        So the grant is seeded by the ONE secret that is never on disk in any
        form: the passphrase the operator just typed. Whoever proved knowledge
        of it gets their PARENT — the shell they typed it into — recorded as an
        authorizing ancestor for this boot, exactly like a registered session.
        A later ``lop secret get`` in that same terminal descends from that
        shell and is served; the detached attacker, reparented to launchd, does
        not descend from it and stays denied.

        The residual risk is stated rather than hidden: anything the operator
        subsequently runs *inside that terminal* can read secrets while the
        broker is unlocked. That is §9.1's acknowledged limit ("a script that
        runs `lop` itself") and §9.4's unlock window, not a new hole — and it
        is strictly narrower than the ancestry-free access this replaces.

        Pinned by identity like every other entry, so a shell that exits and
        whose pid is recycled authorizes nothing.
        """
        parent = parent_pid(identity.pid)
        if parent is None or parent <= 0:
            return
        terminal = process_info(parent)
        if terminal is None:
            return
        with self._lock:
            self._terminals[terminal.pid] = terminal

    def _handle_retrieve(
        self, connection: socket.socket, identity: ProcessIdentity, request: dict[str, Any]
    ) -> None:
        """Decrypt one secret and return it — AFTER the owning session acks.

        **This ordering is the §6 invariant and the reason retrieval goes
        through the broker at all.** A value fetched as ``$(lop secret get X)``
        never passes through the session process, so the session's output
        filters have nothing to match on and the secret would appear in the
        transcript verbatim. The broker closes that: it notifies the session
        that owns this peer and WAITS FOR THE ACK before replying to the child.
        Because the child cannot print a value it has not received, and it
        cannot receive one before the session has confirmed it is scrubbing,
        the registration strictly precedes any possible output.

        **It is an invariant because it fails closed (review R3).** It used to
        serve the value anyway after a 2 s timeout, which made the word
        "guarantee" false: a session that never acked — including one an
        attacker kept from acking — leaked the value into the transcript it was
        meant to be scrubbed from. A descendant that cannot be covered by the
        filter is now DENIED rather than served, because the whole reason this
        path exists is that nothing downstream can catch the value afterwards.
        The cost is bounded and honest: the caller gets a clear error naming the
        wedged session, not a hang.

        **The session retrieving for ITSELF is not subject to this and must not
        be.** That is §6 case (1), where the value is returned into the
        session's own process and it registers the redaction directly — there
        is nothing to notify, and demanding an ack from the very process
        blocked on this reply would deadlock the operator's terminal for a
        notice it does not need.
        """
        name = request.get("name")
        if not isinstance(name, str) or not name:
            self._reply_error(connection, "protocol", "retrieve requires a name")
            return
        try:
            key = self._master_key()
        except BrokerError as exc:
            self._reply_error(connection, "locked", str(exc))
            return

        from local_operator.secrets.store import SecretStore

        session = self._session_for(identity)
        try:
            value = SecretStore(key, base=self._base).get(
                name, session_id=session.session_id if session else None
            )
        except SecretStoreError as exc:
            # **The class name rides along (R11/Q5).** `"store"` alone erased
            # the difference between "no such secret" and "this record is
            # corrupt", and a caller that must distinguish them — the eval
            # `Mapping`, which owes a `KeyError` for a miss — could not. The
            # code stays for wire compatibility; `kind` is what carries the
            # taxonomy across the seam.
            self._reply_error(connection, "store", str(exc), kind=type(exc).__name__)
            return

        # Notify BEFORE replying, and fail closed when the notice is not
        # acknowledged (R3). `is_self` is the session asking for its own value:
        # §6 case (1), which needs no notice and cannot ack itself.
        is_self = session is not None and session.identity.same_process_as(identity)
        if session is not None and not is_self:
            if not self._notify_session(session, name=name, value=value):
                reason = (
                    f"session {session.identity.pid} did not acknowledge the redaction notice "
                    f"within {self._notify_ack_timeout_s:g}s, so this value cannot be kept out "
                    "of its transcript and was not served"
                )
                self._audit_denial(identity, "retrieve", reason)
                self._reply_error(connection, "unredactable", reason)
                return
        send_frame(connection, {"ok": True, "value": encode_bytes(value)})

    def _session_for(self, identity: ProcessIdentity) -> _Session | None:
        """The registered session this peer descends from, if any."""
        from local_operator.secrets.peer import _walk

        with self._lock:
            sessions = dict(self._sessions)
        for ancestor in _walk(identity.pid):
            session = sessions.get(ancestor.pid)
            if session is not None and session.identity.same_process_as(ancestor):
                return session
        return None

    def _notify_session(self, session: _Session, *, name: str, value: bytes) -> bool:
        """Tell the session to redact ``value``, and wait for its ack.

        Bounded by :data:`NOTIFY_ACK_TIMEOUT_S`. Returns whether the session
        acknowledged; the caller DENIES the retrieval when it did not (R3).

        **Why failing closed is right here, having previously served anyway.**
        The earlier reasoning was that denying turns a cosmetic risk into a
        functional failure. It does not hold: §6 exists precisely because a
        value taken through ``$( )`` is invisible to every filter downstream,
        so an unacknowledged notice does not mean "the transcript is slightly
        at risk", it means this secret WILL be written to the operator's
        transcript in plaintext with nothing left to catch it. That is not
        cosmetic, and an attacker who controls whether the session acks could
        choose it deliberately. A denial, by contrast, is visible and
        recoverable: the caller gets an error naming the wedged session.

        The blast radius is deliberately narrow. Only a descendant retrieval
        can be denied this way — a session fetching its own value never reaches
        here (see :meth:`_handle_retrieve`), so a wedged UI cannot lock the
        operator out of their own store, and ``lop secret get`` typed in a
        terminal with no session ancestor has no notice to wait for.

        The per-session write lock serialises concurrent retrievals attributed
        to the same session, so two notifications cannot interleave on one
        stream and read each other's acks. It is held across a BOUNDED socket
        operation only — never across the ancestry walk or the store read.
        """
        try:
            with session.write_lock:
                session.connection.settimeout(self._notify_ack_timeout_s)
                send_frame(
                    session.connection,
                    {
                        "event": "retrieved",
                        "name": name,
                        "value": encode_bytes(value),
                    },
                )
                reply = recv_frame(session.connection)
            return bool(reply.get("ack"))
        except (OSError, ProtocolError, socket.timeout):
            return False

    # --- audit --------------------------------------------------------------

    def _audit(self, identity: ProcessIdentity, event: str, outcome: str) -> None:
        """Record a broker decision against the store's hash chain (§12).

        Best-effort by construction: a store that cannot be opened (locked,
        not yet created, unreadable) must not turn an authorization decision
        into an exception. The decision has already been made by the time this
        runs; losing its audit row is a lesser failure than failing the call.
        """
        try:
            from local_operator.secrets.store import SecretStore

            # Resolve through `_master_key`, not `self._key`. In `keyfile` mode
            # the key arrives lazily from `_key_provider` and `self._key` stays
            # None until something forces it, so a bare None check silently
            # dropped EVERY audit row in the default tier — including the
            # `register` rows R5 asks for. A locked passphrase-tier broker
            # genuinely has no key and still returns early, which is correct:
            # there is no chain to append to until it is unlocked.
            key = self._master_key()
            SecretStore(key, base=self._base).record_broker_event(
                event=event,
                outcome=outcome,
                pid=identity.pid,
                exe=executable_path(identity.pid),
            )
        except Exception:  # noqa: BLE001 - audit must never break a decision
            pass

    def _audit_denial(self, identity: ProcessIdentity, operation: Any, reason: str) -> None:
        self._audit(identity, f"deny:{operation}", "deny")

    def _reply_error(
        self, connection: socket.socket, code: str, message: str, **extra: Any
    ) -> None:
        """Refuse one request. ``extra`` carries fields a caller must act on.

        Kept open-ended rather than growing a parameter per code: the two live
        uses (``kind`` on a store failure, ``pid``/``protocol`` on a version
        refusal) are both "the caller cannot recover without this", and a
        reader unaware of a field ignores it, so adding one is backward
        compatible on the wire.
        """
        try:
            send_frame(connection, {"ok": False, "code": code, "error": message, **extra})
        except (OSError, ProtocolError):  # pragma: no cover - peer already gone
            pass


def executable_path(pid: int) -> str | None:
    """The peer's executable path, for the audit trail (§12).

    ``proc_pidpath`` on macOS (spike 1) and ``/proc/<pid>/exe`` on Linux. Best
    effort: it is evidence for the operator reading an audit log, never an
    authorization input — the path is a property of a pid that may already have
    changed, and §2.1 records that verifying the peer's code is not available
    on this platform at all.
    """
    import sys as _sys

    if _sys.platform == "darwin":
        import ctypes

        from local_operator.secrets.peer import _libc

        libc = _libc()
        if libc is None or not hasattr(libc, "proc_pidpath"):
            return None
        buffer = ctypes.create_string_buffer(4096)
        written = libc.proc_pidpath(ctypes.c_int(pid), buffer, ctypes.c_uint32(4096))
        if written <= 0:
            return None
        return buffer.value.decode("utf-8", "replace")
    if _sys.platform.startswith("linux"):
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None
    return None


def run_broker(base: Path | None = None, *, idle_shutdown_s: float = IDLE_SHUTDOWN_S) -> int:
    """Run a broker in the foreground until it stops. The daemon entry point.

    In ``keyfile`` mode the key is read eagerly through a provider so the first
    request does not pay for it; in ``passphrase`` mode there is no provider
    and the broker starts LOCKED, which is what lets ``lop secret unlock``
    reach a listening daemon.
    """
    from local_operator.secrets.keys import key_mode, load_master_key

    # **`key_mode` and not `key_of_record_is_plaintext`, checked deliberately
    # during the R4-1 caller audit.** This is a startup/availability question —
    # "is there a key I can load eagerly, or must I start locked and wait for
    # `unlock`" — not an authorization one, and it is answered in the daemon's
    # own process about its own store. A planted `master.key` does make a
    # hardened broker start with a provider holding junk, but nothing is
    # disclosed by it: the registration gate below validates against the store's
    # fingerprint and still demands lineage, so the junk is never served, while
    # legitimate callers get the honest "sealed under a key that is not on disk"
    # and `status` names the planted file outright. Validating here instead
    # would ALSO make a keyfile store caught mid-rotation start locked and tell
    # its operator to run `unlock` on a store that has no passphrase, which is a
    # worse answer than the one the fingerprint would be buying.
    provider: Callable[[], bytes] | None = None
    if key_mode(base) == "keyfile":
        provider = lambda: load_master_key(base)  # noqa: E731 - a def here shadows the annotation

    broker = SecretBroker(base, key_provider=provider, idle_shutdown_s=idle_shutdown_s)
    broker.start()
    try:
        broker.serve_forever()
    except KeyboardInterrupt:
        broker.stop()
    return 0


__all__ = [
    "BrokerError",
    "SecretBroker",
    "executable_path",
    "lock_path",
    "run_broker",
    "socket_path",
]
