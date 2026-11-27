"""A WebSocket carrier for the mesh peer link, beneath ``wire``'s record codec.

WHY A TRANSPORT AND NOT A PROTOCOL. A ``PeerLink`` is a byte stream carrying
sealed records — ``wire.FrameReader`` reads the handshake LINES and then the
length-prefixed RECORDS, and ``wire.LinkCrypto`` seals and opens them. Which
byte stream those records ride is not part of the protocol: the framing, the
AEAD, the handshake and the link version stay byte-identical whether the bytes
arrive over TCP or inside WebSocket binary frames. So this module sits BELOW
``wire`` and changes nothing in it — no protocol version bump, no second codec,
no crypto review.

WHY WEBSOCKETS AT ALL (ENG-4225 Slice 1, L1a). A pool member's pod accepts no
inbound connection, and the mesh peer link is raw TCP, so a pod could never be
dialled. The tunnel plane already carries HTTP/WS upgrades from an edge to a
run's loopback target, so the pod's relay listens on loopback and the run-scoped
connector forwards the upgrade to it; the laptop reaches the same pod at a
run-scoped ``wss://`` hostname over 443. This module is that adapter, on both
ends: :func:`dial_ws` for the outbound leg and :class:`WebSocketAcceptServer` for
the loopback listener that hands each accepted connection to the SAME handshake
callback the TCP accept loop uses.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.
* It is not a liveness mechanism. ``ping_interval`` is disabled: the wire's own
  keepalive (``wire.KEEPALIVE_S``) and idle budget (``wire.LINK_IDLE_S``) already
  decide when a link is dead, and a second timer with its own opinion would be
  exactly the kind of disagreement this repo treats as a defect.
* It is not a security boundary. Trust is still the handshake's call: an
  endpoint that accepts bytes is a stranger until ``Handshake`` proves the
  device. ``wss://`` gets the standard VERIFIED TLS context; ``ws://`` is
  plaintext and is therefore only ever dialled on a loopback hop.
* It is not a config surface. The listener is an explicit call
  (``RelayServer.serve_ws``), so L1a adds no settings key, no CLI flag and no
  user-facing copy to localise.

THE IMPORT RULE. ``websockets`` is imported INSIDE the functions that use it, for
the reason ``wire`` gives about ``cryptography``: ``network/cli.py`` imports this
package for argument registration on the CLI startup path.
"""

from __future__ import annotations

import errno
import http
import ipaddress
import logging
import socket
import threading
import time
from typing import Any, Callable, NamedTuple
from urllib.parse import urlsplit

from local_operator.network import wire

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------

#: The only two schemes a mesh peer link ever rides. Anything else — ``http://``,
#: ``tcp://``, a bare ``host:port`` — is the raw TCP path or a mistake, and is
#: refused HERE rather than surfacing later as a dial to a host named ``http``.
WS_SCHEMES = frozenset({"ws", "wss"})

#: The port a scheme implies when the address names none. A run-scoped tunnel
#: hostname is bare (``wss://run-…``) and must mean 443; a loopback listener is
#: always spelled with its port, because there a default would be a guess.
DEFAULT_PORT_BY_SCHEME = {"ws": 80, "wss": 443}

#: The only host a plaintext ``ws://`` dial may name. See :func:`dial_ws`.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: The tag AES-256-GCM appends to every record, and the reason the read bound below
#: is stated against the RIGHT side of the envelope: ``FrameReader
#: .read_record_payload`` bounds the LENGTH PREFIX, which counts the ciphertext —
#: tag included — so a bound that added this tag again would size the library's
#: buffer past anything the wire ever delivers.
AEAD_TAG_BYTES = 16

#: One WebSocket message carries one sealed record: ``wire``'s own 4-byte length
#: prefix plus the ciphertext that prefix measures. This is EXACTLY the largest
#: message the wire will ever hand over — ``read_record_payload`` refuses a prefix
#: above :data:`wire.MAX_RECORD_BYTES` before it allocates a byte — so the library's
#: reader is bounded by the number the wire itself uses, rather than by a default
#: that knows nothing about it (1 MiB) or by a larger number that would only buffer
#: what the wire is about to refuse.
#:
#: THE TAG IS ALREADY COUNTED, once: ``MAX_RECORD_BYTES`` bounds the prefix, the
#: prefix counts ``AES-256-GCM(plaintext)``, and that ciphertext carries
#: :data:`AEAD_TAG_BYTES`. The consequence is worth stating because it looks like a
#: defect the first time it is measured: ``seal`` accepts a plaintext of exactly
#: ``MAX_RECORD_BYTES`` and the reader then refuses its own sender's record by the
#: tag's 16 bytes. That asymmetry is in ``wire``, is byte-identical on TCP, and is
#: not this carrier's to widen — a wider bound here would deliver 16 bytes the
#: reader drops, not a record it keeps.
MAX_MESSAGE_BYTES = wire.MAX_RECORD_BYTES + wire.LOCAL_PREFIX_LEN

#: How long a close may wait for the peer's half of the closing handshake. Short
#: on purpose: every close in the mesh today is a hard ``shutdown`` + ``close``,
#: and a link teardown must not become slower just because the carrier could
#: negotiate politely. :meth:`WebSocketPeerSocket.shutdown` is abortive for the
#: same reason (see its docstring).
CLOSE_TIMEOUT_S = 1.0

#: How long :meth:`WebSocketAcceptServer.stop` waits for its accept thread.
ACCEPT_THREAD_JOIN_S = 5.0

#: Messages the library may queue per connection before it stops reading. Small on
#: purpose: this queue is filled by a connection that has authenticated NOTHING, and
#: its only job is to bridge one record into the reader's hands. At
#: :data:`MAX_MESSAGE_BYTES` one record is up to 8 MiB, so the library's default of
#: 16 lets a stranger park 128 MiB per connection; two bounds that at 16 MiB and
#: leaves the real backpressure where it belongs — the wire's own bounded queues
#: (:data:`wire.QUEUE_FRAMES` / :data:`wire.QUEUE_BYTES`), which exist once a link is
#: up and can say ``net_backpressure`` rather than grow.
MAX_QUEUED_MESSAGES = 2


class WebSocketEndpointError(ValueError):
    """An address that is not a usable mesh WebSocket endpoint.

    A ``ValueError`` because the caller's input is what is wrong; the relay maps
    it to a refusal the same way it maps ``bad_endpoint``.
    """


class WSEndpoint(NamedTuple):
    """A validated ``ws://``/``wss://`` peer address."""

    scheme: str
    host: str
    port: int
    #: Path and query, verbatim, including the leading ``/`` when there is one.
    #: A run-scoped tunnel hostname carries none ("/"); a connector may route by
    #: path, so it is preserved rather than dropped.
    path: str

    @property
    def uri(self) -> str:
        """The address rebuilt for the client, IPv6 literals bracketed again."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}{self.path}"


def parse_ws_endpoint(address: str) -> WSEndpoint:
    """Validate a mesh WebSocket address, or raise :class:`WebSocketEndpointError`.

    SYNTAX ONLY — whether a plaintext hop is allowed is a dial decision
    (:func:`dial_ws`), not a property of the string. Credentials in the URL are
    refused outright: a mesh peer is authenticated by its device key in the
    handshake, and a URL carrying a password would put a secret into a stored
    member row, an audit detail and a log line.
    """
    text = str(address or "").strip()
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise WebSocketEndpointError(f"{address!r} is not a valid WebSocket URL ({exc})") from exc
    scheme = parts.scheme.lower()
    if not scheme:
        raise WebSocketEndpointError(
            f"{address!r} names no scheme; a mesh WebSocket needs ws:// or wss://"
        )
    if scheme not in WS_SCHEMES:
        raise WebSocketEndpointError(
            f"{address!r} uses {scheme}://; a mesh peer link only rides ws:// or wss://"
        )
    if parts.username is not None or parts.password is not None:
        raise WebSocketEndpointError(
            f"{address!r} carries credentials; a mesh peer is authenticated by its device "
            "key in the handshake, never by a URL"
        )
    if parts.fragment:
        raise WebSocketEndpointError(
            f"{address!r} carries a fragment, which is never sent to a server"
        )
    host = (parts.hostname or "").lower()
    if not host:
        raise WebSocketEndpointError(f"{address!r} names no host")
    if port is None:
        port = DEFAULT_PORT_BY_SCHEME[scheme]
    if not 0 < int(port) < 65536:
        raise WebSocketEndpointError(f"{address!r} names port {port}, which is not a port")
    path = parts.path or ""
    if parts.query:
        path = f"{path}?{parts.query}"
    return WSEndpoint(scheme=scheme, host=host, port=int(port), path=path)


def carries_ws_scheme(address: str) -> bool:
    """Whether ``address`` ANNOUNCES a ws/wss scheme, valid or not (never raises).

    Separate from :func:`parse_ws_endpoint` because the dial path asks a different
    question: it must pick the carrier BEFORE parsing, and a MALFORMED
    ``ws://user:pw@host`` that answered "not a WebSocket" would be re-read as
    ``host:port`` — dialling a host whose name begins with its own scheme. So a
    string that claims the scheme is a WebSocket endpoint as far as the carrier
    is concerned, and its parse then fails as ``bad_endpoint`` where the answer
    belongs.
    """
    scheme, sep, _rest = str(address or "").strip().partition("://")
    return bool(sep) and scheme.lower() in WS_SCHEMES


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class WebSocketDialError(OSError):
    """A ``ws://``/``wss://`` dial that never reached an open WebSocket.

    An ``OSError`` subclass because that is what every dial seam in this package
    already catches (``probe_candidates`` records ``connect_failed:<class>``), so
    a failed upgrade and a refused TCP connect are reported the same way.
    """


class WebSocketDialTimeout(WebSocketDialError, TimeoutError):
    """A dial that opened no WebSocket inside its budget."""


# ---------------------------------------------------------------------------
# The socket-shaped shell around one connection
# ---------------------------------------------------------------------------


class WebSocketPeerSocket:
    """One open WebSocket, presented as the byte stream ``wire`` expects.

    THE CONTRACT IS THE FOUR METHODS ``wire.FrameReader`` AND ``PeerLink`` USE:
    ``recv``/``settimeout``/``sendall``/``shutdown`` + ``close``. Nothing here
    re-implements framing: a message's bytes are handed back through a buffer,
    so a 64 KiB read from the middle of an 8 MiB record behaves exactly as it
    does on a TCP socket. THAT BUFFER IS LOAD-BEARING — without it every read
    would either return a whole message regardless of ``bufsize`` (which
    ``FrameReader._read_exactly`` treats as a stream and would then drop the
    tail of) or lose bytes between messages.

    ``recv`` returns ``b""`` when the peer is gone, exactly as a socket does, so
    ``FrameReader`` raises its own ``ConnectionError`` with its own sentence
    rather than this class inventing a second vocabulary for the same event.

    NO ``setsockopt``, DELIBERATELY. There is no TCP option to set through this
    object: ``tune_tcp`` names the one decision the relay makes (Nagle is already
    off), and a caller that asks for a socket option it cannot have —
    ``handshake.set_pair_keepalive``, whose every option is caught — gets ``False``
    ("not set") rather than a lie that it was.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection
        #: Captured at construction: ``close_socket`` may drop the underlying
        #: socket, and both addresses are wanted later for audit rows.
        self._peer = _address_tuple(getattr(connection, "remote_address", None))
        self._local = _address_tuple(getattr(connection, "local_address", None))
        #: Bytes of a message that a smaller ``recv`` could not take yet.
        self._pending = b""
        self._buffer_lock = threading.Lock()
        self._send_lock = threading.Lock()
        #: The per-read bound ``FrameReader`` sets before every ``recv``.
        self._timeout: float | None = None
        self._closed = threading.Event()

    # -- reads --------------------------------------------------------------

    def recv(self, bufsize: int) -> bytes:
        """Up to ``bufsize`` bytes of the record stream; ``b""`` at end of link."""
        if bufsize <= 0:
            return b""
        with self._buffer_lock:
            if self._pending:
                chunk = self._pending[:bufsize]
                self._pending = self._pending[bufsize:]
                return chunk
        message = self._recv_message()
        # AN EMPTY BINARY FRAME IS LEGAL AND IS NOT END OF STREAM. On a socket
        # ``b""`` means the peer is gone, so answering one for a zero-length
        # message would report a live link as dead; read past it instead.
        while message is not None and not message:
            message = self._recv_message()
        if message is None:
            return b""
        if len(message) <= bufsize:
            return message
        with self._buffer_lock:
            self._pending = message[bufsize:]
        return message[:bufsize]

    def _recv_message(self) -> bytes | None:
        # ONE READER THREAD is an assumption ``wire`` already makes (a
        # ``FrameReader`` is not shared), so no read lock is taken across the
        # blocking call: holding one would let a ``settimeout`` from another
        # thread queue behind a read that is waiting for the very frame it
        # bounds.
        from websockets.exceptions import ConnectionClosed

        try:
            message = self._conn.recv(timeout=self._timeout)
        except ConnectionClosed:
            # ONE EVENT, ONE FLAG. ``wait_closed`` and the accept handler learn
            # about the link ending from the same event a read does, so a peer
            # that closes is never a socket whose ``recv`` says gone while its
            # waiter still believes the link is up.
            self._closed.set()
            return None
        if isinstance(message, str):
            # A TEXT frame is not a mesh link: every byte the wire sends is
            # binary. Treating it as bytes would silently reinterpret it, and a
            # peer that sends one is not speaking this protocol.
            raise ConnectionError(
                "a text frame arrived on a mesh link, which only carries binary frames"
            )
        return bytes(message)

    # -- writes -------------------------------------------------------------

    def sendall(self, data: bytes | bytearray | memoryview) -> None:
        """Send ``data`` as ONE binary message (the caller already framed it)."""
        from websockets.exceptions import ConnectionClosed

        payload = bytes(data)
        with self._send_lock:
            try:
                self._conn.send(payload)
            except ConnectionClosed as exc:
                # A closed socket's ``sendall`` raises; this is the same answer
                # for the same event, in the vocabulary the callers already catch.
                self._closed.set()
                raise ConnectionError("the peer closed the connection") from exc

    # -- socket surface -----------------------------------------------------

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = None if timeout is None else float(timeout)

    def gettimeout(self) -> float | None:
        return self._timeout

    def getsockname(self) -> tuple[str, int]:
        return self._local

    def getpeername(self) -> tuple[str, int]:
        return self._peer

    def shutdown(self, how: int) -> None:
        """Close both directions, or refuse: a WebSocket has no half-close.

        Mapping ``SHUT_WR`` onto a full close would truncate the peer's stream
        while hiding it behind the name of a half-close, so only ``SHUT_RDWR`` —
        the one the mesh uses (``PeerLink.close``) — is answered, and abortively:
        the wire's close path is a hard close on a raw socket too, and a polite
        closing handshake here would be a slower carrier for the same event.
        """
        if how != socket.SHUT_RDWR:
            raise OSError(errno.ENOTSUP, "a WebSocket connection has no half-close")
        self.close()

    def close(self) -> None:
        """Drop the connection. Idempotent, and never raises (``_close_quietly``)."""
        if self._closed.is_set():
            return
        self._closed.set()
        from websockets.exceptions import WebSocketException

        try:
            self._conn.close_socket()
        except (OSError, WebSocketException):
            pass

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Block until this adapter is closed; ``False`` when ``timeout`` expired.

        The accept handler needs this: ``websockets`` closes a connection when its
        handler returns, so the handler must outlive the link it handed on.

        NOT ``self._conn.wait_closed``, DELIBERATELY. That method arrives in
        ``websockets`` 17.2 and this package declares ``websockets>=15.0.1``;
        calling it would be an ``AttributeError`` on a supported install, and the
        failure would land in the accept handler — the one thread whose survival
        keeps a live link open. The adapter already owns a closed event that every
        close path sets (``close``, EOF on read, a close error on send, this
        server's handler teardown), so it asks its own object and the carrier's
        minor version stops being part of this module's contract.
        """
        return self._closed.wait(None if timeout is None else max(0.0, float(timeout)))


def _address_tuple(value: Any) -> tuple[str, int]:
    """``(host, port)`` from whatever the library reports as an address."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return str(value[0]), int(value[1])
    return "", 0


def tune_tcp(sock: Any) -> None:
    """Disable Nagle on a raw carrier; a WebSocket carrier already has.

    WHY A NAMED DECISION RATHER THAN A NO-OP ``setsockopt``. The relay sets
    ``TCP_NODELAY`` on every accepted and dialled socket, and a WebSocket
    transport has no such option to set — the library already applies it to the
    socket it opened (``websockets.sync.server``'s per-connection handler does it
    explicitly, and asyncio's transports do it for the client). Answering with a
    no-op ``setsockopt`` would make the caller believe it configured something, so
    the carrier answers the question instead: one place, and no second opinion
    about who set the flag.
    """
    if isinstance(sock, WebSocketPeerSocket):
        return
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


# ---------------------------------------------------------------------------
# Dialing
# ---------------------------------------------------------------------------


def _refuse_plaintext_off_loopback(endpoint: WSEndpoint, sock: socket.socket) -> None:
    """Refuse a plaintext dial whose CONNECTED peer is not this machine.

    THE HOST CHECK IS NOT THIS CHECK, and the gap between them is a redirect. The
    URI's host is answered by whoever wrote the address; the peer IP is answered by
    the kernel after the connect, and it is the only one of the two that says where
    the bytes will actually go. A loopback server replying ``302 Location:
    ws://10.0.0.5:9000`` is the concrete bypass: the address (and every check made
    on it) is loopback, while the upgrade that follows lands somewhere else in the
    clear.

    IT RUNS BEFORE THE UPGRADE, on the socket this module connected, so a refusal
    costs the peer one TCP handshake rather than the handshake's device identities.
    ``wss://`` is not checked: its trust comes from the verified TLS context, which
    holds for any host a redirect could name.
    """
    if endpoint.scheme != "ws":
        return
    peer_ip = str(sock.getpeername()[0])
    try:
        # ``fe80::1%en0`` — a scoped literal is still a literal; the zone is not
        # part of the address the question is about.
        loopback = ipaddress.ip_address(peer_ip.split("%", 1)[0]).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        return
    raise WebSocketDialError(
        f"{endpoint.host}:{endpoint.port} connected to {peer_ip}, which is not a loopback "
        "address; a mesh link may only be plaintext on a loopback hop (use wss:// for "
        "anything else)"
    )


def dial_ws(address: str, *, timeout: float) -> WebSocketPeerSocket:
    """Open a mesh link across ``ws://`` (loopback) or ``wss://`` (443-capable).

    ``timeout`` bounds the WHOLE open: TCP (plus TLS for ``wss://``) and the
    upgrade, which is what the relay's per-candidate budget means.

    PLAINTEXT IS LOOPBACK-ONLY, AND THE CHECK IS ON THE PEER. ``ws://`` names no
    encryption, so dialling one across a network would put the handshake's device
    identities and endpoints on the wire in the clear; the tunnel plane terminates
    TLS at its edge and speaks plain WS only on its own loopback hop, which is the
    one case this allows. Two things enforce it, and neither is enough alone: the
    parsed host is refused unless it is a loopback name (cheap, and catches a typo
    before any socket exists), and the CONNECTED peer IP is refused unless it is
    loopback (:func:`_refuse_plaintext_off_loopback`), which is what a redirect
    cannot talk its way out of.

    NO REDIRECTS ARE FOLLOWED. From ``websockets`` 17 the client follows an HTTP
    3xx on its own (``WEBSOCKETS_MAX_REDIRECTS``, 10 by default) and there is no
    per-call switch for it — the depth is a module constant read at import. So this
    function does the TCP connect ITSELF and hands the socket over as ``sock=``:
    with a caller-supplied socket the library refuses to re-dial and raises instead
    (and 15.x, the declared floor, does not follow redirects at all). The library's
    own ``create_connection`` is what would have consulted a proxy, so nothing here
    can route a peer link through ``HTTP_PROXY``/``HTTPS_PROXY`` either.
    """
    from websockets.exceptions import WebSocketException
    from websockets.sync.client import connect

    endpoint = parse_ws_endpoint(address)
    if endpoint.scheme == "ws" and endpoint.host not in LOOPBACK_HOSTS:
        raise WebSocketDialError(
            f"{address!r} is a plaintext ws:// link to {endpoint.host}; a mesh link may only "
            "be plaintext on a loopback hop (use wss:// for anything else)"
        )
    deadline = time.monotonic() + timeout
    try:
        sock = socket.create_connection((endpoint.host, endpoint.port), timeout=timeout)
    except TimeoutError as exc:
        raise WebSocketDialTimeout(f"{address} opened no WebSocket within {timeout:.1f}s") from exc
    except OSError as exc:
        raise WebSocketDialError(f"{address} could not be dialled: {exc}") from exc
    try:
        _refuse_plaintext_off_loopback(endpoint, sock)
    except WebSocketDialError:
        sock.close()
        raise
    # The TCP leg spent part of the budget above; the upgrade gets what is left,
    # so ``timeout`` still bounds the WHOLE open rather than each leg of it.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        sock.close()
        raise WebSocketDialTimeout(f"{address} opened no WebSocket within {timeout:.1f}s")
    try:
        connection = connect(
            endpoint.uri,
            sock=sock,
            open_timeout=remaining,
            # THE CARRIER MUST NOT ADD LAYERS OF ITS OWN. The payload is AEAD
            # ciphertext, so per-message deflate would burn CPU on incompressible
            # bytes and add a compression oracle for nothing.
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            # Liveness belongs to the wire's keepalive (see the module docstring).
            ping_interval=None,
            close_timeout=CLOSE_TIMEOUT_S,
            # THE LINK OWNS THE CONNECTION, not a ``with`` block: it is handed to
            # the reader and writer threads and closed by whichever of them learns
            # the link died. ``legacy=True`` is the library's name for that
            # direct-ownership form (its context-manager form would close the
            # socket on a scope exit this module never has).
            legacy=True,
        )
    except TimeoutError as exc:
        sock.close()
        raise WebSocketDialTimeout(f"{address} opened no WebSocket within {timeout:.1f}s") from exc
    except (OSError, WebSocketException, ValueError) as exc:
        # ``ValueError`` is the library's refusal to follow a redirect onto the
        # socket it was handed; it is a dial that did not happen, not a bug here.
        sock.close()
        raise WebSocketDialError(f"{address} could not be dialled: {exc}") from exc
    return WebSocketPeerSocket(connection)


# ---------------------------------------------------------------------------
# Accepting (the loopback listener behind a tunnel connector)
# ---------------------------------------------------------------------------


class WebSocketAcceptServer:
    """Accept inbound mesh links over WebSocket on a LOOPBACK port.

    WHY NOT ``RelayServer``'s OWN LISTENER. That listener is the raw-TCP one a
    pod never exposes and that stays byte-identical; this is the second, opt-in
    surface a connector forwards upgrades to. The two share the admission and
    handshake path (``RelayServer._admit_inbound``) so a WebSocket arrival meets
    exactly the same pre-auth caps, membership rules and refusals as a TCP one —
    one admission policy, two carriers.

    A LOOPBACK BIND IS NOT A PREFERENCE. A WebSocket listener is unauthenticated
    until the handshake completes, so binding it to a public interface would be a
    second inbound surface nobody asked for. The check lives HERE, beside
    ``LOOPBACK_HOSTS``, so it holds for every caller of this class rather than for
    one entry point that remembered to restate it.

    THE CAP IS PRE-UPGRADE (``max_connections``). ``RelayServer._admit_inbound``
    refuses past its handshake budget, but only once a connection exists — by then
    the library has already spent a thread and a read buffer on a stranger who
    authenticated nothing. This listener therefore refuses the ``max_connections``th
    held connection with HTTP 503 in ``process_request``, BEFORE the upgrade, which
    is the last moment a refusal costs a request instead of a thread. The number is
    the caller's: :meth:`RelayServer.serve_ws` passes its own ``max_handshakes``, so
    one carrier cannot silently outrun the bound the other honours.
    """

    def __init__(
        self,
        on_accept: Callable[[WebSocketPeerSocket, tuple[str, int]], Any],
        *,
        max_connections: int,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        normalized = str(host or "").strip("[]").lower()
        if normalized not in LOOPBACK_HOSTS:
            raise ValueError(
                f"a mesh WebSocket listener must bind loopback, not {host!r}: it is "
                "unauthenticated until the handshake completes"
            )
        self._on_accept = on_accept
        self._max_connections = max(1, int(max_connections))
        self._host = host
        self._port = int(port)
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._address: tuple[str, int] = ("", 0)
        #: The pre-upgrade cap's ledger: connections this listener is HOLDING, keyed
        #: by identity and holding a strong reference so an id cannot be reused
        #: under a live entry. Entered in ``process_request``, left when the
        #: upgrade is refused or the connection ends.
        self._lock = threading.Lock()
        self._inflight: dict[int, Any] = {}
        #: The adapters handed on, so ``stop`` can close the links it opened.
        self._peers: set[WebSocketPeerSocket] = set()

    @property
    def address(self) -> tuple[str, int]:
        """The address actually bound, which for ``port=0`` is the kernel's choice."""
        return self._address

    def start(self) -> tuple[str, int]:
        """Bind, listen, and start accepting. Returns the bound address."""
        from websockets.sync.server import serve

        self._server = serve(
            self._handle,
            self._host,
            self._port,
            compression=None,
            max_size=MAX_MESSAGE_BYTES,
            max_queue=MAX_QUEUED_MESSAGES,
            # The pre-upgrade cap, and the slot it took handed back when the
            # upgrade does not become a connection.
            process_request=self._hold_slot_or_refuse,
            process_response=self._release_refused_slot,
            ping_interval=None,
            close_timeout=CLOSE_TIMEOUT_S,
        )
        host, port = self._server.socket.getsockname()[:2]
        self._address = (str(host), int(port))
        self._thread = threading.Thread(
            target=self._serve_forever, name="mesh-ws-accept", daemon=True
        )
        self._thread.start()
        return self._address

    def _serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:  # noqa: BLE001 — a dead accept loop must not vanish silently
            if not self._stopping.is_set():
                # Only worth a log while the relay believes it is listening: an
                # exception during ``stop`` is the socket being closed under it.
                logger.exception("the WebSocket accept loop ended")

    def _hold_slot_or_refuse(self, connection: Any, request: Any) -> Any:
        """``process_request``: take one of the pre-upgrade slots, or answer 503.

        THE LAST MOMENT BEFORE A THREAD IS SPENT. ``_admit_inbound`` is the shared
        admission policy and it runs later, once the upgrade has succeeded; a burst
        of strangers would each have bought a thread and a read buffer by then. The
        ledger counts connections this listener is holding — reserved here, released
        by :meth:`_release_refused_slot` when the upgrade is refused and by
        :meth:`_handle` when the connection ends — so the refusal happens before the
        library hands anything to a handler.

        503 rather than the silent close the TCP path uses: this surface already
        speaks HTTP, the client is a peer that asked for an upgrade, and a documented
        status is a better answer than a dropped socket for a peer that is merely
        over a legitimate burst.
        """
        with self._lock:
            if len(self._inflight) >= self._max_connections:
                return connection.protocol.reject(
                    http.HTTPStatus.SERVICE_UNAVAILABLE, "too many connections\n"
                )
            self._inflight[id(connection)] = connection
        return None

    def _release_refused_slot(self, connection: Any, request: Any, response: Any) -> Any:
        """``process_response``: hand the slot back when the upgrade did not happen.

        A refused or malformed handshake never reaches :meth:`_handle`, so this is
        the only place its reservation is released. Without it every plain HTTP probe
        or unsupported client would shrink the cap by one for the life of the
        listener — a bound that decays toward refusing every real peer is a worse
        failure than the one it guards against.
        """
        if int(getattr(response, "status_code", 0)) != http.HTTPStatus.SWITCHING_PROTOCOLS:
            with self._lock:
                self._inflight.pop(id(connection), None)
        return None

    def _handle(self, connection: Any) -> None:
        """One accepted WebSocket: hand it on, then outlive it.

        ``websockets`` closes a connection the moment its handler returns, so
        this handler WAITS for the link's own close. That is what makes the
        handover safe: the relay's threads own the reads and writes, and this
        thread only holds the connection open for them.

        ``False`` FROM THE ADMISSION GATE MEANS STOP. ``_admit_inbound`` answers
        that when the relay has already decided to stop accepting, and then the
        arrival owes the peer nothing: it is closed here and the handler returns
        rather than parking on a link no thread will ever close.
        """
        peer = WebSocketPeerSocket(connection)
        with self._lock:
            self._peers.add(peer)
        try:
            if self._on_accept(peer, peer.getpeername()) is False:
                return
            peer.wait_closed()
        except Exception:  # noqa: BLE001 — one broken arrival must not stop accept
            logger.exception("a WebSocket arrival was refused with an error")
        finally:
            # Orderly in every exit: close first (the adapter is idempotent) so
            # anything waiting on this connection sees it, then give the slot back.
            peer.close()
            with self._lock:
                self._peers.discard(peer)
                self._inflight.pop(id(connection), None)

    def stop(self) -> None:
        """Stop accepting and drop the connections this server opened.

        THE ORDER IS THE POINT. A handler parks in ``wait_closed`` for as long as its
        link is up, and the library's ``shutdown`` closes connections and JOINS those
        handlers — so closing our own links FIRST is what lets those joins finish
        promptly instead of waiting out a shutdown that cannot complete. The accept
        thread is joined under the same bound, so ``stop`` cannot hang on a carrier.
        """
        self._stopping.set()
        server, thread = self._server, self._thread
        if server is None:
            return
        with self._lock:
            peers = list(self._peers)
            # Nothing holds a slot once this listener is stopping, and a handler
            # that outlives the shutdown below would only leak the count.
            self._inflight.clear()
        for peer in peers:
            peer.close()
        try:
            # NO ``close_connections`` ARGUMENT: that keyword arrives with
            # ``websockets`` 17 and the declared floor is 15.0.1, where ``shutdown()``
            # only stops the accept loop (a ``TypeError`` there used to be swallowed
            # below and left the loop running until its join timed out). The links are
            # closed above, which is the part of a shutdown this class owns; the
            # library closes anything still open on the versions that can.
            server.shutdown()
        except Exception:  # noqa: BLE001 — teardown must not raise into a stop path
            logger.exception("the WebSocket accept server refused to stop")
        if thread is not None:
            thread.join(timeout=ACCEPT_THREAD_JOIN_S)
