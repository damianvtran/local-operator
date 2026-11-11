"""The relay connector: one WSS + yamux session to ``tunnel-relay``, piped to the gateway.

Contract: agent-server ``docs/TUNNEL_RELAY.md`` C1 (wire) and C2 (``/connect``).

WHAT IT DOES. :class:`RelayConnector` holds the device's relay session open for
as long as the tunnel unit runs:

1. Dial ``relay.url`` exactly as `/connect` returned it (C1.1: never build it),
   offering one subprotocol, ``radient-tunnel.v1``, with the connector JWT in
   ``Authorization`` and nowhere else.
2. Open yamux stream 2, send ``hello``, wait for ``ready`` (C1.4). Only after
   ``ready`` is the device "connected": :attr:`RelayConnector.connected` is what
   replaces cloudflared's pidfile check in the gateway's health answer.
3. Accept every relay-opened stream and pipe it, unparsed, to
   ``127.0.0.1:<gateway_port>`` (C1.3). A refused gateway dial is a yamux RST,
   which the relay turns into a 502.
4. On ``rotate``/``drain`` make-before-break (C1.5): open a successor, and once
   it is ``ready`` send ``retire`` on the old control stream. The old session
   keeps piping its in-flight streams until the relay closes it — never more
   than two sessions at once.
5. On ``revoked`` (control line, 403 at handshake, or close 4403) stop and set
   :attr:`revoked`: the remedy is a fresh `/connect`, which the tunnel service
   gets by restarting the unit, exactly as its poller already does for a
   version change.

WHY NOTHING HERE TOUCHES DISK. The connector token lives only in memory. It is
scoped to one tunnel's relay handshake and expires in 10 minutes, and keeping it
off disk means there is no new secret file for confinement to protect (unlike
``tunnel/cloudflared.token``).

Retry discipline follows C1.1/C1.6 to the letter, because the relay rate-limits
handshakes per tunnel (6 burst, 0.1/s): a connector that retried faster would
lock itself out. Full-jitter exponential backoff from 1 s to a 30 s cap.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from local_operator.tunnels.relay import yamux

PROTOCOL = "radient-tunnel.v1"
#: The only relay URL production accepts (C1.1). `/connect` is the source of
#: the value; this pin is what stops a tampered or misconfigured control plane
#: from aiming this device's public traffic at another host — the same trust
#: boundary `config.validate_connection` keeps for the origin issuer.
RELAY_URL = "wss://tunnels.radienthq.com/v1/connector"
#: C1.2: the receiver MUST accept messages up to 2 MiB.
MAX_MESSAGE = 2097152
#: C1.6 liveness: WS ping every 20 s, 20 s pong timeout, both under the ALB's
#: 120 s group idle timeout.
PING_INTERVAL = 20
PING_TIMEOUT = 20
#: C1.4: a control line is at most 4096 bytes before its LF.
MAX_LINE = 4096
#: C1.6 token reuse: reuse the current token while it has >= 60 s left.
TOKEN_REUSE_MARGIN = 60
#: Full-jitter backoff cap (C1.1).
BACKOFF_CAP = 30.0
#: How long to wait for ``ready`` after ``hello``. Not a tuned number: the relay
#: answers within its own 10 s hello window plus one Redis ``register.lua`` call
#: (milliseconds), so this only bounds a relay that accepted the WebSocket and
#: then went silent, which would otherwise hold the connector forever.
READY_TIMEOUT = 30.0
#: Opening the WebSocket (TCP + TLS + upgrade); websockets' own default.
OPEN_TIMEOUT = 10.0
#: A session that stayed up this long after `ready` was healthy, and its close
#: resets the reconnect backoff. Above the relay's 60 s superseded/drain close
#: window (C1.5/C1.6), so a supersede fight (each session ending within that
#: window) always reads as consecutive failures.
STABLE_SESSION_SECONDS = 120.0
#: The relay caps a session at 128 concurrent streams (C1.6); twice that bounds a
#: misbehaving peer without ever refusing a legitimate request.
MAX_PIPES = 256

Announce = Callable[[str], None]
Refresh = Callable[[], Awaitable["RelayGrant | None"]]


@dataclass(frozen=True)
class RelayGrant:
    """The ``relay`` object from `/connect` (C2.1), validated."""

    url: str
    # repr=False: a traceback with locals, a debugger or a stray f"{grant}"
    # must never print the connector token.
    token: str = field(repr=False)
    expires_at: float  # Unix seconds
    protocol: str = PROTOCOL

    def fresh(self, now: float | None = None) -> bool:
        return self.expires_at - (time.time() if now is None else now) >= TOKEN_REUSE_MARGIN


def parse_grant(value: object, *, pinned_url: str | None = RELAY_URL) -> RelayGrant | None:
    """The relay grant `/connect` offered, or ``None`` for "no relay" (C2.1).

    Absence and a malformed object read the same way on purpose: the caller's
    fallback for both is the cloudflared-only path that every server still
    supports during the dual-run, so a bad relay block degrades to today's
    behaviour instead of taking the tunnel down. ``pinned_url=None`` is for the
    local E2E rig only; production callers keep the default pin.
    """
    if not isinstance(value, dict):
        return None
    url, token, expires, protocol = (
        value.get("url"),
        value.get("token"),
        value.get("expires_at"),
        value.get("protocol"),
    )
    if protocol != PROTOCOL or not isinstance(url, str) or not isinstance(token, str):
        return None
    if not token or any(c in token for c in "\r\n "):
        return None
    if pinned_url is not None and url != pinned_url:
        return None
    if not isinstance(expires, str):
        return None
    try:
        stamp = datetime.datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    return RelayGrant(url=url, token=token, expires_at=stamp.timestamp(), protocol=protocol)


def control_line(message: dict[str, Any]) -> bytes:
    """One control line, byte-exact with C1.4's emitters (key order as given)."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def hello_line(*, connector: str, tunnel_id: str, version: int) -> bytes:
    return control_line(
        {
            "type": "hello",
            "protocol": PROTOCOL,
            "connector": connector,
            "tunnel_id": tunnel_id,
            "version": version,
        }
    )


def retire_line(successor_session_id: str) -> bytes:
    return control_line({"type": "retire", "successor_session_id": successor_session_id})


def backoff_delay(attempt: int) -> float:
    """C1.1: ``uniform(0, min(30, 2^n))``."""
    return random.uniform(0, min(BACKOFF_CAP, 2.0 ** max(attempt, 0)))


class HandshakeRefused(Exception):
    """The relay answered the upgrade with a refusal (C1.1's table)."""

    def __init__(self, status: int, error: str, retry_after: float | None) -> None:
        super().__init__(f"relay refused the connector handshake: {status} {error}")
        self.status = status
        self.error = error
        self.retry_after = retry_after


class ProtocolViolation(Exception):
    """The relay broke C1; close 4400 with ``code`` as the reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _refusal(failure: BaseException) -> HandshakeRefused | None:
    from websockets.exceptions import InvalidStatus

    if not isinstance(failure, InvalidStatus):
        return None
    response = failure.response
    error = ""
    with contextlib.suppress(ValueError, UnicodeDecodeError, AttributeError):
        body = json.loads(response.body or b"{}")
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            error = body["error"]
    retry_after: float | None = None
    with contextlib.suppress(TypeError, ValueError):
        retry_after = max(0.0, float(response.headers.get("Retry-After", "")))
    return HandshakeRefused(response.status_code, error, retry_after)


async def pipe_to_gateway(stream: yamux.Stream, host: str, port: int) -> None:
    """One relay stream ⇄ one loopback gateway connection, bytes only (C1.3).

    Each direction half-closes independently: the relay's FIN becomes
    ``shutdown(SHUT_WR)`` on the gateway socket, and the gateway's EOF becomes
    our FIN. The stream is NOT closed when one response ends — the relay pools
    idle streams for keep-alive and closes them itself.
    """
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except OSError:
        await stream.reset()
        return

    async def relay_to_gateway() -> None:
        while chunk := await stream.read():
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    async def gateway_to_relay() -> None:
        while chunk := await reader.read(yamux.MAX_DATA_FRAME):
            await stream.write(chunk)
        await stream.close()

    tasks = [asyncio.create_task(relay_to_gateway()), asyncio.create_task(gateway_to_relay())]
    try:
        for finished in asyncio.as_completed(tasks):
            await finished
    except (yamux.StreamReset, ConnectionError, OSError):
        # A reset on either side ends both: a half-dead pipe would hold a
        # gateway connection open with nobody left to answer.
        await stream.reset()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        with contextlib.suppress(OSError, ConnectionError):
            await writer.wait_closed()


class RelaySession:
    """One WebSocket + yamux session with the relay (the S1/S2 of C1.5)."""

    def __init__(self, connector: RelayConnector, grant: RelayGrant) -> None:
        self._connector = connector
        self._grant = grant
        self._ws: Any = None
        self.mux: yamux.Session | None = None
        self._control: yamux.Stream | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._pipes: set[asyncio.Task[None]] = set()
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()
        #: Set by ``rotate``/``drain``: the connector should start a successor.
        self.rotate = asyncio.Event()
        self.rotate_reason = ""
        self.superseded = False
        self.revoked = False
        self.session_id = ""
        self.epoch = 0
        self.close_code: int | None = None
        self.close_reason = ""
        self.ready_at = 0.0

    async def open(self) -> None:
        """Handshake, hello, and wait for ``ready``; raise on any failure."""
        connector = self._connector
        try:
            self._ws = await connector.ws_connect(
                self._grant.url,
                subprotocols=[PROTOCOL],
                additional_headers={"Authorization": f"Bearer {self._grant.token}"},
                user_agent_header=connector.name,
                compression=None,
                max_size=MAX_MESSAGE,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                open_timeout=OPEN_TIMEOUT,
                # An ambient HTTPS_PROXY must not carry the connector token, for
                # the same reason the tunnel's httpx clients run trust_env=False.
                proxy=None,
            )
        except Exception as failure:
            refusal = _refusal(failure)
            if refusal is not None:
                raise refusal from failure
            raise
        if self._ws.subprotocol != PROTOCOL:
            await self._ws.close(4400, "unsupported_protocol")
            raise HandshakeRefused(400, "unsupported_protocol", None)
        self.mux = yamux.Session(self._send)
        self._tasks.append(asyncio.create_task(self._receive()))
        try:
            self._control = await self.mux.open_stream()
            await self._control.write(
                hello_line(
                    connector=connector.name,
                    tunnel_id=connector.tunnel_id,
                    version=connector.version,
                )
            )
            self._tasks.append(asyncio.create_task(self._read_control()))
            self._tasks.append(asyncio.create_task(self._accept()))
            ready = asyncio.create_task(self.ready.wait())
            closed = asyncio.create_task(self.closed.wait())
            try:
                await asyncio.wait(
                    {ready, closed}, timeout=READY_TIMEOUT, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                ready.cancel()
                closed.cancel()
            if not self.ready.is_set():
                raise ConnectionError(
                    f"relay session ended before ready (close {self.close_code} "
                    f"{self.close_reason or 'timeout'})"
                )
        except BaseException:
            await self.shutdown()
            raise

    async def retire(self, successor_session_id: str) -> None:
        """Tell the relay this session's successor is live (C1.5)."""
        if self._control is not None and not self.closed.is_set():
            with contextlib.suppress(yamux.StreamReset, ConnectionError):
                await self._control.write(retire_line(successor_session_id))

    async def shutdown(self, code: int = 1000, reason: str = "normal") -> None:
        """Close locally: GoAway, WS close, and end every pipe."""
        if self.mux is not None and not self.mux.closed:
            with contextlib.suppress(Exception):
                await self.mux.send_go_away(yamux.GO_AWAY_NORMAL)
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._ws.close(code, reason), timeout=5)
        self._finish()
        for task in [*self._tasks, *self._pipes]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._pipes, return_exceptions=True)

    # -- internals -------------------------------------------------------

    async def _send(self, data: bytes) -> None:
        await self._ws.send(data)

    def _finish(self) -> None:
        if self._ws is not None and self.close_code is None:
            self.close_code = self._ws.close_code
            self.close_reason = self._ws.close_reason or ""
        if self.mux is not None:
            self.mux.close()
        self.closed.set()

    async def _violation(self, code: str) -> None:
        self._connector.announce(f"relay protocol error ({code}); closing the session")
        if self.mux is not None and not self.mux.closed:
            with contextlib.suppress(Exception):
                await self.mux.send_go_away(yamux.GO_AWAY_PROTOCOL_ERROR)
        with contextlib.suppress(Exception):
            await self._ws.close(4400, code)
        self.close_code, self.close_reason = 4400, code

    async def _receive(self) -> None:
        from websockets.exceptions import ConnectionClosed

        assert self.mux is not None
        try:
            async for message in self._ws:
                if isinstance(message, str):
                    # C1.2: a text message is a protocol error.
                    await self._violation("bad_message")
                    return
                await self.mux.feed(message)
        except yamux.YamuxError:
            await self._violation("protocol_error")
        except ProtocolViolation as violation:
            await self._violation(violation.code)
        except ConnectionClosed:
            pass
        finally:
            self._finish()

    async def _read_control(self) -> None:
        assert self._control is not None
        pending = bytearray()
        try:
            while chunk := await self._control.read():
                pending += chunk
                while (end := pending.find(b"\n")) >= 0:
                    line = bytes(pending[:end])
                    del pending[: end + 1]
                    if len(line) > MAX_LINE:
                        raise ProtocolViolation("line_too_long")
                    await self._on_control(line)
                if len(pending) > MAX_LINE:
                    raise ProtocolViolation("line_too_long")
        except ProtocolViolation as violation:
            await self._violation(violation.code)
        except yamux.StreamReset:
            pass

    async def _on_control(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raise ProtocolViolation("bad_message") from None
        if not isinstance(message, dict):
            raise ProtocolViolation("bad_message")
        kind = message.get("type")
        if kind == "ready":
            self.session_id = str(message.get("session_id") or "")
            epoch = message.get("epoch")
            self.epoch = epoch if isinstance(epoch, int) else 0
            self.ready_at = time.monotonic()
            self.ready.set()
        elif kind in ("rotate", "drain"):
            self.rotate_reason = f"{kind}:{message.get('reason', '')}"
            self.rotate.set()
        elif kind == "superseded":
            self.superseded = True
        elif kind == "revoked":
            self.revoked = True
            self._connector.announce(
                f"relay revoked this connector (reason={message.get('reason', '')})"
            )
        elif kind == "error":
            self._connector.announce(
                f"relay error {message.get('code', '')}: {message.get('message', '')}"
            )
        # Any other type is forward-compatible and ignored (C1.4).

    async def _accept(self) -> None:
        assert self.mux is not None
        while (stream := await self.mux.accept()) is not None:
            if len(self._pipes) >= MAX_PIPES:
                await stream.reset()
                continue
            task = asyncio.create_task(
                pipe_to_gateway(stream, self._connector.gateway_host, self._connector.gateway_port)
            )
            self._pipes.add(task)
            task.add_done_callback(self._pipes.discard)


class RelayConnector:
    """Keep one relay session (plus, during rotation, its successor) alive."""

    def __init__(
        self,
        grant: RelayGrant,
        *,
        tunnel_id: str,
        version: int,
        gateway_port: int,
        refresh: Refresh,
        name: str,
        announce: Announce = lambda message: None,
        gateway_host: str = "127.0.0.1",
        ws_connect: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        if ws_connect is None:
            from websockets.asyncio.client import connect

            ws_connect = connect
        self.grant = grant
        self.tunnel_id = tunnel_id
        self.version = version
        self.gateway_port = gateway_port
        self.gateway_host = gateway_host
        self.name = name[:64]
        self.announce = announce
        self.ws_connect: Callable[..., Awaitable[Any]] = ws_connect
        self._refresh = refresh
        self._active: RelaySession | None = None
        self._retiring: set[RelaySession] = set()
        self._background: set[asyncio.Task[None]] = set()
        #: Set when the relay or `/connect` says this connector may not run:
        #: the service restarts the unit so `/connect` decides what happens next.
        self.revoked = asyncio.Event()
        #: Set when a refreshed `/connect` stopped offering a relay at all.
        self.withdrawn = asyncio.Event()
        self.stopping = False
        self._unstable_closes = 0

    @property
    def connected(self) -> bool:
        """True while a session that reached ``ready`` is still open (C1.4)."""
        return any(
            s.ready.is_set() and not s.closed.is_set()
            for s in (self._active, *self._retiring)
            if s is not None
        )

    async def run(self) -> None:
        """Hold the relay session until stopped, revoked or withdrawn."""
        attempt = 0
        try:
            while not self.stopping:
                session = await self._establish(attempt)
                if session is None:
                    if self.revoked.is_set() or self.withdrawn.is_set() or self.stopping:
                        return
                    attempt += 1
                    continue
                attempt = 0
                self._active = session
                self.announce(
                    f"relay session ready session_id={session.session_id} epoch={session.epoch}"
                )
                delay = await self._serve(session)
                if delay is None:
                    return
                if delay:
                    await asyncio.sleep(delay)
        finally:
            await self.close()

    async def close(self) -> None:
        self.stopping = True
        sessions = [s for s in (self._active, *self._retiring) if s is not None]
        self._active = None
        self._retiring.clear()
        await asyncio.gather(*(s.shutdown() for s in sessions), return_exceptions=True)
        for task in self._background:
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)

    # -- internals -------------------------------------------------------

    async def _serve(self, session: RelaySession) -> float | None:
        """Run ``session`` until it closes; return the reconnect delay, or None to stop."""
        while True:
            rotate = asyncio.create_task(session.rotate.wait())
            closed = asyncio.create_task(session.closed.wait())
            try:
                await asyncio.wait({rotate, closed}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                rotate.cancel()
                closed.cancel()
            if self.stopping:
                return None
            if session.closed.is_set():
                return self._after_close(session)
            # Make-before-break (C1.5). The successor dial uses the same
            # backoff as any other; if it cannot be ready before the relay's
            # deadline, the relay closes S1 itself and the close path above
            # reconnects.
            session.rotate.clear()
            self.announce(f"relay asked to rotate ({session.rotate_reason})")
            successor = await self._establish(0)
            retries = 0
            # Keep dialling (with backoff) while S1 is still open: one failed
            # dial must not turn make-before-break into break-before-make by
            # idling until the relay closes S1 at its deadline (C1.5).
            while (
                successor is None
                and not session.closed.is_set()
                and not (self.revoked.is_set() or self.withdrawn.is_set() or self.stopping)
            ):
                retries += 1
                successor = await self._establish(retries)
            if successor is None:
                if self.revoked.is_set() or self.withdrawn.is_set() or self.stopping:
                    return None
                continue
            await session.retire(successor.session_id)
            self._retire_in_background(session)
            self._active = successor
            session = successor
            self.announce(
                f"relay session ready session_id={session.session_id} epoch={session.epoch} "
                "(successor)"
            )

    def _retire_in_background(self, session: RelaySession) -> None:
        """Keep piping the old session's in-flight streams until the relay closes it."""
        self._retiring.add(session)

        async def linger() -> None:
            try:
                # C1.5: the relay closes a retired session within its 60 s
                # deadline; the extra margin only covers a relay that died
                # without closing, so a dead session cannot be held forever.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(session.closed.wait(), timeout=90)
            finally:
                await session.shutdown()
                self._retiring.discard(session)

        task = asyncio.create_task(linger())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _after_close(self, session: RelaySession) -> float | None:
        code = session.close_code
        if session.revoked or code == 4403:
            self.revoked.set()
            return None
        self.announce(f"relay session closed code={code} reason={session.close_reason}")
        lived = time.monotonic() - session.ready_at if session.ready_at else 0.0
        if code in (1000, 1001) or lived >= STABLE_SESSION_SECONDS:
            # A normal close, or a session that served long enough to count as
            # healthy, ends any run of failures.
            self._unstable_closes = 0
        if code == 1000:
            return 0.0
        if code == 1001:
            return random.uniform(0, 1)
        # Consecutive short-lived abnormal closes grow the backoff. `run()`
        # resets its dial counter on every `ready`, so without this two
        # connectors superseding each other (4409) would re-take the tunnel
        # every 0-2 s until the relay's handshake bucket throttled both.
        self._unstable_closes += 1
        if code in (4400, 4408):
            # C1.6: back off and call /connect first.
            self.grant = RelayGrant(self.grant.url, self.grant.token, 0.0, self.grant.protocol)
        return backoff_delay(self._unstable_closes)

    async def _fresh_grant(self, *, force: bool = False) -> RelayGrant | None:
        if not force and self.grant.fresh():
            return self.grant
        refreshed = await self._refresh()
        if refreshed is None:
            self.announce("/connect stopped offering a relay; leaving relay mode")
            self.withdrawn.set()
            return None
        self.grant = refreshed
        return refreshed

    async def _establish(self, attempt: int) -> RelaySession | None:
        """One session to ``ready``, retrying per C1.1; ``None`` means "try later"."""
        retried_401 = False
        while not self.stopping:
            try:
                grant = await self._fresh_grant()
            except Exception as failure:  # noqa: BLE001 - a /connect blip is a retry
                self.announce(f"relay token refresh failed: {failure}")
                await asyncio.sleep(backoff_delay(attempt))
                return None
            if grant is None:
                return None
            session = RelaySession(self, grant)
            try:
                await session.open()
                return session
            except HandshakeRefused as refusal:
                self.announce(f"relay handshake refused: {refusal.status} {refusal.error}")
                if refusal.status == 401 and not retried_401:
                    # C1.6: a 401 at handshake always triggers one /connect and
                    # one retry.
                    retried_401 = True
                    self.grant = RelayGrant(grant.url, grant.token, 0.0, grant.protocol)
                    continue
                if refusal.status == 403:
                    self.revoked.set()
                    return None
                if refusal.status in (429, 503) and refusal.retry_after is not None:
                    await asyncio.sleep(refusal.retry_after)
                    return None
                await asyncio.sleep(backoff_delay(attempt))
                return None
            except (OSError, ConnectionError, TimeoutError) as failure:
                self.announce(f"relay connect failed: {type(failure).__name__}: {failure}")
                if session.close_code in (4400, 4408):
                    # C1.6: a 4400/4408 before `ready` (hello_mismatch: the token
                    # no longer matches this tunnel) is answered by /connect
                    # first, then backoff — never a retry with the same token.
                    self.grant = RelayGrant(grant.url, grant.token, 0.0, grant.protocol)
            except Exception as failure:  # noqa: BLE001 - websockets' own errors
                self.announce(f"relay connect failed: {type(failure).__name__}: {failure}")
            await asyncio.sleep(backoff_delay(attempt))
            return None
        return None
