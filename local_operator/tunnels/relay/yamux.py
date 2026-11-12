"""The device half of yamux v0 (hashicorp/yamux ``spec.md``), asyncio-native.

WHY A HAND-WRITTEN YAMUX. The relay is Go and uses ``hashicorp/yamux`` as the
client; the device needs a *server* that interoperates with it byte for byte.
No maintained pure-Python yamux exists, and the protocol is small: four frame
types, four flags, per-stream credit windows. Writing it here keeps the
connector inside the ``lop`` wheel (it ships with every update, including the
remote-compute pod image) instead of adding a native dependency.

ROLES ARE FIXED BY THE CONTRACT (agent-server ``docs/TUNNEL_RELAY.md`` C1.2):
the relay opens odd stream ids, one per public request; the device opens
exactly one even stream, id 2, the control stream. An even SYN from the relay,
or a second device stream, is a protocol error.

TRANSPORT-AGNOSTIC. :class:`Session` never touches a socket: bytes arrive via
:meth:`Session.feed` and leave via the ``send`` coroutine it is given. WS
message boundaries carry no meaning (C1.2) — a frame may span messages and a
message may hold several — so ``feed`` reassembles one byte stream. That same
seam is what lets the golden-transcript tests replay the recorded relay bytes
and compare the emitted device bytes exactly.

FLOW CONTROL mirrors hashicorp/yamux so the Go side's accounting matches ours:
each side starts with an implicit 256 KiB receive window, and SYN/ACK advertise
``MAX_WINDOW - 256 KiB`` (786432) to reach 1 MiB. A reader re-grants credit with
hashicorp's own formula — ``MAX_WINDOW - buffered - remaining`` once that reaches
half the window — so nothing buffers more than 1 MiB per stream and a slow
gateway read is real backpressure all the way to the phone.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import Awaitable, Callable

HEADER = struct.Struct(">BBHII")
HEADER_SIZE = HEADER.size  # 12

VERSION = 0

TYPE_DATA = 0
TYPE_WINDOW_UPDATE = 1
TYPE_PING = 2
TYPE_GO_AWAY = 3

FLAG_SYN = 0x1
FLAG_ACK = 0x2
FLAG_FIN = 0x4
FLAG_RST = 0x8

GO_AWAY_NORMAL = 0
GO_AWAY_PROTOCOL_ERROR = 1
GO_AWAY_INTERNAL_ERROR = 2

#: yamux's implicit per-stream window before any WindowUpdate.
INITIAL_WINDOW = 256 * 1024
#: ``MaxStreamWindowSize`` on both sides (C1.2): SYN/ACK advertise the rest.
MAX_WINDOW = 1048576
#: C1.2 caps one Data frame the device SENDS at 256 KiB; with its 12-byte header
#: that also keeps every WS message far under the 1 MiB send limit.
MAX_DATA_FRAME = 262144
#: The device's only stream (C1.2): the device is the yamux server, so the
#: first even id is 2 and it never opens another.
CONTROL_STREAM_ID = 2
#: C1.2 requires at least 128 pending accepted streams; beyond it a SYN is RST.
#: The relay itself caps a session at 128 concurrent streams (C1.6), so this
#: bound is only reached by a misbehaving peer.
ACCEPT_BACKLOG = 256

Send = Callable[[bytes], Awaitable[None]]


class YamuxError(Exception):
    """The peer broke the protocol; the session must end with GoAway 1."""


class StreamReset(ConnectionError):
    """The stream was reset (RST) or its session ended under it."""


def encode(frame_type: int, flags: int, stream_id: int, length: int) -> bytes:
    return HEADER.pack(VERSION, frame_type, flags, stream_id, length)


class Stream:
    """One yamux stream: an ordered, flow-controlled, half-closable byte pipe."""

    def __init__(self, session: Session, stream_id: int) -> None:
        self.id = stream_id
        self._session = session
        self._buffer = bytearray()
        # Credit WE granted the peer that it has not used yet.
        self._recv_window = INITIAL_WINDOW
        # Credit the PEER granted us; SYN/ACK deltas raise it to 1 MiB.
        self._send_window = INITIAL_WINDOW
        self._readable = asyncio.Event()
        self._writable = asyncio.Event()
        self.remote_closed = False
        self.local_closed = False
        self.was_reset = False

    # -- reading ---------------------------------------------------------

    async def read(self, limit: int = MAX_DATA_FRAME) -> bytes:
        """Up to ``limit`` bytes; ``b""`` at the peer's FIN; raises on RST."""
        while not self._buffer:
            if self.was_reset:
                raise StreamReset(f"yamux stream {self.id} reset")
            if self.remote_closed:
                return b""
            self._readable.clear()
            await self._readable.wait()
        chunk = bytes(self._buffer[:limit])
        del self._buffer[:limit]
        await self._grant()
        return chunk

    async def _grant(self, flags: int = 0) -> None:
        """Return consumed credit to the peer, hashicorp's way.

        Batching to half a window (rather than granting per read) is what the
        Go side does too; granting less often than this would stall a 1 MiB
        burst, granting more often costs a frame per read for nothing.
        """
        if self.was_reset or self._session.closed:
            return
        delta = MAX_WINDOW - len(self._buffer) - self._recv_window
        if delta < MAX_WINDOW // 2 and not flags:
            return
        if self.remote_closed and not flags:
            return
        self._recv_window += delta
        await self._session._send_frame(TYPE_WINDOW_UPDATE, flags, self.id, delta)

    # -- writing ---------------------------------------------------------

    async def write(self, data: bytes) -> None:
        """Send ``data``, waiting for the peer's window; never over-sends."""
        view = memoryview(data)
        while view:
            while self._send_window <= 0:
                self._check_writable()
                self._writable.clear()
                await self._writable.wait()
            self._check_writable()
            size = min(len(view), self._send_window, MAX_DATA_FRAME)
            self._send_window -= size
            await self._session._send_frame(TYPE_DATA, 0, self.id, size, bytes(view[:size]))
            view = view[size:]

    def _check_writable(self) -> None:
        if self.was_reset or self._session.closed:
            raise StreamReset(f"yamux stream {self.id} reset")
        if self.local_closed:
            raise BrokenPipeError(f"yamux stream {self.id} is closed for writing")

    async def close(self) -> None:
        """Half-close: FIN means "no more data from this side" (C1.3 step 4)."""
        if self.local_closed or self.was_reset or self._session.closed:
            return
        self.local_closed = True
        await self._session._send_frame(TYPE_WINDOW_UPDATE, FLAG_FIN, self.id, 0)
        self._maybe_forget()

    async def reset(self) -> None:
        """Abort the stream with RST (C1.3 step 3: the gateway was unreachable)."""
        if self.was_reset:
            return
        self._mark_reset()
        if not self._session.closed:
            await self._session._send_frame(TYPE_WINDOW_UPDATE, FLAG_RST, self.id, 0)

    # -- driven by the session -------------------------------------------

    def _mark_reset(self) -> None:
        self.was_reset = True
        self._readable.set()
        self._writable.set()
        self._session._forget(self.id)

    def _maybe_forget(self) -> None:
        if self.local_closed and self.remote_closed:
            self._session._forget(self.id)

    def _on_data(self, payload: bytes) -> None:
        if len(payload) > self._recv_window:
            raise YamuxError(f"stream {self.id} exceeded its receive window")
        self._recv_window -= len(payload)
        if payload:
            self._buffer += payload
            self._readable.set()

    def _on_window(self, delta: int) -> None:
        if delta:
            self._send_window += delta
            self._writable.set()

    def _on_fin(self) -> None:
        self.remote_closed = True
        self._readable.set()
        self._maybe_forget()


class Session:
    """A yamux server session over an abstract byte transport.

    ``send`` must deliver bytes in order; it is always called under one lock,
    so a frame's header and payload are never interleaved with another frame.
    Accepted relay streams are delivered through :meth:`accept`.
    """

    def __init__(
        self, send: Send, *, accept_backlog: int = ACCEPT_BACKLOG, client: bool = False
    ) -> None:
        # ``client=True`` plays the RELAY's role (odd ids out, even ids in). The
        # product never sets it: it exists so the in-repo fake relay can drive
        # the real connector end to end where the Go relay binary is not built.
        # Interop with hashicorp/yamux is proven by the golden transcripts, not
        # by this symmetric pairing.
        self._client = client
        self._send = send
        self._write_lock = asyncio.Lock()
        self._inbound = bytearray()
        self._streams: dict[int, Stream] = {}
        self._accepts: asyncio.Queue[Stream | None] = asyncio.Queue()
        self._backlog = accept_backlog
        self._next_id = 1 if client else CONTROL_STREAM_ID
        self._next_ping = 0
        self._pings: dict[int, asyncio.Future[None]] = {}
        self.closed = False
        #: The relay's GoAway code once one arrived: no new streams follow it,
        #: and in-flight ones keep piping until the relay closes (C1.5).
        self.remote_go_away: int | None = None
        self.go_away = asyncio.Event()

    # -- public API ------------------------------------------------------

    async def open_stream(self) -> Stream:
        """Open a stream; the device opens exactly one, the control stream (C1.2)."""
        if self.closed:
            raise StreamReset("yamux session closed")
        stream = Stream(self, self._next_id)
        self._next_id += 2
        self._streams[stream.id] = stream
        await stream._grant(FLAG_SYN)
        return stream

    async def accept(self) -> Stream | None:
        """The next relay-opened stream, or ``None`` once the session is closed."""
        if self.closed and self._accepts.empty():
            return None
        return await self._accepts.get()

    async def ping(self) -> None:
        """Ping the relay and wait for its ACK (the device need not, C1.2)."""
        identifier = self._next_ping
        self._next_ping = (self._next_ping + 1) & 0xFFFFFFFF
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pings[identifier] = waiter
        try:
            await self._send_frame(TYPE_PING, FLAG_SYN, 0, identifier)
            await waiter
        finally:
            self._pings.pop(identifier, None)

    async def send_go_away(self, code: int = GO_AWAY_NORMAL) -> None:
        if not self.closed:
            await self._send_frame(TYPE_GO_AWAY, 0, 0, code)

    def close(self) -> None:
        """End the session locally: every stream is reset under its users."""
        if self.closed:
            return
        self.closed = True
        for stream in list(self._streams.values()):
            stream._mark_reset()
        self._streams.clear()
        for waiter in self._pings.values():
            if not waiter.done():
                waiter.set_exception(StreamReset("yamux session closed"))
        self._accepts.put_nowait(None)
        self.go_away.set()

    async def feed(self, data: bytes) -> None:
        """Consume transport bytes; raises :class:`YamuxError` on a protocol fault."""
        if self.closed:
            return
        self._inbound += data
        while len(self._inbound) >= HEADER_SIZE:
            version, frame_type, flags, stream_id, length = HEADER.unpack_from(self._inbound)
            if version != VERSION:
                raise YamuxError(f"unsupported yamux version {version}")
            if frame_type == TYPE_DATA:
                # A Data frame can never legally exceed one stream window, so
                # this bound is also what stops a hostile length from making
                # us buffer without limit while we wait for its body.
                if length > MAX_WINDOW:
                    raise YamuxError("yamux data frame exceeds the stream window")
                if len(self._inbound) < HEADER_SIZE + length:
                    return
                payload = bytes(self._inbound[HEADER_SIZE : HEADER_SIZE + length])
                del self._inbound[: HEADER_SIZE + length]
            else:
                payload = b""
                del self._inbound[:HEADER_SIZE]
            await self._dispatch(frame_type, flags, stream_id, length, payload)

    # -- internals -------------------------------------------------------

    async def _send_frame(
        self, frame_type: int, flags: int, stream_id: int, length: int, payload: bytes = b""
    ) -> None:
        async with self._write_lock:
            if self.closed:
                raise StreamReset("yamux session closed")
            await self._send(encode(frame_type, flags, stream_id, length) + payload)

    def _forget(self, stream_id: int) -> None:
        self._streams.pop(stream_id, None)

    async def _dispatch(
        self, frame_type: int, flags: int, stream_id: int, length: int, payload: bytes
    ) -> None:
        if frame_type == TYPE_PING:
            if flags & FLAG_SYN:
                await self._send_frame(TYPE_PING, FLAG_ACK, 0, length)
            elif flags & FLAG_ACK:
                waiter = self._pings.get(length)
                if waiter is not None and not waiter.done():
                    waiter.set_result(None)
            return
        if frame_type == TYPE_GO_AWAY:
            self.remote_go_away = length
            self.go_away.set()
            return
        if frame_type not in (TYPE_DATA, TYPE_WINDOW_UPDATE):
            raise YamuxError(f"unknown yamux frame type {frame_type}")
        stream = self._streams.get(stream_id)
        if flags & FLAG_SYN:
            stream = await self._incoming(stream_id)
            if stream is None:
                return
        elif stream is None:
            # A late frame for a stream already forgotten (both FINs seen, or
            # reset): hashicorp discards it too. Never an error.
            return
        if frame_type == TYPE_WINDOW_UPDATE:
            stream._on_window(length)
        else:
            stream._on_data(payload)
        if flags & FLAG_RST:
            stream._mark_reset()
        elif flags & FLAG_FIN:
            stream._on_fin()

    async def _incoming(self, stream_id: int) -> Stream | None:
        if stream_id % 2 == (1 if self._client else 0):
            raise YamuxError(f"peer opened stream {stream_id} of the wrong parity")
        if stream_id in self._streams:
            raise YamuxError(f"duplicate SYN for stream {stream_id}")
        stream = Stream(self, stream_id)
        self._streams[stream_id] = stream
        if self.remote_go_away is not None or self._accepts.qsize() >= self._backlog:
            await stream.reset()
            return None
        # ACK at once, before the gateway dial (C1.2): the relay's
        # StreamOpenTimeout must never depend on how fast loopback answers.
        await stream._grant(FLAG_ACK)
        self._accepts.put_nowait(stream)
        return stream


async def pump(session: Session, receive: Callable[[], Awaitable[bytes]]) -> None:
    """Feed ``receive()`` into ``session`` until the transport ends or it errors."""
    with contextlib.suppress(asyncio.CancelledError):
        while not session.closed:
            await session.feed(await receive())
