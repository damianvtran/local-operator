"""A minimal in-process stand-in for agent-server's Go ``tunnel-relay``.

WHY IT EXISTS. The connector's E2E is meant to run against a locally built
``tunnel-relay`` binary (stream A2), but that service is built in parallel with
this one. This fake follows the frozen contract (agent-server
``docs/TUNNEL_RELAY.md`` C1) closely enough to drive the REAL connector and the
REAL gateway end to end: the C1.1 handshake and its refusals, the C1.4 control
stream (hello → ready, rotate/drain, retire, revoked), and one yamux stream per
public request. It is deliberately NOT a re-implementation of the relay's edge
gate: callers sign their own origin assertion, exactly as the relay would.

The relay side uses ``yamux.Session(client=True)``. Pairing our server with our
own client could hide a symmetric bug, which is why byte-level interop with
hashicorp/yamux is proven separately by the golden transcripts in
``test_tunnel_relay.py``; this fake proves the connector's behaviour.

Not a pytest module (no ``test_`` prefix); the E2E rig imports it too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from local_operator.tunnels.relay import yamux
from local_operator.tunnels.relay.connector import MAX_MESSAGE, PROTOCOL, control_line


@dataclass
class Claims:
    """What a connector JWT would carry (C2.3), keyed by the opaque token."""

    tunnel_id: str
    version: int
    revoked: bool = False


@dataclass
class FakeSession:
    ws: ServerConnection
    mux: yamux.Session
    control: yamux.Stream
    session_id: str
    epoch: int
    lines: list[dict[str, Any]] = field(default_factory=list)
    line_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    async def send_control(self, message: dict[str, Any]) -> None:
        await self.control.write(control_line(message))

    async def open(self) -> yamux.Stream:
        return await self.mux.open_stream()

    async def wait_line(self, kind: str, timeout: float = 10) -> dict[str, Any]:
        async def find() -> dict[str, Any]:
            while True:
                for line in self.lines:
                    if line.get("type") == kind:
                        return line
                self.line_arrived.clear()
                await self.line_arrived.wait()

        return await asyncio.wait_for(find(), timeout)


class FakeRelay:
    def __init__(self, tokens: dict[str, Claims]) -> None:
        self.tokens = tokens
        self.sessions: list[FakeSession] = []
        self.session_ready = asyncio.Event()
        self.refusals: list[tuple[int, str]] = []
        self.handshakes = 0
        self._epoch = 0
        self._server: Any = None
        self._public: asyncio.Server | None = None
        self.url = ""
        self.public_port = 0

    async def __aenter__(self) -> FakeRelay:
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            subprotocols=[PROTOCOL],  # type: ignore[list-item]
            process_request=self._check,
            max_size=MAX_MESSAGE,
            compression=None,
            ping_interval=None,
        )
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/connector"
        self._public = await asyncio.start_server(self._public_conn, "127.0.0.1", 0)
        self.public_port = self._public.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._public is not None:
            self._public.close()
        for session in self.sessions:
            with contextlib.suppress(Exception):
                await session.ws.close()
        self._server.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._server.wait_closed(), 5)

    def live(self) -> FakeSession:
        return [s for s in self.sessions if not s.closed.is_set()][-1]

    async def wait_sessions(self, count: int, timeout: float = 10) -> FakeSession:
        async def until() -> FakeSession:
            while len(self.sessions) < count:
                self.session_ready.clear()
                await self.session_ready.wait()
            return self.sessions[count - 1]

        return await asyncio.wait_for(until(), timeout)

    # -- handshake (C1.1) ------------------------------------------------

    def _refuse(self, connection: ServerConnection, status: int, error: str) -> Response:
        self.refusals.append((status, error))
        body = json.dumps({"error": error}).encode()
        return Response(
            status,
            "refused",
            Headers({"Content-Type": "application/json", "Content-Length": str(len(body))}),
            body,
        )

    def _check(self, connection: ServerConnection, request: Request) -> Response | None:
        self.handshakes += 1
        if PROTOCOL not in request.headers.get("Sec-WebSocket-Protocol", ""):
            return self._refuse(connection, 400, "unsupported_protocol")
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return self._refuse(connection, 401, "missing_token")
        claims = self.tokens.get(auth.removeprefix("Bearer "))
        if claims is None:
            return self._refuse(connection, 401, "invalid_token")
        if claims.revoked:
            return self._refuse(connection, 403, "revoked")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        claims = self.tokens[ws.request.headers["Authorization"].removeprefix("Bearer ")]
        mux = yamux.Session(ws.send, client=True)

        async def receive() -> None:
            with contextlib.suppress(Exception):
                async for message in ws:
                    assert isinstance(message, bytes)
                    await mux.feed(message)
            mux.close()

        pump = asyncio.create_task(receive())
        try:
            control = await asyncio.wait_for(mux.accept(), 10)
            assert control is not None and control.id == yamux.CONTROL_STREAM_ID
            buffer = b""
            while b"\n" not in buffer:
                chunk = await control.read()
                if not chunk:
                    return
                buffer += chunk
            line, _, buffer = buffer.partition(b"\n")
            hello = json.loads(line)
            if hello.get("protocol") != PROTOCOL:
                await self._close_error(ws, control, "unsupported_protocol")
                return
            if hello.get("tunnel_id") != claims.tunnel_id or hello.get("version") != (
                claims.version
            ):
                # The negative control: a token minted for another tunnel (or
                # version) never registers a session (C1.4 hello_mismatch).
                await self._close_error(ws, control, "hello_mismatch")
                return
            self._epoch += 1
            session = FakeSession(ws, mux, control, str(uuid.uuid4()), self._epoch)
            await control.write(
                control_line(
                    {
                        "type": "ready",
                        "session_id": session.session_id,
                        "epoch": session.epoch,
                        "rotate_after_s": 3000,
                        "drain_timeout_s": 60,
                    }
                )
            )
            self.sessions.append(session)
            self.session_ready.set()
            while True:
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    session.lines.append(json.loads(line))
                    session.line_arrived.set()
                chunk = await control.read()
                if not chunk:
                    break
                buffer += chunk
            await ws.wait_closed()
        except (yamux.StreamReset, ConnectionError):
            pass
        finally:
            for session in self.sessions:
                if session.ws is ws:
                    session.closed.set()
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

    async def _close_error(self, ws: ServerConnection, control: yamux.Stream, code: str) -> None:
        self.refusals.append((4400, code))
        await control.write(control_line({"type": "error", "code": code, "message": code}))
        await ws.close(4400, code)

    # -- public side: raw TCP in, one yamux stream per connection (C1.3) --

    async def _public_conn(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            stream = await self.live().open()
        except Exception:  # noqa: BLE001 - no session: the relay's 503 offline
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n")
            writer.close()
            return

        async def up() -> None:
            while chunk := await reader.read(65536):
                await stream.write(chunk)
            await stream.close()

        async def down() -> None:
            with contextlib.suppress(yamux.StreamReset):
                while chunk := await stream.read():
                    writer.write(chunk)
                    await writer.drain()
            writer.close()

        tasks = [asyncio.create_task(up()), asyncio.create_task(down())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        await asyncio.wait(tasks, timeout=30)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
