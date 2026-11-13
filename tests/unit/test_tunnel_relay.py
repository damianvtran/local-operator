"""The built-in relay connector against the frozen contract and its golden fixtures.

Contract: agent-server ``docs/TUNNEL_RELAY.md`` (C1, C2); fixtures vendored under
``tests/fixtures/tunnel_relay`` (see its PROVENANCE.md). Three layers:

1. **Golden transcripts** — the relay->device bytes real hashicorp/yamux v0.1.2
   produced are fed to our yamux server, and the bytes it emits must equal the
   recorded device->relay bytes exactly. This is the interop proof.
2. **Connector behaviour** against the in-process fake relay
   (`tunnel_relay_fake.py`): handshake refusals, hello_mismatch (the wrong-
   tunnel token negative control), rotation make-before-break, revocation, RST.
3. **End to end** through the REAL gateway to a real loopback harness: plain
   HTTP, SSE, a WebSocket, and a multi-MiB upload, all over one relay session.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import socket
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.tunnels import config
from local_operator.tunnels.relay import connector as relay_connector
from local_operator.tunnels.relay import yamux
from local_operator.tunnels.relay.connector import (
    RelayConnector,
    RelayGrant,
    hello_line,
    parse_grant,
    retire_line,
)
from local_operator.tunnels.service import CloudflaredOverlap
from tests.unit.tunnel_relay_fake import Claims, FakeRelay

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tunnel_relay"
TUNNEL = "0123456789abcdef0123456789abcdef"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def frame(entry: dict[str, Any]) -> bytes:
    return bytes.fromhex(entry["header_hex"]) + bytes.fromhex(entry.get("payload_hex", ""))


def payload(entry: dict[str, Any]) -> bytes:
    return bytes.fromhex(entry.get("payload_hex", ""))


class Wire:
    """The device's outbound byte stream, as the WebSocket would carry it."""

    def __init__(self) -> None:
        self.out = bytearray()

    async def send(self, data: bytes) -> None:
        self.out += data


def split(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = fixture(f"yamux/{name}.json")["frames"]
    return (
        [f for f in frames if f["dir"] == "relay->device"],
        [f for f in frames if f["dir"] == "device->relay"],
    )


async def handshake(relay: list[dict[str, Any]], device: list[dict[str, Any]], *, bytewise=False):
    """Frames 0..3 of every transcript: control SYN + hello, relay ACK + ready."""
    wire = Wire()
    session = yamux.Session(wire.send)
    control = await session.open_stream()
    await control.write(payload(device[1]))
    inbound = frame(relay[0]) + frame(relay[1])
    if bytewise:
        # C1.2: WS message boundaries carry no meaning; one byte per message is
        # the harshest legal split.
        for index in range(len(inbound)):
            await session.feed(inbound[index : index + 1])
    else:
        await session.feed(inbound)
    assert await control.read() == payload(relay[1])
    return wire, session, control


# -- 1. golden transcripts ---------------------------------------------------


def test_the_hello_and_retire_emitters_match_the_contract_bytes():
    lines = {
        (v["sender"], v["type"]): v["line"].encode()
        for v in fixture("control/messages.json")["valid"]
    }
    assert (
        hello_line(connector="local-operator/2026.11.10", tunnel_id=TUNNEL, version=7)
        == lines[("device", "hello")]
    )
    assert retire_line("c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f") == lines[("device", "retire")]
    _, device = split("01_control_hello_and_http_exchange")
    assert payload(device[1]) == lines[("device", "hello")]


@pytest.mark.asyncio
@pytest.mark.parametrize("bytewise", [False, True])
async def test_transcript_01_control_hello_and_http_exchange(bytewise):
    relay, device = split("01_control_hello_and_http_exchange")
    wire, session, _ = await handshake(relay, device, bytewise=bytewise)
    await session.feed(frame(relay[2]) + frame(relay[3]))
    stream = await session.accept()
    assert stream is not None and stream.id == 1
    assert await stream.read() == payload(relay[3])
    await stream.write(payload(device[3]))
    await session.feed(frame(relay[4]))
    assert await stream.read() == b""
    await stream.close()
    assert bytes(wire.out) == b"".join(frame(f) for f in device)


@pytest.mark.asyncio
async def test_transcript_02_ping_both_ways():
    relay, device = split("02_ping")
    wire, session, _ = await handshake(relay, device)
    await session.feed(frame(relay[2]))  # relay Ping SYN -> device answers ACK
    ping = asyncio.create_task(session.ping())
    await asyncio.sleep(0)
    await session.feed(frame(relay[3]))  # relay ACKs the device's Ping SYN
    await asyncio.wait_for(ping, 1)
    assert bytes(wire.out) == b"".join(frame(f) for f in device)


@pytest.mark.asyncio
async def test_transcript_03_device_rst_on_gateway_refused():
    relay, device = split("03_device_rst_on_gateway_refused")
    wire, session, _ = await handshake(relay, device)
    for entry in relay[2:]:
        await session.feed(frame(entry))
    stream = await session.accept()
    assert stream is not None
    # The real code path: the gateway port refuses, the pipe resets the stream.
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    await relay_connector.pipe_to_gateway(stream, "127.0.0.1", dead_port)
    assert bytes(wire.out) == b"".join(frame(f) for f in device)


@pytest.mark.asyncio
async def test_transcript_04_relay_goaway_normal():
    relay, device = split("04_relay_goaway_normal")
    wire, session, _ = await handshake(relay, device)
    await session.feed(frame(relay[2]))
    assert session.go_away.is_set() and session.remote_go_away == yamux.GO_AWAY_NORMAL
    assert bytes(wire.out) == b"".join(frame(f) for f in device)
    # C1.5: after GoAway no new stream is served; a late SYN is reset.
    await session.feed(yamux.encode(yamux.TYPE_WINDOW_UPDATE, yamux.FLAG_SYN, 3, 786432))
    assert wire.out.endswith(yamux.encode(yamux.TYPE_WINDOW_UPDATE, yamux.FLAG_RST, 3, 0))


@pytest.mark.asyncio
async def test_transcript_05_flow_control_invariants():
    spec = fixture("yamux/05_flow_control_window_update.json")
    relay, device = split("05_flow_control_window_update")
    wire, session, _ = await handshake(relay, device)
    assert bytes(wire.out) == frame(device[0]) + frame(device[1])
    await session.feed(frame(relay[2]))  # SYN stream 1, delta 786432
    stream = await session.accept()
    assert stream is not None
    assert wire.out.endswith(frame(device[2]))  # ACK 786432

    total = 1572864
    body = bytes(i % 251 for i in range(total))
    recorded = [f for f in relay if f["header"]["type"] == "Data" and f["header"]["stream_id"] == 1]
    # The generator matches the recording: each recorded payload's sha256 is
    # the hash of the corresponding slice of `byte i == i mod 251`.
    offset = 0
    for entry in recorded:
        size = entry["header"]["length"]
        assert hashlib.sha256(body[offset : offset + size]).hexdigest() == entry["payload_sha256"]
        offset += size
    assert offset == total

    # The full 1 MiB window in ONE Data frame (contract round 1: receive
    # buffers must take a 1,048,576-byte payload).
    first = 1048576
    await session.feed(yamux.encode(yamux.TYPE_DATA, 0, 1, first) + body[:first])
    mark = len(wire.out)
    received = bytearray()
    granted = 0
    consumed_since_grant = 0
    while len(received) < first:
        chunk = await stream.read()
        received += chunk
        consumed_since_grant += len(chunk)
        updates = wire.out[mark:]
        mark = len(wire.out)
        if updates:
            assert len(updates) == yamux.HEADER_SIZE
            _, kind, flags, sid, delta = yamux.HEADER.unpack(bytes(updates))
            assert (kind, flags, sid) == (yamux.TYPE_WINDOW_UPDATE, 0, 1)
            assert consumed_since_grant >= 524288  # spec invariant 5
            granted += delta
            consumed_since_grant = 0
    # The relay may send no more than the 1 MiB window plus every grant.
    assert granted >= total - first
    await session.feed(yamux.encode(yamux.TYPE_DATA, 0, 1, total - first) + body[first:])
    while len(received) < total:
        received += await stream.read()
    assert bytes(received) == body
    assert "frames[0..3] are byte-exact (control handshake)" in spec["invariants"]


@pytest.mark.asyncio
async def test_a_frame_beyond_the_granted_window_is_a_protocol_error():
    relay, device = split("01_control_hello_and_http_exchange")
    _, session, _ = await handshake(relay, device)
    await session.feed(frame(relay[2]))
    with pytest.raises(yamux.YamuxError):
        await session.feed(yamux.encode(yamux.TYPE_DATA, 0, 1, 1048577) + b"x" * 1048577)


@pytest.mark.asyncio
async def test_an_even_stream_from_the_relay_is_a_protocol_error():
    relay, device = split("01_control_hello_and_http_exchange")
    _, session, _ = await handshake(relay, device)
    with pytest.raises(yamux.YamuxError):
        await session.feed(yamux.encode(yamux.TYPE_WINDOW_UPDATE, yamux.FLAG_SYN, 4, 786432))


# -- C2: the relay grant ------------------------------------------------------


def test_the_connect_relay_object_parses_and_absence_means_no_relay():
    result = fixture("connect/connect_response.json")["result"]
    grant = parse_grant(result["relay"])
    assert grant is not None and grant.url == relay_connector.RELAY_URL
    # The fixture's expires_at is in the past (2025), so the token is not reused.
    assert not grant.fresh()
    run_config = fixture("connect/run_tunnel_config.json")
    run_relay = run_config.get("relay") or run_config.get("result", {}).get("relay")
    assert parse_grant(run_relay) is not None
    assert parse_grant(None) is None
    assert parse_grant({**result["relay"], "protocol": "radient-tunnel.v2"}) is None
    # Production pins the URL: a control plane naming another host is ignored
    # and the device keeps cloudflared, rather than sending public traffic to it.
    assert parse_grant({**result["relay"], "url": "wss://evil.example/v1/connector"}) is None


def test_the_overlap_keeps_cloudflared_until_thirty_minutes_after_the_flip():
    overlap = CloudflaredOverlap()
    assert overlap.keep("cloudflare", 0)
    assert overlap.keep("relay", 100)  # first sight of the flip
    assert overlap.keep("relay", 100 + 29 * 60)
    assert not overlap.keep("relay", 100 + 30 * 60)
    # An unflip brings cloudflared back and restarts the clock.
    assert overlap.keep("cloudflare", 100 + 31 * 60)
    assert overlap.keep("relay", 5000) and overlap.keep("relay", 5000 + 29 * 60)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("", None),
        ("values:\n  tunnel.transport: cloudflare\n", "cloudflare"),
        ("values:\n  tunnel:\n    transport: Cloudflare\n", "cloudflare"),
        ("values:\n  tunnel.transport: relay\n", None),
        ("values:\n  tunnel.transport: bogus\n", None),
    ],
)
def test_the_kill_switch_reads_config_yml(tmp_path, monkeypatch, body, expected):
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    if body:
        (tmp_path / "config.yml").write_text(body)
    assert config.transport_override() == expected
    config.record_relay_offer(True)
    assert config.relay_offered() is (expected is None)
    config.record_relay_offer(False)
    assert config.relay_offered() is False


# -- 2. connector behaviour against the fake relay ----------------------------


def grant_for(relay: FakeRelay, token: str, *, expires_in: float = 600) -> RelayGrant:
    return RelayGrant(url=relay.url, token=token, expires_at=time.time() + expires_in)


async def echo_gateway() -> tuple[asyncio.Server, int]:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def make_connector(relay, grant, port, *, refresh=None, tunnel=TUNNEL, version=7, log=None):
    async def no_refresh():
        return grant

    return RelayConnector(
        grant,
        tunnel_id=tunnel,
        version=version,
        gateway_port=port,
        refresh=refresh or no_refresh,
        name="local-operator/test",
        announce=(log.append if log is not None else (lambda _m: None)),
    )


async def until_connected(connector: RelayConnector) -> None:
    """``ready`` reaches the device a moment after the fake registers the session."""
    for _ in range(500):
        if connector.connected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("connector never reached ready")


@contextlib.asynccontextmanager
async def running(connector: RelayConnector):
    task = asyncio.create_task(connector.run())
    try:
        yield task
    finally:
        await connector.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_relay_stream_is_piped_unparsed_to_the_gateway():
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), port)
        async with running(connector):
            session = await relay.wait_sessions(1)
            await until_connected(connector)
            stream = await session.open()
            await stream.write(b"\x00not http at all\xff")
            assert await stream.read() == b"\x00not http at all\xff"
            await stream.close()
            assert await stream.read() == b""


@pytest.mark.asyncio
async def test_a_token_for_another_tunnel_is_refused_and_never_connects():
    """Negative control: the relay refuses a hello that disagrees with the token."""
    server, port = await echo_gateway()
    log: list[str] = []
    refreshed: list[str] = []
    async with server, FakeRelay({"other": Claims("f" * 32, 7)}) as relay:

        async def refresh():
            # C1.6: after a 4400 the connector must call /connect before it
            # dials again; this /connect still hands out the wrong token.
            refreshed.append("connect")
            return grant_for(relay, "other")

        connector = make_connector(relay, grant_for(relay, "other"), port, log=log, refresh=refresh)
        async with running(connector):
            for _ in range(100):
                if relay.refusals:
                    break
                await asyncio.sleep(0.02)
            assert relay.refusals[0] == (4400, "hello_mismatch")
            assert not connector.connected
            assert relay.sessions == []
            for _ in range(200):
                if refreshed:
                    break
                await asyncio.sleep(0.02)
            assert refreshed, "a 4400 before ready must trigger /connect first"
    assert any("hello_mismatch" in line for line in log)


@pytest.mark.asyncio
async def test_rotation_is_make_before_break_with_retire():
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), port)
        async with running(connector):
            first = await relay.wait_sessions(1)
            inflight = await first.open()
            await first.send_control({"type": "rotate", "reason": "lifetime", "deadline_s": 60})
            second = await relay.wait_sessions(2)
            retire = await first.wait_line("retire")
            assert retire["successor_session_id"] == second.session_id
            # The old session's in-flight stream keeps piping after retire (C1.5).
            await inflight.write(b"still here")
            assert await inflight.read() == b"still here"
            assert second.epoch == first.epoch + 1
            await first.ws.close(1000, "rotated")
            await asyncio.sleep(0.05)
            assert connector.connected
            fresh = await second.open()
            await fresh.write(b"new")
            assert await fresh.read() == b"new"


@pytest.mark.asyncio
async def test_a_revoked_token_stops_the_connector():
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7, revoked=True)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), port)
        async with running(connector) as task:
            await asyncio.wait_for(task, 5)
            assert connector.revoked.is_set()
            assert relay.refusals == [(403, "revoked")]


@pytest.mark.asyncio
async def test_a_401_refreshes_the_token_once_then_connects():
    server, port = await echo_gateway()
    calls = 0
    async with server, FakeRelay({"good": Claims(TUNNEL, 7)}) as relay:

        async def refresh():
            nonlocal calls
            calls += 1
            return grant_for(relay, "good")

        connector = make_connector(relay, grant_for(relay, "stale"), port, refresh=refresh)
        async with running(connector):
            await relay.wait_sessions(1)
    assert calls == 1
    assert relay.refusals == [(401, "invalid_token")]


@pytest.mark.asyncio
async def test_relay_revoked_line_and_4403_stop_the_connector():
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), port)
        async with running(connector) as task:
            session = await relay.wait_sessions(1)
            await session.send_control(
                {"type": "future_thing", "x": 1}  # C1.4: unknown types are ignored
            )
            await session.send_control({"type": "revoked", "reason": "suspended", "version": 7})
            await session.mux.send_go_away()
            await session.ws.close(4403, "revoked")
            await asyncio.wait_for(task, 5)
            assert connector.revoked.is_set()


@pytest.mark.asyncio
async def test_an_unreachable_gateway_resets_the_public_stream():
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    async with FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), dead)
        async with running(connector):
            session = await relay.wait_sessions(1)
            stream = await session.open()
            await stream.write(b"GET / HTTP/1.1\r\n\r\n")
            with pytest.raises(yamux.StreamReset):
                await asyncio.wait_for(stream.read(), 5)
            assert connector.connected  # one refused request never ends the session


# -- 3. end to end through the real gateway -----------------------------------


def bound_socket() -> tuple[socket.socket, int]:
    """A loopback socket already bound to an ephemeral port, and that port.

    The servers in this section are handed the socket itself (uvicorn's
    ``serve(sockets=[...])``) instead of a port number. Picking a port by
    binding to 0, closing the probe and letting uvicorn bind the number later
    left a window in which another xdist worker could take the port: main
    run 43143140065 died in uvicorn's startup with ``[Errno 98] address
    already in use`` and ``SystemExit: 3``. Holding the bound socket from pick
    to serve closes that window instead of retrying around it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
    except BaseException:
        sock.close()
        raise
    return sock, sock.getsockname()[1]


def harness_app():
    """A loopback harness speaking the three shapes the phone needs."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket

    async def events(_request: Request):
        async def stream():
            for index in range(3):
                yield f"data: tick {index}\n\n".encode()
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def upload(request: Request):
        body = await request.body()
        return JSONResponse({"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})

    async def echo(socket: WebSocket):
        await socket.accept()
        while True:
            message = await socket.receive_text()
            await socket.send_text("echo:" + message)

    return Starlette(
        routes=[
            Route("/api/events", events),
            Route("/api/upload", upload, methods=["POST"]),
            WebSocketRoute("/api/ws", echo),
        ]
    )


@contextlib.asynccontextmanager
async def serving(app, sock: socket.socket):
    import uvicorn

    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    # The already-bound socket, never the bare port: see ``bound_socket``.
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        try:
            await task
        finally:
            # uvicorn closes it on a clean shutdown; this covers the rest.
            sock.close()


@pytest.mark.asyncio
async def test_end_to_end_http_sse_websocket_and_upload_through_the_real_gateway():
    from websockets.asyncio.client import connect
    from websockets.typing import Origin

    from local_operator.tunnels.gateway import Gateway
    from tests.unit.test_tunnels import (
        HOST,
        PROOF_HEADER,
        make_connection,
        make_signing_key,
        proof,
    )

    key = make_signing_key()
    connection = make_connection(key)
    gateway_sock, gateway_port = bound_socket()
    harness_sock, harness_port = bound_socket()
    connection["gateway_port"] = connection["tunnel"]["gateway_port"] = gateway_port
    connection["tunnel"]["harnesses"][0]["port"] = harness_port
    tunnel_id, version = connection["tunnel"]["id"], connection["tunnel"]["version"]

    async with (
        httpx.AsyncClient(trust_env=False) as upstream,
        FakeRelay({"tok": Claims(tunnel_id, version)}) as relay,
    ):
        connector = make_connector(
            relay, grant_for(relay, "tok"), gateway_port, tunnel=tunnel_id, version=version
        )
        gateway = Gateway(
            connection,
            upstream,
            mobile_password="private-local-password",
            connector_ready=lambda: connector.connected,
            connector_transport=lambda: "relay",
            cloudflared_running=lambda: False,
        )
        async with (
            serving(harness_app(), harness_sock),
            serving(gateway.app(), gateway_sock),
            running(connector),
        ):
            await relay.wait_sessions(1)
            await until_connected(connector)
            public = f"http://127.0.0.1:{relay.public_port}"
            async with httpx.AsyncClient(trust_env=False, timeout=30) as phone:
                health = (
                    await phone.get(f"http://127.0.0.1:{gateway_port}/_lop_tunnel/health")
                ).json()
                assert health["connected"] is True and health["transport"] == "relay"

                events = []
                async with phone.stream(
                    "GET",
                    public + "/api/events",
                    headers={"host": HOST, PROOF_HEADER: proof(key, target="/api/events")},
                ) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            events.append((line, time.monotonic()))
                assert [e[0] for e in events] == [f"data: tick {i}" for i in range(3)]
                # Streamed, not buffered: the ticks arrive spread over time.
                assert events[-1][1] - events[0][1] >= 0.05

                body = bytes(i % 251 for i in range(8 * 1024 * 1024))
                uploaded = await phone.post(
                    public + "/api/upload",
                    content=body,
                    headers={
                        "host": HOST,
                        "origin": "https://" + HOST,
                        PROOF_HEADER: proof(key, method="POST", target="/api/upload", body=body),
                    },
                )
                assert uploaded.status_code == 200, uploaded.text
                assert uploaded.json() == {
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }

            async with connect(
                f"ws://{HOST}/api/ws",
                host="127.0.0.1",
                port=relay.public_port,
                proxy=None,
                origin=Origin("https://" + HOST),
                additional_headers={PROOF_HEADER: proof(key, target="/api/ws")},
            ) as ws:
                await ws.send("hello")
                assert await asyncio.wait_for(ws.recv(), 10) == "echo:hello"


# -- 4. the tunnel service in relay mode --------------------------------------


def _relay_service(tmp_path, monkeypatch, relay: FakeRelay, *, binary: bool, offer: bool = True):
    """`service.run()` wired to the fake relay, a mock cloud and a stub cloudflared.

    Reuses `test_tunnels`' service fixture so the stored record, port pin and
    mobile password are exactly what the existing service tests use. The relay
    URL pin is relaxed ONLY here (the fake listens on ws://127.0.0.1); the pin
    itself is covered by the parse test above.
    """
    from local_operator.tunnels.errors import LocalPrerequisite
    from tests.unit.test_tunnels import (
        _Connector,
        _service_fixture,
        _synthetic_port,
        make_connection,
        make_signing_key,
    )

    harness = _synthetic_port()
    connection = make_connection(make_signing_key())
    service, served, api = _service_fixture(
        tmp_path, monkeypatch, connection, harness, pinned_port=harness
    )
    if offer:
        served["relay"] = {
            "url": relay.url,
            "token": "tok",
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 600)),
            "protocol": "radient-tunnel.v1",
        }
    relay.tokens["tok"] = Claims(served["tunnel"]["id"], served["tunnel"]["version"])
    record = dict(served["tunnel"])

    async def control_plane(method, path, **kwargs):
        return served if method == "POST" else record

    api.request.side_effect = control_plane
    monkeypatch.setattr(service, "parse_grant", lambda v: parse_grant(v, pinned_url=None))
    monkeypatch.setattr(service, "POLL_SECONDS", 0.02)
    if not binary:

        def missing(*_):
            raise LocalPrerequisite("Install cloudflared 2025.4.0 or newer")

        monkeypatch.setattr(service, "cloudflared_binary", missing)
    children: list[Any] = []

    async def launched(*args, **kwargs):
        children.append(_Connector())
        return children[-1]

    monkeypatch.setattr(service.asyncio, "create_subprocess_exec", launched)
    return service, served, record, children


async def _health(port: int) -> dict[str, Any]:
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(300):
            with contextlib.suppress(httpx.HTTPError):
                body = (await client.get(f"http://127.0.0.1:{port}/_lop_tunnel/health")).json()
                if body.get("connected"):
                    return body
            await asyncio.sleep(0.02)
    raise AssertionError("gateway never reported connected")


async def _stop(task: asyncio.Task[int]) -> int:
    stopped = config.load()
    stopped["stopped"] = True
    config.save(stopped)
    return await asyncio.wait_for(task, timeout=15)


@pytest.mark.asyncio
async def test_relay_offered_without_cloudflared_runs_relay_only(tmp_path, monkeypatch):
    async with FakeRelay({}) as relay:
        service, served, _, children = _relay_service(tmp_path, monkeypatch, relay, binary=False)
        task = asyncio.create_task(service.run())
        health = await _health(served["gateway_port"])
        assert health["transport"] == "relay" and health["cloudflared"] is False
        assert children == []  # the preflight was optional: nothing spawned
        assert config.relay_offered()
        assert await _stop(task) == 0


@pytest.mark.asyncio
async def test_dual_run_then_cloudflared_stops_after_the_overlap(tmp_path, monkeypatch):
    from local_operator.tunnels import service as service_module

    monkeypatch.setattr(service_module, "RELAY_OVERLAP_SECONDS", 0.2)
    async with FakeRelay({}) as relay:
        service, served, record, children = _relay_service(
            tmp_path, monkeypatch, relay, binary=True
        )
        task = asyncio.create_task(service.run())
        await _health(served["gateway_port"])
        await relay.wait_sessions(1)
        assert len(children) == 1 and children[0].returncode is None  # dual-run
        record["transport"] = "relay"  # the flip reconcile ran (C2.2)
        for _ in range(200):
            if children[0].returncode is not None:
                break
            await asyncio.sleep(0.02)
        assert children[0].returncode == 0, "cloudflared outlived the overlap"
        assert not task.done()  # cloudflared stopping never ends the unit
        record["transport"] = "cloudflare"  # unflip: Cloudflare routes again
        for _ in range(200):
            if len(children) == 2:
                break
            await asyncio.sleep(0.02)
        assert len(children) == 2 and children[1].returncode is None
        assert await _stop(task) == 0


@pytest.mark.asyncio
async def test_the_kill_switch_keeps_the_cloudflared_only_path(tmp_path, monkeypatch):
    async with FakeRelay({}) as relay:
        service, served, _, children = _relay_service(tmp_path, monkeypatch, relay, binary=True)
        (tmp_path / "config.yml").write_text("values:\n  tunnel.transport: cloudflare\n")
        task = asyncio.create_task(service.run())
        for _ in range(200):
            if children:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        assert len(children) == 1
        assert relay.handshakes == 0  # the relay was never dialled
        async with httpx.AsyncClient(trust_env=False) as client:
            body = (
                await client.get(f"http://127.0.0.1:{served['gateway_port']}/_lop_tunnel/health")
            ).json()
        assert body["transport"] == "cloudflare"
        assert not config.relay_offered()  # the switch outranks the marker
        assert await _stop(task) == 0


@pytest.mark.asyncio
async def test_an_old_server_without_relay_still_requires_cloudflared(tmp_path, monkeypatch):
    from local_operator.tunnels.errors import LocalPrerequisite

    async with FakeRelay({}) as relay:
        service, _, _, children = _relay_service(
            tmp_path, monkeypatch, relay, binary=False, offer=False
        )
        with pytest.raises(LocalPrerequisite, match="cloudflared"):
            await service.run()
        assert children == [] and relay.handshakes == 0
        assert not config.relay_offered()


@pytest.mark.parametrize(
    "connector, expected",
    [
        (
            {"transport": "relay", "cloudflared": True},
            "Transport: relay (cloudflared still running)",
        ),
        ({"transport": "relay", "cloudflared": False}, "Transport: relay"),
        ({"transport": "cloudflare", "cloudflared": True}, "Transport: cloudflare"),
        ({}, None),  # an older gateway names no transport: print nothing
    ],
)
def test_status_names_the_transport(connector, expected):
    from local_operator.tunnels import cli

    payload = {
        "connector": {"state": "connected", "reason": "", "detail": "", **connector},
        "login": {"state": "ok"},
        "cloud": {"status": "active", "source": "live", "reason": ""},
        "remedy": None,
    }
    lines = [
        line
        for line in cli._status_text(payload, {"status": "active"}, "live").splitlines()
        if line.startswith("Transport:")
    ]
    assert lines == ([expected] if expected else [])


# -- round 1 remediation (PR #3765) --------------------------------------------


def test_the_connector_token_never_appears_in_a_grant_repr():
    """m1: a traceback with locals or a stray f"{grant}" must not print the token."""
    grant = RelayGrant("wss://example.invalid/v1/connector", "s3cret-connector-token", 0.0)
    assert "s3cret-connector-token" not in repr(grant)


@pytest.mark.asyncio
async def test_a_failed_successor_dial_is_retried_before_the_deadline():
    """m3: one refused successor dial must not wait out S1's deadline."""
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), port)
        async with running(connector):
            first = await relay.wait_sessions(1)
            relay.refuse_next.append((503, "draining"))
            await first.send_control({"type": "drain", "reason": "shutdown", "deadline_s": 60})
            second = await relay.wait_sessions(2)
            retire = await first.wait_line("retire")
            assert retire["successor_session_id"] == second.session_id
            assert (503, "draining") in relay.refusals
            assert not first.closed.is_set()  # made before S1 was broken


def test_repeated_supersedes_grow_the_backoff(monkeypatch):
    """m2: consecutive short-lived 4409s back off further each time."""
    seen: list[int] = []
    monkeypatch.setattr(relay_connector, "backoff_delay", lambda n: seen.append(n) or 0.0)

    async def no_refresh():
        return None

    connector = RelayConnector(
        RelayGrant("wss://x.invalid/v1/connector", "t", time.time() + 600),
        tunnel_id=TUNNEL,
        version=7,
        gateway_port=1,
        refresh=no_refresh,
        name="local-operator/test",
    )

    class Closed:
        revoked = False
        close_reason = "superseded"

        def __init__(self, code: int, lived: float) -> None:
            self.close_code = code
            self.ready_at = time.monotonic() - lived

    for _ in range(3):
        connector._after_close(Closed(4409, 1.0))  # type: ignore[arg-type]
    assert seen == [1, 2, 3]
    # A session that stayed up past the stable threshold resets the run.
    connector._after_close(
        Closed(4409, relay_connector.STABLE_SESSION_SECONDS + 1)  # type: ignore[arg-type]
    )
    assert seen[-1] == 1


@pytest.mark.asyncio
async def test_a_stale_cloudflared_path_still_runs_relay_only(tmp_path, monkeypatch):
    """M1: a saved cloudflared_path that no longer exists must not crash-loop the unit."""
    from local_operator.tunnels import service as service_module

    real_binary = service_module.cloudflared_binary
    async with FakeRelay({}) as relay:
        service, served, _, children = _relay_service(tmp_path, monkeypatch, relay, binary=True)
        monkeypatch.setattr(service, "cloudflared_binary", real_binary)
        stored = config.load()
        stored["cloudflared_path"] = str(tmp_path / "removed" / "cloudflared")
        config.save(stored)
        task = asyncio.create_task(service.run())
        health = await _health(served["gateway_port"])
        assert health["transport"] == "relay" and health["cloudflared"] is False
        assert children == []
        assert await _stop(task) == 0


def test_a_missing_or_hung_cloudflared_probe_is_a_local_prerequisite(tmp_path, monkeypatch):
    """M1 at the source: OSError/SubprocessError become LocalPrerequisite."""
    import subprocess

    from local_operator.tunnels import service as service_module
    from local_operator.tunnels.errors import LocalPrerequisite

    with pytest.raises(LocalPrerequisite, match="cloudflared"):
        service_module.cloudflared_binary(str(tmp_path / "gone"))

    def hung(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)

    monkeypatch.setattr(service_module.subprocess, "run", hung)
    with pytest.raises(LocalPrerequisite, match="cloudflared"):
        service_module.cloudflared_binary("/synthetic/cloudflared")


@pytest.mark.asyncio
async def test_a_crashed_relay_connector_is_logged_before_the_restart(
    tmp_path, monkeypatch, capsys
):
    """m4: a connector bug must not be a silent restart loop."""
    async with FakeRelay({}) as relay:
        service, _, _, _ = _relay_service(tmp_path, monkeypatch, relay, binary=False)

        async def boom(self):
            raise RuntimeError("connector bug")

        monkeypatch.setattr(service.RelayConnector, "run", boom)
        assert await asyncio.wait_for(service.run(), timeout=15) == 1
    out = capsys.readouterr().out
    assert "relay connector crashed: RuntimeError; restarting" in out
    assert "connector bug" not in out


# -- 5. the gateway's keep-alive outlasts the relay's idle pool ----------------


async def _read_response(stream: yamux.Stream) -> bytes:
    """One Content-Length HTTP/1.1 response off a relay stream, as the relay reads it."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await asyncio.wait_for(stream.read(), 10)
        if not chunk:
            raise ConnectionError(f"gateway closed the stream before a response: {data!r}")
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = next(
        int(line.split(b":", 1)[1])
        for line in head.split(b"\r\n")
        if line.lower().startswith(b"content-length:")
    )
    while len(body) < length:
        chunk = await asyncio.wait_for(stream.read(), 10)
        if not chunk:
            raise ConnectionError("gateway closed the stream mid-body")
        body += chunk
    return head.split(b"\r\n", 1)[0]


def test_the_gateway_keep_alive_outlasts_every_client_idle_pool():
    """C1.3 + cloudflared: the client, never the gateway, closes an idle connection."""
    from local_operator.tunnels import service

    assert service.GATEWAY_KEEP_ALIVE_SECONDS > relay_connector.RELAY_IDLE_CONN_TIMEOUT
    assert service.GATEWAY_KEEP_ALIVE_SECONDS > service.CLOUDFLARED_IDLE_CONN_TIMEOUT
    built = service.gateway_server_config(object(), 4100)
    assert built.timeout_keep_alive == service.GATEWAY_KEEP_ALIVE_SECONDS
    assert (built.host, built.port, built.access_log) == ("127.0.0.1", 4100, False)


@pytest.mark.asyncio
async def test_a_pooled_relay_stream_still_serves_after_uvicorns_default_keep_alive():
    """Regression: 502 relay_upstream_reset bursts on a loaded machine (2026-11-11).

    The relay reuses an idle stream for up to 30 s (C1.3). The gateway used
    uvicorn's 5 s keep-alive, so after 5 s idle it closed the loopback
    connection under a stream the relay still pooled; a POST sent on it raced
    that close and the connector reset the stream. Here the second request
    goes out on the SAME stream after 5.5 s idle, through the gateway's real
    server configuration, and must be answered.
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from local_operator.tunnels.service import gateway_server_config

    async def echo(request: Request):
        return PlainTextResponse(f"{request.method} {len(await request.body())}")

    app = Starlette(routes=[Route("/api/echo", echo, methods=["GET", "POST"])])
    sock, port = bound_socket()
    server = uvicorn.Server(gateway_server_config(app, port))
    # The already-bound socket, never the bare port: see ``bound_socket``.
    serve = asyncio.create_task(server.serve(sockets=[sock]))
    log: list[str] = []
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        async with FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
            connector = make_connector(relay, grant_for(relay, "tok"), port, log=log)
            async with running(connector):
                session = await relay.wait_sessions(1)
                stream = await session.open()
                await stream.write(b"GET /api/echo HTTP/1.1\r\nHost: t\r\n\r\n")
                assert await _read_response(stream) == b"HTTP/1.1 200 OK"
                # Longer than uvicorn's default keep-alive (5 s), well inside
                # the relay's 30 s idle pool.
                await asyncio.sleep(5.5)
                await stream.write(
                    b"POST /api/echo HTTP/1.1\r\nHost: t\r\nContent-Length: 3\r\n\r\nabc"
                )
                assert await _read_response(stream) == b"HTTP/1.1 200 OK"
                assert not stream.was_reset and not stream.remote_closed
        assert not [line for line in log if "reset by this device" in line]
    finally:
        server.should_exit = True
        try:
            await serve
        finally:
            sock.close()


@pytest.mark.asyncio
async def test_a_device_reset_is_logged_with_its_cause_and_rate_limited():
    """A device RST is named in the service log, once per window (relay logs cannot)."""
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    log: list[str] = []
    async with FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:
        connector = make_connector(relay, grant_for(relay, "tok"), dead, log=log)
        async with running(connector):
            session = await relay.wait_sessions(1)
            for _ in range(3):
                stream = await session.open()
                with pytest.raises(yamux.StreamReset):
                    await asyncio.wait_for(stream.read(), 5)
    resets = [line for line in log if "reset by this device" in line]
    assert len(resets) == 1, log
    assert "gateway dial failed: ConnectionRefusedError" in resets[0]


def test_the_reset_log_counts_what_it_suppressed():
    lines: list[str] = []
    tally = relay_connector.ResetLog(lines.append, window=0.05)
    tally("first")
    tally("second")
    tally("third")
    time.sleep(0.06)
    tally("fourth")
    assert lines == [
        "relay stream reset by this device: first",
        "relay stream reset by this device: fourth (+2 more in the previous 0.05 s)",
    ]


# -- 2026-11-11: an abnormal drop, an empty refresh error ------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        # `str(TimeoutError())` is "", which is what the service log printed.
        (TimeoutError(), "relay token refresh failed: TimeoutError (no message)"),
        (httpx.ReadTimeout(""), "relay token refresh failed: ReadTimeout (no message)"),
    ],
)
async def test_a_refresh_failure_names_its_type_even_when_its_text_is_empty(
    monkeypatch, failure, expected
):
    monkeypatch.setattr(relay_connector, "backoff_delay", lambda _attempt: 0.0)
    log: list[str] = []
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:

        async def refresh():
            raise failure

        # Already expired, so the first dial must call /connect (C1.6).
        connector = make_connector(
            relay, grant_for(relay, "tok", expires_in=0), port, refresh=refresh, log=log
        )
        async with running(connector):
            for _ in range(200):
                if any(line.startswith("relay token refresh failed") for line in log):
                    break
                await asyncio.sleep(0.01)
    refresh_lines = [line for line in log if line.startswith("relay token refresh failed")]
    assert refresh_lines and refresh_lines[0] == expected


def test_a_wrapped_refresh_failure_names_the_chain_by_type_only():
    from local_operator.providers.auth_store import AuthStoreBusyError

    inner = AuthStoreBusyError("Timed out acquiring AuthStore bootstrap process guard")
    outer = ValueError("This computer's credential store was busy.")
    outer.__cause__ = inner
    assert relay_connector.describe_failure(outer) == (
        "ValueError: This computer's credential store was busy. <- AuthStoreBusyError"
    )
    # An inner link's text never reaches the log: it may be third-party prose.
    secret = RuntimeError("https://example.invalid/?token=do-not-log")
    wrapper = ValueError("x")
    wrapper.__cause__ = secret
    assert "example.invalid" not in relay_connector.describe_failure(wrapper)


@pytest.mark.asyncio
async def test_an_abnormal_drop_redials_with_a_still_valid_token_without_connect(monkeypatch):
    """C1.6 token reuse: a 1006 drop with >= 60 s left on the token redials at once
    with that token; `/connect` is not called first."""
    monkeypatch.setattr(relay_connector, "backoff_delay", lambda _attempt: 0.0)
    calls = 0
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:

        async def refresh():
            nonlocal calls
            calls += 1
            return grant_for(relay, "tok")

        connector = make_connector(relay, grant_for(relay, "tok"), port, refresh=refresh)
        async with running(connector):
            first = await relay.wait_sessions(1)
            await until_connected(connector)
            # No close frame: the TCP connection just dies, as at 18:59:10.
            first.ws.transport.abort()
            second = await relay.wait_sessions(2)
            assert second.epoch == first.epoch + 1
            await until_connected(connector)
    assert calls == 0


@pytest.mark.asyncio
async def test_an_abnormal_drop_with_an_expiring_token_calls_connect_first(monkeypatch):
    """The other side of the same rule: under 60 s left, /connect gates the redial."""
    monkeypatch.setattr(relay_connector, "backoff_delay", lambda _attempt: 0.0)
    calls = 0
    server, port = await echo_gateway()
    async with server, FakeRelay({"tok": Claims(TUNNEL, 7)}) as relay:

        async def refresh():
            nonlocal calls
            calls += 1
            return grant_for(relay, "tok")

        connector = make_connector(relay, grant_for(relay, "tok"), port, refresh=refresh)
        async with running(connector):
            first = await relay.wait_sessions(1)
            await until_connected(connector)
            connector.grant = grant_for(relay, "tok", expires_in=30)
            first.ws.transport.abort()
            await relay.wait_sessions(2)
    assert calls == 1
