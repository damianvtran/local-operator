"""Real TCP HTTP/SSE/WebSocket traffic through the gateway, with isolated origins.

These tests do not open an OAuth browser, call a cloud API, start cloudflared,
or scan the operator's registrants. They prove the generic local adapter using
actual socket handshakes and a separate harness server.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import jwt
import pytest
import uvicorn
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from websockets.asyncio.client import connect
from websockets.typing import Origin

from local_operator.tunnels.config import ORIGIN_ISSUER
from local_operator.tunnels.gateway import PROOF_HEADER, Gateway


@asynccontextmanager
async def server(app):
    ready = asyncio.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            ready.set()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    running = ReadyServer(
        uvicorn.Config(
            app,
            log_level="error",
            access_log=False,
            proxy_headers=False,
            timeout_graceful_shutdown=1,
        )
    )
    task = asyncio.create_task(running.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        yield listener.getsockname()[1]
    finally:
        running.should_exit = True
        await task
        listener.close()


@pytest.mark.asyncio
async def test_real_http_sse_websocket_proxy_and_header_isolation():
    seen = []
    streaming = asyncio.Event()
    ticket = None

    async def api(request):
        seen.append(dict(request.headers))
        return JSONResponse({"received": await request.json()})

    async def events(request):
        async def body():
            yield b"event: state\ndata: first\n\n"
            # Until the client has received the first event this stream cannot
            # finish. A buffering proxy deadlocks and trips the outer guard.
            await streaming.wait()
            yield b"event: state\ndata: second\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    async def websocket(ws):
        seen.append(dict(ws.headers))
        await ws.accept()
        await ws.send_text(await ws.receive_text())
        await ws.close()

    async def connect_token(request):
        nonlocal ticket
        # OpenCode 1.18.5 requires this explicit browser marker as well as the
        # checked Origin before issuing a directory-bound, one-use PTY ticket.
        if (
            request.headers.get("x-opencode-ticket") != "1"
            or request.headers.get("origin") != "https://" + host
            or request.headers.get("authorization") != expected
            or request.query_params.get("directory") != "/qa"
        ):
            return JSONResponse({"error": "ticket request denied"}, status_code=403)
        seen.append(dict(request.headers))
        ticket = str(uuid.uuid4())
        return JSONResponse({"ticket": ticket, "expires_in": 60})

    async def pty_connect(ws):
        nonlocal ticket
        if (
            not ticket
            or ws.query_params.get("ticket") != ticket
            or ws.query_params.get("directory") != "/qa"
            or ws.headers.get("origin") != "https://" + host
            or ws.headers.get("authorization") != expected
        ):
            await ws.close(code=1008)
            return
        ticket = None
        seen.append(dict(ws.headers))
        await ws.accept()
        await ws.send_text(await ws.receive_text())
        await ws.close()

    origin = Starlette(
        routes=[
            Route("/api", api, methods=["POST"]),
            Route("/events", events),
            WebSocketRoute("/ws", websocket),
            Route("/pty/qa/connect-token", connect_token, methods=["POST"]),
            WebSocketRoute("/pty/qa/connect", pty_connect),
        ]
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="edge-1", alg="RS256")
    host = "abc123-oc.radienthq.com"
    expected = "Basic " + base64.b64encode(b"opencode:private-test-origin").decode()

    def signed(method, target, body=b""):
        now = int(time.time())
        return jwt.encode(
            {
                "iss": ORIGIN_ISSUER,
                "aud": host,
                "sub": "owner",
                "tunnel_id": "one",
                "harness_id": "opencode",
                "version": 1,
                "method": method,
                "target": target,
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "iat": now,
                "exp": now + 30,
                "jti": str(uuid.uuid4()),
            },
            key,
            algorithm="RS256",
            headers={"kid": "edge-1"},
        )

    async with server(origin) as origin_port:
        connection = {
            "gateway_port": 4099,
            "tunnel": {
                "id": "one",
                "version": 1,
                "harnesses": [
                    {"id": "opencode", "enabled": True, "port": origin_port, "hostname": host},
                ],
            },
            "origin_auth": {
                "issuer": ORIGIN_ISSUER,
                "owner_account_id": "owner",
                "tunnel_id": "one",
                "version": 1,
                "jwks": {"keys": [public]},
            },
        }
        async with httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(10, read=None)
        ) as upstream:
            gateway = Gateway(
                connection,
                upstream,
                opencode_basic={"username": "opencode", "password": "private-test-origin"},
            )
            async with server(gateway.app()) as gateway_port:
                async with httpx.AsyncClient(
                    trust_env=False, base_url=f"http://127.0.0.1:{gateway_port}"
                ) as client:
                    body = b'{"prompt":"hello from phone"}'
                    reply = await client.post(
                        "/api",
                        content=body,
                        headers={
                            "host": host,
                            "origin": "https://" + host,
                            PROOF_HEADER: signed("POST", "/api", body),
                            "cookie": "__Host-radient-grant=cloud-secret",
                        },
                    )
                    assert reply.status_code == 200
                    assert reply.json() == {"received": {"prompt": "hello from phone"}}
                    async with asyncio.timeout(10):
                        async with client.stream(
                            "GET",
                            "/events",
                            headers={"host": host, PROOF_HEADER: signed("GET", "/events")},
                        ) as response:
                            lines = response.aiter_lines()
                            assert await anext(lines) == "event: state"
                            assert await anext(lines) == "data: first"
                            streaming.set()
                            assert "data: second" in [line async for line in lines]
                    async with connect(
                        "ws://" + host + "/ws",
                        host="127.0.0.1",
                        port=gateway_port,
                        proxy=None,
                        origin=Origin("https://" + host),
                        additional_headers={
                            PROOF_HEADER: signed("GET", "/ws"),
                            "cookie": "cloud-secret=hidden",
                        },
                    ) as ws:
                        await ws.send("phone steering")
                        assert await ws.recv() == "phone steering"
                    target = "/pty/qa/connect-token?directory=%2Fqa"
                    # Absence/other values must not synthesize the browser's
                    # request marker, even for an otherwise signed request.
                    for marker in (None, "anything-else", "1"):
                        headers = {
                            "host": host,
                            "origin": "https://" + host,
                            "authorization": "Bearer cloud-secret",
                            "cookie": "__Host-radient-grant=cloud-secret",
                            PROOF_HEADER: signed("POST", target),
                        }
                        if marker is not None:
                            headers["x-opencode-ticket"] = marker
                        reply = await client.post(target, headers=headers)
                        assert reply.status_code == (200 if marker == "1" else 403)
                    assert reply.json()["expires_in"] == 60
                    target = "/pty/qa/connect?directory=%2Fqa&ticket=" + reply.json()["ticket"]
                    async with connect(
                        "ws://" + host + target,
                        host="127.0.0.1",
                        port=gateway_port,
                        proxy=None,
                        origin=Origin("https://" + host),
                        additional_headers={PROOF_HEADER: signed("GET", target)},
                    ) as ws:
                        await ws.send("phone PTY input")
                        assert await ws.recv() == "phone PTY input"
                    streaming.clear()
                    async with asyncio.timeout(10):
                        async with client.stream(
                            "GET",
                            "/events",
                            headers={"host": host, PROOF_HEADER: signed("GET", "/events")},
                        ) as response:
                            lines = response.aiter_lines()
                            assert await anext(lines) == "event: state"
                            assert await anext(lines) == "data: first"
                            gateway.revoked = True
                            streaming.set()
                            assert "data: second" not in [line async for line in lines]
    assert len(seen) == 4
    for headers in seen:
        assert headers["authorization"] == expected
        assert PROOF_HEADER not in headers
        assert "cookie" not in headers
