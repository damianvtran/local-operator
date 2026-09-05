"""Authenticated, streaming loopback adapter for personal harnesses.

The Radient Worker rejects anonymous traffic at the edge; this gateway verifies
its short-lived, request-bound assertion before reaching a harness.
Loopback or a header's presence is never evidence of cloud authentication.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import jwt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.tunnels.config import validate_connection

MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_STREAM_SECONDS = 60
AUTHORIZATION_LEASE_SECONDS = 30
PROOF_HEADER = "x-radient-tunnel-assertion"
# Only presentation/protocol headers cross the boundary. In particular the
# owner's Radient cookies and bearer must never reach a local harness,
# whose plugins/tools may log, reflect, or export request headers.
_REQUEST_HEADERS = {
    "accept",
    "accept-language",
    "content-type",
    "range",
    "if-none-match",
    "if-modified-since",
    "last-event-id",
    "origin",
}
_RESPONSE_HEADERS = {
    "content-type",
    "content-length",
    "content-encoding",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
    "x-accel-buffering",
}


class OriginVerifier:
    """Keys pinned by an authenticated /connect response, never by a request.

    Only RS256 public keys from the trusted control plane are admitted. A
    token-selected kid can select one of them, never trigger a URL retrieval.
    Rotating the origin key increments the tunnel version and reconnects.
    """

    def __init__(self, access: dict[str, Any], client: httpx.AsyncClient) -> None:
        self.access = access
        self.client = client
        self.keys: dict[str, Any] = {
            row["kid"]: jwt.PyJWK.from_dict(row).key for row in access["jwks"]["keys"]
        }
        if any(key.key_size < 2048 for key in self.keys.values()):
            raise ValueError("Origin proof requires RSA keys of at least 2048 bits.")
        self.used: dict[str, float] = {}

    async def verify(
        self,
        token: str,
        *,
        host: str,
        harness_id: str,
        method: str,
        target: str,
        body: bytes,
    ) -> dict[str, Any]:
        if not token or len(token) > 16384:
            raise ValueError("Missing origin assertion.")
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if header.get("alg") != "RS256" or not isinstance(kid, str):
            raise ValueError("Invalid origin assertion.")
        key = self.keys.get(kid)
        if key is None:
            raise ValueError("Origin signing key unavailable.")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=host,
            issuer=self.access["issuer"],
            options={"require": ["exp", "iat", "iss", "aud", "sub", "jti"]},
        )
        expected = {
            "aud": host,
            "sub": self.access["owner_account_id"],
            "tunnel_id": self.access["tunnel_id"],
            "version": self.access["version"],
            "harness_id": harness_id,
            "method": method,
            "target": target,
            "body_sha256": hashlib.sha256(body).hexdigest(),
        }
        if any(claims.get(k) != value for k, value in expected.items()):
            raise ValueError("Origin assertion does not authorize this request.")
        if not isinstance(claims.get("version"), int) or isinstance(claims["version"], bool):
            raise ValueError("Invalid origin assertion version.")
        issued, expires = claims["iat"], claims["exp"]
        if (
            not isinstance(issued, int)
            or isinstance(issued, bool)
            or not isinstance(expires, int)
            or isinstance(expires, bool)
            or not 0 < expires - issued <= 30
        ):
            raise ValueError("Origin assertion lifetime exceeds thirty seconds.")
        if method not in {"GET", "HEAD", "OPTIONS"}:
            # No await between check and insertion: concurrent replays on this
            # event loop cannot both pass. Refuse a full cache rather than
            # evict a still-live nonce and reopen its replay window.
            now = time.time()
            self.used = {key: expiry for key, expiry in self.used.items() if expiry > now}
            nonce = claims["jti"]
            if (
                not isinstance(nonce, str)
                or not nonce
                or nonce in self.used
                or len(self.used) >= 10000
            ):
                raise ValueError("Replayed origin assertion.")
            self.used[nonce] = float(expires)
        return claims


class Gateway:
    def __init__(
        self,
        connection: dict[str, Any],
        client: httpx.AsyncClient,
        *,
        mobile_password: str | None = None,
        opencode_basic: dict[str, str] | None = None,
        connector_ready: Callable[[], bool] | None = None,
    ) -> None:
        self.connection = validate_connection(connection)
        self.client = client
        self.verifier = OriginVerifier(connection["origin_auth"], client)
        self.mobile_password = mobile_password
        self.opencode_basic = opencode_basic
        self.connector_ready = connector_ready or (lambda: False)
        self.authorized_until = time.monotonic() + AUTHORIZATION_LEASE_SECONDS
        self.revoked = False

    def authorize(self) -> None:
        self.authorized_until = time.monotonic() + AUTHORIZATION_LEASE_SECONDS

    @staticmethod
    def target(scope: Mapping[str, Any]) -> str:
        return (
            scope["raw_path"] + (b"?" + scope["query_string"] if scope["query_string"] else b"")
        ).decode("ascii")

    def harness(self, host: str) -> dict[str, Any] | None:
        return next(
            (
                h
                for h in self.connection["tunnel"]["harnesses"]
                if h["enabled"] and h["hostname"] == host
            ),
            None,
        )

    def headers(self, incoming: Any, host: str, harness: dict[str, Any]) -> dict[str, str]:
        headers = {k: v for k, v in incoming.items() if k in _REQUEST_HEADERS}
        headers["host"] = host
        # OpenCode 1.18.5 requires this explicit request marker to issue its
        # short-lived PTY WebSocket ticket. Preserve only its defined value for
        # that harness; never synthesize it or relax Origin/proof verification.
        if harness["id"] == "opencode" and incoming.get("x-opencode-ticket") == "1":
            headers["x-opencode-ticket"] = "1"
        if harness["id"] == "local-operator":
            if not self.mobile_password:
                raise ValueError("Mobile relay is not installed.")
            headers["cookie"] = f"{COOKIE_NAME}={sign_cookie(self.mobile_password)}"
        elif self.opencode_basic:
            credential = self.opencode_basic["username"] + ":" + self.opencode_basic["password"]
            headers["authorization"] = "Basic " + base64.b64encode(credential.encode()).decode()
        return headers

    async def handle(self, request: Request) -> Response:
        host = request.headers.get("host", "").lower()
        if (
            request.url.path == "/_lop_tunnel/health"
            and host == f"127.0.0.1:{self.connection['gateway_port']}"
        ):
            return JSONResponse(
                {
                    "ok": not self.revoked and time.monotonic() < self.authorized_until,
                    "connected": self.connector_ready(),
                }
            )
        if self.revoked or time.monotonic() >= self.authorized_until:
            return JSONResponse({"error": "tunnel authorization unavailable"}, status_code=503)
        harness = self.harness(host)
        if harness is None:
            return JSONResponse({"error": "unknown tunnel host"}, status_code=404)
        expected_origin = "https://" + host
        origin = request.headers.get("origin")
        # A signed document navigation is how a phone opens a saved/shared
        # harness URL. Fetch Metadata marks that cross-site even though the
        # edge already authenticated it. Only safe top-level navigation gets
        # this exception; sibling fetches and browser mutations remain denied.
        navigation = (
            request.method in {"GET", "HEAD"}
            and request.headers.get("sec-fetch-mode") == "navigate"
        )
        if (
            (origin is not None and origin != expected_origin)
            or (
                request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}
                and not navigation
            )
            or (request.method not in {"GET", "HEAD", "OPTIONS"} and origin != expected_origin)
        ):
            return JSONResponse({"error": "same-origin request required"}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request exceeds 10 MiB"}, status_code=413)
        if request.method in {"GET", "HEAD"} and body:
            return JSONResponse({"error": "GET/HEAD bodies are not supported"}, status_code=400)
        try:
            await self.verifier.verify(
                request.headers.get(PROOF_HEADER, ""),
                host=host,
                harness_id=harness["id"],
                method=request.method,
                target=self.target(request.scope),
                body=bytes(body),
            )
        except (ValueError, KeyError, TypeError, jwt.PyJWTError, httpx.HTTPError):
            return JSONResponse(
                {"error": "valid Radient origin assertion required"}, status_code=401
            )
        # Uploads can yield to the policy poller for arbitrarily long periods.
        # Authorization at request arrival cannot authorize a later mutation:
        # recheck after consuming/verifying its body, immediately before any
        # harness request or local side effect begins.
        if self.revoked or time.monotonic() >= self.authorized_until:
            return JSONResponse({"error": "tunnel authorization unavailable"}, status_code=503)
        if request.url.path == "/logout":
            response = RedirectResponse("/_radient/logout", status_code=303)
            response.headers["Clear-Site-Data"] = '"storage"'
            return response
        if request.url.path == "/login" and harness["id"] == "local-operator":
            return RedirectResponse("/", status_code=303)
        # Preserve the checked public Host for the relay's independent Origin
        # check. The destination remains a literal loopback address regardless.
        try:
            headers = self.headers(request.headers, host, harness)
        except ValueError:
            return JSONResponse({"error": "mobile relay is not installed"}, status_code=503)
        target = httpx.URL(
            scheme="http",
            host="127.0.0.1",
            port=harness["port"],
            raw_path=request.scope["raw_path"]
            + (b"?" + request.scope["query_string"] if request.scope["query_string"] else b""),
        )
        try:
            upstream = await self.client.send(
                self.client.build_request(
                    request.method, target, headers=headers, content=bytes(body)
                ),
                stream=True,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return JSONResponse({"error": "local harness unavailable"}, status_code=502)
        response_headers = {k: v for k, v in upstream.headers.items() if k in _RESPONSE_HEADERS}
        location = upstream.headers.get("location")
        if location:
            # No credential-bearing or untrusted absolute redirect can escape
            # to another host. Root-relative harness redirects retain this AUD.
            if (
                location.startswith("/")
                and not location.startswith("//")
                and "\\" not in location
                and not any(ord(char) < 32 for char in location)
            ):
                response_headers["location"] = location
            else:
                await upstream.aclose()
                return JSONResponse({"error": "unsafe harness redirect"}, status_code=502)
        response_headers["cache-control"] = "no-store"
        response_headers["referrer-policy"] = "same-origin"
        response_headers["x-content-type-options"] = "nosniff"
        response_headers["content-security-policy"] = "frame-ancestors 'none'"

        async def stream():
            # SSE is deliberately streamed with backpressure. Reconnects at
            # most every minute recheck edge policy; a backend revoke
            # also stops reads after the next origin chunk (keepalives are 15s).
            try:
                async with asyncio.timeout(MAX_STREAM_SECONDS):
                    async for chunk in upstream.aiter_raw():
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            break
                        yield chunk
            except (TimeoutError, httpx.HTTPError):
                pass
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream(), status_code=upstream.status_code, headers=response_headers
        )

    async def websocket(self, socket: WebSocket) -> None:
        """A bounded generic WebSocket adapter; LO itself uses HTTP and SSE."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException
        from websockets.typing import Origin

        host = socket.headers.get("host", "").lower()
        harness = self.harness(host)
        if (
            harness is None
            or self.revoked
            or time.monotonic() >= self.authorized_until
            or socket.headers.get("origin") != "https://" + host
        ):
            await socket.close(code=1008)
            return
        try:
            target = self.target(socket.scope)
            await self.verifier.verify(
                socket.headers.get(PROOF_HEADER, ""),
                host=host,
                harness_id=harness["id"],
                method="GET",
                target=target,
                body=b"",
            )
            headers = self.headers(socket.headers, host, harness)
            headers.pop("host", None)
            headers.pop("origin", None)
            # URI host supplies the checked HTTP Host while connect()'s socket
            # override ensures DNS can never redirect this to a public server.
            async with connect(
                "ws://" + host + target,
                host="127.0.0.1",
                port=harness["port"],
                proxy=None,
                origin=Origin("https://" + host),
                additional_headers=headers,
                subprotocols=socket.scope.get("subprotocols") or None,
                max_size=MAX_BODY_BYTES,
            ) as upstream:
                await socket.accept(subprotocol=upstream.subprotocol)

                async def to_origin() -> None:
                    while not self.revoked and time.monotonic() < self.authorized_until:
                        message = await socket.receive()
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            return
                        if message["type"] == "websocket.disconnect":
                            return
                        data = message.get("bytes")
                        if data is None:
                            data = message.get("text", "")
                        if len(data) > MAX_BODY_BYTES:
                            return
                        await upstream.send(data)

                async def to_browser() -> None:
                    async for data in upstream:
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            return
                        if isinstance(data, bytes):
                            await socket.send_bytes(data)
                        else:
                            await socket.send_text(data)

                tasks = [asyncio.create_task(to_origin()), asyncio.create_task(to_browser())]
                try:
                    await asyncio.wait(
                        tasks, timeout=MAX_STREAM_SECONDS, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            jwt.PyJWTError,
            httpx.HTTPError,
            WebSocketDisconnect,
            WebSocketException,
        ):
            pass
        finally:
            try:
                await socket.close(code=1000)
            except RuntimeError:
                pass

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    self.handle,
                    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                ),
                WebSocketRoute("/{path:path}", self.websocket),
            ]
        )
