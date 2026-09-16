"""A local token endpoint that ROTATES and runs REFRESH-TOKEN REUSE DETECTION.

Stands in for ``mcp.notion.com`` in the refresh-durability tests. The two
behaviours it models are the ones that make a discarded rotation destructive
rather than merely wasteful:

* every successful ``refresh_token`` exchange ISSUES A NEW REFRESH TOKEN and
  invalidates the previous access token, so an old refresh token is spent the
  moment the exchange lands; and
* presenting anything other than the CURRENT refresh token returns
  ``invalid_grant`` AND revokes the whole family, so every later exchange fails
  too. That is Notion's reuse detection, and it is why one stale POST logs out
  every session at once.

A real socket is deliberate. The defects these tests reproduce are about what
happens to an HTTP request that is still on the wire when its awaiter is
cancelled or times out — a ``MockTransport`` handler is awaited inside the
request coroutine itself, so it is cancelled with it and cannot show whether
the server's rotation survived. The client is a real ``httpx.AsyncClient``
talking to 127.0.0.1, exactly as ``_refresh_oauth_token_locked`` does it.

Rotation happens BEFORE ``response_delay_s`` elapses, so a request cancelled or
timed out inside that window has already had its rotation performed server-side
even though its response never reaches the caller. That is the window the
tests care about.
"""

from __future__ import annotations

import asyncio
import contextlib
import http
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl


class FakeTokenEndpoint:
    """One rotating, reuse-detecting token endpoint on 127.0.0.1.

    Use as an async context manager: ``async with FakeTokenEndpoint() as ep``
    binds a port and exposes ``ep.token_endpoint``. Recorded traffic is on
    ``requests`` (every form body the server parsed), and ``reuse_attempts``
    counts the requests that presented a spent token — the counter the
    durability assertions rest on, because it is the family-revoking request
    itself.
    """

    def __init__(
        self,
        *,
        access_token: str = "access-0",
        refresh_token: str = "refresh-0",
        response_delay_s: float = 0.0,
        on_reject: Callable[[], Awaitable[None]] | None = None,
        commit_then_fail_status: int | None = None,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.response_delay_s = response_delay_s
        #: Awaited just before an ``invalid_grant`` response is written. Lets a
        #: test model the SIBLING process that persists its own rotation while
        #: our request is in flight — the window D3 is about.
        self.on_reject = on_reject
        #: When set, a refresh that the server ACCEPTS is still answered with
        #: this status (an error body) AFTER the rotation is committed — the
        #: provider shape review round 2 (minor 2) was about: a 5xx that leaves
        #: the presented token spent while telling the client nothing. Real
        #: enough to be the failure mode of a proxy in front of a rotating
        #: issuer, and the only way to test the marker's clearing rule without
        #: guessing at one.
        self.commit_then_fail_status = commit_then_fail_status
        self.requests: list[dict[str, str]] = []
        self.reuse_attempts = 0
        self.rotation_count = 0
        self.revoked = False
        #: Set once a rotation has been applied server-side, so a test can
        #: cancel exactly inside the window where the response is still
        #: unwritten.
        self.rotation_applied = asyncio.Event()
        self._server: asyncio.Server | None = None
        self._port = 0

    @property
    def token_endpoint(self) -> str:
        return f"http://127.0.0.1:{self._port}/token"

    async def __aenter__(self) -> FakeTokenEndpoint:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        server = self._server
        if server is not None:
            server.close()
            await server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = {}
            for line in head.decode("latin-1").split("\r\n")[1:]:
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0") or 0)
            body = await reader.readexactly(length) if length else b""
            status, payload = self._decide(dict(parse_qsl(body.decode("utf-8"))))
            if status != 200 and self.on_reject is not None:
                # Awaited BEFORE the rejection is written, so the sibling's
                # write lands after our pre-POST read and before the tombstone
                # write — exactly the window the marker documents.
                await self.on_reject()
            if self.response_delay_s:
                # AFTER the rotation above: a client that walks away now has had
                # its token spent and its rotation performed.
                await asyncio.sleep(self.response_delay_s)
            self._write(writer, status, payload)
        except Exception:  # noqa: BLE001 — a fixture must never fail a test by raising
            pass
        finally:
            # ``Exception``, not ``BaseException``: ``CancelledError`` is not a
            # subclass of it, so a shutdown of the server still propagates.
            with contextlib.suppress(Exception):
                writer.close()

    def _decide(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.requests.append(form)
        if form.get("grant_type") != "refresh_token":
            return 400, {"error": "unsupported_grant_type"}
        if self.revoked or form.get("refresh_token") != self.refresh_token:
            # Reuse detection. A revoked family never recovers, and a spent
            # token presents exactly like a token from a revoked family.
            self.reuse_attempts += 1
            self.revoked = True
            return 400, {"error": "invalid_grant"}
        self.rotation_count += 1
        # Rotate, and invalidate the access token the old grant carried: a
        # rotating provider revokes every previously issued access token when
        # it issues a new one.
        self.refresh_token = f"refresh-{self.rotation_count}"
        self.access_token = f"access-{self.rotation_count}"
        self.rotation_applied.set()
        if self.commit_then_fail_status is not None:
            # The rotation above HAPPENED and the client is told nothing usable:
            # the committed state is the one a next exchange would have to
            # present, which is exactly what makes clearing the send marker on a
            # non-200 a family-revoking mistake.
            return self.commit_then_fail_status, {"error": "server_error"}
        return 200, {
            "access_token": self.access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": self.refresh_token,
            "scope": "fixture",
        }

    @staticmethod
    def _write(writer: asyncio.StreamWriter, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        # The real phrase, not a hardcoded "Bad Request": a 500 written as
        # "500 Bad Request" is a fixture artifact that makes a status-code test
        # look like it is reading a 400.
        reason = http.HTTPStatus(status).phrase
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("utf-8")
            + body
        )
