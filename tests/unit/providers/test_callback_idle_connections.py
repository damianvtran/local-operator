"""An unused browser preconnection must not become an OAuth error page.

These tests use actual loopback sockets and the stock flow lifecycle. Only the
idle read's deadline is injected, so no elapsed-time comparison or tuned sleep
decides success. The outer timeout is solely a deadlock guard. No IdP is called.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest

from local_operator.providers.oauth import callback_server
from local_operator.providers.oauth.callback_server import (
    CallbackFlowOptions,
    LoginCallbacks,
    OAuthCallbackFlow,
)


class FixtureFlow(OAuthCallbackFlow):
    def __init__(self, ready: asyncio.Event) -> None:
        super().__init__(
            CallbackFlowOptions(preferred_port=0),
            LoginCallbacks(on_auth_url=lambda *_args, **_kwargs: ready.set()),
            open_browser=lambda _: None,
        )
        self.state = ""

    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        self.state = state
        return "https://fixture.invalid/authorize"

    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        return {"code": code, "state": state}


@pytest.mark.asyncio
@pytest.mark.parametrize("incomplete", ["timeout", "eof", "partial", "empty_request"])
async def test_unused_connection_closes_without_response_and_real_callback_still_works(
    monkeypatch, incomplete
):
    ready = asyncio.Event()
    flow = FixtureFlow(ready)
    actual_wait_for = asyncio.wait_for
    inject_timeout = incomplete == "timeout"

    async def idle_deadline(awaitable, timeout):
        nonlocal inject_timeout
        if inject_timeout and timeout == 10.0:
            inject_timeout = False
            # This is the real server reader's coroutine. Close the unstarted
            # read instead of leaving a warning or pending task in the test.
            awaitable.close()
            raise TimeoutError
        return await actual_wait_for(awaitable, timeout)

    monkeypatch.setattr(callback_server.asyncio, "wait_for", idle_deadline)
    task = asyncio.create_task(flow.run())
    try:
        async with asyncio.timeout(10):
            await ready.wait()
            reader, writer = await asyncio.open_connection("127.0.0.1", flow.bound_port)
            try:
                if incomplete == "partial":
                    writer.write(b"GET /callback?code=fixture HTTP/1.1\r\nHost:")
                    await writer.drain()
                elif incomplete == "empty_request":
                    writer.write(b"\r\n\r\n")
                    await writer.drain()
                if incomplete in {"eof", "partial"}:
                    writer.write_eof()
                # Chrome can reuse a socket opened before the user consents.
                # Sending an unsolicited 404 poisons that later navigation;
                # EOF lets the browser establish a fresh request connection.
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()
            assert not task.done()
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                base = f"http://127.0.0.1:{flow.bound_port}"
                for method, path in (("GET", "/favicon.ico"), ("POST", "/callback")):
                    unknown = await client.request(method, base + path)
                    assert unknown.status_code == 404
                    assert "<h1>Nothing here</h1>" in unknown.text
                    assert not task.done()
                response = await client.get(
                    base + "/callback", params={"code": "fixture-code", "state": flow.state}
                )
                assert response.status_code == 200
                assert "Authorization complete" in response.text
            assert await task == {"code": "fixture-code", "state": flow.state}
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
