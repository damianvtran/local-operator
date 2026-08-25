"""Regression tests for round-1 review remediation (A1, A5/U1, A3, A4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.browser_bridge.daemon import create_app, pairing_status
from local_operator.browser_bridge.protocol import PROTO_VERSION, Response

EXTENSION_ID = "a" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"


def _hello(socket) -> None:
    socket.send_json(
        {
            "event": "hello",
            "proto": PROTO_VERSION,
            "token": "",
            "extension_version": "0.1.0",
            "browser": "Chrome/151",
        }
    )
    socket.receive_json()


def test_a1_pairing_locks_out_and_rotates_after_max_attempts(tmp_path: Path) -> None:
    """The 5th wrong guess must kill the code, not leave it valid (finding A1)."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            _hello(socket)
            original = pairing_status(tmp_path)["pending_code"]
            for _ in range(5):
                socket.send_json({"event": "pair", "code": "000000"})
                socket.receive_json()
            rotated = pairing_status(tmp_path)["pending_code"]
            assert rotated != original, "code must rotate once the attempt cap is hit"
            # The exhausted code is now dead even if it was the real one.
            socket.send_json({"event": "pair", "code": original})
            result = socket.receive_json()
            assert result["ok"] is False


def test_a5_u1_out_of_process_revoke_severs_live_rpc(tmp_path: Path) -> None:
    """A separate-process reset must fail an in-flight RPC immediately."""
    from local_operator.browser_bridge.daemon import reset_pairing

    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            _hello(socket)
            code = pairing_status(tmp_path)["pending_code"]
            socket.send_json({"event": "pair", "code": code})
            assert socket.receive_json()["ok"] is True
            assert app.state.bridge.link.paired is True
            # Revoke from "another process": only the files change.
            reset_pairing(tmp_path)
            # The very next RPC must be refused, not served.
            response = client.post(
                "/rpc",
                headers={"X-Bridge-Key": key},
                json={"id": "r-after-revoke", "method": "status", "params": {}},
            )
            body = response.json()
            assert body["ok"] is False
            assert body["error"]["code"] == "not_paired"
            assert app.state.bridge.link.paired is False


def test_a5_u1_unpair_event_drops_live_socket(tmp_path: Path) -> None:
    """The options-page unpair event revokes and closes the socket (A5/U1)."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            _hello(socket)
            code = pairing_status(tmp_path)["pending_code"]
            socket.send_json({"event": "pair", "code": code})
            assert socket.receive_json()["ok"] is True
            socket.send_json({"event": "unpair"})
        assert app.state.bridge.link.paired is False
        assert pairing_status(tmp_path)["paired"] is False


@pytest.mark.asyncio
async def test_a3_wait_extends_while_awaiting_origin(tmp_path: Path) -> None:
    """A command blocked on a human decision must not die at the base timeout."""
    from local_operator.browser_bridge.daemon import BridgeService

    service = BridgeService(root=tmp_path)
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[Response]" = loop.create_future()
    service.link.awaiting_origin["r-1"] = "https://example.com"

    async def resolve_late() -> None:
        # Longer than the 0.05s base timeout but the awaiting flag holds it open.
        await asyncio.sleep(0.2)
        future.set_result(Response(id="r-1", ok=True, result={"ok": True}))

    asyncio.create_task(resolve_late())
    result = await service._await_response("r-1", future, base_timeout=0.05)
    assert result.ok is True


@pytest.mark.asyncio
async def test_a3_wait_times_out_without_awaiting_flag(tmp_path: Path) -> None:
    """Without an awaiting-origin flag the base timeout still fires (A3 bound)."""
    from local_operator.browser_bridge.daemon import BridgeService

    service = BridgeService(root=tmp_path)
    future: "asyncio.Future[Response]" = asyncio.get_running_loop().create_future()
    with pytest.raises(asyncio.TimeoutError):
        await service._await_response("r-x", future, base_timeout=0.05)


def test_health_reports_driven_url_and_pending_origin(tmp_path: Path) -> None:
    """U3/U2: /health exposes the driven page and any pending approval."""
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        service.link.current_url = "https://example.com/page"
        service.link.current_title = "Example"
        service.link.awaiting_origin["r-1"] = "https://bank.example"
        health = client.get("/health").json()
        assert health["current_url"] == "https://example.com/page"
        assert health["pending_origin"] == "https://bank.example"
