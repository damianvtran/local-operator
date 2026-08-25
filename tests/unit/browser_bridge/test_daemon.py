from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from local_operator.browser_bridge.daemon import create_app, pairing_status
from local_operator.browser_bridge.protocol import PROTO_VERSION

EXTENSION_ID = "a" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"


def test_rpc_auth_invalid_and_disconnected(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        key = app.state.bridge.state.session_key
        assert (
            client.post("/rpc", json={"id": "r-1", "method": "ping", "params": {}}).status_code
            == 401
        )
        invalid = client.post("/rpc", headers={"X-Bridge-Key": key}, json={"method": "ping"})
        assert invalid.status_code == 422
        disconnected = client.post(
            "/rpc",
            headers={"X-Bridge-Key": key},
            json={"id": "r-2", "method": "open", "params": {"url": "https://example.com"}},
        )
        assert disconnected.json()["error"]["code"] == "extension_disconnected"


def test_pairing_issues_token_then_authenticates(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/126",
                }
            )
            assert socket.receive_json() == {
                "event": "hello_ack",
                "proto": PROTO_VERSION,
                "paired": False,
            }
            code = pairing_status(tmp_path)["pending_code"]
            socket.send_json({"event": "pair", "code": code})
            paired = socket.receive_json()
            assert paired["ok"] is True
            assert len(paired["token"]) >= 32
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": paired["token"],
                    "extension_version": "0.1.0",
                    "browser": "Chrome/126",
                }
            )
            assert socket.receive_json()["paired"] is True


def test_web_page_origin_is_rejected(tmp_path: Path) -> None:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        try:
            with client.websocket_connect(
                "/extension", headers={"origin": "https://evil.example"}
            ) as socket:
                socket.receive_json()
        except Exception as exc:  # Starlette surfaces the explicit 4004 close.
            assert "4004" in str(exc) or getattr(exc, "code", None) == 4004
        else:  # pragma: no cover - rejection is mandatory
            raise AssertionError("web origin was accepted")
