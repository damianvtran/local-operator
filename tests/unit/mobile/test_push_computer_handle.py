"""Registration exposes the live emit binding, never a local registry identity.

Cloud account authorization remains at the existing gateway. These tests drive
its daemon-side boundary with synthetic tunnel/credential bindings; they do not
claim to exercise a hosted gateway or an APNs delivery.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import auth, push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.mobile.push_transport import PushLane, RadientPushTransport
from local_operator.mobile.push_worker import _PendingEmit
from local_operator.paths import config_dir
from local_operator.tunnels.errors import RecordNotFound

HANDLE = "synthetic-account-a-tunnel"
BODY = {
    "platform": "ios",
    "token": "synthetic-apns-token",
    "environment": "production",
    "app_version": "1.0.0 (1)",
    "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
}


@pytest.fixture
def bound_lane(monkeypatch: pytest.MonkeyPatch) -> tuple[PushLane, list[Any]]:
    # Mutable current binding models config changes without touching a live
    # account. No cloud call is made: the unchanged forward is covered in
    # test_push_transport.py, including its exact computer field.
    binding: list[Any] = [(HANDLE, 17)]
    transport = RadientPushTransport(
        tunnel_id=HANDLE, credential_id=17, tunnel_key=lambda: binding[0]
    )
    monkeypatch.setattr(transport, "schedule_register_forward", lambda *args: None)
    return PushLane(transport, config_dir(), unread_count=lambda: 0), binding


def _client(lane: PushLane | None) -> TestClient:
    return TestClient(
        build_app(MobileDaemon(port=0, password="synthetic-password", push_lane=lane)),
        follow_redirects=False,
    )


def _login(client: TestClient) -> None:
    assert client.post("/login", data={"password": "synthetic-password"}).status_code in (
        200,
        303,
    )


def test_registration_handle_equals_composed_attention_and_survives_reconnect(bound_lane) -> None:
    lane, _ = bound_lane
    with _client(lane) as client:
        _login(client)
        first = client.post("/api/push/register", json=BODY)
        assert first.status_code == 200, first.text
        body = first.json()
        # Use the worker's actual composer, not an assertion against a saved
        # registry ID. The same worker is constructed by production PushLane.
        payload = lane.worker._compose(_PendingEmit(key="attention-1", kind="attention"))
        assert payload is not None
        assert body["computer_handle"] == payload["computer"] == HANDLE
        assert body["computer_handle"] != body["device_id"]
        renewed = client.post("/api/push/register", json=BODY)
        assert renewed.status_code == 200, renewed.text
        assert renewed.json()["device_id"] == body["device_id"]
        assert renewed.json()["registered_at"] == body["registered_at"]
        assert renewed.json()["device_key"] != body["device_key"]
    with _client(lane) as reconnected:
        _login(reconnected)
        response = reconnected.post("/api/push/register", json=BODY)
        assert response.json()["computer_handle"] == body["computer_handle"]
    stored = push_devices.store_path(config_dir()).read_text()
    assert "computer_handle" not in stored
    assert BODY["token"] not in stored


@pytest.mark.parametrize("current", [None, ("synthetic-account-b-tunnel", 23), (HANDLE, 23)])
def test_changed_account_or_revoked_tunnel_never_exposes_old_binding(bound_lane, current) -> None:
    lane, binding = bound_lane
    binding[0] = current
    with _client(lane) as client:
        _login(client)
        response = client.post("/api/push/register", json=BODY)
        assert response.status_code == 200, response.text
        assert "computer_handle" not in response.json()


def test_disarmed_lane_does_not_expose_a_handle(bound_lane) -> None:
    lane, _ = bound_lane
    lane.transport._disarmed = True
    with _client(lane) as client:
        _login(client)
        response = client.post("/api/push/register", json=BODY)
        assert response.status_code == 200
        assert "computer_handle" not in response.json()


def test_refusal_during_binding_read_does_not_expose_a_handle(monkeypatch) -> None:
    reading_binding = threading.Event()
    disarmed = threading.Event()

    def read_binding() -> tuple[str, int]:
        reading_binding.set()
        assert disarmed.wait(5), "register forward did not disarm during config read"
        return HANDLE, 17

    transport = RadientPushTransport(tunnel_id=HANDLE, credential_id=17, tunnel_key=read_binding)
    original_disarm = transport._disarm

    def note_disarm(failure: Exception) -> None:
        original_disarm(failure)
        disarmed.set()

    async def refuse_registration(**kwargs) -> None:
        # The real route schedules the forward before reading the binding on
        # a worker thread. Events pin the refusal inside that read, without
        # sleeps or replacing the forward/scheduler that must actually disarm.
        assert await asyncio.to_thread(reading_binding.wait, 5), "binding read never started"
        raise RecordNotFound("synthetic registration refusal")

    monkeypatch.setattr(transport, "_call", refuse_registration)
    monkeypatch.setattr(transport, "_disarm", note_disarm)
    lane = PushLane(transport, config_dir(), unread_count=lambda: 0)
    with _client(lane) as client:
        _login(client)
        response = client.post("/api/push/register", json=BODY)
        assert response.status_code == 200, response.text
        assert disarmed.is_set()
        assert transport._disarmed
        assert set(response.json()) == {"ok", "device_id", "device_key", "registered_at"}


def test_unauthenticated_invalid_and_revoked_device_refusals_have_no_handle(bound_lane) -> None:
    lane, _ = bound_lane
    with _client(lane) as client:
        response = client.post("/api/push/register", json=BODY)
        assert response.status_code == 401
        assert "computer_handle" not in response.json()
        # A cookie from another authenticated route/machine is not this
        # relay's credential. Account authorization at the hosted gateway is
        # unchanged; this proves its daemon-side credential boundary remains.
        client.cookies.set(auth.COOKIE_NAME, auth.sign_cookie("other-route-password"))
        wrong_route = client.post("/api/push/register", json=BODY)
        assert wrong_route.status_code == 401
        assert "computer_handle" not in wrong_route.json()
        assert not push_devices.store_path(config_dir()).exists()
        client.cookies.clear()
        _login(client)
        invalid = client.post("/api/push/register", json={**BODY, "platform": "invalid"})
        assert invalid.status_code == 422, invalid.text
        assert "computer_handle" not in invalid.json()
        registered = client.post("/api/push/register", json=BODY).json()
        device_id = registered["device_id"]
        assert client.delete(f"/api/push/devices/{device_id}").status_code == 200
        before = push_devices.store_path(config_dir()).read_bytes()
        revoked = client.post("/api/push/register", json=BODY)
        assert revoked.status_code == 403, revoked.text
        assert revoked.json()["code"] == "device_revoked"
        assert "computer_handle" not in revoked.json()
        assert push_devices.store_path(config_dir()).read_bytes() == before


def test_without_a_lane_registration_retains_legacy_shape() -> None:
    with _client(None) as client:
        _login(client)
        response = client.post("/api/push/register", json=BODY)
        assert response.status_code == 200
        assert set(response.json()) == {"ok", "device_id", "device_key", "registered_at"}
        assert "computer_handle" not in json.loads(bytes(response.content))
