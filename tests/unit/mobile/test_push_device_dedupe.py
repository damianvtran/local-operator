"""One phone, one row: the registry's defence against a client that forks its identity.

The operator's Settings → Notifications showed 33 identical "iPhone" rows for
one physical phone (2026-11-16, build 1130). The registry already upserted on
``(install_id, platform)``; the 33 rows carried 33 DISTINCT install ids, minted
by an app whose keystore read failed transiently (local-operator-mobile
``install-id.ts`` treated "could not read" as "never minted"). Every one of
those registrations carried the same APNs token, and every one was forwarded
to the cloud as its own device — so one phone was 33 push targets.

These cells drive the real ``POST /api/push/register`` route (the same
``build_app`` + isolated config root as ``test_push_devices.py``) with exactly
what that client sent, and assert what the registry must do about it: rows
that share a push handle with a newer registration are SUPERSEDED (hidden from
the Settings list and reported not-live so delivery stops), legacy rows the
fix inherits collapse on the first registration after the upgrade, a genuinely
different phone keeps its own row, and a revoke still sticks.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from local_operator.mobile import push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.mobile.push_devices import PUSH_DEVICES_STORE_NAME
from local_operator.paths import config_dir

TOKEN = "apns-token-0f1e2d3c4b5a69788796a5b4c3d2e1f0"
OTHER_TOKEN = "apns-token-ffeeddccbbaa99887766554433221100"


def _body(install_id: str, token: str = TOKEN, **extra: Any) -> dict[str, Any]:
    return {
        "platform": "ios",
        "token": token,
        "environment": "production",
        "app_version": "0.0.0 (1088)",
        "install_id": install_id,
        **extra,
    }


def _client() -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _store_path() -> Path:
    return config_dir() / PUSH_DEVICES_STORE_NAME


def _stored() -> list[dict[str, Any]]:
    return json.loads(_store_path().read_text(encoding="utf-8"))["devices"]


def _listed(client: TestClient) -> list[dict[str, Any]]:
    response = client.get("/api/push/devices")
    assert response.status_code == 200, response.text
    return response.json()["devices"]


def _live_ids() -> set[str]:
    """What the emit side delivers to: live state AND a live credential."""
    return {
        fact["device_id"]
        for fact in push_devices.credential_facts(config_dir())
        if fact["state"] == push_devices.STATE_LIVE and fact["credential_live"]
    }


def test_a_burst_of_fresh_install_ids_with_one_token_is_one_listed_device() -> None:
    """The reproduction: the measured burst, six fresh ids in one second, one token."""
    client = _client()
    ids = []
    for _ in range(6):
        response = client.post("/api/push/register", json=_body(str(uuid.uuid4())))
        assert response.status_code == 200, response.text
        ids.append(response.json()["device_id"])

    listed = _listed(client)
    assert [row["device_id"] for row in listed] == [ids[-1]]
    # The superseded rows stay on disk (the cloud must be told they stopped,
    # which only a reported row can say) but none of them is a delivery target.
    assert len(_stored()) == 6
    assert _live_ids() <= {ids[-1]}
    facts = {fact["device_id"]: fact for fact in push_devices.credential_facts(config_dir())}
    for superseded in ids[:-1]:
        assert facts[superseded]["state"] == push_devices.STATE_EXPIRED
        assert facts[superseded]["credential_live"] is False


def test_the_push_handle_digest_is_stored_and_the_token_is_not() -> None:
    client = _client()
    assert client.post("/api/push/register", json=_body(str(uuid.uuid4()))).status_code == 200
    raw = _store_path().read_bytes()
    assert TOKEN.encode() not in raw
    assert b'"token"' not in raw
    (record,) = _stored()
    assert record["push_handle_digest"] == push_devices.push_handle_digest(TOKEN)
    assert record["push_handle_digest"] != TOKEN


def test_a_different_phone_keeps_its_own_row() -> None:
    client = _client()
    first = client.post("/api/push/register", json=_body(str(uuid.uuid4()))).json()
    second = client.post(
        "/api/push/register", json=_body(str(uuid.uuid4()), token=OTHER_TOKEN)
    ).json()
    assert {row["device_id"] for row in _listed(client)} == {
        first["device_id"],
        second["device_id"],
    }
    assert _live_ids() == set() or _live_ids() == {first["device_id"], second["device_id"]}


def test_token_rotation_updates_the_same_row_in_place() -> None:
    client = _client()
    install = str(uuid.uuid4())
    first = client.post("/api/push/register", json=_body(install)).json()
    rotated = client.post("/api/push/register", json=_body(install, token=OTHER_TOKEN)).json()
    assert rotated["device_id"] == first["device_id"]
    (record,) = _stored()
    assert record["push_handle_digest"] == push_devices.push_handle_digest(OTHER_TOKEN)


def test_the_real_install_re_registering_restores_its_row_and_retires_the_ghost() -> None:
    """A transient fork followed by the phone's real identity: the real row wins back."""
    client = _client()
    real = str(uuid.uuid4())
    kept = client.post("/api/push/register", json=_body(real)).json()
    ghost = client.post("/api/push/register", json=_body(str(uuid.uuid4()))).json()
    assert [row["device_id"] for row in _listed(client)] == [ghost["device_id"]]

    again = client.post("/api/push/register", json=_body(real)).json()
    assert again["device_id"] == kept["device_id"]
    listed = _listed(client)
    assert [row["device_id"] for row in listed] == [kept["device_id"]]
    assert listed[0]["state"] == push_devices.STATE_LIVE
    (restored,) = [r for r in _stored() if r["device_id"] == kept["device_id"]]
    assert "superseded_at" not in restored and "expired_at" not in restored


def test_a_revoked_phone_cannot_return_under_a_fresh_install_id() -> None:
    """A revoke sticks to the PHONE: a forked identity with the same token is refused."""
    client = _client()
    first = client.post("/api/push/register", json=_body(str(uuid.uuid4()))).json()
    push_devices.revoke(config_dir(), first["device_id"])

    response = client.post("/api/push/register", json=_body(str(uuid.uuid4())))
    assert response.status_code == 403, response.text
    assert response.json()["code"] == push_devices.DEVICE_REVOKED_CODE
    # Nothing was written for the refused fork, and the tombstone is untouched.
    (record,) = _stored()
    assert record["revoked_at"]

    # A reinstall gets a new APNs token: that is a genuinely new device.
    fresh = client.post("/api/push/register", json=_body(str(uuid.uuid4()), token=OTHER_TOKEN))
    assert fresh.status_code == 200


def test_unrevoke_then_register_still_collapses_to_one_row() -> None:
    client = _client()
    install = str(uuid.uuid4())
    first = client.post("/api/push/register", json=_body(install)).json()
    push_devices.revoke(config_dir(), first["device_id"])
    push_devices.unrevoke(config_dir(), first["device_id"])
    again = client.post("/api/push/register", json=_body(install)).json()
    assert again["device_id"] == first["device_id"]
    assert [row["device_id"] for row in _listed(client)] == [first["device_id"]]


def _legacy_row(install_id: str, stamp: int, **extra: Any) -> dict[str, Any]:
    """A row as the pre-fix build wrote it: no push handle digest."""
    return {
        "device_id": uuid.uuid4().hex,
        "platform": "ios",
        "environment": "production",
        "app_version": "0.0.0 (1088)",
        "install_id": install_id,
        "registered_at": stamp,
        "last_seen_at": stamp,
        "credential_expires_at": stamp + 30 * 86400,
        "device_key": "k" * 43,
        **extra,
    }


def test_the_inherited_ghost_rows_collapse_on_the_first_register_after_upgrade() -> None:
    """The operator's measured store: one real row plus ghosts that never came back."""
    now = int(time.time())
    real_install = str(uuid.uuid4())
    real = _legacy_row(real_install, now - 86400, last_authenticated_at=now - 3600)
    ghosts = [_legacy_row(str(uuid.uuid4()), now - 80000) for _ in range(32)]
    revoked = _legacy_row(str(uuid.uuid4()), now - 90000, revoked_at=now - 85000)
    android = {**_legacy_row(str(uuid.uuid4()), now - 80000), "platform": "android"}
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(
        json.dumps({"devices": [real, *ghosts, revoked, android], "operator_key": "o" * 43}),
        encoding="utf-8",
    )
    _store_path().chmod(0o600)
    client = _client()

    response = client.post("/api/push/register", json=_body(real_install))
    assert response.status_code == 200, response.text
    assert response.json()["device_id"] == real["device_id"]

    listed = {row["device_id"]: row for row in _listed(client)}
    # The real phone, the revoked tombstone (an operator decision, never
    # rewritten) and the other platform's device remain; the 32 ghosts do not.
    assert set(listed) == {real["device_id"], revoked["device_id"], android["device_id"]}
    stored = {row["device_id"]: row for row in _stored()}
    for ghost in ghosts:
        assert stored[ghost["device_id"]]["superseded_at"]
        assert stored[ghost["device_id"]]["expired_at"]
    assert "superseded_at" not in stored[revoked["device_id"]]
    assert "expired_at" not in stored[revoked["device_id"]]


def test_a_legacy_row_that_proved_its_key_is_not_swept() -> None:
    """A pre-fix row whose device key was presented is a real device: keep it."""
    now = int(time.time())
    other = _legacy_row(str(uuid.uuid4()), now - 86400, last_authenticated_at=now - 60)
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(
        json.dumps({"devices": [other], "operator_key": "o" * 43}), encoding="utf-8"
    )
    _store_path().chmod(0o600)
    client = _client()
    mine = client.post("/api/push/register", json=_body(str(uuid.uuid4()))).json()
    assert {row["device_id"] for row in _listed(client)} == {
        other["device_id"],
        mine["device_id"],
    }


def test_superseded_rows_age_out_after_the_cloud_idle_drop() -> None:
    client = _client()
    old = client.post("/api/push/register", json=_body(str(uuid.uuid4()))).json()
    later = time.time() + 1
    push_devices.register(config_dir(), _body(str(uuid.uuid4())), now=later)
    assert len(_stored()) == 2
    beyond = later + (push_devices.CLOUD_IDLE_DROP_DAYS + 1) * 86400
    push_devices.register(config_dir(), _body(str(uuid.uuid4()), token=OTHER_TOKEN), now=beyond)
    assert old["device_id"] not in {row["device_id"] for row in _stored()}
