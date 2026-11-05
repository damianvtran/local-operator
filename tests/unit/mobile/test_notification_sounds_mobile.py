"""The paired phone's notification-sound routes and the custom-sound push fields (C5).

What this file pins (notification-sounds addendum 2 §2.2-2.3):

* ``GET /api/notification-sounds`` — the cookie gate, the list shape, ``no-store``,
  and that the rows are the same set and order the desktop list shows;
* ``GET /api/notification-sounds/{id}/audio?sha=`` — the bytes hash to the listed
  sha, a wrong or stale sha is a 404 (never the current bytes), the cache policy
  is ``immutable`` with ``ETag`` = sha, ``Range`` answers 206, unity gain;
* ``PUT /api/notification-sounds/cached`` — authenticated by the device's own
  ``X-Lop-Device-Key``, refuses a missing/wrong key and a cross-origin write,
  validates the ``c_<sha16>`` stems, and replaces the device's list;
* the push: a custom selection emits ``sound_custom`` beside the soft-pulse
  fallback and ``sound_custom_ready`` per live device; a built-in emits neither.

The real daemon app over the test's isolated HOME/config (``isolate_environment``),
the real custom store, synthetic install ids. No sound is ever played.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import push_credentials, push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.notifications import catalog_view, custom_sounds
from local_operator.notifications import sound as _sound
from local_operator.paths import config_dir

INSTALL_A = "11111111-2222-4333-8444-555555555555"
INSTALL_B = "66666666-7777-4888-9999-aaaaaaaaaaaa"


def tone(seconds: float = 0.4) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(
            b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * 660 * i / 22050)))
                for i in range(int(seconds * 22050))
            )
        )
    return buffer.getvalue()


def _client(*, login: bool = True) -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    if login:
        assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _register(client: TestClient, install_id: str) -> dict[str, Any]:
    response = client.post(
        "/api/push/register",
        json={
            "platform": "ios",
            "token": "synthetic-token",
            "environment": "sandbox",
            "app_version": "1.0.0 (1)",
            "install_id": install_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _select(sound_id: str, *, enabled: bool = True) -> None:
    (config_dir()).mkdir(parents=True, exist_ok=True)
    (config_dir() / "config.yml").write_text(
        "values:\n  notifications:\n    sound:\n"
        f"      enabled: {'true' if enabled else 'false'}\n      id: {sound_id}\n",
        encoding="utf-8",
    )


def _custom() -> custom_sounds.CustomSound:
    return custom_sounds.add(tone(), source_name="qa.wav", name="QA chime")


# -- the list ---------------------------------------------------------------------


def test_the_list_needs_the_cookie() -> None:
    response = _client(login=False).get("/api/notification-sounds")
    assert response.status_code == 401


def test_the_list_names_every_fetchable_sound_by_the_bytes_it_serves() -> None:
    row = _custom()
    _select(row.id)
    client = _client()
    response = client.get("/api/notification-sounds")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    listing = response.json()
    ids = [sound["id"] for sound in listing["sounds"]]
    # Same set and order as every other surface (``sound_ids``).
    assert ids == list(_sound.sound_ids())
    assert listing["selected"] == row.id and listing["enabled"] is True
    assert listing["sound_custom"] == f"c_{row.sha256[:16]}"
    by_id = {sound["id"]: sound for sound in listing["sounds"]}
    custom = by_id[row.id]
    assert custom["builtin"] is False and custom["label"] == "QA chime"
    assert custom["push_name"] == f"c_{row.sha256[:16]}.wav"
    assert by_id["soft-pulse"]["push_name"] == "soft_pulse.wav"
    system = by_id[_sound.SYSTEM_SOUND_ID]
    assert system["sha256"] is None and system["push_name"] is None
    # Every listed sha IS the sha of the bytes the audio route serves.
    for sound in listing["sounds"]:
        if sound["sha256"] is None:
            continue
        audio = client.get(f"/api/notification-sounds/{sound['id']}/audio?sha={sound['sha256']}")
        assert audio.status_code == 200, sound["id"]
        assert hashlib.sha256(audio.content).hexdigest() == sound["sha256"]
        assert len(audio.content) == sound["size_bytes"]


def test_a_built_in_or_disabled_selection_names_no_custom_sound() -> None:
    row = _custom()
    _select("felt-marimba")
    assert _client().get("/api/notification-sounds").json()["sound_custom"] is None
    _select(row.id, enabled=False)
    assert _client().get("/api/notification-sounds").json()["sound_custom"] is None


# -- the audio --------------------------------------------------------------------


def test_the_audio_is_content_addressed_and_ranged() -> None:
    row = _custom()
    client = _client()
    url = f"/api/notification-sounds/{row.id}/audio?sha={row.sha256}"
    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["content-type"] == "audio/wav"
    assert full.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert full.headers["etag"] == f'"{row.sha256}"'
    assert full.headers["x-content-type-options"] == "nosniff"
    # Unity gain: the stored file byte for byte, whatever the computer's volume.
    assert full.content == custom_sounds.sound_path(row.id).read_bytes()  # type: ignore[union-attr]
    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{len(full.content)}"
    assert part.content == full.content[:100]
    again = client.get(url, headers={"If-None-Match": f'"{row.sha256}"'})
    assert again.status_code == 304


def test_a_wrong_or_stale_sha_is_a_404_never_the_current_bytes() -> None:
    row = _custom()
    client = _client()
    wrong = client.get(f"/api/notification-sounds/{row.id}/audio?sha={'0' * 64}")
    assert wrong.status_code == 404
    assert wrong.headers["cache-control"] == "no-store"
    assert client.get(f"/api/notification-sounds/{row.id}/audio?sha=nothex").status_code == 400
    assert (
        client.get("/api/notification-sounds/custom-nope/audio?sha=" + "0" * 64).status_code == 404
    )
    assert (
        client.get(
            f"/api/notification-sounds/{_sound.SYSTEM_SOUND_ID}/audio?sha={'0' * 64}"
        ).status_code
        == 404
    )
    unauthenticated = _client(login=False).get(
        f"/api/notification-sounds/{row.id}/audio?sha={row.sha256}"
    )
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["x-content-type-options"] == "nosniff"


# -- the cache report -------------------------------------------------------------


def _put(client: TestClient, install_id: str | None, key: str | None, shas: Any, **headers: str):
    sent = dict(headers)
    if install_id is not None:
        sent[push_devices.DEVICE_HEADER] = install_id
    if key is not None:
        sent[push_devices.DEVICE_KEY_HEADER] = key
    return client.put("/api/notification-sounds/cached", json={"shas": shas}, headers=sent)


def _stored_cache(install_id: str) -> list[str] | None:
    import json

    store = json.loads((config_dir() / push_devices.PUSH_DEVICES_STORE_NAME).read_text())
    row = next(r for r in store["devices"] if r["install_id"] == install_id)
    return row.get("sound_cached")


def test_the_cache_report_is_the_devices_own_and_needs_its_key() -> None:
    client = _client()
    key = _register(client, INSTALL_A)["device_key"]
    stem = "c_0123456789abcdef"
    assert _put(client, INSTALL_A, None, [stem]).status_code == 401
    assert _put(client, INSTALL_A, "not-the-key", [stem]).status_code == 401
    assert _put(client, INSTALL_B, key, [stem]).status_code == 401
    assert _stored_cache(INSTALL_A) is None
    accepted = _put(client, INSTALL_A, key, [stem, stem])
    assert accepted.status_code == 200
    assert accepted.json()["sound_cached"] == [stem]
    assert _stored_cache(INSTALL_A) == [stem]
    # The list REPLACES: an evicted sound stops counting.
    assert _put(client, INSTALL_A, key, []).json()["sound_cached"] == []
    assert _stored_cache(INSTALL_A) == []


def test_the_cache_report_refuses_bad_stems_cross_origin_and_no_cookie() -> None:
    client = _client()
    key = _register(client, INSTALL_A)["device_key"]
    for bad in (["c-0123456789abcdef"], ["0123456789abcdef"], "c_0123456789abcdef", [1]):
        assert _put(client, INSTALL_A, key, bad).status_code == 422, bad
    too_many = [f"c_{i:016x}" for i in range(push_devices.MAX_SOUND_CACHED + 1)]
    assert _put(client, INSTALL_A, key, too_many).status_code == 422
    cross = _put(client, INSTALL_A, key, ["c_0123456789abcdef"], origin="https://evil.example")
    assert cross.status_code == 403
    anonymous = _put(_client(login=False), INSTALL_A, key, ["c_0123456789abcdef"])
    assert anonymous.status_code == 401
    assert _stored_cache(INSTALL_A) is None


def test_a_revoked_device_cannot_report_its_cache() -> None:
    client = _client()
    registered = _register(client, INSTALL_A)
    assert client.delete(f"/api/push/devices/{registered['device_id']}").status_code == 200
    refused = _put(client, INSTALL_A, registered["device_key"], ["c_0123456789abcdef"])
    assert refused.status_code == 403
    assert refused.json()["code"] == push_devices.DEVICE_REVOKED_CODE


def test_the_capability_is_advertised() -> None:
    response = _client().get("/api/sessions")
    capabilities = response.json().get("capabilities", {})
    assert capabilities.get("notification_sounds") == {"available": True}


# -- the push fields --------------------------------------------------------------


def test_readiness_is_per_device_and_covers_every_live_device() -> None:
    client = _client()
    key_a = _register(client, INSTALL_A)["device_key"]
    _register(client, INSTALL_B)
    stem = "c_0123456789abcdef"
    assert _put(client, INSTALL_A, key_a, [stem]).status_code == 200
    facts = push_devices.credential_facts(config_dir())
    # A device the credential coalescer had nothing new about still gets a row.
    rows = push_credentials.with_sound_readiness([], facts, stem, now=0)
    by_install = {row["device_id"]: row[push_credentials.REPORT_SOUND_READY_FIELD] for row in rows}
    ids = {fact["device_id"]: fact for fact in facts}
    assert len(rows) == 2 and set(by_install) == set(ids)
    ready = [device for device, flag in by_install.items() if flag]
    assert len(ready) == 1
    # A different sha (the selection changed) makes nobody ready.
    other = push_credentials.with_sound_readiness([], facts, "c_fedcba9876543210", now=0)
    assert not any(row[push_credentials.REPORT_SOUND_READY_FIELD] for row in other)


def test_selected_push_stem_follows_the_selection() -> None:
    row = _custom()
    _select(row.id)
    assert catalog_view.selected_push_stem() == f"c_{row.sha256[:16]}"
    _select("soft-pulse")
    assert catalog_view.selected_push_stem() is None
    _select("custom-gone")
    assert catalog_view.selected_push_stem() is None


@pytest.fixture
def worker_harness() -> Any:
    """The push-worker suite's own harness, rooted at the isolated config dir.

    Rooted there (not at ``tmp_path``) because the custom store and the device
    registry both read ``config_dir()``; the harness's ``config_dir`` is the
    worker's, so the three agree.
    """
    from tests.unit.mobile.test_push_worker import Harness

    root = config_dir()
    root.mkdir(parents=True, exist_ok=True)
    harness = Harness(root)
    # The real registry decides who is live here, not the harness's stub.
    harness.worker._live_devices = harness.worker._live_device_ids  # type: ignore[method-assign]
    return harness


def test_a_custom_selection_emits_sound_custom_and_per_device_readiness(
    worker_harness: Any,
) -> None:
    client = _client()
    key_a = _register(client, INSTALL_A)["device_key"]
    _register(client, INSTALL_B)
    row = _custom()
    stem = f"c_{row.sha256[:16]}"
    assert _put(client, INSTALL_A, key_a, [stem]).status_code == 200
    _select(row.id)
    worker_harness.publish("custom")
    assert [r.kind for r in worker_harness.worker.tick()] == ["completion"]
    body = worker_harness.plane.calls[-1][1]
    assert body["sound"] == "soft-pulse"
    assert body["sound_custom"] == stem
    flags = sorted(device["sound_custom_ready"] for device in body["devices"])
    assert flags == [False, True]

    _select("felt-marimba")
    worker_harness.publish("builtin")
    worker_harness.worker.tick()
    body = worker_harness.plane.calls[-1][1]
    assert body["sound"] == "felt-marimba"
    assert "sound_custom" not in body
    assert all("sound_custom_ready" not in device for device in body.get("devices", []))
