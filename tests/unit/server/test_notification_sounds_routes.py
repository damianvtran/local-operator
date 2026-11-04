"""``/v1/notification-sounds*``: status codes, envelopes, the gate and the capability.

The store's own rules are pinned in ``tests/unit/notifications/
test_custom_sounds.py``; this file pins what the desktop switches on: the HTTP
status and ``code`` per refusal (settings spec §5.2), the list shape, the audio
route's content type and ETag, ``selection_reset`` on DELETE, ``require_desktop``
on every route, and ``features.notification_sounds``. It runs the real router
against a real temporary config directory, the ``test_sidebar_layout`` rig.
"""

from __future__ import annotations

import io
import math
import os
import struct
import wave
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.notifications import custom_sounds
from local_operator.server import desktop as desktop_module
from local_operator.server.features import feature_flags
from local_operator.server.routes import notification_sounds

TOKEN = "synthetic-notification-sounds-token"


def tone(seconds: float) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 22050)))
                for i in range(int(seconds * 22050))
            )
        )
    return buffer.getvalue()


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setattr(desktop_module, "_CLAIMED", None)
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name, raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(notification_sounds.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


async def _upload(client: AsyncClient, raw: bytes, filename: str = "clip.wav", name: str = "Mine"):
    return await client.post(
        "/v1/notification-sounds",
        files={"file": (filename, raw, "audio/wav")},
        data={"name": name},
    )


@pytest.mark.asyncio
async def test_list_has_builtins_system_and_the_selection(desktop) -> None:
    client, _ = desktop
    body = (await client.get("/v1/notification-sounds")).json()["result"]
    ids = [row["id"] for row in body["sounds"]]
    assert ids[0] == "soft-pulse" and "system" in ids
    assert body["selected"] == "soft-pulse" and body["volume"] == 70 and body["enabled"] is True
    soft = body["sounds"][0]
    assert soft["builtin"] and soft["is_default"] and soft["label"] == "Soft pulse"


@pytest.mark.asyncio
async def test_upload_rename_audio_delete_round_trip(desktop) -> None:
    client, tmp_path = desktop
    created = await _upload(client, tone(0.5))
    assert created.status_code == 201, created.text
    row = created.json()["result"]
    assert row["id"].startswith("custom-mine-") and row["builtin"] is False

    listed = (await client.get("/v1/notification-sounds")).json()["result"]["sounds"]
    assert listed[-1]["id"] == row["id"]

    renamed = await client.patch(f"/v1/notification-sounds/{row['id']}", json={"name": "Ping"})
    assert renamed.status_code == 200 and renamed.json()["result"]["label"] == "Ping"

    audio = await client.get(f"/v1/notification-sounds/{row['id']}/audio?volume=70")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"
    again = await client.get(
        f"/v1/notification-sounds/{row['id']}/audio?volume=70",
        headers={"If-None-Match": audio.headers["etag"]},
    )
    assert again.status_code == 304

    from local_operator import settings_io

    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["notifications.sound.id"], row["id"]
    )
    deleted = await client.delete(f"/v1/notification-sounds/{row['id']}")
    result = deleted.json()["result"]
    assert result["selection_reset"] is True and result["selected"] == "soft-pulse"
    assert "Soft pulse is selected again" in result["message"]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "filename", "status", "code"),
    [
        (b"RIFF" + b"\0" * (custom_sounds.MAX_INPUT_BYTES + 10), "big.wav", 413, "too_large"),
        (b"OggS" + b"\0" * 24 + b"OpusHead" + b"\0" * 64, "a.opus", 415, "unsupported_format"),
        (None, "long.wav", 422, "too_long"),
        (b"just some text\n" * 50, "text.wav", 422, "undecodable"),
    ],
)
async def test_upload_refusals_carry_their_code(desktop, raw, filename, status, code) -> None:
    client, _ = desktop
    response = await _upload(client, tone(10.6) if raw is None else raw, filename)
    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code and detail["text"]
    assert detail["params"]["file"] == filename


@pytest.mark.asyncio
async def test_builtins_cannot_be_renamed_or_removed(desktop) -> None:
    client, _ = desktop
    assert (await client.delete("/v1/notification-sounds/soft-pulse")).status_code == 409
    response = await client.patch("/v1/notification-sounds/system", json={"name": "x"})
    assert response.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("hostile", ["..%2F..%2Fconfig.yml", "custom-..", "custom-x%2F..%2F..%2Fa"])
async def test_hostile_ids_are_not_found(desktop, hostile) -> None:
    client, _ = desktop
    for method, path in (
        ("get", f"/v1/notification-sounds/{hostile}/audio"),
        ("delete", f"/v1/notification-sounds/{hostile}"),
    ):
        response = await getattr(client, method)(path)
        assert response.status_code == 404, (method, path, response.text)


@pytest.mark.asyncio
async def test_every_route_requires_the_desktop_bearer(desktop) -> None:
    client, _ = desktop
    for method, path in (
        ("get", "/v1/notification-sounds"),
        ("get", "/v1/notification-sounds/soft-pulse/audio"),
        ("post", "/v1/notification-sounds"),
        ("patch", "/v1/notification-sounds/custom-x-000000"),
        ("delete", "/v1/notification-sounds/custom-x-000000"),
    ):
        missing = await client.request(method.upper(), path, headers={"Authorization": ""})
        assert missing.status_code == 401, path
        wrong = await client.request(method.upper(), path, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401, path


def test_the_capability_is_advertised() -> None:
    assert feature_flags()["notification_sounds"] == 1


# -- R2: nothing reads the body before auth, and the body is counted ------------


@pytest.fixture
def parsed_bytes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Bytes the multipart parser was fed, measured at the parser (the reviewer's probe)."""
    from starlette.formparsers import MultiPartParser

    seen = [0]
    original = MultiPartParser.on_part_data

    def counting(self, data: bytes, start: int, end: int) -> None:  # type: ignore[no-untyped-def]
        seen[0] += end - start
        return original(self, data, start, end)

    monkeypatch.setattr(MultiPartParser, "on_part_data", counting)
    return seen


def _multipart(size: int) -> tuple[bytes, str]:
    boundary = "qa-boundary-1234"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    return head + b"R" * size + f"\r\n--{boundary}--\r\n".encode(), boundary


@pytest.mark.asyncio
async def test_an_unauthorized_big_upload_is_refused_before_the_body_is_read(
    desktop, parsed_bytes
) -> None:
    client, _ = desktop
    body, boundary = _multipart(30 * 1024 * 1024)
    response = await client.post(
        "/v1/notification-sounds",
        content=body,
        headers={"Authorization": "", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 401
    assert parsed_bytes[0] == 0


@pytest.mark.asyncio
async def test_a_declared_oversize_body_is_refused_unread(desktop, parsed_bytes) -> None:
    client, _ = desktop
    body, boundary = _multipart(30 * 1024 * 1024)
    response = await client.post(
        "/v1/notification-sounds",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"
    assert parsed_bytes[0] == 0


@pytest.mark.asyncio
async def test_a_chunked_oversize_body_is_cut_off_at_the_cap(desktop, parsed_bytes) -> None:
    """No Content-Length (a chunked or lying sender): the counted stream stops it."""
    client, _ = desktop
    body, boundary = _multipart(30 * 1024 * 1024)

    async def chunks():  # type: ignore[no-untyped-def]
        for start in range(0, len(body), 256 * 1024):
            yield body[start : start + 256 * 1024]

    response = await client.post(
        "/v1/notification-sounds",
        content=chunks(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"
    limit = custom_sounds.MAX_INPUT_BYTES + notification_sounds._MULTIPART_SLACK
    assert 0 < parsed_bytes[0] <= limit + 256 * 1024


@pytest.mark.asyncio
async def test_a_body_without_a_file_part_is_422(desktop) -> None:
    client, _ = desktop
    response = await client.post("/v1/notification-sounds", data={"name": "x"})
    assert response.status_code == 422


# -- R6: If-None-Match per RFC 9110 ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["{etag}", "W/{etag}", '"other", {etag}', "*"])
async def test_if_none_match_lists_weak_tags_and_star(desktop, form) -> None:
    client, _ = desktop
    first = await client.get("/v1/notification-sounds/soft-pulse/audio?volume=40")
    etag = first.headers["etag"]
    again = await client.get(
        "/v1/notification-sounds/soft-pulse/audio?volume=40",
        headers={"If-None-Match": form.format(etag=etag)},
    )
    assert again.status_code == 304
    other = await client.get(
        "/v1/notification-sounds/soft-pulse/audio?volume=40",
        headers={"If-None-Match": '"nope"'},
    )
    assert other.status_code == 200 and other.content[:4] == b"RIFF"
