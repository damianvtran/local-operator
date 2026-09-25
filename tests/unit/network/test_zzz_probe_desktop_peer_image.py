"""PROBE (temporary): the desktop's image read for a PEER's conversation.

THE TWO-MACHINE FACT THIS MODELS. ``Transcript`` externalises an image into
``AttachmentStore()`` == ``config_dir()/attachments`` — the store of the process
that WROTE the row (``session/attachments.py::store_for_transcript_dir`` says so
and derives it from the session directory for that reason). A peer's runtime is
another process on another machine, so the bytes its turn references land in the
PEER's store. The desktop's read (``DesktopSessions.attachment``) resolves
``self.root/attachments`` and nothing else, then answers ``PeerAttachmentUn``.

This probe drives the real route over two real relays and prints:
  1. what the owner's transcript holds after an image-carrying prompt;
  2. what the desktop's own attachment read answers ONCE the bytes are where
     production puts them (the peer's store, not this device's);
  3. the LOCAL equivalent, for contrast: a local session's image read serves.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import shutil
import uuid
import zlib
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    devices,
)
from tests.unit.network.test_session_plane import (
    _create_named_session_on_a_real_peer,
    _NamedRemoteCreate,
    Devices,
)

TOKEN = "probe-desktop-token"


def _png(width: int = 240, height: int = 240) -> tuple[str, bytes]:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    rng = random.Random(7)
    rows = [
        b"\x00"
        + bytes(
            v
            for _ in range(width)
            for v in (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        )
        for _ in range(height)
    ]
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00",
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii"), png


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


@pytest_asyncio.fixture()
async def desktop_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peer_pair: Devices):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    server_a, server_b, _h, _p = peer_pair
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    app = FastAPI()
    app.state.config_manager = ConfigManager(server_a.root)
    app.state.desktop_sessions = DesktopSessions(server_a.root)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, server_a, server_b, app


def _digest(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()[:32]


@pytest.mark.asyncio
async def test_probe_peer_image_read(
    tmp_path: Path, peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, desktop_api: Any
) -> None:
    client, server_a, server_b, _app = desktop_api
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="peer-image-read-probe",
        prompt="warm up",
    )
    try:
        from local_operator.session.peer_rows import clear_cache

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
        await asyncio.to_thread(clear_cache)
        b64, raw = _png()
        digest = _digest(raw)
        response = await client.post(
            f"/v1/desktop/sessions/{created.session_id}/messages",
            json={
                "request_id": str(uuid.uuid4()),
                "text": "here is a picture",
                "images": [{"data_b64": b64, "mime_type": "image/png"}],
            },
        )
        await asyncio.to_thread(created.owner.wait_for_turn)
        rows = [
            entry.get("payload") or {}
            for entry in created.owner.transcript_entries()
            if (entry.get("payload") or {}).get("role") == "user"
        ]
        print("1. PROMPT HTTP:", response.status_code, response.text[:160])
        print("1. OWNER USER ROWS:", json.dumps(rows)[:900])
        print("1. DIGEST OF WHAT WAS SENT:", digest)

        ambient = Path(server_a.root) / "attachments"
        own = Path(server_b.root) / "attachments"
        print(
            "1. AMBIENT STORE (this rig shares one env):",
            sorted(p.name for p in ambient.glob("*")) if ambient.exists() else "MISSING",
        )
        # PUT THE BYTES WHERE PRODUCTION PUTS THEM: the peer's own store, because
        # its runtime is its own process with its own config dir.
        own.mkdir(parents=True, exist_ok=True)
        moved = []
        for suffix in (".bin", ".json"):
            source = ambient / f"{digest}{suffix}"
            if source.exists():
                shutil.move(str(source), str(own / source.name))
                moved.append(source.name)
        print("2. MOVED TO THE PEER'S STORE:", moved)
        print(
            "2. AMBIENT STORE NOW:",
            sorted(p.name for p in ambient.glob("*")) if ambient.exists() else "MISSING",
        )

        read = await client.get(f"/v1/desktop/sessions/{created.session_id}/attachments/{digest}")
        print("2. DESKTOP READ OF THE PEER'S IMAGE:", read.status_code, read.text[:400])

        # THE LOCAL EQUIVALENT. For a session THIS device owns, the writer's store
        # and the reader's store are the same directory (``config_dir()/attachments``
        # — one process, one config dir), so the read resolves. Driving that here is
        # putting the bytes back where a local prompt's own externalise step leaves
        # them, and reading the same two routes.
        from local_operator.server.utils.desktop_sessions import DesktopSessions

        ambient.mkdir(parents=True, exist_ok=True)
        for suffix in (".bin", ".json"):
            source = own / f"{digest}{suffix}"
            if source.exists():
                shutil.copy2(str(source), str(ambient / source.name))
        local_pool = DesktopSessions(server_a.root)
        try:
            local_id = await local_pool.create(str(tmp_path), model=None)
            local_read = await client.get(
                f"/v1/desktop/sessions/{local_id}/attachments/{digest}"
            )
            print(
                "3. LOCAL READ OF THE SAME BYTES:",
                local_read.status_code,
                local_read.headers.get("content-type"),
                len(local_read.content),
                "matches the sent bytes:",
                local_read.content == raw,
            )
            again = await client.get(
                f"/v1/desktop/sessions/{created.session_id}/attachments/{digest}"
            )
            print(
                "4. THE PEER'S SESSION, SAME READ, ONCE THIS DEVICE HOLDS THE BYTES:",
                again.status_code,
                "matches the sent bytes:",
                again.content == raw,
            )
        finally:
            await local_pool.close()
    finally:
        await asyncio.to_thread(created.stop)
