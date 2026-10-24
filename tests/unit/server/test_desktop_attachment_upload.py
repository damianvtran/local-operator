"""``POST /v1/desktop/attachments`` — the composer's upload door.

The route is the only path that moves attachment BYTES: everything after it
carries ``Prompt.files`` references. These tests prove the rules the frozen
contract states (§1.1): content-decided MIME, the 32 MB cap, idempotence by
digest, and a sidecar that remembers MIME and size but never a name or a path
(amendment 7).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_sessions

TOKEN = "desktop-attachments-token"


@pytest_asyncio.fixture
async def upload_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path / "cfg" / "attachments"


def _pdf() -> bytes:
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


async def _upload(client: AsyncClient, raw: bytes, **data):
    files = {"file": (data.pop("filename", "file.bin"), raw, "application/octet-stream")}
    return await client.post("/v1/desktop/attachments", files=files, data=data)


@pytest.mark.asyncio
async def test_the_upload_returns_a_digest_bearing_reference(upload_api) -> None:
    client, store_dir = upload_api
    raw = _pdf()
    response = await _upload(
        client,
        raw,
        name="Complete_With_Docusign.pdf",
        origin_path="/Users/victor/Documents/Complete_With_Docusign.pdf",
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    digest = hashlib.sha256(raw).hexdigest()[:32]
    assert result["digest"] == digest
    assert result["mime_type"] == "application/pdf"
    assert result["bytes"] == len(raw)
    assert result["name"] == "Complete_With_Docusign.pdf"
    assert result["path"] == "/Users/victor/Documents/Complete_With_Docusign.pdf"
    stored = Path(result["stored_path"])
    assert stored == store_dir / f"{digest}.bin"
    assert stored.read_bytes() == raw


@pytest.mark.asyncio
async def test_mime_comes_from_the_bytes_not_the_filename(upload_api) -> None:
    client, _ = upload_api
    response = await _upload(client, _pdf(), name="disguised.txt")
    assert response.status_code == 200, response.text
    assert response.json()["result"]["mime_type"] == "application/pdf"

    markdown = await _upload(client, b"# notes\n", name="notes.md")
    assert markdown.json()["result"]["mime_type"] == "text/markdown"

    opaque = await _upload(client, b"\x00\x01\x02", name="blob.dat")
    assert opaque.json()["result"]["mime_type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_a_repeat_upload_is_the_same_record_without_inheriting_the_first_name(
    upload_api,
) -> None:
    client, store_dir = upload_api
    raw = _pdf()
    first = await _upload(client, raw, name="first.pdf", origin_path="/tmp/first.pdf")
    second = await _upload(client, raw, name="second.pdf", origin_path="/tmp/second.pdf")
    assert first.json()["result"]["digest"] == second.json()["result"]["digest"]
    assert second.json()["result"]["name"] == "second.pdf"
    assert second.json()["result"]["path"] == "/tmp/second.pdf"
    # Amendment 7: the store's sidecar is provenance-free, so a later send can
    # never read an earlier chip's filename back out of it.
    digest = first.json()["result"]["digest"]
    sidecar = json.loads((store_dir / f"{digest}.json").read_text(encoding="utf-8"))
    assert set(sidecar) <= {"digest", "mime_type", "bytes"}
    assert "name" not in sidecar and "origin_path" not in sidecar


@pytest.mark.asyncio
async def test_an_empty_file_is_refused_by_name(upload_api) -> None:
    client, _ = upload_api
    response = await _upload(client, b"", name="empty.pdf")
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == "attachment_empty"
    assert "empty.pdf" in response.json()["detail"]["text"]


@pytest.mark.asyncio
async def test_a_file_over_the_cap_is_refused(upload_api, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.session import attachments

    monkeypatch.setattr(attachments, "MAX_ATTACHMENT_UPLOAD_BYTES", 16)
    client, _ = upload_api
    response = await _upload(client, b"x" * 17, name="big.txt")
    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "attachment_too_large"


# ---------------------------------------------------------------------------
# Review round 1: the door's carriage hop (m2) and ack classification (m1)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def message_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real ``/messages`` door over a pool whose remote records the call."""
    from types import SimpleNamespace

    from local_operator.config import ConfigManager
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)

    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_sessions.router)
    sid = await pool.create(str(tmp_path))

    seen: dict[str, object] = {}

    class Recorder(SimpleNamespace):
        def __getattr__(self, name: str):
            if name.startswith("_"):
                raise AttributeError(name)

            async def inert(*_args: object, **_kwargs: object) -> None:
                return None

            return inert

        async def admit_prompt(self, text: str, **kwargs: object):
            seen["text"] = text
            seen["kwargs"] = kwargs
            return "prompt admitted", False

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client, pool, sid, seen, Recorder
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_the_message_route_hands_files_to_the_owner(message_api) -> None:
    """m2: the one new line at the HTTP door had no test.

    The reproduction is a composer send; this drives ``POST /messages`` with
    ``files`` through the real route and asserts the owner's ``admit_prompt``
    receives them — the hop that, if dropped, is invisible to every unit test
    of the layers below it.
    """
    from types import SimpleNamespace
    from typing import Any, cast

    client, pool, sid, seen, Recorder = message_api
    digest = "a" * 32
    async with pool.session(sid) as bridge:
        bridge.remote = cast(
            Any,
            Recorder(
                is_cold=False,
                frontend_state=SimpleNamespace(epoch="epoch-1"),
            ),
        )
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/messages",
            json={
                "request_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "text": "read this",
                "mode": "prompt",
                "files": [{"digest": digest, "name": "report.pdf", "bytes": 1234}],
            },
        )
    assert response.status_code == 200, response.text
    assert seen["text"] == "read this"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    files = kwargs["files"]
    assert isinstance(files, list)
    assert files[0]["digest"] == digest
    assert files[0]["name"] == "report.pdf"


def test_the_ack_classifies_the_new_refusals() -> None:
    """m1: the published admission outcome named none of the new refusals."""
    from local_operator.server.routes.desktop_sessions import _admission_failure_detail
    from local_operator.session.errors import (
        AttachmentMissing,
        DocumentCarriageUnsupported,
        RemoteDocumentUnsupported,
    )

    missing = _admission_failure_detail(AttachmentMissing(filename="report.pdf"))
    assert "report.pdf" in missing
    assert "could not be found" in missing

    carriage = _admission_failure_detail(DocumentCarriageUnsupported())
    assert "attachment" in carriage.lower() and "document" in carriage.lower()

    remote = _admission_failure_detail(RemoteDocumentUnsupported())
    assert "document" in remote.lower()

    # The name is bounded and stripped: a control-laden 500-char name cannot
    # become an unbounded frame field.
    hostile = _admission_failure_detail(AttachmentMissing(filename="\x00" + "x" * 500))
    assert "\x00" not in hostile
    assert len(hostile) < 200
