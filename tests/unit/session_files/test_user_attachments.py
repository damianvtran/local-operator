"""The user-attachment inventory: scanner rules and the ``…/user-attachments`` route.

Row shapes are copied from real journals (read 2026-10-27), not invented:

* a durable PDF on a user row (session ``519d73620f7b``): ``{mime_type, filename, path,
  stored_path, bytes, injection, type: "document", attachment}``;
* a durable pasted screenshot: ``{attachment, mime_type}`` - NO ``type`` key, because the
  encoder dumps with ``exclude_defaults``;
* an inline screenshot row (written without a store): ``{data, mime_type}``;
* the legacy document shape ``{filename, mime_type, attachment}`` with no ``type``.

The route is driven through the REAL router, admission (``files_target``) and failure ladder,
like ``test_route.py``.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server import features
from local_operator.server.routes import desktop_session_files, desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attachments import store_for_transcript_dir
from local_operator.session.user_attachments import list_user_attachments

from ._fixtures import SESSION, png, write_journal

TOKEN = "user-attachments-test-token"
URL = f"/v1/desktop/sessions/{SESSION}/user-attachments"


def _user(row_id: str, ts: float, blocks: list[Any], **payload: Any) -> dict[str, Any]:
    return {
        "id": row_id,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": blocks, **payload},
    }


def _session(root: Path) -> Path:
    directory = root / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root.parent)}))
    return directory


def _stored(directory: Path, raw: bytes, mime: str) -> tuple[str, Path]:
    ref = store_for_transcript_dir(directory).put_bytes(raw, mime)
    assert ref is not None
    return ref.digest, store_for_transcript_dir(directory).path_for(ref.digest)


def _seed(root: Path) -> tuple[Path, dict[str, Any]]:
    """One journal holding every real shape plus the rows that must NOT be listed."""
    directory = _session(root)
    pdf_digest, pdf_store = _stored(directory, b"%PDF-1.7 hello", "application/pdf")
    shot_digest, _ = _stored(directory, png(seed=3), "image/png")
    inline_retained = png(seed=5)
    retained_digest, _ = _stored(directory, inline_retained, "image/png")
    origin = root.parent / "Downloads" / "IP Assignment.pdf"
    rows = [
        {"id": "c1", "ts": 1.0, "type": "custom", "payload": {"custom_type": "selected_model"}},
        _user(
            "u1",
            10.0,
            [
                {"text": "see attached"},
                {
                    "mime_type": "application/pdf",
                    "filename": "IP Assignment.pdf",
                    "path": str(origin),
                    "stored_path": str(pdf_store),
                    "bytes": 14,
                    "injection": "native",
                    "type": "document",
                    "attachment": pdf_digest,
                },
                {"attachment": shot_digest, "mime_type": "image/png"},
            ],
            input_mode="typed",
        ),
        # An assistant's image and a tool's artifact are agent output, never user attachments.
        {
            "id": "a1",
            "ts": 11.0,
            "type": "message",
            "payload": {"role": "assistant", "content": [{"attachment": shot_digest}]},
        },
        {
            "id": "t1",
            "ts": 12.0,
            "type": "message",
            "payload": {
                "role": "tool",
                "content": [{"kind": "image", "attachment": shot_digest, "content_type": "x"}],
            },
        },
        # A harness-minted user row: never the person's attachment.
        _user(
            "h1",
            13.0,
            [{"attachment": shot_digest, "mime_type": "image/png"}],
            provider_payload={"harness_injected": True},
        ),
        # Legacy document shape, store copy gone; inline screenshot whose bytes ARE retained;
        # inline screenshot with a defaulted (absent) mime whose bytes are NOT retained.
        _user(
            "u2",
            20.0,
            [
                {"filename": "old.txt", "mime_type": "text/plain", "attachment": "0" * 32},
                {"data": base64.b64encode(inline_retained).decode(), "mime_type": "image/png"},
                {"data": base64.b64encode(png(seed=9)).decode()},
                "not-a-block",
                {"type": "document"},  # malformed: no filename/mime
            ],
        ),
    ]
    write_journal(directory, rows)
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"role":"user", this is not json\n')
    return directory, {"pdf": pdf_digest, "shot": shot_digest, "retained": retained_digest}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    return cfg


def test_lists_user_documents_and_images_in_send_order(root: Path) -> None:
    directory, digests = _seed(root)
    listing = list_user_attachments(directory)
    items = listing["items"]
    assert listing["truncated"] is False
    assert [(i["message_id"], i["index"], i["kind"]) for i in items] == [
        ("u1", 1, "document"),
        ("u1", 2, "image"),
        ("u2", 0, "document"),
        ("u2", 1, "image"),
        ("u2", 2, "image"),
        ("u2", 4, "document"),
    ]
    pdf, shot, legacy, inline_kept, inline_lost, malformed = items
    assert pdf["attachment"] == digests["pdf"] and pdf["filename"] == "IP Assignment.pdf"
    assert pdf["bytes"] == 14 and pdf["injection"] == "native" and pdf["stored"] is True
    assert pdf["path"].endswith("IP Assignment.pdf") and pdf["ts"] == 10.0
    assert shot["attachment"] == digests["shot"] and shot["stored"] is True
    assert shot["filename"] is None and shot["mime_type"] == "image/png"
    # A durable image row records no size; the retained copy's length stands in for it.
    assert shot["bytes"] == len(png(seed=3))
    # Legacy: digest on the row, store copy missing -> stored False, path derived from digest.
    assert legacy["filename"] == "old.txt" and legacy["stored"] is False
    assert legacy["stored_path"].endswith("0" * 32 + ".bin")
    # Inline bytes the store holds resolve to their digest (read-only lookup) ...
    assert inline_kept["attachment"] == digests["retained"] and inline_kept["stored"] is True
    # ... and inline bytes it does not hold are reported without a digest, never inlined.
    assert inline_lost["attachment"] is None and inline_lost["stored"] is False
    assert inline_lost["mime_type"] == "image/png" and inline_lost["bytes"] > 0
    assert malformed["filename"] is None
    assert all("data" not in item for item in items), "metadata only - no base64 on the wire"


def test_read_path_never_writes_the_store(root: Path) -> None:
    directory, _ = _seed(root)
    store_dir = store_for_transcript_dir(directory).root
    before = sorted(p.name for p in store_dir.iterdir())
    list_user_attachments(directory)
    assert sorted(p.name for p in store_dir.iterdir()) == before


def test_no_journal_is_an_empty_list(root: Path) -> None:
    assert list_user_attachments(_session(root)) == {"items": [], "truncated": False}


def test_cap_reports_truncation(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.session import user_attachments

    directory, _ = _seed(root)
    monkeypatch.setattr(user_attachments, "MAX_ITEMS", 2)
    listing = list_user_attachments(directory)
    assert len(listing["items"]) == 2 and listing["truncated"] is True


def test_capability_is_advertised() -> None:
    assert features.static_feature_flags()["session_attachments"] == 1


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(cfg)
    app.state.desktop_sessions = DesktopSessions(cfg)
    app.include_router(desktop_session_files.router)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, cfg
    await app.state.desktop_sessions.close()


@pytest.mark.asyncio
async def test_route_lists_items_and_requires_the_bearer(api) -> None:
    client, cfg = api
    _, digests = _seed(cfg)
    response = await client.get(URL)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["state"] == "ready" and result["truncated"] is False
    assert [i["attachment"] for i in result["items"][:2]] == [digests["pdf"], digests["shot"]]
    assert "data" not in response.text
    unauthorised = await client.get(URL, headers={"Authorization": "Bearer wrong"})
    assert unauthorised.status_code in (401, 403)


@pytest.mark.asyncio
async def test_route_unknown_and_malformed_ids(api) -> None:
    client, _ = api
    assert (
        await client.get("/v1/desktop/sessions/aaaaaaaaaaaa/user-attachments")
    ).status_code == 404
    assert (await client.get("/v1/desktop/sessions/NOT-AN-ID/user-attachments")).status_code == 422


@pytest.mark.asyncio
async def test_route_undecodable_archive_is_unavailable_not_empty(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session import compression, user_attachments

    client, cfg = api
    _seed(cfg)

    def boom(_directory: Path) -> dict[str, Any]:
        raise compression.ArchiveUnavailable("zstd missing")

    monkeypatch.setattr(user_attachments, "list_user_attachments", boom)
    result = (await client.get(URL)).json()["result"]
    assert result == {"state": "unavailable", "items": [], "truncated": False}
