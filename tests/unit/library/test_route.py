from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.library import index
from local_operator.server.routes import capabilities, desktop_library
from local_operator.session.attachments import cache_media

from ._fixtures import (
    OTHER,
    SESSION,
    config_root,
    image_rows,
    png,
    session,
    user_row,
    write_rows,
)

pytestmark = pytest.mark.asyncio

TOKEN = "library-test-token"


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = config_root(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_library.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, root


def seed(root: Path) -> tuple[str, str, str]:
    """Return (asset id for the text file, image id, digest). Use the service to warm."""
    text = root.parent / "report.txt"
    text.write_text("a,b,c")
    block = cache_media(png(), "image/png", kind="image", name="cat.png")
    assert block and block.attachment
    session(root, SESSION, write_rows("w1", 1, text) + image_rows("g1", 2, block.attachment))
    session(root, OTHER, [user_row("u1", 3, f"see {text}")])
    catalog = index.build_catalog(root, budget_s=None)
    by_name = {asset.derived_name: asset.id for asset in catalog.assets}
    return by_name["report.txt"], by_name["cat.png"], block.attachment


async def test_list_detail_filters_and_the_bearer(api) -> None:
    client, root = api
    text_id, image_id, _digest = seed(root)

    denied = await client.get(
        "/v1/desktop/library/assets", headers={"Authorization": "Bearer wrong"}
    )
    assert denied.status_code == 401

    response = await client.get("/v1/desktop/library/assets")
    assert response.status_code == 200, response.text
    data = response.json()["result"]
    assert data["state"] == "ready" and data["total"] == 2
    assert data["scan"] == {"sessions_total": 2, "sessions_scanned": 2}
    row = next(row for row in data["rows"] if row["id"] == text_id)
    assert row["ref"]["path"] == str(root.parent / "report.txt")
    assert {item["session_id"] for item in row["occurrences"]} == {SESSION, OTHER}
    assert all("first_seq" in item and "last_seq" in item for item in row["occurrences"])

    filtered = await client.get(
        "/v1/desktop/library/assets",
        params={"kind": "image", "origin_group": "generated"},
    )
    assert filtered.json()["result"]["total"] == 1
    by_session = await client.get(
        "/v1/desktop/library/assets", params={"session_id": f"{SESSION},{OTHER}"}
    )
    assert by_session.json()["result"]["total"] == 2
    bad = await client.get("/v1/desktop/library/assets", params={"kind": "nope"})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "library_bad_filter"
    bad_session = await client.get(
        "/v1/desktop/library/assets", params={"session_id": "not-a-session"}
    )
    assert bad_session.status_code == 422

    detail = await client.get(f"/v1/desktop/library/assets/{image_id}")
    assert detail.status_code == 200
    assert detail.json()["result"]["preview"]["state"] == "ready"
    missing = await client.get("/v1/desktop/library/assets/la_ffffffffffffffff")
    assert missing.status_code == 404


async def test_metadata_tags_and_chat_round_trip(api) -> None:
    client, root = api
    text_id, _image_id, _digest = seed(root)

    patched = await client.patch(
        f"/v1/desktop/library/assets/{text_id}",
        json={"name": "Quarterly", "description": "Numbers", "favorite": True, "organized": True},
    )
    assert patched.status_code == 200, patched.text
    row = patched.json()["result"]
    assert row["name"] == "Quarterly" and row["favorite"] is True
    assert row["organized"]["by"] == "user"

    tag = await client.post("/v1/desktop/library/tags", json={"name": "finance"})
    assert tag.status_code == 200, tag.text
    tag_id = tag.json()["result"]["id"]
    tagged = await client.put(
        f"/v1/desktop/library/assets/{text_id}/tags",
        json={"tags": [tag_id], "primary": tag_id},
    )
    assert tagged.status_code == 200
    assert tagged.json()["result"]["tags"][0]["name"] == "finance"

    tags = await client.get("/v1/desktop/library/tags")
    assert any(item["name"] == "finance" and item["count"] == 1 for item in tags.json()["result"])
    renamed = await client.patch(
        f"/v1/desktop/library/tags/{tag_id}", json={"name": "money"}
    )
    assert renamed.json()["result"]["name"] == "money"

    chat_session = session(root, "111111111111")
    chat = await client.put(
        f"/v1/desktop/library/assets/{text_id}/chat",
        json={"session_id": "111111111111"},
    )
    assert chat.status_code == 200 and chat.json()["result"]["chat"]["session_id"] == "111111111111"
    assert chat_session.is_dir()
    gone = await client.put(
        f"/v1/desktop/library/assets/{text_id}/chat",
        json={"session_id": "222222222222"},
    )
    assert gone.status_code == 422
    cleared = await client.delete(f"/v1/desktop/library/assets/{text_id}/chat")
    assert cleared.json()["result"]["cleared"] is True
    assert cleared.json()["result"]["chat"] is None


async def test_content_range_download_and_thumbnail(api) -> None:
    client, root = api
    text_id, image_id, digest = seed(root)

    whole = await client.get(f"/v1/desktop/library/assets/{text_id}/content")
    assert whole.status_code == 200
    assert whole.content == b"a,b,c"
    assert whole.headers["content-disposition"].startswith("inline")
    assert whole.headers["x-content-type-options"] == "nosniff"
    assert whole.headers["cache-control"] == "private, max-age=60"

    partial = await client.get(
        f"/v1/desktop/library/assets/{text_id}/content",
        headers={"Range": "bytes=0-2"},
    )
    assert partial.status_code == 206
    assert partial.content == b"a,b"
    assert partial.headers["content-range"] == "bytes 0-2/5"

    download = await client.get(
        f"/v1/desktop/library/assets/{text_id}/content", params={"download": 1}
    )
    assert download.headers["content-disposition"].startswith("attachment")

    digest_content = await client.get(f"/v1/desktop/library/assets/{image_id}/content")
    assert digest_content.status_code == 200
    assert digest_content.headers["cache-control"].startswith("private, max-age=31536000")

    thumb = await client.get(
        f"/v1/desktop/library/assets/{image_id}/thumbnail", params={"w": 120}
    )
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] in ("image/webp", "image/png")

    unsupported = await client.get(f"/v1/desktop/library/assets/{text_id}/thumbnail")
    assert unsupported.status_code == 404

    # The wire accepts no path input at all; only the opaque id.
    traversal = await client.get(
        "/v1/desktop/library/assets/la_0123456789abcdef/content", params={"path": "/etc/passwd"}
    )
    assert traversal.status_code in (404, 422)


async def test_capability_is_advertised(api) -> None:
    client, _root = api
    response = await client.get("/v1/capabilities")
    assert response.status_code == 200
    assert response.json()["result"]["features"]["library"] == 1
