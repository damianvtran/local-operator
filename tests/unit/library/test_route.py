from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.library import indexer, service
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
    # Build the index the daemon would have built, then read the ids back from
    # the same SQLite tables the route reads.
    indexer.rebuild_now(root)
    frame = service.list_assets(root, service.Query())
    by_name = {row["name"]: row["id"] for row in frame["rows"]}
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
    renamed = await client.patch(f"/v1/desktop/library/tags/{tag_id}", json={"name": "money"})
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

    thumb = await client.get(f"/v1/desktop/library/assets/{image_id}/thumbnail", params={"w": 120})
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


async def test_patch_bad_fields_are_422_not_500(api) -> None:
    client, root = api
    text_id, _image_id, _digest = seed(root)

    for body in (
        {"organized": "garbage"},
        {"organized": None},
        {"favorite": None},
        {"name": None},
        {"name": ""},
        {"name": "   "},
    ):
        response = await client.patch(f"/v1/desktop/library/assets/{text_id}", json=body)
        assert response.status_code == 422, (body, response.status_code, response.text)
        detail = response.json()["detail"]
        assert detail["code"] == "library_bad_filter"
        assert detail["params"]["field"] in {"organized", "name", "body"}


async def test_patch_works_before_and_after_a_listing(api) -> None:
    client, root = api
    text_id, _image_id, _digest = seed(root)

    # Never listed: the asset exists only in the derived catalog.
    first = await client.patch(f"/v1/desktop/library/assets/{text_id}", json={"name": "First edit"})
    assert first.status_code == 200 and first.json()["result"]["name"] == "First edit"
    await client.get("/v1/desktop/library/assets")
    second = await client.patch(
        f"/v1/desktop/library/assets/{text_id}", json={"name": "Second edit"}
    )
    assert second.status_code == 200 and second.json()["result"]["name"] == "Second edit"


async def test_a_metadata_edit_does_not_duplicate_or_skip_a_page(api) -> None:
    client, root = api
    rows: list[dict[str, Any]] = []
    names = ("alpha.txt", "bravo.txt", "charlie.txt", "delta.txt")
    for number, name in enumerate(names):
        path = root.parent / name
        path.write_text(name)
        rows += write_rows(f"w{number}", number + 1, path)
    session(root, SESSION, rows)
    indexer.rebuild_now(root)

    first = await client.get(
        "/v1/desktop/library/assets", params={"sort": "name", "order": "asc", "limit": 2}
    )
    page = first.json()["result"]
    assert [row["name"] for row in page["rows"]] == ["alpha.txt", "bravo.txt"]
    cursor = page["next_cursor"]
    assert cursor

    target = page["rows"][0]["id"]
    patched = await client.patch(f"/v1/desktop/library/assets/{target}", json={"name": "zulu.txt"})
    assert patched.status_code == 200

    second = await client.get(
        "/v1/desktop/library/assets",
        params={"sort": "name", "order": "asc", "limit": 10, "cursor": cursor},
    )
    assert second.status_code == 200
    result = second.json()["result"]
    # The edit moved the row across the page boundary. Either the old cursor
    # continues correctly or it resets; it may never silently skip/duplicate.
    if result["cursor_reset"]:
        assert [row["name"] for row in result["rows"]] == [
            "bravo.txt",
            "charlie.txt",
            "delta.txt",
            "zulu.txt",
        ]
    else:
        seen = [row["id"] for row in result["rows"]]
        assert len(seen) == len(set(seen))
        assert target not in seen, "the moved row must not be repeated"


async def test_scope_final_narrows_the_route_listing(api) -> None:
    """The UI's default view: generated artifacts only, API default stays all."""
    client, root = api
    seed(root)

    everything = await client.get("/v1/desktop/library/assets")
    final = await client.get("/v1/desktop/library/assets", params={"scope": "final"})
    assert everything.status_code == 200 and final.status_code == 200
    all_data, final_data = everything.json()["result"], final.json()["result"]
    assert all_data["total"] == 2
    assert final_data["total"] == 1
    assert [row["name"] for row in final_data["rows"]] == ["cat.png"]
    assert final_data["counts"]["origin_group"] == {"generated": 1}

    refused = await client.get("/v1/desktop/library/assets", params={"scope": "nope"})
    assert refused.status_code == 422
    assert refused.json()["detail"]["code"] == "library_bad_filter"


async def test_a_corrupt_index_recovers_and_answers_a_building_page(api, monkeypatch) -> None:
    """ "Derived" must mean "rebuildable at runtime", not only in a docstring."""
    from local_operator.library.store import IndexUnavailable, LibraryStore

    client, root = api
    seed(root)
    original = LibraryStore.query_page
    recovered = {"done": False}

    def broken(self, *args, **kwargs):
        if not recovered["done"]:
            raise IndexUnavailable("file is not a database")
        return original(self, *args, **kwargs)

    def recover(self) -> bool:
        recovered["done"] = True
        return True

    monkeypatch.setattr(LibraryStore, "query_page", broken)
    monkeypatch.setattr(LibraryStore, "recover_derived", recover)
    response = await client.get("/v1/desktop/library/assets")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page["state"] == "building" and page["rows"] == []


async def test_a_corrupt_index_that_cannot_recover_is_a_503(api, monkeypatch) -> None:
    from local_operator.library.store import IndexUnavailable, LibraryStore

    client, root = api
    seed(root)

    def boom(self, *args, **kwargs):
        raise IndexUnavailable("file is not a database")

    monkeypatch.setattr(LibraryStore, "query_page", boom)
    monkeypatch.setattr(LibraryStore, "asset", boom)
    monkeypatch.setattr(LibraryStore, "recover_derived", lambda self: False)

    listed = await client.get("/v1/desktop/library/assets")
    assert listed.status_code == 503, listed.text
    assert listed.json()["detail"]["code"] == "library_metadata_unavailable"

    detail = await client.get("/v1/desktop/library/assets/la_0123456789abcdef")
    assert detail.status_code == 503, detail.text


async def test_an_unknown_tag_id_is_not_a_name_grammar_error(api) -> None:
    """An id the caller copied is a missing resource, not a malformed name."""
    client, root = api
    text_id, _image_id, _digest = seed(root)
    response = await client.put(
        f"/v1/desktop/library/assets/{text_id}/tags", json={"tags": ["deadbeef01"]}
    )
    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "library_tag_unknown"


async def test_a_torn_derived_page_is_never_a_500(api, monkeypatch) -> None:
    """A malformed b-tree inside ``idx_occurrences`` used to escape as a 500.

    The store translates ``sqlite3.DatabaseError`` into ``IndexUnavailable`` at
    its read boundary, so the route's one-shot recovery answers a building page.
    """
    import sqlite3

    from local_operator.library.store import LibraryStore

    client, root = api
    seed(root)

    def torn(self, *args, **kwargs) -> None:
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(LibraryStore, "_occurrences", torn)
    response = await client.get("/v1/desktop/library/assets?scope=all")
    assert response.status_code != 500, response.text
    assert response.status_code == 200, response.text
    assert response.json()["result"]["state"] == "building"


async def test_write_routes_map_an_unreadable_index_to_the_503_envelope(api, monkeypatch) -> None:
    from local_operator.library.store import IndexUnavailable, LibraryStore

    client, root = api
    text_id, _image_id, _digest = seed(root)

    def boom(self, *args, **kwargs):
        raise IndexUnavailable("file is not a database")

    monkeypatch.setattr(LibraryStore, "asset", boom)
    monkeypatch.setattr(LibraryStore, "recover_derived", lambda self: False)

    patched = await client.patch(f"/v1/desktop/library/assets/{text_id}", json={"name": "Renamed"})
    tagged = await client.put(f"/v1/desktop/library/assets/{text_id}/tags", json={"tags": []})
    chatted = await client.put(
        f"/v1/desktop/library/assets/{text_id}/chat", json={"session_id": SESSION}
    )
    for response in (patched, tagged, chatted):
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "library_metadata_unavailable"


async def test_an_empty_library_answers_ready_not_building(api) -> None:
    """A fresh install has no sessions: that is an empty READY library."""
    client, _root = api
    response = await client.get("/v1/desktop/library/assets")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page["state"] == "ready" and page["rows"] == [] and page["total"] == 0
