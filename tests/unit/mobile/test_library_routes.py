"""``/api/library*`` on the wire: auth, projection, filters, tags, bytes.

The catalog is SYNTHETIC: a fixed set of assets is written into the derived
index tables the service reads (``library/store.py``), pointing at files this
test created, so every assertion here is about the relay's HTTP contract rather
than about the scanner (which owns its own suite). The metadata store is the
real one under the isolated config root, because tags and edits are the
authoritative half and a fake there would test nothing.

The security property asserted on every content case is the same one the
service exists to guarantee: an asset id resolves to bytes, and no request
shape turns this route into an arbitrary-file read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.library import identity
from local_operator.library.index import Asset, Occurrence
from local_operator.library.store import LibraryStore
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir

SESSION_A = "4e92693767fa"


def _client() -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _asset(
    path: Path,
    *,
    name: str,
    kind: str,
    category: str,
    content_type: str,
    origin: str = "generated",
) -> Asset:
    key = identity.path_key(str(path))
    return Asset(
        key=key,
        id=identity.asset_id(key),
        path=str(path),
        derived_name=name,
        kind=kind,
        category=category,
        content_type=content_type,
        size_bytes=path.stat().st_size,
        origins=[origin],
        occurrences=[
            Occurrence(
                session_id=SESSION_A,
                title=f"{name} session",
                origin=origin,
                tool="write",
                first_ts=1_700_000_000.0,
                last_ts=1_700_000_100.0,
                first_seq=3,
                last_seq=7,
            )
        ],
    )


@pytest.fixture()
def library(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Four real files plus the catalog that names them, under this HOME.

    The files live under this test HOME, NOT under the config root: the
    denylist refuses the config directory on read, so a fixture there would
    assert the refusal path instead of the serving path.
    """
    from local_operator.library.metadata import MetadataStore

    root = Path.home() / "library-fixture"
    root.mkdir(parents=True, exist_ok=True)
    image = root / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    notes = root / "notes.md"
    notes.write_text("# Launch notes\n\nA short document.\n", encoding="utf-8")
    page = root / "page.html"
    page.write_text("<html><body><script>alert(1)</script>hi</body></html>", encoding="utf-8")
    icon = root / "icon.svg"
    icon.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'>"
        "<script>alert(1)</script><rect width='8' height='8' fill='red'/>"
        "</svg>",
        encoding="utf-8",
    )
    image_asset = _asset(
        image, name="photo.png", kind="image", category="image", content_type="image/png"
    )
    notes_asset = _asset(
        notes,
        name="notes.md",
        kind="markdown",
        category="document",
        content_type="text/markdown",
        origin="written",
    )
    page_asset = _asset(
        page,
        name="page.html",
        kind="html",
        category="document",
        content_type="text/html",
        origin="written",
    )
    svg_asset = _asset(
        icon,
        name="icon.svg",
        kind="image",
        category="image",
        content_type="image/svg+xml",
        origin="written",
    )
    assets = [image_asset, notes_asset, page_asset, svg_asset]
    # The scanner normally writes these rows as a side effect of indexing. The
    # synthetic catalog has to do it explicitly: a metadata write with no row to
    # land on returns ``None`` rather than inventing one, and the service reads
    # ONLY the derived tables now.
    store = MetadataStore(config_dir())
    db = LibraryStore(config_dir())
    db.ensure_schema()
    rows: list[dict[str, Any]] = []
    for asset in assets:
        store.ensure_asset(asset.id, asset.key, derived_name=asset.derived_name)
        for occurrence in asset.occurrences:
            rows.append(
                {
                    "asset_id": asset.id,
                    "key": asset.key,
                    "session_id": occurrence.session_id,
                    "title": occurrence.title,
                    "origin": occurrence.origin,
                    "child_id": "",
                    "label": None,
                    "tool": occurrence.tool,
                    "intent": None,
                    "first_ts": occurrence.first_ts,
                    "last_ts": occurrence.last_ts,
                    "first_seq": occurrence.first_seq,
                    "last_seq": occurrence.last_seq,
                    "touches": 1,
                    "highlighted": False,
                    "name": asset.derived_name,
                    "path": asset.path,
                    "kind": asset.kind,
                    "content_type": asset.content_type,
                    "size_bytes": asset.size_bytes,
                    "width": None,
                    "height": None,
                    "duration_s": None,
                }
            )

    # These are HTTP-contract tests over a FIXED set of rows: the background
    # worker would reconcile the synthetic session against a config that has no
    # journal for it and prune exactly what the fixture just inserted. The
    # worker's own suite covers indexing.
    class _NoWorker:
        @staticmethod
        def enqueue_thumbnail(_asset_id: str, _width: int = 480) -> None:
            return None

    monkeypatch.setattr(
        "local_operator.library.service.ensure_started", lambda *a, **k: _NoWorker()
    )
    with db.write() as connection:
        db.replace_session_occurrences(connection, SESSION_A, rows)
        db.recompute_assets(connection, [asset.id for asset in assets])
        db.record_session(
            connection,
            SESSION_A,
            title="fixture",
            sig={"size": 1, "mtime": 1.0, "inode": 1},
            complete=True,
        )
        db.set_state(connection, backfill_done="1", sessions_total="1", revision="7")
        db.bump_revision(connection)
    return {
        "image": image_asset,
        "markdown": notes_asset,
        "html": page_asset,
        "svg": svg_asset,
    }


def test_library_routes_require_authentication(library: dict[str, Any]) -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.get("/api/library/assets").status_code == 401
    assert client.get(f"/api/library/assets/{library['image'].id}").status_code == 401
    assert client.get(f"/api/library/assets/{library['image'].id}/content").status_code == 401
    assert client.post("/api/library/tags", json={"name": "x"}).status_code == 401
    assert (
        client.patch(f"/api/library/assets/{library['image'].id}", json={"name": "x"}).status_code
        == 401
    )


def test_capability_is_advertised(library: dict[str, Any]) -> None:
    client = _client()
    capabilities = client.get("/api/sessions").json()["capabilities"]
    assert capabilities["library"] == 1


def test_list_projects_display_paths_and_counts(library: dict[str, Any]) -> None:
    client = _client()
    payload = client.get("/api/library/assets").json()
    assert payload["state"] == "ready"
    assert payload["total"] == 4
    assert payload["counts"]["kind"]["image"] == 2
    assert {row["kind"] for row in payload["rows"]} == {"image", "markdown", "html"}
    for row in payload["rows"]:
        # The phone projection NEVER carries the absolute spelling.
        assert row["ref"]["path"] is None
        assert row["ref"]["display_path"].startswith("~") or "/" not in row["ref"]["display_path"]
        assert str(config_dir()) not in str(row)
        assert row["occurrences"][0]["session_id"] == SESSION_A
        assert row["occurrences"][0]["first_seq"] == 3
    assert [tag["name"] for tag in payload["tags"]] == []


def test_list_filters_and_refusals(library: dict[str, Any]) -> None:
    client = _client()
    kinds = client.get("/api/library/assets?categories=document").json()
    assert kinds["total"] == 2
    generated = client.get("/api/library/assets?origin_groups=generated").json()
    assert generated["total"] == 1 and generated["rows"][0]["kind"] == "image"
    searched = client.get("/api/library/assets?q=notes").json()
    assert searched["total"] == 1 and searched["rows"][0]["kind"] == "markdown"

    unknown = client.get("/api/library/assets?kinds=document")
    assert unknown.status_code == 422
    assert unknown.json()["code"] == "library_invalid_filter"
    bad_sort = client.get("/api/library/assets?sort=sideways")
    assert bad_sort.status_code == 422
    bad_cursor = client.get("/api/library/assets?cursor=not-a-cursor")
    assert bad_cursor.status_code == 422
    assert bad_cursor.json()["code"] == "library_invalid_cursor"
    bad_limit = client.get("/api/library/assets?limit=999")
    assert bad_limit.status_code == 422


def test_cursor_paging_round_trip(library: dict[str, Any]) -> None:
    client = _client()
    first = client.get("/api/library/assets?limit=1&sort=name").json()
    assert len(first["rows"]) == 1 and first["next_cursor"]
    second = client.get(
        f"/api/library/assets?limit=1&sort=name&cursor={first['next_cursor']}"
    ).json()
    assert len(second["rows"]) == 1
    assert second["rows"][0]["id"] != first["rows"][0]["id"]


def test_detail_and_unknown_ids(library: dict[str, Any]) -> None:
    client = _client()
    image = library["image"]
    detail = client.get(f"/api/library/assets/{image.id}").json()
    assert detail["id"] == image.id and detail["ref"]["path"] is None
    assert detail["preview"]["state"] == "ready"

    missing = client.get("/api/library/assets/la_" + "0" * 16)
    assert missing.status_code == 404
    assert missing.json()["code"] == "library_asset_not_found"
    invalid = client.get("/api/library/assets/not-an-id")
    assert invalid.status_code == 404


def test_metadata_patch_and_reread(library: dict[str, Any]) -> None:
    client = _client()
    asset_id = library["markdown"].id
    patched = client.patch(
        f"/api/library/assets/{asset_id}",
        json={"name": "Launch notes (edited)", "description": "why", "favorite": True},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Launch notes (edited)"
    assert patched.json()["favorite"] is True
    assert patched.json()["ref"]["path"] is None
    reread = client.get(f"/api/library/assets/{asset_id}").json()
    assert reread["name"] == "Launch notes (edited)"

    refused = client.patch(f"/api/library/assets/{asset_id}", json={"favorite": "yes"})
    assert refused.status_code == 422
    assert client.patch(f"/api/library/assets/{asset_id}", json={"name": 4}).status_code == 422
    assert client.patch(f"/api/library/assets/{asset_id}", content=b"[]").status_code == 400


def test_tag_lifecycle_and_assignment(library: dict[str, Any]) -> None:
    client = _client()
    asset_id = library["image"].id
    created = client.post("/api/library/tags", json={"name": "launch", "category": "project"})
    assert created.status_code == 200
    tag_id = created.json()["tag"]["id"]
    assert client.post("/api/library/tags", json={"name": "LAUNCH"}).json()["tag"]["id"] == tag_id

    assigned = client.put(
        f"/api/library/assets/{asset_id}/tags", json={"tags": [tag_id], "primary": tag_id}
    )
    assert assigned.status_code == 200
    assert assigned.json()["ref"]["path"] is None
    assert assigned.json()["tags"][0]["name"] == "launch"
    detail = client.get(f"/api/library/assets/{asset_id}").json()
    assert detail["primary_tag"] == tag_id

    renamed = client.patch(f"/api/library/tags/{tag_id}", json={"name": "q4-launch"})
    assert renamed.status_code == 200 and renamed.json()["tag"]["name"] == "q4-launch"

    other = client.post("/api/library/tags", json={"name": "misc"}).json()["tag"]["id"]
    merged = client.post(f"/api/library/tags/{other}/merge", json={"into": tag_id})
    assert merged.status_code == 200
    assert [tag["id"] for tag in client.get("/api/library/tags").json()["tags"]] == [tag_id]

    unknown_tag = client.put(f"/api/library/assets/{asset_id}/tags", json={"tags": ["tag_nope"]})
    assert unknown_tag.status_code == 422
    assert unknown_tag.json()["code"] == "tag_not_found"
    bad_body = client.put(f"/api/library/assets/{asset_id}/tags", json={"tags": "launch"})
    assert bad_body.status_code == 422


def test_chat_link_requires_a_real_session(library: dict[str, Any]) -> None:
    client = _client()
    asset_id = library["image"].id
    refused = client.put(
        f"/api/library/assets/{asset_id}/chat", json={"session_id": "not-a-session"}
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "bad_session_id"

    absent = client.put(f"/api/library/assets/{asset_id}/chat", json={"session_id": "aaaaaaaaaaaa"})
    assert absent.status_code == 422
    assert absent.json()["code"] == "session_not_found"

    (config_dir() / "sessions" / SESSION_A).mkdir(parents=True, exist_ok=True)
    linked = client.put(f"/api/library/assets/{asset_id}/chat", json={"session_id": SESSION_A})
    assert linked.status_code == 200
    assert linked.json()["chat"]["session_id"] == SESSION_A
    assert client.get(f"/api/library/assets/{asset_id}").json()["chat"]["session_id"] == SESSION_A

    cleared = client.request("DELETE", f"/api/library/assets/{asset_id}/chat")
    assert cleared.status_code == 200 and cleared.json()["cleared"] is True
    assert client.get(f"/api/library/assets/{asset_id}").json()["chat"] is None


def test_bytes_inline_range_and_download(library: dict[str, Any]) -> None:
    client = _client()
    asset_id = library["image"].id
    response = client.get(f"/api/library/assets/{asset_id}/content")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content.startswith(b"\x89PNG")
    assert response.headers["accept-ranges"] == "bytes"

    partial = client.get(f"/api/library/assets/{asset_id}/content", headers={"Range": "bytes=0-3"})
    assert partial.status_code == 206
    assert partial.headers["content-range"].startswith("bytes 0-3/")
    assert partial.content == b"\x89PNG"

    download = client.get(f"/api/library/assets/{asset_id}/content?download=1")
    assert download.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment;")
    assert download.headers["cache-control"] == "no-store"

    missing = client.get("/api/library/assets/la_" + "0" * 16 + "/content")
    assert missing.status_code == 404
    assert missing.json()["code"] == "library_asset_not_found"


def test_html_bytes_are_sandboxed(library: dict[str, Any]) -> None:
    client = _client()
    response = client.get(f"/api/library/assets/{library['html'].id}/content")
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["content-disposition"].startswith("inline;")


def test_thumb_and_non_image_refusal(library: dict[str, Any]) -> None:
    client = _client()
    thumb = client.get(f"/api/library/assets/{library['image'].id}/thumbnail")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"].startswith("image/")
    assert thumb.headers["x-content-type-options"] == "nosniff"

    refused = client.get(f"/api/library/assets/{library['markdown'].id}/thumbnail")
    assert refused.status_code == 404
    assert refused.json()["code"] == "not_an_image"
    assert (
        client.get(f"/api/library/assets/{library['image'].id}/thumbnail?w=nope").status_code == 400
    )


def test_a_missing_file_reports_content_missing(
    library: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client()
    os.unlink(library["markdown"].path)
    response = client.get(f"/api/library/assets/{library['markdown'].id}/content")
    assert response.status_code == 404
    assert response.json()["code"] == "content_missing"


def test_svg_is_sandboxed_on_content_and_thumbnail(library: dict[str, Any]) -> None:
    client = _client()
    asset_id = library["svg"].id

    content = client.get(f"/api/library/assets/{asset_id}/content")
    assert content.status_code == 200
    policy = content.headers["content-security-policy"]
    assert policy.startswith("sandbox") and "default-src 'none'" in policy
    assert b"<script>" in content.content

    thumb = client.get(f"/api/library/assets/{asset_id}/thumbnail")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"].startswith("image/svg+xml")
    assert "default-src 'none'" in thumb.headers["content-security-policy"]

    png = client.get(f"/api/library/assets/{library['image'].id}/content")
    assert png.status_code == 200
    assert "content-security-policy" not in png.headers


def test_facets_without_rows(library: dict[str, Any]) -> None:
    client = _client()
    facets = client.get("/api/library/facets").json()
    assert facets["total"] == 4
    assert "rows" not in facets
    assert facets["counts"]["category"]["document"] == 2
