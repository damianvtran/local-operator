"""C6 of remote files parity: peer sessions' files in the Library index.

docs/design/remote-files-parity.md §5 and the C6 acceptance: I5-1 (a remote asset
never resolves to a path on this disk), I5-2 (list/detail/thumbnail never dial the
mesh), a remote image thumbnail generated once its bytes are cached that SURVIVES
cache eviction, dedup of a digest held locally and remotely into ONE asset with two
occurrences, and removal when the session leaves the catalogue (moved/deleted).

A owns SESSION (``Owner``: two images in A's store, a pad report, a workspace file);
B indexes it through the real relays with ``library.remote.feed`` and serves the
Library routes over B's root.

RED on base (recorded in the PR): ``library.remote`` does not exist and the Library
never lists a peer session's file.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.library import identity, indexer
from local_operator.library import remote as remote_feed
from local_operator.library import service as library_service
from local_operator.remote_files import cache as remote_cache
from local_operator.server.routes import desktop_library
from local_operator.server.utils import desktop_mesh
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mesh_file_pull import mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import SESSION, devices, pair  # noqa: F401
from tests.unit.network.test_remote_files_index import TOKEN, viewer  # noqa: F401
from tests.unit.session_files._fixtures import generate_image_rows, write_journal

LIB = "/v1/desktop/library/assets"


def _catalogue_row(server_a: Any, *, mtime: float = 1_800_000_000.0) -> dict[str, Any]:
    return {
        "id": SESSION,
        "name": "mesh design",
        "mtime": mtime,
        "locality": "remote",
        "reachable": True,
        "archived": False,
        "owner_device": server_a.identity.device_id,
        "owner_device_name": "device-a",
    }


@pytest_asyncio.fixture
async def library(viewer) -> AsyncIterator[tuple[AsyncClient, Any, Any, Any]]:  # noqa: F811
    """B's Library routes plus a fed index; the indexer thread never runs."""
    _client, (server_a, server_b, *_rest), owner, _home = viewer
    indexer.reset_for_tests()
    root = Path(server_b.root)
    worker = indexer.LibraryIndexer(root)
    worker.store.ensure_schema()
    assert remote_feed.feed(worker, catalogue=[_catalogue_row(server_a)]) == 1
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_library.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client, worker, (server_a, server_b), owner
    finally:
        indexer.reset_for_tests()


def _quiet_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A list read lazily starts the indexer; keep its reconcile and feed off the mesh."""
    monkeypatch.setattr(indexer.LibraryIndexer, "start", lambda self: None)


async def _rows(client: AsyncClient) -> dict[str, dict[str, Any]]:
    response = await client.get(LIB, params={"limit": 200})
    assert response.status_code == 200, response.text
    return {row["name"]: row for row in response.json()["result"]["rows"]}


@pytest.mark.asyncio
async def test_peer_files_are_listed_with_remote_and_no_path(library, monkeypatch) -> None:
    client, _worker, (server_a, _b), owner = library
    _quiet_worker(monkeypatch)
    rows = await _rows(client)
    assert {"img-1.png", "img-2.png", "report.md"} <= set(rows), sorted(rows)
    for name in ("img-1.png", "report.md"):
        row = rows[name]
        assert row["remote"] == {
            "device_id": server_a.identity.device_id,
            "device_name": "device-a",
        }, row
        assert row["ref"]["path"] is None, row["ref"]
        assert row["occurrences"][0]["session_id"] == SESSION
    assert rows["img-1.png"]["ref"]["digest"] == owner.digests[0]


@pytest.mark.asyncio
async def test_i5_1_a_remote_path_asset_never_resolves_on_this_disk(library, monkeypatch) -> None:
    """The owner's pad file EXISTS at its absolute path on this test host (both relays
    share one filesystem), which is exactly the trap: the asset must still come
    through the owner, and the viewer must never stat that spelling."""
    client, _worker, (_a, server_b), owner = library
    _quiet_worker(monkeypatch)
    absolute = str(owner.pad / "report.md")
    assert Path(absolute).is_file()
    rows = await _rows(client)
    asset_id = rows["report.md"]["id"]
    asset = library_service._catalog_asset(Path(server_b.root), asset_id)  # noqa: SLF001
    assert asset.path is None and asset.key.startswith("r:"), asset.key
    assert asset.key != identity.path_key(absolute)

    # Both relays run in THIS process, so the OWNER's own stat of its file (its relay
    # projecting the row, on a relay thread) is legitimate; only a stat from the
    # viewer's request path (the event loop or its ``to_thread`` workers) is the bug.
    touched: list[str] = []
    real_stat = library_service.files_service.stat_path

    def spy(path: Any, *a: Any, **k: Any) -> Any:
        name = threading.current_thread().name
        if name == "MainThread" or name.startswith("asyncio"):
            touched.append(str(path))
        return real_stat(path, *a, **k)

    monkeypatch.setattr(library_service.files_service, "stat_path", spy)
    with pytest.raises(library_service.ContentUnavailable) as refused:
        library_service.resolve_content(Path(server_b.root), asset_id)
    assert refused.value.code == "content_remote"
    detail = await client.get(f"{LIB}/{asset_id}")
    assert detail.status_code == 200 and detail.json()["result"]["ref"]["path"] is None
    content = await client.get(f"{LIB}/{asset_id}/content")
    assert content.status_code == 200, content.text
    assert content.content == Path(absolute).read_bytes()
    assert touched == [], f"the viewer stat'd the owner's path on its own disk: {touched}"
    # THROUGH THE OWNER: the bytes landed in B's remote cache by their content address.
    import hashlib

    digest = hashlib.sha256(content.content).hexdigest()[:32]
    assert (remote_cache.cache_root(Path(server_b.root)) / f"{digest}.bin").is_file()


@pytest.mark.asyncio
async def test_i5_2_list_detail_and_thumbnail_never_dial_the_mesh(library, monkeypatch) -> None:
    client, _worker, _servers, _owner = library
    _quiet_worker(monkeypatch)
    dialled: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        dialled.append((args, kwargs))
        raise AssertionError("a Library read dialled the mesh")

    monkeypatch.setattr(desktop_mesh, "_call", refuse)
    response = await client.get(LIB, params={"limit": 200})
    assert response.status_code == 200, response.text
    rows = response.json()["result"]["rows"]
    assert rows
    for row in rows:
        assert (await client.get(f"{LIB}/{row['id']}")).status_code == 200
        thumb = await client.get(f"{LIB}/{row['id']}/thumbnail")
        # No bytes here yet: the designed 404 (the card shows its glyph).
        assert thumb.status_code == 404, (row["name"], thumb.text)
    assert dialled == []


@pytest.mark.asyncio
async def test_a_remote_thumbnail_is_made_after_precache_and_survives_eviction(
    library, monkeypatch
) -> None:
    client, worker, (_a, server_b), owner = library
    _quiet_worker(monkeypatch)
    rows = await _rows(client)
    asset_id = rows["img-1.png"]["id"]
    assert (
        worker.store.ensure_thumbnail(asset_id, width=240, quality=80) is None
    ), "a thumbnail was made with no bytes on this device"
    opened = await client.get(f"{LIB}/{asset_id}/content")
    assert opened.status_code == 200 and opened.content == owner.images[0]
    made = worker.store.ensure_thumbnail(asset_id, width=240, quality=80)
    assert made is not None and Path(made).is_file()
    cached = remote_cache.cache_root(Path(server_b.root)) / f"{owner.digests[0]}.bin"
    cached.unlink()
    assert worker.store.thumb_is_live(asset_id, 240) == Path(
        made
    ), "evicting the cached source invalidated a thumbnail of the same digest"
    thumb = await client.get(f"{LIB}/{asset_id}/thumbnail", params={"w": 240})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"


@pytest.mark.asyncio
async def test_a_digest_held_locally_and_remotely_is_one_asset_with_two_occurrences(
    library, monkeypatch
) -> None:
    client, worker, (_a, server_b), owner = library
    _quiet_worker(monkeypatch)
    root = Path(server_b.root)
    local = "0a1b2c3d4e5f"
    ref = AttachmentStore(root / "attachments").put_bytes(owner.images[0], "image/png")
    assert ref is not None and ref.digest == owner.digests[0]
    write_journal(
        root / "sessions" / local, generate_image_rows(9, 9.0, ref.digest, name="mine.png")
    )
    assert indexer._acquire_writer(root)  # noqa: SLF001 — the scan is the writer's
    assert worker.scan_session(local)
    response = await client.get(LIB, params={"limit": 200})
    matching = [
        row for row in response.json()["result"]["rows"] if row["ref"]["digest"] == ref.digest
    ]
    assert len(matching) == 1, matching
    row = matching[0]
    assert {o["session_id"] for o in row["occurrences"]} == {SESSION, local}
    assert row["remote"] is None, "an asset this device also holds is a LOCAL asset"
    opened = await client.get(f"{LIB}/{row['id']}/content")
    assert opened.status_code == 200 and opened.content == owner.images[0]
    assert not (remote_cache.cache_root(root) / f"{ref.digest}.bin").exists()


@pytest.mark.asyncio
async def test_a_session_gone_from_two_catalogue_reads_is_removed(library, monkeypatch) -> None:
    client, worker, (server_a, _b), _owner = library
    _quiet_worker(monkeypatch)
    assert await _rows(client)
    # An UNREACHABLE peer is not an absence: its device did not answer at all.
    remote_feed.feed(worker, catalogue=[])
    remote_feed.feed(worker, catalogue=[])
    assert await _rows(client), "a silent device's rows were dropped"
    # Its device answers, without the session (moved or deleted): two reads drop it.
    other = {**_catalogue_row(server_a), "id": "fedcba987654"}
    monkeypatch.setattr(remote_feed, "_ask", lambda *a, **k: {"state": "ready", "rows": []})
    remote_feed.feed(worker, catalogue=[other])
    assert await _rows(client), "one missed read is not a removal"
    remote_feed.feed(worker, catalogue=[other])
    assert {name for name in await _rows(client)} == set()
    assert worker.store.session(SESSION) is None
