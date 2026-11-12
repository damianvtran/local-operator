"""C4 of remote files parity: a peer session's Files panel, over the REAL loopback.

docs/design/remote-files-parity.md §2/§3 and the C4 acceptance: I2-1 (a peer page
equals the owner's own display page modulo the remote fields), I2-2 (cursor
passthrough and ``cursor_reset``), I2-3 (every ``remote_read`` row reads, every
``remote_unreadable`` row refuses), the per-row reasons (a cwd file, a missing
file, a credential-shaped pad file), I3-2 (download never overwrites and lands only
verified bytes), the ``~/Downloads``-absent fallback, an owner without
``mesh-file-index-v1`` and a >8 MiB page that is trimmed with the link still up.

A owns SESSION (``mesh`` fixture: two real relays, a real pairing, B a READ member);
B's desktop routes run on an ASGI app over ``DesktopSessions(B.root)``.

RED on base (recorded in the PR): the peer listing answered ``state: "unsupported"``
with no rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import wire
from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.server.routes import desktop_session_files as files_routes
from local_operator.server.routes import desktop_sessions as desktop_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attachments import AttachmentStore
from local_operator.session_files import family as family_mod
from local_operator.session_files import service
from tests.unit.network.test_mesh_file_pull import mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import (  # noqa: F401
    SESSION,
    Devices,
    devices,
    pair,
)
from tests.unit.session_files._fixtures import (
    append_journal,
    assistant,
    call,
    generate_image_rows,
    png,
    tool,
    write_journal,
)

TOKEN = "synthetic-remote-index-token"
REMOTE_FIELDS = ("remote", "origin_device", "remote_read", "remote_unreadable")
URL = f"/v1/desktop/sessions/{SESSION}/files"


def _write_row(n: int, path: str) -> list[dict[str, Any]]:
    return [
        assistant(f"wa{n}", 100.0 + n, [call(f"cw{n}", "write", path=path)]),
        tool(f"wt{n}", 100.5 + n, f"cw{n}", "write", "ok", {"path": path}),
    ]


class Owner:
    """A's session: artifacts in A's store, pad files, a workspace file, a missing one."""

    def __init__(self, server: Any, home: Path) -> None:
        self.root = Path(server.root)
        self.directory = self.root / "sessions" / SESSION
        self.pad = self.directory / "scratchpad"
        self.pad.mkdir(parents=True, exist_ok=True)
        self.store = AttachmentStore(self.root / "attachments")
        self.workspace = home / "work"
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.directory / "desktop.json").write_text(
            json.dumps({"version": 1, "cwd": str(self.workspace)})
        )
        self.images = [png(seed=seed) for seed in (1, 2)]
        self.digests = []
        for image in self.images:
            ref = self.store.put_bytes(image, "image/png")
            assert ref is not None
            self.digests.append(ref.digest)
        (self.pad / "report.md").write_text("# report from the owner\n" * 50)
        (self.workspace / "notes.md").write_text("workspace only")
        rows: list[dict[str, Any]] = []
        for index, digest in enumerate(self.digests, start=1):
            rows += generate_image_rows(index, float(index), digest, name=f"img-{index}.png")
        rows += _write_row(1, str(self.pad / "report.md"))
        rows += _write_row(2, str(self.workspace / "notes.md"))
        rows += _write_row(3, str(self.pad / "gone.txt"))
        rows += _write_row(4, str(self.pad / ".hidden-token.txt"))
        (self.pad / ".hidden-token.txt").write_text("sk-not-for-the-mesh")
        rows += _write_row(5, str(self.pad / ".env"))
        (self.pad / ".env").write_text("API_KEY=x")
        write_journal(self.directory, rows)

    def local_page(self, **query: Any) -> dict[str, Any]:
        """The owner's OWN display page, as its desktop would list it."""
        family, discovery = family_mod.family_blocking(self.root, SESSION, self.directory)
        return service.list_files(
            self.root,
            family,
            service.Query(**{"scope": "family", "limit": 100, **query}),
            display=True,
            budget_s=None,
            expected_parent_revision=discovery.revision if discovery else 0,
        )


@pytest_asyncio.fixture
async def viewer(
    mesh: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, Devices, Owner, Path]]:
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    remote_fetch.STATS["pulls"] = 0
    owner = Owner(mesh[0], home)
    app = FastAPI()
    app.state.config_manager = ConfigManager(Path(mesh[1].root))
    pool = DesktopSessions(Path(mesh[1].root))
    app.state.desktop_sessions = pool
    app.include_router(desktop_routes.router)
    app.include_router(files_routes.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client, mesh, owner, home
    finally:
        await pool.close()


async def _page(client: AsyncClient, **params: Any) -> dict[str, Any]:
    for _attempt in range(40):
        response = await client.get(URL, params=params)
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        if page["state"] != "building":
            return page
    raise AssertionError(f"the peer index never finished building: {page}")


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    """The row minus the remote-only fields; ``ref``'s model defaults (``None``) dropped,
    since the route's response model adds them to BOTH a local and a remote answer."""
    kept = {key: value for key, value in row.items() if key not in REMOTE_FIELDS}
    kept["ref"] = {k: v for k, v in (kept.get("ref") or {}).items() if v is not None}
    return kept


@pytest.mark.asyncio
async def test_i2_1_a_peer_page_is_the_owners_page_plus_the_remote_fields(viewer) -> None:
    client, (server_a, *_rest), owner, _home = viewer
    page = await _page(client)
    local = owner.local_page()
    assert page["remote"] is True and page["unsupported_reason"] is None
    assert page["device_name"] == "device-a"
    assert [_strip(row) for row in page["rows"]] == [_strip(row) for row in local["rows"]]
    for key in ("total", "counts", "revision", "scope", "sort", "order"):
        assert page[key] == local[key], key
    by_name = {row["name"]: row for row in page["rows"]}
    for row in page["rows"]:
        assert row["remote"] is True
        assert row["origin_device"] == {
            "id": server_a.identity.device_id,
            "name": "device-a",
        }
    assert by_name["img-1.png"]["remote_read"] == {"kind": "attachment"}
    assert by_name["report.md"]["remote_read"] == {"kind": "path"}
    # DISPLAY FORM: never the owner's absolute layout (~/… or cwd-relative).
    assert not by_name["report.md"]["ref"]["path"].startswith("/"), by_name["report.md"]["ref"]
    assert by_name["notes.md"]["remote_unreadable"] == "outside_session_files"
    assert by_name["gone.txt"]["remote_unreadable"] == "missing"
    # A credential-shaped pad file is never even LISTED: the owner's index applies its
    # denylist before a row exists (stronger than a ``sensitive`` row). The ``sensitive``
    # reason itself is pinned on the owner's rule below.
    assert ".hidden-token.txt" not in by_name and ".env" not in by_name
    # The owner's absolute layout never crosses: every path is display form.
    assert str(owner.root) not in json.dumps(page)


@pytest.mark.asyncio
async def test_i2_2_the_cursor_passes_through_and_an_append_resets_it(viewer) -> None:
    client, _mesh, owner, _home = viewer
    first = await _page(client, limit=2)
    assert first["next_cursor"] and len(first["rows"]) == 2
    second = await _page(client, limit=2, cursor=first["next_cursor"])
    assert not {r["id"] for r in second["rows"]} & {r["id"] for r in first["rows"]}
    assert second["cursor_reset"] is False
    ref = owner.store.put_bytes(png(seed=9), "image/png")
    assert ref is not None
    append_journal(owner.directory, generate_image_rows(9, 500.0, ref.digest, name="late.png"))
    again = await _page(client, limit=2, cursor=first["next_cursor"])
    assert again["cursor_reset"] is True


@pytest.mark.asyncio
async def test_i2_3_readable_rows_read_and_unreadable_rows_refuse(viewer) -> None:
    client, _mesh, owner, _home = viewer
    page = await _page(client)
    expected = {
        "img-1.png": owner.images[0],
        "img-2.png": owner.images[1],
        "report.md": (owner.pad / "report.md").read_bytes(),
    }
    for row in page["rows"]:
        response = await client.get(f"{URL}/{row['id']}/content")
        if row["remote_read"]:
            assert response.status_code == 200, (row["name"], response.text)
            assert response.content == expected[row["name"]]
            assert response.headers["x-content-type-options"] == "nosniff"
        else:
            assert response.status_code == 409, (row["name"], response.text)
            detail = response.json()["detail"]
            assert detail["code"] == "session_files_remote_unreadable"
            assert detail["params"]["reason"] == row["remote_unreadable"]
    ranged = await client.get(
        f"{URL}/{next(r['id'] for r in page['rows'] if r['name'] == 'report.md')}/content",
        headers={"Range": "bytes=0-9"},
    )
    assert ranged.status_code == 206 and ranged.content == b"# report f"


@pytest.mark.asyncio
async def test_a_local_session_has_no_content_or_download_route(viewer) -> None:
    client, (_a, server_b, *_rest), _owner, _home = viewer
    local = Path(server_b.root) / "sessions" / "aaaaaaaaaaaa"
    write_journal(local, [])
    response = await client.get(f"/v1/desktop/sessions/aaaaaaaaaaaa/files/{'0' * 16}/content")
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_i3_2_download_lands_verified_bytes_and_never_overwrites(viewer) -> None:
    client, (server_a, *_rest), owner, home = viewer
    page = await _page(client)
    row = next(r for r in page["rows"] if r["name"] == "img-1.png")
    url = f"{URL}/{row['id']}/download"

    first = await client.post(url)
    assert first.status_code == 200, first.text
    receipt = first.json()["result"]
    landed = Path(receipt["path"])
    assert landed.parent == home / "Downloads" and landed.read_bytes() == owner.images[0]
    assert receipt["created"] is True and receipt["bytes"] == len(owner.images[0])
    assert receipt["origin_device"]["id"] == server_a.identity.device_id
    assert receipt["sha256"].startswith("sha256:")

    again = (await client.post(url)).json()["result"]
    assert again["path"] == receipt["path"] and again["created"] is False

    landed.write_bytes(b"the person's own edit")
    third = (await client.post(url)).json()["result"]
    assert Path(third["path"]).name == f"{landed.stem} (1){landed.suffix}"
    assert landed.read_bytes() == b"the person's own edit", "an existing file was overwritten"
    assert Path(third["path"]).read_bytes() == owner.images[0]
    assert not [p for p in (home / "Downloads").iterdir() if p.name.endswith(".part")]


@pytest.mark.asyncio
async def test_i3_2_a_tampered_cache_blob_lands_nothing(viewer) -> None:
    client, (_a, server_b, *_rest), owner, home = viewer
    page = await _page(client)
    row = next(r for r in page["rows"] if r["name"] == "img-2.png")
    assert (await client.get(f"{URL}/{row['id']}/content")).status_code == 200
    cached = remote_cache.cache_root(Path(server_b.root)) / f"{owner.digests[1]}.bin"
    cached.write_bytes(b"\x89PNG tampered")
    response = await client.post(f"{URL}/{row['id']}/download")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_files_download_mismatch"
    assert list((home / "Downloads").iterdir()) == []


@pytest.mark.asyncio
async def test_without_a_downloads_folder_the_config_folder_receives_it(viewer) -> None:
    client, (_a, server_b, *_rest), _owner, home = viewer
    (home / "Downloads").rmdir()
    page = await _page(client)
    row = next(r for r in page["rows"] if r["name"] == "report.md")
    response = await client.post(f"{URL}/{row['id']}/download")
    assert response.status_code == 200, response.text
    landed = Path(response.json()["result"]["path"])
    assert landed.parent == Path(server_b.root) / "downloads"
    assert landed.read_text().startswith("# report from the owner")


@pytest.mark.asyncio
async def test_an_owner_without_the_index_capability_is_peer_update_required(viewer) -> None:
    client, (server_a, server_b, *_rest), _owner, _home = viewer
    link = server_b._ensure_link(server_a.identity.device_id)  # noqa: SLF001
    assert link is not None
    link.capabilities = frozenset(c for c in link.capabilities if c != wire.MESH_FILE_INDEX_V1)
    response = await client.get(URL)
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page["state"] == "unsupported" and page["rows"] == []
    assert page["unsupported_reason"] == "peer_update_required"
    assert page["device_name"] == "device-a"


@pytest.mark.asyncio
async def test_a_page_over_one_link_record_is_trimmed_and_the_link_survives(viewer) -> None:
    client, (server_a, server_b, *_rest), owner, _home = viewer
    # Fat rows: an artifact's name is carried verbatim, so 200 names of ~45 KB make a
    # page of ~9 MiB against the link's 8 MiB record (an oversize record kills the link).
    rows: list[dict[str, Any]] = []
    for index in range(200):
        rows += generate_image_rows(
            500 + index, 500.0 + index, f"{index:032x}", name="n" * 45_000 + f"{index}.png"
        )
    append_journal(owner.directory, rows)
    first = await _page(client, limit=200)
    encoded = len(json.dumps(first))
    assert encoded < 8 * 1024 * 1024
    assert first.get("trimmed") is True, f"a {encoded}-byte page was not trimmed"
    assert first["next_cursor"]
    second = await _page(client, limit=200, cursor=first["next_cursor"])
    assert second["cursor_reset"] is False
    assert not {r["id"] for r in second["rows"]} & {r["id"] for r in first["rows"]}
    link = server_b._ensure_link(server_a.identity.device_id)  # noqa: SLF001
    assert link is not None and link.alive


@pytest.mark.asyncio
async def test_user_attachments_of_a_peer_are_listed_ready(viewer) -> None:
    client, _mesh, _owner, _home = viewer
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/user-attachments")
    assert response.status_code == 200, response.text
    listing = response.json()["result"]
    assert listing["state"] == "ready" and listing["remote"] is True


def test_the_owner_rule_names_a_credential_shaped_pad_file_sensitive(tmp_path: Path) -> None:
    """The ``sensitive`` row of §2's table, on the owner's own rule (the listing never
    shows such a row, but ``readability`` is also what ``row`` answers)."""
    from local_operator.network.file_index import readability

    pad = tmp_path / "sessions" / SESSION / "scratchpad"
    pad.mkdir(parents=True)
    (pad / ".env").write_text("API_KEY=x")
    (pad / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    for name in (".env", "id_rsa"):
        read, reason, source = readability(
            tmp_path,
            member_id=SESSION,
            absolute=str(pad / name),
            digest=None,
            size=10,
            exists=True,
            referenced=lambda _member: set(),
        )
        assert (read, reason, source) == (None, "sensitive", None), name
