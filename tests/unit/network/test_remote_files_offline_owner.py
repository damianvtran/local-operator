"""Round-3 remediation of PR #3872 (remote files parity): bytes already here, owner away.

R3-2/Q3-1 and Q3-2: a peer's file that is on THIS device (remote-files cache,
store or replica) must open with its owner offline, without reading the owner's
row or dialling the relay; a miss with the owner silent is the retryable 504
``attachment_owner_unreachable``, never a 404. R3-1/R3-3: a cached peer file's
owner-declared type goes through the remote routes' allowlist and sandbox on the
phone and on the desktop thumbnail.

The owner being offline is modelled at ``remote_open.remote_row_and_silence``,
the one seam every row read goes through: an offline owner's sessions drop out of
the listing and the relay reports the device as unanswered (``peer_rows``'s
``UnansweredPeer``). The cache, the lookup, the routes and the responses are real.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.session.peer_rows import UnansweredPeer

SID = "0123456789ab"
CHILD = "ba9876543210"
OWNER = "d_owner_device"
ASSET = "la_" + "0f" * 8
TOKEN = "t"


def _cache(root: Path, payload: bytes, mime: str) -> str:
    """``payload`` in this device's remote-files cache, as the precache leaves it."""
    directory = remote_cache.cache_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()[:32]
    (directory / f"{digest}.bin").write_bytes(payload)
    (directory / f"{digest}.json").write_text(json.dumps({"mime_type": mime}))
    return digest


@pytest.fixture
def owner_offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """The owner is asleep: no row, the relay names it silent. Counts every dial."""
    from local_operator.server.utils import desktop_mesh
    from local_operator.session import remote_open

    calls: dict[str, list[Any]] = {"rows": [], "relay": [], "pulls": []}

    def silent(session_id: str, _root: Path) -> tuple[None, tuple[UnansweredPeer, ...]]:
        calls["rows"].append(session_id)
        return None, (UnansweredPeer(OWNER, "owner box", "not answering"),)

    def relay(*args: Any, **kwargs: Any) -> Any:
        calls["relay"].append(args[1:2])
        raise AssertionError("a relay call for bytes already on this device")

    async def pull(*_args: Any, **_kwargs: Any) -> Any:
        calls["pulls"].append(1)
        raise remote_fetch.RemoteFetchError("unreachable", "peer_unreachable", "asleep")

    monkeypatch.setattr(remote_open, "remote_row_and_silence", silent)
    monkeypatch.setattr(desktop_mesh, "_call", relay)
    monkeypatch.setattr(remote_fetch, "fetch", pull)
    return calls


# -- R3-2 / Q3-1: the Library open ------------------------------------------------


@pytest_asyncio.fixture
async def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_operator.config import ConfigManager
    from local_operator.library import service as library_service
    from local_operator.server.routes import desktop_library

    assets: dict[str, Any] = {}
    monkeypatch.setattr(library_service, "_catalog_asset", lambda _config, _id: assets["asset"])
    monkeypatch.setattr(
        library_service,
        "_remote_occurrence",
        lambda _config, _id: {"session_id": SID, "origin_device": OWNER},
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_library.router)

    def asset(digest: str, content_type: str, name: str = "evil.svg") -> None:
        assets["asset"] = SimpleNamespace(
            id=ASSET, digest=digest, derived_name=name, content_type=content_type, path=None
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, asset, tmp_path


@pytest.mark.asyncio
async def test_r3_2_a_cached_library_asset_opens_with_its_owner_offline(
    library, owner_offline: dict[str, list[Any]]
) -> None:
    client, asset, root = library
    payload = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    asset(_cache(root, payload, "image/svg+xml"), "image/svg+xml")
    response = await client.get(f"/v1/desktop/library/assets/{ASSET}/content")
    assert response.status_code == 200, response.text
    assert response.content == payload
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert owner_offline == {"rows": [], "relay": [], "pulls": []}, "dialled for local bytes"


@pytest.mark.asyncio
async def test_r3_2_an_uncached_asset_with_its_owner_offline_is_retryable_not_missing(
    library, owner_offline: dict[str, list[Any]]
) -> None:
    client, asset, _root = library
    asset("ab" * 16, "image/png", "pic.png")
    response = await client.get(f"/v1/desktop/library/assets/{ASSET}/content")
    assert response.status_code == 504, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_owner_unreachable"
    assert detail["params"] == {"device": "owner box"}
    assert owner_offline["pulls"] == [], "no pull without the owner's row"


@pytest.mark.asyncio
async def test_r3_2_an_owner_that_answered_without_the_session_is_still_404(
    library, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: silence from ANOTHER device says nothing about this owner."""
    from local_operator.session import remote_open

    monkeypatch.setattr(
        remote_open,
        "remote_row_and_silence",
        lambda _sid, _root: (None, (UnansweredPeer("d_other", "other box", "asleep"),)),
    )
    client, asset, _root = library
    asset("ab" * 16, "image/png", "pic.png")
    response = await client.get(f"/v1/desktop/library/assets/{ASSET}/content")
    assert response.status_code == 404, response.text


# -- Q3-2: the transcript route and its siblings ------------------------------------


def _index_peer_session(root: Path, session_id: str = SID) -> None:
    """What the Library's remote feed recorded while the owner was answering."""
    from local_operator.library.store import LibraryStore

    store = LibraryStore(root)
    with store.write() as connection:
        store.record_session(
            connection,
            session_id,
            title="on the owner",
            sig={"mtime": 1.0},
            complete=True,
            origin_device=OWNER,
        )


@pytest_asyncio.fixture
async def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions as desktop_routes
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    for name in [name for name in __import__("os").environ if name.startswith("CMUX_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    _index_peer_session(tmp_path)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_routes.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client, tmp_path
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_q3_2_a_cached_transcript_attachment_opens_with_its_owner_offline(
    transcript, owner_offline: dict[str, list[Any]]
) -> None:
    client, root = transcript
    payload = b"\x89PNG\r\n\x1a\n" + b"cached-image" * 64
    digest = _cache(root, payload, "image/png")
    for url in (
        f"/v1/desktop/sessions/{SID}/attachments/{digest}",
        f"/v1/desktop/sessions/{SID}/children/{CHILD}/attachments/{digest}",
    ):
        response = await client.get(url)
        assert response.status_code == 200, (url, response.text)
        assert response.content == payload
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-content-type-options"] == "nosniff"
    assert owner_offline == {"rows": [], "relay": [], "pulls": []}, "dialled for local bytes"


@pytest.mark.asyncio
async def test_q3_2_a_cached_svg_on_the_transcript_route_is_opaque(
    transcript, owner_offline: dict[str, list[Any]]
) -> None:
    client, root = transcript
    digest = _cache(root, b"<svg><script>alert(1)</script></svg>", "image/svg+xml")
    response = await client.get(f"/v1/desktop/sessions/{SID}/attachments/{digest}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_q3_2_an_uncached_attachment_with_its_owner_offline_is_504_not_404(
    transcript, owner_offline: dict[str, list[Any]]
) -> None:
    client, _root = transcript
    for url in (
        f"/v1/desktop/sessions/{SID}/attachments/{'cd' * 16}",
        f"/v1/desktop/sessions/{SID}/attachments/{'cd' * 16}/waveform",
        f"/v1/desktop/sessions/{SID}/children/{CHILD}/attachments/{'cd' * 16}",
    ):
        response = await client.get(url)
        assert response.status_code == 504, (url, response.text)
        detail = response.json()["detail"]
        assert detail["code"] == "attachment_owner_unreachable", url
        assert detail["params"] == {"device": "owner box"}, url
    assert owner_offline["pulls"] == [], "no pull without the owner's row"


@pytest.mark.asyncio
async def test_q3_2_a_session_nobody_holds_keeps_its_refusals(
    transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: every device answered and none holds it — unchanged answers."""
    from local_operator.session import remote_open

    monkeypatch.setattr(remote_open, "remote_row_and_silence", lambda _sid, _root: (None, ()))
    client, _root = transcript
    parent = await client.get(f"/v1/desktop/sessions/{SID}/attachments/{'cd' * 16}")
    assert parent.status_code == 404, parent.text
    child = await client.get(f"/v1/desktop/sessions/{SID}/children/{CHILD}/attachments/{'cd' * 16}")
    assert child.json()["detail"]["code"] == "child_not_found", child.text


@pytest.mark.asyncio
async def test_q3_2_an_id_this_device_never_recorded_cannot_read_the_cache(
    transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session id stays an existence check: guessing one reads nothing."""
    from local_operator.session import remote_open

    monkeypatch.setattr(remote_open, "remote_row_and_silence", lambda _sid, _root: (None, ()))
    client, root = transcript
    digest = _cache(root, b"\x89PNG\r\n\x1a\n" + b"someone-elses" * 16, "image/png")
    for url in (
        f"/v1/desktop/sessions/ffffffffffff/attachments/{digest}",
        f"/v1/desktop/sessions/ffffffffffff/children/{CHILD}/attachments/{digest}",
        f"/v1/desktop/sessions/ffffffffffff/attachments/{digest}/waveform",
    ):
        response = await client.get(url)
        assert response.status_code in (404, 409), (url, response.status_code)
        assert b"someone-elses" not in response.content, url


# -- R3-1: the phone's Library routes -----------------------------------------------


@pytest.mark.parametrize("declared", ["text/xml", "application/xml", "text/html"])
def test_r3_1_a_cached_peer_file_on_the_phone_is_opaque_and_sandboxed(
    declared: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from starlette.testclient import TestClient

    from local_operator.library import service as library_service
    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.paths import config_dir

    payload = (
        b'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
        b"<script>fetch('/api/sessions')</script></html>"
    )
    digest = _cache(config_dir(), payload, declared)
    asset = SimpleNamespace(
        id=ASSET, digest=digest, derived_name="report.pdf", content_type=declared, path=None
    )
    monkeypatch.setattr(library_service, "_catalog_asset", lambda _config, _id: asset)
    # The thumbnail's original-bytes fallback, without a real index row.
    monkeypatch.setattr(
        library_service,
        "thumbnail",
        lambda config, asset_id, _w: library_service.resolve_content(config, asset_id),
    )
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    for route in ("content", "thumbnail"):
        response = client.get(f"/api/library/assets/{ASSET}/{route}")
        assert response.status_code == 200, (route, response.text)
        assert response.content == payload
        assert response.headers["content-type"] == "application/octet-stream", route
        assert response.headers["content-security-policy"] == "sandbox; default-src 'none'"
        assert response.headers["x-content-type-options"] == "nosniff"


# -- R3-3: the desktop thumbnail -------------------------------------------------------


@pytest.mark.asyncio
async def test_r3_3_a_cached_peer_thumbnail_is_opaque_and_sandboxed(
    library, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.library import service as library_service

    client, asset, root = library
    payload = b"<?xml version='1.0'?><x/>"
    asset(_cache(root, payload, "application/xml"), "application/xml", "pic.png")
    monkeypatch.setattr(
        library_service,
        "thumbnail",
        lambda config, asset_id, _w: library_service.resolve_content(config, asset_id),
    )
    response = await client.get(f"/v1/desktop/library/assets/{ASSET}/thumbnail")
    assert response.status_code == 200, response.text
    assert response.content == payload
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-security-policy"] == "sandbox"
