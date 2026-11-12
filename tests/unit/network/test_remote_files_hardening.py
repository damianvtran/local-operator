"""Round-1 remediation of PR #3872 (remote files parity): regression tests.

Each test names its ledger id. They run without the mesh: the owner's replies are
stubbed at ``remote_session_files.ask`` / ``_resolve`` / ``fetch._pull``, because
what is under test is what THIS device does with a reply, not the transport (the
loopback suites in ``test_remote_files_{fetch,index,precache}.py`` cover that).
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.remote_files import precache
from local_operator.server.utils import desktop_sessions as desktop_utils
from local_operator.server.utils import remote_session_files as rsf

SID = "0123456789ab"
OWNER = "d_owner_device"


def _row() -> Any:
    return SimpleNamespace(
        owner_device=OWNER,
        owner_device_name="owner box",
        owner_label="owner box",
        mtime=time.time(),
        archived=False,
    )


def _blob(directory: Path, payload: bytes, mime: str = "image/png") -> remote_cache.CachedBlob:
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()[:32]
    path = directory / f"{digest}.bin"
    path.write_bytes(payload)
    return remote_cache.CachedBlob(path=path, size=len(payload), mime=mime, source="cache")


def _stub_row_reply(monkeypatch: pytest.MonkeyPatch, source: dict[str, Any]) -> None:
    async def ask(_config: Path, _sid: str, **_fields: Any) -> tuple[dict[str, Any], Any]:
        return {"row": {"id": "f1", "name": "pic.png"}, "source": source}, _row()

    monkeypatch.setattr(rsf, "ask", ask)


def _never_open(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    opened: list[Any] = []

    def spy(blob: Any) -> Any:
        opened.append(blob)
        raise AssertionError(f"opened {blob.path}")

    monkeypatch.setattr(rsf, "_open", spy)
    return opened


# -- R1-1 -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r1_1_an_owner_digest_that_walks_out_of_the_store_is_409_and_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "config"
    (root / "attachments").mkdir(parents=True)
    (root / "secret.bin").write_bytes(b"LOCAL-ONLY-BYTES")
    _stub_row_reply(monkeypatch, {"attachment": "../secret", "session_id": SID})
    opened = _never_open(monkeypatch)
    with pytest.raises(HTTPException) as caught:
        await rsf.content(root, SID, "f1", None)
    assert caught.value.status_code == 409
    detail: Any = caught.value.detail
    assert detail["code"] == "session_files_remote_unreadable"
    assert opened == []


@pytest.mark.asyncio
async def test_r1_1_a_malformed_member_id_is_409_not_an_uncaught_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_row_reply(monkeypatch, {"attachment": "a" * 32, "session_id": "../../x"})
    opened = _never_open(monkeypatch)
    with pytest.raises(HTTPException) as caught:
        await rsf.content(tmp_path, SID, "f1", None)
    assert caught.value.status_code == 409
    detail: Any = caught.value.detail
    assert detail["code"] == "session_files_remote_unreadable"
    assert opened == []


def test_r1_1_the_cache_never_builds_a_path_from_a_malformed_name(tmp_path: Path) -> None:
    (tmp_path / "attachments").mkdir()
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "notes.bin").write_bytes(b"LOCAL-ONLY-BYTES")
    assert remote_cache.lookup(tmp_path, (SID,), "../home/notes") is None
    assert remote_cache.lookup(tmp_path, ("../../x",), "a" * 32) is None
    with pytest.raises(ValueError):
        remote_cache.locations(tmp_path, SID, "../home/notes")
    with pytest.raises(ValueError):
        remote_cache.locations(tmp_path, "../../x", "a" * 32)


# -- R1-2 -------------------------------------------------------------------------


def _catalogue_row(**overrides: Any) -> dict[str, Any]:
    return {
        "id": SID,
        "mtime": 1_800_000_000.0,
        "locality": "remote",
        "reachable": True,
        "archived": False,
        "owner_device": OWNER,
        **overrides,
    }


@pytest.mark.asyncio
async def test_r1_2_twelve_opens_of_an_unchanged_session_make_one_pass(tmp_path: Path) -> None:
    worker = precache.Precacher(tmp_path)
    passes: list[str] = []

    async def prepare(row: dict[str, Any]) -> int:
        # The real ``prepare``'s first effect: the gate's record of this pass.
        worker.passes[str(row["id"])] = (time.monotonic(), float(row.get("mtime") or 0.0))
        passes.append(str(row["id"]))
        return 0

    worker.prepare = prepare  # type: ignore[method-assign]
    for _ in range(12):
        worker.opened_session(SID, _catalogue_row())
        await asyncio.sleep(0.01)
    assert passes == [SID]
    # Past the rescan window a MOVED session passes again; an unchanged one does not.
    worker.passes[SID] = (time.monotonic() - precache.RESCAN_S - 1, 1_800_000_000.0)
    worker.opened_session(SID, _catalogue_row())
    await asyncio.sleep(0.01)
    assert passes == [SID]
    worker.opened_session(SID, _catalogue_row(mtime=1_800_000_005.0))
    await asyncio.sleep(0.01)
    assert passes == [SID, SID]


def _one_path_candidate(worker: precache.Precacher, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    async def candidates(_row: dict[str, Any]) -> list[tuple[str, dict[str, str], str, int]]:
        return [(SID, {"file_id": "f1"}, "", 10)]

    asked: list[str] = []

    async def ask(_root: Path, _sid: str, **fields: Any) -> tuple[dict[str, Any], Any]:
        asked.append(str(fields.get("phase")))
        return {"row": {}, "source": {"path": "scratchpad://pic.png", "session_id": SID}}, _row()

    async def fetch(*_a: Any, **_k: Any) -> Any:
        raise remote_fetch.RemoteFetchError("refused", "not_referenced", "stub")

    worker.candidates = candidates  # type: ignore[method-assign]
    monkeypatch.setattr(rsf, "ask", ask)
    monkeypatch.setattr(remote_fetch, "fetch", fetch)
    return asked


@pytest.mark.asyncio
async def test_r1_2_a_path_row_lookup_waits_for_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yml").write_text("desktop:\n  remote_files_cache_bytes: 1\n")
    worker = precache.Precacher(tmp_path)
    asked = _one_path_candidate(worker, monkeypatch)
    await worker.prepare(_catalogue_row())
    assert asked == [], "a file the budget cannot take is not asked about"


@pytest.mark.asyncio
async def test_r1_2_a_path_row_lookup_waits_for_the_precache_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = precache.Precacher(tmp_path)
    asked = _one_path_candidate(worker, monkeypatch)
    slot = worker._semaphore(OWNER)  # noqa: SLF001
    await slot.acquire()
    task = asyncio.ensure_future(worker.prepare(_catalogue_row()))
    await asyncio.sleep(0.05)
    assert asked == [], "the row ask ran outside the precache slot"
    slot.release()
    await task
    assert asked == ["row"]


# -- R1-3 -------------------------------------------------------------------------


def _resolve_twice(monkeypatch: pytest.MonkeyPatch, first: Any, second: Any) -> list[int]:
    calls: list[int] = []

    async def resolve(_config: Path, _sid: str, _fid: str) -> tuple[Any, Any, Any]:
        calls.append(1)
        blob = first if len(calls) == 1 else second
        return blob, {"id": "f1", "name": "pic.png"}, _row()

    monkeypatch.setattr(rsf, "_resolve", resolve)
    return calls


def _evicted(tmp_path: Path) -> remote_cache.CachedBlob:
    return remote_cache.CachedBlob(
        path=tmp_path / "gone" / f"{'c' * 32}.bin", size=3, mime="image/png", source="cache"
    )


@pytest.mark.asyncio
async def test_r1_3_content_re_resolves_once_after_an_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = _blob(tmp_path / "cache", b"PNG-BYTES")
    calls = _resolve_twice(monkeypatch, _evicted(tmp_path), blob)
    response = await rsf.content(tmp_path, SID, "f1", None)
    assert response.status_code == 200
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_r1_3_a_second_eviction_is_504_retryable_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_twice(monkeypatch, _evicted(tmp_path), _evicted(tmp_path))
    with pytest.raises(desktop_utils.PeerAttachmentRetryable):
        await rsf.content(tmp_path, SID, "f1", None)


@pytest.mark.asyncio
async def test_r1_3_download_re_resolves_once_after_an_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"PNG-BYTES-FOR-DOWNLOAD"
    blob = _blob(tmp_path / "cache", payload)
    calls = _resolve_twice(monkeypatch, _evicted(tmp_path), blob)
    landing = tmp_path / "Downloads"
    landing.mkdir()
    monkeypatch.setattr(rsf, "downloads_dir", lambda _config: landing)
    receipt = await rsf.download(tmp_path, SID, "f1")
    assert len(calls) == 2
    assert Path(receipt["path"]).read_bytes() == payload
    assert receipt["created"] is True


@pytest.mark.asyncio
async def test_r1_3_library_content_re_resolves_once_after_an_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.library import service as library_service

    blob = _blob(tmp_path / "cache", b"<script>alert(1)</script>", mime="text/html")
    calls = _resolve_twice(monkeypatch, _evicted(tmp_path), blob)
    monkeypatch.setattr(
        library_service,
        "remote_occurrence",
        lambda _config, _asset: {"session_id": SID, "remote_file_id": "f1"},
    )
    monkeypatch.setattr(
        library_service,
        "_catalog_asset",
        lambda _config, _asset: SimpleNamespace(
            digest=None, derived_name="page.html", id="r:x", content_type="text/html"
        ),
    )
    response = await rsf.library_content(tmp_path, "r:x", range_header=None, download=False)
    assert len(calls) == 2
    # R1-5: a type from another device's listing is allowlisted and sandboxed.
    assert response.media_type == "application/octet-stream"
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_r1_3_one_landing_never_evicts_another_before_its_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Budget 0, two pulls: B lands and evicts while A sits between pull and lookup."""
    (tmp_path / "config.yml").write_text("desktop:\n  remote_files_cache_bytes: 0\n")
    directory = remote_cache.cache_root(tmp_path)
    directory.mkdir(parents=True)
    a, b = "a" * 32, "b" * 32
    a_evicted = threading.Event()
    b_evicted = threading.Event()

    def pull(_root: Path, _owner: str, _member: str, source: dict[str, str]) -> dict[str, Any]:
        digest = source["attachment"]
        if digest == b:
            assert a_evicted.wait(5)
        (directory / f"{digest}.bin").write_bytes(digest.encode())
        return {"attachment": digest}

    real_evict, real_lookup = remote_cache.evict, remote_cache.lookup

    passes: list[int] = []

    def evict(root: Path, **kwargs: Any) -> int:
        # B's pull waits for A's pass, so the first pass is A's and the second B's.
        freed = real_evict(root, **kwargs)
        passes.append(1)
        (a_evicted if len(passes) == 1 else b_evicted).set()
        return freed

    def lookup(root: Path, session_ids: tuple[str, ...], digest: str) -> Any:
        if digest == a:
            assert b_evicted.wait(5)
        return real_lookup(root, session_ids, digest)

    monkeypatch.setattr(remote_fetch, "_pull", pull)
    monkeypatch.setattr(remote_fetch.cache, "evict", evict)
    monkeypatch.setattr(remote_fetch.cache, "lookup", lookup)

    async def one(digest: str) -> Any:
        return await remote_fetch.fetch(
            tmp_path,
            owner=OWNER,
            member_id=SID,
            source={"attachment": digest},
            key=digest,
            session_ids=(SID,),
            wait_s=10,
        )

    blob_a, blob_b = await asyncio.gather(one(a), one(b))
    assert Path(blob_a.path).stem == a
    assert Path(blob_b.path).stem == b


# -- R1-4 -------------------------------------------------------------------------


def test_r1_4_a_long_cjk_name_downloads(tmp_path: Path) -> None:
    from local_operator.session_files import service

    payload = b"CJK-NAMED-IMAGE"
    blob = _blob(tmp_path / "cache", payload)
    name = service.sanitise_name("文" * 100 + ".png", blob.path.stem, "image/png")
    assert len(name.encode()) > 255 - 19, "the case this test exists for"
    folder = tmp_path / "Downloads"
    folder.mkdir()
    receipt = rsf.land(blob, name=name, folder=folder, device="owner box")
    assert Path(receipt["path"]).read_bytes() == payload
    assert sorted(p.name for p in folder.iterdir()) == [name], "no temp file left behind"


# -- Q1-1 / I1-3 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q1_1_a_permanent_owner_refusal_carries_the_owners_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.server.routes.desktop_sessions import errors

    async def refused(*_a: Any, **_k: Any) -> Any:
        raise remote_fetch.RemoteFetchError("refused", "not_referenced", "owner said no")

    monkeypatch.setattr(remote_fetch, "fetch", refused)
    app = FastAPI()

    @app.get("/probe")
    async def probe(request: Request) -> Any:
        async with errors(request):
            await desktop_utils._fetch_peer(  # noqa: SLF001
                tmp_path, SID, _row(), SID, {"attachment": "a" * 32}, "a" * 32, (SID,)
            )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.get("/probe")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_on_peer", "the UI keys on the unchanged code"
    assert detail["params"] == {"reason": "not_referenced", "device": "owner box"}
    assert detail["text"] == (
        "This file cannot be shown here: the conversation on owner box no longer refers to it."
    )


def test_i1_3_the_update_command_names_the_device_id() -> None:
    row: Any = SimpleNamespace(owner_device=OWNER, owner_label="unnamed device")
    outdated = desktop_utils.PeerAttachmentOutdated(SID, row)
    assert outdated.peer == OWNER
    assert outdated.device == "unnamed device"


# -- QA round 2: Q2-1 / Q2-2 / Q2-3 (the Library route's CACHED remote branch) ------

ASSET = "la_" + "0f" * 8


@pytest_asyncio.fixture
async def cached_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The Library content route over a peer's digest asset whose bytes are cached.

    The index and catalogue reads are stubbed (what is under test is the route's
    branch for bytes already in ``cache/remote-files``); the cache, the lookup,
    ``_opened`` and the response are real.
    """
    from local_operator.config import ConfigManager
    from local_operator.library import service as library_service
    from local_operator.server.routes import desktop_library

    payload = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    blob = _blob(remote_cache.cache_root(tmp_path), payload, mime="image/svg+xml")
    digest = blob.path.stem
    asset = SimpleNamespace(
        id=ASSET, digest=digest, derived_name="evil.svg", content_type="image/svg+xml", path=None
    )
    monkeypatch.setattr(library_service, "_catalog_asset", lambda _config, _id: asset)
    monkeypatch.setattr(
        library_service, "_remote_occurrence", lambda _config, _id: {"session_id": SID}
    )
    monkeypatch.setattr(rsf, "_row_for", lambda _config, _sid: _row())
    # The digest open reads the row only on a local miss (#3872 R3-2).
    monkeypatch.setattr(desktop_utils, "peer_row", lambda _root, _sid, owner="": _row())
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "t")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_library.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer t"},
    ) as client:
        yield client, blob, payload


@pytest.mark.asyncio
async def test_q2_1_a_cached_remote_svg_is_opaque_and_sandboxed(cached_library) -> None:
    client, _blob_, payload = cached_library
    for _open_number in (1, 2):  # QA saw the first open safe and every later one not
        response = await client.get(f"/v1/desktop/library/assets/{ASSET}/content")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["content-security-policy"] == "sandbox"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.content == payload


@pytest.mark.asyncio
async def test_q2_2_an_eviction_before_the_body_re_resolves_never_truncates(
    cached_library, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.server.routes import desktop_library

    client, blob, payload = cached_library
    evictions: list[str] = []
    fetched: list[str] = []

    def evict_once(path: Path) -> None:
        if not evictions:
            path.unlink()
            evictions.append(path.name)

    # Evict at whichever point sits between this device's lookup and the first
    # body byte: the lazy stream (the old cached branch) or the open (``_opened``).
    real_stream, real_open = desktop_library._stream, rsf._open

    def stream(path: Path, *args: Any) -> Any:
        evict_once(path)
        return real_stream(path, *args)

    def open_(found: Any) -> Any:
        evict_once(found.path)
        return real_open(found)

    async def fetch_peer(root: Path, _sid: str, _row: Any, _member: str, source, key, _r):
        fetched.append(key)
        blob.path.write_bytes(payload)
        return remote_cache.CachedBlob(
            path=blob.path, size=len(payload), mime="image/svg+xml", source="cache"
        )

    monkeypatch.setattr(desktop_library, "_stream", stream)
    monkeypatch.setattr(rsf, "_open", open_)
    monkeypatch.setattr(desktop_utils, "_fetch_peer", fetch_peer)
    response = await client.get(f"/v1/desktop/library/assets/{ASSET}/content")
    assert evictions, "the probe never evicted"
    assert response.status_code == 200, response.text
    assert response.content == payload, "200 headers, then a truncated body"
    assert fetched == [blob.path.stem], "the eviction did not re-resolve through the owner"


@pytest.mark.asyncio
async def test_q2_3_a_second_eviction_says_still_on_its_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_twice(monkeypatch, _evicted(tmp_path), _evicted(tmp_path))
    with pytest.raises(desktop_utils.PeerAttachmentRetryable) as raised:
        await rsf.content(tmp_path, SID, "f1", None)
    assert raised.value.pending is True
    assert raised.value.code == "attachment_fetch_pending"
