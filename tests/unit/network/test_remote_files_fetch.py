"""C2 of remote files parity: a peer session's bytes fetched on miss, over the REAL loopback.

docs/design/remote-files-parity.md §1/§4 and the C2 acceptance list (a)–(l). A owns
SESSION (``mesh`` fixture: two serve-shaped relays, a real pairing, B a READ member);
B is the viewer: its desktop routes run on an ASGI app over ``DesktopSessions(B.root)``
and reach A only through B's relay (``session_file_pull``) and the real link.

RED on base (recorded in the PR): (a) and (b) answered 409 ``attachment_on_peer``;
(i) answered 404.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import struct
import tracemalloc
import wave
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import files, store, sync, wire
from local_operator.remote_files import cache as remote_cache
from local_operator.remote_files import fetch as remote_fetch
from local_operator.server.routes import desktop_sessions as desktop_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mesh_file_pull import mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import (  # noqa: F401
    SESSION,
    Devices,
    devices,
    pair,
)

TOKEN = "synthetic-remote-files-token"
CHILD = "c0ffee000001"


def _wav(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        frames = int(8000 * seconds)
        writer.writeframes(
            b"".join(struct.pack("<h", int(9000 * math.sin(i / 9))) for i in range(frames))
        )
    return buffer.getvalue()


PNG = b"\x89PNG\r\n\x1a\n" + os.urandom(300_000)
WAV = _wav()
MP4 = b"\x00\x00\x00\x18ftypmp42" + os.urandom(1_200_000)


def _own(server: Any, payload: bytes, mime: str, *, session: str = SESSION) -> str:
    """Put ``payload`` in A's store and reference it from ``session``'s transcript."""
    ref = AttachmentStore(Path(server.root) / "attachments").put_bytes(payload, mime)
    assert ref is not None
    directory = Path(server.root) / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        row = {"id": f"e{ref.digest[:6]}", "content": [{"attachment": ref.digest}]}
        handle.write(json.dumps(row) + "\n")
    return ref.digest


def _cache_names(server: Any) -> list[str]:
    directory = remote_cache.cache_root(Path(server.root))
    return sorted(p.name for p in directory.glob("*.bin")) if directory.is_dir() else []


@pytest_asyncio.fixture
async def viewer(
    mesh: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, Devices]]:
    """B's desktop routes over the real relay pair."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    remote_fetch.STATS["pulls"] = 0
    app = FastAPI()
    app.state.config_manager = ConfigManager(Path(mesh[1].root))
    pool = DesktopSessions(Path(mesh[1].root))
    app.state.desktop_sessions = pool
    app.include_router(desktop_routes.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client, mesh
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_g2_png_wav_and_mp4_arrive_into_the_cache_not_the_store(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(a) Each answers 200, byte-identical, landed in cache/remote-files only."""
    client, (server_a, server_b, *_rest) = viewer
    for payload, mime in ((PNG, "image/png"), (WAV, "audio/wav"), (MP4, "video/mp4")):
        digest = _own(server_a, payload, mime)
        response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
        assert response.status_code == 200, response.text
        assert response.content == payload, f"{mime}: the bytes differ"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["content-type"].startswith(mime)
        assert f"{digest}.bin" in _cache_names(server_b)
        assert not (Path(server_b.root) / "attachments" / f"{digest}.bin").exists()
    assert remote_fetch.STATS["pulls"] == 3


@pytest.mark.asyncio
async def test_b_a_replica_held_blob_serves_with_no_relay_call(
    viewer: tuple[AsyncClient, Devices], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) G1: the replica store is read, and nothing asks the owner."""
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png")
    replica = sync.replica_dir(Path(server_b.root), SESSION) / "attachments"
    replica.mkdir(parents=True)
    (replica / f"{digest}.bin").write_bytes(PNG)
    (replica / f"{digest}.json").write_text(json.dumps({"mime_type": "image/png"}))

    def no_pull(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a replica-held blob was pulled from the owner")

    monkeypatch.setattr(remote_fetch, "_pull", no_pull)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
    assert response.status_code == 200, response.text
    assert response.content == PNG
    assert _cache_names(server_b) == []


@pytest.mark.asyncio
async def test_c_a_range_request_answers_206(viewer: tuple[AsyncClient, Devices]) -> None:
    """(c) ``Range: bytes=0-99`` → 206 with exactly 100 bytes."""
    client, (server_a, *_rest) = viewer
    digest = _own(server_a, MP4, "video/mp4")
    response = await client.get(
        f"/v1/desktop/sessions/{SESSION}/attachments/{digest}", headers={"Range": "bytes=0-99"}
    )
    assert response.status_code == 206, response.text
    assert response.content == MP4[:100]
    assert response.headers["content-range"] == f"bytes 0-99/{len(MP4)}"


@pytest.mark.asyncio
async def test_d_an_unreachable_owner_is_504_then_200_once_it_is_back(
    viewer: tuple[AsyncClient, Devices], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) A link that cannot be dialled → 504 owner_unreachable, retryable; then 200.

    The owner is made undialable at B's one dial seam (``_ensure_link`` → ``None``),
    which is the exact branch ``pull_file`` takes for a stopped owner relay.
    """
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png")
    for link in list(getattr(server_b, "_links", {}).values()):
        link.close("test")
    monkeypatch.setattr(server_b, "_ensure_link", lambda _device: None)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
    assert response.status_code == 504, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_owner_unreachable"
    assert "device-a" in detail["text"], detail
    assert _cache_names(server_b) == []
    monkeypatch.undo()
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
    assert response.status_code == 200, response.text
    assert response.content == PNG


@pytest.mark.asyncio
async def test_e_a_slow_pull_answers_pending_and_the_retry_joins_it(
    viewer: tuple[AsyncClient, Devices], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(e) Throttled transfer → 504 fetch_pending + Retry-After; the retry gets 200
    from the SAME single transfer (one owner stat phase, one pull)."""
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png")
    transfer = files._transfer
    monkeypatch.setattr(
        files, "_transfer", lambda *a, **k: (__import__("time").sleep(1.5), transfer(*a, **k))[1]
    )
    stats: list[int] = []
    validated = files._validated_stat
    monkeypatch.setattr(
        files, "_validated_stat", lambda *a, **k: (stats.append(1), validated(*a, **k))[1]
    )
    monkeypatch.setattr(remote_fetch, "REMOTE_FETCH_WAIT_S", 0.3)
    url = f"/v1/desktop/sessions/{SESSION}/attachments/{digest}"
    first = await client.get(url)
    assert first.status_code == 504, first.text
    assert first.json()["detail"]["code"] == "attachment_fetch_pending"
    assert first.headers["retry-after"] == "2"
    assert remote_fetch.pending_count() == 1, "the pull stopped when the wait expired"
    monkeypatch.setattr(remote_fetch, "REMOTE_FETCH_WAIT_S", 10.0)
    second = await client.get(url)
    assert second.status_code == 200, second.text
    assert second.content == PNG
    assert remote_fetch.STATS["pulls"] == 1 and len(stats) == 1, (remote_fetch.STATS, stats)


@pytest.mark.asyncio
async def test_f_a_revoked_view_is_409_with_nothing_cached(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(f) I1-1: without ``view`` the owner refuses and nothing lands anywhere."""
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png")
    record = store.list_networks(server_a.root)[0]
    with store.mutate(record.network_id, server_a.root) as mutable:
        member = mutable.member(server_b.identity.device_id)
        assert member is not None
        member.capabilities = ["list"]
        store.save(mutable, server_a.root)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "attachment_on_peer"
    assert _cache_names(server_b) == []
    staged = files.staging_root(Path(server_b.root))
    assert not staged.is_dir() or list(staged.iterdir()) == []


@pytest.mark.asyncio
async def test_g_an_unreferenced_digest_is_409_with_nothing_cached(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(g) I1-1: a digest in A's shared store the session never referenced."""
    client, (server_a, server_b, *_rest) = viewer
    ref = AttachmentStore(Path(server_a.root) / "attachments").put_bytes(PNG, "image/png")
    assert ref is not None
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{ref.digest}")
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "attachment_on_peer"
    assert _cache_names(server_b) == []


@pytest.mark.asyncio
async def test_h_an_owner_without_file_transfer_is_409_outdated(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(h) No ``mesh-file-v1`` on the link → 409 attachment_peer_outdated, named."""
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png")
    link = server_b._ensure_link(server_a.identity.device_id)  # noqa: SLF001
    assert link is not None
    link.capabilities = frozenset(c for c in link.capabilities if c != wire.MESH_FILE_V1)
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_peer_outdated"
    assert "lop network update device-a" in detail["text"], detail


@pytest.mark.asyncio
async def test_i_a_remote_subagents_image_serves_through_the_child_route(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(i) The child route reaches the owner when the PARENT is a peer's (base: 404)."""
    client, (server_a, *_rest) = viewer
    digest = _own(server_a, PNG, "image/png", session=CHILD)
    response = await client.get(
        f"/v1/desktop/sessions/{SESSION}/children/{CHILD}/attachments/{digest}"
    )
    assert response.status_code == 200, response.text
    assert response.content == PNG


@pytest.mark.asyncio
async def test_j_a_remote_wav_has_a_waveform(viewer: tuple[AsyncClient, Devices]) -> None:
    """(j) Peaks computed beside the cached bytes, never in the authoritative store."""
    client, (server_a, server_b, *_rest) = viewer
    digest = _own(server_a, WAV, "audio/wav")
    response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}/waveform")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["peaks"] and payload["bars"], payload
    assert (remote_cache.cache_root(Path(server_b.root)) / f"{digest}.peaks.json").exists()


@pytest.mark.asyncio
async def test_k_serving_a_64_mib_blob_keeps_ram_flat(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(k) I1-3: a 64 MiB peer blob is streamed; tracemalloc peak < 4 MiB.

    The blob sits in B's own store (lookup step 1), which base ALSO serves for a
    peer session — through ``AttachmentStore.get`` (bytes + base64 + decode), so
    base's peak is several times the file (RED on base).
    """
    client, (_server_a, server_b, *_rest) = viewer
    payload = os.urandom(64 * 1024 * 1024)
    ref = AttachmentStore(Path(server_b.root) / "attachments").put_bytes(payload, "video/mp4")
    assert ref is not None
    expected = hashlib.sha256(payload).hexdigest()
    del payload
    path = f"/v1/desktop/sessions/{SESSION}/attachments/{ref.digest}"
    received = hashlib.sha256()
    sizes = {"total": 0, "status": 0}

    # Driven at the ASGI boundary, NOT through httpx's ASGITransport: that client
    # collects the whole body in memory, which would measure the test, not the route.
    finished = asyncio.Event()
    asked = [0]

    async def receive() -> dict[str, Any]:
        # The request body once; after that, Starlette's disconnect listener
        # waits here until the response is complete (an immediate answer would
        # spin it forever).
        asked[0] += 1
        if asked[0] == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            sizes["status"] = message["status"]
        elif message["type"] == "http.response.body":
            received.update(message.get("body", b""))
            sizes["total"] += len(message.get("body", b""))
            if not message.get("more_body", False):
                finished.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"localhost"), (b"authorization", f"Bearer {TOKEN}".encode())],
        "client": ("127.0.0.1", 1),
        "server": ("localhost", 80),
    }
    app = client._transport.app  # type: ignore[attr-defined]  # noqa: SLF001
    tracemalloc.start()
    try:
        await app(scope, receive, send)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    total = sizes["total"]
    assert sizes["status"] == 200
    assert total == 64 * 1024 * 1024 and received.hexdigest() == expected
    assert peak < 4 * 1024 * 1024, f"serving buffered {peak / 2**20:.1f} MiB"


@pytest.mark.asyncio
async def test_l_eviction_keeps_the_cache_within_its_budget(
    viewer: tuple[AsyncClient, Devices],
) -> None:
    """(l) I4-1: budget = 3 files via the real config key; land 5; the 2 oldest go."""
    client, (server_a, server_b, *_rest) = viewer
    size = 200_000
    blobs = [bytes([index]) * size for index in range(5)]
    (Path(server_b.root) / "config.yml").write_text(
        f"desktop:\n  remote_files_cache_bytes: {3 * size}\n", encoding="utf-8"
    )
    digests = []
    for index, payload in enumerate(blobs):
        digest = _own(server_a, payload, "image/png")
        digests.append(digest)
        response = await client.get(f"/v1/desktop/sessions/{SESSION}/attachments/{digest}")
        assert response.status_code == 200, (index, response.text)
        assert response.content == payload
        # mtime resolution: keep landings strictly ordered for the LRU.
        await asyncio.sleep(0.02)
    present = _cache_names(server_b)
    assert f"{digests[0]}.bin" not in present and f"{digests[1]}.bin" not in present, present
    assert f"{digests[-1]}.bin" in present
    assert remote_cache.total_bytes(Path(server_b.root)) <= 3 * size
