"""Two synthetic mesh peers: mobile schema/media travel through real auth/TCP."""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import MutableMapping
from typing import Any

import pytest

from local_operator.harness.types import (
    AttachmentContent,
    Message,
    TextContent,
    ToolCall,
)
from local_operator.mobile.remote_sessions import (
    RemoteSessionError,
    media_chunk,
    mesh_read,
)
from local_operator.network import relay, store
from local_operator.network.sync import SYNC_CHUNK_BYTES
from local_operator.network.types import PeerRecord
from local_operator.resume import SessionRow
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript
from tests.unit.network.test_relay_e2e import _dialable_devices, devices  # noqa: F401


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("devices")


def seed(server: relay.RelayServer) -> tuple[str, bytes]:
    session_id = "mobile-peer-history"
    raw = bytes(range(256)) * (SYNC_CHUNK_BYTES // 128 + 1)
    attachment_store = AttachmentStore(server.root / "attachments")
    ref = attachment_store.put(base64.b64encode(raw).decode(), "video/mp4")
    assert ref is not None
    directory = server.root / "sessions" / session_id
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    assistant = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[ToolCall(id="clip-call", name="generate_video", arguments={})],
    )
    message = Message(
        role="tool",
        tool_call_id="clip-call",
        content=[
            TextContent(text="retained video"),
            AttachmentContent(kind="video", attachment=ref.digest, content_type="video/mp4"),
        ],
    )

    async def append() -> None:
        await transcript.append_message(assistant)
        await transcript.append_message(message)

    asyncio.run(append())
    return f"{assistant.id}:clip-call", raw


def test_mobile_history_and_multichunk_media_over_real_mesh(
    peer_pair: Any, monkeypatch: Any
) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    entry_id, expected = seed(server_b)
    row = SessionRow(
        id="mobile-peer-history",
        mtime=0.0,
        name="synthetic",
        owner_device=server_b.identity.device_id,
    )
    record = store.find_own_relay(server_a.root)
    assert record is not None
    reply = relay.control_request(
        record,
        "peer_session_history",
        peer=row.owner_device,
        session_id=row.id,
        mobile_session_read="history",
        limit=3,
    )
    assert reply is not None and reply.get("op") == "ack", reply
    page = mesh_read(server_a.root, row, "history", limit=3)
    assert page["entries"][0]["id"] == entry_id
    assert page["entries"][0]["videos"][0]["index"] == 0
    assert "payload" not in page["entries"][0]
    result = bytearray()
    identity = None
    while len(result) < len(expected):
        metadata, data = media_chunk(
            server_a.root,
            row,
            entry_id=entry_id,
            index=0,
            kind="video",
            offset=len(result),
            length=SYNC_CHUNK_BYTES,
            expected_media_id=identity,
        )
        identity = metadata["media_id"]
        assert metadata["session_id"] == row.id
        assert len(data) <= SYNC_CHUNK_BYTES
        result.extend(data)
    assert bytes(result) == expected
    # Media did not tear down the shared encrypted link, and no read engaged a
    # session owner. An ordinary read still traverses that same mesh afterward.
    assert mesh_read(server_a.root, row, "history", limit=3)["entries"][0]["id"] == entry_id
    assert not list((server_b.root / "running").glob("*"))


def test_old_peer_is_explicit_unsupported(peer_pair: Any, monkeypatch: Any) -> None:
    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    seed(server_b)
    link = server_a._ensure_link(server_b.identity.device_id)
    assert link is not None
    monkeypatch.setattr(
        link,
        "capabilities",
        tuple(cap for cap in link.capabilities if cap != "mobile-session-read-v1"),
    )
    row = SessionRow(
        id="mobile-peer-history",
        mtime=0.0,
        name="synthetic",
        owner_device=server_b.identity.device_id,
    )
    with pytest.raises(RemoteSessionError) as caught:
        mesh_read(server_a.root, row, "history", limit=3)
    assert caught.value.code == "remote_unsupported"
    assert caught.value.status == 409


@pytest.mark.asyncio
async def test_live_mobile_view_prompt_image_steer_uuid_and_release(
    peer_pair: Any, monkeypatch: Any
) -> None:
    import io
    import uuid

    from httpx import ASGITransport, AsyncClient
    from PIL import Image

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.mobile.remote_sessions import RemoteViews, resolve_owner
    from tests.unit.network.test_remote_viewer import _user_payload_with, _wait
    from tests.unit.network.test_session_plane import (
        _create_named_session_on_a_real_peer,
    )

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer, peer_pair, monkeypatch, name="mobile view", prompt=""
    )
    initial_threads = set(threading.enumerate())
    pool = RemoteViews()
    view = queue = second = None
    try:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
        row = await asyncio.to_thread(resolve_owner, created.server_a.root, created.session_id)
        assert row is not None
        view, queue = await pool.acquire(created.server_a.root, row)
        assert view.frame is not None and view.frame["session_id"] == created.session_id
        assert view.client is not None and view.client.connected
        same, second = await pool.acquire(created.server_a.root, row)
        assert same is view
        app = build_app(MobileDaemon(port=0, password="test-only"))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            await client.post("/login", data={"password": "test-only"})
            image = io.BytesIO()
            Image.new("RGB", (8, 8), color="red").save(image, format="PNG")
            command_id = str(uuid.uuid4())
            body = {
                "op": "prompt",
                "command_id": command_id,
                "text": "mobile remote image",
                "images": [
                    {"data": base64.b64encode(image.getvalue()).decode(), "mime_type": "image/png"}
                ],
            }
            response = await client.post(f"/api/sessions/{created.session_id}/command", json=body)
            assert response.status_code == 200, response.text
            assert response.json()["ok"] is True
            assert await _wait(
                lambda: _user_payload_with(created, "mobile remote image") is not None
            )
            payload = _user_payload_with(created, "mobile remote image")
            assert payload is not None
            retained = next(block for block in payload["content"] if "data" in block)
            assert Image.open(io.BytesIO(base64.b64decode(retained["data"]))).size == (8, 8)
            # The owner journals the command UUID, not one minted by either relay.
            assert any(
                entry["payload"].get("producer_command_id") == command_id
                for entry in created.owner.transcript_entries()
            )
            again = await client.post(f"/api/sessions/{created.session_id}/command", json=body)
            assert again.status_code == 200 and again.json()["duplicate"] is True, again.text
            await asyncio.to_thread(created.owner.wait_for_turn)
            user_entry = next(
                entry
                for entry in created.owner.transcript_entries()
                if entry["payload"].get("producer_command_id") == command_id
            )
            fetched = await client.get(
                f"/api/sessions/{created.session_id}/image",
                params={"entry": user_entry["id"], "i": "0"},
            )
            assert fetched.status_code == 200, fetched.text
            assert Image.open(io.BytesIO(fetched.content)).size == (8, 8)
            ranged = await client.get(
                f"/api/sessions/{created.session_id}/image",
                params={"entry": user_entry["id"]},
                headers={"Range": "bytes=-7"},
            )
            assert ranged.status_code == 206 and ranged.content == fetched.content[-7:]
            steer_id = str(uuid.uuid4())
            steered = await client.post(
                f"/api/sessions/{created.session_id}/command",
                json={
                    "op": "steer",
                    "command_id": steer_id,
                    "text": "mobile remote steering",
                    "images": body["images"],
                },
            )
            assert steered.status_code == 200, steered.text
            consumed = await client.post(
                f"/api/sessions/{created.session_id}/command",
                json={
                    "op": "prompt",
                    "command_id": str(uuid.uuid4()),
                    "text": "consume remote steer",
                    "images": [],
                },
            )
            assert consumed.status_code == 200, consumed.text
            assert await _wait(
                lambda: _user_payload_with(created, "mobile remote steering") is not None
            )
            steer_payload = _user_payload_with(created, "mobile remote steering")
            assert steer_payload is not None and any(
                "data" in block or "attachment" in block for block in steer_payload["content"]
            )
            assert await _wait(
                lambda: any(
                    "mobile remote image" in str(row)
                    for row in (view.frame or {}).get("transcript", [])
                )
            )
            history = await client.get(f"/api/sessions/{created.session_id}/history")
            assert history.status_code == 200, history.text
            assert any("mobile remote image" in row["text"] for row in history.json()["entries"])
        pool.release(view, queue)
        queue = None
        assert view.client is not None and view.client.connected
        pool.release(view, second)
        second = None
        assert view.client is None and not pool.views
        assert not (created.server_a.root / "sessions" / created.session_id).exists()
    finally:
        if view is not None:
            if queue is not None:
                pool.release(view, queue)
            if second is not None:
                pool.release(view, second)
        await asyncio.to_thread(created.stop)
        assert await _wait(
            lambda: not [
                thread
                for thread in threading.enumerate()
                if thread not in initial_threads and thread.name.startswith("mesh-stream-")
            ]
        )
        from local_operator.analytics import get_recorder

        await asyncio.to_thread(get_recorder().close)


def test_http_stopped_remote_range_matrix(peer_pair: Any, monkeypatch: Any) -> None:
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.mobile.remote_sessions import resolve_owner

    server_a, server_b, _host, _port = peer_pair
    _dialable_devices(peer_pair, monkeypatch)
    entry_id, raw = seed(server_b)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    row = resolve_owner(server_a.root, "mobile-peer-history")
    assert row is not None and row.owner_device == server_b.identity.device_id
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    route = "/api/sessions/mobile-peer-history/attachment"
    params = {"entry": entry_id, "kind": "video", "i": "0"}
    full = client.get(route, params=params)
    assert full.status_code == 200 and full.content == raw
    assert full.headers["content-length"] == str(len(raw))
    for header, expected, content_range in [
        ("bytes=2-9", raw[2:10], f"bytes 2-9/{len(raw)}"),
        ("bytes=-7", raw[-7:], f"bytes {len(raw)-7}-{len(raw)-1}/{len(raw)}"),
        ("bytes=500000-", raw[500000:], f"bytes 500000-{len(raw)-1}/{len(raw)}"),
    ]:
        response = client.get(route, params=params, headers={"Range": header})
        assert response.status_code == 206 and response.content == expected
        assert response.headers["content-range"] == content_range
    unsatisfiable = client.get(route, params=params, headers={"Range": f"bytes={len(raw)}-"})
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(raw)}"
    malformed = client.get(route, params=params, headers={"Range": "bytes=1-2,5-6"})
    assert malformed.status_code == 200 and malformed.content == raw
    assert client.get(route, params={**params, "entry": "not-in-this-session"}).status_code == 404
    from local_operator.harness.types import DocumentContent

    document_bytes = b"%PDF-retained-owner-document\n" * 100
    document = Message(
        role="user",
        content=[
            DocumentContent(
                data=base64.b64encode(document_bytes).decode(),
                filename="owner.pdf",
                mime_type="application/pdf",
            )
        ],
    )
    # The fixture has both devices in one process; a writer runs under its own
    # owner's environment, just as the production owner process does.
    with monkeypatch.context() as owner_env:
        owner_env.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
        asyncio.run(
            Transcript(server_b.root / "sessions" / "mobile-peer-history").append_message(document)
        )
    file_response = client.get(route, params={"entry": document.id, "kind": "file"})
    assert file_response.status_code == 200 and file_response.content == document_bytes
    assert file_response.headers["content-disposition"] == "attachment"
    page = client.get("/api/sessions/mobile-peer-history/history").json()
    assert any(
        row.get("files", [{}])[0].get("filename") == "owner.pdf"
        for row in page["entries"]
        if row.get("files")
    )
    assert not (server_a.root / "sessions" / "mobile-peer-history").exists()
    assert not list((server_b.root / "running").glob("*"))


def test_view_only_media_and_no_view_refusal(peer_pair: Any, monkeypatch: Any) -> None:
    from tests.unit.network.test_relay_e2e import _pair_settled

    server_a, server_b, _host, _port = peer_pair
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host, _port = _pair_settled(
        peer_pair,
        monkeypatch,
        role="read",
        settings=relay.NetworkSettings(port=server_b.settings.port, listen_address="127.0.0.1"),
    )
    entry_id, raw = seed(server_a)
    row = SessionRow(
        id="mobile-peer-history",
        mtime=0.0,
        name="synthetic",
        owner_device=server_a.identity.device_id,
    )
    metadata, data = media_chunk(
        server_b.root, row, entry_id=entry_id, kind="video", index=0, offset=0, length=7
    )
    assert data == raw[:7] and metadata["total"] == len(raw)
    # Revoke through the real receiver-local grant API. Existing open links
    # reread membership on every frame, including the new media purpose.
    with store.mutate(record.network_id, server_a.root) as current:
        relay.set_member_capabilities(
            current, device_id=server_b.identity.device_id, revoke=("view",)
        )
        store.save(current, server_a.root)
    with pytest.raises(RemoteSessionError) as caught:
        media_chunk(
            server_b.root, row, entry_id=entry_id, kind="video", index=0, offset=0, length=7
        )
    assert caught.value.code == "remote_forbidden" and caught.value.status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["http.response.start", "http.response.body"])
async def test_http_sse_cancellation_closes_owner_stream(
    peer_pair: Any, monkeypatch: Any, boundary: str
) -> None:
    from local_operator.analytics import get_recorder
    from local_operator.mobile.daemon import MobileDaemon, build_app
    from tests.unit.network.test_remote_viewer import _wait
    from tests.unit.network.test_session_plane import (
        _create_named_session_on_a_real_peer,
    )

    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="mobile stream",
        prompt="",
    )
    try:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(created.server_a.root))
        from local_operator.mobile.auth import COOKIE_NAME, sign_cookie

        app = build_app(MobileDaemon(port=0, password="test-only"))
        frames: list[dict[str, Any]] = []
        path = f"/api/sessions/{created.session_id}/events"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"cookie", f"{COOKIE_NAME}={sign_cookie('test-only')}".encode())],
            "server": ("localhost", 80),
            "client": ("test", 1),
        }

        async def receive() -> dict[str, Any]:
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(frame: MutableMapping[str, Any]) -> None:
            frames.append(dict(frame))
            if frame["type"] == boundary:
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await app(scope, receive, send)
        assert frames[0]["status"] == 200
        if boundary == "http.response.body":
            assert b"event: projection" in frames[-1]["body"]
            assert created.session_id.encode() in frames[-1]["body"]
        assert await _wait(lambda: not created.server_a._streams and not created.server_b._streams)
    finally:
        await asyncio.to_thread(created.stop)
        await asyncio.to_thread(get_recorder().close)


@pytest.mark.asyncio
async def test_new_http_subscriber_rechecks_revoked_shared_view(
    peer_pair: Any, monkeypatch: Any
) -> None:
    import uuid

    from httpx import ASGITransport, AsyncClient

    from local_operator.analytics import get_recorder
    from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.mobile.remote_sessions import RemoteViews
    from tests.unit.network.test_relay_e2e import _pair_settled
    from tests.unit.network.test_remote_viewer import _wait
    from tests.unit.network.test_session_plane import (
        _serve_real_sessions,
        _start_real_session,
    )

    server_a, server_b, _host, _port = peer_pair
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host, _port = _pair_settled(
        peer_pair,
        monkeypatch,
        role="read",
        settings=relay.NetworkSettings(port=server_b.settings.port, listen_address="127.0.0.1"),
    )
    await asyncio.to_thread(seed, server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    served = _serve_real_sessions(monkeypatch, server_a.root)
    owner = await asyncio.to_thread(
        _start_real_session, server_a.root, "mobile-peer-history", str(server_a.root)
    )
    served["mobile-peer-history"] = owner
    pool = RemoteViews()
    monkeypatch.setattr("local_operator.mobile.remote_sessions.RemoteViews", lambda: pool)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    app = build_app(MobileDaemon(port=0, password="test-only"))
    path = "/api/sessions/mobile-peer-history/events"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"cookie", f"{COOKIE_NAME}={sign_cookie('test-only')}".encode())],
        "server": ("localhost", 80),
        "client": ("synthetic", 1),
    }
    seeded = asyncio.Event()
    held = asyncio.Event()
    frames: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def send(frame: MutableMapping[str, Any]) -> None:
        frames.append(dict(frame))
        if frame["type"] == "http.response.body" and b"event: projection" in frame["body"]:
            seeded.set()
            await held.wait()

    subscriber = asyncio.create_task(app(scope, receive, send))
    view = None
    upstream: Any = None
    try:
        await asyncio.wait_for(seeded.wait(), 15)
        assert frames[0]["status"] == 200
        view = next(iter(pool.views.values()))
        upstream = view.client
        assert upstream is not None and upstream.connected and view.readers == 1
        # This is the receiver's actual local grant mutation, not a simulated
        # denied read. Keep A's HTTP subscription alive to expose cached reuse.
        with store.mutate(record.network_id, server_a.root) as current:
            relay.set_member_capabilities(
                current, device_id=server_b.identity.device_id, revoke=("view",)
            )
            store.save(current, server_a.root)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            assert (await client.get(path)).status_code == 401
            await client.post("/login", data={"password": "test-only"})
            history = await client.get("/api/sessions/mobile-peer-history/history")
            prompt = await client.post(
                "/api/sessions/mobile-peer-history/command",
                json={"op": "prompt", "command_id": str(uuid.uuid4()), "text": "denied"},
            )
            assert history.status_code == prompt.status_code == 403
            assert history.json()["code"] == prompt.json()["code"] == "remote_forbidden"
            # Stop at HTTP headers on the buggy path so a wrongly admitted SSE
            # cannot hang the regression waiting for its never-ending response.
            second: list[dict[str, Any]] = []

            async def second_send(frame: MutableMapping[str, Any]) -> None:
                second.append(dict(frame))
                if frame["type"] == "http.response.start" and frame["status"] == 200:
                    raise asyncio.CancelledError

            try:
                await app(scope, receive, second_send)
            except asyncio.CancelledError:
                pass
            assert second[0]["status"] == 403, second
            body = b"".join(frame.get("body", b"") for frame in second)
            assert b"remote_forbidden" in body and b"event: projection" not in body
        assert view.readers == 1 and len(view.queues) == 1
        assert view.client is upstream and upstream.connected
    finally:
        subscriber.cancel()
        await asyncio.gather(subscriber, return_exceptions=True)
        assert not pool.views
        if view is not None:
            assert view.readers == 0 and not view.queues and view.client is None
        # Capability-only revocation has a pre-existing owner-stream close gap;
        # the new HTTP reference must still unwind without retaining a client.
        await asyncio.to_thread(owner.stop)
        await asyncio.to_thread(get_recorder().close)
        assert await _wait(lambda: upstream is None or not upstream.connected)


@pytest.mark.asyncio
async def test_view_grant_does_not_admit_mobile_prompt(peer_pair: Any, monkeypatch: Any) -> None:
    import uuid

    from httpx import ASGITransport, AsyncClient

    from local_operator.analytics import get_recorder
    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.mobile.remote_sessions import RemoteViews, resolve_owner
    from tests.unit.network.test_relay_e2e import _pair_settled
    from tests.unit.network.test_remote_viewer import _wait
    from tests.unit.network.test_session_plane import (
        _serve_real_sessions,
        _start_real_session,
    )

    server_a, server_b, _host, _port = peer_pair
    server_b.bind()
    server_b.bind_control()
    server_b.start()
    _pair_settled(
        peer_pair,
        monkeypatch,
        role="read",
        settings=relay.NetworkSettings(port=server_b.settings.port, listen_address="127.0.0.1"),
    )
    await asyncio.to_thread(seed, server_a)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    served = _serve_real_sessions(monkeypatch, server_a.root)
    owner = await asyncio.to_thread(
        _start_real_session, server_a.root, "mobile-peer-history", str(server_a.root)
    )
    served["mobile-peer-history"] = owner
    initial_threads = set(threading.enumerate())
    pool = RemoteViews()
    view = queue = None
    try:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
        row = await asyncio.to_thread(resolve_owner, server_b.root, "mobile-peer-history")
        assert row is not None
        view, queue = await pool.acquire(server_b.root, row)
        assert view.client is not None and view.client.connected
        app = build_app(MobileDaemon(port=0, password="test-only"))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            await client.post("/login", data={"password": "test-only"})
            response = await client.post(
                "/api/sessions/mobile-peer-history/command",
                json={
                    "op": "prompt",
                    "command_id": str(uuid.uuid4()),
                    "text": "must not be admitted",
                    "images": [],
                },
            )
            assert response.status_code == 403, response.text
            assert response.json()["code"] == "remote_forbidden"
            assert view.client.connected
            assert not any(
                "must not be admitted" in str(entry) for entry in owner.transcript_entries()
            )
    finally:
        if view is not None and queue is not None:
            pool.release(view, queue)
        await asyncio.to_thread(owner.stop)
        assert await _wait(
            lambda: not [
                thread
                for thread in threading.enumerate()
                if thread not in initial_threads and thread.name.startswith("mesh-stream-")
            ]
        )
        await asyncio.to_thread(get_recorder().close)


@pytest.mark.asyncio
async def test_async_courier_cancellation_closes_its_socket() -> None:
    from local_operator.network import relay
    from tests.unit.network.conftest import NET_WAIT_BUDGET_S

    arrived = asyncio.Event()
    closed = asyncio.Event()
    tasks: set[asyncio.Task[Any]] = set()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        try:
            await reader.readline()  # synthetic local auth, never real secrets
            await reader.readline()
            arrived.set()
            assert await reader.read() == b""
        finally:
            writer.close()
            await writer.wait_closed()
            closed.set()
            tasks.remove(task)

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    pending = asyncio.create_task(
        relay.async_control_request(
            PeerRecord(pid=0, control_port=port, control_key="synthetic-test-key"),
            "peer_session_history",
            timeout=NET_WAIT_BUDGET_S,
        )
    )
    try:
        await asyncio.wait_for(arrived.wait(), NET_WAIT_BUDGET_S)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(closed.wait(), NET_WAIT_BUDGET_S)
        assert not tasks
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
