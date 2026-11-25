"""Remote mobile reads preserve owner identity, schema and HTTP auth semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile import remote_sessions as remote
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.network.types import MeshRefusal
from local_operator.resume import SessionRow
from local_operator.session.transcript import Transcript


def build_history(root: Path, session_id: str = "remote-chat", count: int = 12) -> list[str]:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    ids: list[str] = []

    async def append() -> None:
        for index in range(count):
            message = Message.user(f"turn {index}")
            ids.append(message.id)
            await transcript.append_message(message)

    asyncio.run(append())
    return ids


def test_owner_history_uses_mobile_rows_and_pages_to_beginning(tmp_path: Path) -> None:
    ids = build_history(tmp_path)
    seen: list[str] = []
    before = None
    while True:
        page = remote.owner_read(
            tmp_path,
            "remote-chat",
            {
                "mobile_session_read": "history",
                "before_id": before,
                "limit": 3,
            },
        )
        assert page["mobile_session_read"] == 1
        assert all("payload" not in row and "text" in row for row in page["entries"])
        seen = [row["id"] for row in page["entries"]] + seen
        if not page["has_more"]:
            break
        before = page["entries"][0]["id"]
    assert seen == ids
    assert not (tmp_path / "runtime").exists()


def test_owner_history_forward_and_projection_are_cold(tmp_path: Path) -> None:
    ids = build_history(tmp_path)
    page = remote.owner_read(
        tmp_path,
        "remote-chat",
        {
            "mobile_session_read": "history",
            "after_id": ids[3],
            "limit": 3,
        },
    )
    assert [row["id"] for row in page["entries"]] == ids[4:7]
    result = remote.owner_read(tmp_path, "remote-chat", {"mobile_session_read": "projection"})
    assert result["projection"]["session_id"] == "remote-chat"
    assert result["projection"]["pid"] == 0


@pytest.mark.parametrize("session_id", ["../outside", ".", "", "/outside"])
def test_resolution_refuses_paths_before_mesh(tmp_path: Path, session_id: str) -> None:
    with pytest.raises(remote.RemoteSessionError) as caught:
        remote.resolve_owner(tmp_path, session_id)
    assert caught.value.status == 404


def test_http_remote_failure_is_not_phone_logout(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    def unavailable(*_args: Any) -> Any:
        raise remote.RemoteSessionError("remote_owner_unresolved")

    monkeypatch.setattr(remote, "resolve_owner", unavailable)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    assert client.get("/api/sessions/remote-chat/history").status_code == 401
    client.post("/login", data={"password": "test-only"})
    response = client.get("/api/sessions/remote-chat/history")
    assert response.status_code == 503
    assert response.json()["remote"] is True
    assert response.json()["code"] == "remote_owner_unresolved"


def test_http_remote_history_forwards_exact_cursors(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    row = SessionRow(id="remote-chat", mtime=0.0, name="synthetic", owner_device="peer-b")
    monkeypatch.setattr(remote, "resolve_owner", lambda *_args: row)
    calls: list[dict[str, Any]] = []

    def read(root: Path, owner: Any, purpose: str, **fields: Any) -> dict[str, Any]:
        assert root == tmp_path and owner is row and purpose == "history"
        calls.append(fields)
        return {
            "mobile_session_read": 1,
            "entries": [{"id": "entry", "text": "remote"}],
            "has_more": False,
        }

    monkeypatch.setattr(remote, "mesh_read", read)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    response = client.get("/api/sessions/remote-chat/history?after=previous&limit=3")
    assert response.status_code == 200
    assert response.json()["entries"][0]["text"] == "remote"
    assert calls == [{"before_id": None, "after_id": "previous", "limit": 3}]


@pytest.mark.asyncio
@pytest.mark.parametrize("release_existing_first", [False, True])
async def test_cancelled_shared_authorization_unwinds_tentative_reference(
    tmp_path: Path, monkeypatch: Any, release_existing_first: bool
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    row = SessionRow(id="remote-chat", mtime=0.0, name="synthetic", owner_device="peer-b")
    pool = remote.RemoteViews()
    view = remote.RemoteView(tmp_path, row)
    upstream = SimpleNamespace(connected=True, close=Mock())
    view.client = upstream
    view.frame = {"session_id": row.id, "transcript": []}
    existing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    view.queues.add(existing)
    view.readers = 1
    pool.views[(row.owner_device, row.id)] = view
    checking = asyncio.Event()

    async def pending_read(function: Any, *args: Any) -> Any:
        assert function is remote.mesh_read
        assert args == (tmp_path, row, "projection")
        checking.set()
        await asyncio.Future()

    # Hold precisely the new authorization await, not a wall-clock approximation
    # or a real worker thread that would outlive cancellation of its awaiter.
    monkeypatch.setattr(remote.asyncio, "to_thread", pending_read)
    joining = asyncio.create_task(pool.acquire(tmp_path, row))
    try:
        await asyncio.wait_for(checking.wait(), 5)
        assert view.readers == 2 and len(view.queues) == 1
        if release_existing_first:
            pool.release(view, existing)
            assert view.readers == 1 and view.client is upstream
            upstream.close.assert_not_called()
    finally:
        joining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joining
    if not release_existing_first:
        assert view.readers == 1 and view.client is upstream
        assert view.queues == {existing}
        upstream.close.assert_not_called()
        pool.release(view, existing)
    assert not pool.views and not view.queues and view.readers == 0
    assert view.client is None
    upstream.close.assert_called_once()


def test_unknown_mobile_purpose_refuses_not_legacy_page(tmp_path: Path) -> None:
    build_history(tmp_path)
    with pytest.raises(MeshRefusal):
        remote.owner_read(tmp_path, "remote-chat", {"mobile_session_read": "arbitrary-path"})


@pytest.mark.parametrize(
    "fields",
    [
        {"index": True},
        {"offset": True},
        {"length": True},
        {"offset": -1},
        {"length": 2**100},
        {"index": -1},
        {"path": "/etc/passwd"},
        {"digest": "a" * 32},
        {"before_id": "history-and-media"},
        {"source_read": "metadata"},
    ],
)
def test_media_coordinates_and_mixed_authority_refuse(fields: dict[str, Any]) -> None:
    frame = {
        "mobile_session_read": "media",
        "entry_id": "retained",
        "index": 0,
        "kind": "video",
        "offset": 0,
        "length": 1,
        **fields,
    }
    with pytest.raises(MeshRefusal):
        remote.validate_read(frame)


def test_document_media_is_retained_and_explicitly_rooted(tmp_path: Path, monkeypatch: Any) -> None:
    import base64

    from local_operator.harness.types import DocumentContent
    from local_operator.session.media import session_media

    owner = tmp_path / "owner"
    other = tmp_path / "other"
    directory = owner / "sessions" / "docs"
    directory.mkdir(parents=True)
    raw = b"%PDF-retained synthetic document\n" * 200
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner))
    transcript = Transcript(directory)
    message = Message(
        role="user",
        content=[
            DocumentContent(
                data=base64.b64encode(raw).decode(),
                filename="proof.pdf",
                mime_type="application/pdf",
                path="/untrusted/original/path",
            )
        ],
    )
    asyncio.run(transcript.append_message(message))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(other))
    source = session_media(owner, "docs", message.id, 0, "file")
    assert source is not None and source.mime == "application/pdf"
    assert source.read(2, 31) == raw[2:33]
    assert session_media(other, "docs", message.id, 0, "file") is None
    history = remote.owner_read(owner, "docs", {"mobile_session_read": "history", "limit": 3})
    assert history["entries"][0]["files"] == [
        {"index": 0, "filename": "proof.pdf", "mime_type": "application/pdf", "bytes": len(raw)}
    ]
    from local_operator.mobile.projection import fold_messages_to_entries

    assert fold_messages_to_entries([message])[0].files == history["entries"][0]["files"]
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner))
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    fetched = client.get(
        "/api/sessions/docs/attachment", params={"kind": "file", "entry": message.id}
    )
    assert fetched.status_code == 200 and fetched.content == raw
    assert fetched.headers["content-disposition"] == "attachment"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(other))
    result = remote.owner_read(
        owner,
        "docs",
        {
            "mobile_session_read": "media",
            "entry_id": message.id,
            "index": 0,
            "kind": "file",
            "offset": 0,
            "length": 10,
        },
    )
    assert base64.b64decode(result["data_b64"]) == raw[:10]
    with pytest.raises(MeshRefusal) as caught:
        remote.owner_read(
            owner,
            "docs",
            {
                "mobile_session_read": "media",
                "entry_id": message.id,
                "index": 0,
                "kind": "file",
                "offset": 0,
                "length": 10,
                "expected_media_id": "0" * 64,
            },
        )
    assert caught.value.code == "media_changed"
    asyncio.run(transcript.append_prune(message.id, "redacted"))
    assert session_media(owner, "docs", message.id, 0, "file") is None


def test_oversized_mobile_history_row_keeps_cursor_and_incomplete_fact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from local_operator.network import relay

    directory = tmp_path / "sessions" / "large"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    message = Message.user("large message " * 20_000)
    asyncio.run(transcript.append_message(message))
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", 64 * 1024)
    page = remote.owner_read(tmp_path, "large", {"mobile_session_read": "history", "limit": 3})
    row = page["entries"][0]
    assert row["id"] == message.id
    assert row["text_complete"] is False and "payload" not in row
    assert relay.session_history_reply_bytes(None, page) <= relay.SESSION_HISTORY_REPLY_BUDGET_BYTES


def test_transport_trimmed_forward_pages_do_not_skip_newer_rows(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from local_operator.network import relay

    directory = tmp_path / "sessions" / "forward"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    messages = [Message.user(str(index) + "x" * 20_000) for index in range(6)]

    async def append() -> None:
        for message in messages:
            await transcript.append_message(message)

    asyncio.run(append())
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", 64 * 1024)
    after = messages[0].id
    seen: list[str] = []
    while True:
        page = remote.owner_read(
            tmp_path, "forward", {"mobile_session_read": "history", "after_id": after, "limit": 6}
        )
        seen.extend(row["id"] for row in page["entries"])
        assert (
            relay.session_history_reply_bytes(None, page)
            <= relay.SESSION_HISTORY_REPLY_BUDGET_BYTES
        )
        if not page["has_more"]:
            break
        after = page["entries"][-1]["id"]
    assert seen == [message.id for message in messages[1:]]


def test_unknown_block_mime_uses_retained_sidecar_without_whole_body_read(tmp_path: Path) -> None:
    from local_operator.harness.types import AttachmentContent
    from local_operator.session.attachments import AttachmentStore
    from local_operator.session.media import session_media

    store = AttachmentStore(tmp_path / "attachments")
    ref = store.put_bytes(b"synthetic-webm-content", "video/webm")
    assert ref is not None
    directory = tmp_path / "sessions" / "mime"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    message = Message(
        role="tool",
        tool_call_id="webm",
        content=[AttachmentContent(kind="video", attachment=ref.digest)],
    )
    asyncio.run(transcript.append_message(message))
    source = session_media(tmp_path, "mime", message.id, 0, "video")
    assert source is not None and source.mime == "video/webm"
    assert source.read(0, 9) == b"synthetic"
    assert store.mime_type_for("../not-a-digest") is None


def test_range_numbers_are_bounded_by_representation_not_integer_parser() -> None:
    from local_operator.mobile.daemon import _range_window

    enormous = "9" * 5000
    assert _range_window(f"bytes={enormous}-", 17) == "unsatisfiable"
    assert _range_window(f"bytes=-{enormous}", 17) == (0, 16)
    assert _range_window(f"bytes=2-{enormous}", 17) == (2, 16)
    assert _range_window("bytes=" + "0" * 5000 + "2-9", 17) == (2, 9)


def test_remote_capability_and_unsupported_actions_are_explicit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    row = SessionRow(id="remote-chat", mtime=0.0, name="synthetic", owner_device="peer-b")
    monkeypatch.setattr(remote, "resolve_owner", lambda *_args: row)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    response = client.get("/api/sessions")
    assert response.status_code == 200
    capability = response.json()["capabilities"]["remote_sessions"]
    assert capability == {
        "available": True,
        "version": 1,
        "view": True,
        "history": True,
        "supported_command_ops": ["prompt", "steer"],
        "media": True,
        "supported_media_kinds": ["image", "video", "file"],
        "file_sources": ["retained_document"],
        "document_upload": False,
        "audio_upload": False,
        "skill_invocation": False,
        # Remote Highlight bytes (document/file/sound) on the local URLs; per-owner
        # support is negotiated on the mesh and refused as remote_unsupported.
        "supplements": True,
        "supplement_kinds": ["document", "file", "sound"],
    }
    for body in [
        {"op": "abort"},
        {"op": "prompt", "files": [{"path": "/must-not-be-read"}]},
        {"op": "prompt", "audio": [{"data": "must-not-be-dropped"}]},
        {"op": "prompt", "expand_skills": True},
    ]:
        refused = client.post("/api/sessions/remote-chat/command", json=body)
        assert refused.status_code == 409
        assert refused.json()["code"] == "remote_unsupported" and refused.json()["remote"] is True
    assert not (tmp_path / "sessions" / "remote-chat").exists()


def test_remote_command_drops_the_prompt_wake_opt_in(tmp_path: Path, monkeypatch: Any) -> None:
    """``accept_waking`` is a phone-to-THIS-relay transport flag (agent review
    round 2, R2-3): the remote branch drops it like the local path does, so the
    owner's admission never receives a key it does not define."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    row = SessionRow(id="remote-chat", mtime=0.0, name="synthetic", owner_device="peer-b")
    monkeypatch.setattr(remote, "resolve_owner", lambda *_args: row)
    seen: list[dict[str, Any]] = []

    async def admit(_root: Path, _row: SessionRow, op: str, body: dict[str, Any]) -> dict[str, Any]:
        seen.append({"op": op, **body})
        return {"ok": True, "detail": "admitted"}

    # Patched BEFORE ``build_app``, which binds the name when it builds routes.
    monkeypatch.setattr(remote, "admit_command", admit)
    client = TestClient(build_app(MobileDaemon(port=0, password="test-only")))
    client.post("/login", data={"password": "test-only"})
    response = client.post(
        "/api/sessions/remote-chat/command",
        json={
            "op": "prompt",
            "command_id": "12345678-1234-4678-9234-567812345678",
            "text": "hello",
            "accept_waking": True,
        },
    )
    assert response.status_code == 200, response.text
    assert len(seen) == 1
    assert seen[0]["op"] == "prompt" and seen[0]["text"] == "hello"
    assert "accept_waking" not in seen[0]
