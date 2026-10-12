"""Output artifacts reach the phone the same lazily-fetched way user images do.

The projection emits REFERENCES (index + mime); the daemon resolves them to
bytes on demand. Both walks share ONE index — inline images and artifact
images of kind ``image`` in content order — so these tests pin the pair
together: refs on the live and history folds, and byte resolution through the
daemon's helper against a real on-disk transcript.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from local_operator.harness.types import (
    AgentMessage,
    AttachmentContent,
    Content,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    _image_refs,
    _image_refs_for_content,
    _video_refs,
    _video_refs_for_content,
)
from local_operator.mobile.types import SessionProjection
from local_operator.session.attachments import cache_media


@pytest.fixture(autouse=True)
def _isolate_attachments(tmp_path, monkeypatch):
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    return store_root


#: A 73-byte 1x1 PNG — real bytes for real resolution.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)


def _make_fold() -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1))


def test_refs_count_inline_and_artifact_images_on_one_index():
    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    video = AttachmentContent(kind="video", content_type="video/mp4", attachment="a" * 32)
    content: list[Content] = [
        TextContent(text="generated a still"),
        artifact,
        video,
        ImageContent(data="AAAA", mime_type="image/jpeg"),
    ]

    refs = _image_refs_for_content(content)

    # The video artifact is SKIPPED, so it must not shift the second index.
    assert refs == [
        {"index": 0, "mime_type": "image/png"},
        {"index": 1, "mime_type": "image/jpeg"},
    ]
    assert _image_refs(Message(role="tool", content=content)) == refs


def test_history_fold_attaches_artifact_refs_to_the_settled_row():
    fold = _make_fold()
    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    call = ToolCall(id="c1", name="generate_image", arguments={"prompt": "a fox"})
    history: list[AgentMessage] = [
        Message.assistant("", tool_calls=[call]),
        Message.tool_result(
            ToolResult(
                tool_call_id="c1",
                tool_name="generate_image",
                content=[TextContent(text="Made one image"), artifact],
            )
        ),
    ]

    fold.fold_history(history)

    tool_rows = [e for e in fold.projection.transcript if e.kind == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].images == [{"index": 0, "mime_type": "image/png"}]


def test_live_fold_attaches_artifact_refs_at_settle():
    fold = _make_fold()
    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None

    fold.fold_event(ToolExecutionStartEvent(tool_call_id="t1", tool_name="generate_image"))
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="t1",
            tool_name="generate_image",
            result=ToolResult(
                tool_call_id="t1",
                tool_name="generate_image",
                content=[TextContent(text="Made one image"), artifact],
            ),
        )
    )

    row = fold.projection.transcript[-1]
    assert row.kind == "tool"
    assert row.tool_state == "done"
    assert row.images == [{"index": 0, "mime_type": "image/png"}]


def test_daemon_image_endpoint_resolves_artifact_bytes(tmp_path, monkeypatch):
    """The helper the phone's image URL reads through: entry id + index in,
    artifact bytes out — decoded from the store, not from the row."""
    from local_operator.mobile.daemon import _image_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    session_id = "sess-artifact"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    message = Message.tool_result(
        ToolResult(
            tool_call_id="c1",
            tool_name="generate_image",
            content=[TextContent(text="Made one image"), artifact],
        )
    )
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(message))

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    found = _image_bytes(record, message.id, 0)
    assert found is not None
    data, mime = found
    assert data == PNG_1X1
    assert mime == "image/png"

    # Out-of-range index and unknown entry both miss cleanly.
    assert _image_bytes(record, message.id, 1) is None
    assert _image_bytes(record, "nope", 0) is None


def test_refs_row_ids_resolve_through_the_endpoint(tmp_path, monkeypatch):
    """QA round 1, Q-1: refs ride the TOOL ROW's id, never the result message id.

    The projection emits refs on rows whose ids are
    ``{assistant_message_id}:{call_id}`` (history fold) or ``tc-{call_id}``
    (live fold). The earlier test passed the RESULT message id directly, so
    the endpoint's exact-match lookup looked fine while every real phone fetch
    404'd ('no such image'). This pins the composition the client actually
    walks: row id in, artifact bytes out, for BOTH row shapes.
    """
    from local_operator.mobile.daemon import _image_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    session_id = "sess-rowid"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    assistant = Message.assistant(
        "",
        tool_calls=[ToolCall(id="call_rowid", name="generate_image", arguments={"prompt": "x"})],
    )
    result = Message.tool_result(
        ToolResult(
            tool_call_id="call_rowid",
            tool_name="generate_image",
            content=[TextContent(text="Made one image"), artifact],
        )
    )
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(assistant))
    asyncio.run(transcript.append_message(result))

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    # The two client-visible row-id shapes.
    for row_id in (f"{assistant.id}:call_rowid", "tc-call_rowid"):
        found = _image_bytes(record, row_id, 0)
        assert found is not None, f"no bytes resolved for row id {row_id!r}"
        assert found[0] == PNG_1X1 and found[1] == "image/png"
    # The result's own message id still resolves (older clients/tests), and a
    # junk id is still an ordinary miss.
    assert _image_bytes(record, result.id, 0) is not None
    assert _image_bytes(record, "tc-nope", 0) is None


def test_daemon_image_endpoint_index_matches_refs_walk(tmp_path, monkeypatch):
    """ONE index for both sides: an inline image placed BEFORE the artifact
    makes the artifact index 1 on the refs side, and the endpoint must agree
    — this is the regression pair the module contract promises."""
    from local_operator.mobile.daemon import _image_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    session_id = "sess-mixed"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    message = Message.tool_result(
        ToolResult(
            tool_call_id="c1",
            tool_name="generate_image",
            content=[
                TextContent(text="caption"),
                ImageContent(data=base64.b64encode(b"INLINE").decode(), mime_type="image/jpeg"),
                artifact,
            ],
        )
    )
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(message))

    # The refs walk over the SAME content the row carries: both indexes are
    # 0 (inline) and 1 (artifact) — the exact pair the endpoint resolves.
    refs = _image_refs(message)
    assert [r["index"] for r in refs] == [0, 1]

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    inline = _image_bytes(record, message.id, 0)
    artifact_bytes = _image_bytes(record, message.id, 1)
    assert inline is not None and inline[0] == b"INLINE"
    assert artifact_bytes is not None and artifact_bytes[0] == PNG_1X1


#: A tiny but real MP4 header block. The endpoint does not sniff bytes, so
#: unit media only needs to round-trip through the store; the capture RIG
#: embeds a real decodable clip instead, because the browser is the decoder
#: there.
MP4_HEADER = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 48


def test_video_refs_count_kind_video_artifacts_on_their_own_index():
    """The video family's walk: only artifacts of kind ``video`` count, in
    their OWN index space — neither family shifts the other."""
    video = cache_media(MP4_HEADER, "video/mp4", kind="video", duration_s=1.0)
    second = cache_media(MP4_HEADER + b"2", "video/mp4", kind="video")
    image = cache_media(PNG_1X1, "image/png")
    assert video is not None and second is not None and image is not None
    content: list[Content] = [
        TextContent(text="made a clip"),
        video,
        image,
        ImageContent(data="AAAA", mime_type="image/jpeg"),
        second,
    ]

    assert _video_refs_for_content(content) == [
        {"index": 0, "mime_type": "video/mp4", "duration_s": 1.0},
        {"index": 1, "mime_type": "video/mp4"},
    ]
    # The image side is untouched by the video blocks: parallel indexes, not
    # a shared one.
    assert _image_refs_for_content(content) == [
        {"index": 0, "mime_type": "image/png"},
        {"index": 1, "mime_type": "image/jpeg"},
    ]
    assert _video_refs(Message(role="tool", content=content))


def test_live_fold_attaches_video_refs_at_settle():
    fold = _make_fold()
    artifact = cache_media(MP4_HEADER, "video/mp4", kind="video", duration_s=1.0)
    assert artifact is not None

    fold.fold_event(ToolExecutionStartEvent(tool_call_id="v1", tool_name="generate_video"))
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="v1",
            tool_name="generate_video",
            result=ToolResult(
                tool_call_id="v1",
                tool_name="generate_video",
                content=[TextContent(text="Video ready"), artifact],
            ),
        )
    )

    row = fold.projection.transcript[-1]
    assert row.tool_state == "done"
    assert row.videos == [{"index": 0, "mime_type": "video/mp4", "duration_s": 1.0}]
    assert row.images == []


def test_history_fold_attaches_video_refs_to_the_settled_row():
    fold = _make_fold()
    artifact = cache_media(MP4_HEADER, "video/mp4", kind="video")
    assert artifact is not None
    call = ToolCall(id="v9", name="generate_video", arguments={"prompt": "an otter"})
    history: list[AgentMessage] = [
        Message.assistant("", tool_calls=[call]),
        Message.tool_result(
            ToolResult(
                tool_call_id="v9",
                tool_name="generate_video",
                content=[TextContent(text="Video ready"), artifact],
            )
        ),
    ]

    fold.fold_history(history)

    tool_rows = [e for e in fold.projection.transcript if e.kind == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].videos == [{"index": 0, "mime_type": "video/mp4"}]


def test_attachment_bytes_serves_video_and_the_sidecar_mime(tmp_path, monkeypatch):
    """The video-family resolver: bytes out of the store, and when the block
    carries NO content_type of its own the store's sidecar mime is the
    answer (never a bare default). The families' indexes do not cross."""
    from local_operator.mobile.daemon import _attachment_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.attachments import AttachmentStore
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    # The sidecar mime is deliberately NOT the video fallback (``video/mp4``):
    # with the same value on both sides the assertion below would pass even
    # with the sidecar path dead, proving nothing about it (review round 1,
    # F2). A distinct mime makes the resolution chain discriminate.
    ref = AttachmentStore().put_bytes(MP4_HEADER, "video/webm")
    assert ref is not None
    block = AttachmentContent(kind="video", attachment=ref.digest)
    session_id = "sess-video"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    message = Message.tool_result(
        ToolResult(
            tool_call_id="v1",
            tool_name="generate_video",
            content=[TextContent(text="Video ready"), block],
        )
    )
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(message))

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    found = _attachment_bytes(record, message.id, 0, "video")
    assert found is not None
    data, mime = found
    assert data == MP4_HEADER
    assert mime == "video/webm"

    # Family indexes do not cross: no video at index 1, no image on this
    # message at all, and an unknown kind resolves nothing.
    assert _attachment_bytes(record, message.id, 1, "video") is None
    assert _attachment_bytes(record, message.id, 0, "image") is None
    assert _attachment_bytes(record, message.id, 0, "audio") is None


def test_attachment_route_serves_ranges_and_refuses_cleanly(tmp_path, monkeypatch):
    """The phone-Safari contract, end to end over the REAL route: a full
    200 advertises ``Accept-Ranges``, single ranges answer 206 with
    ``Content-Range``, an unsatisfiable range is a 416, and every miss
    (entry, kind, index) is a clean 404. The image route keeps its exact
    old behavior, Range header ignored."""
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    png = cache_media(PNG_1X1, "image/png")
    video = cache_media(MP4_HEADER, "video/mp4", kind="video")
    assert png is not None and video is not None
    session_id = "sess-route"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    message = Message.tool_result(
        ToolResult(
            tool_call_id="call-range",
            tool_name="generate_video",
            content=[TextContent(text="Video ready"), png, video],
        )
    )
    asyncio.run(Transcript(directory).append_message(message))

    daemon = MobileDaemon(port=0, password="pw123")
    record = SessionRecord(
        pid=7,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry

    client = TestClient(build_app(daemon), follow_redirects=False)
    url = f"/api/sessions/{session_id}/attachment"
    params = {"entry": message.id, "i": "0", "kind": "video"}

    # Auth refusal parity: both attachment families answer the same 401
    # before login.
    unauthed = client.get(url, params=params)
    assert unauthed.status_code == 401
    assert unauthed.json() == {"error": "authentication required"}
    image_unauthed = client.get(
        f"/api/sessions/{session_id}/image", params={"entry": message.id, "i": "0"}
    )
    assert image_unauthed.status_code == 401
    assert image_unauthed.json() == unauthed.json()

    assert client.post("/login", data={"password": "pw123"}).status_code == 303

    size = len(MP4_HEADER)
    full = client.get(url, params=params)
    assert full.status_code == 200
    assert full.content == MP4_HEADER
    assert full.headers["content-type"].startswith("video/mp4")
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["cache-control"] == "public, max-age=31536000, immutable"

    ranged = client.get(url, params=params, headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == MP4_HEADER[:10]
    assert ranged.headers["content-range"] == f"bytes 0-9/{size}"
    assert ranged.headers["accept-ranges"] == "bytes"

    tail = client.get(url, params=params, headers={"Range": "bytes=-4"})
    assert tail.status_code == 206
    assert tail.content == MP4_HEADER[-4:]
    assert tail.headers["content-range"] == f"bytes {size - 4}-{size - 1}/{size}"

    open_end = client.get(url, params=params, headers={"Range": "bytes=4-"})
    assert open_end.status_code == 206
    assert open_end.content == MP4_HEADER[4:]

    refused = client.get(url, params=params, headers={"Range": f"bytes={size}-"})
    assert refused.status_code == 416
    assert refused.headers["content-range"] == f"bytes */{size}"

    # Malformed and multi-range headers are IGNORED (full 200) — a Range the
    # server does not honour must not make the resource un-servable.
    multi = client.get(url, params=params, headers={"Range": "bytes=0-1,5-6"})
    assert multi.status_code == 200
    assert multi.content == MP4_HEADER

    # Every miss is a clean 404; a bad index type is a 400, not a 500.
    assert client.get(url, params={"entry": "nope", "i": "0", "kind": "video"}).status_code == 404
    assert (
        client.get(url, params={"entry": message.id, "i": "0", "kind": "audio"}).status_code == 404
    )
    assert (
        client.get(url, params={"entry": message.id, "i": "5", "kind": "video"}).status_code == 404
    )
    assert (
        client.get(url, params={"entry": message.id, "i": "x", "kind": "video"}).status_code == 400
    )

    # The IMAGE route is unchanged: range headers are ignored (full 200) and
    # its response advertises no ``Accept-Ranges`` (that addition belongs to
    # the attachment route alone), while its bytes and mime are the same as
    # before.
    image = client.get(
        f"/api/sessions/{session_id}/image",
        params={"entry": message.id, "i": "0"},
        headers={"Range": "bytes=0-9"},
    )
    assert image.status_code == 200
    assert image.content == PNG_1X1
    assert image.headers["content-type"].startswith("image/png")
    assert "accept-ranges" not in image.headers
