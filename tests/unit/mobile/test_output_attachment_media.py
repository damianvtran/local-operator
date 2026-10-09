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
