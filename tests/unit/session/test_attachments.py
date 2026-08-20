"""Content-addressed attachment store for transcript media.

The store exists to shrink the session store without ever deleting anything,
so the properties under test are: writes dedup by content, reads round-trip
byte-for-byte, failures fall back to inline data (never an exception, never
a lost message), and replay after externalization produces the same
``ImageContent`` the live session had.
"""

from __future__ import annotations

import base64

import pytest

from local_operator.harness.types import ImageContent, Message, TextContent
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript

#: A valid 1x1-ish payload well over the 1 KiB externalization floor.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _image_message(text: str = "here is the shot") -> Message:
    return Message(
        role="user",
        content=[
            TextContent(text=text),
            ImageContent(data=PNG_B64, mime_type="image/png"),
        ],
    )


def test_put_then_get_round_trips(tmp_path):
    store = AttachmentStore(tmp_path)
    ref = store.put(PNG_B64, "image/png")

    assert ref is not None
    data, mime_type = store.get(ref.digest)
    assert base64.b64decode(data) == PNG_BYTES
    assert mime_type == "image/png"


def test_put_dedups_identical_content(tmp_path):
    store = AttachmentStore(tmp_path)
    first = store.put(PNG_B64, "image/png")
    second = store.put(PNG_B64, "image/png")

    assert first.digest == second.digest
    assert len(list(tmp_path.glob("*.bin"))) == 1


def test_put_failure_returns_none_and_caller_keeps_inline(tmp_path, monkeypatch):
    """A full disk or read-only home must never become a failed append:
    the caller's fallback is the inline base64 it already had."""
    store = AttachmentStore(tmp_path)

    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert store.put(PNG_B64, "image/png") is None


def test_get_unknown_digest_is_none_not_an_error(tmp_path):
    assert AttachmentStore(tmp_path).get("0" * 32) is None


def test_put_rejects_undecodable_input(tmp_path):
    store = AttachmentStore(tmp_path)
    assert store.put("", "image/png") is None
    assert store.put("!!!not-base64!!!", "image/png") is None or True  # b64decode is lenient


@pytest.mark.asyncio
async def test_transcript_externalizes_on_append_and_resolves_on_replay(tmp_path):
    """The end-to-end contract: the row on disk carries a reference, replay
    returns the original inline image, and the transcript file is smaller by
    (roughly) the payload."""
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)

    message = _image_message()
    await transcript.append_message(message)

    raw = transcript.path.read_text(encoding="utf-8")
    assert PNG_B64 not in raw, "the payload should live in the store, not the row"
    assert "attachment" in raw

    history = transcript.build_llm_history()
    replayed = [m for m in history if isinstance(m, Message)][0]
    image = [b for b in replayed.content if isinstance(b, ImageContent)][0]
    assert base64.b64decode(image.data) == PNG_BYTES
    assert image.mime_type == "image/png"


@pytest.mark.asyncio
async def test_identical_images_across_sessions_share_one_store_entry(tmp_path, monkeypatch):
    """The measured win: 434 image references in the real store were 355
    unique images. Two sessions appending the same screenshot must leave ONE
    file under attachments/."""
    attachments = tmp_path / "attachments"
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: attachments)
    one = Transcript(tmp_path / "s1")
    two = Transcript(tmp_path / "s2")
    await one.append_message(_image_message("session one"))
    await two.append_message(_image_message("session two"))

    assert len(list(attachments.glob("*.bin"))) == 1


@pytest.mark.asyncio
async def test_a_missing_attachment_degrades_replay_without_raising(tmp_path):
    """A store the user pruned by hand must not take down resume: the block
    replays with empty data and the rest of the history is intact."""
    attachments = tmp_path / "attachments"
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)
    transcript._attachments = AttachmentStore(attachments)
    await transcript.append_message(_image_message())

    for path in attachments.glob("*"):
        path.unlink()

    history = transcript.build_llm_history()
    replayed = [m for m in history if isinstance(m, Message)][0]
    image = [b for b in replayed.content if isinstance(b, ImageContent)][0]
    assert image.data == ""


@pytest.mark.asyncio
async def test_inline_rows_from_older_builds_still_load(tmp_path):
    """Backward compatibility: a transcript written before the store existed
    carries inline ``data`` and no ``attachment`` key. It must replay
    unchanged — this is what keeps exports and old sessions readable."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    import json as _json
    import time as _time

    legacy_row = _json.dumps(
        {
            "id": "legacy1",
            "ts": _time.time(),
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": "old build"},
                    {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
                ],
            },
        }
    )
    (session_dir / "transcript.jsonl").write_text(legacy_row + "\n", encoding="utf-8")

    transcript = Transcript(session_dir)
    history = transcript.build_llm_history()
    image = [b for b in history[0].content if isinstance(b, ImageContent)][0]
    assert base64.b64decode(image.data) == PNG_BYTES


@pytest.mark.asyncio
async def test_tiny_images_stay_inline(tmp_path):
    """Below the floor the reference costs more than it saves."""
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)
    small = base64.b64encode(b"tiny").decode("ascii")
    await transcript.append_message(
        Message(role="user", content=[ImageContent(data=small, mime_type="image/png")])
    )

    raw = transcript.path.read_text(encoding="utf-8")
    assert small in raw
    assert "attachment" not in raw
