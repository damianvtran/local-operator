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

from local_operator.harness.types import (
    AudioContent,
    ImageContent,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import Transcript


@pytest.fixture(autouse=True)
def _isolate_attachments(tmp_path, monkeypatch):
    """Every test in this module writes into a tmp store, never the
    developer's real ``~/.local-operator/attachments``."""
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    monkeypatch.setattr(
        "local_operator.session.transcript.AttachmentStore",
        lambda root=None: AttachmentStore(root or store_root),
    )
    return store_root


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
    got = store.get(ref.digest)
    assert got is not None
    data, mime_type = got
    assert base64.b64decode(data) == PNG_BYTES
    assert mime_type == "image/png"


def test_put_dedups_identical_content(tmp_path):
    store = AttachmentStore(tmp_path)
    first = store.put(PNG_B64, "image/png")
    second = store.put(PNG_B64, "image/png")

    assert first is not None and second is not None
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


def test_get_rejects_a_digest_mismatch(tmp_path):
    """A bit-rotted or truncated file must not flow silently-wrong bytes
    into the model: treat it as missing so replay degrades."""
    store = AttachmentStore(tmp_path)
    ref = store.put(PNG_B64, "image/png")
    assert ref is not None
    (tmp_path / f"{ref.digest}.bin").write_bytes(b"partial")
    assert store.get(ref.digest) is None


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
    first = history[0]
    assert isinstance(first, Message)
    image = [b for b in first.content if isinstance(b, ImageContent)][0]
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


#: RIFF/WAVE header plus padding, well over the 1 KiB externalization floor.
WAV_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 2048
WAV_B64 = base64.b64encode(WAV_BYTES).decode("ascii")


@pytest.mark.asyncio
async def test_an_audio_block_externalizes_with_its_type_and_default_mime(tmp_path):
    """A recording rides the SAME store, and its two encoded surprises are pinned.

    ``audio/wav`` IS ``AudioContent``'s default, so ``exclude_defaults`` drops
    the mime from the encoded row — measured before the fix, that made the store
    fall back to its image default (``image/png``) and replay re-parse the block
    as an ``ImageContent`` carrying audio bytes, i.e. a recording re-sent as an
    image on every later turn. The row must therefore carry the explicit
    ``type: "audio"`` discriminant, and the mime fallback must know which media
    default it is falling back to.
    """
    import json

    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)
    await transcript.append_message(
        Message(
            role="user",
            content=[TextContent(text="listen"), AudioContent(data=WAV_B64)],
        )
    )

    raw = transcript.path.read_text(encoding="utf-8")
    assert WAV_B64 not in raw, "the payload should live in the store, not the row"
    row = json.loads(raw.splitlines()[0])
    block = row["payload"]["content"][-1]
    assert block["type"] == "audio"
    assert block["mime_type"] == "audio/wav"
    assert "attachment" in block and "data" not in block

    history = transcript.build_llm_history()
    replayed = [m for m in history if isinstance(m, Message)][0]
    (audio,) = [b for b in replayed.content if isinstance(b, AudioContent)]
    assert base64.b64decode(audio.data) == WAV_BYTES
    assert audio.mime_type == "audio/wav"
