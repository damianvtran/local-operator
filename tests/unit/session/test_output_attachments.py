"""Output attachments: binaries a turn produced, cached and referenced.

The output half of the attachment contract. A tool registers bytes once
through ``cache_media`` and the transcript carries a pointer-shaped
``AttachmentContent`` block — so these tests are about the properties that
make the pointer trustworthy: metadata survives the round trip, the durable
row parses back as an attachment (never as empty text), the reference
SURVIVES replay and folds (the legacy image resolution pass must not pop
it), and every failure degrades instead of raising.
"""

from __future__ import annotations

import base64

import pytest

from local_operator.harness.types import (
    AttachmentContent,
    ImageContent,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore, cache_media
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


#: A 73-byte 1x1 PNG — magic, IHDR with declared 1x1 dimensions, IDAT, IEND.
#: Real bytes so the header sniffer (not a mock) supplies the dimensions.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)


def test_cache_media_returns_a_complete_block():
    block = cache_media(
        PNG_1X1,
        "image/png",
        name="generated-01.png",
        source_url="https://provider.example/img/01.png",
    )

    assert block is not None
    assert block.kind == "image"
    assert block.content_type == "image/png"
    assert len(block.attachment or "") == 32
    assert block.size_bytes == len(PNG_1X1)
    # Dimensions come from the header, for free — nothing may decode here.
    assert (block.width, block.height) == (1, 1)
    assert block.name == "generated-01.png"
    assert block.source_url == "https://provider.example/img/01.png"
    # The bytes are IN the store under that digest, already.
    got = AttachmentStore().get_bytes(block.attachment or "")
    assert got is not None and got[0] == PNG_1X1


def test_cache_media_dedups_by_content(_isolate_attachments):
    first = cache_media(PNG_1X1, "image/png")
    second = cache_media(PNG_1X1, "image/png")

    assert first is not None and second is not None
    assert first.attachment == second.attachment
    assert len(list(_isolate_attachments.glob("*.bin"))) == 1


def test_cache_media_refusals_are_none_never_raise():
    assert cache_media(b"", "image/png") is None
    # Non-media major types have no home in the contract's kind vocabulary.
    assert cache_media(b"xx", "application/pdf") is None
    assert cache_media(b"xx", "text/plain") is None
    # An explicit kind outside the vocabulary is refused the same way.
    assert cache_media(b"xx", "image/png", kind="file") is None


def test_cache_media_carries_a_none_when_the_store_refuses(monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert cache_media(PNG_1X1, "image/png") is None


def test_get_bytes_mismatch_is_none():
    store = AttachmentStore()
    block = cache_media(PNG_1X1, "image/png")
    assert block is not None and block.attachment
    (store.root / f"{block.attachment}.bin").write_bytes(b"poisoned")
    assert store.get_bytes(block.attachment) is None


def _artifact_message() -> Message:
    block = cache_media(
        PNG_1X1,
        "image/png",
        name="flux-01.png",
        source_url="https://provider.example/img/01.png",
    )
    assert block is not None
    return Message(
        role="tool",
        content=[TextContent(text="Generated flux-01.png (1x1, 73 B)"), block],
    )


@pytest.mark.asyncio
async def test_transcript_round_trip_keeps_the_artifact_reference(tmp_path):
    """The durable row carries the pointer — digest and metadata — and NEVER
    base64; replay returns the same block, with the digest intact.

    The digest surviving is load-bearing twice over: it is what surfaces
    fetch by, and the legacy image resolution pass must not have popped it
    on the way out (which would also lose it at the next file fold)."""
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)

    await transcript.append_message(_artifact_message())

    raw = transcript.path.read_text(encoding="utf-8")
    assert base64.b64encode(PNG_1X1).decode("ascii") not in raw, "bytes must not be inline"
    assert '"kind":"image"' in raw
    assert '"content_type":"image/png"' in raw
    assert '"name":"flux-01.png"' in raw

    history = transcript.build_llm_history()
    replayed = [m for m in history if isinstance(m, Message)][0]
    artifacts = [b for b in replayed.content if isinstance(b, AttachmentContent)]
    assert len(artifacts) == 1
    block = artifacts[0]
    assert block.attachment and len(block.attachment) == 32
    assert block.kind == "image"
    assert block.size_bytes == len(PNG_1X1)
    # Not inlined, not popped: `data` is not a field, and the reference is
    # still exactly the one written.
    dumped = block.model_dump(exclude_defaults=True)
    assert "data" not in dumped
    assert dumped["attachment"] == json_attachment(raw)


def json_attachment(raw: str) -> str:
    """The digest the row itself carries, read back from the JSONL."""
    import json

    for line in raw.splitlines():
        payload = json.loads(line).get("payload", {})
        for block in payload.get("content", []):
            if isinstance(block, dict) and block.get("kind") == "image":
                return str(block["attachment"])
    raise AssertionError("no artifact block on the row")


@pytest.mark.asyncio
async def test_a_missing_store_degrades_without_dropping_the_reference(tmp_path):
    """A hand-pruned store must not cost the row its metadata: replay keeps
    the block (surfaces can still show 'unavailable' and the source URL);
    only the bytes are gone."""
    session_dir = tmp_path / "session"
    transcript = Transcript(session_dir)
    await transcript.append_message(_artifact_message())

    from local_operator.session.attachments import attachments_dir

    for path in attachments_dir().glob("*"):
        path.unlink()

    history = transcript.build_llm_history()
    replayed = [m for m in history if isinstance(m, Message)][0]
    artifacts = [b for b in replayed.content if isinstance(b, AttachmentContent)]
    assert len(artifacts) == 1
    assert artifacts[0].attachment  # the pointer still names the digest
    assert AttachmentStore().get_bytes(artifacts[0].attachment) is None


def test_relay_resolution_leaves_artifact_references_untouched():
    """The follower's frame resolver inlines LEGACY image refs (popping the
    digest as it goes) — an artifact's digest is durable payload and must
    survive the walk. The guard keys on the legacy mime_type shape, which an
    artifact block does not have; this pins that separation."""
    from local_operator.session.attached import resolve_frame_attachments

    frame = {
        "type": "tool_execution_end",
        "result": {
            "content": [
                {
                    "kind": "image",
                    "content_type": "image/png",
                    "attachment": "a" * 32,
                    "source_url": "https://provider.example/img/01.png",
                },
                {"attachment": "b" * 32, "mime_type": "image/png"},
            ]
        },
    }

    resolved = resolve_frame_attachments(frame, AttachmentStore())

    artifact = resolved["result"]["content"][0]
    assert artifact["attachment"] == "a" * 32, "the digest is payload, not an encoding"
    assert "data" not in artifact, "artifact bytes are never inlined into a frame"
    assert artifact["kind"] == "image"


def test_live_and_durable_shapes_parse_to_one_model():
    """The live frame carries `type`; the durable row cannot (the encoder
    excludes defaults). Both must land on AttachmentContent — the durable
    shape is the one that used to read as EMPTY TEXT without the coercion."""
    live = Message.model_validate(
        {
            "role": "tool",
            "content": [
                {
                    "type": "attachment",
                    "kind": "video",
                    "content_type": "video/mp4",
                    "attachment": "a" * 32,
                }
            ],
        }
    )
    durable = Message.model_validate(
        {
            "role": "tool",
            "content": [
                {
                    "kind": "video",
                    "content_type": "video/mp4",
                    "attachment": "a" * 32,
                    "duration_s": 3.5,
                }
            ],
        }
    )

    for message in (live, durable):
        block = message.content[0]
        assert isinstance(block, AttachmentContent)
        assert block.kind == "video"
    # The durable row's own metadata lands on the block; the live frame,
    # which never carried a duration, does not invent one. (Narrowed per the
    # union rule — `content[0]` is `Content` until isinstance says otherwise,
    # and pyright's whole-tree pass reads this file too.)
    durable_block = durable.content[0]
    live_block = live.content[0]
    assert isinstance(durable_block, AttachmentContent)
    assert isinstance(live_block, AttachmentContent)
    assert durable_block.duration_s == 3.5
    assert live_block.duration_s is None

    # A legacy image reference is NOT an artifact: the coercion must leave
    # the old shapes alone.
    legacy = Message.model_validate(
        {"role": "tool", "content": [{"attachment": "b" * 32, "mime_type": "image/png"}]}
    )
    assert isinstance(legacy.content[0], ImageContent)
