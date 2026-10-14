"""End-to-end wiring of the stored-location hints: wire yes, everywhere else no.

``session/attachment_hints.py`` defines the note; this module pins the two
seams and — more importantly — the INVISIBILITY property end to end, because
that is the requirement the whole placement exists for:

* the TURN request carries the note for a pasted image whose bytes the session
  store holds;
* the durable transcript row, the live ``_context.messages`` and the
  compaction render (which a compaction commit seats back into the live
  context) carry NONE — a note that reached any of them would round-trip into
  later renders and paint in every chat surface as the user's own words;
* a resume re-derives the identical note, so the provider-cache prefix is
  byte-stable across the live/replay boundary;
* the read-only render (asides, clearance, the compaction advisor) carries the
  same note as the turn, because the two must agree ahead of their one
  divergence point;
* a text-only model's render strips the image and earns no note, because there
  is no file part to the interaction it can use.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from local_operator.harness.replay_bound import bound_replay_payloads
from local_operator.harness.types import (
    ImageContent,
    Message,
    StreamEndEvent,
    TextContent,
)
from local_operator.session.attachment_hints import with_attachment_hints
from local_operator.session.attachments import AttachmentStore
from local_operator.session.session import IMAGE_OMITTED_TEXT_ONLY_NOTICE, Session
from local_operator.session.transcript import Transcript

from .test_session import MODEL, TEXT_ONLY_MODEL, ScriptedStream

#: Over the transcript's externalization floor, so the paste is a store file —
#: the regime the note exists for.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
PNG_DIGEST = hashlib.sha256(PNG_BYTES).hexdigest()[:32]

_NOTE_MARKER = "no filesystem search is needed"


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Both the transcript's externalization and the note's lookup resolve the
    store through ``config_dir()``; point it into the test's own tree so
    neither writes to the developer's real ``~/.local-operator``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))


def make_session(tmp_path, stream, **kwargs) -> Session:
    transcript = Transcript(tmp_path / "sess")
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable", "env"],
        **kwargs,
    )


def _notes(messages) -> list[str]:
    return [
        block.text
        for message in messages
        for block in getattr(message, "content", [])
        if isinstance(block, TextContent) and _NOTE_MARKER in block.text
    ]


@pytest.mark.asyncio
async def test_a_pasted_image_earns_a_note_on_the_wire_and_nowhere_durable(tmp_path):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    await session.prompt("make anime of this", [ImageContent(data=PNG_B64, mime_type="image/png")])

    request = stream.requests[0]
    notes = _notes(request.messages)
    assert len(notes) == 1
    note = notes[0]
    store = AttachmentStore()
    assert str(store.path_for(PNG_DIGEST)) in note
    assert f"digest {PNG_DIGEST}" in note
    assert store.path_for(PNG_DIGEST).read_bytes() == PNG_BYTES
    # The copy's id is the derived one; the memo cannot confuse it either way.
    (user_message,) = [m for m in request.messages if m.role == "user"]
    assert user_message.id.endswith("~ah")

    # Durable row: the reference IS there, the note is NOT.
    row = (tmp_path / "sess" / "transcript.jsonl").read_text(encoding="utf-8")
    assert PNG_DIGEST in row
    assert _NOTE_MARKER not in row

    # Live context: untouched.
    (context_user,) = [
        m for m in session._context.messages if isinstance(m, Message) and m.role == "user"
    ]
    assert not any(
        isinstance(block, TextContent) and _NOTE_MARKER in block.text
        for block in context_user.content
    )

    # The compaction rebuild the commit seats (and the turn-end pass persists)
    # must not carry the note either.
    assert _notes(session._render_for_compaction()) == []

    await session.dispose()


@pytest.mark.asyncio
async def test_the_read_only_render_carries_the_same_note_as_the_turn(tmp_path):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    await session.prompt("edit this", [ImageContent(data=PNG_B64, mime_type="image/png")])
    turn_notes = _notes(stream.requests[0].messages)
    assert len(turn_notes) == 1

    _blocks, aside_messages = await session._read_only_prompt([Message.user("aside question")])

    assert _notes(aside_messages) == turn_notes

    await session.dispose()


@pytest.mark.asyncio
async def test_the_read_only_render_is_the_same_composition_as_the_turn(tmp_path):
    """The aside seam spells the composition inline; pin it EQUAL to the
    turn's method. Their byte equality is the property that lets an aside read
    the turn's provider cache, and the two spellings must not drift."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session.prompt("edit this", [ImageContent(data=PNG_B64, mime_type="image/png")])

    turns = [Message.user("and now?")]
    _blocks, aside = await session._read_only_prompt(turns)
    expected = bound_replay_payloads(
        session._render_for_request([*session._wire_legal_snapshot(), *turns])
    )

    assert [m.model_dump() for m in aside] == [m.model_dump() for m in expected]

    await session.dispose()


@pytest.mark.asyncio
async def test_a_resume_re_derives_the_identical_note_bytes(tmp_path):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session.prompt("edit this", [ImageContent(data=PNG_B64, mime_type="image/png")])
    turn_notes = _notes(stream.requests[0].messages)
    await session.dispose()

    # A fresh open of the same session directory replays the row, resolves the
    # attachment from the store, and the pass rebuilds the same bytes — the
    # property the prompt-cache prefix depends on after a resume.
    replayed = with_attachment_hints(
        [m for m in Transcript(tmp_path / "sess").build_llm_history() if isinstance(m, Message)]
    )

    assert _notes(replayed) == turn_notes


@pytest.mark.asyncio
async def test_multiple_pasted_images_earn_one_note_listing_both(tmp_path):
    other_bytes = PNG_BYTES + b"other"
    other_digest = hashlib.sha256(other_bytes).hexdigest()[:32]
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    await session.prompt(
        "compare these",
        [
            ImageContent(data=PNG_B64, mime_type="image/png"),
            ImageContent(
                data=base64.b64encode(other_bytes).decode("ascii"), mime_type="image/jpeg"
            ),
        ],
    )

    (note,) = _notes(stream.requests[0].messages)
    store = AttachmentStore()
    assert note.index(str(store.path_for(PNG_DIGEST))) < note.index(
        str(store.path_for(other_digest))
    )
    assert store.path_for(other_digest).read_bytes() == other_bytes

    await session.dispose()


@pytest.mark.asyncio
async def test_a_text_only_model_strips_the_image_and_earns_no_note(tmp_path):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, model=TEXT_ONLY_MODEL)

    await session.prompt("look at this", [ImageContent(data=PNG_B64, mime_type="image/png")])

    texts = [
        block.text
        for message in stream.requests[0].messages
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert not any(_NOTE_MARKER in text for text in texts)
    assert any(IMAGE_OMITTED_TEXT_ONLY_NOTICE in text for text in texts)

    await session.dispose()


@pytest.mark.asyncio
async def test_a_confined_session_keeps_pixels_but_drops_the_path_promise(tmp_path):
    """The world-root seam: the note no longer promises a read the session's
    own confinement refuses, while the image still rides the wire and the
    aside stays byte-equal to the turn (the provider-cache property)."""
    world = tmp_path / "world"
    world.mkdir()
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, confinement_root=str(world))

    await session.prompt("edit this", [ImageContent(data=PNG_B64, mime_type="image/png")])

    request = stream.requests[0]
    notes = [
        block.text
        for message in request.messages
        for block in getattr(message, "content", [])
        if isinstance(block, TextContent) and "no workspace file" in block.text
    ]
    assert len(notes) == 1
    store = AttachmentStore()
    assert str(store.path_for(PNG_DIGEST)) not in notes[0]
    assert "read or edit" not in notes[0]
    assert f"digest {PNG_DIGEST}" in notes[0]

    # The pixels are still on the wire — only the false promise left.
    assert any(
        isinstance(block, ImageContent)
        for message in request.messages
        if message.role == "user"
        for block in message.content
    )

    # Aside == turn still: the two seams must keep rendering byte-equal.
    turns = [Message.user("and now?")]
    _blocks, aside = await session._read_only_prompt(turns)
    expected = bound_replay_payloads(
        session._render_for_request([*session._wire_legal_snapshot(), *turns])
    )
    assert [m.model_dump() for m in aside] == [m.model_dump() for m in expected]

    await session.dispose()
