"""Stored-location notes on request messages, and the invisibility property.

The note exists so the model can act on a pasted image's FILE — read it, copy
it, or pass its digest to a tool — without searching the filesystem for it
(``session/attachment_hints.py`` carries the measured failure). Four
properties are load-bearing, and each is pinned below:

* the note carries the facts an operation needs: the store path, the digest,
  the mime type and the byte count;
* it rides a COPY and leaves the input message byte-identical (the ``~ah`` id
  keeps the token-estimate memo honest — see ``harness/replay_bound.py``'s
  ``~rb`` for the failure that rule exists for);
* it never names a file the store does not already hold, and never writes
  one — a request render runs per provider step, so populating the operator's
  store from it would be churn as a side effect;
* archive frames (compaction markers) earn no note: they are a rebuilt
  artifact of the transcript, not media somebody attached.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from local_operator.compaction.marker import COMPACTION_MARKER_TYPE
from local_operator.harness.types import ImageContent, Message, TextContent
from local_operator.session.attachment_hints import with_attachment_hints
from local_operator.session.attachments import AttachmentStore

#: A payload well over the transcript's externalization floor, so tests speak
#: the store's real regime rather than the inline one.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
PNG_DIGEST = hashlib.sha256(PNG_BYTES).hexdigest()[:32]


@pytest.fixture(autouse=True)
def _isolate_attachments(tmp_path, monkeypatch):
    """Every test in this module writes into a tmp store, never the
    developer's real ``~/.local-operator/attachments``."""
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    return store_root


def _image_message(*, text: str = "make anime of this", data: str = PNG_B64) -> Message:
    return Message(
        role="user",
        content=[TextContent(text=text), ImageContent(data=data, mime_type="image/png")],
    )


def test_note_names_the_store_file_with_mime_size_and_digest():
    store = AttachmentStore()
    ref = store.put(PNG_B64, "image/png")
    assert ref is not None

    (hinted,) = with_attachment_hints([_image_message()])

    note = hinted.content[-1]
    assert isinstance(note, TextContent)
    assert note.text == (
        "[image attachment stored on disk — read or edit the file directly; "
        "no filesystem search is needed:\n"
        f"- {store.path_for(PNG_DIGEST)} (image/png, {len(PNG_BYTES)} bytes; digest {PNG_DIGEST})]"
    )
    # The path in the note is the file the bytes actually live in.
    assert store.path_for(PNG_DIGEST).read_bytes() == PNG_BYTES


def test_multiple_images_are_listed_in_message_order():
    store = AttachmentStore()
    first = base64.b64encode(PNG_BYTES).decode("ascii")
    second = base64.b64encode(PNG_BYTES + b"second").decode("ascii")
    assert store.put(first, "image/png") is not None
    assert store.put(second, "image/png") is not None

    message = Message(
        role="user",
        content=[
            TextContent(text="two shots"),
            ImageContent(data=first, mime_type="image/png"),
            ImageContent(data=second, mime_type="image/jpeg"),
        ],
    )
    (hinted,) = with_attachment_hints([message])

    note = hinted.content[-1]
    assert isinstance(note, TextContent)
    assert note.text.startswith(
        "[image attachments stored on disk — read or edit the files directly; "
        "no filesystem search is needed:\n"
    )
    lines = note.text.splitlines()[1:]
    assert len(lines) == 2
    assert lines[0].startswith(f"- {store.path_for(PNG_DIGEST)} (image/png,")
    assert lines[1].startswith(
        f"- {store.path_for(hashlib.sha256(PNG_BYTES + b'second').hexdigest()[:32])} "
        "(image/jpeg,"
    )


def test_a_message_without_locatable_images_passes_through_by_identity():
    store = AttachmentStore()
    plain = Message(role="user", content=[TextContent(text="no pictures here")])
    other_role = Message(role="assistant", content=[TextContent(text="hi")])
    unstored = _image_message()  # never put() into the store
    payload_less = Message(
        role="user",
        content=[TextContent(text="empty block"), ImageContent(data="", mime_type="image/png")],
    )
    undecodable = Message(
        role="user",
        content=[
            TextContent(text="junk"),
            ImageContent(data="!!not base64!!", mime_type="image/png"),
        ],
    )

    out = with_attachment_hints([plain, other_role, unstored, payload_less, undecodable])

    assert out[0] is plain
    assert out[1] is other_role
    assert out[2] is unstored
    assert out[3] is payload_less
    assert out[4] is undecodable
    assert list(store.root.glob("*.bin")) == []


def test_the_pass_never_writes_to_the_store():
    """Reporting a location must not be a write: run the pass over content the
    store does not hold and prove no file appears for it."""
    store = AttachmentStore()
    fresh = base64.b64encode(b"\x89PNG\r\n\x1a\nfresh-image" + b"\x00" * 64).decode("ascii")

    with_attachment_hints([_image_message(data=fresh)])

    assert not list(store.root.glob("*.bin"))


def test_compaction_marker_archive_frames_earn_no_note():
    store = AttachmentStore()
    assert store.put(PNG_B64, "image/png") is not None
    marker = Message(
        role="user",
        content=[TextContent(text="summary"), ImageContent(data=PNG_B64, mime_type="image/png")],
        provider_payload={COMPACTION_MARKER_TYPE: {"summary": "summary"}},
    )

    (out,) = with_attachment_hints([marker])

    assert out is marker


def test_the_copy_is_deterministic_and_the_input_is_untouched():
    store = AttachmentStore()
    assert store.put(PNG_B64, "image/png") is not None
    original = _image_message()
    before = original.model_dump()

    first = with_attachment_hints([original])[0]
    second = with_attachment_hints([original])[0]

    assert first.model_dump() == second.model_dump()
    assert original.model_dump() == before
    assert first is not original


def test_the_copy_carries_a_derived_id_for_the_token_memo():
    store = AttachmentStore()
    assert store.put(PNG_B64, "image/png") is not None
    original = _image_message()

    (hinted,) = with_attachment_hints([original])

    # Distinct id: the token-estimate memo is keyed on it and must never
    # answer the hinted size with the unhinted one (or vice versa).
    assert hinted.id != original.id
    assert hinted.id == f"{original.id}~ah"


def test_a_missing_note_for_a_stripped_image_is_the_honest_degrade():
    """The wire-repair paths hand the pass repaired bytes the store does not
    hold (``_rebound_history_images``) — no note, rather than a note naming
    different bytes than the request sends."""
    store = AttachmentStore()
    assert store.put(PNG_B64, "image/png") is not None
    repaired_elsewhere = base64.b64encode(PNG_BYTES[:-1]).decode("ascii")

    message = Message(
        role="user",
        content=[
            TextContent(text="repaired"),
            ImageContent(data=repaired_elsewhere, mime_type="image/png"),
        ],
    )

    (out,) = with_attachment_hints([message])
    assert out is message
