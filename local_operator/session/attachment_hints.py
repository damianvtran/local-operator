"""Stored-location notes for a request's images, invisible to every chat surface.

THE FAILURE THIS REMOVES
------------------------
A user pastes a screenshot and asks for work that needs the FILE ("make an
anime-style version of this", "run OCR on this receipt"). The model receives
the image as an ``ImageContent`` block with no reference to a place on disk,
so it goes looking: on a real session (2026-10-11, session ``609490094595``)
an agent spent twenty-odd turn steps sweeping ``~/.local``, ``/var/folders``,
``Application Support`` and several session directories before it found the
paste in the content-addressed attachment store — which had been sitting at a
stable, computable path the whole time (``<config>/attachments/<digest>.bin``;
see ``session/attachments.py``). The search is the cost of a message that
says where the image is not.

WHAT THIS ADDS, AND WHERE IT RIDES
----------------------------------
:func:`with_attachment_hints` returns the same request messages with ONE extra
``TextContent`` block after each user message's image blocks, naming each
image's store file, its mime type, its byte count and its digest — the same
digest vocabulary the ``generate_image`` caption already uses ("attached to
the session (digest ...)", ``tools/image_tool.py``), so a tool that takes a
``source_attachment`` can be called without touching the filesystem at all.

It is applied at the two seams that build PROVIDER REQUESTS — the turn loop's
``convert_to_llm`` callable (``Session._render_for_request``) and
``Session._read_only_prompt`` — to COPIES of the rendered messages, and
nowhere else. That placement is the invisibility property, BY CONSTRUCTION
rather than by filtering:

* the transcript rows and the live ``_context.messages`` are never touched,
  so nothing a chat surface replays or paints can carry the note — the TUI's
  live echo builds from ``EditorSubmitted``, its replay from transcript rows,
  the desktop paints rows and frames, and the phone projects rows;
* the compaction renders (``Session._render_for_compaction``) never call
  this, so the kept window a compaction commit seats into the live context —
  and which the turn-end pass then persists — cannot bake the note into
  history. A note that reached the transcript would round-trip into every
  future render and would paint as the user's own words.

WHY A COPY, AND WHY THE ``~ah`` ID
----------------------------------
Same reasoning as the ``~rb`` copies in ``harness/replay_bound.py``, which run
at the same stage of the same pipeline: the renderer hands out the
transcript's own ``Message`` objects on the pass-through arm, so appending to
``message.content`` in place would edit the durable record; and
``compaction/tokens.py`` memoizes token estimates by ``message.id``, so a
differently-sized copy must not share its id or the second content is
answered with the first one's size.

WHAT THE NOTE NAMES, AND WHAT IT DELIBERATELY DOES NOT
------------------------------------------------------
The note names a store file the session store ALREADY holds — it is a report,
never a write. ``AttachmentStore.locate`` is the read-only sibling of ``put``
for exactly this reason: a request render (which runs per provider step, and
for token counts beside) must not populate the operator's store as a side
effect. Three exclusions follow, and they are stated rather than hidden:

* a block below the transcript's externalization floor is inline-only (the
  session never wrote a file for it, so there is nothing to name);
* a legacy inline row from a pre-store build has no file either, until it is
  rewritten;
* a frame the wire had to REPAIR or DOWNSCALE (``_rebound_history_images``,
  ``_fit_frames_to_budget``) carries bytes no store file holds, so it earns
  no note rather than a note naming different bytes than the request sends.

A block the render replaced with a notice carries no bytes and earns no note;
a store that has been pruned by hand degrades the same way — the note simply
does not appear for that image.

COST, STATED
------------
One base64-decode, one sha256 and one digest-verified store read per image per
request build, on the loop — measured at ~3.0 ms/image for 700 KB frames
(36.6 ms for twelve, best of five; the per-render repair walk already accepted
beside it, ``_rebound_history_images``, pays 20.7 ms for the same list on the
same host, so this pass is the same order as what already runs there) — and
paid only for user-role messages that actually carry image blocks. The note
itself is roughly 40-90 tokens per image and rides every later request while
the image is in context; for a screenshot-heavy session that is a small
fraction of what the image blocks themselves cost, and it is the price of the
model never having to search.
"""

from __future__ import annotations

from local_operator.compaction.marker import COMPACTION_MARKER_TYPE
from local_operator.harness.types import ImageContent, Message, TextContent
from local_operator.session.attachments import AttachmentStore

#: Suffix on a hinted copy's id. The sibling of ``harness/replay_bound.py``'s
#: ``~rb``: the token-estimate memo is keyed on ``message.id``, and these
#: copies carry more content than the originals, so the two must not share an
#: id. Deterministic, so the hinted view is byte-stable across turns.
_HINT_ID_SUFFIX = "~ah"

#: The note's opening, one line, singular and plural. A shared constant so the
#: two arities cannot drift in wording — the sentence is the whole interface
#: the model reads.
_SINGLE_OPEN = (
    "[image attachment stored on disk — read or edit the file directly; "
    "no filesystem search is needed:\n"
)
_MULTI_OPEN = (
    "[image attachments stored on disk — read or edit the files directly; "
    "no filesystem search is needed:\n"
)


def with_attachment_hints(messages: list[Message]) -> list[Message]:
    """``messages``, with a stored-location note after each message's images.

    Returns a NEW list. A message that earns no note — every non-user role,
    every message without image blocks, a compaction marker's archive frames,
    an image the store cannot locate — is passed through by IDENTITY, so the
    common turn allocates nothing and copies nothing. A message that earns one
    is replaced by a copy carrying an extra trailing ``TextContent``; the
    original message is never mutated (see the module docstring).

    Deterministic by construction: the note is a pure function of the block's
    bytes and the store's own naming rule, so two builds of the same request
    produce byte-identical text and the provider's prompt-cache prefix stays
    stable across turns.
    """
    store = AttachmentStore()
    out: list[Message] = []
    for message in messages:
        if message.role != "user":
            out.append(message)
            continue
        if message.provider_payload and COMPACTION_MARKER_TYPE in message.provider_payload:
            # A snapcompact archive's frames are a rebuilt artifact of the
            # transcript, not media a person just attached; a note about
            # "files" behind them would send the model looking for snapshots
            # nobody asked it to edit.
            out.append(message)
            continue
        lines = [
            line
            for block in message.content
            if isinstance(block, ImageContent)
            if (line := _block_line(store, block)) is not None
        ]
        if not lines:
            out.append(message)
            continue
        opening = _SINGLE_OPEN if len(lines) == 1 else _MULTI_OPEN
        note = opening + "\n".join(lines) + "]"
        out.append(_hinted_copy(message, note))
    return out


def _block_line(store: AttachmentStore, block: ImageContent) -> str | None:
    """The note's line for ``block``, or ``None`` when it earns none.

    ``None`` is a normal outcome, never an error: a payload-less block, an
    undecodable payload, and content the store does not hold all land here,
    and the caller's contract is to leave that image unmentioned rather than
    to write or to name a path that dangles.
    """
    if not block.data:
        return None
    ref = store.locate(block.data, block.mime_type)
    if ref is None:
        return None
    return (
        f"- {store.path_for(ref.digest)} "
        f"({block.mime_type}, {ref.bytes} bytes; digest {ref.digest})"
    )


def _hinted_copy(message: Message, note: str) -> Message:
    """A copy of ``message`` with ``note`` appended as its final text block.

    The id difference is load-bearing, not cosmetic — see
    :data:`_HINT_ID_SUFFIX` — and the rest of the message (role, usage,
    ``provider_payload``, tool pairing fields) rides along untouched, so the
    wire still sees the same turn with one more text block.
    """
    return message.model_copy(
        update={
            "content": [*message.content, TextContent(text=note)],
            "id": f"{message.id}{_HINT_ID_SUFFIX}",
        }
    )
