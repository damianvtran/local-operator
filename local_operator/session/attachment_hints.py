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

THE BOUNDARY (world-root sessions)
----------------------------------
An episode session runs inside a WORLD ROOT: ``SessionSpec.world_root`` is
installed as the session's ``confinement_root``, and the controller store --
the path this note names -- is BY CONSTRUCTION outside it (the SDK refuses a
layout whose store roots sit inside the world). The measured cost of ignoring
that boundary: a confined model read the note, believed
"read or edit the file directly", and spent a tool call on a path its own
confinement must refuse (``tools/confinement.py``; the refusal is correct and
the wasted call was not). The pixels were already in the message; the FALSE
promise was the only thing the note added.

So the hint is boundary-aware: when ``world_root`` is set and the store file
is NOT inside it, the note keeps every fact the model can act on (mime, byte
count, digest, and the do-not-search instruction) and drops the controller
path and the read/edit promise the session cannot keep. The inline image is
untouched. Reachability is judged the way the tools judge it -- on the
RESOLVED path, so a symlink out of the world is outside it -- and an
unresolvable path counts as outside, mirroring
``ToolConfinement.path_denial(resolvable=False)``.

Constraints, stated: ``world_root=None`` (every ordinary exec/TUI/mobile/
server session) renders BYTE-IDENTICAL output to before this parameter
existed, and that arm resolves nothing; no filesystem allowlist is widened,
nothing is copied or written and no provider- or task-specific rule enters
this module.

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

import os
from pathlib import Path

from local_operator.compaction.marker import COMPACTION_MARKER_TYPE
from local_operator.harness.types import (
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore
from local_operator.tools.confinement import ToolConfinement

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

#: Confined world, store file NOT inside it: the note keeps the facts the
#: model can act on (mime/size/digest, and the do-not-search instruction) and
#: drops the controller path plus the read/edit promise the session's own
#: tools would refuse -- see the module docstring's boundary section.
_CONFINED_SINGLE_OPEN = (
    "[image attachment — inline pixels only; it is not available as a file "
    "in this workspace (do not search for one):\n"
)
_CONFINED_MULTI_OPEN = (
    '[image attachments — entries marked "no workspace file" are inline '
    "pixels only and cannot be read or edited (do not search for them):\n"
)

#: Document-bearing messages get their own opening: the note now covers files
#: whose bytes the model may have received natively or as extracted text, so
#: "image" would be a false description of what follows. A message carrying
#: BOTH images and documents takes this opening too — one note per turn, the
#: contract's single trailing block, and the per-row lines say what each entry
#: is.
_FILES_SINGLE_OPEN = (
    "[attached file stored on disk — read or edit the file directly; "
    "no filesystem search is needed:\n"
)
_FILES_MULTI_OPEN = (
    "[attached files stored on disk — read or edit the files directly; "
    "no filesystem search is needed:\n"
)
_FILES_CONFINED_SINGLE_OPEN = (
    "[attached file — it is not available as a workspace file (do not search "
    "for one); the model already has what was delivered below:\n"
)
_FILES_CONFINED_MULTI_OPEN = (
    '[attached files — entries marked "no workspace file" cannot be read or '
    "edited (do not search for them); the model already has what was "
    "delivered below:\n"
)


def with_attachment_hints(
    messages: list[Message], *, world_root: str | Path | None = None
) -> list[Message]:
    """``messages``, with a stored-location note after each message's images.

    Returns a NEW list. A message that earns no note — every non-user role,
    every message without image blocks, a compaction marker's archive frames,
    an image the store cannot locate — is passed through by IDENTITY, so the
    common turn allocates nothing and copies nothing. A message that earns one
    is replaced by a copy carrying an extra trailing ``TextContent``; the
    original message is never mutated (see the module docstring).

    ``world_root`` is the session's installed world boundary (its
    ``confinement_root``; ``None`` for every ordinary session, whose render is
    BYTE-IDENTICAL to before this parameter existed). Set, an image whose
    store file is not inside the boundary keeps its truthful facts and the
    do-not-search instruction but loses the controller path and the read/edit
    promise the confined tools would refuse — see the module docstring's
    boundary section. The inline image is never touched, on either arm.

    Deterministic by construction: the note is a pure function of the block's
    bytes, the store's own naming rule and the session's boundary, so two
    builds of the same request produce byte-identical text and the provider's
    prompt-cache prefix stays stable across turns.
    """
    store = AttachmentStore()
    # The boundary is resolved ONCE per render, the way the tools resolve it
    # (``ToolConfinement.at``); the per-image check below is then pure.
    boundary = ToolConfinement.at(world_root) if world_root and str(world_root).strip() else None
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
        rows: list[tuple[str, bool]] = []
        has_documents = False
        for block in message.content:
            if isinstance(block, ImageContent):
                row = _block_row(store, block, boundary)
            elif isinstance(block, DocumentContent):
                has_documents = True
                row = _document_row(store, block, boundary)
            else:
                continue
            if row is not None:
                rows.append(row)
        if not rows:
            out.append(message)
            continue
        if all(full for _text, full in rows):
            if has_documents:
                opening = _FILES_SINGLE_OPEN if len(rows) == 1 else _FILES_MULTI_OPEN
            else:
                opening = _SINGLE_OPEN if len(rows) == 1 else _MULTI_OPEN
        else:
            if has_documents:
                opening = (
                    _FILES_CONFINED_SINGLE_OPEN if len(rows) == 1 else _FILES_CONFINED_MULTI_OPEN
                )
            else:
                opening = _CONFINED_SINGLE_OPEN if len(rows) == 1 else _CONFINED_MULTI_OPEN
        note = opening + "\n".join(text for text, _full in rows) + "]"
        out.append(_hinted_copy(message, note))
    return out


def _block_row(
    store: AttachmentStore, block: ImageContent, boundary: ToolConfinement | None
) -> tuple[str, bool] | None:
    """The note's line for ``block``, and whether it advertises a real path.

    ``None`` is a normal outcome, never an error: a payload-less block, an
    undecodable payload, and content the store does not hold all land here,
    and the caller's contract is to leave that image unmentioned rather than
    to write or to name a path that dangles. The boolean is ``False`` exactly
    when the boundary makes the store file unreachable, which swaps in the
    reduced line (facts kept, path and read/edit promise dropped).
    """
    if not block.data:
        return None
    ref = store.locate(block.data, block.mime_type)
    if ref is None:
        return None
    path = store.path_for(ref.digest)
    origin = _origin_clause(block.path, boundary)
    if boundary is None or _reachable(boundary, path):
        return (
            f"- {path} ({block.mime_type}, {ref.bytes} bytes; digest {ref.digest}){origin}",
            True,
        )
    return (
        f"- ({block.mime_type}, {ref.bytes} bytes; digest {ref.digest}; no workspace file)"
        f"{origin}",
        False,
    )


def _document_row(
    store: AttachmentStore, block: DocumentContent, boundary: ToolConfinement | None
) -> tuple[str, bool] | None:
    """The note's line for one document, and whether it advertises a real path.

    Facts the model can always act on come first — filename, MIME, byte count
    and the DELIVERY PATH this turn actually took (``injection``, stamped at
    admission) — because a model told "you received a PDF" can reason about
    it even when no file is reachable. The retained copy (the store blob) is
    named only when the session's boundary can reach it, exactly the image
    rule; the ORIGIN path follows the same test, so a confined session is
    never promised a controller path its tools would refuse.

    ``None`` only when the block carries no bytes at all (an empty placeholder
    the wire could not resolve): naming a file nobody can open is the false
    promise this module exists to avoid.
    """
    if not block.data and not block.digest:
        return None
    digest = block.digest
    if not digest and block.data:
        ref = store.locate(block.data, block.mime_type)
        digest = ref.digest if ref is not None else None
    retained = ""
    full = True
    if digest:
        path = store.path_for(digest)
        if boundary is None or _reachable(boundary, path):
            retained = f", retained at {path}"
        else:
            retained = f", retained copy (digest {digest}; no workspace file)"
            full = False
    delivery = _delivery_clause(block.injection)
    origin = _origin_clause(block.path, boundary)
    size = block.bytes or len(block.data)
    return (
        f'- "{block.filename}" ({block.mime_type}, {size} bytes{retained}' f"{origin}; {delivery})",
        full,
    )


def _delivery_clause(injection: str | None) -> str:
    """How the turn's bytes reached the model, in words it can act on.

    Historical (``injection`` is stamped at admission), and phrased as a fact
    about the message rather than a promise about this request: a later
    ``/model`` switch may deliver the same block differently, which is why the
    serializer re-decides and this clause does not claim otherwise.
    """
    if injection == "native":
        return "sent as a native document"
    if injection == "extracted":
        return "delivered as extracted text"
    if injection == "reference":
        return "delivered only as a reference, not its contents"
    return "attached"


def _origin_clause(origin: str | None, boundary: ToolConfinement | None) -> str:
    """``; original at <path>`` when it is present AND reachable.

    Confinement (contract amendment 8): a path outside the session's world is
    withheld from the MODEL-VISIBLE note. History may still show it to the
    user; this function is only ever called on the request render.
    """
    if not origin:
        return ""
    if boundary is not None and not _reachable(boundary, Path(origin)):
        return ""
    return f"; original at {origin}"


def _reachable(boundary: ToolConfinement, path: Path) -> bool:
    """Whether the session's confined tools could reach ``path``.

    Judged on the RESOLVED path, exactly as ``ToolConfinement`` judges tool
    calls (seatbelt compares vnodes; the file tools compare resolved paths —
    ``tools/confinement.py``), so a symlink that leaves the world is outside
    it here too. An unresolvable path counts as outside: "cannot be shown
    inside" is a denial, mirroring ``ToolConfinement.path_denial``'s
    ``resolvable=False`` arm.
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        # RuntimeError is the Python 3.12 spelling of "symlink loop":
        # ``Path.resolve`` raises it there instead of OSError. A loop is
        # precisely the case the boundary must read as outside. This is a
        # judgement about reachability, not error suppression: the contract
        # above says an unresolvable path counts as outside, mirroring
        # ``ToolConfinement.path_denial(resolvable=False)``.
        return False
    # Python 3.13 stopped raising: a non-strict ``resolve`` now hands a loop
    # back UNRESOLVED (the loop link itself, or the loop link plus the rest of
    # the path). That value sits inside the world, so without this check the
    # same loop that 3.12 denies is advertised as readable -- and the project
    # supports both (``requires-python = ">=3.12"``, CI runs 3.13 on ``main``).
    # ``strict=True`` would also deny a not-yet-existing path, which is
    # reachable, so detect the signature instead: a fully resolved path has no
    # symlink among its existing components.
    if _stops_at_symlink(resolved):
        return False
    return boundary.contains(resolved)


def _stops_at_symlink(resolved: Path) -> bool:
    """Whether ``resolved`` still names a symlink at its deepest existing part.

    ``Path.resolve`` replaces every symlink it can follow, so a symlink that
    survives in its result is one it gave up on (a loop on Python 3.13+, which
    returns it unresolved instead of raising). Components that do not exist yet
    are skipped: a missing leaf is ordinary, a surviving link is not.
    """
    probe = resolved
    while not os.path.lexists(probe):
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_symlink()


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
