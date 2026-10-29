"""The compaction-summary marker and how it renders into LLM context.

A compaction pass leaves ONE transcript entry behind — a ``CustomMessage`` of
``custom_type="compaction_summary"`` whose ``details`` carry the summary text
and, for snapcompact, the archive under ``preserve_data["snapcompact"]``.
The session persists that entry and replays it as a user message on every
later request; the evaluation runner rebuilds its history from the same
marker. Both hosts therefore need the same two operations, and they live here
rather than in ``session.py`` so a host that must not import the session
(``run_compaction_pass``'s callers) renders the marker identically.

Nothing here imports ``session``, ``model``, ``providers`` or ``config``: the
compaction package is consumed by hosts that are forbidden those imports.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from local_operator.harness.types import (
    Content,
    CustomMessage,
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ATTACHMENTS_MANIFEST_KEY",
    "CAPTURE_ELISION_FRONTIER_KEY",
    "COMPACTION_MARKER_TYPE",
    "collect_attachment_manifest",
    "with_attachment_manifest",
    "build_compaction_marker",
    "capture_elision_frontier",
    "capture_elision_identities",
    "marker_details",
    "marker_exists",
    "render_compaction_marker",
    "replayed_user_message",
    "split_leading_marker",
    "with_capture_elision_frontier",
]

#: ``CustomMessage.custom_type`` of the entry a compaction pass leaves behind.
COMPACTION_MARKER_TYPE = "compaction_summary"

#: ``custom_type`` of the row a compaction that did NOT run leaves behind.
#:
#: A pass that runs narrates itself through the ``compaction_start``/
#: ``compaction_end`` events, so nothing needs recording for the happy path.
#: A REFUSAL emits no events at all, which is what made it invisible on the
#: routed path: the runtime answered "compacting context…" optimistically and
#: then discarded the outcome, so a user was told a pass had started and
#: nothing ever contradicted it (round 5, U17). Recorded rather than sent as a
#: transient notice because a detached session may have no terminal attached
#: when the refusal lands.
COMPACTION_REFUSED_TYPE = "compaction_refused"


def build_compaction_marker(
    summary: str, preserve_data: dict[str, Any] | None = None
) -> CustomMessage:
    """The marker a pass commits: ``{"summary": ..., "preserve_data": ...}``.

    ``preserve_data`` is only written when present so a context-full marker's
    payload stays exactly what it was before snapcompact existed; the
    transcript persists this shape and older readers key on it.
    """
    details: dict[str, Any] = {"summary": summary}
    if preserve_data is not None:
        details["preserve_data"] = preserve_data
    return CustomMessage(custom_type=COMPACTION_MARKER_TYPE, attribution="system", details=details)


#: ``preserve_data`` key for the ATTACHMENTS manifest a pass commits:
#: ``[{"digest", "filename", "mime_type", "bytes", "path"}, ...]``, newest
#: last, deduped, capped. WHY IT EXISTS: a compaction summary is a model's
#: prose, and "usable after a compaction" (contract §4) must not depend on the
#: summariser happening to copy a file's path. The manifest is data the
#: harness re-renders verbatim on every later request, so a user can ask
#: "where is that PDF" three compactions later and get an answer.
#:
#: ADDITIVE and OPTIONAL, the same discipline ``capture_elisions`` follows: a
#: marker with nothing attached omits the key entirely, so a text-only
#: conversation's marker payload stays byte-identical to before this existed.
ATTACHMENTS_MANIFEST_KEY = "attachments"

#: Bounds. Twenty files and ~2 kB: a longer list is not a context aid, it is
#: a second transcript. The caps are enforced on WRITE, so every reader sees
#: the same bounded payload.
MAX_MANIFEST_ENTRIES = 20
MAX_MANIFEST_CHARS = 2_000


def collect_attachment_manifest(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """The attachment entries carried by ``messages``, newest last.

    Two sources, merged in one pass so a pass cannot lose the previous one's
    record: live ``DocumentContent`` blocks (deduped by digest, else by
    filename+MIME+size) and the manifest already committed on an earlier
    marker (``marker_details``). Order is the conversation's, so truncation
    keeps the LATEST attachments — the ones a follow-up question is most
    likely about.
    """
    entries: list[dict[str, Any]] = []

    def add(entry: dict[str, Any]) -> None:
        key = (
            entry.get("digest")
            or f"{entry.get('filename')}|{entry.get('mime_type')}|{entry.get('bytes')}"
        )
        for index, existing in enumerate(entries):
            existing_key = (
                existing.get("digest")
                or f"{existing.get('filename')}|{existing.get('mime_type')}|{existing.get('bytes')}"
            )
            if existing_key == key:
                # Newest provenance wins: a re-send of the same bytes from a
                # new path should say where the latest send came from.
                entries[index] = entry
                return
        entries.append(entry)

    for message in messages:
        details = marker_details(message)
        if details is not None:
            preserve = details.get("preserve_data")
            if isinstance(preserve, Mapping):
                previous = preserve.get(ATTACHMENTS_MANIFEST_KEY)
                if isinstance(previous, list):
                    for entry in previous:
                        if isinstance(entry, Mapping):
                            add(dict(entry))
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, DocumentContent):
                continue
            add(
                {
                    "digest": block.digest,
                    "filename": block.filename,
                    "mime_type": block.mime_type,
                    "bytes": block.bytes,
                    "path": block.path,
                }
            )
    entries = entries[-MAX_MANIFEST_ENTRIES:]
    # Character cap: drop the OLDEST until the JSON payload fits. Sizers are
    # approximate on purpose; the renderer is what the model sees, and it is
    # derived from the same bounded list.
    import json as _json

    while entries and len(_json.dumps(entries)) > MAX_MANIFEST_CHARS:
        entries.pop(0)
    return entries


def with_attachment_manifest(
    preserve_data: Mapping[str, Any] | None,
    messages: Sequence[Any],
    *,
    world_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """``preserve_data`` with this pass's manifest merged in.

    Copied, never mutated: the caller's payload may be shared with a strategy
    archive. Nothing attached -> the payload is returned as-is (and stays
    ``None`` when it was ``None``), which is what keeps a text-only marker
    byte-identical.

    ``world_root`` filters a confined session's paths AT WRITE, so the stored
    payload itself never carries a path the model must not be shown. That
    matters because the marker is later rendered by the generic
    ``harness/render.py`` converter, which has no session and therefore no
    boundary to consult (contract amendment 8).
    """
    entries = collect_attachment_manifest(messages)
    if not entries:
        return dict(preserve_data) if preserve_data is not None else None
    if world_root and str(world_root).strip():
        from local_operator.tools.confinement import ToolConfinement

        boundary = ToolConfinement.at(world_root)
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            item = dict(entry)
            path = item.get("path")
            if isinstance(path, str) and path and not _manifest_path_reachable(boundary, path):
                item["path"] = None
            filtered.append(item)
        entries = filtered
    merged: dict[str, Any] = dict(preserve_data) if preserve_data else {}
    merged[ATTACHMENTS_MANIFEST_KEY] = entries
    return merged


def render_attachment_manifest(
    preserve_data: Mapping[str, Any] | None, *, world_root: str | Path | None = None
) -> str:
    """The manifest as the model reads it, or ``""`` when there is none.

    Confinement (contract amendment 8): a path outside the session's accessible
    world is withheld here exactly as it is in the per-request hint. History
    may still show it to the user; this is the model-visible render. Judged on
    the RESOLVED path, the same rule ``attachment_hints._reachable`` applies,
    and a boundary we cannot resolve counts as outside.
    """
    if not isinstance(preserve_data, Mapping):
        return ""
    entries = preserve_data.get(ATTACHMENTS_MANIFEST_KEY)
    if not isinstance(entries, list) or not entries:
        return ""
    boundary = None
    if world_root and str(world_root).strip():
        from local_operator.tools.confinement import ToolConfinement

        boundary = ToolConfinement.at(world_root)
    lines = ["Files attached earlier in this conversation:"]
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        filename = str(entry.get("filename") or "an attached file")
        mime = str(entry.get("mime_type") or "unknown type")
        size = entry.get("bytes")
        size_text = f", {size} bytes" if isinstance(size, int) and size > 0 else ""
        path = entry.get("path")
        where = ""
        if isinstance(path, str) and path and _manifest_path_reachable(boundary, path):
            where = f" — {path}"
        lines.append(f"- {filename} ({mime}{size_text}){where}")
    return "\n".join(lines)


def _manifest_path_reachable(boundary: Any, path: str) -> bool:
    if boundary is None:
        return True
    from local_operator.session.attachment_hints import _reachable

    if not _reachable(boundary, Path(path)):
        return False
    return True


#: ``preserve_data`` key for the CAPTURE-ELISION record an epoch commits:
#: ``{"applied": <int>, "identities": [[scope, stream, revision, ordinal], ...]}``
#: -- which superseded-capture blocks the pass elided under the byte band
#: (one occurrence-qualified address per elided block, in elision order) plus
#: the count for diagnostics. ADDITIVE and optional: a record without the key
#: (every record written before this existed) replays with NO capture
#: elisions, which is exactly the old behaviour, and a writer with nothing to
#: record omits the key rather than spelling a zero. REPLAY USES THE
#: ADDRESSES, never a count re-derived from the resumed segment: a later turn
#: can retro-supersede a capture the pass protected, so the pass-time
#: candidate order cannot be rebuilt from the resumed whole (reviewer round
#: 1, C1), and the per-identity ORDINAL names which occurrence was elided
#: when a producer repeated a claim and the never-empty guard skipped the
#: first (reviewer round 1, C2).
CAPTURE_ELISION_FRONTIER_KEY = "capture_elisions"


def capture_elision_frontier(details: Mapping[str, Any] | None) -> int | None:
    """How many superseded captures a marker's epoch elided, or ``None``.

    ``None`` covers both "this record predates capture elision" and "the
    value is malformed"; replay then applies NOTHING, which is the
    conservative reading -- an unknown frontier must never elide more than
    the epoch did, and older readers ignore the key entirely either way.
    """
    if not isinstance(details, Mapping):
        return None
    preserve = details.get("preserve_data")
    if not isinstance(preserve, Mapping):
        return None
    frontier = preserve.get(CAPTURE_ELISION_FRONTIER_KEY)
    if not isinstance(frontier, Mapping):
        return None
    applied = frontier.get("applied")
    # ``bool`` is an ``int``: a spelling of ``True`` here is a malformed
    # record, not a count of one.
    if isinstance(applied, bool) or not isinstance(applied, int) or applied < 0:
        return None
    return applied


def capture_elision_identities(
    details: Mapping[str, Any] | None,
) -> tuple[tuple[str, str, str, int], ...]:
    """The occurrence-qualified addresses of the captures a marker's epoch elided.

    Each entry is ``(producer_scope, stream_id, revision_id, ordinal)``, one
    per elided block in elision order; ``ordinal`` counts how many
    same-identity blocks precede it in canonical arrival order (see the key
    comment for why identity alone cannot name a block). ``()`` covers both
    "this record predates the identities key" and "the value is malformed"
    -- replay then applies NOTHING, the conservative reading: an unknown
    record must never elide anything, and a bare count is not safely
    re-derivable (reviewer round 1, C1/C2). Wrongly-shaped entries are
    skipped individually, so one bad entry cannot discard the rest of a
    valid record.
    """
    if not isinstance(details, Mapping):
        return ()
    preserve = details.get("preserve_data")
    if not isinstance(preserve, Mapping):
        return ()
    record = preserve.get(CAPTURE_ELISION_FRONTIER_KEY)
    if not isinstance(record, Mapping):
        return ()
    identities = record.get("identities")
    if not isinstance(identities, (list, tuple)):
        return ()
    out: list[tuple[str, str, str, int]] = []
    for entry in identities:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            continue
        scope, stream, revision, ordinal = entry
        if not (
            isinstance(scope, str)
            and isinstance(stream, str)
            and isinstance(revision, str)
            and isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and ordinal >= 0
        ):
            continue
        out.append((scope, stream, revision, ordinal))
    return tuple(out)


def with_capture_elision_frontier(
    preserve_data: Mapping[str, Any] | None, identities: Sequence[tuple[str, str, str, int]]
) -> dict[str, Any] | None:
    """``preserve_data`` with the capture-elision record merged in.

    ``identities`` are the stable harness-stamped addresses of the blocks the
    pass ELIDED -- ``(producer_scope, stream_id, revision_id, ordinal)``, one
    entry per elided block in elision order, where ``ordinal`` counts how
    many same-identity blocks preceded it (the never-empty guard can elide a
    duplicate claim's second occurrence while leaving the first, so the
    address needs the ordinal to name a block -- reviewer round 1, C2). The
    persisted form stores them beside their count (``applied``), because
    replay must re-elide exactly the recorded blocks rather than re-derive a
    candidate prefix over the resumed segment (see the key comment above).

    A pass that elided nothing commits NOTHING new: the key is omitted rather
    than written as zero, so a text-only or legacy pass's marker payload
    stays byte-identical to what it was before this existed. Inputs are
    copied, never mutated -- the caller's payload may be shared with a
    strategy archive.
    """
    if not identities:
        return dict(preserve_data) if preserve_data is not None else None
    merged: dict[str, Any] = dict(preserve_data) if preserve_data else {}
    merged[CAPTURE_ELISION_FRONTIER_KEY] = {
        "applied": len(identities),
        "identities": [list(identity) for identity in identities],
    }
    return merged


def replayed_user_message(content: list[Content], entry_id: str | None) -> Message:
    """Build a replayed user message, preserving its transcript entry id.

    A message rendered from a persisted entry MUST keep that entry's id:
    ``first_kept_entry_id`` references it, so minting a fresh uuid here would
    make replay unable to find the cut point. A message with no originating
    entry keeps the model's default id.
    """
    message = Message(role="user", content=content)
    if entry_id:
        message.id = entry_id
    return message


def _stamp_marker_details(message: Message, marker: CustomMessage) -> Message:
    """Attach ``marker``'s details to the message it rendered into.

    The stamp is how a later reader tells a rendered marker apart from an
    ordinary user turn once both are user-role ``Message``s: the cut walker
    lifts it back (``run_compaction_pass``) and the summarization input folds
    it through ``previous_summary`` rather than re-exposing it as conversation
    (:func:`split_leading_marker`). It rides ``provider_payload``, which the
    wire builders never ship, so the model and every provider see identical
    content with or without it.
    """
    message.provider_payload = {COMPACTION_MARKER_TYPE: dict(marker.details)}
    return message


def render_compaction_marker(
    marker: CustomMessage,
    entry_id: str | None = None,
    *,
    world_root: str | Path | None = None,
) -> Message:
    """Render one compaction marker into an LLM-visible message. ``entry_id``
    (the marker's transcript entry id) rides onto the rendered message.

    Every rendered marker carries its details on ``provider_payload`` (see
    :func:`_stamp_marker_details`), so a host that handles the rendered form
    can recover the marker without re-deriving its text shape.

    Snapcompact archives replay via ``history_blocks`` (lazy import; any
    failure degrades to the plain-text summary so a malformed archive never
    breaks the turn).

    The attachments manifest is appended AFTER the summary/archive on both
    branches and omitted entirely when empty, so a conversation with no
    files renders byte-identically. ``world_root`` is the session's
    confinement boundary, used only to withhold an unreachable path from the
    model-visible render (contract amendment 8).
    """
    summary = marker.details.get("summary", "")
    preserve = marker.details.get("preserve_data") or {}
    manifest = render_attachment_manifest(preserve, world_root=world_root)
    archive_payload = preserve.get("snapcompact")
    if archive_payload:
        try:
            from local_operator.compaction import snapcompact

            archive = snapcompact.Archive.model_validate(archive_payload)
            content: list[Content] = []
            for block in snapcompact.history_blocks(archive):
                if block["kind"] == "text":
                    content.append(TextContent(text=block["text"]))
                elif block["kind"] == "images":
                    for frame_b64 in block["frames"]:
                        content.append(ImageContent(data=frame_b64, mime_type="image/png"))
            if manifest:
                content.append(TextContent(text=manifest))
            if content:
                return _stamp_marker_details(replayed_user_message(content, entry_id), marker)
        except Exception:
            logger.warning("snapcompact replay failed; falling back to text summary", exc_info=True)
    # A snapcompact summary is reading instructions for the frames, not a
    # digest of the history — falling back to it ALONE would replay a caption
    # describing images that are not there while the real content vanished.
    # The archive's text edges are plain strings in the same payload and
    # survive whatever made the frame list unrevivable, so salvage them: they
    # are the newest/oldest slices of the actual transcript, which is strictly
    # more useful than any caption.
    salvage = ""
    if isinstance(archive_payload, dict):
        head = archive_payload.get("text_head")
        tail = archive_payload.get("text_tail")
        edges = [edge for edge in (head, tail) if isinstance(edge, str) and edge.strip()]
        if edges:
            joined = "\n[...]\n".join(edges)
            # The summary above may describe pixel-font frames; none are in
            # this message, and a caption describing absent images is a claim
            # the model would waste attention reconciling. Say so explicitly.
            salvage = (
                "\n[note: the archive's image frames could not be replayed here; "
                "the plain-text edges below are what survives]"
                f"\n<archived-transcript-edges>\n{joined}\n</archived-transcript-edges>"
            )
    return _stamp_marker_details(
        replayed_user_message(
            [
                TextContent(
                    text="<previous-context-summary>\n"
                    f"{summary}\n"
                    "</previous-context-summary>"
                    f"{salvage}" + (f"\n\n{manifest}" if manifest else "")
                )
            ],
            entry_id,
        ),
        marker,
    )


def marker_details(message: Any) -> dict[str, Any] | None:
    """The details a rendered marker carries, else ``None``.

    The read side of :func:`_stamp_marker_details`, shared by every consumer
    that must tell a rendered marker apart from an ordinary user turn: the
    pass's cut walker and summarizer (``run_compaction_pass``) and the
    session's own summarize path.
    """
    payload = getattr(message, "provider_payload", None)
    if not isinstance(payload, dict):
        return None
    details = payload.get(COMPACTION_MARKER_TYPE)
    return details if isinstance(details, dict) else None


def marker_exists(messages: Sequence[Any]) -> bool:
    """Whether any message in ``messages`` carries a rendered compaction
    marker (the stamp :func:`render_compaction_marker` puts on every rendered
    marker).

    The frame shed walks up the kept tail until it reaches one, which needs a
    cheap way to find it without re-deriving the tag format.
    """
    for message in messages:
        payload = getattr(message, "provider_payload", None)
        if isinstance(payload, dict) and payload.get(COMPACTION_MARKER_TYPE):
            return True
    return False


def split_leading_marker(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Separate a LEADING rendered marker from a summarization input.

    Returns ``(previous_summary, remaining)``. Every pass after the first
    plans over ``[marker, *kept]``, so the head of the summarized span is a
    rendered marker — and the prior summary must reach the summarizer through
    the fold slot (``previous_summary`` → ``<previous-summary>``) rather than
    re-exposed inside ``<conversation>`` as an ordinary user turn, which is
    what made a chained pass re-derive the summary from a conversation that
    already contained it.

    ``previous_summary`` is the same text the snapcompact chaining folds: the
    archive's accumulated ``text`` when the marker carries one (the bounded
    history it re-renders from), else the marker's own ``summary`` — so a
    marker from either strategy contributes its content exactly once, through
    the fold. The marker message is REMOVED from ``remaining``: lift XOR keep,
    never both.

    ``(None, list(messages))`` when the head is not a rendered marker — a
    first pass, or an input whose head was already handled.
    """
    if not messages:
        return None, []
    details = marker_details(messages[0])
    if details is None:
        return None, list(messages)
    preserve = details.get("preserve_data")
    snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
    previous: str | None = None
    if isinstance(snap, dict) and snap.get("text"):
        previous = str(snap["text"])
    else:
        summary = details.get("summary")
        previous = str(summary) if summary else None
    return previous, list(messages[1:])
