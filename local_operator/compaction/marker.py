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
from typing import Any, Sequence

from local_operator.harness.types import (
    Content,
    CustomMessage,
    ImageContent,
    Message,
    TextContent,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COMPACTION_MARKER_TYPE",
    "build_compaction_marker",
    "marker_details",
    "marker_exists",
    "render_compaction_marker",
    "replayed_user_message",
    "split_leading_marker",
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


def render_compaction_marker(marker: CustomMessage, entry_id: str | None = None) -> Message:
    """Render one compaction marker into an LLM-visible message. ``entry_id``
    (the marker's transcript entry id) rides onto the rendered message.

    Every rendered marker carries its details on ``provider_payload`` (see
    :func:`_stamp_marker_details`), so a host that handles the rendered form
    can recover the marker without re-deriving its text shape.

    Snapcompact archives replay via ``history_blocks`` (lazy import; any
    failure degrades to the plain-text summary so a malformed archive never
    breaks the turn).
    """
    summary = marker.details.get("summary", "")
    preserve = marker.details.get("preserve_data") or {}
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
                    f"{salvage}"
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
