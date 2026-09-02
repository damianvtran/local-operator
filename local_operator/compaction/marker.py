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
from typing import Any

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
    "render_compaction_marker",
    "replayed_user_message",
]

#: ``CustomMessage.custom_type`` of the entry a compaction pass leaves behind.
COMPACTION_MARKER_TYPE = "compaction_summary"


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


def render_compaction_marker(marker: CustomMessage, entry_id: str | None = None) -> Message:
    """Render one compaction marker into an LLM-visible message. ``entry_id``
    (the marker's transcript entry id) rides onto the rendered message.

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
                return replayed_user_message(content, entry_id)
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
    return replayed_user_message(
        [
            TextContent(
                text="<previous-context-summary>\n"
                f"{summary}\n"
                "</previous-context-summary>"
                f"{salvage}"
            )
        ],
        entry_id,
    )
