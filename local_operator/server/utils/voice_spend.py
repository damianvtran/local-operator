"""Route-level voice spend: the §4.2 fallback's emission seam.

WHAT THIS MODULE IS FOR. The speech and transcription routes run in the
DAEMON, while a session's journal is owned by its runtime process. Handing a
record over live would need a new attach-protocol operation (the design note's
§4.2 spike: no daemon-side journal append exists, and the transcript's writer
is an in-process lock with no cross-process counterpart), so until that
operation lands these routes emit the way the design's fallback says:

1. the record goes to the ANALYTICS mirror — best-effort, not a guarantee:
   the recorder drops silently when its queue is full or it is closing
   (design review round 1 / review MINOR 5). It is queryable by session,
   provider, model and time for the analytics retention window;
2. when the session id names a LOCAL session directory, a marker line is
   appended to that directory's ``channel_pending.jsonl`` sidecar, which the
   session's own fold reads when it publishes ``spend_channels`` and reports
   as ``pending: {voice: pending_owner_append, ...}`` — the session's total on
   every surface excludes the record, and the published object says why. The
   marker is the copy the owning session is guaranteed to see; it stands even
   when the mirror dropped its copy, because the charge is real and its
   exclusion from the total is the fact the note tells.

One module rather than a copy per route, because a second entry point that
diverged on the marker rule would make voice money visible on one route and
silent on the other; the routes pass what they know and this file keeps the
rules.

NEVER RAISES and never blocks meaningfully: a bookkeeping failure must not
fail a paid request whose audio or transcript the caller has already been
served, and every step here is best-effort by contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from local_operator.analytics import enqueue_channel_record
from local_operator.clients.radient_cost import RadientCost, channel_record_from_cost
from local_operator.config import ConfigManager
from local_operator.session.channel_spend import (
    new_record_id,
    now_ms,
    write_pending_record,
)

logger = logging.getLogger("local_operator.server.utils.voice_spend")


def session_spend_dir(config_manager: ConfigManager, session_id: str) -> Optional[Path]:
    """The LOCAL session directory a pending marker may be written into, or ``None``.

    Only a session that exists as a local directory gets a marker: the id also
    travels for conversations that live on another device (the desktop's mesh),
    and creating a spurious directory for a foreign id would be worse than the
    marker the local fold loses. ``session_directory_name`` is the shared rule
    for ids that may name a directory at all — discovery metadata must never
    redirect a write outside ``sessions/``.
    """
    if not session_id:
        return None
    from local_operator.session.catalog import session_directory_name

    if not session_directory_name(session_id):
        return None
    directory = Path(str(config_manager.config_dir)) / "sessions" / session_id
    return directory if directory.is_dir() else None


def emit_voice_spend(
    *,
    config_manager: ConfigManager,
    channel: str,
    provider: str,
    units: Optional[float],
    unit: str,
    cost: Optional[RadientCost],
    session_id: str,
    request_id: str,
) -> None:
    """Record ONE successful voice call: analytics always, pending marker if owned.

    ``channel`` is ``tts`` or ``stt``; ``cost`` is the parsed hub object (or
    ``None`` when the leg reported none — an older hub, or a BYO vendor);
    ``units``/``unit`` are the CLIENT's fallback count, upgraded by the
    server-counted values the cost object carries. ``session_id`` may be empty
    (a caller that named none): the record is still made — the money is real —
    it simply has no session to be pending against, and none is guessed.

    The record id is derived from the usage-record id when the server gave one
    (so a retry of the same admitted call dedups), else from ``request_id``.
    """
    try:
        correlation = (cost.usage_record_id if cost is not None else "") or request_id
        record = channel_record_from_cost(
            cost,
            channel=channel,
            record_id=new_record_id(channel, correlation),
            session_id=session_id,
            units=units,
            unit=unit,
            provider=provider,
            request_id=request_id,
            ts_ms=now_ms(),
        )
        enqueue_channel_record(record)
        if session_id:
            directory = session_spend_dir(config_manager, session_id)
            if directory is not None:
                write_pending_record(directory, record)
    except Exception:  # noqa: BLE001 — bookkeeping never fails a paid call
        logger.debug("voice spend emission failed", exc_info=True)
