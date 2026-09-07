"""Immutable sidebar summaries; never prepare a transcript from a paint or click.

Names and runtime marks use the same sources as /resume. Attention is supplied
by the shared completion authority, not inferred from transcript timestamps.
The catalog has no acknowledgement path: listing a conversation is not reading it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from local_operator.resume import SessionRow

logger = logging.getLogger(__name__)

DEFAULT_SIDEBAR_VISIBLE = False
DEFAULT_SIDEBAR_POSITION = "left"
SidebarPosition = Literal["left", "right"]


@dataclass(frozen=True)
class SidebarSettings:
    visible: bool = DEFAULT_SIDEBAR_VISIBLE
    position: SidebarPosition = "left"

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> SidebarSettings:
        section = values.get("tui")
        section = section if isinstance(section, Mapping) else {}
        visible = section.get("sidebar_visible", DEFAULT_SIDEBAR_VISIBLE)
        position = section.get("sidebar_position", DEFAULT_SIDEBAR_POSITION)
        return cls(
            visible=visible if isinstance(visible, bool) else DEFAULT_SIDEBAR_VISIBLE,
            position="right" if position == "right" else "left",
        )


@dataclass(frozen=True)
class CatalogEntry:
    row: SessionRow
    unseen: bool = False
    completion_kind: str = ""

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def rank(self) -> tuple[int, float, str]:
        # Gates are independent of completed turns; acknowledging a completion
        # must never demote a question still waiting for a person.
        tier = 0 if self.row.pending else 1 if self.unseen else 2 if self.row.live_state else 3
        return tier, -self.row.mtime, self.id

    @property
    def status(self) -> str:
        if self.row.pending:
            return "Approval needed" if self.row.pending == "approval" else "Answer needed"
        if self.unseen:
            return {"error": "Unseen error", "interrupted": "Unseen interruption"}.get(
                self.completion_kind, "Unseen completion"
            )
        if self.row.live_state == "wedged":
            return "Not responding"
        if self.row.live_state == "busy":
            return "Working"
        # Follows ``row_state_mark``'s precedence exactly, so the tooltip can
        # never name a different state from the glyph beside it: an armed wake
        # outranks mere presence, a dormant one does not. The glyph is a single
        # character and the description is where a user finds out what it meant,
        # so the two disagreeing is worse than either being terse.
        if self.row.wakes and not self.row.wakes_dormant:
            count = self.row.wakes
            return f"Scheduled ({count} wake{'s' if count != 1 else ''})"
        return {
            "idle": "Ready",
            "attached": "Open",
        }.get(self.row.live_state, "Recent")


def rank_entries(entries: Sequence[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    """Stable identities survive refreshes, including deterministic recency ties."""
    return tuple(sorted(entries, key=lambda entry: entry.rank))


def session_directory_name(session_id: str) -> bool:
    """Discovery metadata cannot redirect a catalog read outside sessions/."""
    return (
        isinstance(session_id, str)
        and bool(session_id)
        and session_id not in {".", ".."}
        and not any(
            character in "/\\\\" or ord(character) < 32 or ord(character) == 127
            for character in session_id
        )
    )


def decorate_rows(
    directory: Path, rows: list[SessionRow], *, include_live: bool = False
) -> list[SessionRow]:
    """Fill in each row's runtime state, and float the ones needing a person.

    Two reads for the whole list: the discovery records say which sessions
    are running, working, attached or wedged, and the wake index says which
    have reminders armed. Best-effort — a picker that cannot read either
    one still lists every session exactly as it did before, because the
    fields are defaulted and the markers simply do not appear.
    """
    from local_operator.session.runtime import registry
    from local_operator.tui.widgets.session_picker import sort_needs_you_first

    try:
        scanned = registry.scan(directory)
    except Exception:  # noqa: BLE001 — markers are an enhancement, never a gate
        logger.debug("picker could not scan session records", exc_info=True)
        scanned = []
    try:
        from local_operator.wakes.store import read_index

        wake_index = read_index(directory)
    except Exception:  # noqa: BLE001
        logger.debug("picker could not read the wake index", exc_info=True)
        wake_index = {}

    live: dict[str, tuple[Any, str]] = {}
    for record, state in scanned:
        session_id = getattr(record, "session_id", "")
        if session_id:
            live[session_id] = (record, state)

    if include_live:
        from local_operator.resume import is_user_session

        known = {row.id for row in rows}
        rows = list(rows)
        for session_id, (record, _state) in live.items():
            if not session_directory_name(session_id):
                continue
            session_dir = directory / "sessions" / session_id
            if session_id not in known and session_dir.is_dir() and is_user_session(session_dir):
                rows.append(
                    SessionRow(
                        session_id,
                        float(getattr(record, "started_at", 0.0) or 0.0),
                        str(getattr(record, "conversation_name", "") or "Untitled conversation"),
                    )
                )
    updated: list[SessionRow] = []
    for row in rows:
        record_state = live.get(row.id)
        live_state = ""
        pending: str | None = None
        if record_state is not None:
            record, state = record_state
            if state == "wedged":
                live_state = "wedged"
            elif getattr(record, "busy", False):
                live_state = "busy"
            elif not getattr(record, "detached", False):
                live_state = "attached"
            else:
                live_state = "idle"
            pending = getattr(record, "pending", None) or None
        entry = wake_index.get(row.id) or {}
        schedules = entry.get("schedules") or () if isinstance(entry, dict) else ()
        updated.append(
            row._replace(
                live_state=live_state,
                pending=pending,
                wakes=len(schedules),
                wakes_dormant=bool(isinstance(entry, dict) and entry.get("stopped_at")),
            )
        )
    return sort_needs_you_first(updated)


#: Rows the poll materialises. The sidebar paints a fixed window (~38 rows at a
#: usual height) and pages within what it holds, so the untruncated answer
#: `/resume` wants is waste here: on a 665-directory store the poll built 56
#: rows every 2 s and spent 92-99% of itself doing it. Headroom well past the
#: viewport keeps paging and ranking honest without materialising the tail.
CATALOG_SCAN_LIMIT = 200

#: `session_id -> ((activity_mtime, transcript_size), SessionRow)`. A row's
#: name and fork mark change only when its transcript does, and the scan
#: already stats that file to rank the session, so the key is free. Only the
#: DURABLE fields are cached: `live_state`, `pending`, `wakes` and `unseen` are
#: layered on afterwards by `decorate_rows`/attention on every poll, because
#: caching a live fact would freeze the list.
_ROW_CACHE: dict[str, tuple[tuple[float, int], SessionRow]] = {}


def _row_stat_key(session_dir: Path) -> tuple[float, int] | None:
    """``(activity_mtime, size)`` for the transcript, or ``None`` if unreadable.

    Deliberately the same file :func:`session.retention.session_activity`
    ranks by, so a row whose key is unchanged is a row whose transcript has not
    been appended to — which is exactly the condition under which its name and
    fork mark cannot have changed. Size is carried alongside mtime because a
    coarse filesystem timestamp can hide an append inside the same second.
    """
    from local_operator.session.retention import TRANSCRIPT_FILENAME

    try:
        stat = (session_dir / TRANSCRIPT_FILENAME).stat()
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


def cached_session_rows(directory: Path, limit: int = CATALOG_SCAN_LIMIT) -> list[SessionRow]:
    """:func:`recent_session_rows` for the poll, memoized on transcript stat.

    The ``O(directories)`` scan underneath is NOT what this avoids — it still
    runs, and bounding it is what ``limit`` does. What this avoids is the
    per-row work above the scan: the bounded head read that builds the name,
    and the fork-title probe, on rows whose transcript has not been appended to
    since the last poll two seconds ago.

    Deliberately reimplements ``recent_session_rows``'s loop instead of calling
    it, because the saving is inside that loop; the scan it calls is the shared
    one, so ranking and visibility stay identical to ``/resume``. Rows absent
    from the current answer are dropped, keeping the cache bounded by the live
    store rather than by every session ever listed.
    """
    from local_operator.resume import (
        ORIGIN_FORK,
        _recent_sessions_with_origin,
        session_name,
        wears_inherited_title,
    )

    rows: list[SessionRow] = []
    fresh: dict[str, tuple[tuple[float, int], SessionRow]] = {}
    for session_id, mtime, origin in _recent_sessions_with_origin(directory, limit):
        session_dir = directory / "sessions" / session_id
        key = _row_stat_key(session_dir)
        cached = _ROW_CACHE.get(session_id)
        if key is not None and cached is not None and cached[0] == key:
            # Same transcript bytes as last poll: the name and the fork mark
            # cannot have changed, so neither read is repeated. `mtime` is
            # taken fresh from the scan regardless — it also tracks the inbox
            # spool, which ranks a row without touching the transcript.
            row = cached[1]._replace(mtime=mtime)
        else:
            row = SessionRow(
                session_id,
                mtime,
                session_name(session_dir),
                forked=origin == ORIGIN_FORK and wears_inherited_title(session_dir),
            )
        rows.append(row)
        if key is not None:
            fresh[session_id] = (key, row)
    _ROW_CACHE.clear()
    _ROW_CACHE.update(fresh)
    return rows


def load_catalog(directory: Path) -> list[CatalogEntry]:
    """One off-loop summary snapshot; never acknowledge or read full histories."""
    from local_operator.session.attention import AttentionStore, conversation_identity

    rows = decorate_rows(directory, cached_session_rows(directory), include_live=True)
    identities = {row.id: conversation_identity(directory / "sessions" / row.id) for row in rows}
    attention = AttentionStore(directory / "attention.db").state_many(identities.values())
    return list(
        rank_entries(
            [
                CatalogEntry(
                    row,
                    bool(attention[identities[row.id]]["unseen"]),
                    str(attention[identities[row.id]]["kind"] or ""),
                )
                for row in rows
            ]
        )
    )
