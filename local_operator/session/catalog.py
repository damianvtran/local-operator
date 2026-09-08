"""Immutable sidebar summaries; never prepare a transcript from a paint or click.

Names and runtime marks use the same sources as /resume. Attention is supplied
by the shared completion authority, not inferred from transcript timestamps.
The catalog has no acknowledgement path: listing a conversation is not reading it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.resume import SessionRow
from local_operator.session.creation import session_category, session_created_at

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogEntry:
    row: SessionRow
    unseen: bool = False
    completion_kind: str = ""
    #: The token and anchor of this row's latest completion, straight from the
    #: attention store. Carried because `unseen` alone is a LEVEL — true on
    #: every poll until the session is read — and an observer that wants to
    #: announce a background session's completion needs the identity of the
    #: specific event to arbitrate on (`AttentionStore.claim_delivery`).
    #: Defaulted so every existing construction site, and the sidebar tests
    #: that build entries positionally, keep working unchanged.
    completion_token: str = ""
    anchor_id: str = ""

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def rank(self) -> tuple[int, int, float, str]:
        """``(tier, wake_rank, -birth, id)`` — a wake orders the PREVIOUS group only.

        ``wake_rank`` is scoped to cold rows (``not self.active``): inside
        Previous an armed wake leads, then a dormant one, then everything else.
        Every ACTIVE row gets the same constant, so the key cannot reorder
        anything the user is currently working with.

        **Why scoped rather than uniform.** The first cut applied the key in
        every tier, on the reasoning that it "only breaks ties within one
        category" and so was free. That premise is empirically false, and three
        reviewers independently reproduced the consequences: rows inside a tier
        are NOT interchangeable, because a tier mixes states that
        :func:`~local_operator.tui.widgets.session_picker.row_state_mark`
        deliberately ranks against each other.

        * tier 5 mixes ``attached`` with ``idle``, so an idle session owning a
          timer sorted above the user's own ATTACHED current session — order
          contradicting the glyph ladder, where ``○`` means *a terminal is
          watching this session* and answers "where am I?".
        * tier 4 mixes ``busy`` with ``wedged``, so a WEDGED (broken) session
          needing a person sank below a merely-busy neighbour with a wake.
        * tier 0 mixes the approval and answer gates, so an armed pending row
          displaced an OLDER pending one — and note this third case defeats a
          guard written as "presence outranks a wake", because a pending row can
          carry an empty ``live_state``.

        Scoping to Previous removes all three at the root instead of enumerating
        the states to dodge. That distinction is the real lesson: an enumerated
        guard has to be updated every time the ``live_state`` vocabulary grows,
        and the tier-0 case above is exactly what such a guard silently misses.

        It is also what the operator actually asked for, which was scoped from
        the start — "under Previous Sessions ... sorted to the top". Tiers 0/4/5
        were never in scope, and reordering them is unrequested behaviour change.

        **Glyph/order parity, stated precisely.** In Active, order defers
        entirely to the existing precedence in ``row_state_mark``; this key
        abstains. In Previous every row is cold, so ``row_state_mark`` paints
        only the wake glyph or nothing at all — the wake is the sole
        forward-looking fact a row can carry, so it leads.

        **Dormant ranks below armed, not nowhere.** ``wakes_dormant`` means the
        session was deliberately stopped and the schedule will never fire, so it
        stays under every armed row: floating it would advertise a future that is
        not coming. It still gets its own band so that every clock glyph in the
        group is CONTIGUOUS. The dim-vs-muted separation between an armed ``◷``
        and a dormant one measures 1.77:1 at the 8x17px cell this UI renders —
        below any discrimination threshold — so a dormant row stranded among the
        plain rows reads as a broken sort rather than as a distinct state.
        Banding it directly under the armed rows puts the block boundary where
        the glyph changes, and costs no row and no chrome.

        The key lives HERE rather than in :func:`session_category` because that
        function is shared with the mobile daemon, whose summaries carry no wake
        data at all (durable rows come from ``recent_session_rows``, not
        ``decorate_rows``, so ``wakes`` is always 0 there). Moving the key into
        the shared categoriser would either be a no-op on mobile or force wake
        plumbing into the daemon; keeping it in the catalog leaves both surfaces
        agreeing on the tier, which is the partition they actually share.

        Stability: a wake is as durable a fact as ``live_state`` and ``pending``,
        and shares their best-effort read — ``decorate_rows`` zeroes ``wakes``
        for a poll whose wake-index read raises, exactly as it empties the live
        map when the registry scan raises, so a transient failure can bounce a
        row for one poll and put it back on the next. That tolerance is
        deliberate (a picker that cannot read either source still lists every
        session), and this key inherits it rather than adding a new fragility.
        What IS new is an asymmetry worth naming: a cold Previous row previously
        had no poll-varying ordering input at all, and now has one. When a wake
        genuinely fires the session becomes live and changes tier, which is a
        real state change rather than churn.

        One consequence reaches MEMBERSHIP, not just order: :func:`load_catalog`
        ranks before applying ``[:limit]``, so at the ``CATALOG_SCAN_LIMIT``
        boundary an ancient session with an armed wake can now enter the window
        and displace a newer row that would otherwise have made it (measured at
        251 sessions). That is arguably the point of the feature — a scheduled
        session is usually an old one, and being unfindable is the report — but
        it is a real behaviour change beyond reordering, so it is recorded here.
        """
        tier = session_category(
            pending=bool(self.row.pending),
            busy=self.row.live_state in ("busy", "wedged"),
            unseen=self.unseen,
            kind=self.completion_kind,
            live=bool(self.row.live_state),
        )
        # Cold rows only. An active row takes the constant, which is what makes
        # the key structurally unable to reorder Active rather than merely
        # declining to today.
        armed = not self.active and bool(self.row.wakes) and not self.row.wakes_dormant
        dormant = not self.active and bool(self.row.wakes) and self.row.wakes_dormant
        wake_rank = 0 if armed else 1 if dormant else 2
        # Activity may update ages and badges, but must not move a click target.
        return tier, wake_rank, -self.row.created_at, self.id

    @property
    def active(self) -> bool:
        """Section membership is independent of the number of ordering categories."""
        return bool(self.row.pending or self.unseen or self.row.live_state)

    @property
    def status_code(self) -> str:
        """Stable transport spelling of the same precedence used by ``status``."""
        if self.row.pending:
            return "approval" if self.row.pending == "approval" else "answer"
        if self.row.live_state in {"wedged", "busy"}:
            return self.row.live_state
        if self.shows_completion_mark:
            return {"error": "error", "interrupted": "interrupted"}.get(
                self.completion_kind, "complete"
            )
        if self.row.live_state == "attached":
            return "attached"
        if self.row.wakes and not self.row.wakes_dormant:
            return "scheduled"
        if self.row.live_state == "idle":
            return "idle"
        if self.row.wakes:
            return "dormant"
        # A receipt is evidence of an outcome even after viewing; no receipt is
        # not evidence of success. Never turn an old unknown transcript green.
        if self.completion_token:
            return {"error": "error", "interrupted": "interrupted"}.get(
                self.completion_kind, "complete"
            )
        return "recent"

    @property
    def shows_completion_mark(self) -> bool:
        """Does an unread completion win the glyph, or does live state?

        THE single arbiter for that question: the sidebar reads it to decide
        whether to override :func:`row_state_mark`, and :attr:`status` reads it
        to decide whether to say "Unseen …". They used to make the decision
        separately, in matching order, held together by a comment asking the
        next author to keep them in step. That is what broke — and the pairing
        invariant this file already treats as load-bearing deserves a predicate
        rather than a promise.

        ``unseen`` is a LEVEL, not an edge: it is true from the moment a turn
        completes until somebody READS that session, and resuming a session
        does not acknowledge it. So the mark alone cannot be allowed to win —
        it says what the session did LAST, and the states below say what it is
        doing NOW:

        * ``pending`` — a parked gate. Already outranked unseen, and still
          does: a person is blocked on this row right now.
        * ``wedged`` — broken NOW. A stale mark from a turn that did finish
          must not hide a runtime that has since stopped answering.
        * ``busy`` — the reported bug. The session is working; painting the
          previous turn's outcome over its own spinner made seven resumed,
          healthy sessions read as seven failures.

        THE COST OF THE ``busy`` RULE, stated plainly because it is a real
        trade and not a free win: it suppresses an unread ``error`` exactly as
        it suppresses an unread ``interrupted``. A turn that FAILED, was
        resumed without being read, and is now running shows a spinner and a
        :attr:`status` of "Working" — the failure has no residue anywhere on
        the row, and the only route back to it is opening the session.

        That is accepted here, deliberately, on three grounds:

        1. Nothing is destroyed. This is a pure function of CURRENT state, not
           a latch: the instant the row stops being ``busy`` the ``✗`` and
           "Unseen error" come back, because ``unseen`` stays true until
           somebody actually reads the session.
        2. The row stays in "Active Sessions": membership is independent of
           ranking. While busy or wedged it uses the in-progress category, below
           unviewed completed outcomes; when it stops, its unread outcome earns
           that outcome's category again. Creation time orders each category.
        3. The alternative IS the reported bug, one class down. Painting an
           error mark over a running session's spinner is the same lie about
           the same row — "this is broken" over a session that is working.

        So the failure mode being accepted (a genuine error is quiet while its
        session runs, and returns when it stops) is strictly milder and strictly
        shorter-lived than the one being fixed (every resumed session claims to
        have failed, indefinitely, until read). If that trade ever needs
        revisiting, the cheap remedy is INK rather than shape — keep the
        spinner, tint it ``danger`` when ``completion_kind == "error"``, or
        widen the tooltip to "Working (last turn failed, unread)". Both are new
        design surface and belong in a round of their own, not here.

        Everything BELOW stays outranked by the mark, deliberately. ``attached``,
        an armed wake and ``idle`` are all facts about residency — true of a
        session that is merely sitting there — while "this finished and you
        have not read it" is a fact about work that is waiting for the
        operator. An unseen completion on a now-idle session is exactly the
        information the sidebar exists to keep, so it is kept.

        Ranking also suppresses stale completions while busy or wedged, so
        a row's category agrees with the live state its glyph communicates.
        """
        return (
            self.unseen and not self.row.pending and self.row.live_state not in ("wedged", "busy")
        )

    @property
    def status(self) -> str:
        if self.row.pending:
            return "Approval needed" if self.row.pending == "approval" else "Answer needed"
        # BEFORE the unseen branch, and mirrored by `shows_completion_mark`,
        # which is what the sidebar suppresses the mark on. A row that is
        # wedged or busy describes itself by what it is doing now.
        if self.row.live_state == "wedged":
            return "Not responding"
        if self.row.live_state == "busy":
            return "Working"
        if self.shows_completion_mark:
            return {"error": "Unseen error", "interrupted": "Unseen interruption"}.get(
                self.completion_kind, "Unseen completion"
            )
        # Follows ``row_state_mark``'s precedence EXACTLY, so the tooltip can
        # never name a different state from the glyph beside it. The glyph is a
        # single character and the description is where a user finds out what it
        # meant, so the two disagreeing is worse than either being terse.
        #
        # Every branch below mirrors one in ``row_state_mark``, in its order:
        # attached outranks an armed wake, which outranks idle, and a DORMANT
        # wake falls through to the cold case — where it is still the glyph, so
        # it must still be the words (round 1, D1: a cold row with a stopped
        # schedule drew the wake mark while the tooltip said "Recent").
        if self.row.live_state == "attached":
            return "Open"
        if self.row.wakes and not self.row.wakes_dormant:
            count = self.row.wakes
            return f"Scheduled ({count} wake{'s' if count != 1 else ''})"
        if self.row.live_state == "idle":
            return "Ready"
        if self.row.wakes:
            # Dormant: the schedule exists but the session was stopped, so it is
            # not going to fire. Named rather than hidden — a user who sees the
            # glyph needs to know why it is not going to act.
            #
            # "dormant" rather than a fresh adjective, because the stop receipt
            # this state comes FROM already says "N wakes dormant until you
            # reopen it" (``control.py`` / ``app.py``). Design review round 2
            # (D5) flagged that an earlier draft here said "paused" and split
            # the vocabulary for one fact across two surfaces; the receipt's
            # word is the established one, so this follows it rather than
            # asking the receipt to move.
            count = self.row.wakes
            return f"Stopped ({count} wake{'s' if count != 1 else ''} dormant)"
        if self.completion_token:
            return {"error": "Error", "interrupted": "Interrupted"}.get(
                self.completion_kind, "Complete"
            )
        return "Recent"


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
                        created_at=session_created_at(session_dir),
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
    return sorted(updated, key=lambda row: 0 if row.pending else 1)


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
_ROW_CACHE: dict[Path, tuple[tuple[float, int], SessionRow]] = {}


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


def cached_session_rows(
    directory: Path,
    limit: int = CATALOG_SCAN_LIMIT,
    *,
    candidates: list[tuple[str, float, str]] | None = None,
) -> list[SessionRow]:
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
    fresh: dict[Path, tuple[tuple[float, int], SessionRow]] = {}
    selected = (
        candidates if candidates is not None else _recent_sessions_with_origin(directory, limit)
    )
    for session_id, mtime, origin in selected:
        session_dir = directory / "sessions" / session_id
        key = _row_stat_key(session_dir)
        cached = _ROW_CACHE.get(session_dir.resolve())
        if key is not None and cached is not None and cached[0] == key:
            # Same transcript bytes as last poll: the name and the fork mark
            # cannot have changed, so neither read is repeated. `mtime` is
            # taken fresh from the scan regardless — it also tracks the inbox
            # spool. It updates displayed age, never immutable creation order.
            row = cached[1]._replace(mtime=mtime)
        else:
            row = SessionRow(
                session_id,
                mtime,
                session_name(session_dir),
                forked=origin == ORIGIN_FORK and wears_inherited_title(session_dir),
                created_at=session_created_at(session_dir),
            )
        rows.append(row)
        if key is not None:
            fresh[session_dir.resolve()] = (key, row)
    _ROW_CACHE.clear()
    _ROW_CACHE.update(fresh)
    return rows


def load_catalog(directory: Path, limit: int = CATALOG_SCAN_LIMIT) -> list[CatalogEntry]:
    """Rank a shared lightweight candidate snapshot before materializing a page.

    Discovery already stats the whole namespace. Applying a recency cap before
    attention lost old unread work; reading names for the entire store would
    undo the sidebar's bounded I/O. Rank cheap rows first, then hydrate only the
    requested prefix through the existing transcript-stat cache.
    """
    from dataclasses import replace

    from local_operator.resume import _recent_sessions_with_origin
    from local_operator.session.attention import AttentionStore, conversation_identity
    from local_operator.session.retention import TRANSCRIPT_FILENAME

    candidates = _recent_sessions_with_origin(directory)
    source = {session_id: (session_id, mtime, origin) for session_id, mtime, origin in candidates}
    # Creation time is the immutable ordering key (#800), so every construction
    # site must stamp it. Rows left at the 0.0 default all tie and fall through
    # to the session-id tie-break, which silently reverses newest-first order.
    rows = [
        SessionRow(
            session_id,
            mtime,
            "",
            created_at=session_created_at(directory / "sessions" / session_id),
        )
        for session_id, mtime, _ in candidates
    ]
    for marker in (directory / "sessions").glob("*/desktop.json"):
        if marker.parent.name in source or not session_directory_name(marker.parent.name):
            continue
        # A marker is a draft fallback, never a competing source of historical
        # title/mtime for a transcript that fell beyond a previous page bound.
        if (marker.parent / TRANSCRIPT_FILENAME).exists():
            continue
        try:
            rows.append(
                SessionRow(
                    marker.parent.name,
                    marker.stat().st_mtime,
                    "",
                    created_at=session_created_at(marker.parent),
                )
            )
        except OSError:
            continue
    rows = decorate_rows(directory, rows, include_live=True)
    identities = {row.id: conversation_identity(directory / "sessions" / row.id) for row in rows}
    attention: dict[str, dict[str, Any]] = {}
    try:
        attention = AttentionStore(directory / "attention.db").state_many(identities.values())
    except (sqlite3.Error, OSError):
        logger.debug("catalog attention unavailable", exc_info=True)
    entries = list(
        rank_entries(
            [
                CatalogEntry(
                    row,
                    bool(attention.get(identities[row.id], {}).get("unseen", False)),
                    str(attention.get(identities[row.id], {}).get("kind") or ""),
                    str(attention.get(identities[row.id], {}).get("completion_token") or ""),
                    str(attention.get(identities[row.id], {}).get("anchor_id") or ""),
                )
                for row in rows
            ]
        )
    )[:limit]
    named = {
        row.id: row
        for row in cached_session_rows(
            directory, candidates=[source[entry.id] for entry in entries if entry.id in source]
        )
    }
    return [
        (
            replace(
                entry,
                row=entry.row._replace(name=named[entry.id].name, forked=named[entry.id].forked),
            )
            if entry.id in named
            else entry
        )
        for entry in entries
    ]
