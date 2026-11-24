"""Asks and inbox rows whose conversation is gone: settle them, never strand them.

WHY THIS MODULE EXISTS (operator report 2026-11-24). The Notifications page
showed a "Waiting on you" ask that could be neither answered nor put away:
every button answered 404 ``requested_not_found``. Two causes were measured in
an isolated store (evidence in the PR):

* an ask raised by a SUBAGENT session is listed by the inbox but refused by the
  desktop pool's door -- fixed on the answers route
  (``server/routes/desktop_sessions._doorless_ask_target``);
* a conversation whose directory leaves this device by any path other than
  ``session.cleanup.remove_session_dir`` (a manual ``rm``, a crash mid-move, a
  restore from an older backup) leaves its ``notifications.db`` rows behind,
  forever, because only that one removal site cascades into the store.

This module is the cascade's three remaining legs:

1. :func:`withdraw_open_asks` -- an EXPLICIT delete of a conversation that still
   has an open ask used to be refused (and with the misleading fallback
   sentence "Whether that conversation is in use could not be checked"); the
   delete now withdraws those asks first, recording ``reason:
   conversation_deleted`` in the session's own log, so the cleanup receipt and
   the removal that follows describe a settled question rather than a
   destroyed one.
2. :func:`settle_deleted_ask` -- the answers route's local settle for a client
   that still shows a row whose conversation is gone.
3. :func:`sweep` -- the bounded backstop for every path nobody instrumented:
   run at ``lop serve`` startup and every :data:`SWEEP_INTERVAL_S`.

THE SWEEP IS INDEX-DRIVEN, NEVER A SCAN (operator requirement 4). Its inputs are
exactly the two cross-session indexes the inbox itself reads -- the ``asks/``
directory (one small JSON per session with anything to show) and
``notifications.db``'s ``items_session`` index -- plus ONE ``stat`` per distinct
session id they name. It never lists ``sessions/`` (thousands of directories)
and never opens a transcript. Cost is O(sessions with asks or inbox rows), and
the incremental pass re-reads an ask entry only when its file changed or one of
its asks reached its deadline.

WHAT "GONE" MEANS (the invariant every leg shares): no ``sessions/<id>``
directory, no ``sessions/handed-away/<id>`` (a conversation mid-hand-off is not
deleted), and no CACHED peer row for the id (``peer_rows.peer_session_row``,
which never dials). A conversation moved to another device and still listed
there is therefore never swept; a peer-mirrored inbox row (``remote-*`` key) is
never even a candidate (``notifications.items.is_remote_key``).

EXPIRY. A past-deadline ``open`` ask is already re-derived ``timed_out`` by every
index reader (``store.index_asks``), so the badge never counts it; the sweep
makes that durable by unlinking an entry the index's own staleness rule
(``store.entry_is_stale``) has retired, and reports how many asks it saw past
their deadline. It never writes a ``timed_out`` event into a session's log:
the timeout NOTICE is owed to the agent and is delivered by that session's
deadline wake, which a cold write from here would pre-empt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.asks import store

logger = logging.getLogger(__name__)

#: Machine reason recorded on a withdrawal the delete cascade writes. A token,
#: not prose: it is read by the fold and by audits, and the user-facing copy is
#: the desktop's own catalogued "conversation deleted" label.
WITHDRAW_REASON_DELETED = "conversation_deleted"

#: How often ``lop serve`` re-runs the sweep after its startup pass. TEN MINUTES
#: because the sweep is a BACKSTOP: every instrumented delete path cascades
#: inline (``cleanup.remove_session_dir``), so this only catches directories
#: removed out of band, which are rare and harmless for minutes. A tighter
#: cadence buys nothing a user can see; the pass is cheap either way (measured
#: in the PR on a clone of a 2.5k-session store).
SWEEP_INTERVAL_S = 600.0

#: The most sessions ONE pass settles. Bounds the write work (each settle is one
#: SQLite transaction plus one unlink) so a store with a large backlog -- the
#: first pass after this ships -- cannot hold the notifications writer for long;
#: the remainder is picked up by the next pass. Stats and the read of the two
#: indexes are not bounded by this: they are the cheap part, and bounding them
#: would make a backlog invisible to the dry-run.
SWEEP_MAX_SETTLES = 200


#: The conversation-id grammar (``uuid4().hex[:12]``, the session store's own
#: ids). ONLY such ids can be orphans: the inbox also files rows under fixed
#: pseudo-sessions with no directory by design (``notifications.store.
#: PSEUDO_SESSION_IDS``: ``network-credentials``, measured with 9 live rows on
#: the reference store) and a sweep keyed on "no directory" would delete them.
_CONVERSATION_ID = re.compile(r"^[a-f0-9]{12}$")


def is_conversation_id(session_id: str) -> bool:
    """Whether ``session_id`` has the shape of a real conversation id."""
    return isinstance(session_id, str) and bool(_CONVERSATION_ID.match(session_id))


def session_gone(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` is provably not a conversation this device can reach.

    See the module docstring's "WHAT GONE MEANS". Blocking (a few stats, one
    cache read); never dials a peer.
    """
    root = Path(config_dir)
    if not is_conversation_id(session_id):
        return False
    sessions = root / "sessions"
    if (sessions / session_id).is_dir():
        return False
    try:
        from local_operator.session.cleanup import HANDED_AWAY_DIRNAME

        if (sessions / HANDED_AWAY_DIRNAME / session_id).exists():
            return False
    except Exception:  # noqa: BLE001 -- unprovable is "not gone"
        return False
    try:
        from local_operator.session.peer_rows import peer_session_row

        if peer_session_row(session_id, root) is not None:
            return False
    except Exception:  # noqa: BLE001 -- unprovable is "not gone"
        logger.debug("orphans: peer cache unreadable for %s", session_id, exc_info=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Delete cascade
# ---------------------------------------------------------------------------


def withdraw_open_asks(directory: Path, *, by: str) -> list[str]:
    """Withdraw every still-answerable ask in ``directory``'s log. Returns their ids.

    Called by ``session.cleanup.delete_session`` immediately before the
    directory is removed, inside the same call (the explicit delete is the
    operator's decision about this conversation, and an open question in it is
    part of what they chose to delete). "Still answerable" is the set the
    deletion guard protects (``cleanup._has_open_asks``): ``open``,
    ``timed_out`` and ``late``. A relayed ask (a pointer into another
    session) is withdrawn locally the same way the queue's own withdraw records
    it; the origin's ask is untouched.

    Raises nothing the caller must handle for a single unwritable append: the
    ids returned are exactly the ones whose event landed, so the caller can
    re-check the guard and refuse when something remained open.
    """
    now = store.now_ms()
    records = store.fold(store.read_events(directory), now)
    answerable = {store.STATUS_OPEN, store.STATUS_TIMED_OUT, store.STATUS_LATE}
    withdrawn: list[str] = []
    for record in records:
        if record.get("status") not in answerable:
            continue
        ask_id = str(record.get("ask_id") or "")
        if not ask_id:
            continue
        payload = {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_WITHDRAWN,
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": by},
            "reason": WITHDRAW_REASON_DELETED,
        }
        if store.append_event(directory, payload):
            withdrawn.append(ask_id)
    return withdrawn


def _drop_ask_from_index(config_dir: Path, session_id: str, ask_id: str) -> bool:
    entry = store.read_entry(config_dir, session_id)
    if entry is None:
        return False
    rows = [dict(row) for row in entry.get("asks") or () if isinstance(row, dict)]
    kept = [row for row in rows if str(row.get("ask_id") or "") != ask_id]
    if len(kept) == len(rows):
        return False
    store.write_entry(config_dir, session_id, cwd=str(entry.get("cwd") or ""), asks=kept)
    return True


def _drop_ask_ref(config_dir: Path, session_id: str, ask_id: str) -> None:
    """Forget the inbox's read mark for one ask (``ref_state``); best-effort."""
    from local_operator.notifications.store import NotificationStore, ask_ref

    notes = NotificationStore(config_dir)
    if not notes.path.exists():
        return
    ref = ask_ref(session_id, ask_id)
    notes._write(  # noqa: SLF001 -- the store's own transaction helper
        lambda conn: conn.execute("DELETE FROM ref_state WHERE ref=?", (ref,)).rowcount,
        "orphan_ask_ref",
    )


async def settle_deleted_ask(config_dir: Path | str, session_id: str, ask_id: str) -> None:
    """Put away an ask whose conversation is gone (desktop Dismiss/Decline).

    There is no log to write (it left with the directory) and no agent to tell,
    so settling is removing the two records outside the directory that can
    still name the ask: its row in the cross-session index (which the reader's
    own sweep would drop on the next read anyway) and the inbox's read mark.
    Idempotent and best-effort: the caller answers 200 either way, because the
    row the user tapped no longer has anything behind it to fail.
    """
    root = Path(config_dir)

    def _settle() -> None:
        try:
            _drop_ask_from_index(root, session_id, ask_id)
        except Exception:  # noqa: BLE001 -- the reader sweeps a gone entry anyway
            logger.debug("orphans: index drop failed for %s", session_id, exc_info=True)
        try:
            _drop_ask_ref(root, session_id, ask_id)
        except Exception:  # noqa: BLE001 -- a stale read mark is invisible
            logger.debug("orphans: ref drop failed for %s", session_id, exc_info=True)

    await asyncio.to_thread(_settle)


def _wake_entry_raw(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    import json

    from local_operator.wakes.store import entry_path

    try:
        payload = json.loads(entry_path(config_dir, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def only_ask_deadline_wakes(config_dir: Path, session_id: str) -> bool:
    """Whether every schedule on this session's wake entry is an ask's deadline row.

    WHY THE DELETE NEEDS THIS. Every queued ask arms a HIDDEN ``ask_timeout``
    wake row (``AskQueue._arm_deadline_wake``) so its deadline survives a
    runtime that is gone, which means the cleanup guard meets "has an armed
    wake" BEFORE it ever reaches "has an open ask" -- an explicit delete of a
    conversation with a waiting question was refused with a sentence about a
    reminder the user never set. Those rows belong to the asks the delete
    withdraws; any OTHER schedule (a reminder, a goal) still refuses.
    Unreadable or empty is ``False``: the guard's refusal stands.
    """
    entry = _wake_entry_raw(Path(config_dir), session_id)
    if entry is None:
        return False
    rows = [row for row in entry.get("schedules") or () if isinstance(row, dict)]
    if not rows or entry.get("goal_only"):
        return False
    return all(
        row.get("kind") == "ask_timeout"
        or str(row.get("id") or "").startswith(store.TIMEOUT_ROW_PREFIX)
        for row in rows
    )


def drop_ask_deadline_wakes(config_dir: Path, session_id: str) -> int:
    """Remove the ``ask_timeout`` rows from this session's wake index entry.

    Called by the explicit delete right after :func:`withdraw_open_asks`: a
    withdrawn ask's deadline fire would be stale anyway (the queue's watermark
    rule), and leaving the row armed is what keeps the cleanup guard refusing.
    Keeps every other row and every unknown key (``preserve``), so a schedule
    this function does not own is never lost. Returns how many rows it dropped.
    """
    from local_operator.wakes.store import write_entry

    root = Path(config_dir)
    entry = _wake_entry_raw(root, session_id)
    if entry is None:
        return 0
    rows = [row for row in entry.get("schedules") or () if isinstance(row, dict)]
    kept = [
        row
        for row in rows
        if not (
            row.get("kind") == "ask_timeout"
            or str(row.get("id") or "").startswith(store.TIMEOUT_ROW_PREFIX)
        )
    ]
    if len(kept) == len(rows):
        return 0
    write_entry(
        root, session_id, cwd=str(entry.get("cwd") or ""), schedules=kept, preserve=entry
    )
    return len(rows) - len(kept)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepReport:
    """What one pass found and (unless ``dry_run``) settled. JSON-able via :meth:`as_dict`."""

    dry_run: bool
    ask_entries_seen: int = 0
    ask_entries_read: int = 0
    note_sessions_seen: int = 0
    orphan_ask_entries: list[dict[str, Any]] = field(default_factory=list)
    stale_ask_entries: list[str] = field(default_factory=list)
    asks_past_deadline: int = 0
    orphan_note_sessions: list[dict[str, Any]] = field(default_factory=list)
    settled: int = 0
    deferred: int = 0
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "ask_entries_seen": self.ask_entries_seen,
            "ask_entries_read": self.ask_entries_read,
            "note_sessions_seen": self.note_sessions_seen,
            "orphan_ask_entries": self.orphan_ask_entries,
            "stale_ask_entries": self.stale_ask_entries,
            "asks_past_deadline": self.asks_past_deadline,
            "orphan_note_sessions": self.orphan_note_sessions,
            "settled": self.settled,
            "deferred": self.deferred,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


#: Incremental memo per config dir: ``{entry name: (mtime_ns, next_deadline_ms)}``.
#: An entry whose file did not change and whose earliest open deadline has not
#: arrived cannot have become stale by the index's own rule, so it is not
#: re-read. Its SESSION is still stat'ed every pass -- deletion does not touch
#: the entry file, so mtime alone cannot prove the session is still there.
_MEMO: dict[str, dict[str, tuple[int, int]]] = {}
_MEMO_LOCK = threading.Lock()


def _next_deadline(entry: dict[str, Any], now: int) -> int:
    """The earliest FUTURE instant this entry's verdict or deadline count changes (ms).

    Only moments after ``now``: a deadline already passed was counted by the
    read that produced this memo, so keeping it would re-read the entry on
    every later pass until its 7-day horizon, which is the cost the memo exists
    to avoid. ``0`` = nothing pending (re-read only when the file changes).
    """
    horizon = store.LATE_WINDOW_S * 1000
    moments: list[int] = []
    for ask in entry.get("asks") or ():
        if not isinstance(ask, dict):
            continue
        expires_at = int(ask.get("expires_at") or 0)
        if not expires_at:
            continue
        if ask.get("status") == store.STATUS_OPEN:
            moments.append(expires_at)
        moments.append(expires_at + horizon)
    future = [moment for moment in moments if moment > now]
    return min(future) if future else 0


def _note_sessions(config_dir: Path) -> list[tuple[str, int]]:
    """Distinct session ids with LOCAL inbox rows or ask read marks, with row counts.

    Read-only and served from ``items_session`` (a covering index on
    ``session_id``), plus ``ref_state``'s primary key for ask marks. A session
    with any peer-mirrored row (``remote-*`` key) is excluded at the source.
    """
    path = config_dir / "notifications.db"
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    try:
        rows = conn.execute(
            "SELECT session_id, SUM(key NOT LIKE 'remote-%'), SUM(key LIKE 'remote-%') "
            "FROM items WHERE session_id <> '' GROUP BY session_id"
        ).fetchall()
        # A session with ANY peer-mirrored row is a conversation another device
        # reports holding: MOVED, not deleted. Measured on the reference store:
        # ``ce6fcb93c28a`` left this device by ``policy: mesh-move`` and kept two
        # local rows from before the move beside two ``remote-*`` rows from its
        # new owner; with a cold peer cache only this evidence keeps the sweep
        # from deleting them.
        counts = {str(sid): int(local or 0) for sid, local, remote in rows if not remote}
        moved = {str(sid) for sid, _local, remote in rows if remote}
        from local_operator.notifications.store import ASK_REF_PREFIX

        for (ref,) in conn.execute(
            "SELECT ref FROM ref_state WHERE ref LIKE ?", (f"{ASK_REF_PREFIX}%",)
        ):
            rest = str(ref)[len(ASK_REF_PREFIX) :]
            sid = rest.split(":", 1)[0]
            if sid and sid not in moved:
                counts.setdefault(sid, 0)
        return sorted(counts.items())
    finally:
        conn.close()


def sweep(
    config_dir: Path | str,
    *,
    dry_run: bool = False,
    now_ms: int | None = None,
    max_settles: int = SWEEP_MAX_SETTLES,
    incremental: bool = True,
) -> SweepReport:
    """One bounded pass over the two indexes. See the module docstring.

    ``dry_run`` reads and reports, writes nothing (and does not advance the
    incremental memo, so a dry run never hides work from the next real pass).
    ``incremental=False`` re-reads every ask entry (the "full pass").
    Blocking: ``lop serve`` runs it in a worker thread.
    """
    started = time.perf_counter()
    root = Path(config_dir)
    now = now_ms if now_ms is not None else store.now_ms()
    report = SweepReport(dry_run=dry_run)
    key = str(root.resolve())
    with _MEMO_LOCK:
        memo = dict(_MEMO.get(key, {})) if incremental else {}
    next_memo: dict[str, tuple[int, int]] = {}
    gone_cache: dict[str, bool] = {}

    def gone(session_id: str) -> bool:
        if session_id not in gone_cache:
            gone_cache[session_id] = session_gone(root, session_id)
        return gone_cache[session_id]

    budget = max(0, int(max_settles))

    # -- the ask index -------------------------------------------------------
    directory = store.asks_dir(root)
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        entries = []
    for dirent in entries:
        name = dirent.name
        if not name.endswith(".json") or name.startswith("."):
            continue
        session_id = name[: -len(".json")]
        report.ask_entries_seen += 1
        try:
            mtime = dirent.stat().st_mtime_ns
        except OSError:
            continue
        if gone(session_id):
            report.orphan_ask_entries.append({"session_id": session_id})
            if dry_run:
                continue
            if budget <= 0:
                report.deferred += 1
                continue
            store.remove_entry(root, session_id)
            report.settled += 1
            budget -= 1
            continue
        cached = memo.get(name)
        if cached is not None and cached[0] == mtime and (not cached[1] or now < cached[1]):
            next_memo[name] = cached
            continue
        entry = store.read_entry(root, session_id)
        report.ask_entries_read += 1
        if entry is None:
            continue
        report.asks_past_deadline += sum(
            1
            for ask in entry.get("asks") or ()
            if isinstance(ask, dict)
            # ``timed_out`` too: an index writer that folded after the deadline
            # already stored the re-derived status; both are "past deadline".
            and ask.get("status") in (store.STATUS_OPEN, store.STATUS_TIMED_OUT)
            and 0 < int(ask.get("expires_at") or 0) <= now
        )
        if store.entry_is_stale(entry, now, session_exists=True):
            report.stale_ask_entries.append(session_id)
            if dry_run:
                continue
            if budget <= 0:
                report.deferred += 1
                continue
            store.remove_entry(root, session_id)
            report.settled += 1
            budget -= 1
            continue
        next_memo[name] = (mtime, _next_deadline(entry, now))

    # -- the inbox store -----------------------------------------------------
    try:
        candidates = _note_sessions(root)
    except sqlite3.Error:
        logger.warning("orphans: notifications store unreadable; inbox leg skipped")
        candidates = []
    report.note_sessions_seen = len(candidates)
    doomed: list[str] = []
    for session_id, count in candidates:
        if not gone(session_id):
            continue
        report.orphan_note_sessions.append({"session_id": session_id, "items": count})
        doomed.append(session_id)
    if not dry_run and doomed:
        from local_operator.notifications.store import NotificationStore

        notes = NotificationStore(root)
        for session_id in doomed:
            if budget <= 0:
                report.deferred += 1
                continue
            try:
                # The SAME call the explicit-delete cascade makes
                # (``cleanup.remove_session_dir``): items, their history and the
                # ask read marks, in one transaction, which moves the store's
                # fingerprint -- the desktop feed's "notifications" frame (badge,
                # dock) republishes on its next tick without a reload.
                notes.delete_session(session_id)
                report.settled += 1
                budget -= 1
            except Exception:  # noqa: BLE001 -- retried next pass
                logger.warning("orphans: could not settle inbox rows for %s", session_id)
                report.deferred += 1

    if not dry_run:
        with _MEMO_LOCK:
            _MEMO[key] = next_memo
    report.elapsed_ms = (time.perf_counter() - started) * 1000
    if report.settled or report.deferred:
        logger.info(
            "orphans: settled %d, deferred %d (ask entries %d, inbox sessions %d) in %.1f ms",
            report.settled,
            report.deferred,
            report.ask_entries_seen,
            report.note_sessions_seen,
            report.elapsed_ms,
        )
    return report


def prime_peer_rows(config_dir: Path | str) -> None:
    """Refresh the cached peer listing ``session_gone`` consults. Best-effort.

    ``peer_rows.peer_session_row`` is cache-only and per process; in ``lop
    serve`` the sidebar poll keeps it warm only while the app is open, and a
    CLI process starts with it empty. Priming once per pass (one bounded relay
    read; ``()`` at once on a device in no network) is what keeps a conversation
    MOVED to another device from reading as deleted. Kept out of :func:`sweep`
    so the pass itself stays local-only and its measured cost is the sweep's.
    """
    try:
        from local_operator.session.peer_rows import peer_session_rows

        peer_session_rows(Path(config_dir))
    except Exception:  # noqa: BLE001 -- a cold cache only narrows "not gone"
        logger.debug("orphans: peer listing unavailable", exc_info=True)


def sweep_with_peers(config_dir: Path | str, **kwargs: Any) -> SweepReport:
    """:func:`prime_peer_rows` then :func:`sweep` -- what serve and the CLI run."""
    prime_peer_rows(config_dir)
    return sweep(config_dir, **kwargs)


async def run_forever(
    config_dir: Path | str, *, interval_s: float = SWEEP_INTERVAL_S, first_delay_s: float = 5.0
) -> None:
    """``lop serve``'s sweeper: one pass shortly after startup, then every ``interval_s``.

    ``first_delay_s`` keeps the startup pass off the boot's critical path (the
    record, the first sidebar poll) without leaving a just-started daemon's
    badge wrong for long. Never raises out: a failed pass is logged and the next
    one runs on schedule.
    """
    await asyncio.sleep(first_delay_s)
    while True:
        try:
            await asyncio.to_thread(sweep_with_peers, config_dir)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- housekeeping never fails the daemon
            logger.warning("orphans: sweep pass failed", exc_info=True)
        await asyncio.sleep(interval_s)


__all__ = [
    "SWEEP_INTERVAL_S",
    "SWEEP_MAX_SETTLES",
    "WITHDRAW_REASON_DELETED",
    "drop_ask_deadline_wakes",
    "is_conversation_id",
    "only_ask_deadline_wakes",
    "SweepReport",
    "run_forever",
    "session_gone",
    "settle_deleted_ask",
    "prime_peer_rows",
    "sweep",
    "sweep_with_peers",
    "withdraw_open_asks",
]
