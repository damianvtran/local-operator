"""``/api/schedules`` — the machine-wide read of what is ARMED on this machine.

WHAT THIS IS. The relay half of ``GET /v1/desktop/wakes`` and
``GET /v1/desktop/monitors`` in ONE answer, because the two families share one
surface everywhere they are drawn: the TUI paints wakes and monitors into the
single wake band (``tui/widgets/wake_panel`` feeds both into one body), and the
phone's Schedules view is one list. One fetch feeds it; two routes would make
the phone do two round trips for one screen with no gain. The payload carries
both listings under ``wakes`` and ``monitors``, each in the exact shape the
desktop serves.

WHY A MACHINE-WIDE INDEX AND NOT A PER-SESSION FIELD. ``GET /api/sessions`` is
ranked by recency and capped, and a session armed once and never opened has its
mtime stamped at arm time — so a schedule armed months ago would silently be
absent from a listing built on that page. The derived index is one small file
per carrying session (``wakes/store.py``, ``monitors/store.py``), so this read
is O(sessions with wakes/monitors) and complete whatever the store's size. It
also needs NO runtime: a schedule outlives the runtime it was armed from, so
this must answer with nothing running — the same durability claim the asks
aggregate is built on.

READ-ONLY, deliberately. Arming, editing and cancelling stay on the terminal
and the desktop plane: nothing here writes, dials an owner or touches a
session, and no route above it does either.

TWO EMPTY ANSWERS, STILL DISTINGUISHABLE. "No schedules" and "this process
could not read the store" are different claims; each listing carries its
store's own ``read_error`` (``read_index_report``) rather than collapsing an
unreadable directory into an empty list. The wake listing also carries the
``supervisor`` block — whether anything on this machine would actually fire a
cold wake — and monitors deliberately do not (they never engage a cold
session, so a supervisor-shaped field would advertise a watcher that does not
exist).

WHY THE DERIVATION IS MIRRORED, NOT IMPORTED. ``desktop_wakes._collect_listing``
and ``desktop_monitors._collect_listing`` are the two references this module
mirrors row for row; they live in FastAPI route modules whose private helpers
deliberately do not cross module boundaries, and importing them here would put
the server's request stack on the daemon's read path. The WIRE SHAPE is what
cannot drift: the rows below are built as the very models the desktop serves
(``server.models.desktop_wakes`` / ``...desktop_monitors``), so nothing the
wire needs goes silently missing — a dump emits every declared field, and a
change required to construct the model fails loudly. Construction alone is not
the whole guard, though: the models are ``extra="allow"``, so a defaulted
addition or a rename is absorbed quietly. That quiet path is pinned instead of
assumed — the declared field sets are asserted in
``tests/unit/mobile/test_schedules_relay.py``
(``test_the_shared_wire_models_field_sets_are_pinned``) — so a field added,
renamed or removed reds a test before it can reach a phone as a default the
desktop never serves. The predicates are the stores' own (``is_held``,
``scheduled_rows``, ``_is_stale_ms``, ``_session_exists``, ``health_hint``,
``unavailable_since_of``) — never a second derivation of a verdict another
process made.

THREADING. Pure filesystem work, like the desktop collectors': the daemon
hands it to a worker thread and it must not run on the event loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.server.models.desktop_monitors import (
    MonitorEntry,
    MonitorListing,
    MonitorRow,
)
from local_operator.server.models.desktop_wakes import (
    SupervisorInfo,
    WakeEntry,
    WakeListing,
    WakeScheduleRow,
)

logger = logging.getLogger(__name__)

#: The desktop listings' own default pages, mirrored (``WAKE_LIST_LIMIT_DEFAULT``
#: / ``MONITOR_LIST_LIMIT_DEFAULT``): the cardinality that matters is sessions
#: CARRYING wakes/monitors, and the caps exist so a pathological store degrades
#: VISIBLY (``truncated``) rather than shipping an unbounded document to a
#: phone. The desktop routes' ``limit`` / ``include_dormant`` query parameters
#: are not mirrored yet — the phone reads the default page (dormant rows
#: included), and a narrower page is a wire change that should ship with the
#: client that asks for it.
WAKE_LIST_LIMIT = 200
MONITOR_LIST_LIMIT = 200


def list_payload(config_dir: Path) -> dict[str, Any]:
    """The one answer: both machine-wide listings, in the desktop wire shapes."""
    return {
        "wakes": _wakes_listing(Path(config_dir), WAKE_LIST_LIMIT),
        "monitors": _monitors_listing(Path(config_dir), MONITOR_LIST_LIMIT),
    }


# ---------------------------------------------------------------------------
# The wake listing (``desktop_wakes._collect_listing``, mirrored)
# ---------------------------------------------------------------------------


def _wakes_listing(config_dir: Path, limit: int) -> dict[str, Any]:
    from local_operator.resume import session_name, session_origin
    from local_operator.wakes.store import is_held, read_index_report, scheduled_rows
    from local_operator.wakes.supervisor import _is_stale_ms, _session_exists

    index, read_error = read_index_report(config_dir)
    now_ms = int(time.time() * 1000)
    entries: list[WakeEntry] = []
    for session_id, raw in index.items():
        if not isinstance(raw, Mapping):
            continue
        # ``is_held`` is the store's one predicate for both park markers —
        # ``stopped_at`` and Aida's ``held_at`` (the desktop listing spells the
        # same two keys inline; behaviourally identical, but the store is the
        # one place a future marker lands).
        dormant = is_held(raw)
        session_dir = config_dir / "sessions" / session_id
        cwd = str(raw.get("cwd") or "")
        schedules = _schedule_rows(raw, now_ms, _is_stale_ms)
        # A session whose ONLY rows are internal timers (patience waits) is not
        # a wake-carrying session on any human surface: the entry would render
        # as an empty row, and the user never armed anything it could show. An
        # entry that shows zero rows without any being filtered is left as it
        # was (the store's own "no wakes" shape, which the listing has always
        # carried).
        raw_rows = list(raw.get("schedules") or ())
        if not schedules and len(scheduled_rows(raw_rows)) != len(raw_rows):
            continue
        entries.append(
            WakeEntry(
                session_id=session_id,
                name=_display_name(session_dir, session_id, cwd, session_name),
                cwd=cwd,
                origin=_origin(session_dir, session_origin),
                updated_at=_as_int(raw.get("updated_at")) or 0,
                dormant=dormant,
                # GHOST, asked with the supervisor's own predicate: the index
                # outlives the session directory it names. A dormant entry is
                # not a ghost — its session is deliberately parked, not gone.
                ghost=not dormant and not _session_exists(config_dir, session_id),
                next_due_at=_next_due_at(schedules),
                schedules=schedules,
            )
        )
    # Soonest first, undateable last, ties by id: the same rule the desktops'
    # pages apply to the same values, so the three surfaces agree.
    entries.sort(
        key=lambda entry: (entry.next_due_at is None, entry.next_due_at or 0, entry.session_id)
    )
    total = len(entries)
    return WakeListing(
        entries=entries[:limit],
        generated_at=now_ms,
        total=total,
        truncated=total > limit,
        supervisor=_supervisor_info(config_dir),
        read_error=read_error,
    ).model_dump()


def _schedule_rows(entry: Mapping[str, Any], now_ms: int, is_stale) -> list[WakeScheduleRow]:
    from local_operator.wakes.store import scheduled_rows

    rows: list[WakeScheduleRow] = []
    # HIDDEN patience waits never appear in a listing: one shared filter with
    # the CLI, the feed and the picker's count (``wakes.store.scheduled_rows``).
    # The supervisor reads the same index unfiltered.
    for raw in scheduled_rows(entry.get("schedules") or ()):
        if not isinstance(raw, Mapping):
            continue
        due = _as_int(raw.get("next_due_at"))
        if due is None:
            continue
        rows.append(
            WakeScheduleRow(
                id=str(raw.get("id") or ""),
                message=str(raw.get("message") or ""),
                next_due_at=due,
                every_ms=_as_int(raw.get("every_ms")),
                until_at=_as_int(raw.get("until_at")),
                limit=_as_int(raw.get("limit")),
                fired_count=_as_int(raw.get("fired_count")) or 0,
                overdue_s=max((now_ms - due) / 1000.0, 0.0),
                stale=is_stale(due, now_ms),
                last_fired_at=_as_int(raw.get("last_fired_at")),
                last_attempt_at=_as_int(raw.get("last_attempt_at")),
            )
        )
    rows.sort(key=lambda row: (row.next_due_at, row.id))
    return rows


def _next_due_at(rows: list[WakeScheduleRow]) -> int | None:
    return min((row.next_due_at for row in rows), default=None)


def _supervisor_info(config_dir: Path) -> SupervisorInfo:
    """Whether anything is actually watching this store's wakes.

    ``verifiable`` rides along although the contract names three fields: on a
    store outside the real home (every sandboxed run) launchd cannot speak
    about it at all, and reporting ``running: false`` there would be a claim
    about someone else's domain.
    """
    from local_operator.wakes.install import is_supported, supervisor_state

    try:
        state = supervisor_state(config_dir)
    except Exception:  # noqa: BLE001 — a probe failure must not fail the listing
        logger.warning("could not read the wake supervisor state", exc_info=True)
        return SupervisorInfo(supported=is_supported(), running=False, detail="unavailable")
    return SupervisorInfo(
        supported=is_supported(),
        running=state.running,
        detail=state.detail,
        verifiable=state.verifiable,
    )


# ---------------------------------------------------------------------------
# The monitor listing (``desktop_monitors._collect_listing``, mirrored)
# ---------------------------------------------------------------------------


def _monitors_listing(config_dir: Path, limit: int) -> dict[str, Any]:
    from local_operator.monitors.store import is_held, read_index_report
    from local_operator.resume import session_name, session_origin
    from local_operator.wakes.supervisor import _session_exists

    index, read_error = read_index_report(config_dir)
    now_ms = int(time.time() * 1000)
    entries: list[MonitorEntry] = []
    for session_id, raw in index.items():
        if not isinstance(raw, Mapping):
            continue
        # ``stopped_at`` is the one park marker monitors carry — there is no
        # Aida engine for them — and ``is_held`` is the store's own spelling
        # for it, so this listing and the cleanup guards cannot disagree.
        dormant = is_held(raw)
        session_dir = config_dir / "sessions" / session_id
        cwd = str(raw.get("cwd") or "")
        monitors = _monitor_rows(raw, now_ms, dormant)
        entries.append(
            MonitorEntry(
                session_id=session_id,
                name=_display_name(session_dir, session_id, cwd, session_name),
                cwd=cwd,
                origin=_origin(session_dir, session_origin),
                updated_at=_as_int(raw.get("updated_at")) or 0,
                dormant=dormant,
                ghost=not dormant and not _session_exists(config_dir, session_id),
                next_due_at=_next_due(monitors),
                monitors=monitors,
            )
        )
    entries.sort(
        key=lambda entry: (entry.next_due_at is None, entry.next_due_at or 0, entry.session_id)
    )
    total = len(entries)
    return MonitorListing(
        entries=entries[:limit],
        generated_at=now_ms,
        total=total,
        truncated=total > limit,
        read_error=read_error,
    ).model_dump()


def _monitor_rows(entry: Mapping[str, Any], now_ms: int, dormant: bool) -> list[MonitorRow]:
    from local_operator.monitors import store as monitor_store

    rows: list[MonitorRow] = []
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping):
            continue
        due = _as_int(raw.get("next_due_at"))
        last = _as_int(raw.get("last_check_at")) or 0
        disabled = bool(raw.get("disabled"))
        until = _as_int(raw.get("until_at"))
        expired = until is not None and until <= now_ms
        # The CLI's precedence (``cli._monitor_state_word``): dormancy wins
        # over disabled, so a failure word cannot point a reader at the wrong
        # remedy; disabled wins over the clock, so a watch that does not tick
        # never reads as merely late. "expired" is the third terminal state.
        if dormant:
            state = "dormant"
        elif disabled:
            state = "disabled"
        elif expired:
            state = "expired"
        else:
            state = "armed"
        arguments = raw.get("arguments")
        ignore = raw.get("ignore")
        rows.append(
            MonitorRow(
                id=str(raw.get("id") or ""),
                name=str(raw.get("name") or ""),
                tool=str(raw.get("tool") or ""),
                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                description=str(raw.get("description") or ""),
                every_ms=_as_int(raw.get("every_ms")),
                until_at=until,
                notify=bool(raw.get("notify")),
                sort_lines=bool(raw.get("sort_lines")),
                ignore=[str(item) for item in ignore] if isinstance(ignore, (list, tuple)) else [],
                cwd=str(raw.get("cwd") or ""),
                created_at=_as_int(raw.get("created_at")) or 0,
                next_due_at=due,
                last_check_at=last,
                checks=_as_int(raw.get("checks")) or 0,
                deliveries=_as_int(raw.get("deliveries")) or 0,
                consecutive_failures=_as_int(raw.get("consecutive_failures")) or 0,
                disabled=disabled,
                disabled_reason=str(raw.get("disabled_reason") or ""),
                due_in_s=None if due is None else (due - now_ms) / 1000.0,
                last_check_age_s=None if not last else max((now_ms - last) / 1000.0, 0.0),
                state=state,
                unavailable_since=monitor_store.unavailable_since_of(raw),
                # The one health sentence every monitor surface shares,
                # rendered from the store rather than re-derived here.
                health=monitor_store.health_hint({**raw, "next_due_at": due}, now_ms),
            )
        )
    rows.sort(key=lambda row: (row.next_due_at is None, row.next_due_at or 0, row.id))
    return rows


def _next_due(rows: list[MonitorRow]) -> int | None:
    return min((row.next_due_at for row in rows if row.next_due_at is not None), default=None)


# ---------------------------------------------------------------------------
# Helpers shared by both listers
# ---------------------------------------------------------------------------


def _display_name(session_dir: Path, session_id: str, cwd: str, read_name) -> str:
    """The conversation's name, or a floor built from its id and directory.

    Best-effort by contract: a name is decoration, and a listing that answered
    500 because one transcript could not be read would hide every other
    schedule on the machine. The floor is not optional either — a nameless row
    is a row the user cannot identify or open with confidence. (The desktop
    listings duplicate this same helper between their route modules by their
    own rule; this is the third surface it serves, kept verbatim.)
    """
    try:
        name = read_name(session_dir)
    except Exception:  # noqa: BLE001
        logger.warning("could not read the name of %s", session_id, exc_info=True)
        name = ""
    if name:
        return name
    basename = Path(cwd).name if cwd else ""
    return f"{session_id[:8]} ({basename})" if basename else session_id[:8]


def _origin(session_dir: Path, read_origin) -> str:
    try:
        return read_origin(session_dir)
    except Exception:  # noqa: BLE001 — grouping metadata, never a reason to fail a listing
        return ""


def _as_int(value: Any) -> int | None:
    """A tolerant int read. Every field here comes from a file another process
    writes, and a hand-edited or half-written row must cost one field rather
    than the whole listing (a cast that raised would 500 the page)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
