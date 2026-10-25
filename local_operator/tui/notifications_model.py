"""The TUI Notifications page's data half: reads, bulk writes, undo, grouping.

NO TEXTUAL HERE, deliberately. Every rule that decides WHAT a gesture touches
lives in this module so it can be tested without an app, and the page widget
(``tui/widgets/notifications_view.py``) only paints and routes keys.

ONE STORE, ONE SET OF RULES. Every read and write goes through
``notifications.items`` — the same functions the desktop routes and the relay
call — so the terminal cannot describe the inbox differently from the other two
surfaces. In particular:

* **A bulk gesture never sweeps.** "Mark all read" and "dismiss all from this
  conversation" enumerate EVERY page of the requested scope first, freezing the
  first page's ``high_water`` as ``through_seq`` on every later page, and then
  write exactly the ``observed`` id/revision tokens that enumeration returned,
  in batches of at most :data:`BULK_BATCH` (the routes' cap). A revision written
  after the page was painted is therefore never acknowledged by a key the
  operator pressed before it existed (``docs/DESKTOP_API.md`` "All means the
  entire explicitly requested scope, not the first 50 rows").
* **An open ask is never dismissed from here.** The store refuses it
  (``skipped: live_ask``); the page reports how many stayed and offers Answer,
  which mounts the existing ask picker. There is no second answer UI.
* **Undo is the inverse op over the same tokens** — ``dismiss`` ↔ ``undismiss``,
  ``read`` ↔ ``unread`` — applied to exactly the ids the original write
  reported as ``changed``. A token whose revision moved since is skipped as
  stale by the store, which is the honest outcome: the operator undoes what
  they did, not whatever arrived afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from local_operator.notifications import items as inbox

#: The routes' per-request token cap (``PATCH /v1/desktop/notifications``); the
#: page writes in batches no larger, so one bulk gesture never depends on a
#: limit only the HTTP layer enforces.
BULK_BATCH = 500

#: How long the footer offers ``z undo`` after a write (design §8.4).
UNDO_WINDOW_S = 6.0

#: Read filters cycled by ``f``; kind filters by ``t``; priority by ``p``.
READ_FILTERS = ("unread", "all")
KIND_FILTERS = ("all", "asks", "updates")
PRIORITY_FILTERS = ("any", "high", "normal_up")
GROUPINGS = ("time", "conversation")


@dataclass(frozen=True)
class Filters:
    """The page's view state. Defaults are the design's: Unread, all kinds, any."""

    read: str = "unread"
    kind: str = "all"
    priority: str = "any"
    grouping: str = "time"
    #: ``/notifications here`` presets this; empty means every conversation.
    session_id: str = ""

    def cycled(self, attribute: str, choices: Sequence[str]) -> "Filters":
        current = getattr(self, attribute)
        index = choices.index(current) if current in choices else -1
        return _replace(self, **{attribute: choices[(index + 1) % len(choices)]})


def _replace(filters: Filters, **changes: Any) -> Filters:
    values = {name: getattr(filters, name) for name in Filters.__dataclass_fields__}
    values.update(changes)
    return Filters(**values)


@dataclass
class Snapshot:
    """One painted state of the page: rows plus the tokens that authorise writes."""

    waiting: list[dict[str, Any]] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    unread_count: int = 0
    waiting_count: int = 0
    waiting_unread_count: int = 0
    high_water: int = 0
    #: True when ``items`` is the whole requested scope; False when the reel has
    #: more pages than the page paints (the header then says "first N").
    complete: bool = True
    error: str = ""

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Every selectable row, in paint order: open asks first (§2.1)."""
        return list(self.waiting) + list(self.items)


@dataclass(frozen=True)
class WriteResult:
    """What a write did, and what an undo of it must touch."""

    op: str
    changed: tuple[str, ...]
    #: ``{id: revision}`` for every changed row, captured from the tokens the
    #: write was authorised by — the undo's own authority.
    tokens: tuple[tuple[str, int], ...]
    skipped_live_asks: int = 0
    skipped_stale: int = 0
    unread_count: int = 0
    error: str = ""

    @property
    def inverse(self) -> str:
        return {"dismiss": "undismiss", "undismiss": "dismiss", "read": "unread", "unread": "read"}[
            self.op
        ]


def _query_kwargs(filters: Filters) -> dict[str, Any]:
    source = {"asks": "ask", "updates": "notify"}.get(filters.kind)
    priority = "high" if filters.priority == "high" else None
    return {
        "state": filters.read,
        "source": source,
        "priority": priority,
        "session_id": filters.session_id or None,
    }


def _keep_priority(row: dict[str, Any], filters: Filters) -> bool:
    # ``normal_up`` hides only low; the store's filter is exact-match, so the
    # "normal and above" arm is applied here, over the rows it returned.
    return not (filters.priority == "normal_up" and row.get("priority") == "low")


def load(
    config_dir: Path | str,
    filters: Filters,
    *,
    limit: int = inbox.LIST_LIMIT_MAX,
    lapsed_visible_ms: int | None = None,
) -> Snapshot:
    """The first page of the requested scope, plus the waiting group.

    The page paints at most ``limit`` updates; bulk gestures do not trust that
    page and enumerate the whole scope themselves (:func:`enumerate_scope`).
    """
    page = inbox.list_page(
        config_dir, limit=limit, lapsed_visible_ms=lapsed_visible_ms, **_query_kwargs(filters)
    )
    # Pending is not unread: the store keeps every live ask in ``waiting`` on
    # the unread tab, and so does the page — a read ask stays pinned and is
    # painted read (design §11).
    waiting = [row for row in page.get("waiting") or [] if _keep_priority(row, filters)]
    items = [row for row in page.get("items") or [] if _keep_priority(row, filters)]
    return Snapshot(
        waiting=waiting if filters.kind != "updates" else [],
        items=items if filters.kind != "asks" else [],
        unread_count=int(page.get("unread_count") or 0),
        waiting_count=int(page.get("waiting_count") or 0),
        waiting_unread_count=int(page.get("waiting_unread_count") or 0),
        high_water=int(page.get("high_water") or 0),
        complete=not page.get("next_cursor"),
    )


def enumerate_scope(
    config_dir: Path | str,
    filters: Filters,
    *,
    session_id: str = "",
    lapsed_visible_ms: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Every observed token in the scope, across every page, under ONE boundary.

    Returns ``(through_seq, observed, rows)``. The FIRST response's
    ``high_water`` is frozen and passed as ``through_seq`` on every later page,
    and the first page's ask refs are kept as they were (a later page carries no
    ``waiting``), so an ask queued mid-enumeration is never in the set.
    ``session_id`` narrows to one conversation ("dismiss all from this
    conversation"); the page's own filters still apply, so a filtered gesture
    never silently widens.
    """
    kwargs = _query_kwargs(filters)
    if session_id:
        kwargs["session_id"] = session_id
    observed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    through: int | None = None
    while True:
        page = inbox.list_page(
            config_dir,
            limit=inbox.LIST_LIMIT_MAX,
            cursor=cursor,
            through_seq=through,
            lapsed_visible_ms=lapsed_visible_ms,
            **kwargs,
        )
        if through is None:
            through = int(page.get("high_water") or 0)
        page_rows = [
            row
            for row in list(page.get("waiting") or []) + list(page.get("items") or [])
            if _keep_priority(row, filters)
            and (filters.kind != "asks" or row.get("source") == "ask")
            and (filters.kind != "updates" or row.get("source") == "notify")
        ]
        keep = {row["id"] for row in page_rows}
        observed.extend(
            {"id": str(token["id"]), "revision": int(token["revision"])}
            for token in page.get("observed") or []
            if token.get("id") in keep
        )
        rows.extend(page_rows)
        if progress is not None:
            progress(len(observed))
        cursor = page.get("next_cursor")
        if not cursor:
            return through, observed, rows


def write(
    config_dir: Path | str,
    op: str,
    *,
    ids: Sequence[Mapping[str, Any]] | None = None,
    scope: tuple[int, Sequence[Mapping[str, Any]]] | None = None,
    session_id: str = "",
) -> WriteResult:
    """Apply ``op`` to explicit tokens (``ids``) or an enumerated ``scope``.

    ``scope`` is ``(through_seq, observed)`` from :func:`enumerate_scope`; it is
    written in batches of :data:`BULK_BATCH`, each carrying the SAME frozen
    boundary. A store error ends the gesture and is reported, never retried
    behind the operator's back — a partial bulk write is stated, not hidden.
    """
    tokens: dict[str, int] = {}
    changed: list[str] = []
    live = stale = 0
    unread = 0
    try:
        if ids is not None:
            tokens = {str(t["id"]): int(t["revision"]) for t in ids}
            answer = inbox.apply_patch(config_dir, op, ids=list(ids))
            changed.extend(answer["changed"])
            live += sum(1 for s in answer["skipped"] if s.get("reason") == "live_ask")
            stale += sum(1 for s in answer["skipped"] if s.get("reason") == "stale_revision")
            unread = int(answer.get("unread_count") or 0)
        elif scope is not None:
            through, observed = scope
            tokens = {str(t["id"]): int(t["revision"]) for t in observed}
            for start in range(0, len(observed), BULK_BATCH):
                batch = list(observed[start : start + BULK_BATCH])
                answer = inbox.apply_patch(
                    config_dir,
                    op,
                    all_items=not session_id,
                    session_id=session_id or None,
                    through_seq=through,
                    observed=batch,
                )
                changed.extend(answer["changed"])
                live += sum(1 for s in answer["skipped"] if s.get("reason") == "live_ask")
                stale += sum(1 for s in answer["skipped"] if s.get("reason") == "stale_revision")
                unread = int(answer.get("unread_count") or 0)
    except Exception as error:  # noqa: BLE001 — the page reports, never crashes
        return WriteResult(
            op=op,
            changed=tuple(changed),
            tokens=tuple((i, tokens[i]) for i in changed if i in tokens),
            skipped_live_asks=live,
            skipped_stale=stale,
            unread_count=unread,
            error=str(error) or type(error).__name__,
        )
    return WriteResult(
        op=op,
        changed=tuple(changed),
        tokens=tuple((i, tokens[i]) for i in changed if i in tokens),
        skipped_live_asks=live,
        skipped_stale=stale,
        unread_count=unread,
    )


def undo(config_dir: Path | str, result: WriteResult) -> WriteResult:
    """The inverse op over exactly the tokens ``result`` changed."""
    if not result.tokens:
        return WriteResult(op=result.inverse, changed=(), tokens=())
    return write(
        config_dir,
        result.inverse,
        ids=[{"id": item_id, "revision": revision} for item_id, revision in result.tokens],
    )


@dataclass(frozen=True)
class Group:
    """One section header and its rows. ``key`` identifies it across repaints."""

    key: str
    label: str
    rows: tuple[dict[str, Any], ...]


def day_bucket(at_ms: int, now: datetime | None = None) -> str:
    """``today`` / ``yesterday`` / ``week`` / ``older`` in local time (design §3)."""
    now = now or datetime.now()
    when = datetime.fromtimestamp(max(0, at_ms) / 1000)
    days = (now.date() - when.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return "week"
    return "older"


def group(
    snapshot: Snapshot,
    filters: Filters,
    labels: Callable[[str, str], str],
    *,
    now: datetime | None = None,
) -> list[Group]:
    """Waiting first, then updates by day (time) or by conversation.

    ``labels(kind, value)`` names a group in the page's copy, so this module
    holds no English; ``kind`` is ``waiting`` / ``day`` / ``conversation``.
    """
    groups: list[Group] = []
    if snapshot.waiting:
        groups.append(Group("waiting", labels("waiting", ""), tuple(snapshot.waiting)))
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in snapshot.items:
        if filters.grouping == "conversation":
            session = row.get("session") or {}
            key = "c:" + str(session.get("id") or "")
        else:
            key = "d:" + day_bucket(int(row.get("updated_at") or 0), now)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    for key in order:
        rows = buckets[key]
        if key.startswith("c:"):
            label = labels("conversation", source_name(rows[0]))
        else:
            label = labels("day", key[2:])
        groups.append(Group(key, label, tuple(rows)))
    return groups


def source_name(row: dict[str, Any]) -> str:
    """The conversation's title, or its id when it has none (never empty)."""
    session = row.get("session") or {}
    return str(session.get("title") or session.get("agent_id") or session.get("id") or "")


def count_in_session(rows: Iterable[dict[str, Any]], session_id: str) -> int:
    return sum(1 for row in rows if (row.get("session") or {}).get("id") == session_id)


def monotonic() -> float:
    """Indirection for the undo window, so tests can drive it without sleeping."""
    return time.monotonic()
