"""What the inbox shows: ``notify`` items plus the open asks agents are blocked on.

``docs/design/notifications-inbox.md`` §2. Every surface (desktop routes, and
later the relay and the TUI page) reads through these functions, so one item
has one shape everywhere (§2.2) and one set of rules for read and dismiss
(§2.3).

TWO SOURCES, NEVER COPIED. ``notify`` rows live in ``notifications.db``; open
asks stay in the ask queue's own derived index (``asks/store.py``) and are read
from it at request time. The store keeps only a reel-side READ mark for an ask
(``ref_state``), never a copy of the ask, because the queue owns its record:
settled asks leave the reel entirely (decision D1) and appear only in an item's
detail under "this conversation's asks".

AN OPEN ASK IS NEVER DISMISSED THROUGH THE REEL. A reel dismiss on an ask id is
refused with ``skipped: live_ask``; settling an ask goes through the existing
reasoned "Dismiss ask…" doors, which tell the agent (§2.3). Nothing in this
module writes the ask log.

Import-light by design: the asks store and the project registry are imported
inside the functions that need them, so the desktop feed's count read never
pays for the project registry.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.notifications.store import (
    ASK_REF_PREFIX,
    NotificationStore,
    ask_ref,
    excerpt,
    parse_ask_ref,
)

logger = logging.getLogger(__name__)

#: An open ask's reel priority. It is the agent blocked on the operator, which
#: is the highest-priority thing the inbox can hold (§2.2).
ASK_PRIORITY = "high"

LIST_LIMIT_MAX = 100
LIST_LIMIT_DEFAULT = 50


def open_asks(
    config_dir: Path | str, *, lapsed_visible_ms: int | None = None
) -> list[dict[str, Any]]:
    """Every OPEN ask across sessions, newest first, from the derived index.

    Only ``OUTSTANDING_STATUSES`` (open): a settled ask is not pending (D1).
    The index re-derives a past-deadline ``open`` as ``timed_out``, so a
    stopped runtime's stale row never reads as waiting here either.
    """
    from local_operator.asks import store as ask_store

    kwargs: dict[str, Any] = {}
    if lapsed_visible_ms is not None:
        kwargs["lapsed_visible_ms"] = lapsed_visible_ms
    rows = ask_store.index_asks(config_dir, **kwargs)
    return [row for row in rows if ask_store.is_outstanding(row.get("status"))]


def _ask_dismissible(row: Mapping[str, Any]) -> bool:
    """Whether the reasoned "Dismiss ask…" flow may be offered for this ask.

    Calls the queue's own verdict (``asks/render.py`` ``dismiss_refusal``): a
    relayed or Team-Bridge-mirrored open ask is refused there, so it is refused
    here, and no second rule is written.
    """
    try:
        from local_operator.asks.render import dismiss_refusal

        return dismiss_refusal(row) == ""
    except Exception:  # noqa: BLE001 — unknown is "do not offer"
        logger.debug("notifications: dismiss verdict unavailable", exc_info=True)
        return False


def _first_question(row: Mapping[str, Any]) -> str:
    for question in row.get("questions") or ():
        if isinstance(question, Mapping):
            text = str(question.get("question") or question.get("header") or "").strip()
            if text:
                return text
    return ""


class SessionFacts:
    """Session title / live / agent facts, read once per session per request.

    Titles prefer the LIVE stored title (a rename after the item was raised
    should show) and fall back to the snapshot the item recorded. ``live`` is
    the liveness marker probe the cold ask door uses (``resume.live_runtime_pid``),
    which reads one small file and never dials a runtime.
    """

    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def get(
        self, session_id: str, *, fallback_title: str = "", agent_id: str = ""
    ) -> dict[str, Any]:
        cached = self._cache.get(session_id)
        if cached is None:
            title = ""
            live = False
            try:
                from local_operator.resume import live_runtime_pid, stored_session_title

                directory = self.config_dir / "sessions" / session_id
                if directory.is_dir():
                    title = stored_session_title(directory)
                live = live_runtime_pid(self.config_dir, session_id) is not None
            except Exception:  # noqa: BLE001 — chrome; the snapshot title stands in
                logger.debug("notifications: session facts unavailable", exc_info=True)
            cached = {"title": title, "live": live}
            self._cache[session_id] = cached
        return {
            "id": session_id,
            "title": cached["title"] or fallback_title,
            "agent_id": agent_id,
            "live": bool(cached["live"]),
        }


class ProjectFacts:
    """``projects_for_session`` over ONE registry snapshot per request (§2.2)."""

    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)
        self._registry: Any = None
        self._failed = False
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def get(self, session_id: str) -> list[dict[str, Any]]:
        if session_id in self._cache:
            return self._cache[session_id]
        rows: list[dict[str, Any]] = []
        if not self._failed:
            try:
                if self._registry is None:
                    from local_operator.projects import ProjectRegistry

                    self._registry = ProjectRegistry(self.config_dir)
                for project in self._registry.projects_for_session(session_id):
                    rows.append(
                        {
                            "key": project.name,
                            "name": project.title or project.name,
                            "status": project.status,
                        }
                    )
            except Exception:  # noqa: BLE001 — related projects are context, not the item
                self._failed = True
                logger.debug("notifications: project registry unavailable", exc_info=True)
        self._cache[session_id] = rows
        return rows


def notify_item(
    row: Mapping[str, Any],
    sessions: SessionFacts,
    projects: ProjectFacts,
) -> dict[str, Any]:
    """One ``notify`` row in the normalized §2.2 shape."""
    session_id = str(row.get("session_id") or "")
    read = bool(row.get("read"))
    dismissed = bool(row.get("dismissed"))
    return {
        "id": str(row["id"]),
        "source": "notify",
        "priority": str(row.get("priority") or "normal"),
        "title": str(row.get("title") or ""),
        "excerpt": excerpt(str(row.get("body_md") or "")),
        "created_at": int(row.get("created_at") or 0),
        "updated_at": int(row.get("updated_at") or 0),
        "seq": int(row.get("seq") or 0),
        "revision": int(row.get("revision") or 1),
        "read": read,
        "read_via": row.get("read_via") if read else None,
        # WHICH surface produced an ``in_view`` read (D11). The desktop, the TUI
        # and the phone all record ``in_view``; this is the provenance that says
        # which of them the evidence came from.
        "read_surface": row.get("read_surface") if read else None,
        "dismissed": dismissed,
        "repeat_count": int(row.get("repeat_count") or 1),
        "key": str(row.get("key") or ""),
        "anchor_id": str(row.get("anchor_id") or ""),
        "session": sessions.get(
            session_id,
            fallback_title=str(row.get("session_title") or ""),
            agent_id=str(row.get("agent_id") or ""),
        ),
        "projects": projects.get(session_id),
        "ask": None,
        "actions": {"dismiss": not dismissed, "answer": False, "read": True},
    }


def ask_item(
    row: Mapping[str, Any],
    marks: Mapping[str, Mapping[str, Any]],
    sessions: SessionFacts,
    projects: ProjectFacts,
) -> dict[str, Any]:
    """One OPEN ask in the normalized shape. ``revision`` is always 1 (§2.2)."""
    session_id = str(row.get("session_id") or "")
    ask_id = str(row.get("ask_id") or "")
    ref = ask_ref(session_id, ask_id)
    mark = marks.get(ref) or {}
    read = mark.get("read_at") is not None
    created = int(row.get("created_at") or 0)
    questions = [q for q in (row.get("questions") or ()) if isinstance(q, Mapping)]
    return {
        "id": ref,
        "source": "ask",
        "priority": ASK_PRIORITY,
        "title": excerpt(_first_question(row), 120),
        "excerpt": excerpt(_first_question(row)),
        "created_at": created,
        "updated_at": created,
        "seq": int(mark.get("seq") or 0),
        "revision": 1,
        "read": read,
        "read_via": "explicit" if read else None,
        "read_surface": None,
        "dismissed": False,
        "repeat_count": 1,
        "session": sessions.get(session_id),
        "projects": projects.get(session_id),
        "ask": {
            "ask_id": ask_id,
            "status": str(row.get("status") or "open"),
            "deadline_at": int(row.get("expires_at") or 0),
            "question_count": len(questions),
            "dismissible": _ask_dismissible(row),
        },
        # The reel never dismisses an open ask (§2.3); it offers Answer.
        "actions": {"dismiss": False, "answer": True, "read": True},
    }


def waiting_items(
    config_dir: Path | str,
    store: NotificationStore,
    sessions: SessionFacts,
    projects: ProjectFacts,
    *,
    session_id: str | None = None,
    lapsed_visible_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Every open ask as an item, newest first (not paged: §2.1)."""
    rows = open_asks(config_dir, lapsed_visible_ms=lapsed_visible_ms)
    if session_id:
        rows = [row for row in rows if str(row.get("session_id") or "") == session_id]
    refs = [ask_ref(str(r.get("session_id") or ""), str(r.get("ask_id") or "")) for r in rows]
    marks = store.ref_states(refs)
    return [ask_item(row, marks, sessions, projects) for row in rows]


def _waiting_counts(rows: Sequence[Mapping[str, Any]], store: NotificationStore) -> tuple[int, int]:
    refs = [ask_ref(str(r.get("session_id") or ""), str(r.get("ask_id") or "")) for r in rows]
    marks = store.ref_states(refs)
    return len(refs), sum(1 for ref in refs if (marks.get(ref) or {}).get("read_at") is None)


def unread_waiting_count(config_dir: Path | str, store: NotificationStore) -> int:
    return _waiting_counts(open_asks(config_dir), store)[1]


def counts(config_dir: Path | str, store: NotificationStore | None = None) -> dict[str, int]:
    """Total pending asks and their unread subset have different meanings.

    The group keeps every pending ask; the badge sums unread_count and
    waiting_unread_count. Reading an ask never settles it.
    """
    store = store or NotificationStore(config_dir)
    unread, high_water = store.counts()
    waiting, waiting_unread = _waiting_counts(open_asks(config_dir), store)
    return {
        "unread_count": unread,
        "waiting_count": waiting,
        "waiting_unread_count": waiting_unread,
        "high_water": high_water,
    }


def cursor_in_range(value: str | None) -> str | None:
    """Refuse a cursor whose stamp cannot be an epoch stamp (QA round 2).

    Length was bounded and magnitude was not, so ``9`` x 79 digits passed the
    wire and died in sqlite. The rule is the store's own: a stamp is a
    non-negative integer inside int64 (``store.CURSOR_STAMP_MAX``). A
    NON-NUMERIC cursor is the documented "from the top" case and stays
    accepted; only a well-formed stamp that cannot exist is refused.
    """
    from local_operator.notifications.store import CURSOR_STAMP_MAX

    if not value:
        return value
    stamp, _, _ = str(value).partition(".")
    try:
        magnitude = int(stamp)
    except ValueError:
        return value
    if not 0 <= magnitude <= CURSOR_STAMP_MAX:
        raise ValueError("cursor stamp is out of range")
    return value


def list_page(
    config_dir: Path | str,
    *,
    state: str = "unread",
    source: str | None = None,
    priority: str | None = None,
    session_id: str | None = None,
    cursor: str | None = None,
    limit: int = LIST_LIMIT_DEFAULT,
    lapsed_visible_ms: int | None = None,
    through_seq: int | None = None,
) -> dict[str, Any]:
    """``GET /v1/desktop/notifications`` (§5.2), as a plain dict."""
    store = NotificationStore(config_dir)
    sessions = SessionFacts(config_dir)
    projects = ProjectFacts(config_dir)
    limit = max(1, min(int(limit), LIST_LIMIT_MAX))
    items: list[dict[str, Any]] = []
    next_cursor = None
    # Capture BEFORE rows: concurrent changes must never enlarge what this
    # response authorizes. The explicit observed set also bounds filters/pages.
    unread, high_water = store.counts()
    if through_seq is not None:
        high_water = min(high_water, through_seq)
    if source in (None, "", "notify"):
        rows, next_cursor = store.list_items(
            state=state,
            priority=priority,
            session_id=session_id,
            cursor=cursor,
            limit=limit,
            through_seq=high_water,
        )
        items = [notify_item(row, sessions, projects) for row in rows]
    waiting_count, waiting_unread = _waiting_counts(
        open_asks(config_dir, lapsed_visible_ms=lapsed_visible_ms),
        store,
    )
    answer: dict[str, Any] = {
        "items": items,
        "next_cursor": next_cursor,
        "unread_count": unread,
        # Counted from the UNFILTERED open set, so a filter on this page never
        # shrinks the badge figure the same answer carries.
        "waiting_count": waiting_count,
        "waiting_unread_count": waiting_unread,
        "high_water": high_water,
    }
    # ``waiting`` rides the FIRST page only (no cursor): asks are not paged, so
    # a second page repeating them would double-render the pinned group. Not
    # under ``state=dismissed`` (an open ask cannot be dismissed through the
    # reel) nor ``source=notify``; a priority filter below high excludes them,
    # because an open ask always reports high.
    if not cursor and source in (None, "", "ask") and state != "dismissed":
        waiting = waiting_items(
            config_dir,
            store,
            sessions,
            projects,
            session_id=session_id,
            lapsed_visible_ms=lapsed_visible_ms,
        )
        if priority and priority != ASK_PRIORITY:
            waiting = []
        # Pending is not unread: even on the unread-updates tab the waiting
        # group keeps every live ask. Reading changes its badge/row treatment,
        # never the fact that an agent is waiting for the operator.
        answer["waiting"] = waiting
    answer["observed"] = [
        {"id": item["id"], "revision": item["revision"]}
        for item in items + answer.get("waiting", [])
    ]
    return answer


def detail(
    config_dir: Path | str,
    item_id: str,
    *,
    lapsed_visible_ms: int | None = None,
    anchor_excerpt: Any = None,
) -> dict[str, Any] | None:
    """``GET /v1/desktop/notifications/{id}`` (§5.2), or ``None`` when unknown.

    ``anchor_excerpt`` is a callable ``(session_dir, anchor_id) -> str`` the
    route supplies (it reads the transcript); absent or failing means
    ``anchor: null`` — honest absence rather than a guessed line.
    """
    from local_operator.asks import store as ask_store

    store = NotificationStore(config_dir)
    sessions = SessionFacts(config_dir)
    projects = ProjectFacts(config_dir)
    kwargs: dict[str, Any] = {}
    if lapsed_visible_ms is not None:
        kwargs["lapsed_visible_ms"] = lapsed_visible_ms
    parsed = parse_ask_ref(item_id)
    if parsed is not None:
        session_id, ask_id = parsed
        rows = [
            row
            for row in ask_store.session_asks(config_dir, session_id, **kwargs)
            if str(row.get("ask_id") or "") == ask_id
            and ask_store.is_outstanding(row.get("status"))
        ]
        if not rows:
            return None
        marks = store.ref_states([item_id])
        item = ask_item(rows[0], marks, sessions, projects)
        history: list[dict[str, Any]] = []
        anchor = None
    else:
        row = store.get(item_id)
        if row is None:
            return None
        item = notify_item(row, sessions, projects)
        session_id = str(row.get("session_id") or "")
        history = store.history(item_id)
        anchor = None
        anchor_id = str(row.get("anchor_id") or "")
        if anchor_id and callable(anchor_excerpt):
            try:
                text = anchor_excerpt(Path(config_dir) / "sessions" / session_id, anchor_id)
            except Exception:  # noqa: BLE001 — honest absence
                text = ""
            if text:
                anchor = {"id": anchor_id, "excerpt": str(text)[:600]}
        item["body_md"] = str(row.get("body_md") or "")
        item["links"] = list(row.get("links") or [])
    return {
        "item": item,
        "history": history,
        "session": {**item["session"], "cwd": _session_cwd(config_dir, session_id)},
        "anchor": anchor,
        "projects": item["projects"],
        "asks": ask_store.session_asks(config_dir, session_id, **kwargs),
    }


def _session_cwd(config_dir: Path | str, session_id: str) -> str:
    try:
        from local_operator.asks import store as ask_store

        entry = ask_store.read_entry(config_dir, session_id)
        if entry and entry.get("cwd"):
            return str(entry["cwd"])
    except Exception:  # noqa: BLE001
        pass
    return ""


def apply_patch(
    config_dir: Path | str,
    op: str,
    *,
    ids: Sequence[Mapping[str, Any]] | None = None,
    session_id: str | None = None,
    through_seq: int | None = None,
    all_items: bool = False,
    observed: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply only explicitly observed revisions, across both source stores.

    Bulk targets echo the list's ``observed`` tokens. A scalar SQLite watermark
    cannot identify an ask (which lives in another store), nor encode a page or
    filter. Never repair missing tokens by discovering the current ask set:
    that would silently include a new, unseen ask. ``session_id`` only narrows.
    """
    store = NotificationStore(config_dir)
    changed: list[str] = []
    skipped: list[dict[str, str]] = []
    notify_ids: list[tuple[str, int]] | None = None
    bulk = ids is None
    if bulk:
        if observed is None or through_seq is None:
            raise ValueError("bulk writes require observed tokens and through_seq")
        # ONE READ FOR THE WHOLE BATCH (CD6): the per-token ``store.get`` this
        # replaces opened a connection per observed entry, up to 500 of them.
        owners = store.session_ids(
            [
                str(entry.get("id") or "")
                for entry in observed
                if not str(entry.get("id") or "").startswith(ASK_REF_PREFIX)
            ]
        )
        bounded = []
        for entry in observed:
            item_id = str(entry.get("id") or "")
            if item_id.startswith(ASK_REF_PREFIX):
                belongs = item_id.split(":", 2)[1]
            else:
                belongs = owners.get(item_id, "")
            if all_items or belongs == session_id:
                bounded.append(entry)
        ids = bounded
    if ids is not None:
        notify_ids = []
        ask_refs: list[str] = []
        open_refs = {
            ask_ref(str(r.get("session_id") or ""), str(r.get("ask_id") or ""))
            for r in open_asks(config_dir)
        }
        for entry in ids:
            item_id = str(entry.get("id") or "")
            if item_id.startswith(ASK_REF_PREFIX):
                if op in ("dismiss", "undismiss"):
                    skipped.append({"id": item_id, "reason": "live_ask"})
                elif item_id not in open_refs:
                    skipped.append({"id": item_id, "reason": "unknown"})
                elif int(entry.get("revision") or 0) != 1:
                    skipped.append({"id": item_id, "reason": "stale_revision"})
                else:
                    ask_refs.append(item_id)
            else:
                notify_ids.append((item_id, int(entry.get("revision") or 0)))
        if ask_refs:
            moved = store.mark_refs(ask_refs, read=op == "read")
            changed.extend(moved)
            skipped.extend(
                {"id": ref, "reason": "already"} for ref in ask_refs if ref not in set(moved)
            )
    if notify_ids:
        result = store.apply(
            op,
            ids=notify_ids,
            through_seq=through_seq if bulk else None,
        )
        changed.extend(result.changed)
        skipped.extend(result.skipped)
    badge = counts(config_dir, store)
    return {
        **badge,
        "changed": changed,
        "skipped": skipped,
    }


def apply_seen(
    config_dir: Path | str,
    items: Iterable[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """``POST /v1/desktop/notifications/seen`` (§6.4): per item, evidence re-checked."""
    store = NotificationStore(config_dir)
    result = store.seen(
        list(items),
        session_id=str(evidence.get("session_id") or ""),
        input_at_ms=int(evidence.get("input_at_ms") or 0),
        visible_since_ms=int(evidence.get("visible_since_ms") or 0),
        surface=str(evidence.get("surface") or ""),
    )
    badge = counts(config_dir, store)
    return {
        **badge,
        "changed": result.changed,
        "skipped": result.skipped,
    }
