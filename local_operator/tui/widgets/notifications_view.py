"""The full-page ``/notifications`` view: open asks and agent updates in one reel.

WHAT IT IS (``docs/design/notifications-inbox.md`` §8; design §8.2-§8.4). A
sibling MODE of ``/schedule`` and ``/library``: it takes the transcript's region,
the dock stays greyed, and the page owns the keyboard until ``esc``. Rows come
from the store (``notifications.items``) through
:mod:`local_operator.tui.notifications_model`, cells from
:mod:`local_operator.tui.notifications_render`; this widget holds state and
routes keys, and decides nothing about what a gesture touches.

LAYOUT. Under :data:`WIDE_COLUMNS` the detail REPLACES the list (``↵`` in, ``esc``
out, the cursor kept). At or above it the page splits 55/45 with an ``edge``
rule and the detail follows the cursor after :data:`DETAIL_DEBOUNCE_S`, so
holding ``↓`` does not read a detail per row.

READ SEMANTICS. Opening the page never marks anything read and never zeroes the
badge (design §11: "a list to work through, not a pile to clear on sight").
Opening one item's detail with ``↵`` is the explicit gesture the design names
and marks THAT revision read, by its observed token. The wide split's
follow-the-cursor preview does not: moving past a row is not reading it. Bulk
reads (``A``, ``/notifications read``) and bulk dismisses (``E``) always ask
first on one ``y/n`` line; ``z`` is the page's own inverse for six seconds.

ASKS ARE ANSWERED, NOT DISMISSED, HERE. ``e`` on an open ask says so, and ``a``
posts :class:`NotificationsViewAnswerRequested`; the app answers through its
EXISTING fleet ask picker. There is no second answer UI.

WRITES ARE OFF THE EVENT LOOP. Every store call runs in a worker thread: an
"all" enumeration walks every page of the reel, and a SQLite hop on a keystroke
is latency the reader feels. One write is in flight at a time; a second gesture
while it runs is ignored rather than queued behind the operator's back.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static

from local_operator.i18n import keys as i18n_keys
from local_operator.paths import config_dir
from local_operator.tui import notifications_model as model
from local_operator.tui import notifications_render as render
from local_operator.tui.i18n_status import display_tr

_t = display_tr(
    "tui.notifications",
    # The literal /commands ride as islands (see model.SHELL_COMMANDS):
    # per-string isolation alone reorders them in Urdu.
    technical_literals=model.SHELL_COMMANDS,
    technical_commands=model.SHELL_COMMAND_TEMPLATES,
)
_k = i18n_keys.tui_notifications

#: The split threshold (design §8.3). Below it the detail replaces the list.
WIDE_COLUMNS = 120
#: How long the wide detail waits for the cursor to settle before reading.
DETAIL_DEBOUNCE_S = 0.15
#: How often an open page looks for new rows. The store file's stat is the
#: doorbell, so an idle page pays one ``stat`` per tick and no SQL.
POLL_S = 2.0


class NotificationsViewDismissed(Message):
    """``esc`` at the page's top level: leave the mode."""


class NotificationsViewAnswerRequested(Message):
    """``a``: answer an open ask through the app's existing ask picker."""

    def __init__(self, session_id: str, ask_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.ask_id = ask_id


class NotificationsViewOpenConversation(Message):
    """``o``: switch to the item's conversation (the page closes first)."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id


class NotificationsViewCountsChanged(Message):
    """The badge figures moved; the app repaints the sidebar note from them."""

    def __init__(self, counts: dict[str, int]) -> None:
        super().__init__()
        self.counts = counts


class _Body(ScrollableContainer):
    """The scrolling list (or the narrow detail). Not focusable: the page is."""

    can_focus = False


def _store_stamp(root: Path) -> tuple[int, ...]:
    """The inbox doorbell: ``notifications.db`` and its WAL, size and mtime.

    Every write (an agent's ``notify``, another surface's read or dismiss)
    lands in the WAL first, so the pair moves on every change the page could
    show.

    THE ASK INDEX IS PART OF THE DOORBELL (T7). Open asks live in
    ``asks/<session>.json``, not in the inbox store, so an ask raised while the
    page is open used to stay invisible until ``r`` — and an ask is the one
    category that BLOCKS an agent, so it is the worst thing to miss. The index
    directory's own stat is one syscall, and every write replaces a session file
    (mtime moves) inside it, so the directory stamp is the whole signal.
    """
    from local_operator.notifications.store import store_path

    base = store_path(root)
    stamp: list[int] = []
    for path in (base, base.with_name(base.name + "-wal")):
        try:
            info = path.stat()
            stamp.extend((info.st_size, info.st_mtime_ns))
        except OSError:
            stamp.extend((0, 0))
    try:
        index = (root / "asks").stat()
        stamp.extend((index.st_size, index.st_mtime_ns))
    except OSError:
        stamp.extend((0, 0))
    return (stamp[0], stamp[1], stamp[2], stamp[3], stamp[4], stamp[5])


class NotificationsView(Vertical):
    """Open asks and agent updates across every conversation on this machine."""

    can_focus = True

    BINDINGS = [
        Binding("up,k", "move(-1)", "Up", show=False),
        Binding("down,j", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("enter", "open", "Open", show=False),
        Binding("space", "toggle_select", "Select", show=False),
        Binding("e", "dismiss", "Dismiss", show=False),
        Binding("E", "dismiss_session", "Dismiss all here", show=False),
        Binding("u", "toggle_read", "Read", show=False),
        Binding("A", "mark_all", "Mark all read", show=False),
        Binding("z", "undo", "Undo", show=False),
        Binding("a", "answer", "Answer", show=False),
        Binding("o", "open_conversation", "Conversation", show=False),
        Binding("f", "cycle('read')", "Read filter", show=False),
        Binding("t", "cycle('kind')", "Kind", show=False),
        Binding("p", "cycle('priority')", "Priority", show=False),
        Binding("g", "cycle('grouping')", "Grouping", show=False),
        Binding("r", "reload", "Re-read", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        filters: model.Filters | None = None,
        now_ms: Callable[[], int] | None = None,
        confirm_mark_all: bool = False,
    ) -> None:
        super().__init__(classes="settings-view notifications-view")
        self._root = Path(root) if root is not None else config_dir()
        self.filters = filters or model.Filters()
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self._title = Static(classes="settings-view-title notifications-view-header")
        self._rule = Static(classes="settings-view-rule")
        self._list_text = Static(classes="notifications-view-text")
        self._list = _Body(self._list_text, classes="settings-view-body notifications-view-list")
        self._edge = Static(classes="notifications-view-edge")
        self._detail_text = Static(classes="notifications-view-text")
        self._detail = _Body(
            self._detail_text, classes="settings-view-body notifications-view-detail"
        )
        self._columns = Horizontal(
            self._list, self._edge, self._detail, classes="settings-view-columns"
        )
        self._notice_text = Static(classes="settings-view-detail notifications-view-notice")
        self._hints = Static(classes="settings-view-hints notifications-view-hints")
        #: The painted state. ``None`` until the first read lands, which is the
        #: page's loading state; an empty Snapshot is the caught-up state.
        self.snapshot: model.Snapshot | None = None
        self.cursor_id = ""
        self.selected_ids: set[str] = set()
        #: ``list`` or ``detail`` (narrow only; wide shows both).
        self.view_mode = "list"
        self._details: dict[tuple[str, int], dict[str, Any] | None] = {}
        #: The row the NARROW detail is showing, pinned at ``↵`` time. Opening an
        #: update marks it read, and on the Unread filter a read row leaves the
        #: list on the next read — so a detail that followed ``current`` jumped to
        #: whichever row slid into the cursor's place (measured on the first
        #: 80x24 detail capture: ``↵`` on an update painted the ask above it).
        self._detail_row: dict[str, Any] | None = None
        #: The cursor's index at the last paint, so a row that leaves the list
        #: hands the cursor to its successor instead of to the top (§14 "a
        #: dismissed row hands focus to the next one").
        self._cursor_index = 0
        self._detail_timer: Timer | None = None
        self._poll_timer: Timer | None = None
        self._stamp: tuple[int, int, int, int] | None = None
        #: ``(prompt, run)`` while a y/n confirm owns the keyboard.
        self._confirm: tuple[str, Callable[[], None]] | None = None
        self.notice = ""
        self._last_write: model.WriteResult | None = None
        self._last_write_at = 0.0
        self.busy = False
        #: ``/notifications read`` opens the page AND arms its confirm, so the
        #: typed command and the key are one gesture with one guard.
        self._arm_mark_all_on_load = confirm_mark_all

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        with self._columns:
            yield self._list
            yield self._edge
            yield self._detail
        yield self._notice_text
        yield self._hints

    def on_mount(self) -> None:
        self._apply_layout()
        self._paint()
        try:
            self.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass
        self.reload()
        self._poll_timer = self.set_interval(POLL_S, self._poll)

    def on_resize(self) -> None:
        self._apply_layout()
        self._paint()

    @property
    def wide(self) -> bool:
        return (self.size.width or 0) >= WIDE_COLUMNS

    def _apply_layout(self) -> None:
        wide = self.wide
        self._edge.display = wide
        self._list.display = wide or self.view_mode == "list"
        self._detail.display = wide or self.view_mode == "detail"
        self.set_class(wide, "-wide")

    # -- data -----------------------------------------------------------------
    def reload(self, *, keep_notice: bool = False) -> None:
        """Re-read the first page (and, wide, the cursor's detail) off-thread."""
        if not keep_notice:
            self.notice = ""
        self.run_worker(self._load(), group="notifications-load", exclusive=True)

    async def _load(self) -> None:
        filters = self.filters
        try:
            snapshot = await asyncio.to_thread(model.load, self._root, filters)
        except Exception as error:  # noqa: BLE001 - the page states the failure
            snapshot = model.Snapshot(error=str(error) or type(error).__name__)
        try:
            self._stamp = await asyncio.to_thread(_store_stamp, self._root)
        except Exception:  # noqa: BLE001
            self._stamp = None
        self._apply_snapshot(snapshot)
        if self._arm_mark_all_on_load:
            self._arm_mark_all_on_load = False
            self.action_mark_all()

    def _apply_snapshot(self, snapshot: model.Snapshot) -> None:
        self.snapshot = snapshot
        rows = snapshot.rows
        ids = [str(row["id"]) for row in rows]
        # The selection survives a re-read only for rows still listed: a row
        # that left (dismissed elsewhere, answered) cannot be acted on, and a
        # count that included it would promise more than the write can touch.
        self.selected_ids &= set(ids)
        if self.cursor_id not in ids:
            index = min(self._cursor_index, len(ids) - 1)
            self.cursor_id = ids[index] if ids else ""
        if self.view_mode == "detail" and self._detail_row is not None:
            # Refresh the pinned row from the new read when it is still listed
            # (its read mark, its revision); keep the pinned copy when it left.
            pinned = str(self._detail_row["id"])
            self._detail_row = next(
                (row for row in rows if str(row["id"]) == pinned), self._detail_row
            )
        self._paint()
        if not snapshot.error:
            self.post_message(
                NotificationsViewCountsChanged(
                    {
                        "unread_count": snapshot.unread_count,
                        "waiting_count": snapshot.waiting_count,
                        "waiting_unread_count": snapshot.waiting_unread_count,
                    }
                )
            )
        if self.wide:
            self._schedule_detail()

    async def _poll(self) -> None:
        if self.busy:
            return
        try:
            stamp = await asyncio.to_thread(_store_stamp, self._root)
        except Exception:  # noqa: BLE001
            return
        if stamp != self._stamp:
            self.reload(keep_notice=True)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.snapshot.rows if self.snapshot is not None else []

    @property
    def current(self) -> dict[str, Any] | None:
        return next((row for row in self.rows if str(row["id"]) == self.cursor_id), None)

    def _index(self) -> int:
        ids = [str(row["id"]) for row in self.rows]
        return ids.index(self.cursor_id) if self.cursor_id in ids else -1

    # -- detail ---------------------------------------------------------------
    def _schedule_detail(self) -> None:
        if self._detail_timer is not None:
            self._detail_timer.stop()
        self._detail_timer = self.set_timer(DETAIL_DEBOUNCE_S, self._load_detail)

    def _load_detail(self, *, mark_read: bool = False) -> None:
        row = self.detail_row
        if row is None:
            return
        self.run_worker(self._read_detail(row, mark_read=mark_read), group="notifications-detail")

    async def _read_detail(self, row: dict[str, Any], *, mark_read: bool) -> None:
        from local_operator.notifications import items as inbox

        token = (str(row["id"]), int(row.get("revision") or 1))
        if row.get("source") == "notify" and token not in self._details:
            try:
                payload = await asyncio.to_thread(inbox.detail, self._root, token[0])
            except Exception:  # noqa: BLE001 - the detail falls back to the row
                payload = None
            self._details[token] = payload
        if mark_read and not row.get("read"):
            # ONE revision, by the token the operator opened. A newer revision
            # written while they read stays unread (the store skips it stale).
            result = await self._write(
                "read", ids=[{"id": token[0], "revision": token[1]}], quiet=True
            )
            pinned = self._detail_row
            if (
                result is not None
                and token[0] in result.changed
                and pinned is not None
                and str(pinned["id"]) == token[0]
            ):
                # The pinned copy predates the write; on the Unread filter the
                # row has left the list, so no re-read will refresh it. Without
                # this the detail footer offered `u read` for a row it had just
                # marked read (measured on the final-head 80x24 detail frame).
                self._detail_row = {**pinned, "read": True}
                self._paint()
            return
        self._paint()

    # -- keys -----------------------------------------------------------------
    def _move_to(self, index: int) -> None:
        rows = self.rows
        if not rows:
            return
        index = max(0, min(len(rows) - 1, index))
        # A MOVE releases an opened row's pin: the operator is browsing again,
        # and the pane follows the cursor from here on (T3).
        self._detail_row = None
        self.cursor_id = str(rows[index]["id"])
        self._cursor_index = index
        self._paint()
        if self.wide:
            self._schedule_detail()

    def action_move(self, delta: int) -> None:
        if self._confirm is not None:
            return
        if self.view_mode == "detail":
            self._detail.scroll_relative(y=delta, animate=False)
            return
        self._move_to(self._index() + delta)

    def action_page(self, delta: int) -> None:
        if self._confirm is not None:
            return
        step = max(1, (self._list.size.height or 2) - 1)
        if self.view_mode == "detail":
            self._detail.scroll_relative(y=delta * step, animate=False)
            return
        self._move_to(self._index() + delta * step)

    def action_jump(self, edge: int) -> None:
        if self._confirm is not None or self.view_mode == "detail":
            return
        self._move_to(0 if edge == 0 else len(self.rows) - 1)

    def action_open(self) -> None:
        row = self.current
        if row is None or self._confirm is not None:
            return
        if row.get("source") == "ask":
            # ``↵`` on an ask is Answer (design §8.2 "a ↵" on the ask line).
            self.action_answer()
            return
        # T3: PIN THE OPENED ROW IN BOTH MODES. On the Unread filter, `↵` marks
        # the row read and it leaves the list, so the cursor lands on its
        # successor — and the wide pane, which followed the cursor, then showed
        # a DIFFERENT item than the one the operator opened. The narrow path was
        # fixed for exactly this; the wide path was not. The pin is released by
        # the cursor's next MOVE (``_move_to``), never by the re-read itself,
        # because the re-read is what the open caused.
        if not self.wide:
            self.view_mode = "detail"
        self._detail_row = row
        self._apply_layout()
        self._paint()
        self._load_detail(mark_read=True)

    def action_toggle_select(self) -> None:
        row = self.current
        if row is None or self._confirm is not None or self.view_mode == "detail":
            return
        row_id = str(row["id"])
        if row_id in self.selected_ids:
            self.selected_ids.discard(row_id)
        else:
            self.selected_ids.add(row_id)
        self._paint()

    def _targets(self) -> list[dict[str, Any]]:
        """The selection when there is one, else the row under the cursor."""
        if self.selected_ids:
            return [row for row in self.rows if str(row["id"]) in self.selected_ids]
        # In the narrow detail the key acts on what the operator is READING,
        # which is the pinned row, not whatever the list cursor now holds.
        row = self.detail_row
        return [row] if row is not None else []

    @staticmethod
    def _token(row: dict[str, Any]) -> dict[str, Any]:
        return {"id": str(row["id"]), "revision": int(row.get("revision") or 1)}

    def action_dismiss(self) -> None:
        if self.busy or self._confirm is not None:
            return
        targets = self._targets()
        if not targets:
            return
        if all(row.get("source") == "ask" for row in targets):
            # The store would refuse it anyway (``live_ask``); saying so here is
            # the difference between a key that does nothing and a key that
            # teaches the one that does.
            self.notice = _t(_k.ask_answer_hint())
            self._paint()
            return
        self._spawn_write("dismiss", ids=[self._token(row) for row in targets])

    def action_toggle_read(self) -> None:
        if self.busy or self._confirm is not None:
            return
        targets = self._targets()
        if not targets:
            return
        # One direction per gesture: if anything targeted is unread, the key
        # marks read; only an all-read target set flips back to unread.
        op = "read" if any(not row.get("read") for row in targets) else "unread"
        self._spawn_write(op, ids=[self._token(row) for row in targets])

    def action_dismiss_session(self) -> None:
        if self.busy or self._confirm is not None:
            return
        row = self.detail_row
        if row is None:
            return
        session = row.get("session") or {}
        session_id = str(session.get("id") or "")
        if not session_id:
            return
        self.busy = True
        self.run_worker(self._arm_dismiss_session(session_id, model.source_name(row)))

    async def _arm_dismiss_session(self, session_id: str, source: str) -> None:
        # The count in the question is the ENUMERATED scope, not this page's
        # first page: the confirm must name what ``y`` will actually touch.
        filters = model._replace(self.filters, kind="updates", read="all")
        try:
            through, observed, _rows = await asyncio.to_thread(
                model.enumerate_scope, self._root, filters, session_id=session_id
            )
        except Exception as error:  # noqa: BLE001
            self.busy = False
            self.notice = _t(_k.write_failed(error=str(error) or type(error).__name__))
            self._paint()
            return
        self.busy = False
        if not observed:
            self.notice = _t(_k.nothing_to_dismiss(source=source))
            self._paint()
            return

        def run() -> None:
            self._spawn_write("dismiss", scope=(through, observed), session_id=session_id)

        self._ask(_t(_k.confirm_dismiss_session(count=len(observed), source=source)), run)

    def action_mark_all(self) -> None:
        if self.busy or self._confirm is not None:
            return
        self.busy = True
        self.run_worker(self._arm_mark_all())

    async def _arm_mark_all(self) -> None:
        # The page's filters bound the gesture: "mark all read" on a filtered
        # view marks what the view lists, never a wider set than it showed.
        filters = model._replace(self.filters, read="unread")
        try:
            through, observed, rows = await asyncio.to_thread(
                model.enumerate_scope, self._root, filters
            )
        except Exception as error:  # noqa: BLE001
            self.busy = False
            self.notice = _t(_k.write_failed(error=str(error) or type(error).__name__))
            self._paint()
            return
        self.busy = False
        # Only UNREAD rows are in the count: a read ask stays pinned in the
        # waiting group, and counting it would promise a change ``y`` cannot make.
        unread = {str(row["id"]) for row in rows if not row.get("read")}
        observed = [token for token in observed if token["id"] in unread]
        if not observed:
            self.notice = _t(_k.nothing_unread())
            self._paint()
            return

        def run() -> None:
            self._spawn_write("read", scope=(through, observed))

        self._ask(_t(_k.confirm_mark_all(count=str(len(observed)))), run)

    def action_undo(self) -> None:
        if self.busy or self._confirm is not None or not self.undo_live:
            return
        last = self._last_write
        assert last is not None
        self._last_write = None
        self.busy = True
        self.run_worker(self._run_undo(last))

    async def _run_undo(self, last: model.WriteResult) -> None:
        try:
            result = await asyncio.to_thread(model.undo, self._root, last)
        finally:
            self.busy = False
        self.notice = (
            _t(_k.write_failed(error=result.error))
            if result.error
            else _t(_k.restored(count=str(len(result.changed))))
        )
        self.reload(keep_notice=True)

    def action_answer(self) -> None:
        if self._confirm is not None:
            return
        row = self.detail_row
        if row is None:
            return
        ask = row.get("ask") if row.get("source") == "ask" else None
        session_id = str((row.get("session") or {}).get("id") or "")
        if ask is None:
            # On an update, ``a`` answers the first live ask in its conversation
            # (design §8.4) — the detail payload lists them.
            payload = self._details.get(self._token_key(row)) or {}
            first = next(
                (a for a in payload.get("asks") or [] if a.get("status", "open") == "open"),
                None,
            )
            if first is None:
                return
            ask = first.get("ask") or first
        ask_id = str((ask or {}).get("ask_id") or "")
        if session_id and ask_id:
            self.post_message(NotificationsViewAnswerRequested(session_id, ask_id))

    @staticmethod
    def _token_key(row: dict[str, Any]) -> tuple[str, int]:
        return (str(row["id"]), int(row.get("revision") or 1))

    def action_open_conversation(self) -> None:
        if self._confirm is not None:
            return
        row = self.detail_row
        session_id = str(((row or {}).get("session") or {}).get("id") or "")
        if session_id:
            self.post_message(NotificationsViewOpenConversation(session_id))

    def action_cycle(self, attribute: str) -> None:
        if self._confirm is not None or self.view_mode == "detail":
            return
        choices = {
            "read": model.READ_FILTERS,
            "kind": model.KIND_FILTERS,
            "priority": model.PRIORITY_FILTERS,
            "grouping": model.GROUPINGS,
        }[attribute]
        self.filters = self.filters.cycled(attribute, choices)
        # A filter change is a new list: the selection named rows of the old one.
        self.selected_ids.clear()
        self.reload()

    def action_reload(self) -> None:
        if self._confirm is not None:
            return
        self._details.clear()
        self.reload()

    def action_leave(self) -> None:
        if self._confirm is not None:
            self._disarm()
            return
        if self.selected_ids:
            self.selected_ids.clear()
            self._paint()
            return
        if self.view_mode == "detail":
            self.view_mode = "list"
            pinned = self._detail_row
            self._detail_row = None
            # Back to the row the detail showed when it is still listed; if it
            # left (read on the Unread filter), the cursor already holds its
            # successor.
            if pinned is not None and any(str(r["id"]) == str(pinned["id"]) for r in self.rows):
                self.cursor_id = str(pinned["id"])
            self._apply_layout()
            self._paint()
            return
        self.post_message(NotificationsViewDismissed())

    # -- confirm --------------------------------------------------------------
    def _ask(self, prompt: str, run: Callable[[], None]) -> None:
        self._confirm = (prompt, run)
        self._paint()

    def _disarm(self) -> None:
        self._confirm = None
        self._paint()

    @property
    def confirming(self) -> bool:
        return self._confirm is not None

    def on_key(self, event: Any) -> None:
        """A y/n confirm owns the keyboard while it is up.

        ``y`` runs; ``n``/``esc``/``enter`` keep everything; every other
        printable key is swallowed so no letter is read as a second meaning
        (the schedule page's rule, same reason).
        """
        if self._confirm is None:
            return
        if event.key == "y":
            event.stop()
            event.prevent_default()
            _prompt, run = self._confirm
            self._confirm = None
            run()
            return
        if event.key in ("escape", "n", "enter"):
            event.stop()
            event.prevent_default()
            self._disarm()
            return
        if event.is_printable:
            event.stop()
            event.prevent_default()

    # -- writes ---------------------------------------------------------------
    def _spawn_write(self, op: str, **kwargs: Any) -> None:
        if self.busy:
            return
        self.busy = True
        self.run_worker(self._write(op, **kwargs))

    async def _write(
        self, op: str, *, quiet: bool = False, **kwargs: Any
    ) -> model.WriteResult | None:
        self.busy = True
        try:
            result = await asyncio.to_thread(model.write, self._root, op, **kwargs)
        finally:
            self.busy = False
        if result.error:
            self.notice = _t(_k.write_failed(error=result.error))
        elif not quiet:
            self.notice = self._receipt(result)
            if result.changed:
                self._last_write = result
                self._last_write_at = time.monotonic()
                self.set_timer(model.UNDO_WINDOW_S + 0.05, self._paint)
        if result.changed:
            self.selected_ids -= set(result.changed)
        self.reload(keep_notice=True)
        return result

    @staticmethod
    def _receipt(result: model.WriteResult) -> str:
        parts: list[str] = []
        count = len(result.changed)
        if count:
            parts.append(
                {
                    "dismiss": _t(_k.dismissed(count=count)),
                    "read": _t(_k.marked_read(count=str(count))),
                    "unread": _t(_k.marked_unread(count=str(count))),
                    "undismiss": _t(_k.restored(count=str(count))),
                }[result.op]
            )
        if result.skipped_live_asks:
            parts.append(_t(_k.asks_stay(count=result.skipped_live_asks)))
        if result.skipped_stale:
            parts.append(_t(_k.stale_skipped(count=result.skipped_stale)))
        return " ".join(parts)

    @property
    def undo_live(self) -> bool:
        return (
            self._last_write is not None
            and time.monotonic() - self._last_write_at <= model.UNDO_WINDOW_S
        )

    # -- paint ----------------------------------------------------------------
    def _paint(self) -> None:
        width = max(20, self.size.width or 80)
        snapshot = self.snapshot
        # TWO cells short of the body, not one: `.settings-view-body` declares
        # `scrollbar-gutter: stable`, so the bar's column is reserved whether or
        # not it shows, plus one cell of breathing room (`SettingsView`'s
        # `_list_width` reservation). Measured on the first 80x24 capture: at
        # one cell short every row's time column wrapped onto its own line.
        list_width = max(20, (self._list.size.width or width) - 2)
        self._title.update(
            render.header_line(
                snapshot or model.Snapshot(),
                self.filters,
                width,
                selected=len(self.selected_ids),
                selection_hint=self._selection_hint(),
            )
        )
        self._rule.update(Text(render.RULE_GLYPH * 400, style="dim"))
        self._edge.update(Text("\n".join([render.EDGE_GLYPH] * 200), style="dim"))
        if snapshot is None:
            self._list_text.update(Text(""))
        elif snapshot.error or not snapshot.rows:
            self._list_text.update(render.empty_text(snapshot, self.filters))
        else:
            groups = model.group(snapshot, self.filters, render.group_label)
            text, offsets = render.list_text(
                groups,
                list_width,
                cursor_id=self.cursor_id,
                selected_ids=self.selected_ids,
                now_ms=self._now(),
            )
            self._list_text.update(text)
            self._scroll_to_line(offsets.get(self.cursor_id))
        self._paint_detail()
        self._notice_text.update(Text(self._footer_text(), style=render.ink("muted")))
        self._hints.update(render.hints_text(render.hint_row(self._hint_labels(), width - 1)))

    @property
    def detail_row(self) -> dict[str, Any] | None:
        """The row the detail paints: the PINNED one, else the cursor's.

        Pinning is the rule in both layouts (T3): a row the operator explicitly
        opened stays in the pane even after it leaves the list it was opened
        from. ``_move_to`` clears the pin, so browsing is unaffected.
        """
        if self._detail_row is not None:
            return self._detail_row
        return self.current

    def _paint_detail(self) -> None:
        if not (self.wide or self.view_mode == "detail"):
            return
        row = self.detail_row
        if row is None:
            self._detail_text.update(Text(""))
            return
        width = max(20, (self._detail.size.width or self.size.width or 80) - 1)
        ids = [str(r["id"]) for r in self.rows]
        index = ids.index(str(row["id"])) if str(row["id"]) in ids else self._index()
        self._detail_text.update(
            render.detail_text(
                self._details.get(self._token_key(row)),
                row,
                width,
                position=(index + 1, len(self.rows)),
                now_ms=self._now(),
                show_back=not self.wide,
            )
        )

    def _selection_hint(self) -> str:
        if not self.selected_ids:
            return ""
        targets = self._targets()
        dismissible = sum(1 for row in targets if row.get("source") == "notify")
        dismiss, read, clear = _t(_k.hint_dismiss()), _t(_k.hint_read()), _t(_k.hint_clear())
        return f"e {dismiss} {dismissible} · u {read} · esc {clear}"

    def _scroll_to_line(self, line: int | None) -> None:
        if line is None:
            return
        viewport = max(1, self._list.size.height or 1)
        y = self._list.scroll_offset.y
        if line < y:
            target = max(0, line - 1)  # keep the group header above in view
        elif line >= y + viewport:
            target = line - viewport + 1
        else:
            return
        try:
            self._list.scroll_to(y=target, animate=False, immediate=True)
        except Exception:  # noqa: BLE001 - scrolling is a nicety
            pass

    def _footer_text(self) -> str:
        if self._confirm is not None:
            return self._confirm[0]
        return self.notice

    def _hint_labels(self) -> list[tuple[str, str]]:
        row = self.detail_row
        labels: list[tuple[str, str]] = []
        if self.undo_live:
            labels.append(("z", _t(_k.hint_undo())))
        if row is not None and self._confirm is None:
            is_ask = row.get("source") == "ask"
            if self.view_mode == "detail" and not self.wide:
                labels.append(("o", _t(_k.hint_conversation())))
                labels.append(("a", _t(_k.hint_answer())))
                if not is_ask:
                    labels.append(("e", _t(_k.hint_dismiss())))
                    labels.append(("E", _t(_k.hint_dismiss_all())))
                    labels.append(
                        ("u", _t(_k.hint_unread()) if row.get("read") else _t(_k.hint_read()))
                    )
            else:
                labels.append(("↵", _t(_k.hint_answer()) if is_ask else _t(_k.hint_open())))
                if not is_ask:
                    labels.append(("e", _t(_k.hint_dismiss())))
                    # T8: the conversation-wide gesture was undiscoverable from
                    # the list (only the narrow detail showed it). It costs no
                    # new copy — `hint_dismiss_all` is the same string the
                    # detail uses — and `hint_row` already drops the widest
                    # hints first, so a narrow footer pays nothing for it.
                    if (row.get("session") or {}).get("id"):
                        labels.append(("E", _t(_k.hint_dismiss_all())))
                read_label = _t(_k.hint_unread()) if row.get("read") else _t(_k.hint_read())
                labels.append(("u", read_label))
                labels.append(("space", _t(_k.hint_select())))
                labels.append(("o", _t(_k.hint_conversation())))
                labels.append(("f", _t(_k.hint_filter())))
        labels.append(("esc", _t(_k.hint_clear()) if self.selected_ids else _t(_k.hint_back())))
        return labels
