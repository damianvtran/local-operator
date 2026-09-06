"""A viewport-sized session list, distinct from the conversation it navigates.

One widget renders only the visible row window: a growing session directory
must not grow the Textual DOM. Cursor, current session and requested session
are separate identities, so a pending attach cannot pretend to have succeeded.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget

from local_operator.resume import format_age
from local_operator.tui import theme as theme_mod
from local_operator.tui.session_catalog import CatalogEntry, rank_entries
from local_operator.tui.terminal_title import SPINNER_FRAMES, SPINNER_INTERVAL_S
from local_operator.tui.widgets.session_picker import row_state_mark
from local_operator.tui.widgets.tool_card import truncate_cells

SIDEBAR_WIDTH = 30

#: How long a requested row waits before its state mark becomes a spinner.
#: The tint itself is immediate — it is the acknowledgement that the click
#: landed. The spinner is only for a switch that is genuinely taking a while,
#: and in practice a warm switch completes well inside this window, so the
#: glyph should almost never be seen: a warm sample that shows it is a
#: regression, not a feature working.
REQUESTED_SPINNER_DELAY_S = 0.15

#: Blank cells between the list and the conversation it sits beside.
#:
#: The list's right-hand age column ("6m", "23h", "1d") ended one cell from the
#: transcript's first character, which read as two columns jammed together
#: rather than two regions — the user's report was that it "looks overcrowded".
#: Three cells is what actually separates them at a glance; two still reads as
#: tight against a full-bleed transcript.
#:
#: ADDED TO THE WIDTH rather than taken out of the content, because the content
#: has none to give: at 28 cells the title column is already 20 after the
#: cursor, state mark and age, and real titles ellipsize there today. Spending
#: the gutter from that budget would have paid for whitespace with the one
#: thing the list exists to show. The main lane can afford it — at 100 columns
#: it keeps 65, above the 60-column floor `SIDEBAR_MAIN_MIN_WIDTH` sets, and
#: below that threshold the drawer is an overlay that does not displace the
#: conversation at all.
#:
#: Whitespace, not a rule: the chrome is borderless, so the gap IS the
#: separator (see the stylesheet's own note on the app's outer inset).
SIDEBAR_GUTTER = 3

#: Below this the gutter is surrendered before the list narrows further. A
#: squeezed terminal needs the cells for a legible title far more than it needs
#: the separation the gutter buys.
SIDEBAR_MIN_CONTENT_WIDTH = 24
SIDEBAR_MAIN_MIN_WIDTH = 60


class SessionSidebar(Widget, can_focus=True):
    BINDINGS = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
        Binding("pageup", "page(-1)", show=False),
        Binding("pagedown", "page(1)", show=False),
        Binding("home", "edge(False)", show=False),
        Binding("end", "edge(True)", show=False),
        Binding("enter", "select", show=False),
        Binding("escape", "leave", show=False),
    ]

    class Selected(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class Dismissed(Message):
        pass

    def __init__(self) -> None:
        super().__init__(id="session-sidebar")
        self.entries: tuple[CatalogEntry, ...] = ()
        self.current_id = ""
        self._requested_id = ""
        self._requested_at = 0.0
        self.cursor_id = ""
        self.error = ""
        self._catalog_loading = True
        self._offset = 0
        self._frame = 0
        self._timer: Timer | None = None
        self._pressed_id: str | None = None
        self._deferred: tuple[CatalogEntry, ...] | None = None
        #: Row under the pointer, by identity rather than by row index: a
        #: catalog refresh reorders rows beneath a stationary pointer, and a
        #: remembered index would light (and describe) whichever session slid
        #: into that slot. `_hover_y` is what re-resolves the identity when the
        #: order changes without the mouse moving.
        self._hover_id: str = ""
        self._hover_y: int | None = None
        #: When the pointer entered the current row. An in-row move restores
        #: the description only once Textual's own show delay has elapsed
        #: since then, so it never appears earlier than Textual would show it.
        self._hover_since: float | None = None
        self.display = False

    @property
    def requested_id(self) -> str:
        """The row a click asked for that has not yet become ``current_id``.

        Set synchronously at ``_sidebar_navigation_pending`` — before any
        await — so the tint is on the very next frame after the click. It is
        an acknowledgement and nothing more: it does not move ``current_id``,
        does not touch the transcript, enables no input and acks nothing.
        Cleared by ``pending("")`` on commit, failure or cancel.
        """
        return self._requested_id

    @requested_id.setter
    def requested_id(self, session_id: str) -> None:
        if session_id != self._requested_id:
            self._requested_at = time.monotonic() if session_id else 0.0
        self._requested_id = session_id
        self._sync_animation()

    def _requested_spinning(self) -> bool:
        return bool(self._requested_id) and (
            time.monotonic() - self._requested_at >= REQUESTED_SPINNER_DELAY_S
        )

    @property
    def page_size(self) -> int:
        return max(1, self.size.height - 2)

    @property
    def visible_entries(self) -> tuple[CatalogEntry, ...]:
        return self.entries[self._offset : self._offset + self.page_size]

    def set_entries(self, entries: Sequence[CatalogEntry]) -> None:
        ordered = rank_entries(entries)
        # A refresh between mouse-down and click must not change what was
        # pressed. Freeze the entire order until that gesture has completed.
        if self._pressed_id is not None:
            self._deferred = ordered
            return
        self.entries = ordered
        self._catalog_loading = False
        self.error = ""
        if not any(entry.id == self.cursor_id for entry in ordered):
            self.cursor_id = self.current_id or (ordered[0].id if ordered else "")
        self._offset = min(self._offset, max(0, len(ordered) - self.page_size))
        # Re-resolve against the NEW order at the pointer's unchanged position:
        # a reorder under a resting pointer must relabel the row it is actually
        # over, not keep describing the session that moved away. Blanking it
        # instead left the affordance and description dark until the user
        # jiggled the mouse.
        self._set_hover(self._hover_y)
        self._sync_animation()
        self.refresh()

    def show_error(self, message: str) -> None:
        # A refresh error does not erase the last usable catalog.
        self.error = message
        self._catalog_loading = False
        self.refresh()

    def set_open(self, opened: bool) -> None:
        self.display = opened
        self._sync_animation()
        self.refresh()

    def on_mount(self) -> None:
        self._timer = self.set_interval(SPINNER_INTERVAL_S, self._advance_spinner, pause=True)
        self._sync_animation()

    def _sync_animation(self) -> None:
        if self._timer is None:
            return
        if self.display and (
            self._catalog_loading
            or bool(self._requested_id)
            or any(entry.row.live_state == "busy" for entry in self.visible_entries)
        ):
            self._timer.resume()
        else:
            self._timer.pause()

    def _advance_spinner(self) -> None:
        self._frame += 1
        self.refresh()

    def _cursor_index(self) -> int:
        return next((i for i, row in enumerate(self.entries) if row.id == self.cursor_id), 0)

    def _reveal(self) -> None:
        index = self._cursor_index()
        self._offset = max(min(self._offset, index), index - self.page_size + 1)
        self._sync_animation()
        self.refresh()

    def action_move(self, delta: int) -> None:
        if self.entries:
            index = max(0, min(len(self.entries) - 1, self._cursor_index() + delta))
            self.cursor_id = self.entries[index].id
            self._reveal()

    def action_page(self, delta: int) -> None:
        self.action_move(delta * self.page_size)

    def action_edge(self, end: bool) -> None:
        if self.entries:
            self.cursor_id = self.entries[-1 if end else 0].id
            self._reveal()

    def action_select(self) -> None:
        if self.cursor_id and any(entry.id == self.cursor_id for entry in self.entries):
            self._reveal()
            self.post_message(self.Selected(self.cursor_id))

    def action_leave(self) -> None:
        self.post_message(self.Dismissed())

    def _entry_at(self, y: int) -> CatalogEntry | None:
        index = y - 1
        visible = self.visible_entries
        return visible[index] if 0 <= index < len(visible) else None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        entry = self._entry_at(event.y)
        if entry is not None:
            self._pressed_id = entry.id
        event.stop()

    def on_click(self, event: events.Click) -> None:
        entry = (
            self._entry_at(event.y)
            if self.region.contains(event.screen_x, event.screen_y)
            else None
        )
        target = (self._pressed_id or (entry.id if entry else "")) if entry is not None else ""
        self._pressed_id = None
        if target:
            self.cursor_id = target
            self.post_message(self.Selected(target))
        if self._deferred is not None:
            deferred, self._deferred = self._deferred, None
            self.set_entries(deferred)
        event.stop()

    def _describe(self, entry: CatalogEntry | None) -> str | None:
        return f"{entry.row.name}\n{entry.status}\n{entry.id}" if entry else None

    def _set_hover(self, y: int | None) -> bool:
        """Point the hover affordance and the tooltip at the row under `y`.

        Called both from pointer movement and from a catalog refresh, because a
        stationary pointer covers a DIFFERENT session once the ranking
        reorders — the row must re-resolve without waiting for a mouse event.

        Returns whether the hovered identity changed, so the caller can repaint
        exactly once instead of on every pointer move within a row.
        """
        self._hover_y = y
        entry = self._entry_at(y) if y is not None else None
        hover_id = entry.id if entry is not None else ""
        description = self._describe(entry)
        # Re-arm the widget's ``tooltip`` only when the row identity changes.
        # This is NOT what keeps it up under pointer movement (round 4, M1:
        # ``Screen._handle_mouse_move`` hides a showing tooltip BEFORE the
        # event reaches this widget, so nothing set here can prevent that —
        # see ``on_mouse_move`` for the restore). What this does fix is the
        # OTHER way the description vanished: the 2 s catalog poll called
        # ``set_entries``, which blanked ``tooltip`` under a perfectly still
        # pointer. Keeping the value stable across a refresh is what lets a
        # resting description survive the poll.
        changed = hover_id != self._hover_id
        if changed:
            self._hover_since = time.monotonic() if hover_id else None
        if changed or self.tooltip != description:
            self._hover_id = hover_id
            self.tooltip = description
        return changed

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._set_hover(event.y):
            self.refresh()
            # A NEW row: Textual's delay timer was (re)armed by this very
            # move only if its tooltip was not showing; when it WAS showing
            # it hid it and armed nothing, so a resting pointer on the new
            # row would never get that row's description. Arm our own
            # one-shot at the same delay, so a row change behaves like a
            # first arrival rather than a dead end.
            if self._hover_id:
                self.set_timer(
                    float(self.app.TOOLTIP_DELAY),
                    self._show_tooltip_if_still_here,
                    name="sidebar-tooltip",
                )
            return
        # Same row, pointer merely moved within it. Textual has ALREADY hidden
        # the tooltip by the time this runs (``Screen._handle_mouse_move``
        # sets ``display = False`` on any move over the widget that owns a
        # showing tooltip, before forwarding the event) and will not re-show
        # it until a further move restarts its delay timer — so one cell of
        # jitter dropped the description and resting did not bring it back.
        # The user's ask is "stays while the mouse is hovered over the
        # session": restore it. Only when it was showing for THIS widget, so
        # a tooltip that was never up (first arrival, or a different widget's)
        # still goes through Textual's normal delay rather than popping.
        if self._hover_id and self._tooltip_due():
            self._show_tooltip_now()

    def _tooltip_due(self) -> bool:
        """Has the description been up (or is it owed) for the hovered row?

        Textual keeps no "was shown" bit a widget can read, and by the time
        this runs it has already hidden the tooltip — so the honest signal is
        time: the pointer entered this row at ``_hover_since`` and Textual's
        own ``TOOLTIP_DELAY`` has elapsed, which is exactly the condition
        under which its timer would have shown it. Using its constant means
        the restore never pops earlier than Textual itself would.
        """
        if self._hover_since is None:
            return False
        return time.monotonic() - self._hover_since >= float(self.app.TOOLTIP_DELAY)

    def _show_tooltip_if_still_here(self) -> None:
        # The one-shot may fire after the pointer moved on or left; only the
        # row it was armed for gets shown, and only if nothing hid it since.
        if self._hover_id and self._tooltip_due() and self.screen.app.mouse_over is self:
            self._show_tooltip_now()

    def _show_tooltip_now(self) -> None:
        """Re-show the description Textual just hid for an in-row move."""
        try:
            from textual.widgets import Tooltip

            tooltip = self.screen.get_child_by_type(Tooltip)
        except Exception:
            return
        if self.tooltip is None:
            return
        tooltip.display = True
        tooltip.absolute_offset = self.app.mouse_position
        tooltip.update(self.tooltip)

    def on_leave(self, event: events.Leave) -> None:
        # The pointer left the list: drop both the affordance and the
        # description rather than leaving a row lit under an absent cursor.
        if self._hover_id or self._hover_y is not None:
            self._hover_id = ""
            self._hover_y = None
            self._hover_since = None
            self.tooltip = None
            self.refresh()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._scroll(1)
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._scroll(-1)
        event.stop()

    def _scroll(self, direction: int) -> None:
        # Wheel moves the viewport, never the keyboard target. Match the live
        # app sensitivity rather than invent a second scrolling speed.
        step = max(1, int(self.app.scroll_sensitivity_y))
        self._offset = max(
            0, min(max(0, len(self.entries) - self.page_size), self._offset + direction * step)
        )
        self._sync_animation()
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        self._offset = min(self._offset, max(0, len(self.entries) - self.page_size))
        self._sync_animation()

    def render(self) -> Text:
        width = max(1, self.size.width)
        result = Text(no_wrap=True, overflow="crop")
        result.append(
            truncate_cells("Sessions", width).ljust(width), style=theme_mod.semantic_color("muted")
        )
        for entry in self.visible_entries:
            result.append("\n")
            current = entry.id == self.current_id
            cursor = self.has_focus and entry.id == self.cursor_id
            hovered = entry.id == self._hover_id and entry.id != ""
            # The row a click asked for, until it becomes current. It shares
            # the cursor's ground so the eye reads "this one is being opened"
            # where it just clicked, and stays distinct from current (bold on
            # `surface`) because it is NOT current yet — readiness is a
            # separate, later fact that only commit may assert.
            requested = entry.id == self._requested_id and entry.id != "" and not current
            # Three states have to stay separable, so they occupy three
            # different grounds: the keyboard cursor keeps `tint-select` (the
            # only tinted one), the attached session keeps `surface` plus bold,
            # and hover is `overlay` — the SAME ground `ToolCard:hover` and
            # `SubagentRow:hover` use, because pointing at a row should look
            # the same everywhere in this app. Hover yields to both of the
            # other two: it is transient pointer feedback, and must not
            # overpaint the identity of where you ARE or where the keyboard is.
            background = (
                "tint-select"
                if cursor or requested
                else "surface" if current else "overlay" if hovered else None
            )
            style = Style(
                color=theme_mod.semantic_color("fg"),
                bgcolor=theme_mod.semantic_color(background) if background else None,
                bold=current,
            )
            line = Text(style=style, no_wrap=True)
            # The requested row gets a caret the cursor does not have. Sharing
            # BOTH `tint-select` and `›` made the two indistinguishable, and
            # they are reachable on opposite rows from real bindings (focus the
            # list, ctrl+shift+down, or press down before commit) — the frame
            # then could not say which row was opening (round 5, D6). `»` is
            # the doubled form of the same caret: one cell, same ink, same
            # column, so nothing reflows and the ramp is untouched. Requested
            # wins when a row is both, because "this is opening" is the fact
            # the user is waiting on.
            line.append("» " if requested else "› " if cursor else "  ")
            mark, ink = row_state_mark(entry.row, self._frame)
            if requested and self._requested_spinning():
                # Same ink a busy row's spinner uses (``row_state_mark``), so one
                # spinner means one thing everywhere in the list.
                # Only after the delay: a switch that is taking long enough to
                # notice gets a spinner ON THE ROW, where the eye is. The
                # footer "Opening…" stays as the textual counterpart.
                mark, ink = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)], "accent"
            elif entry.unseen and not entry.row.pending:
                mark, ink = (
                    ("✗", "danger")
                    if entry.completion_kind in ("error", "interrupted")
                    else ("✓", "success")
                )
            line.append(f"{mark or ' '} ", style=theme_mod.semantic_color(ink))
            age = (
                format_age(max(0, time.time() - entry.row.mtime)).replace(" ago", "")
                if width >= 28
                else ""
            )
            if len(age) > 4:
                age = ""
            title_width = max(1, width - 4 - (len(age) + 1 if age else 0))
            title = truncate_cells(entry.row.name or "Untitled conversation", title_width)
            line.append(title)
            line.pad_right(max(0, width - line.cell_len - len(age)))
            line.append(age, style=theme_mod.semantic_color("dim"))
            result.append_text(line)
        if not self.entries:
            text = (
                "Could not load conversations"
                if self.error
                else (
                    f"{SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]} Loading conversations…"
                    if self._catalog_loading
                    else "No conversations yet"
                )
            )
            result.append("\n" + truncate_cells(text, width), style=theme_mod.semantic_color("dim"))
        while result.plain.count("\n") < self.size.height - 2:
            result.append("\n")
        hint = "esc return" if self.has_focus else "f9 focus · ctrl+b hide"
        if len(self.entries) > self.page_size:
            last = min(len(self.entries), self._offset + self.page_size)
            hint = f"{self._offset + 1}–{last}/{len(self.entries)} · ctrl+b hide"
        footer = "Refresh failed" if self.error else "Opening…" if self.requested_id else hint
        result.append(
            "\n" + truncate_cells(footer, width),
            style=theme_mod.semantic_color("warning" if self.error else "dim"),
        )
        return result
