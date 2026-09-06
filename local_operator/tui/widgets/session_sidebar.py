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
        self.requested_id = ""
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
        self.display = False

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
        # Textual hides a showing tooltip on the next move over the SAME widget
        # (`Screen._handle_mouse_move`), which is why it vanished after about a
        # second of resting on a row. Rows live inside ONE widget, so every
        # in-row move looked like that repeat. Re-arming only when the row
        # identity actually changes keeps the description up while the pointer
        # rests, and still swaps it the moment a different row is under it.
        changed = hover_id != self._hover_id
        if changed or self.tooltip != description:
            self._hover_id = hover_id
            self.tooltip = description
        return changed

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._set_hover(event.y):
            self.refresh()

    def on_leave(self, event: events.Leave) -> None:
        # The pointer left the list: drop both the affordance and the
        # description rather than leaving a row lit under an absent cursor.
        if self._hover_id or self._hover_y is not None:
            self._hover_id = ""
            self._hover_y = None
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
                if cursor
                else "surface" if current else "overlay" if hovered else None
            )
            style = Style(
                color=theme_mod.semantic_color("fg"),
                bgcolor=theme_mod.semantic_color(background) if background else None,
                bold=current,
            )
            line = Text(style=style, no_wrap=True)
            line.append("› " if cursor else "  ")
            mark, ink = row_state_mark(entry.row, self._frame)
            if entry.unseen and not entry.row.pending:
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
