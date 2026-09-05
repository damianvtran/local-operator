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
from local_operator.tui.terminal_title import SPINNER_INTERVAL_S
from local_operator.tui.widgets.session_picker import row_state_mark
from local_operator.tui.widgets.tool_card import truncate_cells

SIDEBAR_WIDTH = 30
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
        self.loading = True
        self._offset = 0
        self._frame = 0
        self._timer: Timer | None = None
        self._pressed_id: str | None = None
        self._deferred: tuple[CatalogEntry, ...] | None = None
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
        self.loading = False
        self.error = ""
        if not any(entry.id == self.cursor_id for entry in ordered):
            self.cursor_id = self.current_id or (ordered[0].id if ordered else "")
        self._offset = min(self._offset, max(0, len(ordered) - self.page_size))
        self._sync_animation()
        self.refresh()

    def show_error(self, message: str) -> None:
        # A refresh error does not erase the last usable catalog.
        self.error = message
        self.loading = False
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
        if self.display and any(entry.row.live_state == "busy" for entry in self.visible_entries):
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
        entry = self._entry_at(event.y)
        target = self._pressed_id or (entry.id if entry else "")
        self._pressed_id = None
        if target:
            self.cursor_id = target
            self.post_message(self.Selected(target))
        if self._deferred is not None:
            deferred, self._deferred = self._deferred, None
            self.set_entries(deferred)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        entry = self._entry_at(event.y)
        self.tooltip = f"{entry.row.name}\n{entry.status}\n{entry.id}" if entry else None

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
            background = "tint-select" if cursor else "surface" if current else "bg"
            style = Style(
                color=theme_mod.semantic_color("fg"),
                bgcolor=theme_mod.semantic_color(background),
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
            text = "Loading conversations…" if self.loading else "No conversations yet"
            result.append("\n" + truncate_cells(text, width), style=theme_mod.semantic_color("dim"))
        while result.plain.count("\n") < self.size.height - 2:
            result.append("\n")
        footer = (
            "Refresh failed" if self.error else "Opening…" if self.requested_id else "ctrl+b hide"
        )
        result.append(
            "\n" + truncate_cells(footer, width),
            style=theme_mod.semantic_color("warning" if self.error else "dim"),
        )
        return result
