"""The ``/resume`` picker: choose a past conversation by NAME, not by hash.

Why a screen rather than a notice. The recovery list used to be printed INTO
the transcript: a block of ``<12-hex id>   3h ago`` rows that pushed the
conversation up, could not be navigated, stayed on screen after it had been
used, and left the user to retype an id they had to read off the scrollback.
Choosing a conversation is a two-way question — the app offers the options,
the user picks one — so it takes a surface that can hold a selection and hand
an answer back. That is exactly a modal screen, and it is what the trajectory
viewer already does for the other "read a list, pick a row" case.

Why names. A column of hex ids is not something anyone recognises their own
work in; the id is what the machine resumes, not what a human picks by. The
name is the session's opening user message (see
:func:`local_operator.resume.session_name`), which is both the only per-session
title on disk and the thing the user actually remembers about the session.

The list is filterable by typing because the ids are unmemorable and the names
are not: with a hundred sessions, "asteroids" finds the one you mean faster
than paging can. Filtering narrows; it never reorders, so a row does not move
under the cursor as the query grows.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.resume import SessionRow, format_age
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells

#: Width the card asks for. Wide enough for a readable name plus the age and
#: id columns; the stylesheet caps it against narrow terminals.
PICKER_WIDTH = 74

#: Rows shown before the list scrolls. A page that fills the terminal makes the
#: modal feel like a mode switch rather than a popup; ten is enough to scan.
PAGE_ROWS = 10

#: The cursor. A filled caret rather than a reversed row: the transcript behind
#: this card is dim, and a block of inverted colour reads as a selection the
#: user made rather than as the position they are on.
CURSOR = "❯ "
NO_CURSOR = "  "


def filter_rows(rows: Sequence[SessionRow], query: str) -> list[SessionRow]:
    """Rows whose name or id contains ``query``, case-insensitively.

    Substring rather than fuzzy on purpose: the two fields are a sentence the
    user wrote and a hex id, and fuzzy matching over free text produces
    confident nonsense ("asteroids" matching a row about "assets ordering").
    Order is preserved — filtering must never move a row under the cursor.
    """
    needle = query.strip().lower()
    if not needle:
        return list(rows)
    return [row for row in rows if needle in row.name.lower() or needle in row.id.lower()]


def render_rows(
    rows: Sequence[SessionRow],
    selected: int,
    width: int,
    now: float,
) -> list[Text]:
    """One line per session: cursor, name, age, id.

    The name takes whatever the age and id columns do not, and truncates
    rather than wrapping — a two-row entry would break the arithmetic that
    maps a cursor index to a row.
    """
    fg = Style(color=theme_mod.semantic_color("fg"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    faint = Style(color=theme_mod.semantic_color("faint"))
    accent = Style(color=theme_mod.semantic_color("label"))

    ages = [format_age(max(0.0, now - row.mtime)) for row in rows]
    age_col = max((cell_len(age) for age in ages), default=0)
    # The id is fixed-width by construction (uuid4().hex[:12]) but is measured
    # rather than assumed, so an older session written by a build with a
    # different id length still lines up instead of ragging the column.
    id_col = max((cell_len(row.id) for row in rows), default=0)
    name_col = max(8, width - len(CURSOR) - age_col - id_col - 4)

    lines: list[Text] = []
    for index, (row, age) in enumerate(zip(rows, ages)):
        current = index == selected
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(CURSOR if current else NO_CURSOR, style=accent if current else faint)
        # An unnamed session is one whose transcript could not be read or that
        # has no user turn yet. Saying so is better than an empty cell, which
        # reads as a rendering fault.
        name = row.name or "(unnamed session)"
        name_style = fg if current else muted
        if not row.name:
            name_style = dim
        line.append(truncate_cells(name, name_col).ljust(name_col), style=name_style)
        line.append("  ")
        line.append(age.rjust(age_col), style=dim if current else faint)
        line.append("  ")
        line.append(row.id, style=faint)
        lines.append(line)
    return lines


class SessionPickerScreen(ModalScreen[str | None]):
    """Pick a conversation to resume; dismisses with its id, or ``None``.

    Two-way by construction: the caller pushes the screen and acts on what it
    returns, so the picker owns navigation and the caller owns resuming. Esc
    (or a filter that matches nothing, then Esc) answers ``None`` and the
    session on screen is left exactly as it was.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "choose", "Resume", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        # Ctrl+P/Ctrl+N as well as the arrows: every printable key belongs to
        # the filter, so the readline pair is the only other way to move a
        # hand that is already typing.
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("backspace", "backspace", "Edit filter", show=False),
    ]

    def __init__(self, rows: Sequence[SessionRow], now: float) -> None:
        super().__init__()
        self._all = list(rows)
        self._now = now
        self._query = ""
        self._selected = 0
        self._offset = 0
        self._body: Static

    # -- state ---------------------------------------------------------------
    # ``visible_rows``/``filter_query``/``_card_text``, not ``visible``/``query``/
    # ``_render``: all three of the shorter names are already Textual's
    # (``Widget.visible``, the ``DOMNode.query`` selector method, and the
    # internal ``Widget._render``), and shadowing them breaks the framework's
    # own focus, query and paint paths from inside the screen.
    @property
    def visible_rows(self) -> list[SessionRow]:
        """The rows the current filter admits, in their original order."""
        return filter_rows(self._all, self._query)

    @property
    def filter_query(self) -> str:
        return self._query

    @property
    def selected_index(self) -> int:
        return self._selected

    def selected_id(self) -> str | None:
        """The highlighted session's id, or ``None`` when nothing matches."""
        rows = self.visible_rows
        if not rows:
            return None
        return rows[min(self._selected, len(rows) - 1)].id

    # -- actions -------------------------------------------------------------
    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_choose(self) -> None:
        # Enter on an empty result set is not a choice. Dismissing with None
        # here (rather than ignoring the key) means Enter always closes the
        # picker, which is what a user who has typed a bad filter expects.
        self.dismiss(self.selected_id())

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        self._move_to(self._selected + delta * PAGE_ROWS)

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self.visible_rows) - 1 if to_end else 0)

    def action_backspace(self) -> None:
        if self._query:
            self.set_query(self._query[:-1])

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the filter.

        Handled here rather than as bindings because the filter accepts every
        character; a binding per key would be a table of ninety-five entries
        that still missed the ninety-sixth.
        """
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self.set_query(self._query + char)

    # The wheel moves the cursor a row at a time, which scrolls the window
    # with it (``_move_to`` keeps the selection on screen). Clamped, like
    # every other movement here: a scroll gesture that wrapped to the other
    # end of the list would read as the picker resetting itself.
    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(1)

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(-1)

    # -- internals -----------------------------------------------------------
    def set_query(self, query: str) -> None:
        """Apply a filter, keeping the cursor on a row that still exists."""
        self._query = query
        # The cursor is clamped rather than preserved by identity: a filter
        # that removes the selected row has to leave the cursor somewhere, and
        # the nearest surviving row is less surprising than jumping to the top.
        self._move_to(self._selected)
        self._repaint()

    def _move_to(self, index: int) -> None:
        rows = self.visible_rows
        if not rows:
            self._selected = 0
            self._offset = 0
            self._repaint()
            return
        # Clamped, never wrapping: a Down at the bottom that silently returned
        # to the top reads as the list having reset itself.
        self._selected = max(0, min(len(rows) - 1, index))
        # Scroll only far enough to keep the cursor on screen, so the list is
        # stable while paging through the middle of it.
        if self._selected < self._offset:
            self._offset = self._selected
        elif self._selected >= self._offset + PAGE_ROWS:
            self._offset = self._selected - PAGE_ROWS + 1
        self._offset = max(0, min(self._offset, max(0, len(rows) - PAGE_ROWS)))
        self._repaint()

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="session-picker"):
            self._body = Static(self._card_text(), id="session-picker-body")
            yield self._body

    def on_mount(self) -> None:
        self._repaint()

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        body.update(self._card_text())

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads."""
        return [line.plain for line in self._card_text().split("\n")]

    def _card_text(self) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        rows = self.visible_rows

        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Resume a conversation", style=Style(color=theme_mod.semantic_color("fg")))
        if self._query:
            # The filter is echoed in the header rather than in a separate
            # input row: it is one short string, and a dedicated row would
            # cost a line of the list on every terminal to show nothing most
            # of the time.
            header.append(f"  /{self._query}", style=Style(color=theme_mod.semantic_color("label")))
        header.append(f"  {len(rows)} of {len(self._all)}", style=faint)

        out = Text()
        out.append_text(header)
        out.append("\n")
        out.append("─" * PICKER_WIDTH, style=Style(color=theme_mod.semantic_color("edge")))
        out.append("\n")

        if not self._all:
            out.append("no previous sessions to resume", style=dim)
        elif not rows:
            out.append(f"nothing matches {self._query!r}", style=dim)
        else:
            window = rows[self._offset : self._offset + PAGE_ROWS]
            selected_in_window = self._selected - self._offset
            for index, line in enumerate(
                render_rows(window, selected_in_window, PICKER_WIDTH, self._now)
            ):
                if index:
                    out.append("\n")
                out.append_text(line)
            if len(rows) > PAGE_ROWS:
                out.append("\n")
                shown = self._offset + len(window)
                out.append(f"{shown} of {len(rows)}", style=faint)

        out.append("\n\n")
        out.append("↑↓ move · enter resume · type to filter · esc cancel", style=faint)
        return out
