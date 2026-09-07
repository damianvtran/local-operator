"""The ``/move`` picker: choose the directory this session works in.

A sibling of :class:`~local_operator.tui.widgets.session_picker.SessionPickerScreen`
and deliberately built to its pattern rather than to a new one: a modal over
the dimmed transcript, a cursor the screen owns, a filter every printable key
types into, and a dismissal that hands the caller a value or ``None``. The
answer is an absolute PATH (``str``), the way the resume picker's answer is a
session id — the caller decides what to do with it.

WHY IT ALSO COMPLETES PATHS. Every other picker in this app chooses from a
closed set: the sessions on disk, the models a provider lists, the blocks in a
message. A directory list is not closed — the destination the user wants may be
anywhere on the filesystem — so a card that could only filter its suggestions
would be abandoned the first time someone wanted a directory it had not
guessed. The input therefore has two modes, split by
:func:`~local_operator.tui.move_targets.looks_like_path`: anything anchored at
``~``/``/``/``.`` or containing a separator COMPLETES against the filesystem,
and anything else FILTERS the suggestions. One input, and the mode is
predictable from what the user typed rather than from a key they have to know.

The card states which mode it is in (the header says ``filter`` or ``path``)
because the two answer differently to the same keystroke, and a user who cannot
tell which one they are in cannot tell whether an empty list means "no such
directory" or "no such suggestion".
"""

from __future__ import annotations

from collections.abc import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.move_targets import MoveTarget, filter_targets, looks_like_path
from local_operator.tui.widgets.tool_card import truncate_cells

#: Width the card takes when the terminal allows it, and the floor it will not
#: go below. Both are CELL counts of the card's content, inside its padding.
#: Wider than the resume picker's 74 because a path is the whole row here where
#: a session name shares its row with an age and an id — and a truncated path,
#: unlike a truncated name, is not recognisable at all.
PICKER_MAX_WIDTH = 80
PICKER_MIN_WIDTH = 30

#: Cells between the card and the terminal edges, on top of its own padding, so
#: it reads as floating. Same value and same reason as the resume picker's.
PICKER_WIDTH_MARGIN = 6

#: Cells of the card's own padding on EACH side, mirroring ``padding: 1 2``.
PICKER_PADDING_CELLS = 2

#: Rows of directories shown before the list scrolls, when there is room. A
#: CEILING, not the page size — see :meth:`MovePickerScreen._page_rows`.
PAGE_ROWS_MAX = 10

#: Fraction of the screen height the card may occupy, mirroring the stylesheet.
CARD_MAX_HEIGHT_FRACTION = 0.8

#: Rows the card spends on things that are not directories: the header, the
#: rule under it, the blank spacer and the footer. Reserved FIRST so a short
#: terminal loses list rows (which scroll) rather than the footer (which is the
#: only statement of how to leave) — the discipline `session_picker` records.
CARD_CHROME_ROWS = 4

#: Rows of the card's own padding, mirroring the vertical half of ``padding: 1 2``.
CARD_PADDING_ROWS = 2

#: The cursor column, present on every row so a selection cannot shift the text.
CURSOR_CELLS = 2

#: What the card says when the filter admits nothing. Distinct sentences for
#: the two modes because they mean different things and imply different fixes:
#: a filter that matches nothing is a typo in a word, a path that matches
#: nothing is a directory that is not there.
NO_MATCH_NOTICE = "no suggestion matches that"
NO_PATH_NOTICE = "no directory matches that path"


#: Tiers whose note states WHAT THE ROW IS rather than why it was offered.
#: Rendered at the paths' own weight so the grouping reads as structure; the
#: structural tiers stay faint. See ``render_rows`` (design D5).
_IDENTITY_KINDS = frozenset({"current", "recent"})


def render_rows(
    targets: list[MoveTarget],
    selected: int,
    width: int,
    hovered: int | None = None,
) -> list[Text]:
    """One line per directory: cursor, label, and the note saying why.

    The note is right-aligned and DROPPED before the label is cut: the label is
    the thing being chosen and the note is only the reason it was offered, so
    on a narrow card the reason goes first. The label is then truncated from
    the LEFT (``…/repos/lo-move-cmd``) rather than the right, because the tail
    of a path is what distinguishes two siblings while the head is the part
    every row shares.
    """
    dim = Style(color=theme_mod.semantic_color("dim"))
    faint = Style(color=theme_mod.semantic_color("faint"))
    accent = Style(color=theme_mod.semantic_color("accent"))
    fg = Style(color=theme_mod.semantic_color("fg"))

    out: list[Text] = []
    for index, target in enumerate(targets):
        is_selected = index == selected
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("❯ " if is_selected else "  ", style=accent if is_selected else faint)
        room = max(1, width - CURSOR_CELLS)
        note = target.detail
        # ``current`` is EXEMPT from the note drop, unlike every other note.
        # The others say why a row was offered and are expendable beside the
        # label being chosen; `current` answers a different question — "where
        # am I now?" — and it was dropped first on exactly the deep paths this
        # machine is full of, so the marker vanished for the users most likely
        # to need it (design D4). It costs 7 cells and the label is truncated
        # to pay for them.
        pinned = target.kind == "current"
        note_cells = cell_len(note) + 2 if note else 0
        if note and not pinned and cell_len(target.label) + note_cells > room:
            note, note_cells = "", 0
        label = _truncate_head(target.label, max(1, room - note_cells))
        line.append(label, style=fg if is_selected else dim)
        if note:
            pad = room - cell_len(label) - cell_len(note)
            if pad > 0:
                line.append(" " * pad)
            # The IDENTITY notes (`current`, `recent`) are lifted to `dim`,
            # the weight the paths carry; the structural ones (`home`,
            # `parent`, `in this folder`) stay `faint`. The tiers were legible
            # only through the notes, and the notes were the lowest-contrast
            # thing on the row, so the ordering's meaning was invisible
            # (design D5). A gradient gives the grouping without spending rows
            # on separators, which a 10-row budget cannot afford.
            line.append(note, style=dim if target.kind in _IDENTITY_KINDS else faint)
        if hovered is not None and index == hovered and not is_selected:
            line.stylize(Style(bgcolor=theme_mod.semantic_color("raised")))
        out.append(line)
    return out


def _truncate_head(text: str, width: int) -> str:
    """``text`` cut from the LEFT to ``width`` cells, marked with an ellipsis.

    ``truncate_cells`` cuts the tail, which is right for a sentence and wrong
    for a path: two rows under the same parent differ only in their last
    segment, so a tail cut renders them as the same string.
    """
    if cell_len(text) <= width:
        return text
    if width <= 1:
        return "…"
    kept = text
    while kept and cell_len(kept) > width - 1:
        kept = kept[1:]
    return f"…{kept}"


class MovePickerScreen(ModalScreen[str | None]):
    """Pick a working directory; dismisses with its absolute path, or ``None``.

    Two-way like the resume picker: the caller pushes it and acts on what comes
    back, so this screen owns navigation and the caller owns the move. Esc
    answers ``None`` and the session is left exactly where it was.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "choose", "Move", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        # The readline pair as well as the arrows, for the reason the resume
        # picker carries them: every printable key belongs to the filter, so
        # these are the only other way to move a hand that is already typing.
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("backspace", "backspace", "Edit filter", show=False),
        # A PATH is long, so the readline word/line kills are not a nicety
        # here: backing out of a completed path one character at a time cost 15
        # backspaces in the UX walk, and they were the only editing key bound.
        # `ctrl+w` deletes the last path SEGMENT rather than the last word,
        # which is what "back up one directory" means on this card.
        Binding("ctrl+w", "kill_segment", "Delete segment", show=False),
        Binding("ctrl+u", "kill_query", "Clear", show=False),
        # Tab COMPLETES the highlighted row into the query rather than choosing
        # it, which is what makes the card a navigator: completing
        # `~/workspace` and pressing tab again lists inside it, so a user can
        # walk down a tree without knowing the full path. Enter still chooses,
        # so tab never costs a keystroke that Enter would have finished.
        Binding("tab", "complete", "Complete", show=False),
    ]

    def __init__(
        self,
        targets: list[MoveTarget],
        current: str = "",
        *,
        complete: Callable[[str], list[MoveTarget]] | None = None,
        self_target: Callable[[str], MoveTarget | None] | None = None,
    ) -> None:
        super().__init__()
        self._all = list(targets)
        self._current = current
        self._query = ""
        self._selected = 0
        self._offset = 0
        self._hovered: int | None = None
        # Supplied by the host, which knows the session's cwd to resolve
        # relative paths against. Optional so the widget stays testable — and
        # usable by an embedder — without one: no completer means the card
        # filters its suggestions and nothing more, which is a smaller feature
        # rather than an error.
        self._complete = complete
        # Resolves a typed path to ITSELF when it names a usable directory, so
        # a directory with no subdirectories is still selectable rather than
        # reported as "no match" (D1/U1). Supplied by the host for the same
        # reason ``complete`` is: it needs the session's cwd for relative paths.
        self._self_target = self_target
        # Filtering runs on every keystroke and again on every paint, so the
        # result is cached against the query that produced it. The path tier
        # is cached on the SAME key because it is the same keystroke's work,
        # and it is filesystem I/O — re-listing a directory per repaint is
        # exactly the cost this cache exists to avoid.
        self._rows: list[MoveTarget] = list(targets)
        self._rows_for = "\x00 never a real query"
        self._body: Static

    # -- state ---------------------------------------------------------------
    # ``visible_rows``/``filter_query``/``_card_text``, not ``visible``/``query``/
    # ``_render``: all three shorter names are Textual's own (``Widget.visible``,
    # ``DOMNode.query``, ``Widget._render``) and shadowing them breaks focus,
    # query and paint from inside the screen.
    @property
    def visible_rows(self) -> list[MoveTarget]:
        """The rows the current query admits, in the order they are offered.

        Two tiers, chosen by what was typed (see the module docstring): a path
        completes against the filesystem, anything else filters the
        suggestions. Order is never re-ranked for a fixed query, so a row
        cannot move out from under the cursor between repaints.
        """
        if self._rows_for != self._query:
            self._rows = self._resolve_rows(self._query)
            self._rows_for = self._query
        return self._rows

    def _resolve_rows(self, query: str) -> list[MoveTarget]:
        if query and looks_like_path(query) and self._complete is not None:
            try:
                rows = list(self._complete(query))
            except Exception:  # noqa: BLE001 — a failed listing is an empty one
                rows = []
            if rows:
                return rows
            # NO CHILDREN IS NOT NO MATCH. ``action_complete`` appends a
            # separator so the next tab descends, and completing INSIDE a
            # directory with no subdirectories returns nothing — so the
            # directory the user just walked to rendered as "no directory
            # matches that path", one keystroke after being a selectable row,
            # and Enter there dismissed with ``None`` (read by the caller as
            # Esc: no move, no notice). That is the dead end design D1 and UX
            # U1 found independently, and it made the picker strictly weaker
            # than the typed-path route it fronts.
            #
            # A query that RESOLVES to a readable directory is offered as
            # itself, so Enter moves there. Typing the path by hand lands here
            # too, which is why this sits in row resolution rather than in
            # ``action_complete``.
            here = self._resolve_self(query)
            if here is not None:
                return [here]
        return filter_targets(self._all, query)

    def _resolve_self(self, query: str) -> MoveTarget | None:
        """``query`` as its own row when it names a usable directory.

        Returns ``None`` for a path that does not exist or cannot be entered —
        those are genuinely "no match" and keep the existing sentence.
        """
        if self._self_target is None:
            return None
        try:
            return self._self_target(query)
        except Exception:  # noqa: BLE001 — an unresolvable path is simply not offered
            return None

    @property
    def is_path_query(self) -> bool:
        """Whether the typed text is being COMPLETED rather than filtered."""
        return bool(self._query) and looks_like_path(self._query)

    @property
    def filter_query(self) -> str:
        return self._query

    @property
    def selected_index(self) -> int:
        return self._selected

    def selected_path(self) -> str | None:
        """The highlighted directory's absolute path, or ``None`` when empty."""
        rows = self.visible_rows
        if not rows:
            return None
        return rows[min(self._selected, len(rows) - 1)].path

    # -- actions -------------------------------------------------------------
    def _dismiss_result(self, result: str | None) -> None:
        """Dismiss after releasing a hovered row's pointer shape."""
        self.styles.pointer = "default"
        self.dismiss(result)

    def action_cancel(self) -> None:
        self._dismiss_result(None)

    def action_choose(self) -> None:
        # Enter on an empty result set is not a choice, but it still CLOSES the
        # picker — which is what a user who has typed a bad filter expects, and
        # what the resume picker does.
        self._dismiss_result(self.selected_path())

    def action_complete(self) -> None:
        """Put the highlighted row's path into the query, ready to descend.

        A trailing separator is appended so the very next keystroke — or an
        immediate second tab — lists INSIDE the completed directory rather than
        re-matching it among its siblings. That is the whole walking gesture,
        and without the separator tab would appear to do nothing on the second
        press.
        """
        rows = self.visible_rows
        if not rows:
            return
        row = rows[min(self._selected, len(rows) - 1)]
        # The LABEL, not the absolute path. The row the user pressed tab on
        # reads ``~/workspace/repos/x``; inserting ``path`` put a long absolute
        # string in the header, which then head-truncated into something that
        # did not resemble the row they had just chosen (design D3).
        # ``complete_path`` expands ``~`` itself, so the tidy form round-trips.
        text = row.label or row.path
        self.set_query(text.rstrip("/") + "/")

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        self._move_to(self._selected + delta * self._page_rows())

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self.visible_rows) - 1 if to_end else 0)

    def action_backspace(self) -> None:
        if self._query:
            self.set_query(self._query[:-1])

    def action_kill_segment(self) -> None:
        """Drop the last path segment — "back up one directory".

        A trailing separator is dropped with the segment it closes, so a query
        left at ``~/a/b/`` by ``tab`` goes to ``~/a/`` in one press rather than
        stripping the separator and leaving the user where they were.
        """
        if not self._query:
            return
        trimmed = self._query.rstrip("/")
        head, sep, _ = trimmed.rpartition("/")
        self.set_query(head + sep if sep else "")

    def action_kill_query(self) -> None:
        """Clear the query, returning the card to its suggestions."""
        if self._query:
            self.set_query("")

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the query.

        Handled here rather than as bindings for the reason the resume picker
        gives: the query accepts every character, and a binding per key is a
        table of ninety-five entries that still misses the ninety-sixth.
        """
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self.set_query(self._query + char)

    # -- mouse ---------------------------------------------------------------
    # The wheel moves the cursor a row at a time, which scrolls the window with
    # it. CLAMPED, like every other movement here: a scroll gesture that
    # wrapped to the other end would read as the list resetting itself. Every
    # handler stops the event so one gesture does not also scroll the
    # transcript behind the card.
    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(1)

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self.action_move(-1)

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Primary-button click on a row moves to it.

        Button 1 only, the guard the resume picker documents: the action behind
        this rebuilds the session's runtime, which is not something a
        right-click asking for a context menu should trigger.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        rows = self.visible_rows
        if 0 <= index < len(rows):
            self._selected = index
            self._dismiss_result(rows[index].path)

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """List index under a mouse event, or ``None`` anywhere else.

        The same three guards ``session_picker._index_at`` documents, and they
        are load-bearing for the same reason: the modal's backdrop covers the
        whole screen and bubbles clicks from outside the card, the footer and
        spacer sit below the last row at offsets that resolve to real rows, and
        a false positive here MOVES THE SESSION.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        row = event.screen_y - region.y - self._header_rows()
        rows = self.visible_rows
        drawn = min(self._page_rows(), max(0, len(rows) - self._offset))
        if not 0 <= row < drawn:
            return None
        index = self._offset + row
        return index if 0 <= index < len(rows) else None

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        """The box the card's percentage sizes actually resolve in.

        ``self.size`` (this Screen's CONTENT box), not ``self.app.size``:
        ``Screen { padding: 1 }`` insets the content box, and percentage
        heights resolve against that box — so measuring the terminal
        over-counts the room and Textual clips the difference silently, off the
        bottom, taking the footer with it. Reported honestly, with the width
        floor applied where the preference is (``_card_width``) rather than in
        the measurement it is applied to.
        """
        try:
            size = self.size
            if not size.width or not size.height:  # not laid out yet
                size = self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        return max(1, size.width), max(8, size.height)

    def _card_width(self) -> int:
        """Content cells the card may use, measured against the terminal.

        The floor applies only while it FITS: a minimum width is a preference
        and the terminal is not, so below it the card gives up the breathing
        margin before it gives up content.
        """
        width, _ = self._screen_size()
        padding = PICKER_PADDING_CELLS * 2
        room = width - PICKER_WIDTH_MARGIN - padding
        if room < PICKER_MIN_WIDTH:
            return max(1, width - padding)
        return min(PICKER_MAX_WIDTH, room)

    def _page_rows(self) -> int:
        """Directory rows the card can actually DRAW right now.

        Chrome is reserved FIRST and the list takes what is left, so the cursor
        can never sit on a row the card did not render — Enter on one would
        move somewhere the user could not see.
        """
        _, height = self._screen_size()
        budget = int(height * CARD_MAX_HEIGHT_FRACTION) - CARD_PADDING_ROWS - CARD_CHROME_ROWS
        return max(1, min(PAGE_ROWS_MAX, budget))

    def _header_rows(self) -> int:
        """Rows above the first directory row: the header and its rule."""
        return 2

    # -- internals -----------------------------------------------------------
    def set_query(self, query: str) -> None:
        """Apply a query and put the cursor on the FIRST match.

        Not the nearest surviving row: clamping the old index means narrowing a
        list usually lands the cursor on the LAST match, so the row Enter would
        take is the least related one still standing. Every finder the user has
        met — and this app's own command, ask and resume pickers — answers a
        narrowing query with its best match at the top.
        """
        if query == self._query:
            return
        self._query = query
        self._selected = 0
        self._offset = 0
        self._repaint()

    def _move_to(self, index: int) -> None:
        rows = self.visible_rows
        if not rows:
            self._selected = 0
            self._offset = 0
            self._repaint()
            return
        # CLAMPED, never wrapping, because `session_picker._move_to` clamps and
        # this card is its twin down to the CSS — AGENTS.md names that picker
        # explicitly ("`session_picker._move_to` already clamps, and it is a
        # full surface too"). NOT on the `/settings` exception, which is about
        # a full PAGE and would equally license clamping `/model` and
        # `/command` — the same paragraph forbids those by name. A Down at the
        # bottom that silently returned to the top reads as the list having
        # reset itself.
        self._selected = max(0, min(len(rows) - 1, index))
        # Scroll only far enough to keep the cursor on screen, so the list is
        # stable while paging through the middle of it.
        page = self._page_rows()
        if self._selected < self._offset:
            self._offset = self._selected
        elif self._selected >= self._offset + page:
            self._offset = self._selected - page + 1
        self._offset = max(0, min(self._offset, max(0, len(rows) - page)))
        self._repaint()

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="move-picker"):
            self._body = Static(self._card_text(), id="move-picker-body")
            yield self._body

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: every column and the page size come from the screen."""
        self._move_to(self._selected)

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
        label_style = Style(color=theme_mod.semantic_color("label"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        width = self._card_width()
        rows = self.visible_rows

        # The typed query is the user's only receipt that typing reached this
        # modal, so a narrow card sheds the title before it sheds the query —
        # truncating the assembled line does the opposite. The MODE word
        # ("filter"/"path") rides with it because the two answer the same
        # keystroke differently, and an empty list means different things in
        # each.
        title = "Move to a directory"
        header = Text(no_wrap=True, overflow="ellipsis")
        if self._query:
            lead = "  path " if self.is_path_query else "  filter "
            compact_lead = lead.strip() + " "

            # A long PATH is cut from the LEFT, a long filter word from the
            # right. The reason is the same one the rows use: the end of a path
            # is the fragment the user is currently typing and the part the
            # completion is matching on, so a tail cut hides the only characters
            # that are changing — measured on a real capture, where a deep path
            # rendered as `…/ho…` and the user could not see what they had
            # typed. A filter word has its meaning at the front instead.
            def _fit(text: str, room: int) -> str:
                return (
                    _truncate_head(text, room) if self.is_path_query else truncate_cells(text, room)
                )

            titled = cell_len(title) + cell_len(lead) + cell_len(self._query)
            if titled <= width:
                header.append(title, style=fg)
                header.append(lead, style=faint)
                header.append(self._query, style=label_style)
            elif cell_len(compact_lead) < width:
                header.append(compact_lead, style=faint)
                header.append(
                    _fit(self._query, width - cell_len(compact_lead)),
                    style=label_style,
                )
            else:
                header.append(_fit(self._query, width), style=label_style)
        else:
            header.append(title, style=fg)

        out = Text()
        out.append_text(header)
        out.append("\n")
        out.append("─" * width, style=faint)
        out.append("\n")

        page = self._page_rows()
        counter: tuple[int, int, int] | None = None
        if not rows:
            # Two sentences for two situations: a filter that matched nothing
            # is a mistyped word, a path that matched nothing is a directory
            # that does not exist. The header already echoes the query, so
            # neither repeats it.
            #
            # A path that RESOLVES is no longer one of these cases: it comes
            # back as its own row from ``_resolve_rows``, so "no directory
            # matches that path" is now only said about a path that genuinely
            # does not resolve — which is what made the sentence false before
            # (design D1, UX U3: it was shown for a directory the user had
            # just selected).
            out.append(NO_PATH_NOTICE if self.is_path_query else NO_MATCH_NOTICE, style=dim)
        else:
            window = rows[self._offset : self._offset + page]
            for index, line in enumerate(
                render_rows(
                    window,
                    self._selected - self._offset,
                    width,
                    None if self._hovered is None else self._hovered - self._offset,
                )
            ):
                if index:
                    out.append("\n")
                out.append_text(line)
            if len(rows) > page:
                counter = (self._offset + 1, self._offset + len(window), len(rows))

        # Body, one quiet row, then the card's META — the position and the key
        # hints, which are statements ABOUT the list rather than entries in it,
        # so they travel together at the bottom. This is the usage and resume
        # cards' grammar. The counter is emitted ONLY when the list scrolls:
        # printing an empty line in its place leaves two blank rows and pushes
        # the keys away from the block they belong to.
        out.append("\n\n")
        if counter is not None:
            first, last, total = counter
            out.append("showing ", style=faint)
            out.append(f"{first:,}–{last:,}", style=dim)
            out.append(" of ", style=faint)
            out.append(f"{total:,}", style=dim)
            out.append("\n")
        for index, (key, what) in enumerate(
            _footer_hints(width, scrolls=counter is not None, empty=not rows)
        ):
            if index:
                out.append(" · ", style=faint)
            out.append(key, style=dim)
            if what:
                out.append(f" {what}", style=faint)
        return out


#: Footer hints, MOST disposable first. ``enter``/``esc`` are never dropped:
#: between them they are how the card is used and how it is left. ``tab`` sits
#: above them because completion is the half of this picker that reaches a
#: directory the suggestions never guessed — a user who does not know it exists
#: cannot discover it from the list.
_FOOTER_HINTS: tuple[tuple[str, str], ...] = (
    ("↑↓", "move"),
    ("pgup/pgdn", "page"),
    ("type", "to filter or path"),
    ("tab", "complete"),
    ("enter", "move"),
    ("esc", "cancel"),
)
#: ``type to filter or path`` sheds LATE, and that is the whole of design D2.
#: It is the only thing anywhere on this card that says a second input mode
#: exists, and path completion is the half of the feature that makes any
#: directory on the machine reachable. Shedding it first meant the mode was
#: advertised only to users who had already found it.
#:
#: Paging is inferable from a scrollbar-less list with a `showing N–M of T`
#: counter; a hidden second input mode is not inferable from anything. So
#: ``pgup/pgdn`` goes first in both orders and ``type`` outlives ``tab``.
_FOOTER_DROP_ORDER = ("pgup/pgdn", "tab", "type", "↑↓")

#: Drop order for a list that SCROLLS — which a freshly opened picker almost
#: always is: ``PAGE_ROWS_MAX`` is 10 and the default suggestion set runs to
#: 11-12 rows, so this is the order the OPENING frame uses at every width.
#: That is precisely why ``type`` may not lead it. ``pgup/pgdn`` is kept ahead
#: of ``tab`` here (the reverse of the non-scrolling order) because paging is
#: real on a scrolling list and is the fastest way through it.
_FOOTER_DROP_ORDER_SCROLLING = ("tab", "pgup/pgdn", "type", "↑↓")


#: The footer for a query that matched nothing. Movement, paging, completion
#: and `enter move` all describe a list that is not there, so the only honest
#: thing the row can say is how to get back to one. `backspace` is the key that
#: widens the query and the one a user in this state is already reaching for.
#: Stated as a hint pair like every other so it sheds and renders identically —
#: the same correction `session_picker._EMPTY_HINT` records.
_EMPTY_HINT: tuple[str, str] = ("backspace", "to widen")


def _footer_hints(
    width: int, *, scrolls: bool = False, empty: bool = False
) -> list[tuple[str, str]]:
    """The key hints that fit in ``width`` cells, dropping the least needed."""
    if empty:
        # Offering movement and `enter move` for an empty list advertises
        # actions that do nothing, and `tab complete` promises to complete a
        # row that is not there. `esc` stays: leaving is still available and is
        # the other thing a user wants here.
        return _shed_to_width([_EMPTY_HINT, ("esc", "cancel")], (_EMPTY_HINT[0],), width)
    hints = list(_FOOTER_HINTS)
    drop_order = _FOOTER_DROP_ORDER_SCROLLING if scrolls else _FOOTER_DROP_ORDER
    return _shed_to_width(hints, drop_order, width)


def _shed_to_width(
    hints: list[tuple[str, str]], drop_order: tuple[str, ...], width: int
) -> list[tuple[str, str]]:
    """``hints`` reduced to fit ``width`` cells, dropping in ``drop_order``.

    The last resort drops the LABELS and keeps the keys: two bare keys still
    say which keys exist, which is more than a clipped row says. Mirrors
    ``session_picker._shed_to_width`` so the two footers cannot drift into two
    shed policies.
    """

    def cells(pairs: list[tuple[str, str]]) -> int:
        return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
            0, len(pairs) - 1
        )

    for droppable in drop_order:
        if cells(hints) <= width:
            return hints
        hints = [pair for pair in hints if pair[0] != droppable]
    if cells(hints) <= width:
        return hints
    return [(key, "") for key, _ in hints]
