"""The `/links` picker: choose a URL out of the conversation and open it.

A content-sized card over the dimmed transcript, following
:class:`~local_operator.tui.widgets.move_picker.MovePickerScreen`: the caller
pushes it with a callback, the screen owns navigation, and it dismisses with the
chosen :class:`~local_operator.tui.link_targets.LinkTarget` or ``None``.

WHY A LIST RATHER THAN A CLICK
==============================

The gesture the user reaches for — click the link — is the one gesture the
TERMINAL cannot perform. Textual claims the mouse at startup
(``\\x1b[?1000h``/``\\x1b[?1003h``), and a terminal that is reporting mouse
events to an application does not run its own click-to-open, so the OSC-8
hyperlink is on screen and correct and clicking it does nothing. Ghostty's
shift+click bypass is the terminal's own and is documented as undetectable by
the program, so the app has to offer the URL itself, and a LIST is the shape
that needs no pointer: the transcript is a scrollback with no notion of "the
cell under the cursor", and a key that guessed at "the nearest URL" would open
the wrong one silently. Choosing from the URLs of the conversation is explicit,
works the same whether the composer or a transcript row holds focus, and needs
nothing of the terminal's mouse protocol.

The card still ANSWERS a click — the app receives mouse events even though the
terminal withholds its own gesture, so a click on a row chooses that row. That
is a route to the URL rather than a click on the URL, which is what makes it
work where the link itself does not: nothing depends on where in the string the
pointer landed.

``enter`` opens; the row's whole URL is what is opened, never the truncated
string the card paints.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.link_targets import LinkTarget
from local_operator.tui.widgets.tool_card import truncate_cells

#: Rows the card spends on anything but a link: the title, the rule under it,
#: the blank row above the footer, and the footer itself. Counted from what
#: :meth:`LinkPickerScreen._card_text` emits rather than derived, because the
#: row budget is the screen height minus exactly this.
CHROME_ROWS = 4

#: `padding: 1 2` on the card in the stylesheet: four cells, both sides.
CARD_PADDING_CELLS = 4

#: `padding: 1 2` again, counted in ROWS: the card spends two of them before
#: any content, which is why the row budget starts below this.
CARD_PADDING_ROWS = 2

#: Most links the card will show at once. The list is ordered newest-first, so
#: a longer conversation scrolls rather than growing a card that covers the
#: transcript it is describing — and AGENTS.md's rule that a card must not grow
#: past its box makes an unbounded list a defect, not a preference.
MAX_VISIBLE_ROWS = 12

#: Widest the card will grow, before the screen's own width is considered.
#: The same 100 the stylesheet carries — see :meth:`LinkPickerScreen._card_width`
#: for why the number is stated in both places rather than read back.
CARD_MAX_WIDTH = 100

#: Cells the `❯ ` cursor column costs on every row, selected or not.
CURSOR_CELLS = 2

#: Shown in place of the card on a terminal too small to hold it — a sibling of
#: the card, so the card's own contract ("it drew, or it drew nothing") is
#: untouched. One row, no ground of its own.
TOO_SMALL_NOTICE = "terminal too small for /links · esc"

#: The same notice for a terminal too NARROW to hold the full one. `esc` is the
#: actionable half and it is what a truncation sheds first, leaving
#: `terminal too small for` — the command named, the way out gone, on a screen
#: where the notice is the only thing painted (design round 1, D1). `copy_picker`
#: keeps the two spellings for exactly this case and chooses on the width the
#: screen actually resolved; this is the same discipline rather than a second
#: one. Measured on the dark ramp, the long form needs 35 content cells, i.e. 37
#: terminal columns, so 36x8 and below is where it used to clip and 38x8 — the
#: narrowest size that draws the card at all — is where it is whole.
TOO_SMALL_NOTICE_SHORT = "too small · esc"

#: Footer hints, in the order the app's other cards list them: movement, the
#: action, the way out — `session_picker`'s and `move_picker`'s order, so a user
#: meets the same three clauses in the same sequence on every card.
_FOOTER_HINTS: tuple[tuple[str, str], ...] = (
    ("↑↓", "move"),
    ("enter", "open"),
    ("esc", "cancel"),
)

#: Most disposable first, the rule `session_picker` records on its own table:
#: `enter` and `esc` are never shed, because between them they are how the card
#: is used and how it is left. `↑↓` is the one that can go — the cursor is drawn
#: on the row it is on, so a user who never reads the hint can still see where
#: they are standing, while `enter` is the only way to act.
_FOOTER_DROP_ORDER = ("↑↓",)


class LinkPickerScreen(ModalScreen[LinkTarget | None]):
    """Pick a URL; dismisses with the chosen target, or ``None``.

    The list is a SNAPSHOT taken when the screen is built, for the reason
    ``CopyPickerScreen`` records: a message settling underneath would insert at
    the top of a newest-first list and shift the rows below it, including the
    one the user is aiming at.

    Esc dismisses, and ``ctrl+c`` is deliberately NOT bound: it is the global
    interrupt in this app, and a modal claiming it would change what stopping a
    turn means depending on whether a picker happened to be open.

    ``ALLOW_SELECT = False`` for the reason ``CopyPickerScreen`` states in full:
    the card is CHROME, and Textual copies the whole widget on a chained click
    wherever it is selectable. On this card that would put the link list itself
    on the clipboard on a double-click.
    """

    ALLOW_SELECT = False

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "choose", "Open", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
    ]

    def __init__(self, targets: list[LinkTarget]) -> None:
        super().__init__()
        self._targets = list(targets)
        self._selected = 0
        #: First row of the window on screen. Derived from the cursor in
        #: :meth:`_move_to`, so the two cannot disagree about which row the
        #: user is on.
        self._offset = 0
        self._body: Static | None = None
        #: List index under the pointer, or ``None`` when it is not over a row.
        #: The card answers a click by opening a URL, so the highlight is the
        #: only thing that says a row is clickable and which one a click will
        #: hit — `copy_picker` calls that pair "the ENTIRE affordance for the
        #: mouse", and this card had neither half of it (design round 1, D3).
        self._hovered: int | None = None

    # -- geometry ------------------------------------------------------------

    def _content_size(self) -> tuple[int, int]:
        """The Screen CONTENT box — the terminal, inset by ``Screen { padding: 1 }``.

        Guarded rather than read straight: a screen built standalone in a test
        has no app, and the fallback is the CONTENT box of the 80x24 host the
        other cards assume (``overlay.FALLBACK_SCREEN``, less the screen's own
        padding), so a measurement taken without a pilot is a smaller answer
        rather than an exception.

        The floors are 1x1 and NOT a comfortable size. ``overlay.screen_size``
        floors at 20x8 for arithmetic sanity, and copying that here made
        :meth:`is_drawable` unreachable: with a floor of 8 rows there was always
        room for a one-link card, so the too-small notice was dead code and a
        genuinely short terminal got a card clipped off the bottom — the one
        outcome the notice exists to prevent. A card that cannot fit must be
        able to SAY so.
        """
        try:
            size = self.app.size
        except Exception:  # noqa: BLE001 — no app: the test-host fallback
            return (78, 22)
        return (max(1, size.width - 2), max(1, size.height - 2))

    def _card_width(self) -> int:
        """Text columns the card draws in: its own content, capped to the screen.

        CONTENT-SIZED, like `/move`. The card is an overlay on a conversation the
        user has not left, so it takes the columns its widest row needs and gives
        the transcript back the rest — a full-width card for three short URLs
        reads as a panel that lost its content.

        The title and the FULL footer are part of the measurement, not just the
        rows: a footer that shed a hint to fit a card the footer itself had sized
        would put the two rules in each other's way, and a card narrower than its
        own key hints is the one state it must not reach.

        The `max-width: 100` in the stylesheet is repeated here rather than read
        back, because the card is sized by this number. AGENTS.md records what a
        second cap costs when the two disagree silently
        (``SessionPickerScreen``: the shipped sheet capped at 80 over a Python
        budget of 74, and the narrower won with nothing to read back), so the
        number is stated once in the sheet and once here, in the same breath.
        """
        room = max(1, min(self._content_size()[0], CARD_MAX_WIDTH) - CARD_PADDING_CELLS)
        natural = max(
            [cell_len("Open a link"), _hint_cells(list(_FOOTER_HINTS))]
            + [
                CURSOR_CELLS + cell_len(target.url) + 2 + cell_len(target.sender)
                for target in self._targets[: self._row_budget()]
            ]
        )
        return max(1, min(natural, room))

    def _row_budget(self) -> int:
        """Link rows the card may draw, or ``0`` when it cannot draw one.

        ``avail`` is the rows left for link rows AND, when the list is windowed,
        the counter that says so. The branch is written this way rather than as a
        subtraction so the two answers cannot oscillate — a budget that dropped a
        row for a counter which then no longer appeared was the shape this
        avoids — and a list that needs a counter but has room for only one row
        returns ``0`` rather than one row plus the card's own overflow: the
        footer is the thing that would be clipped, and a card that cannot state
        how to leave it is the trap :meth:`is_drawable` exists to refuse.
        """
        avail = min(MAX_VISIBLE_ROWS, self._content_size()[1] - CARD_PADDING_ROWS - CHROME_ROWS)
        if avail < 1:
            return 0
        if len(self._targets) <= avail:
            return len(self._targets)
        return avail - 1 if avail >= 2 else 0

    def is_drawable(self) -> bool:
        """Whether the card can be drawn at all.

        One link row plus the chrome is the floor: below it there is no frame in
        which the card says anything a user can act on, and a dimmed screen with
        an empty card on it reads as a crash.
        """
        return self._row_budget() >= 1 and self._card_width() >= cell_len("esc cancel")

    # -- navigation ----------------------------------------------------------

    def _move_to(self, index: int) -> None:
        """Move the cursor to ``index``, WRAPPING at both ends.

        Wrapping, not clamping, and the distinction is the one AGENTS.md draws:
        this card is a short list OVERLAID on a screen the user is still looking
        at, so coming round on a deliberate arrow press is a shortcut to a row
        that is already visible — the case the wrap rule is written for, and the
        rule ``model_picker`` and ``command_picker`` keep. The full-surface
        pickers clamp because their list is the whole page; this card's list is
        one to twelve rows over a conversation the user has not left.
        """
        count = len(self._targets)
        if count == 0:
            return
        self._selected = index % count
        # The window follows the cursor and nothing else: there is no second
        # position (no scrollbar, no independent viewport) that could disagree
        # with it, which is why "one gesture owns the viewport" is satisfied by
        # construction here rather than by a rule.
        rows = self._row_budget()
        if rows < 1:
            # Nothing is painted at this size (see `is_drawable`), so the window
            # is a value nothing reads. Pinned to the top rather than left to the
            # arithmetic below, which would walk the offset past the end of a
            # list it is not showing.
            self._offset = 0
            self._repaint()
            return
        if self._selected < self._offset:
            self._offset = self._selected
        elif self._selected >= self._offset + rows:
            self._offset = self._selected - rows + 1
        self._repaint()

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        """Page CLAMPED — a page is a scroll gesture, and a scroll that wrapped
        to the other end of the list reads as the list resetting itself."""
        count = len(self._targets)
        if count == 0:
            return
        step = delta * self._row_budget()
        self._move_to(max(0, min(count - 1, self._selected + step)))

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self._targets) - 1 if to_end else 0)

    def action_cancel(self) -> None:
        self._dismiss_result(None)

    def action_choose(self) -> None:
        rows = self._targets
        if not rows:
            self.app.bell()
            return
        self._dismiss_result(rows[min(self._selected, len(rows) - 1)])

    def _dismiss_result(self, result: LinkTarget | None) -> None:
        """Dismiss once, and only while still mounted.

        A chained click (or a key landing on a screen that is already leaving)
        can reach these handlers twice; the second call would push a duplicate
        dismissal into the app's callback. ``is_mounted`` is the honest test —
        it is the same guard ``CopyPickerScreen._dismiss_result`` carries.
        """
        if self.is_mounted:
            self.dismiss(result)

    # -- mouse ---------------------------------------------------------------

    # Every handler stops the event, so one gesture cannot also scroll or select
    # the transcript behind the card, and the wheel CLAMPS for the reason
    # ``action_page`` does: a wheel flick that jumped to the other end of the
    # list would read as the list resetting itself.

    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        count = len(self._targets)
        if count:
            self._move_to(min(count - 1, self._selected + 1))

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._move_to(max(0, self._selected - 1))

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Primary-button click on a row opens it.

        This works where clicking the LINK does not, and that is the whole
        point: the terminal's own gesture is suppressed while the app holds
        mouse reporting, but the app receives the click and can act on it. The
        row is the target rather than the URL's cells, so nothing depends on
        where in the string the pointer landed.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        rows = self._targets
        if 0 <= index < len(rows):
            self._selected = index
            self._dismiss_result(rows[index])

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        """Highlight the row under the pointer, and hold up the hand there.

        `move_picker.on_mouse_move`'s shape, deliberately: same hit-test (the
        one :meth:`on_click` already uses, so the highlight and the click
        cannot disagree about where a row is), same repaint-on-change, and the
        hand only over a row. The pointer shape is set on every move rather
        than only on a change — it is cheap, and the property's own observer
        no-ops when the shape did not move, which is what keeps the hand from
        sticking to a row the pointer has left.
        """
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        """The hand and the highlight both end at the card's edge.

        A shape that never resets is a cursor the user stops trusting, which
        is `test_pointer_shapes`' rule for every picker here.
        """
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """List index under a mouse event, or ``None`` anywhere else.

        The guards are ``move_picker._index_at``'s, and they are load-bearing
        for a sharper reason here: the backdrop covers the whole screen and
        bubbles clicks from outside the card, and a false positive on this card
        hands a URL to the browser.
        """
        body = self._body
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        row = event.screen_y - region.y - self._header_rows()
        drawn = min(self._row_budget(), max(0, len(self._targets) - self._offset))
        if not 0 <= row < drawn:
            return None
        index = self._offset + row
        return index if 0 <= index < len(self._targets) else None

    def _header_rows(self) -> int:
        """Title plus rule — the rows above the first link row."""
        return 2

    # -- render --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Container(classes="link-picker"):
            self._body = Static(self._card_text(), id="link-picker-body")
            yield self._body
        # A sibling of the card, never a row inside it: the card's contract is
        # two-sided and pinned (either it drew a usable frame or it drew
        # nothing), and a notice inside it would satisfy "the card drew
        # something".
        self._too_small = Static(TOO_SMALL_NOTICE, id="link-picker-too-small")
        yield self._too_small

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: the row budget and both columns come from the screen."""
        self._move_to(self._selected)

    def _repaint(self) -> None:
        body = self._body
        if body is None or not body.is_mounted:
            return
        drawable = self.is_drawable()
        body.update(self._card_text() if drawable else Text())
        # Hidden, not merely emptied: a card painting nothing still claims its
        # two padding rows, and at these sizes that is how a screen grows past
        # its own height and becomes scrollable, which AGENTS.md calls always a
        # bug here.
        card = body.parent
        if card is not None:
            card.display = drawable
        notice = getattr(self, "_too_small", None)
        if notice is not None and notice.is_mounted:
            notice.display = not drawable
            if not drawable:
                notice.update(self._too_small_text(notice))

    def _too_small_text(self, notice: Static) -> str:
        """The notice, in the widest form the screen actually resolved.

        `copy_picker`'s rule, and it is taken from the NARROWEST width any
        resolved source reports rather than the screen's own: the card is
        measured inside `Screen { padding: 1 }`, so a width read from the
        terminal alone can be two cells larger than the box the notice is
        painted in — and an uncertain measurement must not select the form
        that can clip. Nothing resolved (not yet laid out) prefers the short
        one, which fits everywhere the long one does.
        """
        candidates = [
            size.width
            for size in (getattr(notice, "size", None), getattr(self, "size", None))
            if size is not None and size.width
        ]
        columns = min(candidates) if candidates else 0
        return TOO_SMALL_NOTICE if cell_len(TOO_SMALL_NOTICE) <= columns else TOO_SMALL_NOTICE_SHORT

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads.

        Empty when the card is not drawn, so a test cannot assert against rows
        that never reached the terminal.
        """
        if not self.is_drawable():
            return []
        return [line.plain for line in self._card_text().split("\n")]

    def _card_text(self) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        fg = Style(color=theme_mod.semantic_color("fg"))

        width = self._card_width()
        rows = self._targets

        out = Text()
        out.append("Open a link", style=fg)
        out.append("\n")
        out.append("─" * width, style=faint)
        out.append("\n")

        window = rows[self._offset : self._offset + self._row_budget()]
        for index, line in enumerate(
            self._row_lines(
                window,
                width,
                None if self._hovered is None else self._hovered - self._offset,
            )
        ):
            if index:
                out.append("\n")
            out.append_text(line)

        # Body, one quiet row, then the card's META — the position and the key
        # hints, which are statements ABOUT the list rather than entries in it
        # and so travel together at the bottom. This is the `/move` and
        # `/usage` cards' grammar. The counter is emitted only while the list
        # scrolls: an empty line in its place would leave two blank rows and
        # push the keys away from the block they belong to.
        #
        # EVERY WORD HERE IS AT LEAST `dim`, the step `session_picker` took on
        # this same ground and for the reason it records there: the card's
        # ground is `$lo-overlay`, and `faint` on it measures 1.49:1 — an
        # overlay lifts the ground without lifting the text with it. That put
        # the two words telling a user how to act and how to leave among the
        # faintest pixels on the screen. Only the ` · ` SEPARATORS stay
        # `faint`, which is what "meta separator" names (design round 1, D2).
        out.append("\n\n")
        if len(rows) > len(window):
            last = self._offset + len(window)
            out.append("showing ", style=dim)
            out.append(f"{self._offset + 1}–{last}", style=dim)
            out.append(" of ", style=dim)
            out.append(f"{len(rows)}", style=dim)
            out.append("\n")
        for index, (key, what) in enumerate(_footer_hints(width)):
            if index:
                out.append(" · ", style=faint)
            out.append(key, style=dim)
            if what:
                out.append(f" {what}", style=dim)
        return out

    def _row_lines(
        self, window: list[LinkTarget], width: int, hovered: int | None = None
    ) -> list[Text]:
        """One line per link: cursor, URL, and which side it came from.

        The URL is truncated from the RIGHT and the sender hint is dropped
        before it is cut. Both are deliberate: the head of a URL is the host,
        which is what tells two rows apart at a glance — a tail cut would render
        two links to different repositories as the same string — and the hint is
        only the reason a row is offered, so on a narrow card it is what the
        user can spare. The same two rules ``move_picker.render_rows`` applies
        to a path and its note.

        ``hovered`` is the window-relative row under the pointer, and the
        highlight is `raised` on it — the ground `move_picker` and
        `session_picker` use, and NOT the selection's `tint-select`: hover is a
        pointer cue and the cursor is the selection, so the two must not be
        confusable. The selected row is skipped because it already carries the
        ground (and the `❯` that says which row ``enter`` takes).
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        accent = Style(color=theme_mod.semantic_color("accent"))
        fg = Style(color=theme_mod.semantic_color("fg"))

        lines: list[Text] = []
        for index, target in enumerate(window):
            is_selected = index + self._offset == self._selected
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("❯ " if is_selected else "  ", style=accent if is_selected else faint)
            room = max(1, width - CURSOR_CELLS)
            hint = target.sender
            hint_cells = cell_len(hint) + 2 if hint else 0
            if hint_cells and cell_len(target.url) + hint_cells > room:
                hint, hint_cells = "", 0
            line.append(
                truncate_cells(target.url, max(1, room - hint_cells)),
                style=fg if is_selected else dim,
            )
            if hint:
                pad = room - cell_len(target.url) - cell_len(hint)
                if pad > 0:
                    line.append(" " * pad)
                # The sender is `dim`, the step the `move` and `session`
                # cards' identity notes take on this ground: at `faint` it
                # measured 1.49:1 on the card and read as a smudge rather
                # than as part of the row it belongs to (design round 1, D2).
                line.append(hint, style=dim)
            if hovered is not None and index == hovered and not is_selected:
                line.stylize(Style(bgcolor=theme_mod.semantic_color("raised")))
            lines.append(line)
        return lines


def _hint_cells(pairs: list[tuple[str, str]]) -> int:
    """Cells a row of key hints occupies, separators included."""
    return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
        0, len(pairs) - 1
    )


def _footer_hints(width: int) -> list[tuple[str, str]]:
    """The key hints that fit in ``width`` cells, shedding the least needed.

    Movement goes first and ``esc`` never goes, because the last hint standing
    has to be the way out — the same shed policy ``move_picker`` and
    ``session_picker`` share.
    """
    hints = list(_FOOTER_HINTS)
    for droppable in _FOOTER_DROP_ORDER:
        if _hint_cells(hints) <= width:
            return hints
        hints = [pair for pair in hints if pair[0] != droppable]
    if _hint_cells(hints) <= width:
        return hints
    return [(key, "") for key, _ in hints]
