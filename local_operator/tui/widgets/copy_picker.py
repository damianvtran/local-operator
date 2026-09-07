"""The `/copy` picker: choose an answer, a code block or a quote to copy.

A fullscreen modal over the dimmed transcript, following
:class:`~local_operator.tui.widgets.session_picker.SessionPickerScreen`: the
caller pushes it with a callback, the screen owns navigation, and it dismisses
with the chosen :class:`CopyTarget` or ``None``.

It returns the TARGET, not its text, so the caller can read ``truncated`` and
``copy_message`` without re-deriving them. The clipboard write is the caller's:
it goes through ``_put_on_clipboard``, the one receipt shared with the drag and
composer gestures, so a per-gesture toast cannot reappear here.

The tree itself is a SNAPSHOT taken when the screen was built. See
:class:`CopyPickerScreen` for why it does not live-update.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.console import Console
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.copy_targets import CopyTarget, FlatNode, flatten_targets
from local_operator.tui.markdown_theme import IslandSyntaxTheme
from local_operator.tui.widgets.tool_card import truncate_cells

#: Rows the tree is guaranteed even on a short terminal. It floors the ROW
#: BUDGET the tree and preview divide, not the tree's own share — porting the
#: constant onto ``tree_rows`` instead yields a visibly different widget on
#: every short tree, which is the common case.
MIN_TREE_ROWS = 3
#: Rows the card spends below the tree that are NOT part of the row budget:
#: the rule under the tree, the rule under the preview, and the footer. The
#: preview's own header row comes out of the preview's share, matching the
#: reference. Counted from what :meth:`CopyPickerScreen._card_text` actually
#: emits — the reference's equivalent constant counts hand-drawn borders,
#: which are CSS here, so it cannot be copied.
CHROME_ROWS_BELOW_TOP = 3
#: Title row plus the rule under it, matching `/analytics`' card.
HEADER_ROWS = 2
#: `padding: 1 2` on the card: two rows and four cells.
CARD_PADDING_ROWS = 2
CARD_PADDING_CELLS = 4
#: Fraction of the screen the card occupies; mirrors the stylesheet.
CARD_HEIGHT_FRACTION = 0.9
#: Each level of depth indents by three cells (`│  ` or `├─ `).
GUTTER_CELLS = 3
#: Cells the `❯ ` cursor column costs on every row, selected or not.
CURSOR_CELLS = 2
#: Source lines wrapped around the preview's current offset. A 20,000-line
#: answer costs 2.8 s of loop CPU to wrap whole (measured), which is why a
#: budget exists at all; 200 lines costs 23 ms.
#:
#: It is a WINDOW, not a prefix. It used to cap the wrap at the first 200
#: source lines, which was invisible while only ~15 rows were ever shown —
#: but the moment the preview scrolls, a prefix budget becomes a dead end the
#: user walks into: scrolling a 600-line answer stops at source line 200 with
#: a remainder that will not go down, and the surface looks broken. Sliding
#: the window with the offset keeps the same bounded cost while every line
#: stays reachable. The overflow marker is still computed from the true SOURCE
#: line count, never from the wrapped rows, so the budget remains an
#: optimisation the user cannot observe.
PREVIEW_WRAP_BUDGET = 200
#: Granularity the wrap window's start is snapped to. Half the budget, so
#: every reachable offset has at least this many lines wrapped below it —
#: more than any terminal draws — while the memo holds one entry per stride
#: rather than one per scrolled line. See
#: :meth:`CopyPickerScreen._wrap_window_start`.
PREVIEW_WRAP_STRIDE = PREVIEW_WRAP_BUDGET // 2
#: Content rows the preview keeps whatever the tree would like to have, plus
#: its header row. The tree used to take ``available // 2`` unconditionally
#: while the preview's cap-at-``len`` generosity only ever ran downward, so a
#: 19-row tree at 100x30 hid ten rows while SEVEN preview rows sat empty under
#: a one-line preview (design round 1, D1). Reserving the preview a floor and
#: giving the tree the rest inverts that: five content rows is enough to judge
#: "is this the block I meant" (the preview's whole job), and the tree gets
#: every row it can actually fill.
MIN_PREVIEW_ROWS = 6
#: Cells the scroll gutter costs a tree row: the track column plus one cell of
#: gap so the thumb does not touch the hint.
GUTTER_THUMB_CELLS = 2
#: Spare cells above :meth:`CopyPickerScreen._min_flat_width` before the gutter
#: is drawn at all. The gutter widens the narrowest drawable row, so drawing it
#: unconditionally would move the drawability threshold and newly HIDE the card
#: on terminals that draw one today. It is shed instead, the discipline
#: :meth:`CopyPickerScreen._footer_text` already applies to hints.
GUTTER_SHED_HEADROOM = 4
#: Footer hints, widest first, paired with the order they are SHED in. The
#: footer is the only statement of how to leave, so a narrow card drops the
#: movement hints rather than the whole row — `esc quit` is eight cells and
#: fits terminals the full forty-five-cell footer does not. This mirrors
#: `session_picker._shed_to_width`, which drops hints in a fixed order for the
#: same reason; hiding the card over a footer that merely needed shortening
#: would blank a modal the user is looking at.
#:
#: `shift+↑↓ preview` sits at index 1 deliberately: `_footer_text` sheds from
#: the FRONT, so the ladder is `↑↓ move` → `shift+↑↓ preview` → `enter copy`,
#: which is the order these earn their cells. A user on a card narrow enough
#: to shed has a smaller preview to scroll, and the arrows are the thing a
#: list surface is assumed to have anyway.
#:
#: The MOUSE is deliberately absent. `↑↓/click move · enter/dblclick copy` is
#: 46 cells spent on something a mouse user does not need to be told: the
#: hover highlight and the hand pointer ARE the affordance and they
#: demonstrate themselves the instant the pointer crosses a row. Neither
#: `session_picker` nor `command_picker` advertises its click either. The
#: scarce cells go to the preview scroll, which is genuinely undiscoverable
#: (design round 1, D8; UX round 1, §3.3).
#:
#: `shift+↑↓`, not `⇧↑↓`: `settings_view` spells this modifier out in both of
#: its footers (`shift+↑↓ reorder · d delete`), and one glyph style per app
#: beats a shorter row. At 50 cells it still fits the 50-cell card a 60-column
#: terminal draws.
#: Named because it is the one hint that is CONDITIONAL — see `_footer_text`.
PREVIEW_HINT = "shift+↑↓ preview"
FOOTER_HINTS = ("↑↓ move", PREVIEW_HINT, "enter copy", "esc quit")
#: Never shed: without it the card cannot say how to leave.
FOOTER_ESSENTIAL = "esc quit"
#: Rows the card cannot do without: title, rule, one tree row, rule, the
#: preview's header, rule, footer. Below this the card draws NOTHING rather
#: than laying out rows the box cannot hold — `overflow: hidden` clips from
#: the bottom, so a card one row too tall loses the footer specifically.
MIN_CARD_INNER_ROWS = 7
#: Shown INSTEAD of the card when the terminal cannot hold one. It names the
#: command so the frame explains itself, and `esc` because that is the only
#: thing the user can do from here — the same two facts the footer carries,
#: in the cells that remain. Deliberately does not contain "Copy to
#: clipboard": that string is what the pinned tests use to assert the card is
#: absent, and a notice that tripped it would make the guard unable to fail.
TOO_SMALL_NOTICE = "terminal too small for /copy · esc"


class CopyPickerScreen(ModalScreen[CopyTarget | None]):
    """Pick what to copy; dismisses with the chosen target, or ``None``.

    The tree is built once, by the caller, and never rebuilt while the screen
    is open. That is deliberate: new answers insert at the TOP of a
    most-recent-first list, so a live rebuild would shift every row below the
    insertion point — including the one the user is currently aiming at. A
    message that settles while the picker is open is simply not listed, and
    reopening picks it up.

    Esc dismisses. **`ctrl+c` is deliberately NOT bound**, though the reference
    binds it alongside Esc: in this app `ctrl+c` is the global interrupt
    (``app.py``), and a modal claiming it would change what stopping a turn
    means depending on whether an overlay happened to be open. Both existing
    modals (`SessionPickerScreen`, `AnalyticsScreen`) dismiss on Esc alone, so
    Esc alone is also the consistent answer.

    ``shift+↑↓`` scrolls the PREVIEW. It is the one shape that gives the
    second pane a keyboard without giving this screen a focus model: `left`,
    `right` and `tab` are all free, and all three plausible uses of them
    ("move focus to the preview") would create a mode in which `up`/`down`
    mean two different things depending on state nothing in the frame shows —
    the user presses `down` and sees nothing move. `shift+↑↓` reads as "the
    other pane's version of this key" and needs no mode. `j/k`, `space` and
    `/` are deliberately not bound: this app is not modal-vim anywhere, and
    there is no filter here to open.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "choose", "Copy", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        # Keyboard parity for the preview wheel. Every mouse capability needs
        # a key or the mouse becomes the only way to do something, which on a
        # terminal UI is a regression rather than a feature.
        Binding("shift+up", "scroll_preview(-1)", "Preview up", show=False),
        Binding("shift+down", "scroll_preview(1)", "Preview down", show=False),
        Binding("shift+pageup", "page_preview(-1)", "Preview page up", show=False),
        Binding("shift+pagedown", "page_preview(1)", "Preview page down", show=False),
    ]

    def __init__(self, targets: list[CopyTarget]) -> None:
        super().__init__()
        # Flattened once: the row list cannot change while the screen is open,
        # so re-walking the tree per repaint would be work with no input.
        self._flat: list[FlatNode] = flatten_targets(targets)
        self._selected = 0
        # The capped tree height from the last paint, reused as the page step
        # so pageDown moves by exactly what the user can see. Seeded with the
        # floor so a page key pressed before the first paint still moves.
        self._tree_rows = MIN_TREE_ROWS
        # Rows the preview drew last paint, reused as its page step for the
        # same reason `_tree_rows` is.
        self._preview_rows = MIN_PREVIEW_ROWS
        # First SOURCE line of the preview on screen. Source lines, not
        # wrapped rows: everything the user is shown about the preview (the
        # header total, the `… N more lines` remainder) is counted in source
        # lines, and an offset in the other unit would make the remainder
        # disagree with the header at every width — the exact defect
        # `_preview_lines` documents.
        self._preview_offset = 0
        # Row index under the pointer, in `_flat` coordinates. Distinct from
        # `_selected`: hover says "clickable", the cursor says "Enter takes
        # this", and painting them identically would show two selected rows.
        self._hovered: int | None = None
        # Wrapped preview rows, keyed `(target.id, width)`. See
        # `_wrap_preview` for why that key is total and why the cache exists.
        self._wrap_cache: dict[tuple[str, int, int], tuple[list[Text], list[int]]] = {}
        self._body: Static
        self._too_small: Static

    # -- state ---------------------------------------------------------------
    # `visible_rows`/`_card_text`, not `visible`/`_render`: both short names
    # are Textual's own (`Widget.visible`, `Widget._render`) and shadowing them
    # breaks focus and paint from inside the screen.
    @property
    def visible_rows(self) -> list[FlatNode]:
        """Every row in the tree, in draw order."""
        return self._flat

    def selected_target(self) -> CopyTarget | None:
        """The highlighted node, or ``None`` for an empty tree."""
        if not self._flat:
            return None
        return self._flat[self._selected].target

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        """The box the card's percentage sizes actually resolve in.

        ``self.size`` (this Screen's CONTENT box), not ``self.app.size``: the
        stylesheet's ``Screen { padding: 1 }`` insets the content box, and
        percentage heights resolve against the content box, so measuring the
        terminal over-counts the room and Textual clips the difference
        silently — off the bottom, taking the footer with it. Both
        `SessionPickerScreen` and `UsagePanel` measure the screen for this.
        """
        try:
            size = self.size
            if not size.width or not size.height:  # not laid out yet
                size = self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        return size.width, size.height

    def _card_width(self) -> int:
        width, _ = self._screen_size()
        return max(20, min(100, width - 4) - CARD_PADDING_CELLS)

    def _card_rows(self) -> int:
        """Rows the card's box actually gives its content.

        ``CARD_HEIGHT_FRACTION`` mirrors the stylesheet's ``height: 90%``; the
        padding comes back off because Textual sizes border-box.
        """
        _, height = self._screen_size()
        return int(height * CARD_HEIGHT_FRACTION) - CARD_PADDING_ROWS

    @property
    def is_drawable(self) -> bool:
        """Whether the box can hold the card's irreducible chrome.

        False on a terminal too short or too NARROW to paint the whole card,
        where it draws nothing rather than laying out rows the screen will
        clip. Both axes are one failure: ``overflow: hidden`` clips from the
        bottom, so a card that asks for one row more than it has loses the
        footer — the only statement of how to leave — and says nothing.

        The height limb was measured at 8-11 rows, where the fixed chrome
        (:data:`MIN_CARD_INNER_ROWS`) exceeds the box however the row budget is
        divided; the tree and preview already floor at one row each, so the
        split cannot recover it.

        The width limb is the same clip reached differently: a hint is never
        truncated (it carries the counts and the ``truncated`` marker), so on a
        narrow pane a row whose hint alone overruns the width WRAPS in the real
        paint, and each wrapped row pushes the footer further off the bottom. A
        row is only laid out flat when the cursor, the deepest gutter, one cell
        of label and the widest hint fit together, so that is what is measured
        — the composed-string helpers cannot see this, because wrap happens in
        the compositor and not in the ``Text`` they build.
        """
        if self._card_rows() < MIN_CARD_INNER_ROWS:
            return False
        return self._card_width() >= self._min_flat_width()

    def _footer_text(self, width: int) -> str:
        """The footer, shed to ``width`` — never dropped entirely.

        Hints come off the front (movement first, then Enter) because the last
        one standing has to be ``esc quit``: a card that cannot say how to
        leave is the defect the whole drawability check exists to prevent.

        The preview hint is GATED on the preview actually overflowing, the same
        rule the `↓ N more` cue follows. Advertising a scroll on a preview that
        is entirely on screen sends the user to press a key that does nothing
        visible, which reads as the key being broken rather than as the
        document being short — and it spends cells a narrow card needs for the
        hints that do apply.
        """
        hints = [
            hint
            for hint in FOOTER_HINTS
            if hint != PREVIEW_HINT or self._preview_source_lines() > self._preview_rows
        ]
        while len(hints) > 1 and cell_len(" · ".join(hints)) > width:
            hints.pop(0)
        return " · ".join(hints)

    def _min_flat_width(self) -> int:
        """Cells the widest row needs to paint on ONE line.

        The footer contributes only its IRREDUCIBLE form, because it sheds
        (:meth:`_footer_text`) where a tree row cannot: a hint is never
        truncated, so a row whose hint overruns the pane wraps in the real
        paint and pushes the footer off the bottom.
        """
        widest = 0
        for node in self._flat:
            hint = node.target.hint
            # One cell of label, plus the two-cell gap a hint always keeps.
            need = CURSOR_CELLS + GUTTER_CELLS * node.depth + 1
            if hint:
                need += cell_len(hint) + 2
            widest = max(widest, need)
        return max(widest, cell_len(FOOTER_ESSENTIAL))

    def _row_budget(self) -> int:
        """Rows the tree and the preview divide between them.

        ``MIN_TREE_ROWS + 1`` floors THIS, not the tree's share — that is the
        reference's shape, and applying the constant to the tree instead gives
        a visibly different widget on every short tree.

        The floor is then capped by the room that actually exists, because the
        reference's unconditional version overflowed the card by one row at a
        14-row terminal and Textual clipped the footer off the bottom
        SILENTLY. Below :data:`MIN_CARD_INNER_ROWS` no division of the budget
        helps and :attr:`is_drawable` hides the card instead.
        """
        room = self._card_rows() - HEADER_ROWS - CHROME_ROWS_BELOW_TOP
        # `max(1, room)`, written plainly: the floor above is what the
        # reference applies unconditionally, and capping it to the room that
        # exists removes it entirely on a short card. An earlier revision kept
        # a `min(MIN_TREE_ROWS + 1, room)` conditional here that computed the
        # identical value for every input while reading as though the floor
        # survived.
        return max(1, room)

    def _split_rows(self) -> tuple[int, int]:
        """``(tree_rows, preview_rows)`` for the current size.

        ``tree_rows`` is capped at the number of rows that EXIST, so a two-node
        tree takes two rows and donates the remainder to the preview rather
        than sitting in a half-height pane padded with blanks.

        The other cap is :data:`MIN_PREVIEW_ROWS`, and it replaced an
        unconditional ``available // 2``. The half share only ever LOST rows to
        the preview — the ``min(len)`` cap donates downward and nothing donated
        back — so a tree longer than half the budget was starved while the
        preview sat padded with blanks: measured at 100x30 mixed, ten of
        nineteen tree rows hidden under SEVEN empty preview rows (design round
        1, D1). Reserving the preview its floor and giving the tree the rest
        recovers those rows in exactly the cases that were wasting them; five
        of the seven measured shapes (mixed@80x24, mixed@100x14, short, code,
        long) render byte-identically, because there the ``min(len)`` cap and
        the ``max(1, …)`` floor still do all the work.

        **The expression must not read the cursor or the selected target.**
        Only ``len(self._flat)`` and ``available`` appear here, and both are
        invariant while the screen is open. Sizing the preview from the
        selected target's own line count would look tempting — it would fill
        the pane exactly — and it would change the tree's height on every
        arrow press, shifting rows sideways under a cursor the user is aiming
        at. That is `session_picker.render_rows`' documented column defect
        moved onto the time axis, where it is worse: motion draws the eye.
        """
        available = self._row_budget()
        tree_rows = max(1, min(len(self._flat), available - MIN_PREVIEW_ROWS))
        preview_rows = max(1, available - tree_rows)
        return tree_rows, preview_rows

    def _page_rows(self) -> int:
        """The page step: the tree's CAPPED height, as last painted."""
        return max(1, self._tree_rows)

    # -- navigation ----------------------------------------------------------
    def _move_to(self, index: int) -> None:
        """Move the cursor, CLAMPED at both ends.

        Not wrapped, diverging from the reference. AGENTS.md's wrap rule is
        written for a short list overlaid on a screen the user is still
        reading; it carries a documented exception for a list that IS the whole
        page, whose stated rationale — the far end is a destination the user
        travels to deliberately — describes this surface exactly. The nearest
        precedent in this repo, `session_picker._move_to`, already clamps. Page
        keys clamp under either reading, so clamping the arrows too is also the
        only choice that leaves one uniform rule on the page. `home`/`end` are
        the better answer to "take me to the other end" anyway.
        """
        # The preview is a DIFFERENT DOCUMENT once the selection changes, so
        # the offset resets on every cursor move — including one that lands
        # where it already was, which costs nothing and removes a state where
        # the reset depends on whether the clamp happened to move anything.
        # Carrying an offset across targets would open the next preview
        # part-way down a document the user never scrolled.
        self._preview_offset = 0
        if not self._flat:
            self._selected = 0
            self._repaint()
            return
        self._selected = max(0, min(len(self._flat) - 1, index))
        self._repaint()

    def _preview_source_lines(self) -> int:
        """Source lines in the selected target's preview.

        The unit `_preview_offset` is measured in, and the unit the header
        total and the `… N more lines` remainder are both counted in.
        """
        target = self.selected_target()
        if target is None or not target.preview:
            return 0
        return len(target.preview.splitlines())

    def _scroll_preview_to(self, offset: int) -> None:
        """Move the preview, CLAMPED at both ends — never wrapped.

        Clamped for the same reason `_move_to` is, and the ceiling is the LAST
        SOURCE LINE rather than ``total - preview_rows``: the pane's row count
        is in wrapped rows and the offset is in source lines, so subtracting
        one from the other would over- or under-shoot by however much the
        document wraps. Stopping at the last line means a fully scrolled
        preview can show a single line above blank rows, which is the honest
        frame — the remainder marker has vanished by then, so the blank space
        reads as "that is the end" rather than as a rendering fault.
        """
        ceiling = max(0, self._preview_source_lines() - 1)
        clamped = max(0, min(ceiling, offset))
        if clamped == self._preview_offset:
            return
        self._preview_offset = clamped
        self._repaint()

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        self._move_to(self._selected + delta * self._page_rows())

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self._flat) - 1 if to_end else 0)

    def action_scroll_preview(self, delta: int) -> None:
        self._scroll_preview_to(self._preview_offset + delta)

    def action_page_preview(self, delta: int) -> None:
        # By the preview's own drawn height, so a page moves by what the user
        # can see — the same rule `action_page` follows for the tree.
        self._scroll_preview_to(self._preview_offset + delta * max(1, self._preview_rows))

    def _dismiss_result(self, result: CopyTarget | None) -> None:
        """Dismiss once, releasing a hovered row's pointer shape first.

        Two guards in three lines, both load-bearing.

        ``is_active`` is the dismiss-race guard. Two dismissals with no event
        loop turn between them — a fast double-click on a busy machine, or a
        user hammering Enter — reach ``dismiss`` after the screen has already
        popped, and Textual raises ``ScreenStackError: Can't pop screen``.
        Reproduced on both paths before this existed: two queued ``Click``
        messages crash, and so do two bare ``action_choose()`` calls and a
        choose-then-cancel, which means the KEYBOARD path could crash the app
        without a mouse anywhere near it. Guarding only the mouse handler
        would have left that in place, so the guard lives here, where both
        paths pass through. (``pilot.click(times=2)`` does NOT reproduce it —
        it pauses between clicks — which is why the regression test posts the
        messages itself.)

        The pointer reset is the other half: the modal leaves without another
        mouse move, so the inline-rule assignment has to restore OSC 22 while
        this screen still owns the pointer. Without it a user who clicks to
        copy is left with a hand cursor over their transcript.
        """
        if not self.is_active:
            return
        self.styles.pointer = "default"
        self.dismiss(result)

    def action_cancel(self) -> None:
        self._dismiss_result(None)

    def action_choose(self) -> None:
        """Copy the highlighted node.

        A node with nothing to copy refuses rather than dismissing with a
        payload the caller cannot write. FALSY, not just ``None``: an empty
        fence (```` ```py ```` closed on the next line) builds a child whose
        content is ``""``, and dismissing on it wrote nothing to the clipboard
        with no toast and no notice — silence is right for a zero-width drag
        and wrong for a command the user typed deliberately, which is the
        position ``_cmd_copy``'s docstring already argues.

        Refusing here rather than dropping the empty child from the tree: the
        block IS in the message, and a `Block 2` that vanishes from the list
        makes the remaining numbers disagree with what the user is reading.
        """
        target = self.selected_target()
        if target is None or not target.content:
            return
        self._dismiss_result(target)

    # -- rendering -----------------------------------------------------------
    # -- mouse ---------------------------------------------------------------
    # Handler names are PUBLIC (`on_click`, `on_mouse_move`, `on_leave`,
    # `on_mouse_scroll_*`), per `command_picker`'s note: Textual's own `Widget`
    # hover bookkeeping runs off these names, and a private one would take the
    # events without letting it keep its state.
    #
    # Every handler stops its event. The wheel is measured NOT to bubble to the
    # transcript today — the modal already absorbs it — so this fixes nothing
    # observable; it is what keeps a future change to the transcript's own
    # scrolling from re-opening the one-gesture-two-surfaces defect, which is
    # why both sibling pickers do it too.

    def _wheel_step(self) -> int:
        """Rows per wheel notch, read LIVE from the app.

        Not a hardcoded 1, though both sibling pickers step by one: it is a
        per-instance attribute set in `App.__init__` (measured at 2.0 here), so
        a constant silently desynchronises the moment anything changes it and
        one gesture then travels at two speeds depending on where the pointer
        sat. `settings_view` records the same reasoning and the measurement
        behind it.
        """
        try:
            return max(1, int(self.app.scroll_sensitivity_y))
        except Exception:
            return 1

    def _wheel(self, event, direction: int) -> None:  # type: ignore[no-untyped-def]
        """Route a notch by WHERE THE POINTER IS, not by which widget saw it.

        The card is one `Static`, so the event's widget is identical over both
        panes and cannot discriminate them — the pointer's row against the
        card's own layout is the only thing that can. Over the tree the notch
        moves the cursor through `_move_to`, the single movement path every
        other gesture and key already uses; over the preview it moves the
        preview and leaves the cursor alone, because a wheel over text you are
        reading must not change what is selected under you.

        The row comes off the EVENT rather than off the last remembered
        pointer position: a wheel can arrive without a preceding mouse move
        (the pointer is already resting where the user wants to scroll), and
        routing on stale state would send the first notch to the wrong pane.
        """
        row = self._pointer_row_of(event)
        if row is not None and row >= self._preview_top_row:
            self._scroll_preview_to(self._preview_offset + direction * self._wheel_step())
            return
        # Anywhere else on the card — including the chrome and the backdrop —
        # moves the cursor. That is the generous reading and it is safe:
        # movement is clamped, reversible and visible, unlike the click, which
        # is why only the click carries `_index_at`'s three guards.
        self.action_move(direction * self._wheel_step())

    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._wheel(event, 1)

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._wheel(event, -1)

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Click selects and previews; DOUBLE-click copies and dismisses.

        **A deliberate divergence from `session_picker`, where a single click
        acts immediately.** That shape is right there because its list has no
        preview — the row text is all the information that exists, so a click
        cannot rob the user of anything. Here the preview pane is the picker's
        entire reason to exist: it is how you confirm you have the right block
        before overwriting your clipboard. A click that copied would remove
        the confirmation step FROM THE MOUSE PATH ONLY, handing the mouse user
        a less careful version of the feature than the keyboard user gets —
        backwards, since the mouse was invited in to make this easier.

        The two mistakes also differ in how they fail. Resuming the wrong
        session is loud and recoverable: you see the wrong conversation.
        Copying the wrong thing is SILENT — the clipboard is overwritten, the
        modal is gone, and the user finds out at the paste site with the prior
        contents already destroyed.

        "Click the already-selected row copies" was considered and rejected:
        it makes one physical gesture mean two different things depending on
        invisible prior state.

        Button 1 only, and the button is tested BEFORE any state changes: a
        right-click asking for a context menu is measured to arrive here, and
        it must not move the cursor on its way to being ignored.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        if index != self._selected:
            self._move_to(index)
        # `Click.chain` is Textual's own double-click count (0.5 s threshold),
        # so the contract needs no timing invented here.
        if getattr(event, "chain", 1) >= 2:
            self.action_choose()

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            # Tree rows only: `_tree_lines` is 0.07 ms against the preview's
            # 31 ms, and a hover changes nothing the preview draws. Repainting
            # the whole card per row the pointer crosses is what made a mouse
            # sweep cost 365 ms of loop CPU.
            self._repaint()
        # A hand over a tree row, the default shape everywhere else INCLUDING
        # the preview: the preview is scrollable, not clickable, and a hand
        # there would promise a click it does not keep. The inline-rule
        # assignment drives `Screen.update_pointer_shape()` through the
        # property's own observer and no-ops when the shape did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Tree row under a mouse event, or ``None`` anywhere else.

        Measured against the BODY's region rather than the event's own widget:
        the card is one ``Static``, so an event anywhere in it reports a ``y``
        relative to the whole block — title, rules, preview and footer
        included.

        Three guards, all load-bearing, and all three are `session_picker`'s.
        Its docstring records that its first cut without them resolved a click
        on the footer to session #12 and the dimmed backdrop to row 0, and
        BOTH of those coordinates were re-measured as still delivered here —
        so these are mandatory rather than defensive:

        - the point must be inside the body's region (the modal's backdrop
          covers the whole screen and bubbles events from well outside the
          card, including the columns to its left where ``y`` alone still
          looks valid);
        - the row must be inside the DRAWN page, not merely inside ``_flat`` —
          a short tree caps the pane at the rows that exist, and the blank
          remainder below them would otherwise resolve to real targets;
        - and the resulting index must still be a row that exists.

        Title, rules, preview, footer, card padding and backdrop are therefore
        all inert. The backdrop deliberately does NOT dismiss: the app's other
        two modals have no mouse handling at all, so backdrop-dismissal would
        exist on exactly one of three and a user who learned it here would get
        stuck on `/resume`. If the app wants it, it lands on all three.
        """
        row = self._pointer_row_of(event)
        if row is None:
            return None
        tree_rows, _ = self._split_rows()
        offset = row - HEADER_ROWS
        start = self._window_start(tree_rows)
        drawn = min(tree_rows, max(0, len(self._flat) - start))
        if not 0 <= offset < drawn:
            return None
        index = start + offset
        return index if 0 <= index < len(self._flat) else None

    def _pointer_row_of(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Row of the CARD under the event, or ``None`` outside the card.

        Kept separate from `_index_at` because the wheel and the click ask
        different questions of the same coordinate: the wheel needs to know
        which PANE the pointer is over (including its chrome), the click needs
        an exact tree row.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        return event.screen_y - region.y

    @property
    def _preview_top_row(self) -> int:
        """First card row belonging to the preview pane, its header included.

        Derived from the same split the paint uses, so the routing boundary
        cannot drift from the drawn one.
        """
        tree_rows, _ = self._split_rows()
        # title + rule + tree + rule
        return HEADER_ROWS + tree_rows + 1

    def compose(self) -> ComposeResult:
        with Container(classes="copy-picker"):
            self._body = Static(self._card_text(), id="copy-picker-body")
            yield self._body
        # A SIBLING of the card, not a row inside it, and that placement is
        # the whole trick. The card's contract is two-sided and pinned — either
        # its footer is on screen or it drew nothing — so the notice must not
        # be able to satisfy "the card drew something". Outside the card it
        # cannot: `render_lines_for_test` reads the card body and still
        # reports `[]`, `is_drawable` is untouched, and "Copy to clipboard"
        # stays out of a frame too small to hold the card.
        #
        # It exists because the hidden card is correct and unexplained. From
        # the user's chair, `/copy` on a small terminal is "I typed a command
        # and my screen went dim and blank": the dimmed backdrop is the only
        # sign anything happened, and nothing says why or what to do. One dim
        # line fits widths the card does not (8 cells of `esc` against a
        # 32-cell narrowest row), so the degraded frame becomes legible
        # without laying out a single row the box cannot hold.
        self._too_small = Static(TOO_SMALL_NOTICE, id="copy-picker-too-small")
        yield self._too_small

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: the row split and every column come from the screen.

        The wrap cache and the preview offset join that re-derivation because
        both are width-dependent. The cache is keyed on width so a stale entry
        cannot be *read*, but the old-width entries would sit there for the
        life of the screen; dropping them keeps the cache the size of one
        layout. The offset is clamped rather than kept, because a narrower
        card wraps the same source into more rows and a preview scrolled to a
        position the document no longer has would paint blank.
        """
        self._wrap_cache.clear()
        # Before `_move_to`, which resets the offset to 0 anyway — the clamp
        # is here so the invariant holds even if that reset is ever narrowed
        # to "only when the index actually changed".
        self._scroll_preview_to(self._preview_offset)
        self._move_to(self._selected)

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        drawable = self.is_drawable
        body.update(self._card_text() if drawable else Text())
        # Hidden, not merely emptied. A card with no drawable line still claims
        # its two padding rows, and at these sizes that pushed the screen's
        # virtual height past its own size — a scrollable screen, which
        # AGENTS.md calls always a bug here — around a card painting nothing.
        # `ask_picker._repaint` hides its card for the same reason. The next
        # resize brings it back; Esc works throughout, which is what keeps this
        # a degraded frame rather than a trap.
        card = body.parent
        if card is not None:
            card.display = drawable
        notice = getattr(self, "_too_small", None)
        if notice is not None and notice.is_mounted:
            notice.display = not drawable

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads.

        Empty when the card is not drawn. This method re-derives the text
        rather than reading back what was painted, so without these guards it
        reports rows that never reached the terminal — measured at 80x10,
        where it claimed the footer against a frame that had clipped it, and a
        test asserting on it therefore could not see the defect.

        Three guards, following `ask_picker.render_lines_for_test`: the body
        must be mounted, the card must not be HIDDEN (``display`` is the
        answer to "is this drawn", so this defers to it rather than keeping a
        second opinion), and the composed text must be non-empty, because
        ``Text`` splits an empty card into one empty line.

        It still cannot see WRAP — the compositor folds a too-narrow row, this
        builds strings — which is why the narrow case is gated by
        :attr:`is_drawable` above rather than detected here, and why a test
        about wrapping has to read the painted frame.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return []
        card = body.parent
        if card is not None and card.is_mounted and not card.display:
            return []
        if not self.is_drawable:
            return []
        text = self._card_text()
        if not text.plain:
            return []
        return [line.plain for line in text.split("\n")]

    def _window_start(self, tree_rows: int) -> int:
        """First visible row: the cursor centred, clamped to the ends."""
        return max(0, min(self._selected - tree_rows // 2, max(0, len(self._flat) - tree_rows)))

    def _card_text(self) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        label_style = Style(color=theme_mod.semantic_color("label"))
        width = self._card_width()
        tree_rows, preview_rows = self._split_rows()
        # Remembered for the page step, which must move by what the user can
        # actually see rather than by an uncapped half-height. The preview's
        # height is remembered for the same reason, by `shift+pageup/down`.
        self._tree_rows = tree_rows
        self._preview_rows = max(1, preview_rows - 1)

        rule = Style(color=theme_mod.semantic_color("edge"))
        start = self._window_start(tree_rows)
        above = start
        below = max(0, len(self._flat) - start - tree_rows)

        out = Text(no_wrap=True, overflow="ellipsis")
        out.append("Copy to clipboard", style=label_style)
        out.append("\n")
        # The two cues ride the rules that already exist, so a tree that
        # overflows costs no row to say so — which matters because the row
        # budget is exactly what the split above is fighting to recover, and
        # at 80x24 the tree is six rows, two of which would have been signage.
        self._append_rule(out, width, rule, f"↑ {above}" if above else "")
        out.append("\n")

        for line in self._tree_lines(width, tree_rows):
            out.append_text(line)
            out.append("\n")

        # A rule, not a blank row: without it the preview's header sat directly
        # under the last tree row and read as one more row of the tree.
        #
        # BOTH directions are cued, unlike `todo_panel`'s single remainder,
        # because `_window_start` centres the cursor: this list is very often
        # clipped at both ends at once, and a down-only cue would then be
        # actively misleading — it would say the list continues below while
        # silently hiding the rows above.
        self._append_rule(out, width, rule, f"↓ {below} more" if below else "")
        out.append("\n")

        target = self.selected_target()
        for line in self._preview_lines(width, target, preview_rows):
            out.append_text(line)
            out.append("\n")

        out.append("─" * width, style=rule)
        out.append("\n")
        out.append(self._footer_text(width), style=dim)
        return out

    def _append_rule(self, out: Text, width: int, rule: Style, cue: str) -> None:
        """A full-width rule, with ``cue`` right-aligned into its last cells.

        The cue's SLOT is sized on the widest value the cue can ever take, not
        on the value it has now. Sizing it on the current text makes the rule
        change length as the digit count does, so a user who is only scrolling
        watches the divider twitch under them for no reason they can see —
        `todo_panel`'s U1, worth not re-learning. Reserving the slot costs
        nothing: what is not cue is rule, and the rule is the same glyph
        either way.
        """
        if not cue:
            out.append("─" * width, style=rule)
            return
        # Widest rendering: every row hidden in that direction, in the longer
        # of the two cue shapes.
        slot = cell_len(f"↓ {max(1, len(self._flat))} more")
        keep = max(0, width - slot - 1)
        out.append("─" * keep, style=rule)
        pad = max(0, width - keep - cell_len(cue))
        out.append("─" * pad, style=rule)
        out.append(cue, style=Style(color=theme_mod.semantic_color("dim")))

    def _gutter_drawn(self, width: int, rows: int | None = None) -> bool:
        """Whether the scroll gutter is drawn at this width and tree height.

        Two gates, for two different reasons.

        It is drawn only when the tree ACTUALLY OVERFLOWS, the same rule the
        `↑ N`/`↓ N more` cues follow: a track with no thumb on a list that
        cannot scroll is a scrollbar for a document that fits, and it makes
        the cue's absence stop meaning "this is everything".

        And it is a SHED, not an optional decoration. The gutter costs
        :data:`GUTTER_THUMB_CELLS` off every tree row, which raises the
        narrowest row the card can lay out — so drawing it unconditionally
        would push :meth:`is_drawable`'s width limb up and newly HIDE the card
        on terminals that draw one today. Dropping the gutter instead is the
        same discipline :meth:`_footer_text` applies to hints: a card that
        merely needed narrowing must not be blanked.
        """
        if rows is None:
            rows = self._tree_rows
        if len(self._flat) <= rows:
            return False
        return width - self._min_flat_width() >= GUTTER_SHED_HEADROOM

    def _tree_lines(self, width: int, rows: int) -> list[Text]:
        """The windowed tree: cursor, ancestor gutter, connector, label, hint.

        Each row is painted on a GROUND spanning the full card width, in
        `session_picker.render_rows`' exact three-way shape — selected,
        hovered, or neither — rather than a fourth state invented here. A bare
        caret is adequate for a keyboard user and inadequate the moment the
        mouse arrives: hover has to say a row is live BEFORE the click, and a
        caret gives it nothing to lift. Hover and selection share
        ``tint-select`` and **the caret discriminates them**, which is this
        app's established convention: if hover painted like selection the user
        would see two selected rows and could not tell which one Enter takes.
        """
        accent = Style(color=theme_mod.semantic_color("accent"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        # The caret is MUTED, not accent: once the ground says "selected", the
        # mark only has to point. Two signals competing for the same fact is
        # what `command_picker`'s D17 and `session_picker`'s D5 both record.
        muted = Style(color=theme_mod.semantic_color("muted"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        start = self._window_start(rows)
        gutter = self._gutter_drawn(width, rows)
        row_width = width - GUTTER_THUMB_CELLS if gutter else width
        thumb = self._thumb_span(rows, start) if gutter else range(0)

        lines: list[Text] = []
        for offset in range(rows):
            index = start + offset
            line = Text(no_wrap=True, overflow="ellipsis")
            if index >= len(self._flat):
                lines.append(line)
                continue
            node = self._flat[index]
            selected = index == self._selected
            hovered = index == self._hovered
            if selected:
                ground = theme_mod.semantic_color("tint-select-hi" if hovered else "tint-select")
            elif hovered:
                ground = theme_mod.semantic_color("tint-select")
            else:
                ground = theme_mod.semantic_color("overlay")
            row_bg = Style(bgcolor=ground)

            prefix = ""
            for level in range(max(0, node.depth - 1)):
                # A vertical guide only while that ancestor still has a row
                # below it; three spaces once it does not, so the guides do not
                # run past the end of their own subtree.
                prefix += "│  " if node.ancestor_has_next[level] else " " * GUTTER_CELLS
            if node.depth > 0:
                prefix += "└─ " if node.is_last else "├─ "

            hint = node.target.hint
            # The hint is never truncated and the label is: the hint carries
            # the line/block counts and the `truncated` marker, which are what
            # a narrow terminal must not lose. Two cells of gap minimum.
            hint_cells = cell_len(hint) + 2 if hint else 0
            used = CURSOR_CELLS + cell_len(prefix)
            label = truncate_cells(node.target.label, max(1, row_width - used - hint_cells))

            line.append("❯ " if selected else "  ", style=row_bg + (muted if selected else fg))
            line.append(prefix, style=row_bg + dim)
            # An aggregate row is DERIVED — it copies a join of rows listed
            # above it rather than something that exists in the message — and
            # one ramp step down is the quietest way to say so without adding
            # a glyph (design round 1, D4).
            if selected:
                label_ink = accent + Style(bold=True)
            elif node.target.derived:
                label_ink = muted
            else:
                label_ink = fg
            line.append(label, style=row_bg + label_ink)
            if hint:
                gap = max(1, row_width - used - cell_len(label) - cell_len(hint))
                line.append(" " * gap, style=row_bg)
                self._append_hint(line, hint, row_bg, dim, warning)
            # The ground spans the FULL card width. A highlight that stops at
            # the text is ragged, and on a mouse surface it also lies about
            # where the row's click target ends.
            painted = cell_len(line.plain)
            if painted < row_width:
                line.append(" " * (row_width - painted), style=row_bg)
            if gutter:
                self._append_gutter(line, offset in thumb, row_bg)
            lines.append(line)
        return lines

    @staticmethod
    def _append_hint(line: Text, hint: str, row_bg: Style, dim: Style, warning: Style) -> None:
        """The hint, with `truncated` lifted out of the count ink.

        `truncated` is a caveat about the payload — the app raises a `warning`
        notice about it after the copy — and the whole hint was `dim`, which
        styled it as though it were one more statistic in a column full of
        statistics. The counts stay `dim`; only the caveat moves. `_message_hint`
        puts `truncated` first for its own reason, which is what makes it
        separable with a prefix test rather than a parse.
        """
        marker = "truncated"
        if hint.startswith(marker):
            line.append(marker, style=row_bg + warning)
            line.append(hint[len(marker) :], style=row_bg + dim)
            return
        line.append(hint, style=row_bg + dim)

    def _thumb_span(self, rows: int, start: int) -> range:
        """Rows of the drawn page the scroll thumb covers.

        Proportional: the thumb is as much of the track as the page is of the
        list, floored at one row so a very long tree still shows a mark. The
        thumb is the only CONTINUOUS signal on this surface — the `↓ N more`
        cue says how much is left, the thumb says WHERE you are and moves as
        you go — which is why both exist and neither replaces the other.
        """
        total = len(self._flat)
        if total <= rows:
            return range(0)
        span = max(1, round(rows * rows / total))
        # Anchored on the window rather than the cursor, since the window is
        # what the track represents. Clamped so a thumb at the end of the list
        # cannot be drawn past the bottom of the track.
        top = min(rows - span, round(start * rows / total))
        return range(max(0, top), max(0, top) + span)

    @staticmethod
    def _append_gutter(line: Text, on_thumb: bool, row_bg: Style) -> None:
        """One cell of gap, then the track cell for this row."""
        track_ink = theme_mod.semantic_color("dim" if on_thumb else "edge-hi")
        line.append(" ", style=row_bg)
        line.append("█" if on_thumb else "│", style=row_bg + Style(color=track_ink))

    def _preview_lines(self, width: int, target: CopyTarget | None, rows: int) -> list[Text]:
        """The highlighted node's text from ``_preview_offset``, wrapped.

        Never hard-truncated. What is BELOW the pane is reported as
        `… N more lines` on the last row, which COSTS a row: showing one more
        line and hiding the fact that more exist is the failure this marker
        prevents. What is ABOVE it is reported in the header, which already
        carries the total and so costs no row at all.

        Both cues are LIVE — they are recomputed from the offset every paint
        and they VANISH at their ends, so the absence of the marker means
        "this is the end of the document" and the absence of the header's
        position means "you are at the top". That is `todo_panel`'s rule, and
        it is what makes a remainder self-checking without ordinals: scroll,
        and the number goes down.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))

        header = Text(no_wrap=True, overflow="ellipsis")
        caption = f"Preview · {target.hint}" if target and target.hint else "Preview"
        header.append(caption, style=dim)
        lines = [header]

        content_rows = rows - 1
        if target is None or content_rows <= 0:
            while len(lines) < rows:
                lines.append(Text())
            return lines

        source_total = len(target.preview.split("\n"))
        # Clamped HERE as well as in `_scroll_preview_to` because the pane's
        # height is a paint-time fact: a resize that shrinks the document's
        # room reaches paint before any gesture can re-clamp.
        offset = max(0, min(self._preview_offset, max(0, source_total - 1)))
        window_start = self._wrap_window_start(offset)
        wrapped, rows_per_source = self._wrap_preview(target, max(1, width - 2), window_start)

        # Rows are dropped, not re-wrapped, to reach the offset: the wrap
        # window starts at a stride boundary, so the offset is usually some
        # lines into it. `rows_per_source` is the map from source lines to the
        # rows they produced, which is exactly what converts one unit to the
        # other.
        skipped_rows = sum(rows_per_source[: offset - window_start])
        visible = wrapped[skipped_rows:]
        # Source lines still wrapped below the window (the map covers at most
        # PREVIEW_WRAP_BUDGET of them), plus the ones the window never reached.
        mapped_from_offset = rows_per_source[offset - window_start :]

        below_window = source_total - (window_start + len(rows_per_source))
        has_more = len(visible) > content_rows or below_window > 0
        shown = content_rows - 1 if has_more else min(len(visible), content_rows)

        def _remaining_below(shown_rows: int) -> int:
            """Source lines below the pane when ``shown_rows`` rows are drawn.

            Counted in SOURCE lines, not wrapped rows. The header beside it
            reports the source line count, so a marker counting rows
            contradicted it in the same pane — at 100 cols a 79-line answer
            claimed "144 more lines", more than the document has. It also
            saturated: the wrap budget capped the rows, so a 600-line and a
            1000-line answer both reported the same 183. An OFFSET-AWARE
            marker can re-introduce that defect in one line, which is why the
            conversion runs through ``rows_per_source`` — the row-to-line map —
            rather than subtracting row counts from a row total.
            """
            consumed = 0
            source_shown = 0
            for count in mapped_from_offset:
                if consumed + count > shown_rows:
                    break
                consumed += count
                source_shown += 1
            return max(0, source_total - offset - source_shown)

        # Reserving the marker's row is only right if the marker has something
        # to say. A pane whose last drawn row completes the document leaves
        # `remaining` at 0, and spending a row on a marker that will not be
        # painted would hide a line for nothing — the inverse of the defect
        # the marker exists to prevent.
        if has_more and _remaining_below(shown) == 0:
            has_more = False
            shown = min(len(visible), content_rows)

        remaining = _remaining_below(shown)

        # The header says where the pane sits once it is not at the top. It is
        # the symmetric cue to the foot marker: without it a scrolled preview
        # looks like a document that simply starts at "line 40", and the user
        # has no way to tell there is anything above. Free — the header row
        # already exists and carries only the total.
        if offset > 0:
            header.append(f" · from line {offset + 1}", style=dim)

        for row in range(content_rows):
            if row < shown:
                line = visible[row]
                # Highlighted code keeps its own colours; prose is muted so the
                # tree above stays the brighter surface.
                lines.append(line if target.language else Text(line.plain, style=muted))
            elif row == shown and has_more and remaining > 0:
                plural = "line" if remaining == 1 else "lines"
                lines.append(Text(f"… {remaining} more {plural}", style=dim))
            else:
                lines.append(Text())
        return lines

    def _wrap_window_start(self, offset: int) -> int:
        """First source line of the wrap window holding ``offset``.

        Snapped to :data:`PREVIEW_WRAP_STRIDE` rather than taken as the offset
        itself, purely to bound the cache: keying on the raw offset would give
        a 20,000-line answer 20,000 entries, each holding up to
        :data:`PREVIEW_WRAP_BUDGET` rows, as the user scrolls it. Snapping caps
        it at ``ceil(lines / STRIDE)`` per target per width, and because the
        stride is half the budget, every offset in the window still has at
        least a stride's worth of lines wrapped BELOW it — far more than any
        terminal can draw at once, so the snapping is invisible.
        """
        return (max(0, offset) // PREVIEW_WRAP_STRIDE) * PREVIEW_WRAP_STRIDE

    def _wrap_preview(
        self, target: CopyTarget, width: int, window_start: int = 0
    ) -> tuple[list[Text], list[int]]:
        """``(wrapped rows, rows each source line produced)`` from ``window_start``.

        The second list is what lets the overflow marker be quoted in SOURCE
        lines: it maps a row cutoff back to the line the user would count. It
        is measured by rendering each source line separately, which produces
        exactly the same total as rendering the block whole — verified across
        widths 20-100 for prose, long words, unicode and highlighted code
        before this was relied on.

        :data:`PREVIEW_WRAP_BUDGET` source lines are wrapped, starting at
        ``window_start`` — a WINDOW, not a prefix; see the constant for why
        that distinction became user-visible the moment the preview could
        scroll. The caller compares the windowed map against the true source
        length, so budgeting changes how much is WRAPPED and never the number
        the user is shown.

        **Memoised on ``(target.id, width, window_start)``**, which supersedes
        this method's former "no per-target cache" note. That note's objection
        was staleness, and it does not survive the key: ``CopyTarget`` is
        frozen, the tree is a snapshot taken when the screen was pushed and
        cannot change while it is open, and width and window are the only
        other inputs — so a hit is byte-identical to a recomputation by
        construction. The cost of not caching is not theoretical either. Every
        repaint re-wrapped the whole preview through ``rich.Syntax``: measured
        at 31 ms against ``_tree_lines``' 0.07 ms, so one mouse sweep across
        the tree spent 365 ms of loop CPU with the loop frozen throughout, and
        the same sweep warm costs 2 ms. Hover made that trivial to hit without
        meaning to press anything, which is what turned a pre-existing
        inefficiency into a blocker for the mouse work. Cleared on resize,
        where width changes wholesale.
        """
        key = (target.id, width, window_start)
        cached = self._wrap_cache.get(key)
        if cached is not None:
            return cached

        source = target.preview.expandtabs(4)
        budgeted = source.split("\n")[window_start : window_start + PREVIEW_WRAP_BUDGET]
        text = "\n".join(budgeted)

        console = Console(width=width, no_color=False)
        if target.language:
            # The repo's one syntax theme. A second palette here would make a
            # code block read differently in the picker than in the transcript.
            renderable = Syntax(
                text,
                target.language,
                theme=IslandSyntaxTheme(),
                word_wrap=True,
                padding=0,
                background_color=theme_mod.semantic_color("bg"),
            )
        else:
            renderable = Text(text)

        options = console.options.update(width=width, height=None, highlight=False)
        lines: list[Text] = []
        for segments in console.render_lines(renderable, options, pad=False):
            line = Text()
            for segment in segments:
                line.append(segment.text, style=segment.style)
            lines.append(line)

        # Plain `Text` per source line even for code: only the ROW COUNT is
        # wanted here, and the highlighted rows above are what actually gets
        # painted. Measuring the same wrap twice with two renderables would be
        # two chances to disagree.
        rows_per_source = [
            len(console.render_lines(Text(entry), options, pad=False)) for entry in budgeted
        ]
        self._wrap_cache[key] = (lines, rows_per_source)
        return lines, rows_per_source
