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
#: Cap on source lines wrapped for the preview. A 500-line answer would
#: otherwise fold thousands of rows to display about fifteen. The overflow
#: marker is computed from the SOURCE line count, never from the wrapped rows,
#: so budgeting the wrap cannot change the number the user is shown.
PREVIEW_WRAP_BUDGET = 200
#: Footer hints, widest first, paired with the order they are SHED in. The
#: footer is the only statement of how to leave, so a narrow card drops the
#: movement hints rather than the whole row — `esc quit` is eight cells and
#: fits terminals the full thirty-one-cell footer does not. This mirrors
#: `session_picker._shed_to_width`, which drops hints in a fixed order for the
#: same reason; hiding the card over a footer that merely needed shortening
#: would blank a modal the user is looking at.
FOOTER_HINTS = ("↑↓ move", "enter copy", "esc quit")
#: Never shed: without it the card cannot say how to leave.
FOOTER_ESSENTIAL = "esc quit"
#: Rows the card cannot do without: title, rule, one tree row, rule, the
#: preview's header, rule, footer. Below this the card draws NOTHING rather
#: than laying out rows the box cannot hold — `overflow: hidden` clips from
#: the bottom, so a card one row too tall loses the footer specifically.
MIN_CARD_INNER_ROWS = 7


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
        self._body: Static

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
        """
        hints = list(FOOTER_HINTS)
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
        """
        available = self._row_budget()
        tree_rows = max(1, min(len(self._flat), available // 2))
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
        if not self._flat:
            self._selected = 0
            self._repaint()
            return
        self._selected = max(0, min(len(self._flat) - 1, index))
        self._repaint()

    def action_move(self, delta: int) -> None:
        self._move_to(self._selected + delta)

    def action_page(self, delta: int) -> None:
        self._move_to(self._selected + delta * self._page_rows())

    def action_jump(self, to_end: int) -> None:
        self._move_to(len(self._flat) - 1 if to_end else 0)

    def action_cancel(self) -> None:
        self.dismiss(None)

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
        self.dismiss(target)

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="copy-picker"):
            self._body = Static(self._card_text(), id="copy-picker-body")
            yield self._body

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: the row split and every column come from the screen."""
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
        # actually see rather than by an uncapped half-height.
        self._tree_rows = tree_rows

        out = Text(no_wrap=True, overflow="ellipsis")
        out.append("Copy to clipboard", style=label_style)
        out.append("\n")
        out.append("─" * width, style=Style(color=theme_mod.semantic_color("edge")))
        out.append("\n")

        rule = Style(color=theme_mod.semantic_color("edge"))
        for line in self._tree_lines(width, tree_rows):
            out.append_text(line)
            out.append("\n")

        # A rule, not a blank row: without it the preview's header sat directly
        # under the last tree row and read as one more row of the tree.
        out.append("─" * width, style=rule)
        out.append("\n")

        target = self.selected_target()
        for line in self._preview_lines(width, target, preview_rows):
            out.append_text(line)
            out.append("\n")

        out.append("─" * width, style=rule)
        out.append("\n")
        out.append(self._footer_text(width), style=dim)
        return out

    def _tree_lines(self, width: int, rows: int) -> list[Text]:
        """The windowed tree: cursor, ancestor gutter, connector, label, hint."""
        accent = Style(color=theme_mod.semantic_color("accent"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        start = self._window_start(rows)

        lines: list[Text] = []
        for offset in range(rows):
            index = start + offset
            line = Text(no_wrap=True, overflow="ellipsis")
            if index >= len(self._flat):
                lines.append(line)
                continue
            node = self._flat[index]
            selected = index == self._selected

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
            label = truncate_cells(node.target.label, max(1, width - used - hint_cells))

            line.append("❯ " if selected else "  ", style=accent if selected else fg)
            line.append(prefix, style=dim)
            line.append(label, style=accent + Style(bold=True) if selected else fg)
            if hint:
                gap = max(1, width - used - cell_len(label) - cell_len(hint))
                line.append(" " * gap)
                line.append(hint, style=dim)
            lines.append(line)
        return lines

    def _preview_lines(self, width: int, target: CopyTarget | None, rows: int) -> list[Text]:
        """The highlighted node's text, wrapped — never hard-truncated.

        Overflow is reported as `… N more lines` on the last row, which COSTS
        a row: showing one more line and hiding the fact that more exist is
        the failure this marker prevents.
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

        wrapped, rows_per_source = self._wrap_preview(target, max(1, width - 2))
        source_total = len(target.preview.split("\n"))
        has_more = len(wrapped) > content_rows or len(rows_per_source) < source_total
        shown = content_rows - 1 if has_more else min(len(wrapped), content_rows)

        # Counted in SOURCE lines, not wrapped rows. The header beside it
        # reports the source line count, so a marker counting rows contradicted
        # it in the same pane — at 100 cols a 79-line answer claimed "144 more
        # lines", more than the document has. It also saturated: the wrap
        # budget capped the rows, so a 600-line and a 1000-line answer both
        # reported the same 183. Both numbers now measure the same thing, and
        # the budget stays an optimisation the user cannot observe.
        consumed = 0
        source_shown = 0
        for count in rows_per_source:
            if consumed + count > shown:
                break
            consumed += count
            source_shown += 1
        remaining = max(0, source_total - source_shown)

        for row in range(content_rows):
            if row < shown:
                line = wrapped[row]
                # Highlighted code keeps its own colours; prose is muted so the
                # tree above stays the brighter surface.
                lines.append(line if target.language else Text(line.plain, style=muted))
            elif row == shown and has_more:
                plural = "line" if remaining == 1 else "lines"
                lines.append(Text(f"… {remaining} more {plural}", style=dim))
            else:
                lines.append(Text())
        return lines

    def _wrap_preview(self, target: CopyTarget, width: int) -> tuple[list[Text], list[int]]:
        """``(wrapped rows, rows each source line produced)``.

        The second list is what lets the overflow marker be quoted in SOURCE
        lines: it maps a row cutoff back to the line the user would count. It
        is measured by rendering each source line separately, which produces
        exactly the same total as rendering the block whole — verified across
        widths 20-100 for prose, long words, unicode and highlighted code
        before this was relied on.

        Only the first :data:`PREVIEW_WRAP_BUDGET` source lines are wrapped: a
        long answer would otherwise fold thousands of rows to display about
        fifteen. The caller compares the budgeted map against the true source
        length, so budgeting changes how much is WRAPPED and never the number
        the user is shown.

        No per-target cache. The reference caches because it re-renders on
        every frame; Textual repaints on state change, so a cache here would
        buy nothing measurable and could serve a stale preview.
        """
        source = target.preview.expandtabs(4)
        budgeted = source.split("\n")[:PREVIEW_WRAP_BUDGET]
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
        return lines, rows_per_source
