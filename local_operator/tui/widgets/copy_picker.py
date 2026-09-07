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

from typing import NamedTuple

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
#: Shortest track that can carry a PROPORTION. On one or two rows the thumb
#: either fills the track — the visual language for "everything fits", beside
#: a `↓ N more` cue saying it does not — or occupies half of it regardless of
#: where the window sits. Three rows is the first height at which a thumb has
#: somewhere to be that is neither of those. See `_gutter_drawn`.
MIN_GUTTER_TRACK_ROWS = 3
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
#: The same notice for a card too narrow to hold the full one. `esc` is the
#: ACTIONABLE half and it used to shed FIRST, leaving `terminal too small for
#: /copy ·` — a dangling separator that reads as a rendering fault — and then
#: dropping the one word that tells the user how to leave (round 1, U8). The
#: footer already solves this by shedding hints and keeping `esc quit` last;
#: this is the same discipline in two forms rather than a truncation, because
#: there is only one thing here worth keeping. 14 cells against 34.
TOO_SMALL_NOTICE_SHORT = "too small · esc"


class _PointerAt(NamedTuple):
    """A bare screen coordinate with the two attributes the hit-tests read.

    :meth:`CopyPickerScreen._index_at` and :meth:`_pointer_row_of` take an
    "event" but use only ``screen_x``/``screen_y``. Re-resolving the hover
    after the window moved has a coordinate and no event — synthesising a real
    ``events.MouseMove`` for it would mean inventing a widget, a button state
    and deltas that no hit-test reads, and posting it would re-enter the
    handler. This is the coordinate, and nothing else.
    """

    screen_x: int
    screen_y: int


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
        # Last pointer position seen on this screen, in SCREEN coordinates.
        # Kept because `_hovered` is an index into `_flat` and the window
        # moves under a resting pointer: a real terminal sends no MouseMove
        # while only the wheel turns, so the highlight has to be recomputed
        # from the coordinate rather than carried (round 1, U4/Q3).
        self._pointer_at: tuple[int, int] | None = None
        # `(index, screen coordinate)` the current click chain is bound to.
        # A double-click copies THIS row rather than re-resolving the
        # coordinate, because the first click recentres the tree under the
        # pointer — see `on_click` (round 1, U1/Q1).
        self._click_anchor: tuple[int, tuple[int, int]] | None = None
        # Set when something moved the window while an anchor was live, which
        # Textual's chain cannot see: it breaks a chain on a changed screen
        # offset or the clock, and a wheel notch or an arrow key is neither
        # (round 2, U10). `on_click` reads it to refuse a copy for a chain
        # that is no longer one gesture, rather than copying a row the frame
        # stopped showing.
        self._anchor_disturbed = False
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

        The gate reads :meth:`_preview_overflows`, the SAME fact the pane's
        `… N more lines` marker is drawn from. It used to compare
        ``_preview_source_lines() > _preview_rows``, which is source lines
        against wrapped rows: on ordinary prose (one long source line per
        paragraph) the pane overflowed with FEWER source lines than rows, so
        the frame drew the marker while the footer — the only place
        `shift+↑↓` is ever named — declined to say the preview scrolled, in
        one screenshot (round 1, MAJOR-1). One overflow fact, two cues.
        """
        hints = [hint for hint in FOOTER_HINTS if hint != PREVIEW_HINT or self._preview_overflows()]
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
        if not self._flat:
            self._selected = 0
            self._preview_offset = 0
            self._repaint()
            return
        clamped = max(0, min(len(self._flat) - 1, index))
        # ANY movement DISTURBS a live double-click anchor. Textual breaks a
        # click chain on only two things — a changed screen offset and the
        # clock — and a wheel notch or an arrow key changes neither, so the
        # second click still arrived as `chain=2` and copied the PRE-WHEEL row
        # while the caret and the preview had moved on (round 2, U10: 5/5
        # frame/clipboard disagreements with a wheel between the clicks, 5/5
        # with a key, 0/5 plain). That is the same silent wrong-clipboard
        # failure U1 was filed for, through a narrower but entirely natural
        # gesture: click a row, realise it is not the one, nudge the wheel,
        # click again. This is the single path every gesture and key moves
        # through, so it is the one place that can see them all; `on_click`
        # re-arms after its own call, so a click cannot erase the anchor it is
        # establishing.
        if self._click_anchor is not None:
            self._anchor_disturbed = True
        self._click_anchor = None
        # The preview is a DIFFERENT DOCUMENT once the selection changes, so
        # the offset resets — but ONLY when the index actually changed.
        # Resetting unconditionally meant a movement that is a complete visual
        # no-op (`up`/`home`/`pageup` already at the top row, `down`/`end` at
        # the last, or a wheel notch at either end) still threw away the
        # user's reading position, with no history and no way back: 90
        # `shift+down` to recover a place lost to a key that moved nothing
        # (round 1, U2). It bites hardest on this surface's own best case, a
        # single long answer, where the tree is ONE row so every arrow press
        # is such a no-op. The old comment argued the unconditional form
        # "removes a state where the reset depends on whether the clamp
        # happened to move anything" — but that state is exactly what a user
        # expects: nothing moved, so nothing should have changed.
        if clamped != self._selected:
            self._preview_offset = 0
        self._selected = clamped
        # The cursor move may have scrolled the window under a resting
        # pointer, which is what left the highlight on a row the pointer was
        # not over. Resolved before the repaint so one paint carries both.
        self._refresh_hover()
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

    def _preview_tail_offset(self) -> int:
        """The LAST useful offset: the one whose pane ends on the last line.

        This is the single definition of "the preview overflows" that the
        whole surface derives from — the footer's `shift+↑↓ preview` hint, the
        `… N more lines` marker's existence, and the scroll clamp all read it,
        so the frame cannot contradict itself in the way review round 1 found
        it doing (MAJOR-1: the pane drew `… 4 more lines` while the footer
        declined to say the preview scrolled, because the two were computed in
        DIFFERENT UNITS — source lines against wrapped rows — on ordinary
        prose that wraps, which is most assistant answers).

        Returning **0** when the document already fits is the other half of
        that fix (MAJOR-2/U5). The previous ceiling was ``source_lines - 1``
        unconditionally, so a three-line preview in a sixteen-row pane still
        scrolled, pushing content it had room for off the top of the frame and
        leaving blank rows behind — the `less` contract, and the tree's own
        `down` contract, are both that scrolling stops with the last line at
        the bottom.

        The count walks ``rows_per_source`` BACKWARDS from the end, because
        the two units only convert through that map: filling a pane of
        ``content_rows`` wrapped rows consumes however many source lines those
        rows came from, which varies per line. Walking from the end asks the
        question the clamp actually has — "what is the first source line whose
        remainder still fills the pane?" — and answers it exactly, at any
        width, without ever subtracting a row count from a line count. That
        subtraction is the defect `_preview_lines` documents at length and the
        one this method exists not to re-introduce.

        The walk crosses WRAP WINDOWS rather than stopping at the last one,
        and that is what makes the ceiling hold on long documents.
        `_wrap_window_start` snaps to :data:`PREVIEW_WRAP_STRIDE`, so a
        document of ``100k + 1 … 100k + content_rows`` lines has a final
        window holding fewer lines than the pane draws: walking only that
        window ran out of MAP before it ran out of PANE and returned
        ``window_start``, which for that band is exactly the
        ``source_lines - 1`` ceiling MAJOR-2 removed — a fully scrolled
        205-line answer ended on five lines above eleven blank rows at 100x30
        (round 2, BLOCKER-1/U11). The band is ``content_rows`` wide, so it
        grew with the terminal: 17/60 sampled lengths at 80x24, 30/60 at
        100x30, 43/60 at 140x44. Stepping to the previous window and
        continuing costs one further wrap (cached, and only on documents past
        the stride) and makes the ceiling independent of where the document's
        length happens to fall against it.

        Reaching ``window_start == 0`` with the pane still unfilled is the
        only proof the DOCUMENT fits, and it returns **0** — the "does not
        overflow" answer the whole surface gates on.
        """
        target = self.selected_target()
        source_total = self._preview_source_lines()
        if target is None or source_total <= 0:
            return 0
        # `_preview_rows` is already the pane's CONTENT height — `_card_text`
        # stores it net of the header row — and the tail is the offset whose
        # remainder fills exactly that, with no marker row needed because
        # there is nothing left to announce.
        content_rows = max(0, self._preview_rows)
        if content_rows <= 0:
            return max(0, source_total - 1)

        # Wrapped from the LAST window, so a long document costs one wrap of
        # the tail rather than of the whole body: only the final rows can
        # decide where the tail begins. The cache makes a repeat free.
        width = max(1, self._card_width() - 2)
        line = max(0, source_total - 1)
        window_start = self._wrap_window_start(line)
        rows = 0
        while True:
            _, rows_per_source = self._wrap_preview(target, width, window_start)
            if not rows_per_source:
                return max(0, source_total - 1)
            # The map covers at most PREVIEW_WRAP_BUDGET lines from
            # `window_start`, and `line` is the highest one this pass may
            # consume, so a window the budget truncated is walked over the
            # part it does cover rather than trusted past its end.
            top = min(line, window_start + len(rows_per_source) - 1)
            for index in range(top, window_start - 1, -1):
                rows += rows_per_source[index - window_start]
                if rows >= content_rows:
                    # The lines from `index` to the last one fill the pane, so
                    # the tail starts at `index`; one line earlier overflows it.
                    return index
            if window_start <= 0:
                # Walked to the document's first line without filling the pane.
                return 0
            line = window_start - 1
            window_start = self._wrap_window_start(line)

    def _preview_overflows(self) -> bool:
        """Whether the preview has anything the pane is not already showing.

        The one fact both cues are gated on. See :meth:`_preview_tail_offset`.
        """
        return self._preview_tail_offset() > 0

    def _scroll_preview_to(self, offset: int) -> None:
        """Move the preview, CLAMPED at both ends — never wrapped.

        Clamped for the same reason `_move_to` is. The ceiling is
        :meth:`_preview_tail_offset` — the offset whose pane ends on the last
        source line — rather than ``source_lines - 1``, so scrolling stops
        with the document's end at the bottom of the pane instead of sliding
        it up into blank rows (round 1, MAJOR-2/U5). It is deliberately NOT
        ``total - preview_rows``: that subtracts wrapped rows from source
        lines, the unit confusion this file documents throughout.
        """
        clamped = max(0, min(self._preview_tail_offset(), offset))
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

        The refusal RINGS. Marking the row (`py · empty` in the hint column,
        and `Preview · py · empty` in the header) predicts the refusal, and
        that prediction does real work — but a user who did not read the hint
        column pressed Enter and got nothing at all: no notice, no bell, no
        frame change, which is an application that appears to have stopped
        responding to the key (round 1, U6). `App.bell` is the right register
        for it — a toast would demand a dismissal for a keypress the user can
        simply repeat elsewhere, and this screen deliberately raises no
        notices of its own. On a terminal with the bell disabled the marking
        is still there, which is why both halves exist.
        """
        target = self.selected_target()
        if target is None or not target.content:
            self.app.bell()
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
        # OFF THE CARD IS INERT, exactly as it is for the click. The notch
        # used to fall through to `action_move`, so a pointer ONE COLUMN
        # outside the card did something categorically different and
        # destructive: it moved the cursor and discarded the reading position
        # (round 1, U3/Q2). At 100x30 that zone is three cells left of the
        # preview text the user is reading, and trackpad drift of three cells
        # while scrolling is routine rather than adversarial; on a
        # single-node tree it was a PURE loss, since nothing moved and the
        # page was simply gone.
        #
        # The old justification — "movement is clamped, reversible and
        # visible" — is true of the cursor and false of the offset it silently
        # discarded. `_index_at` had already decided the backdrop is inert to
        # clicks, and one gesture map per surface is easier to learn than two.
        if row is None:
            return
        if row >= self._preview_top_row:
            self._scroll_preview_to(self._preview_offset + direction * self._wheel_step())
            return
        # The tree pane and its own chrome (title and rules) move the cursor:
        # inside the card the generous reading is right, and it is the pane
        # the pointer is visibly over.
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

        **The copy is bound to the row the FIRST click resolved, not to a
        fresh hit-test on the second.** `_window_start` centres the cursor, so
        click 1 recentres the tree UNDER A STATIONARY POINTER and the same
        screen coordinate then names a different row: measured at ~41% of the
        pane on any overflowing tree, and it copied `Answer 10` when the user
        aimed at `Quote 1` (round 1, U1/Q1 — the clipboard bytes were
        captured). Two clicks of one gesture must mean one row, so the chain
        remembers which. This does not re-open the arbitration that made a
        double-click the copy gesture; it is what makes that gesture honest.

        The anchor binds only while the chain is UNDISTURBED. A chained click
        is by definition the same physical spot, but a spot is not a row: the
        window can move under a still pointer, so anything that moves it
        clears the anchor in `_move_to` and the second click SELECTS WITHOUT
        COPYING (round 2, U10). What survives is the case the anchor was built for —
        the app moved the row under a hand that did not move — and what no
        longer survives is the case where the USER moved it, where the frame
        they are looking at is the honest answer. QA round 2 (Q6) noted this
        boundary while it still cut the other way and recommended documenting
        rather than changing it; the UX round measured the same mechanism
        destroying the clipboard silently, so it is a fix here and not a note.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            # A chain broken off the tree cannot be continued onto it: the
            # next click on a row must read as a fresh first click — which is
            # why the disturbance flag clears with it rather than surviving to
            # refuse that click.
            self._click_anchor = None
            self._anchor_disturbed = False
            return
        event.stop()
        # `Click.chain` is Textual's own double-click count, so the contract
        # needs no timing invented here. The window is `App.CLICK_CHAIN_TIME_
        # THRESHOLD`, which this app deliberately widens to **0.9 s** (see the
        # constant's own rationale in `app.py`); round 2 corrected a "0.5 s"
        # written here, which understated the exposure by 80%. Cited rather
        # than restated so a Textual upgrade cannot make this comment false.
        #
        # Textual increments the chain only while the pointer stays on ONE
        # screen offset, so a chained click is the same physical SPOT. That is
        # not the same as the same ROW: the window can move under a still
        # pointer, which is why `_move_to` clears the anchor and why the fall
        # back below re-hit-tests rather than trusting a stale one (U10).
        chain = getattr(event, "chain", 1)
        anchor = self._click_anchor
        coordinate = (event.screen_x, event.screen_y)
        chained = chain >= 2 and anchor is not None and anchor[1] == coordinate
        # A chain the window moved under is no longer ONE gesture, so it does
        # not copy — it selects and previews, exactly as a chain broken by the
        # clock already does (verified in round 2 as "the right refusal").
        # Re-hit-testing instead was considered and measured: because a wheel
        # over the tree moves the CURSOR and `_window_start` recentres it, the
        # row under the pointer equals the row the preview is showing in only
        # **1 of 9** pane positions, so that fallback copies a third row that
        # is neither aimed at nor displayed. Refusing is the conservative
        # direction for the reason this docstring already gives: a wrong copy
        # is silent and destroys the clipboard, a refusal is visible in the
        # frame and one more click away from the right answer.
        disturbed = chain >= 2 and not chained and self._anchor_disturbed
        target_index = anchor[0] if chained and anchor is not None else index
        if target_index != self._selected:
            self._move_to(target_index)
        if not chained:
            # Re-armed AFTER the move, which disturbs any anchor: this click is
            # the one binding the chain, so its anchor must outlive the
            # movement it causes.
            self._click_anchor = (index, coordinate)
            self._anchor_disturbed = False
        if chain >= 2 and not disturbed:
            self.action_choose()

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pointer_at = (event.screen_x, event.screen_y)
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            # Tree rows only: the tree is roughly two orders of magnitude
            # cheaper to draw than the preview (see `_wrap_preview`), and a
            # hover changes nothing the preview draws. Repainting the whole
            # card per row the pointer crosses is what made a mouse sweep
            # dominate the loop before the wrap was memoised.
            self._repaint()
        # A hand over a tree row, the default shape everywhere else INCLUDING
        # the preview: the preview is scrollable, not clickable, and a hand
        # there would promise a click it does not keep. The inline-rule
        # assignment drives `Screen.update_pointer_shape()` through the
        # property's own observer and no-ops when the shape did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def _refresh_hover(self) -> bool:
        """Re-resolve the highlight against the LAST KNOWN pointer position.

        `_hovered` is an index into `_flat`, but the window moves under a
        resting pointer — the wheel scrolls the tree and a click recentres it
        — and a real terminal sends NO `MouseMove` while only the wheel turns.
        So the highlight was left painted on a row the pointer was not over,
        and six notches scrolled it off the pane entirely while the hand
        cursor stayed (round 1, U4/Q3).

        That matters more here than it looks: the hover highlight is the
        ENTIRE affordance for the mouse on this screen, deliberately — the
        footer does not advertise the click — so a highlight that is not under
        the pointer undermines the one thing teaching the feature. It also hid
        U1 in the exact frame where it was still catchable, by confirming an
        aim the second click did not honour.

        Called from every path that can move the window without a pointer
        event. Cheap: it is one hit-test plus the repaint the caller was
        already doing, and it no-ops when the row did not change.

        `session_picker` has the identical staleness and is NOT fixed here —
        that is its own change, deliberately out of this PR's scope.

        Returns whether the highlight moved, so a caller that is about to
        repaint anyway does not pay for a second one.
        """
        if self._pointer_at is None:
            return False
        index = self._index_at(_PointerAt(*self._pointer_at))
        self.styles.pointer = "pointer" if index is not None else "default"
        if index == self._hovered:
            return False
        self._hovered = index
        return True

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pointer_at = None
        # A chain cannot survive the pointer leaving the card. Cleared as a
        # pair: the next click is a fresh first click, so it must not inherit
        # a disturbance recorded against the anchor being dropped here.
        self._click_anchor = None
        self._anchor_disturbed = False
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

        Three guards, all three `session_picker`'s. Its docstring records that
        its first cut without them resolved a click on the footer to session
        #12 and the dimmed backdrop to row 0, and BOTH of those coordinates
        were re-measured as still delivered here:

        - the point must be inside the body's region (the modal's backdrop
          covers the whole screen and bubbles events from well outside the
          card, including the columns to its left where ``y`` alone still
          looks valid). **Mandatory**: without it those measured coordinates
          resolve to real rows;
        - the row must be inside the DRAWN page, not merely inside ``_flat`` —
          a short tree caps the pane at the rows that exist, and the blank
          remainder below them would otherwise resolve to real targets.
          **Defensive on this screen, not mandatory.** Review round 1
          brute-forced the reachable state space (``_flat`` 0-39 x
          ``available`` 1-39 x every cursor position) and found NO state where
          it changes the outcome: ``_window_start``'s clamp to
          ``len(_flat) - tree_rows`` and ``_split_rows``' ``min(len(_flat), …)``
          cap already guarantee ``start + tree_rows <= len(_flat)``, so the
          third guard subsumes it. It is kept as defence against a future
          change to either invariant — both live in other methods — and this
          note replaces an earlier one calling all three "mandatory rather
          than defensive", which sent a reader hunting for a case that cannot
          occur (MINOR-2);
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
        # `_move_to` first, so the split and the window are re-derived at the
        # new size, and it no longer resets the offset for a resize (the index
        # does not change, U2). Then the clamp, which now OBSERVABLY does what
        # its docstring always claimed: the position is kept and clamped
        # rather than dropped, so widening the terminal to read a long block
        # more comfortably no longer throws the reader back to line 1
        # (round 1, U7).
        #
        # The order is DEFENSIVE, not load-bearing. The reasoning for it — the
        # clamp reads `_preview_rows`, which `_card_text` only refreshes
        # during that repaint — is sound, but review round 2 (MINOR-1) swapped
        # the two statements and found no frame where it shows: identical
        # rows, tail, offset and painted output at 140x44, 100x30, 80x24 and
        # 90x14 with the offset at the tail, and the whole file green. It is
        # kept in this order because deriving geometry before reading it is
        # the right dependency, and recorded as unobservable so the next
        # reader does not go looking for the frame that needs it.
        self._move_to(self._selected)
        self._scroll_preview_to(self._preview_offset)

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
            if not drawable:
                # The notice is pinned to ONE row in the stylesheet, so a
                # string wider than the box it lands in wraps and the pin
                # clips it — the notice that exists to explain a degraded
                # frame becoming degraded itself. Choosing the form that fits
                # keeps `esc` down to fourteen columns instead of shedding it
                # first (U8, NIT-1).
                #
                # Measured against the NOTICE'S OWN box, not the screen's.
                # `_screen_size` agrees with it once laid out, but its two
                # fallbacks do not: it answers `self.app.size` before layout
                # and a hardcoded (80, 24) on exception, both of which report
                # the terminal rather than the content box `Screen
                # { padding: 1 }` insets by two cells — so a 34- or 35-column
                # terminal could be told it had room for the 34-cell long form
                # and clip it back to `terminal too small for /copy ·`, the
                # exact dangling separator U8 removed. Design round 2 (D17)
                # kept a captured frame of that rendering but could not
                # reproduce it on demand across 22 repeats, so this closes the
                # question rather than chasing the race: take the NARROWEST
                # width any resolved source reports, and when nothing is
                # resolved prefer the short form — it fits everywhere the long
                # one does, so an uncertain measurement must not select the
                # one that can clip.
                candidates = [
                    size.width
                    for size in (getattr(notice, "size", None), getattr(self, "size", None))
                    if size is not None and size.width
                ]
                columns = min(candidates) if candidates else 0
                notice.update(
                    TOO_SMALL_NOTICE
                    if cell_len(TOO_SMALL_NOTICE) <= columns
                    else TOO_SMALL_NOTICE_SHORT
                )

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
        # `↑ N more`, not `↑ N`: a bare number reads as an ordinal ("row 6")
        # rather than as a remainder, which is the exact hazard that ruled out
        # an `N of M` position indicator on this card. `↓ N more` cannot be
        # misread that way, so the asymmetry was worth closing rather than
        # preserving (design round 1, D12). It is free — the slot below is
        # already sized on the wider `↓` form.
        self._append_rule(out, width, rule, self._cue("↑", above) if above else "")
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
        self._append_rule(out, width, rule, self._cue("↓", below) if below else "")
        out.append("\n")

        target = self.selected_target()
        for line in self._preview_lines(width, target, preview_rows):
            out.append_text(line)
            out.append("\n")

        out.append("─" * width, style=rule)
        out.append("\n")
        out.append(self._footer_text(width), style=dim)
        return out

    def _cue(self, arrow: str, count: int) -> str:
        """A remainder cue whose COLUMNS do not move as its digits change.

        The count is right-aligned in a field as wide as the largest it can
        ever reach (``len(_flat)``), so `↑  3 more` and `↑ 14 more` occupy the
        same cells and the digits line up under each other. Reserving the
        rule's slot alone was not enough: the cue is right-aligned into it, so
        a one-digit value started a column further right than a two-digit one
        and the text visibly stepped sideways as the user crossed 9→10 while
        merely scrolling (round 1, U9). That is precisely what item C1 asked
        to avoid, and `todo_panel`'s U1 is the same lesson.

        The field is sized from the row count rather than from the current
        value for the same reason the rule's slot is: a width that tracks the
        present number is a width that changes.
        """
        digits = len(str(max(1, len(self._flat))))
        return f"{arrow} {count:>{digits}} more"

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
        slot = cell_len(self._cue("↓", max(1, len(self._flat))))
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

        A track shorter than :data:`MIN_GUTTER_TRACK_ROWS` is dropped for the
        same reason, one step further on. At 100x14 the split gives the tree
        ONE row with eighteen hidden, and a one-row track can only paint a
        thumb that fills it — the visual language for "everything fits", while
        `↓ 18 more` on the rule says the opposite (design round 1, D14). The
        contradiction is forced by geometry rather than by the anchor, so
        fixing D11 does not fix it: below three rows no thumb can express a
        proportion at all. The card at that height is a degraded frame either
        way; dropping the gutter removes the one element in it that is
        actively wrong and leaves `↓ N more` to do the work alone.

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
        if rows < MIN_GUTTER_TRACK_ROWS:
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

        The top is anchored on SCROLL PROGRESS (``start / max_start``) across
        the track's free travel, not on the window's fraction of the list.
        Those are two different mappings, and the earlier
        ``min(rows - span, round(start * rows / total))`` let the clamp win one
        window position early: the thumb bottomed out at ``start=2`` of 3 while
        the rule still read `↓ 1 more`, so two distinct list positions painted
        an IDENTICAL gutter and the user's final arrow press moved the
        continuous signal not at all (design round 1, D11). This is the model
        `ask_picker._scrollbar_thumb` and `usage_panel._scrollbar_thumb`
        already use — span from the viewport fraction, top from the offset's
        fraction of its own range — so the app's three bars agree.

        The invariant is SYMMETRIC, and both halves are load-bearing:

            thumb flush with the bottom of the track ⟺ nothing below
            thumb flush with the top of the track    ⟺ nothing above

        Round 2 (reviewer MAJOR-1, design D16) found that a bare floored
        proportion holds the first and breaks the second. ``floor`` puts
        ``top == travel`` exactly at ``start == max_start`` — the bottom
        invariant, and why round 1 chose it over ``round`` — but it also
        yields ``top == 0`` for a RANGE of early windows, so the thumb sat
        pinned at the top of the track while `↑ N more` said rows were hidden
        above. That is verbatim D11's own defect, mirrored: at the 19-row
        shape D11 was filed against, ``start=0`` and ``start=1`` painted an
        identical gutter while the cue changed, and at 80x24 three
        consecutive positions did. Enumerated over ``total`` 2–399 ×
        ``rows`` 1–59, floor scored 0 bottom contradictions and 427,381 top
        ones.

        So the extremes of the track are RESERVED for the extremes of the
        list, and the interior is mapped over the interior cells only. Over
        the same space, restricted to shapes that actually draw a gutter
        (``rows >= MIN_GUTTER_TRACK_ROWS``): **0 contradictions at either end
        across 3,898,762 window positions**, bar the five noted below.

        The interior mapping spreads ``1 … max_start-1`` over
        ``1 … travel-1`` rather than re-flooring the full-range proportion.
        Both forms satisfy the invariant identically — the difference is only
        how evenly interior windows land on interior cells, where this one
        measures 1.17x the ideal crowding against 1.78x for a clamped
        full-range floor, i.e. the thumb tracks the list more faithfully in
        the middle where the user is actually reading.

        The five residuals are all ``travel == 1`` with ``max_start > 1``
        (only ``rows``/``total`` of 3/5, 3/6, 4/6, 5/7): a one-cell travel has
        two expressible positions for three or more states, so by pigeonhole
        SOME window must share a cell with an extreme. It is a geometric
        limit, not a formula defect, and it is resolved deliberately in favour
        of the BOTTOM — an interior window shares the top cell rather than the
        bottom one — because the bottom is where the list ends and where the
        user stops, and because `↑`/`↓` cues carry the fine signal at sizes
        that small (see ``MIN_GUTTER_TRACK_ROWS``, which sheds the track
        entirely once it cannot say anything true at all).
        """
        total = len(self._flat)
        if total <= rows:
            return range(0)
        span = max(1, round(rows * rows / total))
        travel = rows - span
        max_start = total - rows
        if travel <= 0 or max_start <= 0:
            top = 0
        elif start <= 0:
            top = 0
        elif start >= max_start:
            top = travel
        elif travel < 2:
            # One cell of travel cannot express an interior at all. Share the
            # TOP cell, so "flush with the bottom" keeps meaning "nothing
            # below" — see the docstring on which end this forfeits and why.
            top = 0
        else:
            top = 1 + ((start - 1) * (travel - 2)) // max(1, max_start - 2)
            top = max(1, min(travel - 1, top))
        return range(top, top + span)

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

        Two entries are live on the paint path of a long preview, not one:
        `_preview_lines` wraps the window holding the CURRENT offset while
        `_footer_text` → `_preview_overflows` → `_preview_tail_offset` walks
        the window(s) holding the document's END, and on a document scrolled
        to its middle those are different keys (review round 2, MINOR-2). So
        the first paint of a long preview pays two cold wraps rather than one,
        and the walk added for BLOCKER-1 can touch a third on the band of
        lengths that straddles a stride boundary. The memo absorbs the repeats
        — warm hits are ~0.00025 ms below — and the cold cost is bounded by
        the same per-window budget, so the ratio the table reports still holds
        per entry; the count of entries is what changed.

        **Memoised on ``(target.id, width, window_start)``**, which supersedes
        this method's former "no per-target cache" note. That note's objection
        was staleness, and it does not survive the key: ``CopyTarget`` is
        frozen, the tree is a snapshot taken when the screen was pushed and
        cannot change while it is open, and width and window are the only
        other inputs — so a hit is byte-identical to a recomputation by
        construction.

        The cost of not caching is not theoretical either. Every repaint
        re-wrapped the preview through ``rich.Syntax``, which is two orders of
        magnitude dearer than the tree beside it. Measured with
        ``time.thread_time`` (CPU, not wall — AGENTS.md), median of 9, on an
        M-series laptop at 100x30, so treat the RATIO as the durable fact and
        the absolute numbers as this machine's:

        ===============================  =========  =========  =========
        preview shape                    wrap cold  wrap warm  _tree_lines
        ===============================  =========  =========  =========
        plain, 121 lines                   3.40 ms   0.00025 ms   0.015 ms
        plain, 3002 lines                  3.58 ms   0.00025 ms   0.014 ms
        python block, 499 lines            4.19 ms   0.00025 ms   0.024 ms
        python block, 499 lines @200col    4.31 ms   0.00025 ms   0.023 ms
        ===============================  =========  =========  =========

        Note what the third column says: the wrap is ~200x the tree's cost,
        and a hover changes ONLY what the tree draws — which is why
        `on_mouse_move` repaints without touching the preview, and why the
        memo exists at all. `_card_text` as a whole goes from ~4.06 ms cold to
        ~0.10 ms warm on the 121-line answer, so a mouse sweep across the tree
        costs one wrap rather than one per row.

        The figures are also flat in the document's length, which is
        :data:`PREVIEW_WRAP_BUDGET` doing its job: a 3002-line answer wraps no
        more source than a 121-line one, so neither the cache entry nor the
        wrap grows with the message. An earlier revision of this comment
        quoted 31 ms / 0.07 ms / a 365 ms sweep; those do not reproduce at the
        shapes described here (QA round 1, Q4), and 31 ms was only reachable
        on a wide card before the budget bounded the work. An unqualified
        number that does not reproduce is worse than no number, so the shape,
        the clock and the machine are stated above.

        Cleared on resize, where width changes wholesale.
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
