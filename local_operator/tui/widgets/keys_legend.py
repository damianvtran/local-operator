"""The ``? Keys`` legend — a read-only overlay of the app's keyboard vocabulary.

Issue #1944 asks for a named affordance for "what can I press here": the
focused footer ladder and the ``/help`` key rows cover the FUNCTION of the
app's chords, but neither is the legend the issue names. This card is it, and
it exists to carry the rows the other surfaces structurally cannot:

* the sidebar-scoped chords — ``ctrl+a``, ``ctrl+o`` and ``ctrl+k`` fire ONLY
  while the sessions list owns the focus chain (F9 mode), so ``/help``
  deliberately does not list them ("a row for a chord that only fires in f9
  mode would be a lie", see the block above its ``f9`` row). A section HEADING
  that says which surface a row belongs to is the qualifier a bare key row
  cannot carry, which is why the list's chords live under "the sessions list
  (f9)" here.
* ``f10``/``ctrl+k`` pin — the footer ladder teaches pin only from 30 cells of
  list width up and drops out entirely below it, which is exactly the narrow
  terminal the issue calls out. A card sized against the SCREEN cannot drop a
  row for width reasons; it wraps or scrolls.

Reachability is deliberately narrow, and that is the characters-first rule
(issue #1357's contract, kept by this slice): ``?`` opens the card wherever a
typist cannot claim the character — read-only/full-page states and gated
modals — while every state where a typist CAN claim it (a focused composer,
the list's typing-home) keeps typing the character, unchanged. The typed
``/keys`` command is the char-safe route that covers the typing states, and
this card's own footer names both.

The card is a plain ``Static`` that composes and windows its own rows — the
same split ``UsagePanel`` makes, for the same reasons (see below): the widget
owns only the scroll offset, the section spec below is the single source the
rows are built from, and every geometry decision is pinned against the
measured screen rather than left to ``auto``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.transcript import wrap_cells

#: Horizontal padding cells, mirroring the ``padding: 1 2`` the stylesheet pins
#: on ``KeysLegend``. Textual sizes BORDER-BOX, so every width/height pinned in
#: ``_repaint`` has to add these back or the gutters eat the last rows.
PADDING_CELLS = 4

#: Vertical padding rows, same mirror (`padding: 1 2` is one row top and
#: bottom). Spent only when the ground can afford it — see ``KEYS_SQUEEZE_ROWS``.
PADDING_ROWS = 2

#: Below this many rows of ground the vertical gutter is spent rather than
#: kept: at these sizes the gutter is a fifth of the card, and the honest trade
#: (the one ``GoalPanel`` and ``UsagePanel`` already make at their own derived
#: thresholds) is to keep the CONTENT rows and drop the decoration.
KEYS_SQUEEZE_ROWS = 10

#: Cells between the key column and its description column.
KEY_GAP = 2

#: The one hint row every card in this family carries: the way OUT, with the
#: two routes back IN beside it. A dismissal-only footer would be a dead end
#: for the reader who closes the card and wants it again later — `/keys`
#: otherwise rides `/help` and the app binding alone, and this footer is where
#: the card can name its own reopen route (design note R1).
_HINT_SEGMENTS: tuple[str, ...] = ("↑↓ scroll", "esc close", "? or /keys")


@dataclass(frozen=True)
class LegendSection:
    """One titled block of ``(key, description)`` rows."""

    heading: str
    rows: tuple[tuple[str, str], ...]


#: THE single definition of what the card shows. Pure data, exported so tests
#: assert against the spec rather than a frame of glyphs.
#:
#: Order is load-bearing twice over. The sessions list comes FIRST because it
#: is the priority when space is tight: its ``ctrl+a``/``ctrl+o``/``ctrl+k``
#: rows are the ones no other surface teaches (see the module docstring), and
#: the card scrolls, so "first" means "visible without a scroll on the narrow
#: terminals the issue names". Within a section, rows run in the order a user
#: meets them: enter the list, move, act (pin, layer), leave.
#:
#: Every row is a LIVE binding on the branch that ships this file. ``f10`` is
#: kept beside ``ctrl+k`` rather than folded into it because f10 is the
#: compatibility route that still works and may still be the one a returning
#: user has in muscle memory; the description marks it as the alternate.
LEGEND_SECTIONS: tuple[LegendSection, ...] = (
    LegendSection(
        heading="the sessions list (f9)",
        rows=(
            ("f9", "enter the list; f9 again returns"),
            ("esc", "return to the composer"),
            ("enter", "open the row under the cursor"),
            ("up/down", "move the cursor; pageup/pagedown, home/end jump"),
            ("ctrl+a", "show or hide the ⌥ subagent runs"),
            ("ctrl+o", "jump to the first subagent run"),
            ("ctrl+k", "pin or unpin the row"),
            ("f10", "pin or unpin (compatibility)"),
        ),
    ),
    LegendSection(
        heading="the app",
        rows=(
            ("f8", "open an aside"),
            ("ctrl+b", "show or hide the sessions list"),
            ("ctrl+t", "expand or collapse the todo panel"),
            ("ctrl+g", "cycle the subagent panel"),
            ("ctrl+l", "clear the transcript (history stays)"),
            ("ctrl+home/end", "jump to the transcript top or end"),
            ("ctrl+n/ctrl+s", "start a new session / resume one"),
        ),
    ),
)


def key_column() -> int:
    """Width of the key column: the widest key in the spec, in cells."""
    return max(cell_len(key) for section in LEGEND_SECTIONS for key, _ in section.rows)


def natural_width() -> int:
    """The card width the content would take if the screen were unlimited.

    ``key column + gap + widest description + both gutters`` — the widest
    single line the card can paint, heads included. ``panel_width`` clamps it
    to the screen; nothing here assumes it was allowed.
    """
    widest_desc = max(
        cell_len(description) for section in LEGEND_SECTIONS for _, description in section.rows
    )
    return key_column() + KEY_GAP + widest_desc + PADDING_CELLS


def build_legend_rows(width: int) -> list[Text]:
    """The card's whole content as painted rows, composed to ``width`` cells.

    Pure function of ``(spec, width)`` — no app, no widget — so the wrap and
    the key column are testable without a running terminal, the same split
    ``usage_panel``'s builders keep. A description wraps to the column beside
    its key and every continuation line is indented to that same column, so a
    long row reads as one entry rather than as rows of a second list.
    """
    key_col = key_column()
    desc_width = max(1, width - key_col - KEY_GAP)
    muted = Style(color=theme_mod.semantic_color("muted"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    rows: list[Text] = []
    for index, section in enumerate(LEGEND_SECTIONS):
        if index:
            # One air row between sections: a heading flush against the
            # previous section's last row reads as one more entry of it.
            rows.append(Text(""))
        rows.append(Text(section.heading, style=muted))
        for key, description in section.rows:
            parts = wrap_cells(description, desc_width) or [""]
            first = Text()
            first.append(key + " " * (key_col - cell_len(key)), style=muted)
            first.append(" " * KEY_GAP)
            first.append(parts[0], style=dim)
            rows.append(first)
            for part in parts[1:]:
                line = Text(" " * (key_col + KEY_GAP))
                line.append(part, style=dim)
                rows.append(line)
    return rows


class KeysLegendDismissed(Message):
    """Esc or ``?`` in the card — the app restores the stashed focus."""


class KeysLegend(Static):
    """The overlay itself. State is only the scroll offset and visibility.

    It holds no vocabulary of its own: the rows come from the module-level
    spec, so a chord cannot exist in the card and not in the bindings it
    describes (or the reverse) without both moving in one commit.

    Scroll model mirrors ``UsagePanel``: the widget composes EVERY row and
    windows it against the measured ground, instead of asking Textual to
    scroll a content-sized child. Two reasons, both measured elsewhere in this
    family — ``auto`` heights measure against a guessed width and settle a row
    tall, and the pinned height is what keeps the card from walking up and
    down the screen as the reader scrolls it.
    """

    can_focus = True

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
        # `?` toggles: the same key that opens the card closes it, which is
        # the one gesture the narrow reachable set makes memorable. Bound on
        # the WIDGET so it fires while the card holds focus whichever surface
        # opened it; the app-level binding cannot double-fire because the
        # widget's own binding wins the chain while it is focused.
        Binding("question_mark,question", "dismiss", "Close", show=False),
        Binding("up", "scroll_rows(-1)", show=False),
        Binding("down", "scroll_rows(1)", show=False),
        Binding("pageup", "scroll_page(-1)", show=False),
        Binding("pagedown", "scroll_page(1)", show=False),
        Binding("home", "scroll_home", show=False),
        Binding("end", "scroll_end", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id="keys-legend")
        self._offset = 0
        # Screen size plus the live dock ceiling used for the last paint, the
        # same fingerprint `UsagePanel` keeps: the dock can grow without
        # resizing this overlay, so Textual emits no resize event for it.
        self._layout_shown: tuple[int, int, int] | None = None
        self.display = False

    # -- state ---------------------------------------------------------------
    def open(self) -> None:
        """Show the card from the top and repaint."""
        self._offset = 0
        self.display = True
        self._repaint()

    def close(self) -> None:
        """Hide the card. Purely a surface — nothing is written anywhere."""
        self.display = False

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    # -- geometry ------------------------------------------------------------
    def panel_width(self) -> int:
        """The card's width: the content's natural width, clamped to the screen.

        ``screen − 2`` is the hard bound from the design note (a card that
        touches the screen edges reads as a mode, not a card); below that the
        frame wins — a legend that cannot be read to its right edge teaches
        nothing, and the same rule `GoalPanel.panel_width` records for its own
        narrow case.
        """
        screen_width, _ = overlay.screen_size(self)
        return min(natural_width(), screen_width - 2)

    def _content_width(self) -> int:
        return max(1, self.panel_width() - PADDING_CELLS)

    def _fit(self) -> tuple[int, int, int]:
        """``(ground rows, gutter rows, budget)`` — one measurement.

        The budget is what is left of the ground ABOVE THE DOCK after the
        gutter. The ground, not the screen: every card on this layer centres
        in — and may only cover — the region above the docked composer
        (``widgets/overlay`` states the rule), and a legend that covered the
        composer would hide the very surface two of its routes are typed into.
        """
        ground = overlay.rows_above_dock(self)
        gutter = PADDING_ROWS if ground >= KEYS_SQUEEZE_ROWS else 0
        return ground, gutter, max(1, ground - gutter)

    # -- composition ---------------------------------------------------------
    def _hint_row(self, scrolled: bool) -> Text:
        """The footer facts that fit, least-load-bearing dropped first.

        ``esc close`` never yields — it is the way out of a card that holds
        the keyboard — and the reopen route yields only to it. The scroll cue
        is first to go: it matters only while scrolled, and the position it
        stands for is recoverable by pressing the key it names.
        """
        width = self._content_width()
        candidates = list(_HINT_SEGMENTS)
        if not scrolled:
            candidates = [segment for segment in candidates if segment != "↑↓ scroll"]

        def painted(items: list[str]) -> str:
            return " · ".join(items)

        kept = list(candidates)
        for disposable in ("↑↓ scroll", "? or /keys"):
            if cell_len(painted(kept)) <= width:
                break
            kept = [segment for segment in kept if segment != disposable]
        faint = Style(color=theme_mod.semantic_color("faint"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        row = Text()
        for index, segment in enumerate(kept):
            if index:
                row.append(" · ", style=faint)
            # The keys read in the same ink the rows above use for keys; the
            # words beside them are the card's own copy.
            row.append(segment, style=dim)
        # Belt-and-braces behind the drop loop: below the narrowest screen the
        # card supports (screen − 2, minus the gutters) even `esc close` could
        # overflow, and a hint cropped mid-word reads as a rendering fault.
        row.truncate(max(1, width), overflow="ellipsis")
        return row

    def _window(self) -> tuple[bool, int]:
        """``(show_hint, window rows)`` for the current ground — one source.

        The hint is pinned chrome (the window scrolls under it), and on a
        ground too short for both it is dropped rather than squeezing the
        content to nothing: two content rows with no named exit is a worse
        card than three content rows and no hint.
        """
        budget = self._fit()[2]
        show_hint = budget >= 2
        return show_hint, max(1, budget - (1 if show_hint else 0))

    def _compose_rows(self) -> list[Text]:
        """The painted rows: a window of the content plus the pinned hint."""
        content = build_legend_rows(self._content_width())
        show_hint, window_budget = self._window()
        max_offset = max(0, len(content) - window_budget)
        self._offset = max(0, min(self._offset, max_offset))
        scrolled = max_offset > 0
        rows = list(content[self._offset : self._offset + window_budget])
        if show_hint:
            rows.append(self._hint_row(scrolled))
        return rows

    def render_lines_for_test(self) -> list[str]:
        """The painted card as plain strings — what a user reads."""
        return [row.plain for row in self._compose_rows()]

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        rows = self._compose_rows()
        width = self.panel_width()
        self.styles.width = width
        _, gutter, _ = self._fit()
        self.set_class(gutter == 0, "-squeezed")
        # Pinned rather than `auto`, and the padding rows are added back:
        # Textual is border-box, so the pinned height is
        # `content rows + gutter` or the gutter consumes the hint row (the
        # measured failure `GoalPanel._repaint` documents at the same step).
        outer_height = len(rows) + gutter
        self.styles.height = outer_height
        overlay.recentre(self, width, outer_height)
        screen_width, screen_height = overlay.screen_size(self)
        self._layout_shown = (screen_width, screen_height, overlay.rows_above_dock(self))
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        self.update(out)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """A terminal resize changes both the card width and the row window."""
        self._repaint()

    def sync_layout(self, *, force: bool = False) -> None:
        """Repaint when the screen or live dock changed around the open card.

        Same signature and same guard as the usage card's, because the app
        drives both from the same places (dock changes emit no resize for an
        overlay that floats on a layer).
        """
        if not self.display or not self.is_mounted:
            return
        width, height = overlay.screen_size(self)
        fingerprint = (width, height, overlay.rows_above_dock(self))
        if force or fingerprint != self._layout_shown:
            self._repaint()

    # -- actions -------------------------------------------------------------
    def action_dismiss(self) -> None:
        self.close()
        self.post_message(KeysLegendDismissed())

    def action_scroll_rows(self, delta: int) -> None:
        self._scroll_by(delta)

    def action_scroll_page(self, delta: int) -> None:
        budget = self._fit()[2]
        self._scroll_by(delta * max(1, budget - 1))

    def action_scroll_home(self) -> None:
        self._jump_to(0)

    def action_scroll_end(self) -> None:
        self._jump_to(len(build_legend_rows(self._content_width())))

    def _jump_to(self, target: int) -> None:
        """Clamp ``target`` into range and repaint.

        Named ``_jump_to`` rather than ``_scroll_to`` on purpose: ``Widget``
        owns ``_scroll_to(x, y, animate=…)`` and an incompatible override is
        the first defect class the widget conventions name — pyright's
        ``reportIncompatibleMethodOverride`` caught this one, which is why the
        name is not the obvious one.
        """
        rows = len(build_legend_rows(self._content_width()))
        self._offset = max(0, min(self._max_offset(rows), target))
        self._repaint()

    def _scroll_by(self, delta: int) -> None:
        self._jump_to(self._offset + delta)

    def _max_offset(self, row_count: int) -> int:
        _, window = self._window()
        return max(0, row_count - window)

    # -- mouse ---------------------------------------------------------------
    # The card floats over the transcript, so a gesture left to bubble would
    # move both the legend and the conversation behind it — the reason
    # `UsagePanel` stops the same events at the same boundary.
    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(1)

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(-1)
