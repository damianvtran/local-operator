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
this card's own footer names the routes back in — ``/keys`` unconditionally,
``?`` qualified to the states where nothing is typing.

The card is a plain ``Static`` that composes and windows its own rows — the
same split ``UsagePanel`` makes, for the same reasons (see below): the widget
owns only the scroll offset, the section spec below is the single source the
rows are built from, and every geometry decision is pinned against the
measured screen rather than left to ``auto``.
"""

from __future__ import annotations

from collections.abc import Mapping
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

#: The hint row's segments, in paint order. The reopen route leads with the
#: typed `/keys` — the unconditional one — and marks `?` as its conditional
#: sibling rather than naming it bare (UX round 1, U2: the card's own footer
#: advertised `?` flatly, and `?` types wherever a typist can claim it, so a
#: reader who follows the footer from a composer gets a stray character in
#: their draft instead of the card). `/keys` has no such state.
_HINT_SEGMENTS: tuple[str, ...] = (
    "↑↓ scroll",
    "esc close",
    "/keys reopens",
    "? when not typing",
)


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
            # The two rows a stuck user reaches for FIRST, ahead of the
            # navigation set. /help's "esc stop the agent" row is outside
            # this card's two scopes only because /help is a different
            # surface: a card answering "what can I press" that cannot stop a
            # runaway turn is missing its most load-bearing chord (design
            # round 1, D7 / UX round 1, U4). `ctrl+c`'s first press interrupts
            # the turn (and clears a draft); a second press within the window
            # leaves — `action_interrupt`, which is the truth the words
            # carry.
            ("esc", "stop the agent"),
            ("ctrl+c", "interrupt the turn; twice exits"),
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


#: Rows whose key is the DEFAULT of a remappable action, mapped to the keymap
#: ids that can change it. Every other row's key is an unconditional binding
#: that cannot drift; these two are the card's only rows that could become a
#: lie after `lop config edit keymap.<id>` or the settings page (design round
#: 1, D6), so they are resolved against the LIVE keymap at composition time.
#: `tests/unit/tui/test_keys_legend.py` asserts these ids exist in
#: `KEY_ACTIONS`, which is the drift guard for the string pair.
_REMAPPABLE_ROWS: dict[str, tuple[str, ...]] = {
    "ctrl+n/ctrl+s": ("keymap.new_session", "keymap.resume"),
}


def resolve_key(key: str, keymap: Mapping[str, str] | None = None) -> str:
    """The key a row should PAINT: the live remap for the remappable rows.

    Falls back to the spec literal whenever the row is static, the keymap is
    absent, or an id is missing from it — a stripped harness (no app, no
    keymap) paints exactly the defaults, which is what the pure-builder tests
    rely on.
    """
    ids = _REMAPPABLE_ROWS.get(key)
    if not ids or not keymap:
        return key
    resolved = [keymap.get(identifier) for identifier in ids]
    if any(part is None for part in resolved):
        return key
    return "/".join(part for part in resolved if part is not None)


def _longest_description_word() -> int:
    """The longest single word across every description, in cells.

    ``wrap_cells`` breaks a word that does not fit its column rather than
    overflowing it, so this number IS the narrowest description column the
    card can paint without splitting a word: below it the columnar layout
    force-breaks its own copy mid-token (design round 1, D1's measured
    ``compose`` / ``r`` rows), and the stacked layout takes over instead.
    """
    return max(
        cell_len(word)
        for section in LEGEND_SECTIONS
        for _, description in section.rows
        for word in description.split(" ")
    )


def key_column(keymap: Mapping[str, str] | None = None) -> int:
    """Width of the key column: the widest key in the spec, in cells."""
    return max(
        cell_len(resolve_key(key, keymap)) for section in LEGEND_SECTIONS for key, _ in section.rows
    )


def natural_width(keymap: Mapping[str, str] | None = None) -> int:
    """The card width the content would take if the screen were unlimited.

    ``key column + gap + widest description + both gutters`` — the widest
    single line the card can paint, heads included. ``panel_width`` clamps it
    to the screen; nothing here assumes it was allowed.
    """
    widest_desc = max(
        cell_len(description) for section in LEGEND_SECTIONS for _, description in section.rows
    )
    return key_column(keymap) + KEY_GAP + widest_desc + PADDING_CELLS


def build_legend_rows(width: int, keymap: Mapping[str, str] | None = None) -> list[Text]:
    """The card's whole content as painted rows, composed to ``width`` cells.

    Pure function of ``(spec, width, keymap)`` — no app, no widget — so the
    wrap, the key column and the narrow-width STACKING are testable without a
    running terminal, the same split ``usage_panel``'s builders keep.

    Two shapes, switched by the one measured threshold: while the description
    column can hold the longest word in the copy, rows paint ``key  description``
    with continuation lines indented to the same column; below that the row
    STACKS — the key on its own line, the description wrapped at the FULL
    content width beneath it — because a 13-cell key column inside a 22-cell
    box leaves a column narrower than the card's own words, and every
    description force-breaks mid-token (design round 1, D1, measured:
    ``compose`` / ``r`` at 30 columns).

    Ink tiers are the reading hierarchy, not decoration: keys in ``fg`` and
    descriptions in ``muted`` (the family's secondary-text rung; ``dim`` is
    the micro-label/separator rung — 3.43:1 on this card's ground — and the
    body copy was demoted to it, design round 1 D3). Headings take the
    family's title ink (``label``) plus bold (D4); the sibling cards also
    draw a rule under theirs, which this card skips knowingly: two rule rows
    are two content rows the 100×30 window's budget does not have.
    """
    key_col = key_column(keymap)
    bare_width = width - key_col - KEY_GAP
    stacked = bare_width < _longest_description_word()
    desc_width = width if stacked else max(1, bare_width)
    heading_ink = Style(color=theme_mod.semantic_color("label"), bold=True)
    key_ink = Style(color=theme_mod.semantic_color("fg"))
    desc_ink = Style(color=theme_mod.semantic_color("muted"))
    rows: list[Text] = []
    for index, section in enumerate(LEGEND_SECTIONS):
        if index:
            # One air row between sections: a heading flush against the
            # previous section's last row reads as one more entry of it.
            rows.append(Text(""))
        rows.append(Text(section.heading, style=heading_ink))
        for key, description in section.rows:
            painted_key = resolve_key(key, keymap)
            parts = wrap_cells(description, desc_width) or [""]
            if stacked:
                rows.append(Text(painted_key, style=key_ink))
                for part in parts:
                    rows.append(Text(part, style=desc_ink))
                continue
            first = Text()
            first.append(painted_key + " " * (key_col - cell_len(painted_key)), style=key_ink)
            first.append(" " * KEY_GAP)
            first.append(parts[0], style=desc_ink)
            rows.append(first)
            for part in parts[1:]:
                line = Text(" " * (key_col + KEY_GAP))
                line.append(part, style=desc_ink)
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
        return min(natural_width(self._keymap_snapshot()), screen_width - 2)

    def _content_width(self) -> int:
        return max(1, self.panel_width() - PADDING_CELLS)

    def _keymap_snapshot(self) -> dict[str, str]:
        """The live keymap for the rows that can be remapped (review D6).

        ``App.set_keymap`` stores the normalized ``id -> key`` mapping on the
        app's ``_keymap``, and the app applies it from config — the same
        source its own bindings are re-keyed from. Read per repaint rather
        than cached: it is a handful of strings, and a remap applied while
        the card is open should not need a reopen to be reflected.
        ``resolve_key`` falls back to the spec's literals, so a stripped
        harness with no keymap still paints the defaults.
        """
        keymap = getattr(self.app, "_keymap", None)
        return dict(keymap) if isinstance(keymap, dict) else {}

    def _compose_content(self) -> list[Text]:
        """The content rows with the live keymap — the one composition path."""
        return build_legend_rows(self._content_width(), self._keymap_snapshot())

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

        The scroll cue OUTLIVES the reopen route: an overflowing card must
        say it overflows — the cue is the only in-card sign there is more
        below, and it is exactly the narrow sizes where the overflow is worst
        that used to be left without it (design round 1, D2, measured: the
        cue painted only from content 34, i.e. never on the 30–41-col card).
        The reopen route yields first among the named routes — closing and
        wanting the card back is recoverable from `/help` and the binding —
        and ``esc close`` never yields: it is the way out of a card that
        holds the keyboard. The `?` qualifier yields before either, because
        it is the one segment that is only conditionally true.
        """
        width = self._content_width()
        candidates = list(_HINT_SEGMENTS)
        if not scrolled:
            candidates = [segment for segment in candidates if segment != "↑↓ scroll"]

        def painted(items: list[str]) -> str:
            return " · ".join(items)

        kept = list(candidates)
        for disposable in ("? when not typing", "/keys reopens", "↑↓ scroll"):
            if cell_len(painted(kept)) <= width:
                break
            kept = [segment for segment in kept if segment != disposable]
        faint = Style(color=theme_mod.semantic_color("faint"))
        words = Style(color=theme_mod.semantic_color("muted"))
        row = Text()
        for index, segment in enumerate(kept):
            if index:
                row.append(" · ", style=faint)
            # The hint is copy, not decoration: it names the way out and the
            # routes back in, so it rides the same `muted` rung as the card's
            # descriptions (design D3 — it was `dim`, 3.43:1 on the ground).
            row.append(segment, style=words)
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
        content = self._compose_content()
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

    def sync_layout(self, *, force: bool = False) -> None:
        """Repaint when the screen or live dock changed around the open card.

        Same signature and same guard as the usage card's, because the app
        drives ALL THREE cards from the same places — ``_sync_overlay_layout``,
        fed by the 1 Hz band tick and the resize timer. There is deliberately
        no ``on_resize`` handler here: this widget lives in a ``width: auto``
        host, so Textual delivers it no resize event and a handler would be
        dead code (the premise is stated in the app's own resize comment;
        review round 1, F1/U1 — the missing sync caller was the defect, not a
        missing event).
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
        self._jump_to(len(self._compose_content()))

    def _jump_to(self, target: int) -> None:
        """Clamp ``target`` into range and repaint.

        Named ``_jump_to`` rather than ``_scroll_to`` on purpose: ``Widget``
        owns ``_scroll_to(x, y, animate=…)`` and an incompatible override is
        the first defect class the widget conventions name — pyright's
        ``reportIncompatibleMethodOverride`` caught this one, which is why the
        name is not the obvious one.
        """
        rows = len(self._compose_content())
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
