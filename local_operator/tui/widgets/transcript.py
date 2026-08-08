"""Transcript container and the FINALIZED-BLOCK protocol.

The finalization protocol: blocks appended to the transcript
declare when they are done mutating. A block exposing ``is_finalized()`` is
treated as immutable by the container — its content is never updated again —
and ``settled_rows()`` reports how many of its rows are provably stable now
(used later for scroll accounting).

Spacing rhythm (the brand): blocks own NO uniform outer margin — the
container decides every gap. Separation is ADAPTIVE and opt-in: a block
takes a single blank row above it when the block before it was a different
KIND of thing, when that block rendered taller than one row, or when the
block itself is AIRY (:attr:`TranscriptBlock.SPACING_AIRY`). Tool rows are
airy: each one is a separate action, and stacked flush a run of them reads
as one wrapped block rather than as a ledger — reported from the field as
"there should be one line spacing between each". A list of one-line
notices still stacks tight, because that IS one thing said in parts. The
gap rides one CSS class (:data:`GAP_CLASS`); the base block selectors stay
margin-free so no "filler row everywhere" regression can slip back in, and
so the class can never be doubled by a margin underneath it.

Layout rhythm (D20): user blocks sit at the gutter (``❯`` at column 0);
everything else indents two cells so the turn spine reads at a glance.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Literal

from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.style import Style
from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import Static

from local_operator.tui import theme as theme_mod

#: The turn spine (D20): user prompts sit at the gutter; everything else
#: indents two cells so the ``❯`` column reads at a glance.
SPINE_INDENT = 2


def wrap_cells(text: str, width: int) -> list[str]:
    """Word-wrap ``text`` into rows of at most ``width`` CELLS.

    The wrapping sibling of :func:`truncate_cells`, and for the same reason:
    every measurement in this app goes through ``rich.cells.cell_len``, so a
    caller that needs rows instead of one clipped row must not fall back to
    ``textwrap`` (which counts codepoints and mis-wraps CJK by a factor of two).

    A word longer than the row — a URL, a resolved path, a session id — is broken
    rather than allowed to overhang, because the caller's whole reason for
    wrapping is that the overhang lands in another widget's column.
    """
    if width <= 0:
        return [text]
    rows: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if cell_len(candidate) <= width:
            current = candidate
            continue
        if current:
            rows.append(current)
            current = ""
        while cell_len(word) > width:
            head = ""
            used = 0
            for char in word:
                size = cell_len(char)
                if used + size > width:
                    break
                head += char
                used += size
            # Per-character sizes do not add up for a grapheme CLUSTER: `cell_len`
            # counts `1️⃣` (digit + VS16 + keycap) as 2 when handed the whole
            # string and as 1+0+0 per character, so a row built from the running
            # sum overhung its frame by a cell. Measure the finished head ONCE —
            # one call over at most `width` characters, so linear in the word,
            # not the quadratic re-measure of a growing string this loop used to
            # avoid — and shed trailing characters until it fits.
            while head and cell_len(head) > width:
                head = head[:-1]
            if not head:
                # A single character WIDER than the row (any CJK ideograph or
                # emoji at width 1). Taking nothing appended "" forever and grew
                # the row list without bound — a hung UI thread instead of a
                # mis-wrap, on exactly the inputs this function exists to handle.
                # One overhanging cell is the honest outcome: the caller's width
                # cannot hold this character at all.
                head = word[0]
            rows.append(head)
            word = word[len(head) :]
        current = word
    if current or not rows:
        rows.append(current)
    return rows


#: Tool-ledger name column: the floor every card agrees on, and the ceiling it
#: may grow to when the frame has room. Defined here rather than in the card
#: because the transcript owns the shared value (`tool_card` imports from this
#: module, so the constant cannot live there without a cycle).
TOOL_NAME_COL = 8
TOOL_NAME_COL_MAX = 24

#: CSS class opening exactly one blank row above a block. Applied by the
#: container, never by a block itself: only the container knows what came
#: before, and "what came before" is the entire spacing rule.
GAP_CLASS = "gap-above"


class TranscriptBlock(Static):
    """Base class for one transcript entry (assistant, tool, user, notice).

    Content is applied through :meth:`set_content`; once :meth:`finalize` is
    called the block is immutable — further :meth:`set_content` calls are
    ignored, which is the container's guarantee that committed rows never
    change under scroll.

    Row accounting (TUI-011): ``settled_rows`` is LAZY. ``set_content`` does
    not measure the renderable; the count is estimated from the renderable
    only when :meth:`settled_rows` is actually read (and memoized). The hot
    streaming path never pays for measurement.
    """

    DEFAULT_CSS = ""  # all styling lives in local_operator.tcss

    #: Grouping key for adaptive spacing. Blocks that share a kind stack
    #: tight while each stays one row; a change of kind always opens a gap.
    SPACING_KIND: ClassVar[str] = "block"
    #: True for a block that always opens a gap above itself regardless of
    #: what preceded it — the turn boundary, not a content difference.
    SPACING_LEAD: ClassVar[bool] = False
    #: True for blocks that appear and vanish within a turn. They neither
    #: take a gap nor anchor one, so nothing flickers when they are lifted.
    SPACING_TRANSIENT: ClassVar[bool] = False
    #: True for a block that is a ROW OF THE TOOL LEDGER — one settled or running
    #: tool call, drawn in the shared name column. Only these size that column:
    #: an approval prompt also carries a ``tool_name``, and letting it count made
    #: a pending question widen every settled row beneath it for a call that had
    #: not run and might yet be refused.
    LEDGER_ROW: ClassVar[bool] = False
    #: True for a block that takes a blank row above itself even after its
    #: OWN kind. Tool rows are the case: a run of them is a list of separate
    #: actions, not a paragraph of one, and stacked flush they read as a
    #: single wrapped block — the user reported exactly that ("there should
    #: be one line spacing between each"). Distinct from ``SPACING_LEAD``,
    #: which also fires against the empty transcript's top edge because it
    #: marks a turn boundary; this one only separates neighbours.
    SPACING_AIRY: ClassVar[bool] = False

    #: Set False once the block will never mutate again.
    _finalized: bool = False
    #: Last applied content, kept for lazy settled_rows measurement.
    _content: RenderableType | None = None
    #: Memoized settled row count (None = not measured yet).
    _settled_rows_cache: int | None = None
    #: Memoized "taller than one row?" answer for the spacing decision.
    _multirow_cache: bool | None = None

    def set_content(self, renderable: RenderableType) -> None:
        """Apply ``renderable`` as the block content (no-op once finalized)."""
        if self._finalized:
            return
        self._content = renderable
        self.invalidate_row_measurements()
        self.update(renderable)

    def invalidate_row_measurements(self) -> None:
        """Drop the memoized row counts (content or WIDTH changed).

        Width matters as much as content: the same renderable is one row at 90
        columns and three at 40, and the spacing rule asks this question of a
        block whose width may have been unknown when it first answered.
        """
        self._settled_rows_cache = None
        self._multirow_cache = None

    @property
    def renderable(self) -> RenderableType | None:
        """The current content renderable (rich) — inspection/test hook.

        Textual 8's ``Static`` no longer exposes a public ``renderable``;
        blocks keep their own reference so tests and exporters can read the
        exact rich object last applied via :meth:`set_content`.
        """
        return self._content

    def finalize(self) -> None:
        """Freeze the block; the container never re-renders it afterwards."""
        self._finalized = True

    def is_finalized(self) -> bool:
        """True when the block is immutable (FINALIZED-BLOCK protocol)."""
        return self._finalized

    def settled_rows(self) -> int:
        """Leading rows provably byte-stable now (all rows once finalized).

        Lazy: measured on first read after the last content change, at the
        block's OWN width when mounted (D3: no hardcoded reference width).
        """
        if not self._finalized:
            return 0
        if self._settled_rows_cache is None:
            self._settled_rows_cache = _count_rows(self._content, self.size.width or 80)
        return self._settled_rows_cache

    def spans_multiple_rows(self) -> bool:
        """True when the block currently renders taller than a single row.

        The ONE question adaptive spacing asks — deliberately a predicate
        rather than a row count, so a block that can answer it from its own
        state (a tool card knows; a streaming message knows from its source
        text) never pays for a full render just to be spaced correctly.
        The default measures whatever renderable is applied, memoized per
        content revision.
        """
        if self._multirow_cache is None:
            self._multirow_cache = _count_rows(self._content, self.size.width or 80) > 1
        return self._multirow_cache


class UserBlock(TranscriptBlock):
    """One user prompt at the gutter: a dim ``❯`` chevron at column 0."""

    #: A prompt starts a turn — always give it air, whatever came before.
    SPACING_KIND = "user"
    SPACING_LEAD = True

    def __init__(self, text: str) -> None:
        super().__init__()
        self.add_class("user-block")
        line = Text()
        line.append("❯ ", style=Style(color=theme_mod.semantic_color("dim")))
        line.append(text, style=Style(color=theme_mod.semantic_color("fg")))
        self.set_content(line)
        self.finalize()


#: The notice kinds, by ROLE rather than by volume. A typed alias, not a bare
#: ``str``, because the old signature let a wrong kind through silently: the app
#: passed ``"success"`` after a login and ``_KIND_TOKENS.get(kind, "dim")``
#: rendered it byte-identically to a nonsense kind, while the theme carried an
#: unused `success` green. A `Literal` makes that a type error at the call site.
NoticeKind = Literal["info", "note", "success", "warning", "error"]

#: Notice kind glyphs (D14): structure from symbols, not prefixes.
NOTICE_GLYPHS: dict[str, str] = {
    "info": "·",
    # Same glyph as `info` on purpose: `note` is the same KIND of statement (a
    # receipt), one weight up. A second symbol would claim a distinction of
    # meaning where the only difference is how much it wants reading.
    "note": "·",
    "success": "✓",
    "warning": "!",
    "error": "✗",
}


class NoticeBlock(TranscriptBlock):
    """One notice line: glyph + text, tinted by kind (D14), on the spine."""

    SPACING_KIND = "notice"

    #: Five tiers. ``info`` is `dim` — the quietest ink in the app, a step below a
    #: settled tool summary — which is right for a receipt nobody needs to read
    #: and wrong for one that answers a question the user is actively asking
    #: ("did my text just get thrown away?"). ``note`` is that middle weight:
    #: readable at a glance, not an alarm. Reaching for `warning` instead is what
    #: inverted the frame's colour budget, putting routine receipts in the
    #: loudest ink in the palette.
    #:
    #: Choose by ROLE, which is what makes the choice repeatable: ``info`` for a
    #: receipt nobody has to read, ``note`` for the answer to something the user
    #: just did, ``success`` for a completed action worth confirming, ``warning``
    #: for a state they must act on or know about, ``error`` for a failure.
    _KIND_TOKENS: ClassVar[dict[NoticeKind, str]] = {
        "info": "dim",
        "note": "muted",
        # NOT the theme's green. `success` #57c785 is already spent two rows up
        # on a diff's added lines, and the composer's caret carries the accent
        # #38c96a five cells below that — three greens on one surface, two of
        # them 5 dE2000 apart, none of them meaning what the others mean. The
        # completed action is carried by the ✓ GLYPH and the brightest plain ink,
        # which is the same trade the band already made when it moved its healthy
        # rung off this colour.
        "success": "fg",
        "warning": "warning",
        "error": "danger",
    }

    def __init__(self, text: str, kind: NoticeKind = "info") -> None:
        super().__init__()
        self.add_class("notice-block")
        self._text = text
        self._token = self._KIND_TOKENS.get(kind, "dim")
        self._glyph = NOTICE_GLYPHS.get(kind, "·")
        self.set_content(self._build())
        self.finalize()

    def on_resize(self, event: object) -> None:
        """Re-wrap at the new width (the same discipline as the tool row).

        A re-wrap is a HEIGHT change, so the spacing rule has to be asked again:
        the same notice is one row at 90 columns and three at 40, and adaptive
        spacing gaps a multi-row block where it packs single-row ones.
        """
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build())
        finally:
            self._finalized = was_finalized
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_around(self)

    def _build(self) -> RenderableType:
        """Glyph + text on the spine, WRAPPED with a hanging indent.

        A notice is the one block whose text can exceed its row and whose author
        cannot know that in advance ("ctrl+c again to exit - resume with: …" grows
        with the session id). Left to Rich's own fold it wrapped to column ZERO —
        the gutter the composer's own ``❯`` lives in — so the app's two loudest
        rows (one keystroke from exit, and a disarmed approval gate) were also the
        two that broke the spine. Wrapping here keeps every continuation row under
        the first character of the text, which is what makes a long notice read as
        one statement instead of as several.
        """
        style = Style(color=theme_mod.semantic_color(self._token))
        indent = " " * SPINE_INDENT
        hanging = " " * (SPINE_INDENT + 2)
        width = max((self.size.width or 80) - 2, 12)
        body = max(width - cell_len(hanging), 8)
        rows = wrap_cells(self._text, body)
        line = Text(no_wrap=True, overflow="ellipsis")
        for index, row in enumerate(rows):
            if index:
                line.append("\n")
                line.append(hanging, style=style)
            else:
                line.append(indent, style=style)
                line.append(f"{self._glyph} ", style=style)
            line.append(row, style=style)
        return line


class RichBlock(TranscriptBlock):
    """A finalized block wrapping one pre-built rich renderable.

    Used where the app needs multi-style content (``/help`` columns,
    structured listings) that the single-tint NoticeBlock cannot express.
    Content rides the spine indent (D20).
    """

    SPACING_KIND = "rich"

    def __init__(self, renderable: RenderableType) -> None:
        super().__init__()
        self.add_class("rich-block")
        from rich.padding import Padding

        self.set_content(Padding(renderable, (0, 0, 0, SPINE_INDENT)))
        self.finalize()


class WorkingBlock(TranscriptBlock):
    """The ONE aggregate working line (D25): shimmer sweeps it at 30 fps.

    A single working message, never per-row animation. When
    shimmer is disabled (settings/env), the line falls back to a static dim
    marker so the running state stays legible in a still frame (D26).
    """

    #: Lifted at turn end; it neither takes a gap nor anchors one, so no
    #: blank row appears and then vanishes underneath the settled rows.
    SPACING_TRANSIENT = True

    #: Repaint cadence — repaints animated loader text at 30 fps.
    _FRAME_MS = 33

    def __init__(self) -> None:
        super().__init__()
        self.add_class("working-block")
        self._frame_ms: float = 0.0
        self._timer = None
        self._paint()

    def on_mount(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled

        if shimmer_enabled():
            self._timer = self.set_interval(self._FRAME_MS / 1000, self._tick)

    def _tick(self) -> None:
        self._frame_ms += self._FRAME_MS
        self._paint()

    def _paint(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled, shimmer_text

        line = Text(" " * SPINE_INDENT)
        if shimmer_enabled():
            line.append_text(shimmer_text("working…", self._frame_ms))
        else:
            dim = Style(color=theme_mod.semantic_color("dim"))
            line.append("· ", style=dim)
            line.append("working…", style=dim)
        self.set_content(line)

    def stop(self) -> None:
        """Stop the repaint timer and settle on the static frame."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


def needs_gap_above(
    previous: "TranscriptBlock | None", block: "TranscriptBlock", *, splash_above: bool = False
) -> bool:
    """Whether ``block`` opens with one blank row given what preceded it.

    The whole adaptive-spacing rule, in one pure function so it can be
    reasoned about (and tested) without a running app:

    - nothing above → no gap; the transcript meets the top edge
    - nothing above but the VISIBLE empty state → a gap; the splash is a block
      too, and a receipt flush against its last hint row reads as a line that
      fell out of the lockup rather than as the answer to what the user just did
    - either side transient (the working line) → no gap; it will vanish
    - a turn-leading block (a user prompt) → always a gap
    - a different KIND of block → a gap; the change of subject is the cue
    - an AIRY block (a tool row) → a gap even after its own kind; each row
      is a separate action and flush rows read as one wrapped block
    - same kind, previous was ONE row → no gap; a list of one-line notices
      is a list, not a stack of paragraphs
    - same kind, previous was taller → a gap; multi-row output needs air or
      the next block reads as its continuation
    """
    if previous is None:
        return splash_above
    if block.SPACING_TRANSIENT or previous.SPACING_TRANSIENT:
        return False
    if block.SPACING_LEAD:
        return True
    if previous.SPACING_KIND != block.SPACING_KIND:
        return True
    if block.SPACING_AIRY:
        return True
    # Either side being tall opens the gap. Asking only about the block ABOVE
    # separated tall→short correctly and left short→tall packed flush, which is
    # the same wall the rule exists to prevent — just built in the other order.
    return previous.spans_multiple_rows() or block.spans_multiple_rows()


class TranscriptView(ScrollableContainer):
    """The scrolling column every block appends into.

    Separation is ADAPTIVE, decided by :func:`needs_gap_above` at the moment
    a block is appended (and re-decided for the one block below a block that
    changes height after the fact). Tool rows each take a blank row; a run of
    one-line notices stays flush. Nothing else pads: the base block selectors
    in the tcss declare no margin at all, so the gap can only ever come from
    the deliberate class. Appends scroll to the bottom unless the user has
    scrolled up to read.

    The container also owns KEYBOARD movement between focusable blocks
    (:meth:`focus_neighbour`): only it knows the append order, and a card
    asked to hand focus on has no other way to find its neighbour.

    ``clear_blocks`` notifies an optional ``on_clear`` hook (TUI-009) so the
    app can reset its streaming/tool-card bookkeeping.
    """

    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__()
        self._blocks: list[TranscriptBlock] = []
        self._on_clear: Callable[[], None] | None = None
        # The ledger's shared name column, recomputed lazily. Cached because it
        # is read once per card per repaint and only changes when the set of tool
        # names on screen does.
        self._name_col_cache: int | None = None

    def set_on_clear(self, hook: Callable[[], None] | None) -> None:
        """Install the hook fired after every :meth:`clear_blocks`."""
        self._on_clear = hook

    def append_block(self, block: TranscriptBlock) -> None:
        """Mount ``block`` at the bottom and keep the tail in view.

        Scrolling is deferred through ``call_after_refresh`` so the freshly
        mounted block's layout settles BEFORE ``scroll_end`` measures the
        virtual size (TUI-022) — an immediate scroll would target the stale
        pre-mount extent and land short.
        """
        self._apply_gap(self._anchor_before(len(self._blocks)), block)
        self._blocks.append(block)
        if hasattr(block, "tool_name"):
            self._invalidate_name_col()
        stick_to_bottom = self._is_near_bottom()
        self.mount(block)
        # The gap above was decided while the block was still UNMOUNTED, where
        # `spans_multiple_rows()` has no width to measure against and falls back
        # to 80 columns. That answer is right for most blocks and wrong for every
        # wrapping one in a narrow terminal, so the decision is retaken once the
        # block has a real width. Idempotent: the common case re-applies the same
        # class and nothing repaints.
        self.call_after_refresh(self._settle_gap, block)
        self._remeasure_empty_state()
        if stick_to_bottom:
            self.call_after_refresh(self.scroll_end, animate=False)

    def _remeasure_empty_state(self) -> None:
        """Re-measure the empty state after the block count in this region changed.

        The empty state budgets its height against the rows its siblings take
        (WelcomeView.get_content_height), and it reads those from their PLACED
        sizes — which the block mounted a line above this does not have yet. So
        the measurement is asked for again once the mount has been laid out;
        without it the splash keeps the height it was measured at when it was
        alone in the region, and overdraws it by the new block's rows.
        """
        for child in self.children:
            if not isinstance(child, TranscriptBlock) and child.display:
                # `layout=True`: a measured height is cached per container size, so
                # a plain repaint would redraw the new block into the old count.
                self.call_after_refresh(child.refresh, layout=True)

    def _settle_gap(self, block: TranscriptBlock) -> None:
        """Re-decide ``block``'s gap now that it has been laid out."""
        if block.parent is not self:
            return
        block.invalidate_row_measurements()
        self.refresh_gap_around(block)

    def refresh_gap_around(self, block: TranscriptBlock) -> None:
        """Re-decide the gaps ABOVE and BELOW ``block`` after it changed height.

        ``refresh_gap_after`` alone stopped being enough once notices began
        wrapping: the spacing rule reads the multi-row state of BOTH neighbours,
        so a block that grows from one row to three changes the answer for its
        own gap as well as for its follower's. Looking only downward left a
        wrapped notice flush against whatever it followed.
        """
        try:
            index = self._blocks.index(block)
        except ValueError:
            return
        self._apply_gap(self._anchor_before(index), block)
        self.refresh_gap_after(block)

    @property
    def tool_name_col(self) -> int:
        """Cells the tool ledger gives its name column, shared by every card.

        Grown to the longest name currently on screen, floored at ``NAME_COL`` and
        capped so the column stays a spine. Recomputed on demand and cached until
        the ledger changes, because it is read once per card per repaint.
        """
        if self._name_col_cache is None:
            from local_operator.tui.glyphs import display_name

            longest = 0
            for block in self._blocks:
                # `LEDGER_ROW`, not "has a tool_name": the approval prompt has one
                # too, and a PENDING question was widening the column for every
                # settled row beneath it — for a tool that had not run and might
                # be refused, and the widening survived the refusal.
                if not getattr(block, "LEDGER_ROW", False):
                    continue
                name = getattr(block, "tool_name", "")
                if isinstance(name, str) and name:
                    longest = max(longest, cell_len(display_name(name)))
            self._name_col_cache = max(TOOL_NAME_COL, min(longest, TOOL_NAME_COL_MAX))
        return self._name_col_cache

    def invalidate_name_col(self) -> None:
        """Public entry point: a card's NAME changed, so the column may have.

        A composing row follows the tool name as its fragments arrive, and the
        column is derived from those names — without this the first fragment's
        width outlived it for the rest of the session.
        """
        self._invalidate_name_col()

    def _invalidate_name_col(self) -> None:
        """Forget the cached column and repaint the ledger if it moved.

        Only the cards repaint, and only when the number actually changed: a
        ledger that reflowed on every append would undo the point of a spine.
        """
        previous = self._name_col_cache
        self._name_col_cache = None
        if previous is not None and previous != self.tool_name_col:
            for block in self._blocks:
                repaint = getattr(block, "refresh_row", None)
                if callable(repaint):
                    repaint()

    def refresh_gap_after(self, block: TranscriptBlock) -> None:
        """Re-decide the gap for the first real block below ``block``.

        Called when a block changes height after the fact — a tool card
        expanding from one row to many. Only the immediate neighbour can
        change, so this stays O(1) rather than restyling the transcript.
        """
        try:
            index = self._blocks.index(block)
        except ValueError:
            return
        for following in self._blocks[index + 1 :]:
            if following.SPACING_TRANSIENT:
                continue
            self._apply_gap(block, following)
            return

    def focus_neighbour(self, block: TranscriptBlock, delta: int) -> bool:
        """Focus the nearest focusable block ``delta`` steps from ``block``.

        The keyboard half of the expand affordance. Returns False when there
        is nothing in that direction, which is the caller's cue to fall
        through to the screen's ordinary tab order — walking off the bottom
        of the ledger lands in the composer, which is where a user who just
        finished reading wants to be, and walking off the top lands on the
        transcript itself so the scroll keys take over.

        Skips blocks that cannot take focus (prose, notices) rather than
        stopping at them: what the arrow keys traverse is the list of
        ACTIONABLE rows, and a stop on an inert paragraph would read as the
        key having failed.
        """
        try:
            index = self._blocks.index(block)
        except ValueError:
            return False
        step = 1 if delta > 0 else -1
        cursor = index + step
        while 0 <= cursor < len(self._blocks):
            candidate = self._blocks[cursor]
            if candidate.focusable:
                candidate.focus()
                return True
            cursor += step
        return False

    def _anchor_before(self, index: int) -> TranscriptBlock | None:
        """The last block before ``index`` that counts as "what came before".

        Transient blocks are invisible to spacing: the working line sits
        between a tool row and the next one for a second and must not change
        how they relate.
        """
        for candidate in reversed(self._blocks[:index]):
            if not candidate.SPACING_TRANSIENT:
                return candidate
        return None

    def _apply_gap(self, previous: TranscriptBlock | None, block: TranscriptBlock) -> None:
        block.set_class(
            needs_gap_above(previous, block, splash_above=self._empty_state_visible()), GAP_CLASS
        )

    def _empty_state_visible(self) -> bool:
        """True when the empty-state view is showing above the blocks.

        Identified by what it is NOT — every other child of this container is a
        transcript block — rather than by importing the view, which imports this
        module for its notice glyphs. It is also the more honest predicate: what
        spacing needs to know is whether something is drawn above the first block,
        not which widget that something happens to be.
        """
        return any(
            child.display for child in self.children if not isinstance(child, TranscriptBlock)
        )

    def remove_block(self, block: TranscriptBlock) -> None:
        """Remove one block (used to lift the boot hint, D9)."""
        if block not in self._blocks:
            return
        index = self._blocks.index(block)
        self._blocks.remove(block)
        block.remove()
        # Same reason `clear_blocks` does it: the name column is derived FROM the
        # blocks, so a removal can only ever make it too wide.
        if getattr(block, "LEDGER_ROW", False):
            self._name_col_cache = None
        # Whatever fell into the removed block's place now has a different
        # neighbour above it — most visibly the very first block, which must
        # never carry a gap once the boot hint is lifted off the top.
        for offset in range(index, len(self._blocks)):
            following = self._blocks[offset]
            if following.SPACING_TRANSIENT:
                continue
            self._apply_gap(self._anchor_before(offset), following)
            return

    def blocks(self) -> list[TranscriptBlock]:
        """Blocks in append order (live and finalized)."""
        return list(self._blocks)

    def clear_blocks(self) -> None:
        """Remove every block (the ``/clear`` command)."""
        for block in self._blocks:
            block.remove()
        self._blocks.clear()
        # Every derived measurement goes with them. The name column is computed
        # FROM the blocks, so a stale one made the next ledger inherit the width
        # of a transcript the user just cleared.
        self._name_col_cache = None
        self.scroll_home(animate=False)
        if self._on_clear is not None:
            self._on_clear()

    def _is_near_bottom(self) -> bool:
        """True when the viewport sits at (or within 2 rows of) the bottom."""
        max_offset = self.virtual_size.height - self.size.height
        if max_offset <= 0:
            return True
        return self.scroll_offset.y >= max_offset - 2


def _count_rows(renderable: RenderableType | None, width: int = 80) -> int:
    """Row count a renderable occupies at ``width``, measured through rich.

    WRAPPING COUNTS. Counting source newlines only reported 1 for a 400-char
    single-line notice that actually paints nine rows, which fed
    ``spans_multiple_rows`` and silently disabled the adaptive gap below tall
    blocks — the exact "decays back into uniform filler" failure the spacing
    rule is defended against, one layer lower.

    Only called lazily from ``settled_rows``/``spans_multiple_rows`` — never on
    the streaming path.
    """
    if renderable is None:
        return 0
    inner = max(width, 10)
    if isinstance(renderable, str):
        renderable = Text(renderable)
    if isinstance(renderable, Text):
        # cell-aware, so CJK and emoji account correctly; ceil-divide each
        # logical line by the available width.
        rows = 0
        for line in renderable.plain.splitlines() or [""]:
            cells = cell_len(line)
            rows += max(1, -(-cells // inner))  # ceil without float error
        return max(1, rows)
    console = Console(width=inner)
    try:
        segments = console.render(renderable, console.options)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — measurement must never break a render
        return 1
    rows = 1
    for segment in segments:
        rows += segment.text.count("\n")
    return rows
