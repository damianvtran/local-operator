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

Layout rhythm (D20): a user prompt owns column 0 — a full-height ``▌`` rule
beside every one of its rows — and its prose sits two cells in, exactly where
the old ``❯ `` prefix put it, so the turn spine reads at a glance.
"""

from __future__ import annotations

import time
from typing import Callable, ClassVar, Literal

from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.style import Style
from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import Static

from local_operator.harness.intent import ACTIVITY_THINKING
from local_operator.tui import theme as theme_mod

#: The turn spine (D20): user prompts sit at the gutter; everything else
#: indents two cells so the gutter column reads at a glance.
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

        This does NOT settle the block's layout height. Textual keeps its own
        copy of that in ``Widget._content_height_cache``, keyed on the WIDTH
        ALONE, and clearing it here was tried and is not enough — a block that
        re-wraps itself has to PIN its height instead (see
        :meth:`UserBlock._build`).
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
    """One user prompt behind a full-height rule in the gutter column.

    Reported from the field: "have user messages have a more obvious
    delineation […] consider that they will often be multi-line with
    paragraphs". The old treatment was a ``❯`` on the FIRST row only, so a
    three-paragraph prompt read as three assistant paragraphs the moment it
    wrapped — measured at 60 columns, even the opening paragraph lost its
    marker on its second row.

    The four decisions this block turns on, and why:

    **The rule runs down EVERY row**, wrapped continuations and the blank rows
    between paragraphs included. A marker that appears once marks a LINE; a
    marker on every row marks a BLOCK, and the block is what the reader is
    trying to find when scrolling back. The blank paragraph rows are the case
    that decides it: skip them and the rule breaks into segments, which is the
    same "three separate things" failure in a new costume.

    The strongest case for this is not prose but a PASTED SNIPPET, which is why
    the indentation is kept verbatim (see :meth:`_rows`). The rule sits OUTSIDE
    the text field — column 0-1, in a different ink — so the field has exactly
    one origin, at column 2, and every indent level measures from it. Without
    the bar, ``def f():`` two cells right of the assistant's column 0 is
    ambiguous: the reader cannot tell the app's indent from the paste's. With
    it, the bar says where the content field starts and the ladder inside is
    unmistakably the author's. A rule at column 2 with prose at 4, or prose left
    at 0, would have made the two systems share an origin and collide.

    **No background tint.** Elevation-as-a-background-step is already spent, in
    full, on the tool ledger — every tool row is a filled slab, and that fill
    CARRIES the outcome. A second slab kind on the same surface makes the
    transcript a stack of cards and demotes the one element whose fill means
    something. The gutter column carries the whole signal instead, which is
    affordable because nothing else paints there: assistant prose starts at
    column 0 with no glyph, notices and tool rows start at :data:`SPINE_INDENT`.
    A column no other block touches does not need a hue to be unmistakable.

    **The rule is ``dim``, not the accent.** The accent green is spent on
    exactly five sites (enumerated in ``local_operator.tcss``) and means "a turn
    is live"; a green column beside every prompt would be the largest accent
    surface in the app and would mean nothing. ``dim`` is the sheet's own
    separator ink, and ``▌`` (LEFT HALF BLOCK) buys the weight back through the
    GLYPH — half a cell of solid ink, where ``│`` would draw the left edge of a
    box the minimalism contract forbids.

    **Spacing is unchanged.** :attr:`SPACING_LEAD` already opens a row above
    every prompt and the block below is always a different
    :attr:`SPACING_KIND`, so the existing adaptive rule brackets the rule with
    one row of GROUND on each side — which is precisely what keeps it reading
    as a sidebar rather than as a card with padding.
    """

    #: A prompt starts a turn — always give it air, whatever came before.
    SPACING_KIND = "user"
    SPACING_LEAD = True

    #: The gutter glyph and the cells it claims. The width is exactly
    #: :data:`SPINE_INDENT` so the prose lands in the same text column the old
    #: ``❯ `` prefix put it in and no other block moves.
    RULE = "▌"
    RULE_COLS = SPINE_INDENT
    #: Semantic ink for the rule and for the prose beside it. Named rather than
    #: inlined so the two candidate weights could be rendered and COMPARED at
    #: 120 and 60 columns instead of argued about; see the class docstring for
    #: why the answer is not the accent.
    RULE_TOKEN = "dim"
    TEXT_TOKEN = "fg"

    #: Narrowest body the text is wrapped into. Below ``RULE_COLS + MIN_BODY``
    #: — a 10-column terminal — rows are built wider than the frame and Rich
    #: CLIPS them with an ellipsis (``overflow="ellipsis"`` in :meth:`_build`).
    #: That is the deliberate trade: wrapping into the two or three cells left
    #: over turns a sentence into a column of single characters, and dropping
    #: the rule to buy them back loses the delineation exactly where the frame
    #: is most crowded and the reader needs it most.
    MIN_BODY = 8

    def __init__(self, text: str) -> None:
        super().__init__()
        self.add_class("user-block")
        self._text = text
        self.set_content(self._build())
        self.finalize()

    def text(self) -> str:
        """The prompt as submitted, newlines and indentation intact."""
        return self._text

    def on_resize(self, event: object) -> None:
        """Re-wrap at the new width, then re-ask the spacing question.

        Same discipline as :class:`NoticeBlock`: this block wraps ITSELF (Rich's
        own fold would return the continuation rows to column 0 and eat the
        rule), so a width change is a content change. It is also a HEIGHT
        change, and adaptive spacing gaps a multi-row block where it packs
        single-row ones — one prompt is one row at 120 columns and four at 60.

        The height change feeds back: rebuilding at a new width changes the row
        count, which is itself a resize, so one drag step costs TWO builds
        (measured: a 400-word prompt dragged 120→60 rebuilds at 56 cells twice,
        then settles — the second pass produces the same rows, so it converges
        rather than oscillating). A HEIGHT-only terminal resize costs none: the
        block's width is unchanged, so it is never sent a resize at all.
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

    def _rows(self, body: int) -> list[str]:
        """The prompt's text rows at ``body`` cells, gutter not yet applied.

        Paragraphs are preserved as authored: the text is split on newlines
        first and each paragraph wrapped independently, so a blank line in the
        prompt stays a blank row here — which is what keeps the rule continuous
        across a paragraph break instead of segmenting it.

        Blank rows are trimmed at the ENDS, though, because a blank row is only
        meaningful BETWEEN paragraphs: at the edge it separates content from
        nothing, and paints a stub of rule beside no text. Every caller in
        ``app.py`` strips its text today, so this is not reachable from the
        product — it is here because the block owns its own rendering and a
        future caller should not be able to produce a dangling bar. At least one
        row always survives, so an empty prompt is still a block with a height.
        """
        rows: list[str] = []
        for paragraph in self._text.split("\n"):
            # Leading spaces are lifted out before wrapping and put back on
            # every row. `wrap_cells` splits on " " and rebuilds with a single
            # separator, which preserves an INTERIOR run of spaces (the empty
            # words re-add their separator once `current` is truthy) and drops a
            # LEADING one (`current` is still "" and falsy through every empty
            # word). Rich preserved it before this block started wrapping
            # itself, so a pasted code snippet used to keep its shape and then
            # came out flush left, one line at a time — the exact multi-line
            # case the delineation exists for. Fixed here, not in `wrap_cells`:
            # that helper also lays out notices and tool rows, which have no
            # authored indentation to keep.
            stripped = paragraph.lstrip(" ")
            indent = " " * min(len(paragraph) - len(stripped), max(body - 1, 0))
            room = max(body - len(indent), 1)
            rows.extend(indent + row if row else "" for row in wrap_cells(stripped, room))
        while len(rows) > 1 and not rows[0]:
            rows.pop(0)
        while len(rows) > 1 and not rows[-1]:
            rows.pop()
        return rows

    def _build(self) -> RenderableType:
        """The prompt, every row prefixed by the gutter rule.

        The height is PINNED to the row count rather than left to ``auto``, the
        same trade ``ToolCard`` and the command picker already make and for a
        sharper version of the same reason. Under ``auto`` the layout engine
        MEASURES this widget, and its measurement is cached on
        ``Widget._content_height_cache`` keyed on the WIDTH ALONE, which
        ``Static.update`` never clears. The block is built before it is laid
        out, so the first measurement is taken of the 80-column fallback build
        folded to fit the real width — inflated — and the correct rebuild that
        arrives with the resize then paints fewer rows into the reserved space.
        Reported from the subagent page: a three-paragraph prompt reserved 10
        rows and painted 8, leaving a two-row hole mid-transcript, and it
        survived clearing the cache because nothing re-ran layout afterwards.
        A block that authors its own rows KNOWS its height; measuring it is the
        bug. Writing the style is itself a layout refresh, so the correction
        lands on the next pass.
        """
        rule_style = Style(color=theme_mod.semantic_color(self.RULE_TOKEN))
        text_style = Style(color=theme_mod.semantic_color(self.TEXT_TOKEN))
        body = max((self.size.width or 80) - self.RULE_COLS, self.MIN_BODY)
        gutter = self.RULE + " " * (self.RULE_COLS - cell_len(self.RULE))
        rows = self._rows(body)
        self.styles.height = len(rows)
        line = Text(no_wrap=True, overflow="ellipsis")
        for index, row in enumerate(rows):
            if index:
                line.append("\n")
            line.append(gutter, style=rule_style)
            if row:
                line.append(row, style=text_style)
        return line


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


#: What the working line says before any event has named something narrower.
#: A turn always opens with a model call in flight, so this is a statement of
#: fact and not a placeholder. No trailing ellipsis: the clock that follows
#: every label already says the thing is ongoing, and says it with a number.
DEFAULT_ACTIVITY = ACTIVITY_THINKING


class WorkingBlock(TranscriptBlock):
    """The ONE aggregate working line (D25): what the turn is doing, right now.

    A single working message, never per-row animation. Shimmer sweeps it at
    30 fps; when shimmer is disabled (settings/env), the line falls back to a
    static dim marker so the running state stays legible in a still frame (D26).

    It carries an ACTIVITY and a clock, not the word "working". The gaps this
    line exists to cover are the ones with no ledger row at all — the wait for
    the first token, and the model call between one tool batch and the next —
    and through those a constant "working…" said nothing the animation had not
    already said. Every label it shows is derived from an event the app actually
    received (``OperatorApp._current_activity``); the line never invents one.

    What it deliberately does NOT do is restate the row above it. Pinned to the
    foot of the transcript it sits directly under the live tool card, and a
    label built from the same arguments painted that call's description twice in
    consecutive rows, which read as a rendering fault. It names the KIND of work
    and how many of them, and leaves the detail to the ledger; the clock and the
    count are then the two things on this row that appear nowhere else.

    It also has to STAY at the foot of the conversation, which is
    :meth:`TranscriptView.pin_tail`'s job, not this widget's — mounted once at
    turn start and left where it landed, it was stranded under the prompt that
    opened the turn while the ledger grew past it off the bottom of the screen.
    """

    #: Lifted at turn end. It takes a blank row above it like any other change
    #: of kind: the suppression this used to carry was justified by the line
    #: sitting MID-transcript, where the gap would appear and vanish under the
    #: settled rows — the tail pin made it permanently last, so the gap is
    #: constant for the life of the turn and goes when the turn does. Flush, it
    #: inherited the exact failure the airy rule exists to prevent and read as a
    #: caption on the card above it. It still never ANCHORS a gap (nothing is
    #: ever below it), which is what ``previous.SPACING_TRANSIENT`` covers.
    SPACING_TRANSIENT = True

    #: Repaint cadence — repaints animated loader text at 30 fps.
    _FRAME_MS = 33

    #: Repaint cadence with shimmer OFF. The band is still, but the clock is
    #: not: an elapsed reading frozen at the second the activity began asserts
    #: a stale age for a gap that is still growing, which is the one number
    #: this line exists to report. One repaint a second is what the clock's own
    #: resolution needs and no more.
    _STATIC_FRAME_MS = 1000

    #: The head glyph, cycled off the same timer as the shimmer. The braille
    #: spinner is already this app's word for "running" (the status band and the
    #: subagent panel both use this exact tuple), it is plain Unicode so it
    #: survives a terminal with no patched font, and it spends no accent —
    #: motion, not colour, says alive.
    #:
    #: It replaced a "· ", which was not the differentiator its comment claimed:
    #: `·` is NOTICE_GLYPHS["info"], painted in the same dim ink at the same
    #: column, so the working line was shaped byte-identically to an inert
    #: receipt — and on the compaction path the notice `· compacting context…`
    #: sat directly above the line `· compacting context`, a word-for-word
    #: double print.
    _SPINNER = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
    #: Glyph advance, independent of the repaint rate: 30 fps is the shimmer's
    #: cadence and would spin this into a blur.
    _SPIN_MS = 80

    #: Cells reserved for the clock, CONSTANT so the label's clip point does not
    #: move as the number grows: an unreserved clock re-clipped the label at 10s,
    #: at 1m40s and at the hour, creeping the text leftward under the eye. Two
    #: spaces plus the five the tool ledger gives its own duration column
    #: (``tool_card.DURATION_COL``), inlined because that module imports this one.
    _CLOCK_COL = 7

    def __init__(self, activity: str = DEFAULT_ACTIVITY, phase: str = DEFAULT_ACTIVITY) -> None:
        super().__init__()
        self.add_class("working-block")
        self._frame_ms: float = 0.0
        self._tick_ms: float = self._FRAME_MS
        self._animated = True
        self._timer = None
        self._activity = activity or DEFAULT_ACTIVITY
        self._phase = phase
        # The clock times the CURRENT PHASE, not the turn and not the label: how
        # long the agent has been busy altogether is the status band's
        # `duration` segment, and the question this line answers is the other
        # one — how long the thing on screen has been the thing on screen.
        self._phase_started = time.monotonic()
        self._clock = ""
        self._paint()

    @property
    def activity(self) -> str:
        """The label currently on the line (what the turn is doing)."""
        return self._activity

    def set_activity(self, activity: str, phase: str | None = None) -> None:
        """Name what the turn is doing now.

        The clock restarts only when the PHASE changes, not whenever the label
        does. Keying it to the rendered string made the row refute itself: one
        call of a three-call batch settling showed ``✓ 4.0s`` on its receipt
        while the line two rows below reset to ``running 2 tools  0s``, and a
        batch that shed a call every twenty seconds could never show a clock
        past twenty — which is exactly the "has this been stuck" question the
        clock exists to answer. A count changing, a tool name arriving in
        fragments and an intent being revised are all the same phase.
        """
        activity = activity or DEFAULT_ACTIVITY
        phase = phase or activity
        if phase != self._phase:
            self._phase = phase
            self._phase_started = time.monotonic()
        elif activity == self._activity:
            return
        self._activity = activity
        self._paint()

    def on_mount(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled

        self._animated = shimmer_enabled()
        self._tick_ms = self._FRAME_MS if self._animated else self._STATIC_FRAME_MS
        self._timer = self.set_interval(self._tick_ms / 1000, self._tick)

    def on_resize(self, event: object) -> None:
        """Re-truncate at the new width (the label is clipped, never wrapped)."""
        self._paint()

    def _tick(self) -> None:
        self._frame_ms += self._tick_ms
        if self._animated:
            self._paint()
            return
        # With shimmer off the spinner is frozen too (D26 pins a still frame),
        # so the clock is the only thing that can change and a repaint landing
        # on the same second is one nobody can see.
        if self._clock_text() != self._clock:
            self._paint()

    def _clock_text(self) -> str:
        """How long the current PHASE has run, in the ledger's own grammar."""
        from local_operator.tui.widgets.tool_card import format_duration

        return format_duration(time.monotonic() - self._phase_started)

    def _paint(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled, shimmer_text
        from local_operator.tui.widgets.tool_card import truncate_cells

        dim = Style(color=theme_mod.semantic_color("dim"))
        animated = shimmer_enabled()
        # ALWAYS shown, from the first frame. It is the one fact this row has
        # that nothing else on screen does — a running tool's own card carries
        # no duration until it settles, and the band's clock is the session's
        # cumulative active time, not this phase's age.
        self._clock = self._clock_text()
        # A frozen frame rather than no glyph when shimmer is off: the braille
        # head is unique to this row either way, which is what a still terminal
        # needs to tell it from an info notice.
        head = self._SPINNER[int(self._frame_ms // self._SPIN_MS) % len(self._SPINNER)]
        head = f"{head} " if animated else f"{self._SPINNER[0]} "
        # ONE row, always. The block is SPACING_TRANSIENT, so nothing below it
        # re-measures against its height; a label that wrapped would take rows
        # it had told the transcript it would not, and the intents it shows are
        # model-supplied and of no bounded length. The clock's cells are
        # reserved rather than measured, so the clip point holds still.
        width = max(
            (self.size.width or 80) - SPINE_INDENT - cell_len(head) - self._CLOCK_COL,
            8,
        )
        label = truncate_cells(self._activity, width)
        line = Text(" " * SPINE_INDENT)
        line.append(head, style=dim)
        if animated:
            line.append_text(shimmer_text(label, self._frame_ms))
        else:
            line.append(label, style=dim)
        line.append(f"  {self._clock}", style=dim)
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
    - the block BELOW is transient (nothing is; the working line is pinned
      last) → no gap; a blank row that appeared and then vanished under the
      settled rows is the flicker this rule was written to avoid
    - the block ITSELF is transient (the working line) → a gap; it is pinned to
      the foot for the whole turn, so the row is constant rather than a
      flicker, and flush it read as a caption on the card above it — the exact
      "flush rows read as one wrapped block" failure the airy rule exists for
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
    if previous.SPACING_TRANSIENT:
        return False
    if block.SPACING_TRANSIENT:
        return True
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

    def __init__(
        self, *, id: str | None = None, classes: str | None = None  # noqa: A002 (Textual's name)
    ) -> None:
        # Keyword-only and optional so the app can tell its two transcripts
        # apart. There are two once the full-page subagent view is open (the
        # main conversation, hidden, and the child's), and `query_one` on the
        # TYPE would then be ambiguous — which is exactly the failure mode a
        # docked, always-present transcript must not have. The child's is
        # identified by CLASS instead: it is created and removed as the mode
        # opens and closes, and `remove()` only posts a prune, so a reopen
        # inside that window would collide on a unique id.
        super().__init__(id=id, classes=classes)
        self._blocks: list[TranscriptBlock] = []
        self._on_clear: Callable[[], None] | None = None
        # The ledger's shared name column, recomputed lazily. Cached because it
        # is read once per card per repaint and only changes when the set of tool
        # names on screen does.
        self._name_col_cache: int | None = None
        #: The block held at the BOTTOM as later blocks arrive (the working
        #: line). Pinned rather than re-appended so it is never unmounted and
        #: remounted mid-turn, which would restart its timer and its clock.
        self._tail: TranscriptBlock | None = None

    def set_on_clear(self, hook: Callable[[], None] | None) -> None:
        """Install the hook fired after every :meth:`clear_blocks`."""
        self._on_clear = hook

    def pin_tail(self, block: TranscriptBlock) -> None:
        """Append ``block`` and hold it last as the transcript grows.

        The working line has to travel with the conversation: appended once at
        turn start it stayed under the prompt that opened the turn, and by the
        time the turn had run three tools the only live thing on screen sat
        somewhere in the scrollback. A pin costs one branch in
        :meth:`append_block` and keeps the widget itself untouched, which is
        what a re-append could not do — remounting resets the repaint timer and
        the elapsed clock the line is reporting.

        Only one block is ever pinned; a second pin replaces the first rather
        than stacking, because "the bottom" admits one occupant.
        """
        self._tail = None
        self.append_block(block)
        self._tail = block

    def append_block(self, block: TranscriptBlock) -> None:
        """Mount ``block`` at the bottom and keep the tail in view.

        "The bottom" means above the PINNED tail when there is one — the
        transcript grows underneath the working line, not past it.

        Scrolling is deferred through ``call_after_refresh`` so the freshly
        mounted block's layout settles BEFORE ``scroll_end`` measures the
        virtual size (TUI-022) — an immediate scroll would target the stale
        pre-mount extent and land short.
        """
        tail = self._tail if self._tail is not block else None
        index = self._blocks.index(tail) if tail in self._blocks else len(self._blocks)
        self._apply_gap(self._anchor_before(index), block)
        self._blocks.insert(index, block)
        if hasattr(block, "tool_name"):
            self._invalidate_name_col()
        stick_to_bottom = self._is_near_bottom()
        # `before=None` is Textual's own "append" — one mount call either way.
        self.mount(block, before=tail if tail in self._blocks else None)
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
                # A call the model is still DICTATING is the same case one state
                # over, and it arrived with the rename fix: the name is
                # model-controlled and arrives in fragments, so a single announced
                # 200-character name took the column to its cap and shifted every
                # settled receipt sixteen cells right — and, exactly as with the
                # refusal, the widening outlived the row when it settled as
                # `never sent`. A row contributes its name once the call it names
                # has actually started.
                if getattr(block, "contributes_name", True) is False:
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
        # A pin naming a block that is gone would send every later append to a
        # widget the container no longer holds.
        if self._tail is block:
            self._tail = None
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
        # of a transcript the user just cleared. The pin goes too — the block it
        # named was just removed, and the hook below is where a live turn's
        # working line is mounted again.
        self._name_col_cache = None
        self._tail = None
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
