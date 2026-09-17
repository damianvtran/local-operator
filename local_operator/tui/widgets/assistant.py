"""Assistant message block — rich Markdown with the FROZEN-PREFIX trick.

Naive re-render-the-whole-message-per-token is quadratic and is the dominant
streaming cost (a real-world quadratic-re-render bug). Instead we keep a ``(frozen_text,
frozen_rendered)`` pair and re-render only the *tail* after the last settled
block boundary on each update.

A settled boundary is a blank line that is not inside a code fence. Block
tokenization is local across such a boundary, so ``render(prefix) +
render(tail) == render(prefix + tail)``, which lets us cache the prefix's
render and only re-wrap the tail. Turns quadratic streaming reveal into
linear.

Hot-path hygiene (TUI-011):

- Fence coverage is INCREMENTAL: append-only updates scan for fence markers
  only in the NEW text while carrying the running ``in_fence`` state and the
  covered-line set; a full re-scan runs only when the update was not a pure
  append. The frozen prefix is never re-lexed on the streaming path.
- Row accounting is exact and free: the frozen prefix is already flattened, so
  ``settled_rows`` COUNTS its rows instead of re-rendering the markdown at a
  guessed width to measure them.

Splice preconditions (TUI-010): the freeze is REFUSED when it would break
markdown semantics across the boundary — a reference-link definition line
(``[label]: url``) in the frozen prefix can pair with a link in the tail, and
a list item closing the prefix can continue in the tail. In both cases the
boundary defers (or is refused outright).

An equality guard short-circuits identical re-emits: providers re-emit the
same text on no-delta ticks, and without the guard the full parse + wrap runs
per tick. Theme epochs (TUI-016) invalidate the frozen renderable.
"""

from __future__ import annotations

import re

from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from textual.content import Content
from textual.selection import Selection

from local_operator.tui import theme as theme_mod
from local_operator.tui.markdown_theme import brand_markdown_theme
from local_operator.tui.settings import settings_get
from local_operator.tui.widgets import _copy_markdown
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

#: Reference-link definition line: ``[label]: target`` (TUI-010 refusal).
_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s", re.MULTILINE)
#: A line that continues a list: bullet or ordered item (TUI-010 deferral).
_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d{1,9}[.)])\s")
#: Fence marker: 3+ backticks or tildes (commonmark fenced-code opener).
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: Width a block builds at before it has been laid out. The correct width
#: arrives with the first resize, which rebuilds (see
#: :meth:`AssistantBlock.on_resize`).
FALLBACK_WIDTH = 80

#: The assistant rail: the same two cells ``UserBlock`` spends on its prompt
#: rule, in a DIFFERENT ink and a NARROWER glyph. The transcript's complaint was
#: that a user message has a line down its side and the model's answer has
#: nothing, so the answer gets the same delineation.
#:
#: ``U+258E`` (quarter block), not the rule's ``U+258C`` (half block), and the
#: difference is not decorative. Colour alone did not carry the distinction:
#: ``label`` and ``signal`` are ISOLUMINANT in the shipped ramps — measured
#: 1.08:1 contrast against each other on dark and 1.00:1 on light — so in
#: greyscale, and under every CVD simulation (never above 1.24:1), a one-line
#: user message and a one-line assistant reply were the same glyph in the same
#: column at the same weight, which is the exact confusion this rail exists to
#: remove. Weight is a SECOND, non-colour channel: it survives greyscale, it
#: survives colour blindness, and it costs nothing in the long-block case that
#: colour already handled.
#:
#: Changed here rather than on ``UserBlock``: the prompt rule is shipped and
#: restyling it is not this slice's to do. The glyph is ALSO not the blockquote
#: bar, and that is now THE MECHANISM rather than a secondary benefit: a quote
#: row carries both marks and they are told apart by WIDTH — quarter block
#: against half block, in different columns — which is what let the old
#: reserve-the-cells-and-paint-nothing exception go (see :func:`rail_rows`).
RAIL = "\u258e"
#: The glyph RICH paints for a blockquote bar — not ours, and named here so the
#: tests can assert that the rail and the bar are DIFFERENT glyphs in DIFFERENT
#: columns, which is what makes a railed quote row read as nesting. It is also
#: ``UserBlock.RULE``; that shared identity was the collision the rail used to
#: step around, back when the rail was this same glyph.
QUOTE_BAR = "\u258c"
#: Exactly :data:`SPINE_INDENT`, imported rather than re-spelled as ``2``. The
#: prompt rule and this rail must be the same width BY CONSTRUCTION — the field
#: has one text origin, and two gutters that agree by coincidence drift the
#: first time one of them is tuned (``UserBlock`` docstring, transcript.py).
RAIL_COLS = SPINE_INDENT
#: ``label``, and deliberately neither ``signal`` nor ``accent``.
#:
#: ``signal`` is the PROMPT's rule: painting the answer in it was rendered,
#: compared and rejected, because it makes a user block and an assistant block
#: look alike and so removes the only distinction this rail exists to draw.
#: ``accent`` is a closed five-site budget this is not entitled to spend.
#: Measured as a graphical object against ``bg`` across all 54 registered
#: themes: no theme below the 3:1 non-text floor, worst ``tokyo-night-day`` at
#: 4.32:1 — and ``label`` is never equal to ``signal`` or ``accent`` in any of
#: them, so the rail can never accidentally read as a prompt.
RAIL_TOKEN = "label"
#: Narrowest body the prose is folded into WHEN THE LANE CAN AFFORD IT, the
#: floor ``UserBlock.MIN_BODY`` established for the same trade: below it,
#: wrapping into the two or three cells the rail leaves turns a sentence into a
#: column of single characters.
#:
#: It is a floor on legibility, NOT a licence to paint outside the block. Where
#: the lane cannot pay it, :meth:`AssistantBlock._body_width` clamps it to what
#: the lane has — see that method for why containment wins the trade (design
#: round 2, D9).
MIN_BODY = 8
#: Default for ``display.rail``. Lives beside the rail it governs so the
#: registry entry and the render edge can be pinned to ONE constant by test.
DEFAULT_RAIL = True


def rail_rows(text: Text) -> Text:
    """``text`` with the rail prepended to EVERY row, quote rows included.

    Mirrors :meth:`UserBlock._build`'s geometry: the gutter runs down wrapped
    continuations and the blank rows between paragraphs alike, because a marker
    on ONE row marks a line while a marker on every row marks a BLOCK — and the
    block is what a reader scrolling back is looking for. Skipping the blank
    rows breaks the rail into one segment per paragraph, which is the same
    "three separate things" failure wearing the new treatment.

    **A blockquote row carries BOTH marks**, and there is no exception for it:
    the rail in the block's own gutter at painted columns 0–1, Rich's bar in the
    quote's own column at painted column 2. They never touch, and they are
    distinguishable because ``U+258E`` and ``U+258C`` are different WIDTHS — a
    quarter block beside a half block reads as one rail with a quote nested
    inside it rather than as a double paint.

    Design round 1 ruled the opposite — reserve the rail's cells on a quote row
    and paint nothing into them — and that ruling was correct for the tree it
    was made in, where the rail and the bar were the SAME glyph (``U+258C``) in
    the same token and so read as a double-paint glitch. The isoluminance fix
    (design finding D2) moved the rail to ``U+258E`` and made the collision
    rationale stale; the reserved hole it left behind broke the column on
    exactly the rows a reader is scanning past, which is what the maintainer
    reported on PR #1229. The glyph difference IS the mechanism now, so the
    reservation is gone. Do not re-introduce it without first changing the
    glyphs back.

    The colour is resolved HERE, at paint time, and never cached on the
    instance: :meth:`AssistantBlock.retheme` re-enters ``_apply_rows``, so a
    theme change re-inks the rail for free — a ``Style`` held on the block is
    exactly what would break that.
    """
    glyph_style = Style(color=theme_mod.semantic_color(RAIL_TOKEN))
    gutter = RAIL + " " * max(RAIL_COLS - cell_len(RAIL), 0)
    railed = Text(end="", no_wrap=True, overflow="ellipsis")
    for index, row in enumerate(text.split("\n")):
        if index:
            railed.append("\n")
        railed.append(gutter, style=glyph_style)
        railed.append_text(row)
    return railed


def flatten(renderable: RenderableType, width: int, console: Console | None = None) -> Text:
    """A rich renderable's rendered rows, as ONE styled ``Text``.

    Why this exists at all: Textual decides per widget whether its content can
    be selected, and it decides it from the TYPE of the visual. ``visualize()``
    (``textual/visual.py``) promotes ``str`` and ``rich.text.Text`` to a
    ``Content``, and wraps every other rich renderable in a ``RichVisual``.
    ``Content`` applies ``options.selection`` while formatting and tags each
    segment with its content offset; ``RichVisual.render_strips`` ignores the
    selection argument entirely and tags nothing. A ``Markdown`` therefore
    cannot highlight and cannot be copied — ``Widget.get_selection`` bails on
    the first line, because the visual is not a ``Text`` or ``Content``.

    So the markdown is rendered ONCE, here, at a known width, and handed to
    Textual as the one renderable it treats as selectable. The rows are byte
    identical to what ``RichVisual`` painted before (both walk the same segment
    stream from the same console), so this buys selection without moving a
    single cell.

    Width is BAKED IN, which is the cost: the caller owns rebuilding on resize
    and pinning its height, the same bargain ``UserBlock`` and ``ToolCard``
    already make.

    ``console``: the app's own, so the brand markdown theme and the terminal's
    encoding are the ones in force. Detached (tests holding a block directly,
    a block built before mount) falls back to a private console carrying the
    same theme.
    """
    if console is None:
        console = Console(width=width, theme=brand_markdown_theme())
    options = console.options.update(width=width, height=None, highlight=False)
    text = Text(end="")
    # Cells emitted on the row currently being built. Rich pads a row that has
    # CONTENT out to the full width, but emits a row that has none as nothing
    # at all — so a blank line between two paragraphs was zero cells wide.
    # Selection paints the cells a row actually has, so a multi-paragraph
    # answer highlighted as a stack of disconnected slabs with unpainted gaps
    # between them, while `get_selection` returned one continuous string. The
    # highlight has to describe what gets copied; padding the blank rows is
    # what makes the band continuous.
    #
    # Safe for the clipboard: `TranscriptBlock.get_selection` drops each row's
    # trailing pad, which it already had to do for the content rows Rich pads.
    row_cells = 0
    for segment in console.render(renderable, options):
        if segment.control:
            continue
        for index, part in enumerate(segment.text.split("\n")):
            if index:
                if row_cells == 0:
                    text.append(" " * width)
                text.append("\n")
                row_cells = 0
            if part:
                text.append(part, segment.style)
                row_cells += cell_len(part)
    # Rich closes every block with a newline; kept, that is a blank row the
    # markdown never had, and one row of height the block would reserve and
    # never paint.
    while text.plain.endswith("\n"):
        text.right_crop(1)
    return text


def _scan_fences(
    lines: list[str],
    start_line: int,
    in_fence: bool,
    fence_marker: str,
    covered: set[int],
    end_line: int | None = None,
) -> tuple[bool, str]:
    """Advance fence state over ``lines[start_line:end_line]``, marking rows.

    Returns ``(in_fence, fence_marker)`` after the last scanned line. Lines
    inside a fence (including the marker rows themselves) land in
    ``covered``. Used incrementally so append-only updates scan only the new
    text (TUI-011a); the full re-scan path starts at ``start_line=0`` with
    fresh state. ``end_line`` bounds the scan to COMPLETED lines (their
    newline has arrived) so a fence marker split across two deltas is never
    toggled twice.
    """
    stop = len(lines) if end_line is None else end_line
    for i in range(start_line, stop):
        line = lines[i]
        if in_fence:
            covered.add(i)
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        marker_char = match.group(1)[0]
        marker_len = len(match.group(1))
        if not in_fence:
            in_fence = True
            fence_marker = marker_char * marker_len
            covered.add(i)
        elif marker_char == fence_marker[0] and marker_len >= len(fence_marker):
            # Closing fence: same character, at least the opener's length,
            # and nothing but marker characters on the line (commonmark).
            if set(line.strip()) <= {marker_char}:
                in_fence = False
                fence_marker = ""
    return in_fence, fence_marker


def _line_offsets(lines: list[str]) -> list[int]:
    """Char offset of each line start."""
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    return offsets


def _candidate_boundaries(
    lines: list[str], offsets: list[int], covered: set[int], text_len: int
) -> list[tuple[int, int]]:
    """``(boundary_offset, blank_line_index)`` for every settled blank line.

    A settled boundary is a blank line outside any fence whose NEXT line
    starts real content; end-of-text is deferred (more may arrive).
    """
    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if line.strip() != "" or i in covered:
            continue
        if i + 1 >= len(lines):
            continue  # trailing blank — defer, more may arrive
        if lines[i + 1].strip() == "":
            continue  # next line is blank too; keep scanning forward
        boundary = offsets[i + 1]
        if 0 < boundary < text_len:
            candidates.append((boundary, i))
    return candidates


def _last_preceding_list_item(lines: list[str], covered: set[int], before: int) -> bool:
    """True when the last non-blank line above ``before`` is list syntax."""
    for j in range(before - 1, -1, -1):
        if j in covered or lines[j].strip() == "":
            continue
        return _LIST_ITEM_RE.match(lines[j]) is not None
    return False


def find_stable_boundary(text: str) -> int:
    """Char offset of the last settled block boundary, or 0 if none.

    The returned offset is the start of the first content line after the
    last settled blank line, so ``text[:offset]`` is the freezable prefix
    and ``text[offset:]`` the live tail.

    Splice preconditions (TUI-010) REFUSE or DEFER a candidate:

    - refusal: the frozen prefix contains a reference-link definition line
      (``[label]: target``) — the tail may carry the referencing link, so
      splitting would render a dangling definition.
    - deferral: the last frozen block is a list item and the tail's next
      line continues list syntax — the boundary backs off one block.
    """
    if not text:
        return 0
    lines = text.split("\n")
    covered: set[int] = set()
    _scan_fences(lines, 0, False, "", covered)
    return _stable_boundary(text, lines, covered)


def _stable_boundary(text: str, lines: list[str], covered: set[int]) -> int:
    """Shared boundary resolution with TUI-010 preconditions applied.

    Walks settled candidates from the LAST blank line backward. A candidate
    is skipped only when the block immediately above the blank is a list
    item AND the tail starts with list syntax (freezing would split the list
    in two). Once the boundary sits after the list's closing blank the list
    is entirely inside the frozen prefix, so render(prefix)+render(tail) ==
    render(prefix+tail) holds and no further pinning is needed — a permanent
    "any list above pins the boundary" rule re-rendered the whole tail on
    every flush for any message opening with bullets.
    """
    offsets = _line_offsets(lines)
    candidates = _candidate_boundaries(lines, offsets, covered, len(text))
    if not candidates:
        return 0

    for boundary, blank_line in reversed(candidates):
        if _last_preceding_list_item(lines, covered, blank_line):
            if _LIST_ITEM_RE.match(text[boundary:]):
                continue  # tail continues the list: back off one block
        # Refusal: a reference-link definition in the frozen prefix can pair
        # with a link anywhere in the tail — never freeze across that.
        if _REF_DEF_RE.search(text[:boundary]):
            return 0
        return boundary
    return 0


class AssistantBlock(TranscriptBlock):
    """One streaming assistant message rendered as rich Markdown.

    Call :meth:`update_text` with the FULL accumulated text on each flush;
    the block re-renders only the volatile tail. Call :meth:`finalize_text`
    once at ``message_end`` to commit a single full render and freeze.

    The frozen renderable is kept together with the theme epoch it was built
    under (TUI-016): when the epoch changes, the cache is dropped so the
    next update re-renders against the new ramp.

    The Markdown is FLATTENED to a ``Text`` before it is applied (see
    :func:`flatten`), because that is what makes agent prose selectable and
    copyable — reported from the field as "I can't seem to highlight the agent
    messages which is important to be able to copy/paste agent content". The
    rows are unchanged; only the type Textual sees is. Two consequences the
    block now owns:

    * **Width is baked in**, so :meth:`on_resize` rebuilds — the discipline
      ``UserBlock``, ``NoticeBlock`` and ``ToolCard`` already follow.
    * **Height is pinned** to the row count, for the reason
      ``TranscriptBlock.invalidate_row_measurements`` records: a block that
      authors its own rows must not be MEASURED, because the measurement is
      cached on width alone and the first one is taken of the fallback build.

    The frozen prefix is cached as its FLATTENED text, not just as a
    ``Markdown``, and the flush concatenates. That is exact rather than
    approximate: rich ends every renderable with a newline, so flattening
    ``Group(prefix, tail)`` and joining ``flatten(prefix)`` to
    ``flatten(tail)`` with one newline produce the same rows (asserted in
    ``test_transcript_selection.py``). It also makes streaming CHEAPER than
    before — the prefix's markdown was re-rendered by the compositor on every
    repaint, and now it is rendered once per settled block.
    """

    SPACING_KIND = "assistant"

    def __init__(self) -> None:
        super().__init__()
        self.add_class("assistant-block")
        self._full_text: str = ""
        #: Whether the message ENDED EARLY — the turn was aborted or the
        #: provider stopped mid-sentence — as opposed to running to a normal
        #: stop. Distinct from ``_finalized``, which is the FINALIZED-BLOCK
        #: protocol's "this block is immutable" and says nothing about whether
        #: the model finished talking: the abort path finalizes a truncated
        #: message exactly as the clean path finalizes a whole one, so the two
        #: questions have to be asked separately. Consumers that reproduce the
        #: message elsewhere (``/copy``) need "is it COMPLETE", not "is it
        #: frozen".
        self._truncated: bool = False
        self._frozen_text: str = ""
        self._frozen_rendered: Markdown | None = None
        #: The frozen prefix's FLATTENED rows, and the width they were built
        #: at. Both, because the flatten bakes the width in: a cached prefix
        #: from a 120-column frame is wrong rows at 60, and the epoch check
        #: alone would never notice.
        self._frozen_flat: Text | None = None
        self._frozen_width: int = -1
        self._frozen_epoch: int = -1
        # Incremental fence tracking (TUI-011a): state as of the last scan.
        self._scanned_len: int = 0
        self._scanned_lines: int = 0  # completed lines already fence-scanned
        self._in_fence: bool = False
        self._fence_marker: str = ""
        self._covered: set[int] = set()
        #: The row count last written to ``styles.height``. ``_apply_rows``
        #: compares against it to decide whether the content update needs a
        #: LAYOUT pass or only a repaint; -1 means nothing is pinned yet, so
        #: the first apply always lays out.
        self._pinned_rows: int = -1
        #: The width the applied rows were flattened at. ``on_resize`` compares
        #: against it so a height-only resize — which every height pin raises
        #: — does not re-flatten a message to reproduce identical rows.
        self._built_width: int = -1
        #: The gutter width the applied rows were actually PAINTED with, in the
        #: ``_built_width`` mould: what the frame on screen is, not what the
        #: setting says now. The copy path measures against this rather than
        #: re-reading ``display.rail``, because the two reads happen at
        #: different times and nothing forces them to agree — see
        #: :meth:`copy_gutter` (review round 2, M1).
        #:
        #: ``-1`` means "never painted", which the copy path resolves live: a
        #: block with no rows has no frame to be faithful to, and answering
        #: from the setting is the only defined answer there.
        self._painted_rail_cols: int = -1

    def update_text(self, text: str) -> None:
        """Apply ``text`` as the accumulated message content.

        Equality guard first: identical text is a no-op (providers re-emit on
        no-delta ticks). Otherwise re-render only the tail after the last
        settled blank-line block boundary — append-only updates scan only
        the NEW text for fence markers and never re-lex the frozen prefix.
        """
        if self._finalized:
            return
        if text == self._full_text:
            return  # equality guard — no work for identical re-emits

        # Theme epoch changed since the freeze: the cached renderable was
        # built under another ramp — drop it (TUI-016).
        epoch = theme_mod.get_theme_epoch()
        if self._frozen_rendered is not None and epoch != self._frozen_epoch:
            self._frozen_text = ""
            self._frozen_rendered = None
            self._frozen_flat = None

        append_only = text.startswith(self._full_text) and self._scanned_len <= len(text)
        self._track_fences(text, append_only)
        self._full_text = text

        lines = text.split("\n")
        boundary = _stable_boundary(text, lines, self._covered)
        prefix = text[:boundary] if boundary > 0 else ""
        if prefix != self._frozen_text:
            # The prefix moved (grew, or was dropped by an epoch change): the
            # cached flatten is stale. `_flat_rows` re-renders it once.
            self._frozen_text = prefix
            self._frozen_flat = None
            self._frozen_epoch = epoch
        self._apply_rows(self._flat_rows(self._flat_width()))

    def _rail_cols(self) -> int:
        """:data:`RAIL_COLS` when the rail is on, 0 when it is off.

        The block-level gutter width, read at the same rate as the paint in
        :meth:`_apply_rows` so a flip cannot leave the fold and the paint
        disagreeing about how many cells the gutter owns.

        This is what makes OFF mean pre-rail rendering rather than pre-rail
        rendering minus two columns: an ungated subtraction folds the prose two
        cells narrower than the box while nothing is painted in the space it
        left, which is an indent the reader did not ask for and cannot explain.

        Read per call rather than cached on the instance, the same argument
        :func:`rail_rows` makes about resolving colour at paint time — it is
        what makes a mid-session flip apply to mounted blocks for free.
        """
        return RAIL_COLS if settings_get("display.rail", DEFAULT_RAIL) else 0

    def _body_width(self, lane: int) -> int:
        """``lane`` less the gutter, floored at :data:`MIN_BODY` — but never
        wider than the lane can actually hold.

        Two rules that can disagree, resolved here once so every caller gets the
        same answer. The floor keeps prose legible; the CLAMP keeps the painted
        row inside the block's own region, and the clamp wins.

        The floor alone breaks containment at narrow widths, because it floors
        the TEXT LANE while the row that gets painted is ``rail_cols + lane``.
        Measured before the clamp, with the rail on: a 9-column terminal gives
        this block a 5-column region, the lane floors at 8, and the row paints
        10 cells — five cells outside its own container, over the scrollbar. It
        overflowed at every terminal width up to 13 with the rail on and up to
        11 with it off, where the pre-rail build (which had no floor at all)
        always fit.

        So where the lane cannot afford the floor, the floor gives way: fold to
        what is actually there. That yields text which is nearly unreadable at a
        2-cell body, and that is the deliberate trade — unreadable text inside
        the block is recoverable by widening the terminal, whereas a row painted
        outside its region corrupts the frame around it and the reader cannot
        tell which widget is lying. Applied in BOTH rail states rather than by
        special-casing the rail off: containment is not a property one state is
        allowed to skip.

        Not solved by dropping the rail under a width threshold: a rail that
        vanishes at some width is a second behaviour to explain, and the block
        would still have to decide what to do at the width below that one
        (design round 2, D9).
        """
        body = lane - self._rail_cols()
        if body < MIN_BODY:
            # The floor would widen the fold past the lane and paint outside the
            # block. Never negative, and 0 is a real answer rather than a
            # degenerate one: at a 2-cell region the gutter is the whole lane,
            # and a 0-cell body paints the rail alone and stays contained
            # (measured). Asking for 1 there puts a cell back outside the block.
            return max(body, 0)
        return body

    def _flat_width(self) -> int:
        """The width the rows are built at — the block's own once laid out.

        Before layout it is the width this block is ABOUT to be given, which
        :meth:`TranscriptBlock.fold_width` derives from the container it was
        appended into. Going straight to :data:`FALLBACK_WIDTH` here is what
        made every mount-then-stream path fold at 80 columns, pin that fold as
        the block's height, and re-fold one frame later when ``on_resize``
        landed: at 140 columns the block measurably built at 80 and settled at
        134, which a reader sees as the message flashing narrow. The pin is
        untouched — ``_apply_rows`` still pins the count of whatever rows it is
        handed — so the invariant that a self-authoring block is never MEASURED
        holds exactly as before; only the width those rows are folded at is
        better informed.

        The rail's cells come off the top: the markdown is folded into the BODY
        the gutter leaves, not into the whole lane. Folding at the full lane and
        then pushing two cells onto every row overhangs the block by two on
        every wrapped row. :meth:`_body_width` applies the floor and the
        containment clamp together, so this and :meth:`authored_width` cannot
        disagree about how narrow is too narrow.
        """
        return self._body_width(self.fold_width(FALLBACK_WIDTH))

    def authored_width(self, lane: int) -> int:
        """The lane, less the rail — this block's box is not the whole lane.

        The container's lane walk (``TranscriptView._refit_authored_blocks``)
        asks each authored block for its rebuild width rather than handing it
        the lane, exactly so a block whose box is narrower than the lane can say
        so. Without this override the walk re-folds the prose two cells wider
        than the box the rail leaves it, and the two rebuild triggers — the lane
        walk and this block's own ``on_resize`` — would name DIFFERENT widths
        for one lane change, which is what turns ``refit_width``'s equality
        guard into a rebuild loop rather than a no-op.

        Same subtraction and same floor as :meth:`_flat_width`, through the same
        :meth:`_body_width`, so whichever trigger fires first the rows are folded
        at one number.
        """
        return self._body_width(super().authored_width(lane))

    def _flat_console(self) -> Console | None:
        """The app's console, or ``None`` when this block is detached.

        Detached is not exotic: a block is constructed and given its first
        text before it is mounted, and the unit tests hold blocks with no app
        at all. :func:`flatten` falls back to a private console carrying the
        same markdown theme, so the rows are the same either way.
        """
        try:
            return self.app.console
        except Exception:
            return None

    def _apply_rows(self, text: Text) -> None:
        """Apply flattened rows and PIN the height to the count of them.

        Pinned for the reason ``UserBlock._build`` records at length: a block
        that authors its own rows must not be MEASURED, because Textual caches
        the measurement on ``_content_height_cache`` keyed on the WIDTH ALONE
        and the first measurement is taken of the fallback-width build. Under
        ``height: auto`` (which the sheet still gives this block, and which is
        now only the pre-first-content default) a message built at
        :data:`FALLBACK_WIDTH` and then laid out narrower reserves the inflated
        count forever and paints a hole under itself.

        The height pin is also what makes the LAYOUT pass skippable. Textual's
        ``Static.update`` reflows by default, and a reflow re-arranges every
        widget in the transcript — measured at 7.8 ms across 173 widgets on a
        161-block screen. A pinned block's footprint is exactly its pin, so a
        delta that lands inside the same number of rows changes nothing the
        container has to re-place and needs a repaint only. Deltas arrive at
        30 Hz and most of them add a few characters to a line that already
        exists, so this is the common case, not the rare one: it took the cost
        of a streaming delta from 4.54 ms to 1.98 ms at the median and from
        56.4 ms to 11.4 ms at the worst.

        **The rail is painted HERE and nowhere else**, and that placement is
        load-bearing rather than convenient. This method is the single choke
        point every row-producing path funnels through — :meth:`update_text`,
        :meth:`refit_width` and :meth:`retheme` all end in it — so the gutter is
        applied once per assembled frame. Applying it inside :meth:`_flat_rows`
        instead would bake it into ``_frozen_flat``, which that method caches
        and CONCATENATES on every delta: the cached prefix would arrive already
        railed and be railed again, growing by two cells per flush. Painting the
        assembled output leaves the cache holding pure prose, so double-painting
        is impossible by construction rather than by a guard.
        """
        self._built_width = self._flat_width()
        # ONE read of the flag per paint, recorded beside the width it was
        # painted at. Every later question about THIS frame — what the copy
        # must strip, where a selection column starts — is answered from the
        # record rather than by reading the setting again, so the frame and the
        # clipboard cannot disagree even when the setting changes underneath a
        # block that has not repainted (review round 2, M1). Recorded here
        # because this method is the single funnel every row-producing path
        # ends in, so one assignment covers update_text, refit_width, retheme
        # and finalize_text alike.
        rail_cols = self._rail_cols()
        self._painted_rail_cols = rail_cols
        if rail_cols:
            text = rail_rows(text)
        # Counted from the RAILED text, because that is what gets painted. The
        # gutter adds no rows, but the pin must describe the frame it reserves
        # space for rather than the one before the gutter went on.
        rows = text.plain.count("\n") + 1
        moved = rows != self._pinned_rows
        self._pinned_rows = rows
        self.styles.height = rows
        self.set_content(text, layout=moved)

    def _flat_rows(self, width: int) -> Text:
        """The block's rows at ``width``, from the state the last update left.

        The frozen prefix is cached as FLATTENED TEXT and this concatenates —
        exact rather than approximate, because rich ends every renderable with
        a newline, so ``flatten(Group(prefix, tail))`` and
        ``flatten(prefix) + "\\n" + flatten(tail)`` produce identical rows
        (pinned in ``test_transcript_selection.py``). It also makes streaming
        CHEAPER than the ``Group`` it replaces: that re-rendered the prefix's
        markdown on every repaint, and this renders it once per settled block.
        """
        if not self._full_text:
            return Text("")
        if not self._frozen_text:
            # The threaded width goes down with it: `_flat_whole` folds the whole
            # message, and a rebuild has to author at the width its guard
            # compared against rather than at a second derivation of it.
            return self._flat_whole(width)
        console = self._flat_console()
        if self._frozen_flat is None or self._frozen_width != width:
            self._frozen_rendered = Markdown(self._frozen_text)
            self._frozen_flat = flatten(self._frozen_rendered, width, console)
            self._frozen_width = width
        tail = self._full_text[len(self._frozen_text) :]
        rows = self._frozen_flat.copy()
        rows.append("\n")
        rows.append_text(flatten(Markdown(tail), width, console))
        return rows

    def on_resize(self, event: object) -> None:
        """Rebuild at the new width — the flatten baked the old one in.

        The same discipline ``UserBlock`` and ``NoticeBlock`` already follow,
        and it arrives with the flatten rather than being a new cost: a
        ``Markdown`` re-folded itself per repaint, so nothing had to be told
        the width had changed. A ``Text`` carries the fold it was built with.

        Finalized blocks rebuild too. The FINALIZED-BLOCK protocol promises
        the container that committed ROWS never change under scroll, and at a
        new width they are a different set of rows whatever this does; the
        alternative is a settled message that keeps a stale fold and either
        clips or leaves a hole.

        Guarded on the WIDTH, because a resize is not evidence that the rows
        moved. The rows are a pure function of the text and the width, and
        ``_apply_rows`` pins the height — so every height pin raises a Resize
        of its own and this handler re-ran the whole flatten to reproduce the
        rows it had just been given. Measured on a session replay: 122 rebuilds
        for 75 blocks, ~175 ms, all of the excess for a width that never
        changed.
        """
        self.refit_width(self._flat_width())

    def refit_width(self, width: int) -> None:
        """Re-flatten when ``width`` is not the width these rows were built at.

        The container's lane walk (:meth:`TranscriptView._refit_authored_blocks`)
        reaches this the same way ``on_resize`` does, because this block is the
        one that made the tear general: the rows are a flattened ``Text``, so
        the fold is baked in rather than re-derived per repaint, and the only
        thing that re-fits it is the ``Resize`` the compositor can drop. A
        settled agent reply is the most-common wrapping block in the transcript,
        so without this the lane fix would repair the tool rows above a reply
        and leave the reply itself stopped short of them — the same one-viewport
        two-right-edges frame, on the prose.

        ``width`` is used as the REBUILD width rather than only as the trigger,
        so a rebuild cannot land on a third number: rows are a pure function of
        the text and the width, which is also what makes the ``_built_width``
        equality a sound guard.
        """
        if not self._full_text:
            return
        if width == self._built_width:
            return
        was_finalized = self._finalized
        self._finalized = False
        try:
            rows = self._flat_whole(width) if was_finalized else self._flat_rows(width)
            self._apply_rows(rows)
        finally:
            self._finalized = was_finalized

    @property
    def frozen_renderable(self) -> Markdown | None:
        """The cached frozen-prefix render (theme-epoch tracked, TUI-016)."""
        return self._frozen_rendered

    def _track_fences(self, text: str, append_only: bool) -> None:
        """Incrementally update fence coverage for ``text`` (TUI-011a).

        Append-only updates scan ONLY the new suffix (carrying the running
        ``in_fence`` state); anything else re-scans from the top so the
        coverage stays authoritative.

        The incremental resume rewinds to the start of the line containing
        ``_scanned_len`` and carries the fence state from BEFORE that line
        was first scanned: resuming at the line with the state it produced
        double-toggles a closing fence whose newline arrives in the next
        delta, pinning ``in_fence`` True forever (the frozen prefix then
        never advances and every flush re-parses the whole message).
        """
        lines = text.split("\n")
        # Only lines whose newline has arrived are scanned: a fence marker
        # split across two deltas must not toggle until it is complete, and
        # a boundary needs a blank line, which needs its newline. Resuming
        # at the first never-completed line with the carried state keeps the
        # scan O(new text) and the state authoritative.
        completed = max(0, len(lines) - 1)  # the last element has no newline
        if append_only and self._scanned_lines > 0:
            self._in_fence, self._fence_marker = _scan_fences(
                lines,
                self._scanned_lines,
                self._in_fence,
                self._fence_marker,
                self._covered,
                completed,
            )
        else:
            self._covered = set()
            self._in_fence, self._fence_marker = _scan_fences(
                lines, 0, False, "", self._covered, completed
            )
        self._scanned_lines = completed
        self._scanned_len = len(text)

    @property
    def in_fence(self) -> bool:
        """Whether the streamed text currently sits inside a code fence."""
        return self._in_fence

    def finalize_text(self) -> None:
        """Commit the full text as one render and freeze the block.

        One render of the WHOLE message, not the concatenation: the splice was
        only ever a streaming economy, and a settled message is re-lexed once.
        """
        if self._finalized:
            return
        self._full_text = self._full_text or ""
        self._apply_rows(self._flat_whole())
        self.finalize()

    def mark_truncated(self) -> None:
        """This message ended early — aborted, or cut off by the provider.

        Named and shaped after ``ToolCard.mark_interrupted``, which answers the
        same question for the other kind of live row: the turn ended before the
        thing finished, and the record has to say so rather than looking like a
        completed one. One vocabulary for "ended early" across both block types.

        The TEXT is deliberately untouched. What streamed is what the user
        read, and rewriting or annotating it here would put words in the
        model's row; the flag is metadata for consumers that reproduce the
        message somewhere the frame's context is missing (``/copy``), where a
        half sentence is indistinguishable from a short complete answer.
        """
        self._truncated = True

    def is_truncated(self) -> bool:
        """Whether this message ended early (see :meth:`mark_truncated`)."""
        return self._truncated

    def _flat_whole(self, width: int | None = None) -> Text:
        """The whole message as one flatten, at ``width`` or the block's own.

        ``width`` is a lane a caller has already published (:meth:`refit_width`)
        and is used verbatim when given, for the reason that method records;
        ``None`` keeps the ladder, which is what the streaming and retheme paths
        want.
        """
        if not self._full_text:
            return Text("")
        lane = width if width is not None else self._flat_width()
        return flatten(Markdown(self._full_text), lane, self._flat_console())

    def retheme(self) -> None:
        """Re-flatten in the new ramp, dropping every theme-baked cache.

        The flatten renders through the app console, whose markdown theme the
        switch has already re-pushed — but the frozen prefix is cached as
        FLATTENED TEXT with the old ramp's styles baked into every span, so
        the caches go first (the same invalidation ``update_text`` performs
        when it notices an epoch change) and the rebuild re-lexes from source.
        """
        if not self._full_text:
            return
        self._frozen_rendered = None
        self._frozen_flat = None
        self._frozen_epoch = theme_mod.get_theme_epoch()
        was_finalized = self._finalized
        self._finalized = False
        try:
            rows = self._flat_whole() if was_finalized else self._flat_rows(self._flat_width())
            self._apply_rows(rows)
        finally:
            self._finalized = was_finalized

    def settled_rows(self) -> int:
        """Rows provably stable now: the frozen prefix's render while live."""
        if self._finalized:
            return super().settled_rows()
        # While streaming, only the frozen prefix is byte-stable — and it is
        # already flattened, so the count is COUNTED rather than re-measured
        # through rich. Exact, and it is the same number the compositor will
        # paint, which a second render at a guessed width was not.
        if self._frozen_flat is not None:
            return self._frozen_flat.plain.count("\n") + 1
        return 0

    def spans_multiple_rows(self) -> bool:
        """Answered from the source text, never by rendering the Markdown.

        Spacing only needs "one row or more"; a full render of a message
        that may be thousands of lines long to learn that is waste. Any
        embedded newline settles it; otherwise the single line is multi-row
        exactly when it is wider than the block — measured at the LADDER's
        width, so a block whose destination is already named is judged at it
        rather than at the 80-column fallback
        (`TranscriptBlock.spans_multiple_rows` records why that matters to the
        gaps an insert settles before its mount).
        """
        text = self._full_text.strip()
        if not text:
            return False
        if "\n" in text:
            return True
        return cell_len(text) > max(self.fold_width(80), 10)

    def text(self) -> str:
        """The accumulated message text (for tests and export)."""
        return self._full_text

    def copy_gutter(self, index: int) -> int:
        """The rail's columns, on every row — the BLOCK-level constant.

        :func:`rail_rows` prefixes the same two cells to every row this block
        paints, blank paragraph rows included, so the count is uniform and needs
        no row bookkeeping — the same argument ``UserBlock.copy_gutter`` makes
        for its prompt rule.

        This is only half the answer, and the smaller half.
        :meth:`_furniture_width` is the PER-ROW one on top of it: a quote row
        also carries a painted ``▌`` that is not this gutter, a list row carries
        a ``•``. Returning the right number here while ``get_selection`` still
        aligns against railed rows is exactly the shape of the bug this block's
        copy path was fixed for — the constant is necessary and nowhere near
        sufficient.

        Zero when the rail was off when these rows were painted: there is no
        gutter to strip, and reporting two would take two cells of real content
        off the clipboard.

        **Answered from what was PAINTED, not from what the setting says now**
        (review round 2, M1). The paint and the copy happen at different times,
        and nothing forces the flag to hold still between them: an external
        write — another pane, ``lop config edit`` — drops the settings cache
        without repainting anything, and the in-app repaint sweep can skip a
        block too (a swallowed per-block ``retheme``, or the offscreen skip).
        Re-reading the flag here then measures the CURRENT setting against rows
        painted under the OLD one, and the reader silently gets the wrong
        document: measured at 60 columns, painted-on/copied-off put the rail
        itself on the clipboard, and painted-off/copied-on ate the first two
        characters of every row and lost row 0 entirely. Reading the record
        makes the frame and the clipboard unable to disagree by construction,
        which is the argument this file makes everywhere else.

        Falls back to the live answer only when this block has NEVER painted
        (``-1``). There is no frame to be faithful to in that case, and the
        empty-message path returns before any of this anyway.
        """
        if self._painted_rail_cols < 0:
            return self._rail_cols()
        return self._painted_rail_cols

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """The selected text as MARKDOWN, so it pastes cleanly anywhere.

        The base :meth:`TranscriptBlock.get_selection` copies the rendered
        frame, which is the right rule for the transcript's plain-text blocks
        but wrong for a markdown message: the frame turns a blockquote into a
        ``▌`` bar, a bullet into ``•``, a table into box-drawing and a heading
        into bare bold text. Pasted into a messenger or an email that is
        furniture, not content — the ``▌`` welded to every quoted line is the
        report this method answers. The block already holds the message's
        source, so the clipboard carries that instead, mapped to the rows the
        reader actually highlighted.

        The selection and the frame stay the same computation: the highlighted
        rows come from ``Selection.get_span`` exactly as the base method reads
        them, and those row indices are aligned back to source lines by
        :func:`_copy_markdown.align`. A partial selection is sliced out of the
        source and re-fenced / re-quoted so it is valid markdown on its own.
        When the source cannot be aligned (an empty message, or a frame the
        walker cannot place), the method falls back to the base frame copy so a
        copy never comes back empty-handed.

        **The one selection markdown cannot answer: a SUB-LINE take.** Reported
        from the field — dragging the eight cells of one word out of a bullet
        announced ``copied 115 characters`` and put the whole source line on the
        clipboard. The cause was structural rather than an off-by-one: the row
        walk kept only ``first_row``/``last_row`` and dropped the column pair
        ``get_span`` returns, and ``slice_markdown`` is row-granular by
        contract, so every partial row was widened to the source line under it.

        Column-trimmed markdown source is not the missing third option — it is
        **impossible**, not merely unimplemented, and that is what decides the
        rule. A rendered column does not index a source column: measured on the
        reported bullet, ``frontend`` sits at rendered column 57 and source
        column 58, because ``- `` paints as `` • `` (+1) and the ``**`` around
        the word vanishes (-2). The offset is content-dependent AND signed, and
        :func:`_copy_markdown.align` deliberately maps rows to source *lines*,
        never claiming a column correspondence. So for a partial row there are
        exactly two truthful answers — the glyphs that were highlighted, or a
        whole line the reader did not select — and the second is the bug.

        Hence the boundary, drawn on SOURCE LINES rather than on rendered rows:
        a selection that does not cover the full content of the rows it touches
        **and** touches at most one source line copies the highlighted glyphs,
        per :meth:`TranscriptBlock.copy_gutter`'s rule. Everything wider stays
        markdown. Counting source lines rather than rows is also what covers a
        phrase dragged across a wrapped paragraph's fold — one source line
        painted as several rows, the same defect, and measured at width 60 a
        row-count gate still copied all 128 characters of it.

        **The glyph path strips painted furniture per row**, via
        :func:`_copy_markdown.furniture_width` rather than :meth:`copy_gutter`.
        The inherited gutter is 0 and has to be: an assistant message has no
        fixed one, because whether a leading ``▌`` is decoration or content
        depends on the construct the row belongs to, and only the alignment
        knows. Clamping to 0 stripped nothing, so a sub-line drag across a
        wrapped quote's fold pasted the ``▌`` and a column-0 drag pasted the
        ``•`` — furniture the markdown path never emitted, found independently
        by review round 1 (R1-1) and design round 1 (D1). It also meant one
        cell of difference in where a drag STARTED silently switched the paste
        between markdown and rendered glyphs, which is why the full-coverage
        predicate above measures against the same painted width.

        **Rows of one source line rejoin with what the terminal consumed**, not
        with a newline: the gate guarantees they share a source line, so each
        break between them is a soft wrap at the current width rather than a
        character in the document (design round 1, D2). A space is not always
        right — a token wider than the render segment is folded mid-token and
        nothing is consumed — so each fold's separator is decided by walking
        the row against its source line, :func:`_copy_markdown.wrap_separators`.
        The decision is POSITIONAL: an earlier version asked only whether the
        two rows' adjoining tokens appeared welded ANYWHERE in the line, which
        destroyed a real word boundary on any line using both ``file system``
        and ``filesystem`` (review round 2, R2-1; design round 2, D2-2).

        **The accepted cost**, stated so it is chosen rather than rediscovered:
        a drag from the middle of one bullet to the middle of the next copies
        both bullets WHOLE. The reader gets more than they highlighted. That is
        deliberate — trimming those ends needs the column mapping that does not
        exist, and the markdown path is the only one that can state a multi-line
        take as valid markdown at all. Reviewed and accepted as shippable in
        design round 1 (D3). A sub-line take also loses inline markers: dragging
        one bold word yields ``frontend``, not ``**frontend**``. That is the
        base rule — the clipboard is what the highlight covered — and it was
        accepted in design round 1 (D4) for the reason that the reader pointed
        at a frame with no asterisks anywhere on it.

        A sub-line take across a TABLE row is the same rule and the one place
        it costs something real: the rendered row has no ``|``, so the paste is
        ``alpha  0.91`` and no longer reads as a table row (design round 1, D5).
        It is left as the rule rather than special-cased because the whole-row
        gesture — the one a reader makes to take a row AS a row — still covers
        the row and still copies ``| alpha | 0.91 |`` from the source.

        **A drag that lies entirely in a row's trailing pad copies nothing**,
        deliberately: it selects no glyph, and Rich's pad is not content the
        reader can see. ``_put_on_clipboard`` drops the empty payload, so there
        is no write and no receipt — the same answer a zero-width click gets
        (review round 1, R1-2).
        """
        visual = self._render()
        if not isinstance(visual, Content):
            return None
        if not self._full_text.strip():
            return super().get_selection(selection)
        rows = visual.plain.split("\n")
        # ONE COORDINATE CONVENTION, and everything below depends on it: the
        # alignment, the furniture measurement and every BLANKNESS test work in
        # BARE columns — the row as Rich folded it, before :func:`rail_rows`
        # prefixed this block's gutter. :data:`RAIL_COLS` is added back only
        # where a column is compared against a SELECTION span, because the
        # reader drags over the painted frame and their columns include the
        # rail. The private helpers below are handed ``bare`` for the same
        # reason and document that they expect it.
        #
        # The rail cannot simply be left on, for TWO independent reasons.
        #
        # 1. ``▌`` is not a neutral glyph here: :func:`_copy_markdown.align`
        #    reads it as the BLOCKQUOTE bar, so a rail on every row makes every
        #    row look like a quote row. Measured on a paragraph + bullets +
        #    blockquote message, railed rows align to ``[0, 0, 2, 3, 3, 5]``
        #    where bare rows align to ``[0, 1, 2, 3, 4, 5]``, and
        #    ``furniture_width`` drops from 3 to 0 on the list rows — a copy
        #    then pastes the bullet glyph AND lands on the wrong source line.
        # 2. The rail makes no row BLANK any more. A separator row paints as
        #    ``▌`` plus pad, which is truthy under ``.strip()``, so every
        #    blankness predicate in this file inverts: blank rows would enter
        #    ``content``, a railed blank would ``rstrip()`` to length 1 instead
        #    of 0, and :meth:`_furniture_width`'s previous-painted-row scan —
        #    which exists precisely so a separator row does not make the row
        #    after it read as a list opener — would never skip anything,
        #    flipping ``opens_line`` after every paragraph break and
        #    re-introducing the issue #395 mismeasurement class.
        #
        # Bound ONCE here and used for every compensation below, because with
        # ``display.rail`` off no gutter is painted: slicing a constant two
        # would eat the first two characters of real content and every
        # ``+ RAIL_COLS`` comparison would be off by the same two.
        #
        # Taken from :meth:`copy_gutter`, which answers from what was PAINTED
        # rather than from the setting's current value. ``rows`` above came out
        # of the rendered frame, so the number that de-rails them has to be the
        # number that railed them — re-reading the flag here is how a copy ends
        # up measuring this frame against a setting that changed after it was
        # painted (review round 2, M1).
        rail_cols = self.copy_gutter(0)
        bare = [row[rail_cols:] for row in rows]
        mapping = _copy_markdown.align(self._full_text, bare)

        # The same ``Selection.get_span`` the band paints with, chrome rows
        # dropped, so the clipboard and the highlight cannot disagree.
        selected: list[tuple[int, tuple[int, int]]] = []
        for index in range(len(rows)):
            span = selection.get_span(index)
            if span is not None and not self.copy_row_is_chrome(index):
                selected.append((index, span))
        if not selected:
            return None

        # Blank rows carry no glyphs, so they neither prove nor disprove a
        # sub-line take: a whole-message copy legitimately spans the blank
        # separator rows between paragraphs, and letting one veto the markdown
        # path would degrade every multi-paragraph copy to rendered text.
        # Blankness judged on the BARE row (reason 2 above): every painted row
        # opens with the gutter, so ``rows[i].strip()`` is true even of a blank
        # paragraph row and this filter would stop filtering anything.
        content = [(i, span) for i, span in selected if bare[i].strip()]

        sub_line = False
        if content:
            # Source LINES, not rows: a paragraph that wraps is one line painted
            # as several, and a phrase dragged across that fold is as much a
            # sub-line take as one inside a single row.
            #
            # An UNPLACED row (``align`` returned ``None``) is absence of
            # evidence, not agreement. Dropping it from the set let a selection
            # spanning an unplaced row and a placed one collapse to one element
            # and read as a single-source-line take it never was: two lines of a
            # fenced block were then rejoined into ONE RUNNABLE COMMAND (review
            # round 3, R3-2; design round 3, D3-2). It now disqualifies the gate,
            # sending the take down the markdown path, which states a multi-line
            # selection as multiple lines.
            #
            # MIXED evidence is what disqualifies: when some rows are placed and
            # others are not, nothing can say whether the unplaced ones belong to
            # the placed one's line, so a single-element set is not agreement.
            # When NO row is placed the situation is different in kind rather
            # than in degree -- there is no line-boundary evidence at all, which
            # is ordinary for a CJK paragraph where no anchor word survives, and
            # the markdown path has nothing to offer either. That case keeps the
            # glyph answer (review round 2, R2-2).
            row_sources = [mapping[i] if i < len(mapping) else None for i, _ in content]
            sources = {source for source in row_sources if source is not None}
            mixed = bool(sources) and None in row_sources
            if not mixed and len(sources) <= 1:
                first_index, (first_start, _) = content[0]
                last_index, (_, last_end) = content[-1]
                # ``rstrip()`` is the only honest measure of the row's actual
                # content here. Rich pads each row out to its RENDER SEGMENT's
                # width, which is not the block's — measured, prose rows pad to
                # 76 while table rows in the same message pad to 14 — so any
                # predicate against the block width or raw ``len(row)`` is
                # wrong. ``end`` also arrives three ways: -1 for end-of-row, a
                # column inside the pad when the drag overran the last glyph, or
                # a column short of it. Only the glyph count settles all three.
                # The gutter is the row's PAINTED furniture, not the block's
                # ``copy_gutter`` of 0: a full-content take must read as full
                # whether or not the reader's drag began on the ``▌`` cell, or
                # the same gesture one cell left would fall to the glyph path
                # and paste a different document (design round 1, D1).
                # Converted INTO selection columns: the furniture and the row
                # content are measured bare, the drag is reported painted, and
                # the rail is exactly the difference. ``rstrip`` on the bare row
                # rather than the painted one for reason 2 — a railed blank row
                # rstrips to ``▌``, a length of 1 where the honest answer is 0.
                starts_full = first_start <= rail_cols + self._furniture_width(
                    bare, mapping, first_index
                )
                ends_full = last_end == -1 or last_end >= rail_cols + len(bare[last_index].rstrip())
                sub_line = not (starts_full and ends_full)

        if sub_line:
            # Sliced as ``TranscriptBlock.get_selection`` slices it (``-1``
            # meaning end of row), but over the content rows only — a blank row
            # caught at the edge of the drag would contribute a line break the
            # reader never highlighted — and clamped past each row's PAINTED
            # furniture rather than past ``copy_gutter``.
            #
            # ``copy_gutter`` is the rail and NOTHING MORE. It used to be 0
            # here, and the note that it "cannot be anything else" was true
            # while the block painted no gutter of its own; the rail made it a
            # fixed two cells, but only those two. What is furniture BEYOND the
            # rail still depends on the construct the row belongs to, so the
            # clamp remains per-row: clamping to the block constant alone
            # stripped nothing beyond the gutter, and a sub-line take across a
            # wrapped quote's fold put the ``▌`` on the clipboard while a
            # column-0 drag picked up the ``•`` — furniture the base commit
            # never copied, and the exact leak this method's docstring claims to
            # prevent (review round 1, R1-1; design round 1, D1).
            # Sliced out of the PAINTED rows, because ``start`` and ``end`` are
            # painted columns — so the clamp is the rail plus the construct's
            # own furniture, which is what keeps the gutter off the clipboard
            # when the reader's drag began on it.
            glyphs = [
                rows[index][
                    max(start, rail_cols + self._furniture_width(bare, mapping, index)) : (
                        None if end == -1 else end
                    )
                ]
                for index, (start, end) in content
            ]
            trimmed = [row.rstrip() for row in glyphs]

            # Rejoined with what the TERMINAL consumed at each fold, not with a
            # newline. The gate above guarantees these rows share one source
            # line, so every break between them is a SOFT WRAP — an artifact of
            # the current width, not a character in the document. Pasting it
            # sent a phrase to Slack as two lines and turned the receipt into
            # ``copied 2 lines`` for part of one sentence, the mirror of the
            # composer bug ``_put_on_clipboard`` already fixed (design round 1,
            # D2). Once rejoined the receipt falls into the character branch by
            # itself, so the unit needs no separate fix.
            # The separators are decided from EVERY row of the source line, not
            # from the rows the reader's drag happened to touch. The walk is
            # end-anchored: it asks the rows to consume the line's whole visible
            # content, so handing it a selection-shaped subset violates its
            # precondition and it cannot place them. That is what a drag stopping
            # one row short of a paragraph's end used to do, and the guess it
            # fell to put a space through a URL and a filesystem path (review
            # round 3, R3-1; design round 3, D3-1). Supplying the full row set
            # makes the precondition hold BY CONSTRUCTION rather than by hope --
            # the mapping already knows every row of the line.
            # A single-row take has no fold inside it, so it needs no evidence
            # about what a fold consumed. Asking anyway would make an unplaceable
            # line degrade a take that was never at risk.
            source_line = self._source_line(mapping, content[0][0])
            fold_rows, first_offset = self._source_line_rows(bare, mapping, content)
            separators: list[str] | None = []
            if len(trimmed) > 1:
                separators = _copy_markdown.wrap_separators(fold_rows, source_line)

            # ``None`` is a refusal, not a separator: the walk could not place
            # these rows, so nothing here knows what the fold consumed. Inventing
            # a plausible character is precisely the failure of the last three
            # rounds, so the take falls back to the markdown source, which is
            # truthful without needing to know the fold.
            if separators is None:
                # No placement, so this code does not know what the fold
                # consumed. There are two ways to be in that position and they
                # have DIFFERENT safe answers:
                #
                # * ``align`` placed the rows but the walk could not consume the
                #   line (a clipped or unusual selection). The markdown source
                #   is still available and is truthful, so use it.
                # * ``align`` placed NOTHING, which is ordinary for a CJK
                #   paragraph where no anchor word survives. There is no
                #   markdown to fall back to, so the rendered glyphs are the
                #   only truthful answer and the script heuristic decides the
                #   fold -- chosen explicitly here rather than fallen into.
                if source_line is None:
                    # Look for the evidence directly before assuming there is
                    # none: a line the walk CAN place these rows against is proof
                    # they came from it. Only when no line places them, or
                    # several do, is there genuinely nothing to know.
                    source_line = _copy_markdown.locate_source_line(fold_rows, self._full_text)
                    if source_line is not None:
                        separators = _copy_markdown.wrap_separators(fold_rows, source_line)

                if separators is None:
                    if source_line is not None or self._may_hide_a_line_boundary():
                        copied = self._markdown_for_rows(mapping, selected[0][0], selected[-1][0])
                        if copied:
                            return copied, "\n"
                        return super().get_selection(selection)
                    separators = _copy_markdown.separators_without_source(fold_rows)

            joined = trimmed[0]
            for offset, text in enumerate(trimmed[1:]):
                if not text:
                    continue
                # Indexed against the FULL row set, so the reader's first row is
                # at ``first_offset``. No length guard with a ``" "`` default:
                # that default is the same invented character in miniature, and a
                # skew here is a bug to surface, not to paper over (review round
                # 3, R3-3). The slice above guarantees the index is in range.
                separator = separators[first_offset + offset]
                joined = f"{joined}{separator}{text}" if joined else text
            return joined, "\n"

        # Widened to whole source lines on purpose — see the accepted cost
        # above. The bounds come from ``selected`` rather than ``content`` so
        # the markdown path spans exactly the rows it has always spanned.
        copied = self._markdown_for_rows(mapping, selected[0][0], selected[-1][0])
        if not copied:
            return super().get_selection(selection)
        return copied, "\n"

    def _markdown_for_rows(self, mapping: list[int | None], first_row: int, last_row: int) -> str:
        """Markdown for ``first_row..last_row``, widened over UNPLACED edge rows.

        :func:`_copy_markdown.slice_markdown` collects source lines through the
        mapping, so a row ``align`` could not place contributes nothing. When
        such a row is at the EDGE of the selection its source line is dropped
        entirely and content the reader lit vanishes from the clipboard --
        reachable on the continuation rows of an over-long line inside a fence,
        which is where R3-2's welded shell command was found.

        Widening each end out to the nearest placed row recovers those lines
        through the existing gap fill. It errs toward copying MORE than was
        highlighted, which is this path's already-accepted cost (design round 1,
        D3) and the only safe direction: a paste with an extra line is one the
        reader can see and trim, a silently missing or welded line is not.

        GATED ON EVIDENCE THAT THERE IS SOMETHING TO RESCUE. Widening only helps
        where the selection already yields a line and an unplaced EDGE row would
        drop a neighbour. Where NO selected row is placed there is no such line:
        the un-widened slice is ``""``, the caller already falls back to the
        frame copy -- exactly the lit glyphs -- and widening pre-empts that
        correct answer. Worse, because ``slice_markdown`` collects through the
        mapping, widening over unplaced rows adds no line of its own; it drags
        the window onto whichever line IS placed, so the copy becomes a
        DIFFERENT line than the one highlighted. A reader who lit one shell
        command then pastes another and runs it (review round 4, R4-1; design
        round 4, D4-1) -- the same harm class as R3-2's weld, and nothing on the
        clipboard signals the substitution. Rescuing a lit line needs evidence
        that a line was there to rescue; with no placed row there is none, which
        is the same guess-rather-than-know shape this path exists to refuse.
        """
        placed = any(
            mapping[row] is not None
            for row in range(first_row, min(last_row, len(mapping) - 1) + 1)
            if row < len(mapping)
        )
        if not placed:
            return ""

        first, last = first_row, last_row
        while first > 0 and (first >= len(mapping) or mapping[first] is None):
            first -= 1
        while last < len(mapping) - 1 and mapping[last] is None:
            last += 1
        return _copy_markdown.slice_markdown(self._full_text, mapping, first, last)

    def _may_hide_a_line_boundary(self) -> bool:
        """Could an unplaced run of rows straddle TWO source lines?

        The script heuristic in :func:`_copy_markdown.separators_without_source`
        answers what a FOLD consumed, which is only a meaningful question if the
        rows really are folds of one line. When the message has a single
        non-blank source line that is certain by construction -- the ordinary CJK
        paragraph, where no anchor word survives for ``align`` to place.

        With several source lines it is not certain, and assuming it welded three
        lines of a fenced block into one (review round 3, R3-2). Then the answer
        must come from the markdown source, which needs no fold knowledge.
        """
        return len([line for line in self._full_text.split("\n") if line.strip()]) > 1

    def _source_line_rows(
        self,
        rows: list[str],
        mapping: list[int | None],
        content: list[tuple[int, tuple[int, int]]],
    ) -> tuple[list[str], int]:
        """Every rendered row of the selected source line, and the drag's offset.

        ``rows`` is the BARE row list — the rail stripped — for the convention
        :meth:`get_selection` states: the blankness filter below and the
        furniture measurement both have to see a blank separator row as blank,
        and a railed row never is.

        :func:`_copy_markdown.wrap_separators` is end-anchored, so it can only
        place rows that are the WHOLE of the source line. The selection is not
        that: a reader highlighting exactly a URL stops before the paragraph's
        last row, and handing those clipped rows to the walk is what made it
        give up and fall to a guess that split the URL (review round 3, R3-1).

        The mapping already knows which rows belong to the line, so the full set
        is recovered from it rather than inferred from the drag. The second
        element is the index of the reader's FIRST row within that set, which is
        what lets the caller read the separator for its own folds out of the
        full-line answer.

        The gate above guarantees a single source line and no unplaced content
        rows, so ``content`` cannot straddle two lines here.
        """
        source = mapping[content[0][0]] if content[0][0] < len(mapping) else None
        indices = [i for i, mapped in enumerate(mapping) if mapped == source and rows[i].strip()]
        if not indices:
            indices = [index for index, _ in content]
        first_offset = indices.index(content[0][0]) if content[0][0] in indices else 0
        # Furniture is measured from the FRAME rather than from any row subset:
        # ``opens_line`` asks whether the previous painted row came from another
        # source line, which is a property of what Rich painted and not of the
        # reader's drag. Answering it from a selection called a continuation row
        # a marker row whenever the reader started mid-line (issue #395), so
        # :meth:`_furniture_width` now scans the mapping itself.
        full_rows = [
            rows[index][self._furniture_width(rows, mapping, index) :].rstrip() for index in indices
        ]
        return full_rows, first_offset

    def _source_line(self, mapping: list[int | None], row: int) -> str | None:
        """The source line rendered row ``row`` came from, or ``None``.

        ``None`` for a row :func:`_copy_markdown.align` could not place, which
        every caller treats as "assume nothing": the alignment is the only
        evidence about what a painted glyph means, so without it the row is
        returned verbatim rather than guessed at.
        """
        if row >= len(mapping):
            return None
        source = mapping[row]
        if source is None:
            return None
        lines = self._full_text.split("\n")
        return lines[source] if source < len(lines) else None

    def _furniture_width(
        self,
        rows: list[str],
        mapping: list[int | None],
        row: int,
    ) -> int:
        """Painted-furniture columns on rendered row ``row``, in BARE columns.

        ``rows`` is the rail-stripped list and the answer excludes the rail:
        this is the CONSTRUCT's own furniture, and :meth:`copy_gutter` is the
        block-level constant on top of it. Callers comparing against a selection
        column add :data:`RAIL_COLS` back. Handing this the painted rows breaks
        it two ways — ``▌`` would be read as a quote bar, and the blank-row skip
        below would never fire.

        The per-row gutter this block cannot express as a constant. See
        :func:`_copy_markdown.furniture_width` for why the answer needs the
        source line: the same glyph is decoration on a quote row and content
        inside a fence.

        A row OPENS its source line when the previous content row came from a
        different one — which is what distinguishes a list item's marker row
        from its wrapped continuations, whose furniture is indent alone.

        **``opens_line`` is read from the FRAME, not from ``content``.** Whether
        a row carries a painted marker is a property of what Rich painted, so it
        cannot depend on where the reader's drag happened to start. Deriving it
        from the selection made the FIRST selected row always look like a marker
        row: a drag over a wrapped list item's continuation row alone saw no
        previous row, called the continuation an opener, and measured its
        furniture as 0 instead of the painted indent. ``starts_full`` then
        answered differently either side of that indent, so one cell of
        difference in the drag start flipped the paste between the whole
        markdown item and the row's glyphs, and columns inside the indent leaked
        it as leading spaces (issue #395). Scanning the mapping instead keeps the
        D2-4 property that a whole-row gesture gives the SAME answer wherever
        inside the painted furniture it began — the furniture is now measured
        identically for every start column, because it never consults the drag.
        """
        source = mapping[row] if row < len(mapping) else None
        # The nearest preceding row that paints something, taken from the frame.
        # Blank rows are skipped rather than treated as a different source line:
        # a separator row between two constructs would otherwise make the row
        # after it read as an opener regardless of what it actually carries.
        previous: int | None = None
        for index in range(row - 1, -1, -1):
            if index < len(rows) and rows[index].strip():
                previous = mapping[index] if index < len(mapping) else None
                break
        covered, _ = _copy_markdown.classify(self._full_text.split("\n"))
        return _copy_markdown.furniture_width(
            rows[row],
            self._source_line(mapping, row),
            opens_line=source != previous,
            fenced=source is not None and source in covered,
        )
