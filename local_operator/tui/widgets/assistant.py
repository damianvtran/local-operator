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
from rich.text import Text
from textual.content import Content
from textual.selection import Selection

from local_operator.tui import theme as theme_mod
from local_operator.tui.markdown_theme import brand_markdown_theme
from local_operator.tui.widgets import _copy_markdown
from local_operator.tui.widgets.transcript import TranscriptBlock

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
        """
        return self.fold_width(FALLBACK_WIDTH)

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
        """
        self._built_width = self._flat_width()
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
            return self._flat_whole()
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
        if not self._full_text:
            return
        if self._flat_width() == self._built_width:
            return
        was_finalized = self._finalized
        self._finalized = False
        try:
            rows = self._flat_whole() if was_finalized else self._flat_rows(self._flat_width())
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

    def _flat_whole(self) -> Text:
        """The whole message as one flatten, at the block's current width."""
        if not self._full_text:
            return Text("")
        return flatten(Markdown(self._full_text), self._flat_width(), self._flat_console())

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
        exactly when it is wider than the block.
        """
        text = self._full_text.strip()
        if not text:
            return False
        if "\n" in text:
            return True
        return cell_len(text) > max(self.size.width or 80, 10)

    def text(self) -> str:
        """The accumulated message text (for tests and export)."""
        return self._full_text

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
        mapping = _copy_markdown.align(self._full_text, rows)

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
        content = [(i, span) for i, span in selected if rows[i].strip()]

        sub_line = False
        if content:
            # Source LINES, not rows: a paragraph that wraps is one line painted
            # as several, and a phrase dragged across that fold is as much a
            # sub-line take as one inside a single row. A row ``align`` cannot
            # place contributes nothing, which leaves the conservative answer
            # (glyphs are always truthful) if alignment itself ever regresses.
            sources = {
                mapping[i] for i, _ in content if i < len(mapping) and mapping[i] is not None
            }
            if len(sources) <= 1:
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
                starts_full = first_start <= self._furniture_width(
                    rows, mapping, first_index, content
                )
                ends_full = last_end == -1 or last_end >= len(rows[last_index].rstrip())
                sub_line = not (starts_full and ends_full)

        if sub_line:
            # Sliced as ``TranscriptBlock.get_selection`` slices it (``-1``
            # meaning end of row), but over the content rows only — a blank row
            # caught at the edge of the drag would contribute a line break the
            # reader never highlighted — and clamped past each row's PAINTED
            # furniture rather than past ``copy_gutter``.
            #
            # ``copy_gutter`` is 0 on this block and cannot be anything else:
            # an assistant message has no fixed gutter, because what is
            # furniture depends on the construct the row belongs to. Clamping
            # to it stripped nothing, so a sub-line take across a wrapped
            # quote's fold put the ``▌`` on the clipboard and a column-0 drag
            # picked up the ``•`` — furniture the base commit never copied, and
            # the exact leak this method's docstring claims to prevent (review
            # round 1, R1-1; design round 1, D1).
            glyphs = [
                rows[index][
                    max(start, self._furniture_width(rows, mapping, index, content)) : (
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
            # The separators are decided from the FULL rows of the source line,
            # not from the clipped ends the reader highlighted. The walk that
            # replaced the old substring test is positional and end-anchored, so
            # it needs the same text Rich painted: a trimmed first row cannot be
            # located in the source line, and a partial take is exactly where a
            # membership test used to guess wrong (review round 2, R2-1).
            source_line = self._source_line(mapping, content[0][0])
            full_rows = [
                rows[index][self._furniture_width(rows, mapping, index, content) :].rstrip()
                for index, _ in content
            ]
            separators = _copy_markdown.wrap_separators(full_rows, source_line)

            joined = trimmed[0]
            for offset, text in enumerate(trimmed[1:]):
                if not text:
                    continue
                separator = separators[offset] if offset < len(separators) else " "
                joined = f"{joined}{separator}{text}" if joined else text
            return joined, "\n"

        # Widened to whole source lines on purpose — see the accepted cost
        # above. The bounds come from ``selected`` rather than ``content`` so
        # the markdown path spans exactly the rows it has always spanned.
        first_row = selected[0][0]
        last_row = selected[-1][0]
        copied = _copy_markdown.slice_markdown(self._full_text, mapping, first_row, last_row)
        if not copied:
            return super().get_selection(selection)
        return copied, "\n"

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
        content: list[tuple[int, tuple[int, int]]],
    ) -> int:
        """Painted-furniture columns on rendered row ``row``.

        The per-row gutter this block cannot express as a constant. See
        :func:`_copy_markdown.furniture_width` for why the answer needs the
        source line: the same glyph is decoration on a quote row and content
        inside a fence.

        A row OPENS its source line when the previous content row came from a
        different one — which is what distinguishes a list item's marker row
        from its wrapped continuations, whose furniture is indent alone.
        """
        source = mapping[row] if row < len(mapping) else None
        previous: int | None = None
        for index, _ in content:
            if index == row:
                break
            previous = mapping[index] if index < len(mapping) else None
        covered, _ = _copy_markdown.classify(self._full_text.split("\n"))
        return _copy_markdown.furniture_width(
            rows[row],
            self._source_line(mapping, row),
            opens_line=source != previous,
            fenced=source is not None and source in covered,
        )
