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
        """The width the rows are built at — the block's own once laid out."""
        return self.size.width or FALLBACK_WIDTH

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
        """
        visual = self._render()
        if not isinstance(visual, Content):
            return None
        if not self._full_text.strip():
            return super().get_selection(selection)
        rows = visual.plain.split("\n")
        first_row: int | None = None
        last_row: int | None = None
        for index in range(len(rows)):
            if selection.get_span(index) is not None:
                if first_row is None:
                    first_row = index
                last_row = index
        if first_row is None or last_row is None:
            return None
        mapping = _copy_markdown.align(self._full_text, rows)
        copied = _copy_markdown.slice_markdown(self._full_text, mapping, first_row, last_row)
        if not copied:
            return super().get_selection(selection)
        return copied, "\n"
