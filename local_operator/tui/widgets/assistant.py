"""Assistant message block — rich Markdown with the FROZEN-PREFIX trick.

Naive re-render-the-whole-message-per-token is quadratic and is the dominant
streaming cost (omp issue #4353). Instead we keep a ``(frozen_text,
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
- Row accounting is lazy (inherited from ``TranscriptBlock``): streaming
  updates never measure the renderable; ``settled_rows`` estimates via rich
  measurement only when asked.

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

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import TranscriptBlock

#: Reference-link definition line: ``[label]: target`` (TUI-010 refusal).
_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s", re.MULTILINE)
#: A line that continues a list: bullet or ordered item (TUI-010 deferral).
_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d{1,9}[.)])\s")
#: Fence marker: 3+ backticks or tildes (commonmark fenced-code opener).
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


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


def _candidate_boundaries(lines: list[str], offsets: list[int], covered: set[int],
                          text_len: int) -> list[tuple[int, int]]:
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
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_class("assistant-block")
        self._full_text: str = ""
        self._frozen_text: str = ""
        self._frozen_rendered: Markdown | None = None
        self._frozen_epoch: int = -1
        # Incremental fence tracking (TUI-011a): state as of the last scan.
        self._scanned_len: int = 0
        self._scanned_lines: int = 0  # completed lines already fence-scanned
        self._in_fence: bool = False
        self._fence_marker: str = ""
        self._covered: set[int] = set()

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

        append_only = text.startswith(self._full_text) and self._scanned_len <= len(text)
        self._track_fences(text, append_only)
        self._full_text = text

        lines = text.split("\n")
        boundary = _stable_boundary(text, lines, self._covered)
        if boundary > 0:
            prefix = text[:boundary]
            tail = text[boundary:]
            if prefix != self._frozen_text:
                # Prefix grew: a new block settled. Re-render frozen once.
                self._frozen_text = prefix
                self._frozen_rendered = Markdown(prefix)
                self._frozen_epoch = epoch
            assert self._frozen_rendered is not None
            # The tail is rebuilt per flush; a keyed cache here retained a
            # full copy of the message per flush (O(n^2) bytes) for a cache
            # that can never hit — the equality guard already no-ops
            # identical re-emits and append-only text only moves forward.
            renderable = Group(self._frozen_rendered, Markdown(tail))
        else:
            renderable = Markdown(text)
        self.set_content(renderable)

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
        """Commit the full text as one render and freeze the block."""
        if self._finalized:
            return
        self._full_text = self._full_text or ""
        self.set_content(Markdown(self._full_text) if self._full_text else Text(""))
        self.finalize()

    def settled_rows(self) -> int:
        """Rows provably stable now: the frozen prefix's render while live."""
        if self._finalized:
            return super().settled_rows()
        # While streaming, only the frozen prefix is byte-stable. Lazy: the
        # count is estimated only when asked, at the block's own width (D3).
        if self._frozen_rendered is not None:
            return _measure_rows(self._frozen_rendered, self.size.width or 80)
        return 0

    def text(self) -> str:
        """The accumulated message text (for tests and export)."""
        return self._full_text


def _measure_rows(renderable: object, width: int = 80) -> int:
    """Row count a rich renderable occupies, measured lazily via rich."""
    from rich.console import Console

    console = Console(width=max(width, 10))
    try:
        segments = console.render(renderable, console.options)  # type: ignore[arg-type]
    except Exception:
        return 1
    rows = 1
    for segment in segments:
        rows += segment.text.count("\n")
    return rows
