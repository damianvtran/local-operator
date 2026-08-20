"""Markdown-source copy for the assistant block.

A copy of an assistant message should paste as MARKDOWN, not as the rendered
frame: the frame turns a blockquote into a ``▌`` bar, a bullet into ``•``, a
table into box-drawing and a heading into bare bold text — none of which a
messenger or email client recognises, and all of which interrupt a paste with
ASCII furniture. The block already holds the message's source
(``AssistantBlock._full_text``), so the clipboard can carry that, mapped to
whatever the reader actually highlighted.

Rendered→markdown is lossy in BOTH directions (a quote's two source lines
merge into one wrapped row; a table becomes box-drawing; ``**`` vanishes), so
the frame cannot be reverse-parsed back to source. The only honest source of
markdown is the source itself. The work here is therefore ALIGNMENT: given the
flattened rows the block painted and the source lines it was built from,
decide which source lines each rendered row came from, so a selection can be
sliced out of the source.

The alignment leans on what Rich's markdown renderer guarantees (verified
against ``flatten`` at several widths): a paragraph WRAPS, so its continuation
rows carry no marker; but every LIST ITEM, BLOCKQUOTE LINE, HEADING and FENCED
CODE line starts a fresh rendered row, and a fence's backtick rows vanish.
Those are anchors. Between anchors, rows wrap. Blank source lines render as
blank rows except at the very top and bottom of the message, where the
renderer trims them.
"""

from __future__ import annotations

import re

#: A line that opens or closes a fenced code block.
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
#: A list item (bullet or ordered).
_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d{1,9}[.)])\s")
#: A blockquote line (one or more ``>`` levels).
_QUOTE_RE = re.compile(r"^\s{0,3}(?:>\s?)+")
#: An ATX heading.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
#: A horizontal rule.
_HR_RE = re.compile(r"^\s{0,3}(?:([-*_])\s*){3,}$")

#: How far ahead of the walk pointer a row may find its source line. Wide
#: enough to step over a fence marker, a table divider and a run of blank
#: lines; the monotonic guard keeps a wide window from skipping content.
_LOOKAHEAD = 12


def _strip_markup(line: str) -> str:
    """The visible text a structural source line contributes, marker removed.

    A list item renders its text after the bullet, a quote after the bar, a
    heading after the ``#``s. Inline markup is left alone — the renderer keeps
    the visible characters of ``**bold**`` even though it drops the asterisks,
    so anchoring compares on the first plain WORD (see :func:`_first_word`).
    """
    line = _QUOTE_RE.sub("", line)
    line = _LIST_RE.sub("", line)
    line = _HEADING_RE.sub("", line)
    return line.strip()


#: A rendered row's leading structure: a quote bar (``▌ ``), a bullet (``• ``)
#: or an ordered marker (``1. ``), any of them nested. Stripped before the
#: row's first CONTENT word is read so it can be anchored to its source line.
_RENDERED_PREFIX_RE = re.compile(r"^\s*(?:(?:▌|•|◦|▪|\d{1,9}[.)])\s*)*")

#: A leading LIST marker on a rendered row (bullet or number). A new list item
#: always opens with one and a wrapped continuation never does, so its presence
#: means the row starts a new source line. The quote bar is NOT in this set:
#: Rich repeats ``▌`` on every row of a quote, continuations included, so a bar
#: says nothing about whether the row is a new source line.
_RENDERED_MARKER_RE = re.compile(r"^\s*(?:•|◦|▪|\d{1,9}[.)])(?:\s|$)")

#: A single whitespace-delimited token that is ONLY a structure marker (quote
#: bar, bullet, or ordered number) — skipped when reading a row's content word.
#: Rich renders an ordered marker as a bare number (`` 1 alpha`` splits to the
#: token ``1``, the ``.`` is not painted), so the marker token is a bullet, a
#: bar, an ordered marker WITH its dot, or a bare integer.
_MARKER_TOKEN_RE = re.compile(r"(?:▌|•|◦|▪|\d{1,9}[.)]|\d{1,9})")


def _first_word(text: str) -> str:
    """The first run of non-space, non-markup characters, lowercased.

    ``**bold** lead`` anchors on ``bold``; the asterisks never reach the frame.
    A token that is ONLY a list marker (``1.``, ``-``) or only punctuation is
    skipped, so an ordered item anchors on its first content word (``alpha`` in
    ``1. alpha``), not on the marker — stripping ``.`` off a bare ``1`` used to
    leave an empty anchor and the item never matched (the F3 review finding).
    As a last resort a token that stripping empties entirely is returned raw,
    so a punctuation-led line still has something to anchor on.
    """
    fallback = ""
    for token in text.split():
        word = token.strip("*_`~#>|-+.")
        if word:
            return word.lower()
        if not fallback and token.strip("*_`~"):
            fallback = token.lower()
    return fallback


def _row_word(rendered_row: str) -> str:
    """A rendered row's first content word, structure markers stripped.

    The marker strip and the word read are one loop rather than a regex sub
    followed by a scan, because a marker and its text can be separated by a
    single space (``1 alpha``) — a ``\\s+``-after-marker pattern misses that and
    leaves the bare number as the "word", which is why an ordered list anchored
    to nothing (the F3 review finding).
    """
    for token in rendered_row.split():
        if _MARKER_TOKEN_RE.fullmatch(token):
            continue  # a bullet, bar or ordered marker, not content
        return _first_word(token)
    return ""


def classify(lines: list[str]) -> tuple[set[int], set[int]]:
    """(fence-covered lines, fence-marker lines) for ``lines``.

    Covered is every line inside a fenced block, markers included; markers is
    the subset that opens or closes one (they render NOTHING — the backticks
    are not painted). Both are needed: a body line copies verbatim, a marker
    line is re-added around a mid-fence slice rather than copied as text.
    """
    covered: set[int] = set()
    markers: set[int] = set()
    in_fence = False
    fence_char = ""
    for i, line in enumerate(lines):
        match = _FENCE_RE.match(line)
        if match is not None:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                markers.add(i)
                covered.add(i)
            elif marker[0] == fence_char and set(line.strip()) <= {fence_char}:
                in_fence = False
                markers.add(i)
                covered.add(i)
            continue
        if in_fence:
            covered.add(i)
    return covered, markers


def align(source: str, rendered_rows: list[str]) -> list[int | None]:
    """Map each rendered row to the source line it came from (or ``None``).

    ``None`` marks a blank separator row (or a row the walk could not place),
    which a copy slices around rather than from. The walk is monotonic —
    rendered rows and source lines are in the same order — so a single pointer
    advances through the source as rows are consumed. A non-blank row first
    tries to CONTINUE the source line it is on (a wrap), then looks ahead a
    bounded window for the source line that starts it. A blank row consumes a
    blank source line when one is next, else marks a separator.
    """
    src_lines = source.split("\n")
    covered, markers = classify(src_lines)
    n = len(src_lines)

    def paints(i: int) -> bool:
        """Does source line ``i`` put any glyph on the frame?"""
        if i in markers:
            return False
        return bool(src_lines[i].strip())

    # Lead/trail trim: the renderer drops blank rows at the message edges, so
    # the walk starts at the first line that paints and stops after the last.
    first_paint = 0
    while first_paint < n and not paints(first_paint):
        first_paint += 1

    mapping: list[int | None] = []
    src = first_paint  # next unconsumed source line
    current: int | None = None  # source line rows are currently wrapping within
    budget = 0  # rendered rows the current source line may still wrap to
    for row in rendered_rows:
        word = _row_word(row)

        if not row.strip():
            # A blank row consumes the next blank source line when one is here
            # (a paragraph break), else it is a renderer separator (around a
            # quote or table) with no source line of its own.
            if src < n and src not in covered and not src_lines[src].strip():
                mapping.append(src)
                src += 1
            else:
                mapping.append(None)
            current = None  # any break ends a wrap in progress
            budget = 0
            continue

        placed: int | None = None
        # A fence body line and a table row never wrap — each occupies exactly
        # its own row — so a row inside either is never a continuation.
        no_wrap_now = current is not None and (current in covered or "|" in src_lines[current])
        # A STRUCTURAL source line (list item, quote line, heading) always opens
        # a fresh rendered row, so if this row's first content word matches one
        # ahead, it STARTS that line — it is never a wrap of the current one.
        # That is what stops a quote line's wrap budget swallowing the next
        # ``>`` line (whose first word matches) while still letting the quote's
        # own continuation (whose first word does not) wrap.
        starts_structural = False
        if not no_wrap_now:
            for look in range(src, min(src + _LOOKAHEAD, n)):
                line = src_lines[look]
                if look in covered or not line.strip():
                    continue
                if _LIST_RE.match(line) or _QUOTE_RE.match(line) or _HEADING_RE.match(line):
                    anchor = _first_word(_strip_markup(line)) or _first_word(line)
                    if word and anchor and word == anchor:
                        starts_structural = True
                    break
        if (
            current is not None
            and budget > 0
            and not no_wrap_now
            and not starts_structural
            and not _RENDERED_MARKER_RE.match(row)
        ):
            placed = current
            budget -= 1
        else:
            # Scan in SOURCE ORDER and take the first line this row can anchor
            # to. Rendered rows and source lines are monotonic, so the earliest
            # match is the right one — a plain paragraph anchors to ITSELF, never
            # to a later quote that merely shares its first word (the F1 review
            # finding, where a two-pass structural-first scan let the quote steal
            # the paragraph's row). A fence body line matches on its exact text;
            # anything else on its first content word, markup stripped.
            for look in range(src, min(src + _LOOKAHEAD, n)):
                line = src_lines[look]
                if look in markers:
                    continue  # backtick rows paint nothing; step over
                if look in covered:
                    if line.strip() and row.strip() == line.strip():
                        placed = look
                        break
                    continue
                if not line.strip():
                    continue  # content is further on; blanks pair with blank rows
                anchor = _first_word(_strip_markup(line)) or _first_word(line)
                if word and anchor and word == anchor:
                    placed = look
                    break
                # An hr or a table divider renders as a bar with no anchor word.
                if _is_rule_like(line) and _is_rule_like(row):
                    placed = look
                    break
                # A table body row renders as its cells with the pipes dropped;
                # match it on the first cell's word. Rich truncates a long cell
                # with ``…``, so also match a truncated stem: the rendered word
                # is the cell's prefix, not a substring of the full line (the F4
                # review finding).
                if "|" in line and word:
                    stem = word.rstrip("…")
                    if word in line.lower() or (stem and stem in line.lower()):
                        placed = look
                        break
            if placed is not None:
                src = placed + 1
                # A fence body line and a table row occupy exactly one row.
                # Everything else — paragraphs, list items and quote lines alike
                # — may wrap; the starts-structural check above re-anchors the
                # next item or quote line when the wrap runs out.
                budget = 0 if (placed in covered or "|" in src_lines[placed]) else _LOOKAHEAD
        mapping.append(placed)
        current = placed
    return mapping


def _is_rule_like(text: str) -> bool:
    """A row whose visible content is only rule/divider characters."""
    body = text.strip().strip("|")
    return bool(body) and set(body) <= {"-", "─", ":", " ", "=", "_"}


def slice_markdown(source: str, mapping: list[int | None], first_row: int, last_row: int) -> str:
    """The markdown for rendered rows ``first_row..last_row`` inclusive.

    Takes the source lines those rows map to, then re-fences and re-quotes the
    slice so it is valid markdown on its own: a slice that opens inside a code
    fence gets the fence's backticks wrapped around it, and a slice from inside
    a blockquote gets the ``> `` prefix re-applied to each line. Without the
    repair a mid-fence or mid-quote selection would paste as unformatted text
    with no sign of what it was.
    """
    if first_row > last_row:
        return ""
    lines = source.split("\n")
    covered, markers = classify(lines)
    picked: set[int] = set()
    for row in range(first_row, last_row + 1):
        if row >= len(mapping):
            break
        src = mapping[row]
        if src is not None:
            picked.add(src)
    if not picked:
        return ""

    # Fill the source lines between the first and last picked line. The walk
    # places rows by POSITION (earliest matching source line wins), so a gap
    # line here is always inside the selected span, never an unselected block
    # that happens to share a first word — the F1 failure came from the walk
    # mis-anchoring, not from filling. Filling recovers the lines that render
    # nothing of their own: fence markers, blank separators, reference-link
    # definitions, and the quote lines Rich merged into one wrapped block.
    lo, hi = min(picked), max(picked)
    picked.update(range(lo, hi + 1))

    # A selection that reaches the LAST rendered row covers the whole message,
    # so it also covers any trailing source lines that render nothing of their
    # own — a reference-link definition or a trailing blank. Without this those
    # are dropped from a full-message copy (the F1b review finding).
    n_rows = len(mapping)
    if last_row >= n_rows - 1:
        picked.update(range(hi, len(lines)))

    # Rich renders a quote's consecutive ``>`` lines as ONE wrapped block, so a
    # selection that reaches into a quote has highlighted the WHOLE merged block
    # even though only the first source line anchored. Extend the slice to the
    # end of the quote run — stopping at the first line that is not itself a
    # quote (a blank ends the run, so a following paragraph or heading is never
    # absorbed). The run-stop is what the F1 finding's blanket fill lacked.
    n = len(lines)
    last = max(picked)
    if _QUOTE_RE.match(lines[last]):
        j = last + 1
        while j < n and _QUOTE_RE.match(lines[j]):
            picked.add(j)
            j += 1

    first = min(picked)
    in_fence_at_start = first in covered and first not in markers
    fence_marker = ""
    fence_lang = ""
    open_marker_line: int | None = None
    if in_fence_at_start:
        for i in range(first, -1, -1):
            if i in markers:
                m = _FENCE_RE.match(lines[i])
                fence_marker = m.group(1)  # type: ignore[union-attr]
                fence_lang = lines[i].strip()[len(fence_marker) :]
                open_marker_line = i
                break
    # The quote prefix to re-apply to a mid-quote slice. It is derived from the
    # first QUOTE line in the slice and applied only to lines that are themselves
    # quotes — a trailing heading or paragraph after the quote run keeps its own
    # form (the F1c review finding, where every non-covered line was re-quoted).
    quote_prefix = ""
    for i in sorted(picked):
        if i not in covered:
            qm = _QUOTE_RE.match(lines[i])
            if qm is not None:
                quote_prefix = qm.group(0)
                break

    # Prepend the opener only when the slice does not already include the
    # fence's own opening marker line; append the closer only when it does not
    # include the closing marker. Without the symmetric end check a fence whose
    # body runs to a blank line gets a spurious closing fence AFTER the trailing
    # paragraph, swallowing it into the code block on paste (the F2 finding).
    includes_open = open_marker_line in picked if open_marker_line is not None else False
    includes_close = any(
        i in markers and i != open_marker_line and i > (open_marker_line or -1) for i in picked
    )
    out: list[str] = []
    if in_fence_at_start and fence_marker and not includes_open:
        out.append(fence_marker + fence_lang)
    for i in sorted(picked):
        text = lines[i]
        # Re-apply the quote prefix only to lines that are quotes in the source;
        # everything else (headings, paragraphs, blanks, fence lines) is emitted
        # as written.
        if quote_prefix and i not in covered and _QUOTE_RE.match(text):
            body = _QUOTE_RE.sub("", text)
            out.append((quote_prefix + body) if body else quote_prefix.rstrip())
        else:
            out.append(text)
    if in_fence_at_start and fence_marker and not includes_close:
        out.append(fence_marker)
    return "\n".join(out).strip("\n")
