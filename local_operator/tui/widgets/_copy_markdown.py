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
_RENDERED_PREFIX_RE = re.compile(r"^\s*(?:(?:▌|•|◦|▪|\d{1,9}[.)])\s+)*")

#: A leading LIST marker on a rendered row (bullet or number). A new list item
#: always opens with one and a wrapped continuation never does, so its presence
#: means the row starts a new source line. The quote bar is NOT in this set:
#: Rich repeats ``▌`` on every row of a quote, continuations included, so a bar
#: says nothing about whether the row is a new source line.
_RENDERED_MARKER_RE = re.compile(r"^\s*(?:•|◦|▪|\d{1,9}[.)])\s")


def _first_word(text: str) -> str:
    """The first run of non-space, non-markup characters, lowercased.

    ``**bold** lead`` anchors on ``bold``; the asterisks never reach the frame.
    """
    for token in text.split():
        word = token.strip("*_`~#>|-+.")
        if word:
            return word.lower()
    return ""


def _row_word(rendered_row: str) -> str:
    """A rendered row's first content word, structure markers stripped."""
    return _first_word(_RENDERED_PREFIX_RE.sub("", rendered_row))


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
            # Two passes so a STRUCTURAL start wins over a plain paragraph whose
            # first word happens to match — otherwise a quote line's first word
            # can steal a row meant for a later item.
            for want_structure in (True, False):
                for look in range(src, min(src + _LOOKAHEAD, n)):
                    line = src_lines[look]
                    if look in markers:
                        continue  # backtick rows paint nothing; step over
                    if look in covered:
                        if not want_structure and line.strip() and row.strip() == line.strip():
                            placed = look
                            break
                        continue
                    if not line.strip():
                        continue  # content is further on; blanks pair with blank rows
                    structural = bool(
                        _LIST_RE.match(line) or _QUOTE_RE.match(line) or _HEADING_RE.match(line)
                    )
                    if want_structure and not structural:
                        continue
                    anchor = _first_word(_strip_markup(line)) or _first_word(line)
                    if word and anchor and word == anchor:
                        placed = look
                        break
                    # An hr or a table divider renders as a bar with no anchor word.
                    if not want_structure and _is_rule_like(line) and _is_rule_like(row):
                        placed = look
                        break
                    # A table body row renders as its cells with the pipes
                    # dropped; match it on the first cell's word. The divider
                    # (a rule-like line) is caught above.
                    if not want_structure and "|" in line and word and word in line.lower():
                        placed = look
                        break
                if placed is not None:
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
    # maps the rendered rows it could place; the lines between them belong to
    # the selection too — a fence's marker rows, the blank before a trailing
    # paragraph, and the quote lines Rich merged into one wrapped block (a
    # quote's consecutive ``>`` lines render as a single paragraph, so only the
    # first anchors and the rest must be recovered here). Filling the whole
    # contiguous span is what keeps that content instead of closing up.
    lo, hi = min(picked), max(picked)
    picked.update(range(lo, hi + 1))

    # Rich renders a quote's consecutive ``>`` lines as ONE wrapped block, so a
    # selection that reaches into a quote has highlighted the WHOLE merged block
    # even though only the first source line anchored. Extend the slice to the
    # end of the quote run so the copy carries every line the reader saw, not
    # just the one that happened to anchor. (Same recovery for a list, whose
    # items likewise reflow together at some widths.)
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
    if in_fence_at_start:
        for i in range(first, -1, -1):
            if i in markers:
                m = _FENCE_RE.match(lines[i])
                fence_marker = m.group(1)  # type: ignore[union-attr]
                fence_lang = lines[i].strip()[len(fence_marker) :]
                break
    quote_prefix = ""
    if first not in covered:
        qm = _QUOTE_RE.match(lines[first])
        if qm is not None:
            quote_prefix = qm.group(0)

    out: list[str] = []
    if in_fence_at_start and fence_marker:
        out.append(fence_marker + fence_lang)
    for i in sorted(picked):
        text = lines[i]
        if quote_prefix and i not in covered:
            body = _QUOTE_RE.sub("", text)
            out.append((quote_prefix + body) if body else quote_prefix.rstrip())
        else:
            out.append(text)
    if in_fence_at_start and fence_marker:
        out.append(fence_marker)
    return "\n".join(out).strip("\n")
