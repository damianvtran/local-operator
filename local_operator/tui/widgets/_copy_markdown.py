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
import unicodedata

#: A line that opens or closes a fenced code block.
#:
#: The indent is deliberately UNBOUNDED rather than CommonMark's three spaces.
#: Three is the limit relative to the block's CONTAINER, and these lines arrive
#: flat with no container context, so a fence indented under a list item — four
#: spaces under ``- `` , five under ``1. `` , more when nested — read as ordinary
#: text under a bounded pattern. That made ``classify`` return an EMPTY covered
#: set for one of the most common shapes this app paints (numbered steps with a
#: fenced command under each), which in turn let ``furniture_width`` take its
#: list branch over CODE and strip a leading digit: ``3 rows expected`` copied as
#: ``rows expected`` (design round 2, D2-1).
#:
#: The trade-off, taken knowingly: a literal `````` ``` ```` inside a FOUR-SPACE
#: INDENTED code block is now read as a fence. That construct is vanishingly rare
#: in assistant output (Rich's markdown renders indented code as code either
#: way), while list-nested fences are routine, and the failure directions are not
#: symmetric — over-recognising a fence makes the copy verbatim, which is the
#: conservative answer for a clipboard, while under-recognising one deletes
#: characters from the user's code.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
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
    tokens = rendered_row.split()
    # Whether the row is indented. Rich paints an ordered-list marker with a
    # leading space (`` 1 alpha``, `` 100 item``); a number-led PARAGRAPH is
    # flush left (``2026 roadmap``). The indent is what tells a bare-number
    # marker from number-led content, at ANY marker width — a length cap broke
    # three-digit lists (the H1 review finding).
    indented = rendered_row[:1] in (" ", "\t")
    for index, token in enumerate(tokens):
        if _MARKER_TOKEN_RE.fullmatch(token):
            if not token.isdigit():
                continue  # a bullet, bar, or ordered marker-with-dot: structure
            # A bare number is an ordered-list marker only when it is indented
            # and followed by the item's text. A number that begins CONTENT —
            # a flush-left number-led paragraph (``2026 roadmap``), or an item's
            # own text after a bullet (``- 3 ways``) — is the row's anchor and
            # must be returned, not skipped (the G1/G2 review findings).
            preceded_by_marker = index > 0 and bool(_MARKER_TOKEN_RE.fullmatch(tokens[index - 1]))
            if indented and index + 1 < len(tokens) and not preceded_by_marker:
                continue  # standalone ordered marker
            # number-led content: fall through and return it
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
                    # A bare ``>`` separating two quote lines paints no word of
                    # its own, so it cannot answer whether this row starts a new
                    # structural line. Stopping the scan there hid the quoted
                    # LIST behind it: every row of the list was attributed to the
                    # quote's opening line, so ``furniture_width`` was handed a
                    # source line that is a quote but not a list item and kept
                    # the ``•`` (design round 2, D2-4).
                    if _QUOTE_RE.match(line) and not _strip_markup(line):
                        continue
                    anchor = _first_word(_strip_markup(line)) or _first_word(line)
                    if word and anchor and word == anchor:
                        starts_structural = True
                    break

        # A FENCE BODY line likewise always opens its own rendered row, and it
        # has to be checked separately because the structural scan above skips
        # covered lines. Without this a fence indented under a list item was
        # absorbed as a WRAP of the item that opens it: the item's wrap budget
        # was still live, the code row matched no structural anchor, so the code
        # was attributed to ``- Run the check:``. That mis-attribution is what
        # made ``fenced`` False over code even once ``classify`` saw the fence,
        # so ``furniture_width`` read the code's leading ``1`` as an ordered
        # marker and deleted it (design round 2, D2-1, second cause).
        #
        # Matched on EXACT stripped text, the same key the anchor scan below
        # uses for a covered line: a code line has no markup to normalise, and
        # an exact match cannot steal a row from the prose it wraps.
        starts_fence_body = False
        if not starts_structural:
            for look in range(src, min(src + _LOOKAHEAD, n)):
                if look in markers or look not in covered:
                    continue
                line = src_lines[look]
                if line.strip() and row.strip() == line.strip():
                    starts_fence_body = True
                    break
        if (
            current is not None
            and budget > 0
            and not no_wrap_now
            and not starts_structural
            and not starts_fence_body
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


#: The quote bar as Rich paints it, repeated once per nesting level. A quote's
#: CONTINUATION rows carry it too, which is what makes it furniture on every row
#: of the construct rather than a marker that identifies the first one.
_RENDERED_QUOTE_PREFIX_RE = re.compile(r"^(?:▌ ?)+")

#: A rendered list marker and the space after it, with its leading indent. Only
#: the row that OPENS the item carries this; a continuation is indented to the
#: same column with spaces, which ``str.lstrip`` handles instead.
_RENDERED_LIST_PREFIX_RE = re.compile(r"^\s*(?:[•◦▪]|\d{1,9})\s+")


def furniture_width(row: str, source_line: str | None, *, opens_line: bool, fenced: bool) -> int:
    """Leading cells of ``row`` that Rich PAINTED rather than the model wrote.

    The row-level answer to the question :meth:`TranscriptBlock.copy_gutter`
    asks per block, and it has to live here because it is only answerable with
    the alignment: the same ``▌`` is furniture on a quote row and content on a
    row of a fenced diagram, and nothing about the glyph itself says which.
    ``source_line`` is the line :func:`align` mapped the row to, so the decision
    is made from what the model actually wrote.

    Why not a glyph regex over the row alone — the approach
    ``test_a_quote_dragged_from_column_zero_keeps_the_bar`` rejects: inside a
    fence it turns ``  1 / 0`` into ``/ 0``, because a code line that starts
    with a digit is indistinguishable from a rendered ordered marker by glyph
    alone. Keying on the source line makes ``fenced`` an explicit refusal
    instead of a coincidence, so code copies byte for byte.

    ``opens_line`` separates an item's first rendered row from its wrapped
    continuations. Rich paints `` • `` once and indents the rest to the same
    column, so a continuation's furniture is pure indent — stripping a marker
    pattern there would eat content that merely looks like one (a continuation
    beginning ``2026 was the year`` is the reported shape of that bug).
    """
    if source_line is None or fenced:
        # Unplaceable rows and code are returned verbatim: with no source line
        # to attribute a prefix to, every glyph is content until proven
        # otherwise, and that is the conservative direction for a clipboard.
        return 0
    # Constructs COMPOSE, so the branches are applied in painted order rather
    # than as an either/or. A list nested in a blockquote paints ``▌  • text``:
    # testing the quote first and returning stripped the bar but kept the ``•``,
    # leaking a glyph that is nowhere in the user's document and flipping the
    # paste format on a one-cell change of drag start (design round 2, D2-4).
    width = 0
    remainder = source_line
    if _QUOTE_RE.match(remainder):
        match = _RENDERED_QUOTE_PREFIX_RE.match(row)
        if match:
            width = match.end()
        # Peel the quote marker so the line can be re-tested for what it CONTAINS
        # — the same source line is both a quote line and a list item.
        remainder = _QUOTE_RE.sub("", remainder, count=1)

    if _LIST_RE.match(remainder):
        if opens_line:
            match = _RENDERED_LIST_PREFIX_RE.match(row[width:])
            return width + (match.end() if match else 0)
        # A continuation is indented to the marker's column with plain spaces.
        tail = row[width:]
        return width + (len(tail) - len(tail.lstrip(" ")))
    return width


#: Inline markers Rich DROPS when it paints, so a rendered row is not a literal
#: substring of its source line. Stepping over these is what lets the positional
#: walk below place a row of ``**bold text** and `code```.
_INLINE_MARKUP = "*_`~"


def _skip_inline_markup(source: str, i: int) -> int:
    """Advance past inline markers at ``i`` that render as nothing.

    Handles emphasis/code runs and a link's ``](target)`` tail, whose label is
    painted while the target is not.
    """
    while i < len(source):
        char = source[i]
        if char in _INLINE_MARKUP or char == "[":
            i += 1
            continue
        if char == "]" and source.startswith("](", i):
            close = source.find(")", i + 2)
            if close < 0:
                return i
            i = close + 1
            continue
        break
    return i


def _match_visible(source: str, start: int, text: str) -> int | None:
    """End offset of ``text`` rendered from ``source`` at ``start``, or ``None``.

    Compares the LITERAL character first and only treats a mismatch as markup.
    The order matters: intraword ``_`` is literal in CommonMark, so a walk that
    skipped markers unconditionally could not place
    ``some_verylongtoken_here`` and fell back to guessing on exactly the
    mid-token folds the separator has to get right.
    """
    i, j = start, 0
    while j < len(text):
        if i < len(source) and source[i] == text[j]:
            i += 1
            j += 1
            continue
        nxt = _skip_inline_markup(source, i)
        if nxt == i:
            return None
        i = nxt
    return i


def _separator_for_scripts(left: str, right: str) -> str:
    """Fallback separator judged from the characters either side of a fold.

    Only reached when the positional walk could not place the rows. A terminal
    breaks CJK between any two characters because those scripts do not write
    spaces, so a space there is never something the user had; Latin text breaks
    at a space it then consumes, so one has to go back. Asking the adjacent
    characters is a HEURISTIC, but a far narrower one than assuming a space
    everywhere: it is what stops a Chinese or Japanese paragraph gaining an
    invented space at every fold (review round 2, R2-2; design round 2, D2-3),
    while leaving Latin prose — emoji and double-width cells included, which are
    placed by the walk and never reach here — rejoining with the space it lost.
    """
    for char in (left, right):
        if not char or unicodedata.east_asian_width(char) not in ("W", "F"):
            continue
        # Width alone is the WRONG test: an emoji is double-width too, and emoji
        # sit in space-delimited Latin prose that genuinely lost a space at the
        # fold. Review round 2 verified emoji already copied correctly, so the
        # rule is narrowed to wide characters that are TEXT (letters and the
        # CJK punctuation that ends a clause) rather than symbols (category
        # ``So``, which is where emoji live).
        if unicodedata.category(char)[0] in ("L", "P"):
            return ""
    return " "


def wrap_separators(row_texts: list[str], source_line: str | None) -> list[str]:
    """What the terminal CONSUMED at each fold between rows of ONE source line.

    Returns one separator per fold, so ``len(row_texts) - 1`` entries.

    A soft wrap is not always a space. Rich breaks at a space when it can and
    consumes it, so rejoining needs one put back; but a token wider than the
    render segment is broken MID-TOKEN and nothing is consumed, so putting a
    space back would invent a character the user never had. Both are reachable
    in ordinary prose at ordinary widths: a bare URL folds mid-token at 40
    columns while the sentence around it folds at spaces.

    **The decision is POSITIONAL.** The rows are walked against the source line
    with a cursor, and the separator for a fold is the whitespace actually
    sitting at that point in the line. The previous test — does
    ``last_token_of_A + first_token_of_B`` occur ANYWHERE in the line — ignored
    position, so a line using both ``file system`` and ``filesystem`` judged a
    real space-fold to be mid-token and welded two words shut: ``filesystem
    layer``, silently wrong and unrepairable by the reader (review round 2,
    R2-1; design round 2, D2-2, 30 reproductions across five prose lines).

    The walk is **end-anchored**: the rows must consume the line's whole visible
    content, which is what forces a single correct alignment instead of letting
    an early partial match stand. ``row_texts`` must therefore be the FULL rows
    of the source line, not a selection's clipped ends.

    When the walk cannot place the rows the answer falls to
    :func:`_separator_for_scripts`, and that fallback IS a heuristic — stated
    plainly here because an earlier version of this docstring claimed the
    source-asking approach was exact when it was not (design round 2, D2-2).
    """
    folds = max(len(row_texts) - 1, 0)
    if not folds:
        return []

    stripped = [text.strip() for text in row_texts]
    if source_line is not None:
        walked = _walk_folds(stripped, source_line)
        if walked is not None:
            return walked

    # No placement, so decide each fold from the scripts meeting at it. This is
    # also the branch an unplaceable row takes (``align`` returned ``None``),
    # where returning ``" "`` unconditionally is what put a space into CJK.
    return [_separator_for_scripts(stripped[i][-1:], stripped[i + 1][:1]) for i in range(folds)]


def _walk_folds(row_texts: list[str], source_line: str) -> list[str] | None:
    """Separators from a positional walk of ``row_texts`` over ``source_line``.

    ``None`` when no start offset lets the rows consume the line exactly, which
    leaves the caller its documented fallback rather than a wrong answer.
    """
    for start in range(len(source_line) + 1):
        cursor = _match_visible(source_line, start, row_texts[0])
        if cursor is None:
            continue
        separators: list[str] = []
        for text in row_texts[1:]:
            # Step over whatever sits between the rows: the whitespace the fold
            # consumed, plus any inline marker that paints nothing. Whether that
            # run held WHITESPACE is the whole question — that is the character
            # the rejoin has to put back.
            gap = cursor
            saw_space = False
            while gap < len(source_line):
                if source_line[gap].isspace():
                    saw_space = True
                    gap += 1
                    continue
                nxt = _skip_inline_markup(source_line, gap)
                if nxt == gap:
                    break
                gap = nxt
            end = _match_visible(source_line, gap, text)
            if end is None:
                separators = []
                break
            separators.append(" " if saw_space else "")
            cursor = end
        else:
            if source_line[cursor:].strip() == "":
                return separators
    return None


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
    # Every picked line is emitted AS WRITTEN. Quote lines already carry their
    # own ``>``/``>>`` prefix in the source, so nested levels survive the copy
    # (the G3 review finding, where re-applying a single prefix flattened them).
    for i in sorted(picked):
        out.append(lines[i])
    if in_fence_at_start and fence_marker and not includes_close:
        out.append(fence_marker)
    return "\n".join(out).strip("\n")
