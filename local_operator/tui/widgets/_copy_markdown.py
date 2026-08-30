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


#: A whitespace-delimited token that is ONLY quote-bar glyphs. In a SOURCE line
#: such a token is the model's own text -- a document may legitimately open a
#: quoted line with ``▌`` -- but the row Rich paints from it is
#: indistinguishable from the bar Rich paints as furniture, and ``_row_word``
#: skips both. The source-side anchor has to skip it too, or the two sides can
#: never agree on a word and the row is left unplaced (issue #392).
_BAR_TOKEN_RE = re.compile(r"▌+")


def _source_anchor(line: str) -> str:
    """The word :func:`align` anchors a rendered row to, for source ``line``.

    ``_row_word`` reads the rendered side and skips every leading structure
    glyph, the quote bar included. Reading the source side with ``_first_word``
    alone is therefore ASYMMETRIC: a quoted line whose own text opens with
    ``▌`` anchors on that bar while its rendered row anchors on the word after
    it, so the two never match and the row is never placed. ``align`` then hands
    ``furniture_width`` no source line, its documented ``source_line is None``
    branch strips nothing, and the whole rendered row -- painted bar included --
    reaches the clipboard (issue #392).

    Skipping the bar tokens restores the symmetry. It is safe in the direction
    that matters: ``_row_word`` can never RETURN a bar (they are all in
    ``_MARKER_TOKEN_RE``), so an anchor of ``▌`` could only ever have matched
    nothing. The original chain is kept as the fallback so a line that is
    NOTHING but bars still has something to anchor on rather than an empty word,
    which matches no row and would silently lose the line.
    """
    body = _strip_markup(line)
    tokens = body.split()
    while tokens and _BAR_TOKEN_RE.fullmatch(tokens[0]):
        tokens.pop(0)
    return _first_word(" ".join(tokens)) or _first_word(body) or _first_word(line)


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
    # The frame column at which the table being walked paints its FIRST cell,
    # learned from a row already placed on one of that table's source lines.
    # This is the positional evidence the numeric path needs; see
    # ``_number_opens_row`` for why value equality alone is not sound (R2-1).
    table_column: int | None = None
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
            table_column = None  # a break also ends the table the column described
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
                    anchor = _source_anchor(line)
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
                anchor = _source_anchor(line)
                if word and anchor and word == anchor:
                    placed = look
                    break
                # An hr or a table divider renders as a bar with no anchor word.
                if _is_rule_like(line) and _is_rule_like(row):
                    placed = look
                    break
                # A table body row renders as its cells with the pipes dropped;
                # match it on the FIRST CELL ONLY, never the whole line. Table
                # cells WRAP -- the ``no_wrap_now`` claim above is about a row
                # never CONTINUING, not about its cells fitting -- so searching
                # the whole line let a continuation carrying some later cell's
                # tail match the NEXT table row on any shared word and copy it
                # entire (issue #399). The first cell is what Rich paints at the
                # row's START, so it is the only part of the line a row-opening
                # rendered row can be identified by; a continuation matches no
                # first cell and is left UNPLACED, which the copy path already
                # reads as "assume nothing" and answers with the lit glyphs.
                if "|" in line and (word or (_is_table_row(line) and _row_number(row))):
                    first_cell = _first_cell(line)
                    # Compared against the cell's FIRST WORD, and by EQUALITY.
                    # Containment is not evidence that this row opens the line:
                    # a first cell is short, so a word from one cell is easily a
                    # substring of a DIFFERENT row's cell -- ``'10' in '100'``,
                    # ``'at' in 'api-gateway'``, ``'on' in 'monitoring'`` -- and
                    # the row that matched was then copied whole, handing back a
                    # line the reader never lit. That is #399's payload one
                    # mechanism over, and it is sharpest on a NUMERIC header
                    # (``| 1 | 10 |``): ``_row_word`` reads a leading bare number
                    # as an ordered-list marker, so the header's word comes from
                    # its SECOND cell and cannot match its own line at all --
                    # leaving a lookahead substring hit as the only match it can
                    # make (R3-1).
                    #
                    # Prefix, not equality, ONLY when Rich actually truncated the
                    # rendered word: a cell too wide for its column is painted
                    # with a trailing ``…`` and the row word is then a genuine
                    # PREFIX of the cell (the F4 review finding). Gating on that
                    # ellipsis is what keeps the relaxation from re-admitting the
                    # substitutions above -- ``'10'`` is not a truncation of
                    # ``'100'``, so an ungated prefix test would still place the
                    # numeric header on a body row.
                    #
                    # ``_first_word`` rather than the raw cell because the cell
                    # is MARKDOWN and the row is what Rich PAINTED: ``**alpha**``
                    # and ``` `reload` ``` and ``--quiet`` all render without
                    # their markup, so a raw comparison fails on the very rows it
                    # is meant to place.
                    head = _first_word(first_cell)
                    stem = word.rstrip("…")
                    truncated = stem != word
                    matched = (
                        bool(head)
                        and bool(word)
                        and (head == word or (truncated and bool(stem) and head.startswith(stem)))
                    )
                    # A numeric first column (``| 1 | step |``, ids, ranks,
                    # years) is invisible to ``word``: ``_row_word`` skips a
                    # leading bare number as an ordered-list MARKER, so the row
                    # word for `` 1   the first detail`` is ``the`` and no first
                    # cell matches it. The row is a genuine OPENER, and dropping
                    # it cost the markdown answer at 124 sweep sites (R1-1).
                    #
                    # Judged POSITIONALLY, never by the number's value alone.
                    # Value equality is not sound here: a wrapped row's note may
                    # contain a number equal to a LATER row's id, and the fold
                    # can land that number at the start of a continuation, which
                    # is indented like every table continuation and so matched
                    # the later row EXACTLY -- handing back a row the reader
                    # never lit, which is #399's own payload (R2-1).
                    if not matched and _is_table_row(line):
                        matched = _number_opens_row(row, first_cell, table_column)
                    if matched:
                        placed = look
                        break
            if placed is not None:
                src = placed + 1
                # A fence body line and a table row occupy exactly one row.
                # Everything else — paragraphs, list items and quote lines alike
                # — may wrap; the starts-structural check above re-anchors the
                # next item or quote line when the wrap runs out.
                budget = 0 if (placed in covered or "|" in src_lines[placed]) else _LOOKAHEAD
                # Learn (or forget) where this table paints its first cell. Rich
                # left-aligns every row of a table on the same column, so a row
                # ALREADY placed on a table line -- the header, the divider, or
                # a body row the word path resolved -- reports the column an
                # opener must start at. Taken from the first such row and held
                # for the rest of the table: a continuation is indented past it
                # to a later cell's column, which is the distinction value
                # equality could not make (R2-1).
                if _is_table_row(src_lines[placed]):
                    if table_column is None:
                        table_column = _painted_column(row)
                else:
                    table_column = None  # left the table; the column no longer applies
        mapping.append(placed)
        current = placed
    return mapping


def _is_rule_like(text: str) -> bool:
    """A row whose visible content is only rule/divider characters."""
    body = text.strip().strip("|")
    return bool(body) and set(body) <= {"-", "─", ":", " ", "=", "_"}


def _first_cell(line: str) -> str:
    """A table source line's first cell, lowercased, escapes resolved.

    Splits on UNESCAPED pipes only, then unescapes to the literal characters the
    reader sees, so the returned text is what Rich paints in that cell rather
    than the markdown that produced it. Splitting on the raw character cuts a
    cell like ``ps aux \\| grep x`` in half and reports ``ps aux \\`` (R1-2).

    Scanned rather than matched with a ``(?<!\\\\)`` lookbehind, because that
    lookbehind is wrong one escape deeper: in ``\\\\|`` the backslash is itself
    escaped and the pipe IS a real delimiter, but the lookbehind sees a
    backslash and skips it (R2-2). A left-to-right scan consuming ``\\x`` as a
    unit cannot make that mistake, since the first backslash has already eaten
    the second by the time the pipe is read.
    """
    body = line.strip()
    # Only a LEADING delimiter is stripped by position; ``strip("|")`` would
    # also eat a trailing escaped pipe's character and shift the escape.
    if body.startswith("|"):
        body = body[1:]
    cell: list[str] = []
    i = 0
    while i < len(body):
        char = body[i]
        # Only ``\|`` and ``\\`` are escapes THIS parse cares about, and the
        # restriction matters: a backslash before anything else is literal
        # content, so treating every ``\x`` as an escape would eat the space in
        # a Windows path cell (``c:\``) and report ``c:``. Consuming ``\\`` as a
        # unit is what stops an escaped BACKSLASH from shielding the real
        # delimiter after it (R2-2).
        if char == "\\" and i + 1 < len(body) and body[i + 1] in "|\\":
            cell.append(body[i + 1])
            i += 2
            continue
        if char == "|":
            break
        cell.append(char)
        i += 1
    return "".join(cell).strip().lower()


#: A source line that is part of a markdown TABLE, as opposed to prose that
#: merely mentions a pipe. Requires a pipe with content on at least one side of
#: it in a line whose shape is a row: the numeric read is meaningless off a
#: table and must not be offered a prose line to match against (R2-4).
_TABLE_ROW_RE = re.compile(r"^\s{0,3}\|.*\|?\s*$|^[^|\n]*\|[^|\n]*\|")


def _is_table_row(line: str) -> bool:
    """Does this source line render as a table row rather than as prose?

    A pipe alone is not evidence: ``run a | b to filter`` is a paragraph. Rich
    only builds a table from lines that delimit cells, so requiring the row
    SHAPE keeps the numeric path off prose entirely instead of relying on the
    match downstream to fail (R2-4).
    """
    return "|" in line and bool(_TABLE_ROW_RE.match(line.strip()))


#: A rendered row whose first painted token is a bare number, carrying the
#: leading pad Rich puts before a table cell. Anchored so it cannot match a
#: number later in the row, and bounded to a digit run so a ``0.91`` score cell
#: and ``2026-01`` do not read as one.
#:
#: The leading ``[ \t]+`` only establishes that the row is INDENTED. That is a
#: weak gate on its own -- every table continuation, list continuation and quote
#: body satisfies it too -- so it is not by itself the thing that keeps this
#: path off a continuation. It excludes exactly one shape: a flush-left
#: number-led paragraph (``2026 roadmap``), which already reaches ``word`` and
#: must not get a second, looser way to claim a table line. Soundness comes from
#: the caller, which requires a table source line and POSITIONAL agreement via
#: ``_number_opens_row`` (R2-3).
_ROW_LEADING_NUMBER_RE = re.compile(r"^[ \t]+(\d{1,9})(?:\s|$)")


def _row_number(rendered_row: str) -> str:
    """The rendered row's leading bare number, or ``""`` if it has none."""
    match = _ROW_LEADING_NUMBER_RE.match(rendered_row)
    return match.group(1) if match else ""


def _painted_column(rendered_row: str) -> int:
    """The frame column at which ``rendered_row`` starts painting."""
    return len(rendered_row) - len(rendered_row.lstrip())


def _number_opens_row(rendered_row: str, first_cell: str, table_column: int | None) -> bool:
    """Does this rendered row OPEN the table line whose first cell is given?

    Positional, not value-based, and that distinction is the whole point. A bare
    number carries no evidence that a row opens anything: ids, years, versions,
    ranks and counts are routinely repeated inside the table's own prose, so a
    wrapped row whose note mentions a LATER row's id folds that id to the start
    of a continuation, where value equality matched the later row exactly and a
    whole-row take copied a row the reader never lit (R2-1, the shape #399 is
    about).

    Rich left-aligns every row of a table on one column, so an OPENER paints its
    first cell at that column while a continuation is indented past it to a
    later cell's column. ``table_column`` is that column, learned from a row
    already placed on this table. Both checks are required:

    * value equality, so the number still has to name THIS row's cell; and
    * column agreement, so a continuation carrying the same number is refused.

    Without a known column the answer is REFUSED rather than guessed. Losing the
    markdown answer is truthful and the copy path already answers such a row
    with the lit glyphs; substituting a different row is not.

    **The column is not always available, and the refusal above is the whole
    answer when it is not.** It is learned from a row ALREADY placed on one of
    this table's source lines, so on the FIRST row of a table it is still
    ``None`` and this returns False (R3-2). For a word-led header that is
    invisible — the header places itself through the word path and seeds the
    column for every row after it. For a NUMERIC-led header nothing ever seeds
    it, because ``_row_word`` reads the leading bare number as an ordered-list
    marker and the header's word comes from a later cell. Such a table keeps the
    lit glyphs instead of its markdown, which is the safe direction of the trade
    and deliberately left that way: the alternative is guessing a row from a bare
    number, which is exactly the substitution R2-1 removed.
    """
    if table_column is None:
        return False  # no positional evidence; refuse rather than guess
    if _painted_column(rendered_row) != table_column:
        return False  # indented past the first cell: a continuation, not an opener
    number = _row_number(rendered_row)
    if not number or not first_cell:
        return False
    head = first_cell.split()[0] if first_cell.split() else ""
    return head == number


#: One level of source quote marker, so the levels on a line can be COUNTED.
#: ``_QUOTE_RE`` matches the whole run at once and cannot report its depth.
_QUOTE_LEVEL_RE = re.compile(r">\s?")


def _quote_depth(source_line: str) -> int:
    """How many quote levels ``source_line`` opens with.

    Rich paints exactly one ``▌`` per level, so this is the number of bars on
    the row that are FURNITURE. Every bar after them is the model's own text.
    """
    match = _QUOTE_RE.match(source_line)
    if match is None:
        return 0
    return len(_QUOTE_LEVEL_RE.findall(match.group(0)))


def _painted_quote_width(row: str, depth: int) -> int:
    """Leading cells of ``row`` that are the quote bars RICH painted.

    Counted by DEPTH rather than matched by glyph, which is the distinction
    issue #392 is about. ``_RENDERED_QUOTE_PREFIX_RE`` is greedy and the glyph
    carries no evidence of who wrote it, so on ``> ▌ literal bar`` -- painted
    ``▌ ▌ literal bar`` -- it consumed BOTH bars and the copy lost the one the
    model typed. The source line says how many levels Rich opened, and that is
    the only thing that can tell a painted bar from a literal one.

    Stops early if the row runs out of bars, so a row whose furniture Rich
    painted differently than the depth suggests strips what is actually there
    rather than slicing into content.
    """
    width = 0
    for _ in range(depth):
        if not row.startswith("▌", width):
            break
        width += 1
        # The single space Rich puts after each bar. Guarded rather than assumed:
        # the last bar of a row whose content is empty carries no trailing space.
        if row.startswith(" ", width):
            width += 1
    return width


#: A rendered list marker and the space after it, with its leading indent. Only
#: the row that OPENS the item carries this; a continuation is indented to the
#: same column with spaces, which ``str.lstrip`` handles instead.
_RENDERED_LIST_PREFIX_RE = re.compile(r"^\s*(?:[•◦▪]|\d{1,9})\s+")

#: A painted BULLET glyph opening a rendered row. Unambiguous evidence that the
#: row opens its list item: Rich paints one only on an item's first row, and a
#: continuation is indented to the same column with plain spaces. Deliberately
#: EXCLUDES the ordered marker, which Rich paints as a bare number and which a
#: continuation can therefore imitate (``2026 was the year`` is the reported
#: shape) -- that ambiguity is what keeps the mapping in the decision below.
_RENDERED_BULLET_RE = re.compile(r"^\s*[•◦▪]\s")


def opens_list_row(row: str, *, mapping_opens: bool) -> bool:
    """Does this rendered row OPEN its list item, judged from the FRAME first?

    ``furniture_width``'s list branch needs to tell an item's marker row from
    its wrapped continuations, and the mapping is not a reliable witness: a
    continuation whose first word matches the NEXT list item's first word is
    mis-anchored to that item by ``align``'s structural scan, which makes the
    continuation look like an opener and the real marker row look like a
    continuation. A quoted list whose second item begins ``a second ...`` behind
    a continuation beginning ``a continuation ...`` reproduces it, and the
    resulting furniture was measured two cells short, so a whole-row gesture
    flipped format either side of the painted indent -- the exact wart issue
    #395 is about, one construct over.

    The frame answers directly where it can, and the two directions are not
    symmetric:

    * a painted BULLET is proof the row opens the item -- Rich paints it once;
    * no marker-shaped prefix AT ALL is proof it does not -- an opening row
      always carries one.

    Only the ambiguous middle (a leading bare number, which is either an ordered
    marker or a continuation that starts with a year) defers to the mapping,
    which is the case the mapping actually gets right.
    """
    if _RENDERED_BULLET_RE.match(row):
        return True
    if not _RENDERED_LIST_PREFIX_RE.match(row):
        return False
    return mapping_opens


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
        # By the source line's quote DEPTH, not by a greedy glyph match: the two
        # differ exactly when the model's own quoted text opens with ``▌``, and
        # the greedy match then strips a bar the reader typed (issue #392).
        width = _painted_quote_width(row, _quote_depth(remainder))
        # Peel the quote marker so the line can be re-tested for what it CONTAINS
        # — the same source line is both a quote line and a list item.
        remainder = _QUOTE_RE.sub("", remainder, count=1)

    if _LIST_RE.match(remainder):
        # The quote bar is already consumed, so this asks the LIST question of
        # the row's remainder. ``opens_line`` is the mapping's opinion and is
        # only consulted where the frame is genuinely ambiguous -- see
        # :func:`opens_list_row` for why a mis-anchored continuation otherwise
        # measures its furniture short and flips the paste format (issue #395).
        tail_row = row[width:]
        if opens_list_row(tail_row, mapping_opens=opens_line):
            match = _RENDERED_LIST_PREFIX_RE.match(tail_row)
            return width + (match.end() if match else 0)
        # A continuation is indented to the marker's column with plain spaces.
        tail = row[width:]
        return width + (len(tail) - len(tail.lstrip(" ")))
    return width


#: Inline markers Rich DROPS when it paints, so a rendered row is not a literal
#: substring of its source line. Stepping over these is what lets the positional
#: walk below place a row of ``**bold text** and `code```.
_INLINE_MARKUP = "*_`~"


def _is_word_char(char: str) -> bool:
    """A character CommonMark counts as word-interior for emphasis purposes."""
    return bool(char) and (char.isalnum() or char == "_")


def _is_intraword_underscore(source: str, i: int) -> bool:
    """``_`` at ``i`` sits inside a word, so CommonMark paints it LITERALLY.

    CommonMark deliberately exempts ``_`` from intraword emphasis so that
    ``snake_case`` survives, which ``*`` does not need. Treating such an
    underscore as a dropped marker made the walk unable to place a row that
    BEGINS on one -- exactly what Rich produces when it folds a long
    ``some_very_long_identifier_name`` mid-token -- so the line fell to the
    guess and every fold on it gained a space (review round 3, R3-2).
    """
    before = source[i - 1] if i else ""
    after = source[i + 1] if i + 1 < len(source) else ""
    return _is_word_char(before) and _is_word_char(after)


def _skip_inline_markup(source: str, i: int) -> int:
    """Advance past inline markers at ``i`` that render as nothing.

    Handles emphasis/code runs and a link's ``](target)`` tail, whose label is
    painted while the target is not. An intraword ``_`` is content, not a
    marker, and is left for the literal comparison to consume.
    """
    while i < len(source):
        char = source[i]
        if char == "_" and _is_intraword_underscore(source, i):
            break
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


def _skip_painted_nothing(source: str, i: int) -> tuple[int, str]:
    """Advance past everything at ``i`` that paints NO glyph.

    Returns the new offset and the WHITESPACE that was among what was skipped --
    at a fold that is the whole question, since the consumed whitespace is what
    a rejoin has to put back.

    The whitespace is returned VERBATIM rather than as a flag. Answering "was
    there a space?" made the rejoin put back exactly one, so a fold landing on a
    run of two or more spaces silently collapsed it: aligned columns in prose
    lost their alignment and a double-spaced sentence lost a space the document
    had (review round 4, R4-2). The walk knows what the run was, so reporting a
    plausible single space instead of the real one is the same
    guess-rather-than-know shape one level down from the refusals above.

    One definition of "paints as nothing" serves both places the walk needs it:
    the gap between two rows, and the tail after the last one. They were
    written separately, and the tail test compared the RAW residual, so a line
    ending in `` ` ``, ``**`` or ``](url)`` left a marker standing after the
    final row, failed to place, and fell to the guess -- which is how a
    mid-token fold on such a line gained a space that split a URL or a
    filesystem path (review round 3, R3-1; design round 3, D3-1).
    """
    skipped: list[str] = []
    while i < len(source):
        if source[i].isspace():
            skipped.append(source[i])
            i += 1
            continue
        nxt = _skip_inline_markup(source, i)
        if nxt == i:
            break
        i = nxt
    return i, "".join(skipped)


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


def locate_source_line(row_texts: list[str], source: str) -> str | None:
    """The one source line these rows are the full wrapping of, or ``None``.

    Used when :func:`align` placed none of the rows. That is absence of evidence
    from the aligner, not proof the rows share a line, so instead of loosening
    the rule the evidence is sought DIRECTLY: a line that the end-anchored walk
    can place these rows against is a line they demonstrably came from.

    ``None`` when no line places them or when several do -- an ambiguous answer
    is still no answer, and the caller must fall back to something that does not
    need to know. Without this, a selection over rows the aligner could not place
    was judged a single-line take on no evidence at all, which welded three
    separate lines of a fenced block into one (review round 3, R3-2).
    """
    if len(row_texts) < 2:
        return None
    found: str | None = None
    for line in source.split("\n"):
        if not line.strip():
            continue
        if _walk_folds([text.rstrip() for text in row_texts], line) is None:
            continue
        if found is not None:
            return None
        found = line
    return found


def separators_without_source(row_texts: list[str]) -> list[str]:
    """Last-resort separators when there is NO source line to walk at all.

    Reachable only when ``align`` placed none of the selected rows, so the
    markdown path has nothing to offer either and the rendered glyphs are the
    only truthful answer available. The name states the precondition that
    :func:`wrap_separators` refuses to guess past: this function is chosen
    deliberately by a caller that has checked there is no evidence, never
    reached by falling through.

    A terminal breaks CJK between any two characters because those scripts do
    not write spaces, so a space there is never something the user had; Latin
    text breaks at a space it then consumes, so one has to go back. Asking the
    adjacent characters IS a heuristic, and it is kept only for the case where
    the alternative is refusing to copy anything: it is what stops a Chinese or
    Japanese paragraph gaining an invented space at every fold (review round 2,
    R2-2; design round 2, D2-3).
    """
    stripped = [text.strip() for text in row_texts]
    return [
        _separator_for_scripts(stripped[i][-1:], stripped[i + 1][:1])
        for i in range(max(len(stripped) - 1, 0))
    ]


def _separator_for_scripts(left: str, right: str) -> str:
    """One fold's separator judged from the characters meeting at it."""
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


def wrap_separators(row_texts: list[str], source_line: str | None) -> list[str] | None:
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

    **``None`` means "I cannot know", and is not a separator.** When the rows
    cannot be placed against the line there is no evidence about what the fold
    consumed, and every previous attempt to answer anyway shipped a defect:
    guessing ``" "`` split a URL and a filesystem path, guessing from the
    adjacent scripts welded a compound, and both were silent because a
    plausible character looks like an answer. Three rounds of findings
    (R2-1, R2-2, R3-1, R3-2) are one root cause — the precondition above was
    documented but not enforced, so the caller's clipped or unplaced rows
    reached a fallback that invented a character rather than refusing.
    Refusing hands the decision back to the caller, which has a truthful
    answer available (the markdown source) that this function does not.
    """
    folds = max(len(row_texts) - 1, 0)
    if not folds:
        # No fold, nothing to rejoin: knowable without any evidence at all, so
        # a single-row sub-line take never depends on the walk.
        return []

    if source_line is None:
        return None
    # Rows keep their LEADING whitespace on purpose. When a fold's space does not
    # fit at the end of a row Rich moves it to the start of the next one instead
    # of consuming it, so that space is still on screen and still on the
    # clipboard; adding a separator too would double it.
    return _walk_folds([text.rstrip() for text in row_texts], source_line)


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
            # LITERAL FIRST, the same order :func:`_match_visible` uses and for
            # the same reason: a row that already carries the fold's whitespace
            # at its start consumed nothing, so the rejoin must add nothing.
            # Only when the row cannot be matched where the cursor stands is the
            # gap skipped and its whitespace put back.
            end = _match_visible(source_line, cursor, text)
            if end is not None:
                separators.append("")
                cursor = end
                continue
            gap, consumed = _skip_painted_nothing(source_line, cursor)
            end = _match_visible(source_line, gap, text)
            if end is None:
                separators = []
                break
            # The consumed run VERBATIM, not a single space standing in for it
            # (review round 4, R4-2). Newlines cannot appear here -- the walk
            # runs within one source line -- so this only ever restores the
            # spaces and tabs the fold ate.
            separators.append(consumed)
            cursor = end
        else:
            # End-anchor on the VISIBLE residual. Markers that paint nothing are
            # not content the rows failed to cover, so a line closing on
            # `` `code` ``, ``**bold**`` or ``[label](url)`` must still count as
            # fully consumed (review round 3, R3-1).
            tail, _ = _skip_painted_nothing(source_line, cursor)
            if source_line[tail:].strip() == "":
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

    # Rich REFLOWS a quote's consecutive ``>`` lines into one paragraph, so a
    # rendered row can carry text from two source lines and a selection that
    # lights one row has highlighted both. Extend the slice over the lines that
    # were merged into the picked one so the copy states what the reader lit.
    #
    # Bounded to lines the mapping placed NOWHERE, which is what makes the
    # extension answer the merge instead of the whole construct. A merged line
    # has no rendered row of its own -- its text was folded into a sibling's
    # rows -- so ``align`` never places it, and that absence is the evidence
    # this needs. A line that DID get its own row is a separate item the reader
    # did not light, and absorbing it is issue #416: every item of a quoted list
    # renders on its own row and every one was placed, so a one-row take on the
    # first of three copied all three. Measured across widths 30-72: the merged
    # paragraph leaves its second line unplaced at every width, while the quoted
    # list places all three items at every width -- so the narrower rule is
    # exactly as strong on the merge and silent on the list.
    #
    # ``placed_lines`` is read from the WHOLE mapping rather than from the
    # selected rows: a line placed on a row outside this selection is still a
    # line with its own row, and judging only the selection would call it
    # unplaced and re-absorb it.
    #
    # A quote line with no text of its own (a bare ``>`` separating two quoted
    # paragraphs) is excluded for the same reason, one step further: it paints
    # nothing, so it was never reflowed into anybody's row and is unplaced for
    # a different reason than a merged line is. Absorbing it appended a stray
    # ``>`` to the copy. This is the predicate ``align`` already applies to a
    # bare ``>`` in its structural scan.
    n = len(lines)
    placed_lines = {source for source in mapping if source is not None}
    last = max(picked)
    if _QUOTE_RE.match(lines[last]):
        j = last + 1
        while (
            j < n
            and _QUOTE_RE.match(lines[j])
            and j not in placed_lines
            and _strip_markup(lines[j])
        ):
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
