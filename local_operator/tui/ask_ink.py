"""Make a cited ask id in agent prose a link with its state beside it.

WHY THIS EXISTS
===============

An agent says what happened to a question by ID — "the old one, a-745e, is
withdrawn" — because that is the handle its own tool receipt gave it and the
only one that names one ask among several. In the TUI that id was plain prose:
no state, no route to the ask, and (when backticked) an ink that looked like a
link and did nothing (T-3).

So a token that RESOLVES to an ask of this conversation is painted in the
app's link style, followed by a no-break space and a ONE-CELL state mark —
``●`` waiting, ``✓`` answered, ``○`` closed without an answer. One cell on
purpose: a state change repaints in place and never reflows the paragraph or
shifts the scrollback (design §3).

WHAT IS LEFT ALONE
==================

* **Ids that do not resolve.** Ids are unique per conversation only, and the
  same ``a-745e`` exists in another session's log. Decoration resolves against
  THIS conversation's asks and nothing else: an unknown id stays exactly the
  text the model wrote.
* **Fenced code blocks.** A shell sample quoting an id is an argument being
  shown, not a citation — the same rule :mod:`local_operator.tui.link_markup`
  states for bare URLs, and it reuses that module's fence lexer so the two
  cannot disagree about which rows are code. Inline code IS decorated: a
  backticked id is the model citing one.
* **URLs.** ``https://example.com/a-745e`` is an address, not a citation; the
  token inside it is masked with the same scan ``link_markup`` uses to find
  URL bodies.

THE COPY CONTRACT
=================

The mark is inserted into the markdown SOURCE at render time, so the message
itself — what ``/copy`` reads — never carries it. A manual selection paints the
mark, and :func:`strip_marks` removes it on the way to the clipboard, because
the mark is state-at-paint-time, not something the model said.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.asks import store
from local_operator.tui import theme as theme_mod
from local_operator.tui.link_markup import _REF_DEF_RE, _fenced_spans
from local_operator.tui.link_targets import _URL_START_RE, _body_end

__all__ = [
    "ASK_ID_PATTERN",
    "MARKS",
    "decorate_source",
    "mark_for_status",
    "state_for_status",
    "strip_marks",
    "style_marks",
    "without_marks",
]

#: An ask id as :mod:`local_operator.asks.store` mints it: ``a-`` and four hex.
#:
#: The boundaries matter in both directions. A preceding ``.``/``/`` keeps the
#: token out of a path or a hostname (``https://x.test/a-745e``), and a
#: following word character keeps ``a-cafe``-shaped words inside prose from
#: being read as ids — ``a-cafe`` IS id-shaped, which is why resolution, not the
#: pattern, is what decides.
ASK_ID_PATTERN = r"a-[0-9a-f]{4}"

#: ``a-745e`` at a boundary: not preceded by a word character, a dot or a
#: slash, and not followed by a word character or a hyphen.
_ASK_TOKEN_RE = re.compile(rf"(?<![\w/.@-])({ASK_ID_PATTERN})(?![\w-])")

#: The mark each state wears. ONE cell each — the whole reason the state is a
#: glyph and not a word: a repaint must not re-wrap the paragraph.
MARKS: dict[str, str] = {
    "waiting": "●",
    "answered": "✓",
    "closed": "○",
}

#: The ink each mark takes (T-3's table). ``●`` is the list's own open glyph
#: (``ask_queue.STATUS_MARKS``), so the prose and the row agree by eye.
_MARK_INK: dict[str, str] = {
    "●": "warning",
    "✓": "success",
    "○": "muted",
}

#: A no-break space joins the id to its mark, so a fold can never separate the
#: pair and leave a mark looking like it belongs to the next word.
NBSP = "\u00a0"

#: A backtick RUN, matched after an id to find the close of an inline code span
#: the id sits inside.
_TICKS_RE = re.compile(r"`+")

#: The tail an ORPHAN mark must follow: an ask id at the end of the previous
#: row, with the no-break space optional.
#:
#: The space is optional because Rich TRIMS it at a fold: at 16 cells
#: ``word word a-745e`` / ``○ a-e53b● word`` is a real fold, with the pair's
#: space gone and the mark alone at column 0. That shape is still the
#: decorator's — nothing else puts a state glyph at a row start after an id.
#:
#: The ID must also sit at a real token boundary (the same left context
#: :data:`_ASK_TOKEN_RE` requires). A rule keyed on "ends with an id-shaped
#: word" would strip a mark after prose that merely runs into one —
#: ``xa-745e`` is a word, not an id — and only the token the decorator could
#: have resolved is allowed to orphan a mark.
_ORPHAN_TAIL_RE = re.compile(rf"(?<![\w/.@-]){ASK_ID_PATTERN}{re.escape(NBSP)}?$")

#: ``id`` + no-break space + mark, as it appears in the flattened rows.
_MARKED_RE = re.compile(rf"({ASK_ID_PATTERN}){NBSP}([●✓○])")

#: The decoration as SOURCE carries it: a backticked id takes its mark
#: after the closing backtick run, so the run is part of the match. Used by
#: :func:`strip_marks`, which must remove the pair in either shape.
_DECORATED_RE = re.compile(rf"({ASK_ID_PATTERN})(`*){NBSP}[●✓○]")

#: The state a queue status folds to. The list's own vocabulary is the source
#: (``asks/store``): an answered ask is DONE once its answer is delivered, but
#: "delivering" is the same fact mid-flight, and a lapsed or withdrawn ask is
#: closed with no answer — which is a different thing from "answered", and the
#: one distinction a reader came to the prose to settle.
_WAITING = frozenset({store.STATUS_OPEN})
_ANSWERED = frozenset({store.STATUS_ANSWERED, store.STATUS_LATE})
_CLOSED = frozenset(
    {
        store.STATUS_DECLINED,
        store.STATUS_DISMISSED,
        store.STATUS_EXPIRED,
        store.STATUS_TIMED_OUT,
        store.STATUS_WITHDRAWN,
    }
)


def state_for_status(status: str) -> str:
    """``"waiting"``/``"answered"``/``"closed"`` for an ask's queue status.

    Unknown statuses fold to ``closed`` rather than to waiting: a state this
    module has not been taught cannot ask the reader for anything.
    """
    if status in _WAITING:
        return "waiting"
    if status in _ANSWERED:
        return "answered"
    if status in _CLOSED:
        return "closed"
    return "closed"


def mark_for_status(status: str) -> str:
    """The one-cell mark an ask's queue status wears in prose."""
    return MARKS[state_for_status(status)]


def _scan_label(text: str, open_at: int) -> int | None:
    """Index just past the ``]`` that closes the label opened at ``open_at``.

    BRACKET-BALANCED and escape-aware, which the regex it replaces was not
    (review round 4): ``[nested [a-745e]](url)`` and ``[abc \\] a-745e](url)``
    both put a cited id inside link text, and a pattern stopping at the first
    ``]`` read the first as ``[nested [a-745e]`` — a construct that ends before
    the id is even closed — and decorated it, so a click ran the ask action AND
    the URL opener on the same cell.

    A blank line ends the search: a label may hold at most one newline in
    CommonMark, and no construct may reach across a paragraph break.
    """
    depth = 1
    index = open_at + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "\n" and text.startswith("\n\n", index):
            return None
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _scan_destination(text: str, open_at: int) -> int | None:
    """Index just past the ``)`` closing the destination opened at ``open_at``.

    Parentheses are balanced (CommonMark's one-level allowance, and deeper
    balance costs nothing here) and backslash escapes are skipped; a blank line
    ends it, for the same reason as :func:`_scan_label`.
    """
    depth = 1
    index = open_at + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "\n" and text.startswith("\n\n", index):
            return None
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _scan_autolink(text: str, open_at: int) -> int | None:
    """Index just past ``>`` when ``<…>`` is an autolink, else ``None``.

    CommonMark's shape: no whitespace and no ``<`` inside, and either a scheme
    (``<https://x>``) or an address (``<me@x.test>``).
    """
    index = open_at + 1
    length = len(text)
    while index < length and text[index] not in "<>\n":
        index += 1
    if index >= length or text[index] != ">":
        return None
    body = text[open_at + 1 : index]
    if not body or " " in body or "\t" in body:
        return None
    if ":" not in body and "@" not in body:
        return None
    return index + 1


def _link_constructs(text: str) -> list[tuple[int, int]]:
    """Every Markdown link/autolink range that starts outside a fence.

    ONE left-to-right pass (review round 4's perf constraint): each construct
    consumes the text it covers, and a ``[`` that never closes costs one
    character of advance, so the walk is linear in the text. This is what the
    masking needs to be Markdown-aware rather than pattern-shaped — the shapes
    that matter are the ones a *reader* sees as a link, and those are the ones
    a regex over ``[^]]*`` cannot see.
    """
    spans: list[tuple[int, int]] = []
    fences = _fenced_spans(text)

    def in_fence(position: int) -> bool:
        return any(low <= position < high for low, high in fences)

    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "<":
            end = _scan_autolink(text, index)
            if end is not None and not in_fence(index):
                spans.append((index, end))
                index = end
                continue
            index += 1
            continue
        if char == "[" and not in_fence(index):
            label_end = _scan_label(text, index)
            if label_end is not None and label_end < length:
                if text[label_end] == "(":
                    end = _scan_destination(text, label_end)
                    if end is not None:
                        spans.append((index, end))
                        index = end
                        continue
                elif text[label_end] == "[":
                    ref_end = _scan_label(text, label_end)
                    if ref_end is not None:
                        spans.append((index, ref_end))
                        index = ref_end
                        continue
        index += 1
    for match in _REF_DEF_RE.finditer(text):
        if not in_fence(match.start()):
            spans.append((match.start(), match.end()))
    return spans


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """Ranges a cited id must NOT be decorated inside: fences, links and URLs.

    LINK TEXT IS MASKED, not just its target (review N-4). An id inside
    ``[a-745e](url)`` was given BOTH a ``@click`` action and the link style, and
    a real click dispatches ``Widget._on_click`` (the action) and
    ``TranscriptBlock.on_click`` (the URL) on the same widget, so one click both
    navigated and jumped the TUI to the ask. The rule is the module docstring's:
    links stay plain. Links are read with :func:`_link_constructs`, a
    Markdown-aware scanner, because the shapes that matter (nested brackets,
    escapes, reference definitions) are exactly the ones a regex mis-reads.
    """
    spans = list(_fenced_spans(text))
    spans.extend(_link_constructs(text))
    scan = 0
    while True:
        match = _URL_START_RE.search(text, scan)
        if match is None:
            break
        body_to = _body_end(text, match.end())
        if body_to == match.end():
            scan = match.end()
            continue
        spans.append((match.start(), body_to))
        scan = body_to
    return sorted(spans)


def decorate_source(text: str, states: Mapping[str, str]) -> str:
    """``text`` with every resolvable ask id followed by NBSP and its mark.

    ``states`` maps an ask id of this conversation to its queue status; an id
    absent from it is left exactly as written, which is what keeps the same id
    in another conversation's log from being decorated here.

    Idempotent: an id that already carries a mark is matched by its own
    ``NBSP + mark`` suffix, so a re-render — which the streaming path performs
    on every flush and the resize path per width — cannot stack marks. For a
    BACKTICKED id the suffix sits after the closing backtick, so that is where
    the check looks: reading it directly after the id let a backticked id
    collect a second mark on every flush (review minor).
    """
    if not states or "a-" not in text:
        return text
    masked = _masked_spans(text)
    out: list[str] = []
    cursor = 0
    for match in _ASK_TOKEN_RE.finditer(text):
        start, end = match.span(1)
        if start < cursor or any(low <= start < high for low, high in masked):
            continue
        status = states.get(match.group(1))
        if status is None:
            continue
        # A BACKTICKED id takes its mark OUTSIDE the code span. Inserted inside,
        # the mark would be painted in the code ink and copied as part of the
        # sample — the id is what the model cited, the state is this surface's.
        insert_at = end
        close = _TICKS_RE.match(text, end)
        if close is not None and text[:start].endswith(close.group(0)):
            insert_at = close.end()
        if text[insert_at:].startswith(NBSP):
            # Already decorated by an earlier pass.
            continue
        out.append(text[cursor:insert_at])
        out.append(NBSP + mark_for_status(status))
        cursor = insert_at
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)


def strip_marks(text: str) -> str:
    """``text`` with the ``NBSP + mark`` decoration removed from each id.

    The clipboard is the model's words, not this surface's state glyph. Two
    shapes have to go: the PAINTED form (``a-745e`` + NBSP + mark) and the
    SOURCE form, where a backticked id carries the mark after its closing
    backtick (`` `a-745e` `` + NBSP + mark). One pattern covers both — the
    backtick run is optional and kept — so a round trip through
    :func:`decorate_source` returns the original text exactly (review minor).
    """
    if NBSP not in text:
        return text
    return _DECORATED_RE.sub(r"\1\2", text)


def style_marks(text: Text, *, action: Callable[[str], str | None]) -> None:
    """Paint every decorated id in the link style and make it a jump.

    ``action`` maps an ask id to the Textual action that opens it, or ``None``
    for an id this surface will not route. The action rides as style META
    (``@click``), never as a rich ``link=``: a rich link makes Textual hand the
    terminal an OSC-8 URL, and a ``lop-ask:`` scheme would be offered to the
    desktop as something to open (design §5.4).

    Styles are APPENDED, so they outrank the ink a code span or an emphasis
    painted under the id — a backticked id is the case that needed it.
    """
    plain = text.plain
    if NBSP not in plain:
        return
    signal = theme_mod.semantic_color("signal")
    for match in _MARKED_RE.finditer(plain):
        target = action(match.group(1))
        if target is None:
            continue
        text.stylize(
            Style(color=signal, underline=True, meta={"@click": target}),
            match.start(1),
            match.end(1),
        )
        ink = _MARK_INK.get(match.group(2))
        if ink is not None:
            text.stylize(
                Style(color=theme_mod.semantic_color(ink), meta={"@click": target}),
                match.start(2),
                match.end(2),
            )


def without_marks(row: str, *, start: int, end: int, previous_row: str | None = None) -> str:
    """``row[start:end]`` (CELL columns) with every decoration cell removed.

    The glyph copy path in ``AssistantBlock.get_selection`` hands the clipboard
    the cells a drag covered, and the decoration is cells the reader never typed
    — a drag that begins, ends or passes through one must not carry it (QA
    round 1, Q2): ``a-745e○, i``, ``745e○`` and ``○, is withdrawn;`` all came
    out of the same paragraph.

    Two shapes are removed, both derived from how the line was BUILT rather than
    from a pattern over the result:

    * the pair itself — the no-break space and the mark that follows it. The
      no-break space exists only in a decoration, so the pair is unambiguous
      even when the drag starts inside the id.
    * the ORPHAN at a row's first cell: rich may fold between the no-break
      space and the mark, leaving the mark at column 0 of the next row with no
      pair to match. It is recognised by the previous row ending in the
      no-break space, which is the only way that cell is ever produced.

    ``end`` follows the caller's convention: ``-1`` means end of row.
    """
    limit = cell_len(row) if end < 0 else end
    removed: list[tuple[int, int]] = []
    for match in _MARKED_RE.finditer(row):
        # The pair starts at the NO-BREAK SPACE, one cell before the mark the
        # pattern's second group names. Removing only the mark left the space
        # inside a drag that crossed it (``a-745e\xa0,``).
        mark_cell = cell_len(row[: match.start(2)])
        removed.append((mark_cell - 1, mark_cell + cell_len(row[match.start(2) : match.end(2)])))
    # ``rstrip(" ")`` and not a bare ``rstrip()``: NBSP IS whitespace to
    # ``str.rstrip``, so the bare form ate the very cell it was looking for and
    # a wrapped pair was never recognised (review N-1).
    #
    # AND the row has to end in an ID before the no-break space: a mark after
    # ANY NBSP was dropped, so a drag over a fold that landed after French
    # ``fait\u00a0✓`` lost a glyph the model wrote (review round 3, minor).
    # The decoration only ever inserts ``id + NBSP + mark``, so an id is the
    # only tail that can have produced an orphan.
    if (
        previous_row is not None
        and row[:1] in MARKS.values()
        and _ORPHAN_TAIL_RE.search(previous_row.rstrip(" ")) is not None
    ):
        removed.append((0, 1))
    # A row that ENDS in the no-break space is the other half of a folded pair:
    # the decoration always inserts NBSP + mark together, so a trailing NBSP at
    # end-of-row has no mark after it in this row, and the reader did not type
    # it either (review N-1, the ``id ends the row`` position).
    trimmed_row = row.rstrip(" ")
    if trimmed_row.endswith(NBSP):
        tail = cell_len(trimmed_row) - 1
        removed.append((tail, tail + 1))
    out: list[str] = []
    cell = 0
    for char in row:
        width = cell_len(char)
        # CELL columns throughout (review minor): a char slice read a wide
        # glyph's 2 cells as one, so a CJK prefix shifted every later column.
        # A glyph is kept when its span meets the selection's.
        if (
            cell + width > start
            and cell < limit
            and not any(low <= cell < high for low, high in removed)
        ):
            out.append(char)
        cell += width
        if cell >= limit:
            break
    return "".join(out)
