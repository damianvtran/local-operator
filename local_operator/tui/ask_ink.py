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

from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.link_markup import _fenced_spans
from local_operator.tui.link_targets import _URL_START_RE, _body_end

__all__ = [
    "ASK_ID_PATTERN",
    "MARKS",
    "decorate_source",
    "mark_for_status",
    "state_for_status",
    "strip_marks",
    "style_marks",
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

#: ``id`` + no-break space + mark, as it appears in the flattened rows.
_MARKED_RE = re.compile(rf"({ASK_ID_PATTERN}){NBSP}([●✓○])")

#: The state a queue status folds to. The list's own vocabulary is the source
#: (``asks/store``): an answered ask is DONE once its answer is delivered, but
#: "delivering" is the same fact mid-flight, and a lapsed or withdrawn ask is
#: closed with no answer — which is a different thing from "answered", and the
#: one distinction a reader came to the prose to settle.
_WAITING = frozenset({"open"})
_ANSWERED = frozenset({"answered", "late"})
_CLOSED = frozenset({"declined", "dismissed", "expired", "timed_out", "withdrawn"})


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


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """Ranges a cited id must NOT be decorated inside: fences and URLs."""
    spans = list(_fenced_spans(text))
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
    on every flush and the resize path per width — cannot stack marks.
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
        following = text[end:]
        if following.startswith(NBSP) and len(following) > 1:
            # Already decorated by an earlier pass.
            continue
        # A BACKTICKED id takes its mark OUTSIDE the code span. Inserted inside,
        # the mark would be painted in the code ink and copied as part of the
        # sample — the id is what the model cited, the state is this surface's.
        insert_at = end
        close = _TICKS_RE.match(text, end)
        if close is not None and text[:start].endswith(close.group(0)):
            insert_at = close.end()
        out.append(text[cursor:insert_at])
        out.append(NBSP + mark_for_status(status))
        cursor = insert_at
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)


def strip_marks(text: str) -> str:
    """``text`` with the painted ``NBSP + mark`` suffix removed from each id.

    The clipboard is the model's words, not this surface's state glyph.
    """
    if NBSP not in text:
        return text
    return re.sub(rf"({ASK_ID_PATTERN}){NBSP}[●✓○]", r"\1", text)


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
