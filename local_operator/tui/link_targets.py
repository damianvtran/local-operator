"""The URLs in a conversation, for the action that hands one to the browser.

WHY THIS EXISTS
===============

The transcript paints OSC-8 hyperlinks and Ghostty honours them, but the
terminal never sees the click: Textual's driver claims the mouse at startup and
a terminal that is reporting mouse events to an application does not run its own
click-to-open gesture (``textual/drivers/linux_driver.py`` writes ``\\x1b[?1000h``
SET_VT200_MOUSE and ``\\x1b[?1003h`` SET_ANY_EVENT_MOUSE). Holding shift to
bypass that is the terminal's gesture, not ours — Ghostty documents that the
application cannot even detect it (``xtshiftescape``) — so the app has to offer
its own route to the URL. This module is the part of that route that decides
WHICH strings are URLs.

THE TEXT IS THE SOURCE, NOT THE RENDERED FRAME
==============================================

Extraction runs over ``block.text()``, which every transcript block returns as
the text it was BUILT from — for an assistant answer that is the message's
markdown source. Two reasons, and the first is a hard one:

* **A bare URL has no link span at all.** Rich's markdown renderer sets
  ``Style(link=...)`` for ``[text](url)`` and for the explicit autolink
  ``<url>``, and nothing for ``https://example.com`` written plainly —
  verified against rich 15.0.0 in this tree, where a message holding both a
  markdown link and a bare URL produced exactly one span and one hyperlink.
  Measured on the real thing: `PLAIN` kept the bare URL as ordinary text with
  no blue and no underline. A picker built from the spans would therefore miss
  the shape users paste most, and would miss it silently.

* Reading the source needs no render, no width and no mounted widget, so the
  list is the same whether the block is 20 or 200 columns wide, and it can be
  filtered and tested without a pilot. This is the same argument
  ``widgets/_copy_markdown.py`` makes for copying markdown rather than the
  frame: the frame is lossy in both directions, the source is not.

SCHEME SAFETY
=============

Every URL here is handed to the user's browser, so only ``http``/``https``
pass: a block's text is attacker-influenceable (a tool result, a peer message,
a model's answer), and an opener that forwards whatever it is given is a
one-token file path or ``javascript:`` away from being an execution vector.
The posture matches the mobile renderer's (``mobile/web/src/components/
markdown.tsx``: "Links are forced https? and open in a new tab"). The guard is
:func:`is_openable`, applied to every capture and again by the caller at the
boundary — one function, two call sites, no second spelling of the rule.

WHAT COUNTS AS A URL
====================

Three rules, each stated once:

* :func:`_body_end` — the characters a URL is made of, and where it stops,
  shared by BOTH captures so a markdown link's target and a bare URL are cut at
  the same place. A parenthesised run is included only when it is BALANCED, at
  any depth, which is what keeps ``…/wiki/Foo_(bar)`` and ``…/a_(b_(c))_d``
  whole. Review round 1 found what happens when that rule lived on one path
  only: the markdown capture stopped at the link's own terminator, and the
  picker painted two rows for one link with the cursor on the truncated one
  (MAJOR-1). Review round 2 found the other half — a rule written as a
  nesting-level PATTERN rather than as a balance is a rule with a silent edge,
  one level past its bound.
* :func:`_trim` — the punctuation that belongs to the prose or the emphasis
  AROUND the URL (``…/docs.``, ``**…/x**``), applied to both captures for the
  same reason.
* :func:`is_openable` — the scheme.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: What a URL begins with, in ONE literal: the guard that decides what may
#: reach the browser and the finder that decides what is offered are then the
#: same characters rather than two spellings of "http or https" that can drift.
_SCHEME = r"https?://"

#: The only schemes that may reach the browser. Deliberately a prefix test on
#: the raw string rather than ``urlparse(...).scheme``: ``urlparse`` accepts
#: almost anything as a scheme, and the point here is to REFUSE, not to parse.
_OPENABLE_RE = re.compile(rf"^{_SCHEME}", re.IGNORECASE)

#: Where a bare URL starts.
_URL_START_RE = re.compile(_SCHEME, re.IGNORECASE)

#: Characters that end a body outright: the brackets and quotes a URL inside
#: prose is very often wrapped in. ``<>`` because a URL in angle brackets is an
#: autolink and its closing ``>`` is punctuation; the quotes because the URL is
#: usually inside them. Whitespace is left to :meth:`str.isspace` in the scan,
#: so every character Python counts as whitespace ends a URL rather than the
#: four an ASCII list would name.
_BODY_STOP = frozenset("<>\"'`")


def _body_end(text: str, start: int) -> int:
    """The index just past the URL body that begins at ``start``.

    WHY THIS IS A SCAN AND NOT A PATTERN, because the next reader will reach
    for the pattern. Python's ``re`` cannot match arbitrarily nested
    parentheses — the language has no recursion — so the balance has to be
    either written out for a FIXED number of levels or walked. The fixed
    pattern is the trap: it matches ``…/Foo_(bar)`` and then says nothing past
    its bound, so a URL one level deeper is not refused, it is TRUNCATED, and
    the truncated string is a legal ``https://`` URL that the picker paints as
    a row and ``enter`` opens as a 404. That is review round 2's BLOCKER: the
    round-1 fix added a parenthesised alternative that could match ONE level,
    and it turned ``https://a.test/a_(b_(c))_d`` — which the round-1 head got
    right — into ``https://a.test/a_``. Nesting depth belongs to the text and
    not to the rule, so NO bound is the right bound; the balance is, and a loop
    is what can hold it.

    The balance is the rule the pattern stated: a ``(`` opens a run that must
    close, and a run that never closes is not URL text — that ``(`` and
    everything after it belong to the prose, so the body ends before it
    (``…/x_(y`` → ``…/x_``). A ``)`` reached with nothing open ends the body
    there, which is what makes ``[label](url)``'s own terminator a terminator
    and keeps the closer of ``(see …)`` out of the URL.

    Shared by both captures, and it has to be: ``[label](…)``'s own terminator
    is a ``)``, so a body that stopped at the FIRST ``)`` truncated the target
    — and because the bare finder then found the correct form at the same
    offset, the picker painted TWO rows for one link with ``❯`` on the
    truncated one, so a plain ``enter`` opened a 404 (review round 1,
    MAJOR-1). One shared body is what makes the two paths agree by
    construction.
    """
    depth = 0
    # Where the outermost still-open ``(`` sits, or -1 when nothing is open.
    unclosed_at = -1
    index = start
    limit = len(text)
    while index < limit:
        char = text[index]
        if char in _BODY_STOP or char.isspace():
            break
        if char == "(":
            if depth == 0:
                unclosed_at = index
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
            if depth == 0:
                unclosed_at = -1
        index += 1
    return unclosed_at if depth else index


#: ``[label](`` — the markdown link's head. Its TARGET is then read from the
#: text by :func:`_body_end`, exactly as a bare URL is, so the two paths cannot
#: cut the same characters in two places. Matched against the SOURCE, so the
#: label may be anything Rich would render.
_MARKDOWN_OPEN_RE = re.compile(r"\[[^\]]*\]\(\s*", re.IGNORECASE)

#: What may follow a markdown target: an optional title, then the ``)`` that
#: closes the link. ``[^)]`` still excludes a ``)`` from the title, as before.
_MARKDOWN_TAIL_RE = re.compile(r"(?:\s+[^)]*)?\)")

#: Trailing characters that belong to the SENTENCE (or to emphasis), not the
#: URL. A URL written at the end of a clause arrives as ``…/docs.`` and opening
#: that exact string is a 404; the period is the most common case by far, and
#: the rest are the closers a human types after a link in prose.
#:
#: ``*`` and ``~`` are here for the emphasis marks a model writes around a URL
#: (``**https://a.test/x**``) — review round 1, MAJOR-2. They cost a real URL
#: that ENDS in one of those characters, which is rarer than the bold link.
#: Unbalanced ``)`` is NOT here: :func:`_body_end` cannot return a body that
#: ends on one — it breaks before a ``)`` with nothing open, and backs up past a
#: ``(`` that never closes — which is why there is no second trimmer for it.
_TRAILING_JUNK = ".,;:!?'\"*~"

#: What each side of the conversation is called in the picker's hint column.
_AGENT = "agent"
_USER = "you"


@dataclass(frozen=True)
class LinkTarget:
    """One URL, with what the picker needs to describe it.

    Frozen because the list is a SNAPSHOT: it is built once when the picker
    opens and never rebuilt, for the reason ``copy_targets.CopyTarget`` is
    frozen — a message settling under an open picker would renumber the rows
    beneath it, including the one the user is aiming at.
    """

    url: str
    #: ``"agent"`` or ``"you"`` — which side of the conversation it came from.
    sender: str
    #: 1 for the MOST RECENT message that carried a URL, counting up with age.
    #: An ordinal rather than a message id because the picker never navigates
    #: back to the message: the row's job is to tell two same-host URLs apart.
    rank: int


def is_openable(url: str) -> bool:
    """Whether ``url`` may be handed to the browser.

    The single spelling of "http or https", called by extraction and again by
    the opener. Case-insensitive because ``HTTPS://`` is a legal URL and a
    guard that only refused lowercase would be a guard with a hole in it.
    """
    return bool(_OPENABLE_RE.match(url))


def _trim(url: str) -> str:
    """Drop the punctuation that belongs to the surrounding prose.

    Applied to BOTH patterns' captures, so a markdown link and a bare URL that
    point at the same characters come out as the same string — which is also
    what lets the dedupe collapse them to one row.

    Nothing here handles parentheses. A balanced run is part of the URL by
    :func:`_body_end`, and an unbalanced closer can never reach this function:
    the scan breaks before a ``)`` with nothing open and backs up past a ``(``
    that never closes, so a capture cannot end on one. The trimmer this used to
    carry was dead code the moment the body learned the rule, and two places
    deciding the same question is how they come to disagree.
    """
    while url and url[-1] in _TRAILING_JUNK:
        url = url[:-1]
    return url


def extract_links(text: str) -> list[str]:
    """Every openable URL in ``text``, in the order it appears, deduped.

    Markdown links and bare URLs are found by two heads and then merged on
    POSITION rather than concatenated: a message that says "see [the
    docs](https://a.test/x) or https://b.test/y" has an order the reader can
    see, and running one head's results after the other's would list it wrong.
    A markdown link's target is also inside the bare head's reach, so the merge
    is what keeps each URL to one entry.

    Both captures take their body from the SAME :func:`_body_end` and then go
    through the SAME :func:`_trim`, which is what makes the two paths produce
    the same string for the same characters — the property the dedupe depends
    on, and the one whose absence put a truncated row under the cursor in
    review round 1 (MAJOR-1).

    The bare pass resumes AFTER the body it just read rather than after the
    ``https://`` that started it, so a URL containing another one —
    ``…/r?u=https://b.test/y``, the shape of every redirector and share link —
    stays ONE row instead of being found a second time from its own interior.

    Both captures are also filtered through :func:`is_openable`, though
    :data:`_SCHEME` already starts the body: the rule that only http(s) may
    become a ROW is stated rather than implied by a pattern, and a later edit
    to the body cannot quietly admit a scheme.
    """
    found: list[tuple[int, str]] = []
    for match in _MARKDOWN_OPEN_RE.finditer(text):
        # The target begins where the head ends, past any space after the ``(``.
        body_at = match.end()
        body_to = _body_end(text, body_at)
        # No body at all is ``[label]()``; a body the link's own ``)`` does not
        # follow is prose that happens to look like a link's head. Either way
        # this is not a markdown link.
        if body_to == body_at or not _MARKDOWN_TAIL_RE.match(text, body_to):
            continue
        url = _trim(text[body_at:body_to])
        if is_openable(url):
            found.append((body_at, url))
    cursor = 0
    while True:
        match = _URL_START_RE.search(text, cursor)
        if match is None:
            break
        body_to = _body_end(text, match.end())
        if body_to == match.end():
            # ``https://`` with nothing a URL can be made of after it.
            cursor = match.end()
            continue
        url = _trim(text[match.start() : body_to])
        if is_openable(url):
            found.append((match.start(), url))
        cursor = body_to
    # Ranked by position, then deduped by URL. The second step is load-bearing
    # rather than tidiness: a markdown link's TARGET is inside the bare head's
    # reach, so `[docs](https://a.test/x)` arrives twice, once from each head —
    # once at the label's offset and once at the target's — and the earlier
    # offset is the one the reader sees first.
    seen: set[str] = set()
    urls: list[str] = []
    for _, url in sorted(found, key=lambda pair: pair[0]):
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _source_label(block: object) -> str | None:
    """``"you"``/``"agent"`` for a block, or ``None`` if it is not a message.

    ``isinstance`` against the two real classes, because the question here IS
    "which widget is this" — ``copy_targets`` reaches for a Protocol instead
    only because it must accept a mixture it cannot import, and a Protocol
    would answer the wrong question here: ``text()`` and ``is_finalized()`` are
    present on EVERY transcript block, so "is a message" is not something
    attribute presence can decide.

    Anything else — a tool card, a system notice, a peer receipt — is skipped.
    A URL in tool output is not a link the user was reading, and listing it
    would bury the three they were.
    """
    # Imported here, not at module scope: this module is read by the picker and
    # by the app, and the widgets import each other in a chain (`assistant`
    # imports `transcript`). A deferred import keeps this module's position in
    # that chain a caller's business rather than a rule.
    from local_operator.tui.widgets.assistant import AssistantBlock
    from local_operator.tui.widgets.transcript import UserBlock

    if isinstance(block, UserBlock):
        return _USER
    if isinstance(block, AssistantBlock):
        return _AGENT
    return None


def _guarded(block: object, name: str) -> object:
    """``block.<name>()``, or ``None`` if it cannot answer.

    A ``runtime_checkable`` protocol tests attribute PRESENCE only, so an object
    can satisfy the check and still raise on the call — ``copy_targets`` records
    the live ``/copy`` this cost once already (``'UserBlock' object has no
    attribute 'is_truncated'``). An action that opens a browser must not take a
    turn down either, and a block that cannot answer is not a message.
    """
    method = getattr(block, name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:  # noqa: BLE001 — see the docstring: presence is not callability
        return None


def build_link_targets(blocks: Sequence[object]) -> list[LinkTarget]:
    """The picker's rows: every URL in the conversation, newest message first.

    ``blocks`` is the transcript's own list, walked in REVERSE, so "most recent"
    means last-appended — the same rule ``build_copy_targets`` uses and for the
    same reason: a resumed conversation replays its history into the same
    column, so append order is the order the reader sees.

    Only FINALIZED messages are read. A streaming answer is still changing, and
    the half-written tail of a URL is not a link the user can open — listing it
    would put a string on screen that fails when chosen. (``/copy`` makes the
    same cut for the same reason.)

    One entry per URL, not per mention: the list answers "which link do I want",
    and the same URL twice is a row the user has to think about for nothing. The
    FIRST mention wins, which — walking newest-first — means the most recent
    one, so the row's rank describes where the reader most likely saw it.
    """
    seen: set[str] = set()
    targets: list[LinkTarget] = []
    rank = 0
    for block in reversed(list(blocks)):
        sender = _source_label(block)
        if sender is None:
            continue
        if _guarded(block, "is_finalized") is not True:
            continue
        text = _guarded(block, "text")
        if not isinstance(text, str):
            continue
        urls = extract_links(text)
        if not urls:
            continue
        rank += 1
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            targets.append(LinkTarget(url=url, sender=sender, rank=rank))
    return targets
