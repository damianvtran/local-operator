"""Bounded receipt for a ``data:`` URI that arrived inside a message BODY.

WHY THIS EXISTS. A Team Bridge body is free-form text a member (or an agent
acting for one) pasted or encoded, and a ``data:image/…;base64,…`` URI is a
legal way to write an image inline. The desktop client already collapses such
a run into a bounded receipt before Markdown ever sees it
(``inline-data-receipt.tsx``), but CORE was handing the raw body straight to
the model through the history and ``take_up`` results: one 600 KB string of
base64 that is unreadable, unbounded, and costs context on every read of that
page. That is the operator's original complaint, second door.

WHAT IT IS NOT. This is a REPRESENTATIONAL filter, not a security control. It
does not strip an injection (a body's text is still body text and still travels
under ``BRIDGE_UNTRUSTED_CLAUSE``), it does not admit or screen anything, and
it does not touch the wire, the hub row, ``body_hash`` or ``details``. Message
rows are immutable by contract and ``body_hash`` is the screening commitment
every admission binds to, so rewriting a body anywhere but at the model
boundary would invalidate verdicts that were computed over the original bytes.
That is why this runs on the way INTO model context and nowhere else.

WHY IT NEVER DECODES. The bytes are worthless here — the model is being told
they were withheld, not shown them — so the decoded size is derived
arithmetically from the base64 length. Decoding a sender-controlled payload
would be the one part of this that could be made expensive or hostile.

THE SHAPE, AND WHY IT IS THIS SHAPE. Rounds 1–3 of review found the same class
of defect three times: a matcher that tried to UNDERSTAND the header could be
defeated by a header shape it did not anticipate (an unbounded quantifier that
went quadratic; a MIME-wrapped, URL-safe or uppercase spelling that kept the
whole blob; then a long parameter list, an over-long media type, and a CRLF
right after the marker, each of which left the payload unmarked because the
header grammar refused to match and the code read "no match" as "no URI").

The lesson is that failing OPEN is unacceptable and validating a header is not
this function's job, so there is no header grammar left to fail. The scanner
now:

1. finds the LITERAL ``;base64,`` token, case-insensitively;
2. walks back a bounded distance for the ``data:`` that introduces it (the
   media type is decoration for the receipt, never a gate);
3. skips whitespace and line breaks after the marker;
4. consumes a base64 run with a monotonic alphabet scan — ``A-Z a-z 0-9 + / =
   - _`` plus folding whitespace — until a character outside that alphabet.

Nothing in that sequence can be defeated by parameter count, parameter length,
media-type length, whitespace before or after the marker, or casing: those
inputs change what the RECEIPT says, never whether the payload is withheld.

WHY THE COST IS LINEAR. The input is attacker-sized and delivered synchronously
on an ``async`` tool path, so a scan that stalls the event loop is a worse
defect than the blob it removes. Every step above only ever moves a cursor
forward over a region it then skips: a marker match costs O(1), the backward
``data:`` search is amortised over the gap since the previously consumed run,
and the payload walk never revisits a character. Total work is therefore linear
in the body, which is also why nothing is truncated — a body with no data URI is
returned as the SAME object at any length, including one that merely mentions
``metadata:``.
"""

from __future__ import annotations

import re
import string
from typing import Any

__all__ = [
    "bound_inline_data_payload",
    "bound_inline_data_uris",
    "inline_data_marker",
]

#: Media types with no ``/`` are still legal in the header (the RFC allows an
#: omitted type, which defaults to ``text/plain``).
_DEFAULT_MIME = "text/plain"

#: The one token that starts a base64 data URI's payload. Deliberately a literal
#: rather than a parsed header: see the module docstring. ``[ \t]*`` inside it is
#: bounded and only tolerates spacing around the literal.
_BASE64_MARKER = re.compile(r";[ \t]*base64[ \t]*,", re.IGNORECASE)

#: Cheap pre-test so the no-op path (the overwhelmingly common one) allocates
#: nothing: a lowercased copy of a 2 MB body would be a 2 MB allocation per call.
_ANY_DATA = re.compile(r"data:", re.IGNORECASE)

#: How far back from ``;base64,`` the introducing ``data:`` may sit. Generous for
#: any plausible header (a media type plus many parameters) and bounded so the
#: backward search cannot walk a whole body; past it the marker is treated as a
#: mention in prose, not a URI.
_MAX_HEADER_BACK = 8192

#: Payload alphabet. Both standard and URL-safe base64, plus padding: a sender
#: may paste either spelling, and ``-_`` is what URL-safe encoders emit.
_PAYLOAD_CHARS = frozenset(string.ascii_letters + string.digits + "+/-_=")

#: Whitespace that can separate folded lines of ONE payload.
_FOLD_WS = " \t\r\n"

#: A fold is resumed only when the continuation line carries at least this many
#: payload characters. 16 is chosen low on purpose so a NARROW wrap (40 columns,
#: or a terminal-width paste) still continues — a high floor silently stopped at
#: the first line of a 40-column blob and left the rest behind with a receipt
#: claiming a wrong exact size.
#:
#: THE TRADEOFF, stated because there is no way to remove it: a payload followed
#: by a newline and a word of 16+ base64-alphabet characters will swallow that
#: word (``…base64,AAAA\ninternationalization``). A shorter word — ``thanks``, a
#: closing fence, a sentence — is prose and survives. Over-deleting a sender's
#: own words is a real cost, and the floor is the only lever: it is set at the
#: point where wrapped lines outnumber the prose words it could eat, and the
#: final fragment of a wrap narrower than this is left in place and reported as
#: ``at least`` rather than silently miscounted.
_FOLD_MIN_LINE = 16

#: How much of the sender-chosen media type the marker echoes. This is
#: SENDER-CHOSEN text going into model content, so it is bounded for display, not
#: just for cost — and clamped BELOW the 64-character run length a blob detector
#: (including this repo's own tests) uses, so the receipt itself is never mistaken
#: for the thing it replaced.
_MAX_MARKER_MIME_CHARS = 40

#: Placeholder for one replaced URI. Names the type, the decoded size and that
#: the bytes are gone — no base64, no hash, no filename (the hub never gives an
#: agent a display name, so there is none here to leak).
_MARKER = "[inline data withheld: {mime}, {size} bytes decoded]"

#: Same marker when the payload was cut mid-quantum, so the count is a floor and
#: the reader is told so instead of being given a wrong exact number.
_MARKER_AT_LEAST = "[inline data withheld: {mime}, at least {size} bytes decoded]"


def _decoded_size(payload: str) -> int:
    """Decoded byte count from a base64 string's LENGTH, without decoding it.

    Every 4 base64 characters carry 3 bytes, and each trailing ``=`` pads one
    of those bytes away. ``payload`` is already whitespace-free (the fold
    scanner never appends the separators), so this is the true length. A
    truncated payload (a length that is not a multiple of 4) is rounded down,
    which is why the caller marks such a size "at least".
    """

    if not payload:
        return 0
    padding = 0
    if payload.endswith("=="):
        padding = 2
    elif payload.endswith("="):
        padding = 1
    return max(0, len(payload) * 3 // 4 - padding)


def inline_data_marker(mime: str, size: int, *, at_least: bool = False) -> str:
    """The short structured placeholder that stands in for one withheld URI."""

    template = _MARKER_AT_LEAST if at_least else _MARKER
    shown = (mime or _DEFAULT_MIME).strip()[: _MAX_MARKER_MIME_CHARS]
    return template.format(mime=shown or _DEFAULT_MIME, size=size)


def _fold_resume(text: str, index: int) -> int | None:
    """Index just past a fold separator at ``index``, or ``None`` if there is none.

    TWO FORMS, because the same fold reaches this module in two states. While a
    body is still a plain Python string the separator is a REAL newline; once
    ``json.dumps`` has serialised it the newline becomes the two characters
    ``\\`` ``n``. Accepting both is the defensive half of the serialisation fix:
    the primary fix bounds the string BEFORE serialisation
    (:func:`bound_inline_data_payload`), and this keeps the serialised path
    correct too, so a future change to how a page is built cannot silently
    reopen the hole that shipped a whole blob.

    The cursor only moves forward over separators, so this cannot backtrack.
    """

    length = len(text)
    i = index
    seen = False
    while i < length:
        char = text[i]
        if char in _FOLD_WS:
            i += 1
            seen = True
            continue
        # The escaped spellings json.dumps emits for whitespace.
        if char == "\\" and i + 1 < length and text[i + 1] in "nrt":
            i += 2
            seen = True
            continue
        break
    return i if seen else None


def _skip_fold_ws(text: str, index: int) -> int:
    """``index`` moved past any fold separator; ``index`` itself when there is none."""

    return _fold_resume(text, index) or index


def _consume_payload(text: str, start: int) -> tuple[str, int, bool]:
    """``(whitespace-free payload, end index, tail_left)``; one forward pass.

    ``start`` must already be past the whitespace that follows ``;base64,`` —
    so a payload that begins on the next line (a CRLF straight after the marker,
    which used to make the whole URI look like a mention) is consumed normally.

    A base64 run is "alphabet characters, optionally separated by line breaks"
    because wrapping is what real encoders emit. Expressing that as a regex is
    the textbook catastrophic-backtracking shape, so it is a cursor walk here:
    the cursor only ever advances, and the cost is the length of the payload.

    ``tail_left`` is True when a continuation was declined AFTER the wrap was
    already established — that short run is the wrap's own final line, so the
    reported size is a floor. With no fold at all it is prose and the payload is
    whole, so the size is exact.
    """

    length = len(text)
    index = start
    line_start = start
    folded = False
    pieces: list[str] = []
    while True:
        end = index
        while end < length and text[end] in _PAYLOAD_CHARS:
            end += 1
        pieces.append(text[index:end])
        index = end
        # A fold is only a fold if the line it would continue was a full line of
        # payload; otherwise this separator ends the blob (see _FOLD_MIN_LINE).
        if (index - line_start) < _FOLD_MIN_LINE:
            break
        resume = _fold_resume(text, index)
        if resume is None:
            break
        next_end = resume
        while next_end < length and text[next_end] in _PAYLOAD_CHARS:
            next_end += 1
        # The NEXT line must also reach the floor. That is what tells a wrap
        # apart from a single blob followed by a short word: prose is never a run
        # this long of unbroken base64 characters.
        if (next_end - resume) >= _FOLD_MIN_LINE:
            line_start = resume
            index = resume
            folded = True
            continue
        break
    tail_left = folded and _fold_resume(text, index) is not None
    return "".join(pieces), index, tail_left


def _header_mime(text: str, ceiling: int) -> str:
    """The media type introducing the marker that ends at ``ceiling``.

    Decoration ONLY — the caller has already decided to withhold the payload
    before this runs. Walking back to ``data:`` is bounded by
    :data:`_MAX_HEADER_BACK` and by the caller's cursor, so a header of any
    length or parameter count cannot turn a decided withhold into a skip.
    """

    floor = max(0, ceiling - _MAX_HEADER_BACK)
    at = text.rfind("data:", floor, ceiling)
    if at < 0:
        # Case-insensitive fallback: rfind on a slice keeps the cost local, and
        # this only runs for the (rare) headers not spelled in lower case.
        window = text[floor:ceiling]
        lowered = window.lower()
        offset = lowered.rfind("data:")
        if offset < 0:
            return _DEFAULT_MIME
        at = floor + offset
    header = text[at + len("data:") : ceiling]
    return header.split(";", 1)[0].strip()[: _MAX_MARKER_MIME_CHARS]


def bound_inline_data_uris(text: str) -> str:
    """Return ``text`` with every inline ``;base64`` data URI replaced.

    Pure: no I/O, no decoding, no globals mutated. ``text`` that carries no data
    URI is returned as the SAME object, unscanned in the sense that matters —
    not one byte is dropped from it, at any length. That identity is asserted in
    the tests because a body mentioning ``metadata:`` contains the substring
    ``data:`` and used to be truncated for it.

    Cost is linear in ``len(text)``: a literal-marker search, an amortised
    bounded backward hop for the media type, and one monotonic payload scan per
    URI (see the module docstring).
    """

    if _ANY_DATA.search(text) is None:
        return text
    pieces: list[str] = []
    cursor = 0
    for marker in _BASE64_MARKER.finditer(text):
        if marker.start() < cursor:
            # Inside a payload a previous iteration already replaced.
            continue
        mime = _header_mime(text, marker.start())
        payload_start = _skip_fold_ws(text, marker.end())
        payload, end, tail_left = _consume_payload(text, payload_start)
        if not payload:
            # A marker with no payload is a mention of a data URI, not a blob to
            # withhold: replacing it would turn readable text into a false claim
            # that bytes were dropped.
            continue
        uri_start = _uri_start(text, cursor, marker.start())
        pieces.append(text[cursor:uri_start])
        pieces.append(
            inline_data_marker(
                mime,
                _decoded_size(payload),
                at_least=tail_left or len(payload) % 4 != 0,
            )
        )
        cursor = end
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _uri_start(text: str, cursor: int, ceiling: int) -> int:
    """Where the ``data:`` introducing the marker at ``ceiling`` begins.

    Bounded by ``cursor`` and :data:`_MAX_HEADER_BACK`. Falls back to ``ceiling``
    (replace the marker only) when no ``data:`` precedes it, which keeps a
    bare ``;base64,`` mention in prose from eating the words before it.
    """

    floor = max(cursor, ceiling - _MAX_HEADER_BACK)
    at = text.rfind("data:", floor, ceiling)
    if at >= 0:
        return at
    window = text[floor:ceiling]
    offset = window.lower().rfind("data:")
    return floor + offset if offset >= 0 else ceiling


def bound_inline_data_payload(value: Any) -> Any:
    """Bound every string INSIDE a JSON-ish payload, BEFORE it is serialised.

    WHY BEFORE SERIALISATION, and why that is the shipping fix rather than a
    nicety. Every history op builds its model-visible page with
    ``json.dumps(result)``, and ``json.dumps`` rewrites a newline in a body to the
    two characters ``\\`` ``n``. A MIME-wrapped payload therefore reaches the model
    as one long line of visible ``\\n`` escapes, and scanning only the serialised
    text meant the fold was never seen: the first 76-char line matched on its own
    (a false exact ``57 bytes`` receipt) and the remaining ~700 KB shipped
    unchanged. Bounding the plain Python string here means the escaped form never
    carries a payload at all.

    :func:`bound_inline_data_uris` also understands the escaped form, so a caller
    that only has serialised text is still covered — two layers because this hole
    was reachable by exactly one of them.

    COPY-ON-WRITE. A payload with no data URI is returned as the SAME object at
    every level, so the overwhelmingly common case allocates nothing and the
    pass-through guarantee is observable by identity, not just by equality.
    """

    if isinstance(value, str):
        return bound_inline_data_uris(value)
    if isinstance(value, dict):
        changed = False
        mapped: dict[Any, Any] = {}
        for key, item in value.items():
            bounded = bound_inline_data_payload(item)
            changed = changed or bounded is not item
            mapped[key] = bounded
        return mapped if changed else value
    if isinstance(value, list):
        bounded_items = [bound_inline_data_payload(item) for item in value]
        return value if all(new is old for new, old in zip(bounded_items, value)) else bounded_items
    if isinstance(value, tuple):
        bounded_items = tuple(bound_inline_data_payload(item) for item in value)
        return value if all(new is old for new, old in zip(bounded_items, value)) else bounded_items
    return value
