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

WHY THE COST IS LINEAR, AND WHY THAT IS THE WHOLE DESIGN. The input is
attacker-sized, so the work must not be super-linear: a body is delivered
synchronously on an ``async`` tool path, and a scan that stalls the event loop
is a worse defect than the blob it was meant to remove. Two rules buy that,
and both are load-bearing:

* **Every quantifier in :data:`_HEADER` is bounded.** An unbounded ``+`` inside
  a group that can fail (``[^;,\\s]+`` before a required ``;base64,``) retries
  from every start position and is quadratic in the body — measured at 3.5 s
  for a 25 KB body and ~1.5 h at 1 MiB. A real media type and its parameters
  are short, so the caps (:data:`_MAX_MIME_CHARS`, :data:`_MAX_PARAM_CHARS`,
  :data:`_MAX_PARAMS`) cost nothing a real URI needs and make each attempt
  O(1) instead of O(rest-of-body).
* **The payload is consumed by a forward scanner, not by the regex.** One
  ``while`` pass (:func:`_consume_payload`) with a monotonic cursor cannot
  backtrack, so the payload grammar — which includes line folding — cannot
  reintroduce the cost the header caps just removed.

The result is ~2 passes over the text and no truncation: NOTHING is dropped
from a body that carries no data URI, however long it is (a page that merely
mentions ``metadata:`` is returned identical, by identity), which is why there
is no scan window here any more.
"""

from __future__ import annotations

import re
import string

__all__ = [
    "bound_inline_data_uris",
    "inline_data_marker",
]

#: Media types with no ``/`` are still legal in the header (the RFC allows an
#: omitted type, which defaults to ``text/plain``).
_DEFAULT_MIME = "text/plain"

#: Header caps. These exist for COST, not for validation (see the module
#: docstring): they turn a quadratic retry into a constant one. A DNS-style
#: media type is far under 127 characters and RFC 2397 allows no unbounded
#: parameter list, so no URI a sender can actually mean is refused by them.
_MAX_MIME_CHARS = 127
_MAX_PARAM_CHARS = 127
_MAX_PARAMS = 8

#: One inline data URI HEADER — everything up to and including ``;base64,``.
#: Case-insensitive because RFC 2397/5234 ABNF literals are, and browsers parse
#: ``DATA:image/png;BASE64,`` exactly like the lowercase spelling.
#:
#: The parameter group requires ``name=value`` so the literal ``;base64`` cannot
#: be absorbed into it and then backtracked out of. The payload is deliberately
#: NOT part of this pattern; :func:`_consume_payload` reads it.
_HEADER = re.compile(
    rf"data:(?P<mime>[^;,\s]{{0,{_MAX_MIME_CHARS}}})?"
    rf"(?P<params>(?:;[^;,\s=]{{0,{_MAX_PARAM_CHARS}}}="
    rf"[^;,\s]{{0,{_MAX_PARAM_CHARS}}}){{0,{_MAX_PARAMS}}})"
    r";[ \t]*base64[ \t]*,",
    re.IGNORECASE,
)

#: Cheap pre-test so the no-op path (the overwhelmingly common one) allocates
#: nothing: a lowercased copy of a 2 MB body would be a 2 MB allocation per call.
_ANY_DATA = re.compile(r"data:", re.IGNORECASE)

#: Payload alphabet. Both standard and URL-safe base64, plus padding: a sender
#: may paste either spelling, and ``-_`` is what URL-safe encoders emit.
_PAYLOAD_CHARS = frozenset(string.ascii_letters + string.digits + "+/-_=")

#: Whitespace that can separate folded lines of ONE payload.
_FOLD_WS = " \t\r\n"

#: A fold is resumed only when the continuation line carries at least this
#: many payload characters. Wrapped encoders emit 64- or 76-column lines
#: (``base64``, ``textwrap``, MIME), all above this; a short run followed by a
#: newline is the END of the blob. Resuming there would swallow the first word
#: of the prose after it (``…base64,AAAA\nthanks``), and over-deleting a
#: sender's own words is a worse failure than leaving a fragment: a final
#: wrap line under this width is therefore LEFT in place, which bounds what can
#: survive to under 60 characters — under the 64-char blob threshold, and three
#: orders of magnitude below the 700 KB this exists to remove.
_FOLD_MIN_LINE = 60

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
    return template.format(mime=mime or _DEFAULT_MIME, size=size)


def _consume_payload(text: str, start: int) -> tuple[str, int, bool]:
    """``(whitespace-free payload, end index, tail_left)``; one forward pass.

    WHY A HAND-WRITTEN SCANNER. Base64 payloads are legally folded across lines
    (MIME wrapping, ``textwrap``), so the grammar is "alphabet chars, optionally
    separated by a newline". Expressing that as a nested quantifier is the
    textbook catastrophic-backtracking shape, and this input is sender-chosen.
    A cursor that only ever moves forward cannot backtrack, so the cost is the
    length of the payload and nothing else.

    The cursor always advances or returns, so callers may treat this as
    monotonic and never rescan a region.
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
        # payload; otherwise this newline ends the blob (see _FOLD_MIN_LINE).
        if index < length and text[index] in "\r\n" and (index - line_start) >= _FOLD_MIN_LINE:
            resumed = index
            while resumed < length and text[resumed] in _FOLD_WS:
                resumed += 1
            next_end = resumed
            while next_end < length and text[next_end] in _PAYLOAD_CHARS:
                next_end += 1
            # The NEXT line must also be a full-width line. That is what tells a
            # wrap apart from a single blob followed by a short word: prose is
            # never 60+ unbroken base64 characters.
            if (next_end - resumed) >= _FOLD_MIN_LINE:
                line_start = resumed
                index = resumed
                folded = True
                continue
        break
    # ``tail_left`` only when a SHORT continuation was declined AFTER the wrap
    # was already established: that short run is the wrap's final line, so the
    # size above is a floor. With no fold it is prose, and the payload is whole.
    tail_left = folded and index < length and text[index] in "\r\n"
    return "".join(pieces), index, tail_left


def bound_inline_data_uris(text: str) -> str:
    """Return ``text`` with every inline ``;base64`` data URI replaced.

    Pure: no I/O, no decoding, no globals mutated. ``text`` that carries no data
    URI is returned as the SAME object, unscanned in the sense that matters —
    not one byte is dropped from it, at any length. That identity is asserted in
    the tests because a body mentioning ``metadata:`` contains the substring
    ``data:`` and used to be truncated for it.

    Cost is linear in ``len(text)``: a bounded-quantifier header search plus one
    monotonic payload scan per URI (see the module docstring). Nothing here can
    be made super-linear by a sender, which is what makes it safe to call
    synchronously from the ``async`` tool path.
    """

    if _ANY_DATA.search(text) is None:
        return text
    pieces: list[str] = []
    cursor = 0
    for header in _HEADER.finditer(text):
        if header.start() < cursor:
            # Inside a payload a previous iteration already replaced.
            continue
        payload, end, tail_left = _consume_payload(text, header.end())
        if not payload:
            # A header with no payload is a mention of a data URI, not a blob to
            # withhold: replacing it would turn readable text into a false claim
            # that bytes were dropped.
            continue
        pieces.append(text[cursor : header.start()])
        pieces.append(
            inline_data_marker(
                (header.group("mime") or _DEFAULT_MIME).strip(),
                _decoded_size(payload),
                at_least=tail_left or len(payload) % 4 != 0,
            )
        )
        cursor = end
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)
