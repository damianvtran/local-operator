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

WHY THE SCAN IS WINDOWED. The input is attacker-sized, so the work must not be.
:data:`INLINE_DATA_SCAN_MAX_CHARS` bounds one body's scan and the size of what
this can emit; a body longer than the window keeps its scanned prefix and its
unscanned remainder is replaced by an explicit marker rather than passed
through, because the whole point is that nothing unscanned reaches the model
as a blob. Total work is then linear in the page the client has already read
into memory, never quadratic in it.
"""

from __future__ import annotations

import re

__all__ = [
    "INLINE_DATA_SCAN_MAX_CHARS",
    "bound_inline_data_uris",
    "inline_data_marker",
]

#: Characters of ONE body this will look at. A body at or under the window is
#: scanned whole, so the common pathological case (the operator's ~600 KB
#: receipt) still reports its TRUE decoded size instead of an "at least"
#: figure. Above it the remainder is withheld with a stated marker.
INLINE_DATA_SCAN_MAX_CHARS = 1024 * 1024

#: Media types with no ``/`` that are still legal in the header (the RFC
#: allows an omitted type, which defaults to ``text/plain``).
_DEFAULT_MIME = "text/plain"

#: One inline data URI. Deliberately narrow on purpose: it matches only the
#: ``;base64,`` form, because the percent-encoded form is human-readable-ish
#: text and carries no base64 run to blow up a context window. ``mime`` and
#: the ``;param=value`` list are captured only so the receipt can name the type.
_INLINE_DATA_URI = re.compile(
    r"data:(?P<mime>[^;,\s]+)?(?P<params>(?:;[^;,\s]+)*);base64,(?P<payload>[A-Za-z0-9+/]*={0,2})"
)

#: How many bytes of a body were left unscanned, and that they were withheld.
#: Stated rather than silent: an agent must never believe it read a body it saw
#: only the prefix of (same rule the file-inlining design fixes for files).
_TAIL_MARKER = "[{count} more bytes of this body were not scanned for inline data and are withheld]"

#: Placeholder for one replaced URI. Names the type, the decoded size and that
#: the bytes are gone — no base64, no hash, no filename (the hub never gives an
#: agent a display name, so there is none here to leak).
_MARKER = "[inline data withheld: {mime}, {size} bytes decoded]"

#: Same marker when the payload was cut by the scan window, so the count is a
#: floor and the reader is told so instead of being given a wrong exact number.
_MARKER_AT_LEAST = "[inline data withheld: {mime}, at least {size} bytes decoded]"


def _decoded_size(payload: str) -> int:
    """Decoded byte count from a base64 string's LENGTH, without decoding it.

    Every 4 base64 characters carry 3 bytes, and each trailing ``=`` pads one
    of those bytes away. A truncated payload (a length that is not a multiple
    of 4) is rounded down, which is why the caller marks such a size "at least".
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


def _replace_one(match: re.Match[str], *, truncated: bool) -> str:
    payload = match.group("payload")
    # A header with no payload is a mention of a data URI, not a blob to
    # withhold: replacing it would turn readable text into a false claim that
    # bytes were dropped. Leave those alone.
    if not payload:
        return match.group(0)
    cut = truncated and match.end() == len(match.string)
    return inline_data_marker(
        (match.group("mime") or _DEFAULT_MIME).strip(),
        _decoded_size(payload),
        at_least=cut or len(payload) % 4 != 0,
    )


def bound_inline_data_uris(text: str) -> str:
    """Return ``text`` with every inline ``;base64`` data URI replaced.

    Pure: no I/O, no decoding, no globals mutated. ``text`` that contains no
    ``data:`` at all is returned unchanged and unscanned — a plain long body
    keeps every byte, so this cannot silently truncate ordinary content.

    A body longer than :data:`INLINE_DATA_SCAN_MAX_CHARS` is scanned only up to
    the window and its remainder is replaced by an explicit marker. That tail is
    withheld rather than forwarded because an unscanned tail is exactly the
    unbounded blob this module exists to keep out of context; stating the count
    keeps the omission honest instead of invisible.
    """

    if "data:" not in text:
        return text
    scanned = text[:INLINE_DATA_SCAN_MAX_CHARS]
    unscanned = len(text) - len(scanned)
    bounded = _INLINE_DATA_URI.sub(
        lambda match: _replace_one(match, truncated=bool(unscanned)), scanned
    )
    if unscanned:
        bounded = f"{bounded}\n\n{_TAIL_MARKER.format(count=unscanned)}"
    return bounded
