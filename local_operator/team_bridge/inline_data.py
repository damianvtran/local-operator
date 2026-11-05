"""Withhold an inline ``;base64,`` body from model context, without parsing it.

WHY THIS EXISTS. A Team Bridge body is free-form text a member (or an agent
acting for one) pasted or encoded, so a ``data:image/…;base64,…`` URI can
arrive inside one. The desktop client already collapses such a run into a
bounded receipt before Markdown sees it (``inline-data-receipt.tsx``), but CORE
was handing the raw body straight to the model through the history and
``take_up`` results: one 600 KB string of base64 that is unreadable, unbounded,
and costs context on every read of that page. That is the operator's original
complaint, second door.

WHAT IT IS NOT. A representational filter, not a security control. It does not
strip an injection (a body is still body text and still travels under
``BRIDGE_UNTRUSTED_CLAUSE``), it does not admit or screen anything, and it does
not touch the wire, the hub row, ``body_hash`` or ``details``. Message rows are
immutable by contract and ``body_hash`` is the screening commitment every
admission binds to, so rewriting a body anywhere but at the model boundary
would invalidate verdicts computed over the original bytes.

THE RULE — AND WHY IT IS THIS BLUNT. If a body contains the marker
``;base64,`` case-insensitively ANYWHERE, it is not parsed at all: the body is
replaced by at most :data:`MAX_PREFIX_CHARS` of its original prefix verbatim,
then a marker stating how many characters the original held and that the body
was truncated. Everything after the prefix is dropped. A body without the
marker is returned byte-identical — the same object — after one substring
search.

Four review rounds produced four different bypasses of a matcher that tried to
UNDERSTAND the header. Round 1: an unbounded quantifier went quadratic (3.5 s
at 25 KB). Round 2: MIME-wrapped, URL-safe and upper-case spellings kept the
whole blob. Round 3: a 300-character parameter, an over-long media type, and a
CRLF straight after the marker each skipped the URI entirely, because the
header grammar refused to match and the code read "no match" as "no URI".
Round 4: the backward search for ``data:`` anchored on the ``data:`` inside
``metadata:`` and mangled the prose. Every one of those was a shape the grammar
did not anticipate, and each fix moved the bypass rather than removing it.

The marker is the only thing that reliably distinguishes a data URI from prose,
so it is the only thing consulted. No media type, parameter count, casing,
quoting, fold width, whitespace or boundary case can defeat it, and the cost is
one linear pass that cannot backtrack. The price is one known false positive,
accepted deliberately and stated below.

THE KNOWN FALSE POSITIVE, stated because it is real. A message that merely
MENTIONS ``;base64,`` in prose — a pasted command line, a question about
encodings — is truncated like a blob. That loses the tail of a legitimate
message, and it is the trade this design makes on purpose: the alternative is a
parser that fails open on the next unanticipated header shape, and an
over-truncated message is visible, bounded and self-describing, while a shipped
blob is invisible, unbounded and exactly the defect being fixed. The truncation
says so, in words, in the body the model reads.

WHY IT NEVER DECODES. The bytes are worthless here — the model is told they
were withheld, not shown them — so no base64 is ever decoded. Decoding
sender-controlled input would be the one part of this that could be made
expensive or hostile.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "MAX_PREFIX_CHARS",
    "bound_inline_data_payload",
    "bound_inline_data_uris",
    "withheld_body_marker",
]

#: The only thing this module looks for, and the whole of its decision procedure
#: (see the module docstring). Case-insensitive because RFC 2397/5234 ABNF
#: literals are, and browsers parse ``;BASE64,`` exactly like the lower-case
#: spelling.
_MARKER = re.compile(r";base64,", re.IGNORECASE)

#: How much of the original body survives a truncation. Enough to keep the
#: opening sentence — the part a reader needs to know what the message WAS —
#: while bounding what a sender can push into context with the rest discarded.
MAX_PREFIX_CHARS = 512
#: The statement that replaces everything past the prefix. It names the cost in
#: CHARACTERS, not decoded bytes: after truncation there are no bytes to size,
#: and a character count is what the reader can check against the message they
#: sent. ``body truncated`` is explicit because a silent cut is the failure this
#: module exists to prevent.
_MARKER_TEXT = (
    "\n…[inline base64 data withheld: {count} characters in this message; body truncated]"
)


def withheld_body_marker(count: int) -> str:
    """The statement standing in for a body that carried inline base64 data."""

    return _MARKER_TEXT.format(count=count)


def bound_inline_data_uris(body: str) -> str:
    """Withhold a body that carries the ``;base64,`` marker.

    Returns ``body`` ITSELF (same object, not a copy) when the marker is absent —
    that identity is asserted in the tests, because a body mentioning
    ``metadata:`` must not be touched and an ordinary long message must never be
    shortened.

    When the marker is present the result is the body's prefix followed by
    :func:`withheld_body_marker` carrying the ORIGINAL character count. The prefix
    ends at the EARLIER of :data:`MAX_PREFIX_CHARS` and the marker itself — still
    "at most 512 characters of the original, verbatim", but it means the prefix
    cannot carry a fragment of the payload it is supposed to be removing. A
    blob's first 490 characters are payload, and shipping them while claiming the
    bytes were withheld would be a smaller version of the same lie.

    One substring search plus at most one slice: linear, and with nothing that
    can fail open (see the module docstring for why the parser was removed).
    """

    found = _MARKER.search(body)
    if found is None:
        return body
    return f"{body[: min(MAX_PREFIX_CHARS, found.start())]}{withheld_body_marker(len(body))}"


def bound_inline_data_payload(value: Any) -> Any:
    """Withhold every string INSIDE a JSON-ish payload, BEFORE serialisation.

    WHY BEFORE SERIALISATION. Every history op builds its model-visible page with
    ``json.dumps(result)``, and ``json.dumps`` rewrites a newline in a body to the
    two characters ``\\`` ``n``. Applying this to the plain Python strings means
    the serialised page never carries a marker to begin with — the reason the
    earlier matcher had to understand escaped folds at all.

    COPY-ON-WRITE. A payload with no marker is returned as the SAME object at
    every level, so the common case allocates nothing and the pass-through
    guarantee is observable by identity, not just by equality.
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
