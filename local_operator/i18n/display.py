"""Opt-in display projection; raw messages, catalogues and wire bytes stay clean.

This source-only prototype intentionally takes explicit offsets AFTER ICU has
formatted native values. It is not a regex tokenizer, a second ICU renderer or
full parameter-span support. Token ownership belongs to the narrow paint caller;
ASCII human names must not be labelled technical merely because they are ASCII.
No bidi reordering/shaping is performed: terminal support must be validated later.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

from .messages import RenderedMessage

_RLI, _LRI, _PDI = "\u2067", "\u2066", "\u2069"
_BIDI_CONTROLS = frozenset(
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


def _validate_token_pairs(value: str) -> None:
    """Check a deliberately small balanced-delimiter subset, not shell syntax.

    Openers ANYWHERE in a declared token must close inside its island, including
    fn(arg) and --x=[id]. Quote pairs may occur internally (mode='own'); bracket
    characters inside quotes are literal. Backslash escape rules and unquoted
    apostrophes are ambiguous without a language parser, so reject rather than
    invent shell/identifier semantics. This validates pairing, not authenticity
    or whether a caller has classified a human name as a technical identifier.
    """
    paired = {"{": "}", "[": "]", "(": ")"}
    stack: list[str] = []
    quote: str | None = None
    for index, char in enumerate(value):
        if char == "\\":
            raise ValueError("escaped technical tokens are outside the display subset")
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            if index and value[index - 1] not in "=([{":
                raise ValueError("ambiguous embedded quote is outside the display subset")
            quote = char
        elif char in paired:
            stack.append(paired[char])
        elif char in "}])":
            if not stack or stack.pop() != char:
                raise ValueError("technical token delimiters must balance inside the island")
    if quote is not None or stack:
        raise ValueError("technical token quotes/delimiters must close inside the island")


@dataclass(frozen=True)
class TechnicalSpan:
    """Owner-declared complete ASCII token at Python string offsets, not bytes.

    Expected bytes pin a hint to THIS rendered record; stale offsets cannot
    silently isolate a different value. Include leading flags/slashes and paired
    technical quotes/braces. Overlaps, controls, multiline and partial tokens
    are rejected rather than guessed or repaired.
    """

    start: int
    end: int
    expected: str


def isolated_status_text(record: RenderedMessage, technical: Sequence[TechnicalSpan] = ()) -> str:
    """Project plain Urdu-source status prose, balancing isolates on every line.

    Never feed this projection to Markdown/ANSI parsing, logs, receipts or copy
    semantics. This is only text for a plain Rich paint adapter. English fallback
    and unknown/passthrough output remain byte-identical, even if preference is ur.
    Existing controls/escape sequences are rejected on this opt-in boundary so
    external input cannot close an isolate or smuggle terminal control state.
    """
    text = record.text
    # LF is the only admitted separator: U+2029 ends a bidi paragraph and
    # U+2028 is a renderer-dependent line break. Neither may cross our LF-only
    # isolate balance. Reject at paint, never sanitize the shared raw result.
    if any(
        c in _BIDI_CONTROLS
        or c in "\u2028\u2029"
        or (unicodedata.category(c) == "Cc" and c != "\n")
        for c in text
    ):
        raise ValueError("status display requires plain text without control sequences")
    if any(
        not isinstance(span, TechnicalSpan)
        or type(span.start) is not int
        or type(span.end) is not int
        for span in technical
    ):
        raise ValueError("technical status hints require integer-offset TechnicalSpan records")
    spans = sorted(technical, key=lambda span: span.start)
    previous = 0
    for span in spans:
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not 0 <= span.start < span.end <= len(text)
            or span.start < previous
            or not isinstance(span.expected, str)
            or not span.expected
            # ASCII, but a declared COMMAND may carry arguments
            # (`/network new <name>`); the whole command must be one island or
            # bidi re-orders its words. Spaces are allowed INSIDE the span;
            # the boundary checks below still forbid a partial token.
            or not all(0x20 <= ord(c) <= 0x7E for c in span.expected)
            or span.expected != span.expected.strip()
            or text[span.start : span.end] != span.expected
        ):
            raise ValueError("invalid or overlapping technical status span")
        _validate_token_pairs(span.expected)
        # A token cannot begin inside --force, /session, quoted IDs or {names}.
        # End punctuation may belong to the prose; quotes/braces belong to the
        # technical island and must be supplied in full by its owner.
        if span.start and not text[span.start - 1].isspace():
            raise ValueError("technical status span must include its whole prefix")
        suffix = span.end
        # Urdu/Arabic sentence punctuation belongs to the PROSE that follows
        # the token, exactly as ASCII . , ; : ! ? does. Without these the
        # native `… /update۔` copy failed the whole-suffix rule (B-1).
        while suffix < len(text) and text[suffix] in ".,;:!?۔؟،":
            suffix += 1
        if suffix < len(text) and not text[suffix].isspace():
            raise ValueError("technical status span must include its whole suffix")
        previous = span.end
    if record.source_locale != "ur":
        return text
    chunks: list[str] = []
    cursor = 0
    for span in spans:
        chunks.extend((text[cursor : span.start], _LRI, span.expected, _PDI))
        cursor = span.end
    chunks.append(text[cursor:])
    # split('\n'), not splitlines(): retain exact delimiters and trailing empty
    # lines. Controls are inserted only within a line, never across paragraphs.
    return "\n".join(_RLI + line + _PDI for line in "".join(chunks).split("\n"))
