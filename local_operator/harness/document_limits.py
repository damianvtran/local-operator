"""Document-rendering limits shared by the token ruler and the serializers.

Lives under ``harness`` because ``compaction.tokens`` must stay free of any
``providers`` import (the import-isolation tests in ``compaction`` and the
evaluation runner enforce it), yet it has to charge a document by the SAME
class table and bounds the provider serializers render with. Defining them in
one place is what keeps ruler and render from drifting apart.
"""

from __future__ import annotations

#: How much of a text-ish file the fallback keeps. Text has no page structure
#: to count, and the truncation notice names the cut.
MAX_FALLBACK_TEXT_CHARS = 400_000

#: One document may claim at most ``1 / DOCUMENT_WINDOW_SHARE`` of the model's
#: context window once rendered. A quarter leaves the rest of the window to the
#: conversation, the system prompt and the reply; without it a single large
#: file filled a small-window model by itself and compaction, which can only
#: shed OTHER history, could never make the next request fit.
DOCUMENT_WINDOW_SHARE = 4

#: Characters per token an ASCII character is budgeted at. Measured on
#: o200k_base, in characters per token: English prose ~4.5, minified JSON ~2.3,
#: hex ~1.8, base64 ~1.5, random printable ASCII ~1.4, control bytes ~1.0. The
#: ratio 2 covers prose, code and JSON; it does NOT cover hex, base64 or control
#: bytes, which cost 0.57-1.0 tokens a character against the 0.5 charged. That
#: is a deliberate heuristic, not a bound: changing it would shorten every
#: prose and code render, and the damage is limited because one document is
#: capped at a quarter of the window (measured worst case ~63% of the window,
#: control bytes at 32k) and a history that large is shrinkable by compaction.
#: Non-ASCII text is a different case, bounded by UTF-8 bytes: see
#: :func:`text_weight`.
DOCUMENT_CHARS_PER_TOKEN = 2


def text_weight(text: str) -> int:
    """Budget units ``text`` costs: ASCII is 1 unit a character, anything else
    ``DOCUMENT_CHARS_PER_TOKEN`` units per UTF-8 byte.

    A unit is ``1 / DOCUMENT_CHARS_PER_TOKEN`` of a token. Non-ASCII text is
    therefore charged one token per UTF-8 byte, which IS an upper bound: a
    byte-level tokenizer can always fall back to one token per byte, so
    ``tokens <= utf8 bytes`` for the non-ASCII part (the rule
    ``compaction.tokens`` applies to its upper bound). A cap in CHARACTERS gives
    no such bound, which is how 400k characters of Hangul or CJK Extension B
    filled (or overran) a 200k window.

    The ASCII part is a heuristic, not a bound (see
    :data:`DOCUMENT_CHARS_PER_TOKEN`): dense ASCII such as base64, hex or
    control bytes costs more than 0.5 tokens a character. One document is still
    bounded by the render cap, ``DOCUMENT_WINDOW_SHARE`` of the window.
    """
    encoded = len(text.encode("utf-8"))
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars + DOCUMENT_CHARS_PER_TOKEN * (encoded - ascii_chars)


def weighted_prefix_len(text: str, budget: int) -> int:
    """Longest prefix of ``text`` (in characters) whose :func:`text_weight`
    fits ``budget`` units.

    Pure-ASCII text, the common case, is cut at exactly ``budget`` characters.
    Otherwise bisect on the prefix weight (monotonic in the prefix length):
    ~19 passes over at most the 400k-character fallback bound.
    """
    if budget <= 0:
        return 0
    if len(text) <= budget and text_weight(text) <= budget:
        return len(text)
    if text.isascii():
        return min(len(text), budget)
    low, high = 0, min(len(text), budget)
    while low < high:
        mid = (low + high + 1) // 2
        if text_weight(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


#: Most pages a PDF may carry NATIVELY, whatever the window. At the per-page
#: charge it is also the ceiling of one PDF's token charge
#: (``compaction.tokens.MAX_DOCUMENT_TOKEN_ESTIMATE``), so the ruler cannot
#: under-count a native PDF on a window big enough to allow more: a 2M window's
#: quarter-share would otherwise permit 200 native pages (500k tokens) against
#: a 250k charge. Longer PDFs are extracted, which is bounded separately.
MAX_NATIVE_PDF_PAGES = 100

#: MIME types the fallback can read as text. PDF is handled by extraction
#: first; these are the "text-ish" second arm of contract §3.
_TEXT_MIMES = frozenset({"application/json", "application/xml", "application/yaml"})


def is_text_mime(mime_type: str) -> bool:
    """Whether the fallback can decode ``mime_type`` directly as text."""
    return mime_type.startswith("text/") or mime_type in _TEXT_MIMES
