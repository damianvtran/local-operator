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
#: o200k_base: English prose ~4.5, minified JSON ~2.3, hex ~1.8, base64 ~1.5.
#: 2 covers all of them, so ASCII text is cut at ``window * 2 / 4`` characters.
#: It is NOT the budget for non-ASCII text, which no ratio can bound: CJK runs
#: 1.3 chars/token for common ideographs but ~2 for random full-range ones,
#: Hangul and emoji ~2.3, and CJK Extension B ~4. See :func:`text_weight`.
DOCUMENT_CHARS_PER_TOKEN = 2


def text_weight(text: str) -> int:
    """Budget units ``text`` costs: ASCII is 1 unit a character, anything else
    ``DOCUMENT_CHARS_PER_TOKEN`` units per UTF-8 byte.

    A unit is ``1 / DOCUMENT_CHARS_PER_TOKEN`` of a token, so non-ASCII text is
    charged one token per UTF-8 byte. That is a hard ceiling rather than a
    measurement: a tokenizer can always fall back to one token per byte, so
    ``tokens <= len(text.encode("utf-8"))`` holds for every text and tokenizer
    (the rule ``compaction.tokens`` already applies to its upper bound). A
    cap in CHARACTERS cannot give that guarantee, which is how a 400k-character
    file of Hangul or CJK Extension B filled (or overran) a 200k window.
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
