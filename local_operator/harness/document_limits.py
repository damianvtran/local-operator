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

#: Characters per token the fallback text is budgeted at. This ratio sizes the
#: REFUSAL (``document_render_excess``) and the render cap
#: (``documents.document_text_cap``) together, so it must not flatter dense
#: text: measured on o200k_base, English prose is ~4.5, minified JSON ~2.3, hex
#: ~1.8, base64 ~1.5 and CJK ~1.3. A ratio of 3 under-counted a CJK file at its
#: cap by 2.3x, which a refusal reads as headroom and the provider answers with
#: a 400. 2 covers JSON and hex and leaves CJK ~1.6x under at the cap (and still
#: inside the window: the cap is a quarter of it), at the price of rendering
#: prose at a smaller character count than a 4-5 ratio would allow.
DOCUMENT_CHARS_PER_TOKEN = 2

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
