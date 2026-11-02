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

#: Characters per token the fallback text is budgeted at. Deliberately the
#: pessimistic end (English prose is ~4, code and CJK nearer 2-3): a cap that
#: under-counts would let a rendered document exceed its share of the window.
DOCUMENT_CHARS_PER_TOKEN = 3

#: MIME types the fallback can read as text. PDF is handled by extraction
#: first; these are the "text-ish" second arm of contract §3.
_TEXT_MIMES = frozenset({"application/json", "application/xml", "application/yaml"})


def is_text_mime(mime_type: str) -> bool:
    """Whether the fallback can decode ``mime_type`` directly as text."""
    return mime_type.startswith("text/") or mime_type in _TEXT_MIMES
