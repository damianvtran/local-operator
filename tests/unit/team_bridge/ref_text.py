"""The per-ask ``ref`` line the tool appends to an outgoing message.

``dm`` and ``channel_post`` end every ask with ``\\n— ref XXXXX`` after the
caller's own text; tests that compare a sent text with what the caller wrote
strip that line first, so they keep pinning the caller's text and not the
random ref.
"""

from __future__ import annotations

import re

_REF_LINE = re.compile(r"\n— ref [0-9A-HJKMNP-TV-Z]{5}\Z")


def without_ref(text: str) -> str:
    return _REF_LINE.sub("", text)
