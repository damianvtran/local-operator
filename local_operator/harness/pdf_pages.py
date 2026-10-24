"""PDF page counts shared by the two rulers that need them.

WHY A NEUTRAL MODULE, rather than one of the callers owning it. Two
independent decisions need the page count of the same bytes:

* ``providers/documents.py`` decides whether a PDF fits a provider's page
  ceiling before sending it natively;
* ``compaction/tokens.py`` charges a document by its pages, because a flat
  byte estimate either undercounts a text PDF badly (a 100-page document
  costs 150k-300k tokens) or overcounts a scan.

``compaction`` deliberately does not import ``session``/``model``/``providers``
(its evaluation-runner host asserts that boundary in a subprocess probe), so
the parser cannot live in ``documents.py`` without either duplicating the
parse or breaking that probe. This module imports stdlib plus a LAZY
``pypdf``, which both sides may use.

The cache is per DIGEST, not per path: the attachment store is
content-addressed, so the answer cannot change for a digest for the life of
the process, and a session resends its history on every turn.
"""

from __future__ import annotations

import hashlib
import io
import logging

logger = logging.getLogger(__name__)

#: Parsed page counts, including the ``None`` answer for an unparseable PDF
#: (``None`` is an answer too: retrying the parse on every turn would pay the
#: xref read again for a document that will never get a count).
_PAGE_COUNT_CACHE: dict[str, int | None] = {}


def known_pdf_page_count(digest: str | None) -> tuple[bool, int | None]:
    """``(cached, count)`` for ``digest`` — lets a caller skip decoding bytes.

    A token estimate that already knows the page count must not base64-decode
    a 20 MB PDF just to hand ``pdf_page_count`` bytes it will not read: the
    digest-keyed answer is the whole reason the cache exists (agent review
    round 3, minor 1).
    """
    if digest is None or digest not in _PAGE_COUNT_CACHE:
        return False, None
    return True, _PAGE_COUNT_CACHE[digest]


def pdf_page_count(raw: bytes, digest: str | None = None) -> int | None:
    """Pages in ``raw``, or ``None`` when the document cannot be parsed.

    ``None`` is not an error: a scanned/encrypted/corrupt PDF still has bytes
    and can still be sent to a provider that accepts PDFs (Anthropic,
    Google). It only means the page LIMIT cannot be checked, which is why the
    delivery decision treats an unknown count as "inside" for providers whose
    only meaningful limit is bytes, and the token estimator falls back to its
    byte-derived charge.
    """
    key = digest or hashlib.sha256(raw).hexdigest()[:32]
    if key in _PAGE_COUNT_CACHE:
        return _PAGE_COUNT_CACHE[key]
    try:
        from pypdf import PdfReader  # lazy: only a document's reader pays it

        count: int | None = len(PdfReader(io.BytesIO(raw), strict=False).pages)
    except Exception:  # noqa: BLE001 — an unparseable PDF is a normal document
        logger.debug("document: page count unavailable", exc_info=True)
        count = None
    _PAGE_COUNT_CACHE[key] = count
    return count
