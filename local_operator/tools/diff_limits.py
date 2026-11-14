"""Bounds for the write/edit diff's SOURCE-EVIDENCE redaction.

WHY. ``write``/``edit`` results carry a rendered unified diff in
``details['diff']`` (``tools/builtin._diff_details``). The publication seam
(``harness.loop._redact_diff_detail``) judges each hunk's old/new view, but a
hunk holds only two context lines: a registered multi-line value that the hunk
window cuts never appears whole, nothing matches, and its fragments were
published verbatim (on dfc7411: 5 of 10 lines of a registered literal whose
middle line was edited; separately, the first line of a 3-line literal at the
200-row preview cut). The proof only exists in the COMPLETE file sides, so the
producer hands both sides to the loop privately and the loop redacts each side
once, masking every diff row whose source line the redactor touched
(``AgentLoop._settle_write_diff``).

A side larger than :data:`DIFF_SOURCE_MAX_BYTES` cannot be judged within this
bound, so its diff is WITHHELD (counters/path stay exact) rather than published
on the per-hunk judgement alone. The ceiling is 4 MiB per side, the spill
store's own per-entry cap; the redactor's whole-text pass is linear (the shape
pass is documented at ~83 ms/MB in ``redaction_shapes``), so one call costs a
bounded fraction of a second on a worker thread.
"""

from __future__ import annotations

#: Raw UTF-8 bytes per file side the source-evidence pass will redact whole.
DIFF_SOURCE_MAX_BYTES = 4 * 1024 * 1024

#: The seam's fixed withholding receipt (``harness.loop`` uses this object), so
#: the producer can withhold a diff without importing the loop.
DIFF_WITHHELD_TEXT = "[output withheld: credential redaction requires a complete bounded diff]"


def fits_source_evidence(before: str, after: str) -> bool:
    """Whether both complete sides are within :data:`DIFF_SOURCE_MAX_BYTES`."""
    for side in (before, after):
        # ``len`` is a cheap lower bound; encode only when it could fit.
        if len(side) > DIFF_SOURCE_MAX_BYTES:
            return False
        if len(side.encode("utf-8", errors="surrogatepass")) > DIFF_SOURCE_MAX_BYTES:
            return False
    return True
