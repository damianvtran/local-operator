"""Framework-free catalogue HTTP mechanics shared by core and relay.

The mobile daemon must not import FastAPI just to compare a validator. Both
routes emit the same raw-file revision and accept the same weak/list/wildcard
If-None-Match forms; keeping that comparison here prevents cache divergence.
"""

from __future__ import annotations


def etag_matches(header: str, sha: str) -> bool:
    """RFC 9110 §13.1.2 weak comparison for ``If-None-Match``.

    Emitted validators remain strong, quoted raw-file SHA-256 revisions. This
    comparison is deliberately identical to the original core route behavior.
    """
    for raw in header.split(","):
        tag = raw.strip()
        if tag == "*":
            return True
        if tag.startswith("W/"):
            tag = tag[2:].strip()
        if len(tag) >= 2 and tag.startswith('"') and tag.endswith('"') and tag[1:-1] == sha:
            return True
    return False
