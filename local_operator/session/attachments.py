"""Content-addressed store for message attachments (images and other media).

Transcripts reference images by base64 inline in ``transcript.jsonl``. On a
screenshot-heavy install that payload dominates the session store — measured
at 102 of 134 MB (76%) across 142 real sessions — and it is the single most
redundant data on the disk:

- **Base64 inflation.** One third of every stored image byte is encoding
  overhead; the decoded bytes are 25% smaller before anything else happens.
- **Cross-session duplication.** The same screenshot is re-stored by every
  session that pasted or captured it. Measured: 434 image references reduced
  to 355 unique images, ~20 MB of exact duplicates.

This module is the answer. ``<config>/attachments/<digest>.bin`` holds each
unique image ONCE, named by the sha256 of its decoded bytes, with a small
``<digest>.json`` sidecar carrying the mime type. A transcript row then
carries a reference — ``{"type": "image", "attachment": "<digest>",
"mime_type": ...}`` — instead of the payload. At the measured ratios this
takes the session store from 134 MB to ~52 MB, with the saving growing
faster than the store itself as duplicates accumulate.

Two properties make this safe where the old retention ceilings were not:

- **Nothing is ever deleted.** There is no eviction, no sweep, no ceiling.
  An attachment lives for as long as any transcript references it, and after
  that too — reclaiming orphaned bytes is a user's explicit choice, exactly
  like session transcripts themselves.
- **Reads are fully backward compatible.** A row carrying inline ``data``
  loads exactly as before, so transcripts written by older builds, exports,
  and any external tool that reads the JSONL directly all keep working.
  Re-attaching the inline bytes on load (rather than rewriting the file)
  would break the dedup the store exists for, so rows are externalized on
  write only.

The store deliberately does NOT reuse the spill store (``tools/spill.py``):
spill is LRU-evicted under a byte ceiling and is allowed to forget content
a transcript still references, which is exactly the failure this work
exists to remove from the session store.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

#: Directory under the config dir holding deduplicated attachment content.
ATTACHMENTS_DIRNAME = "attachments"

#: Characters of the hex digest used for file names. 32 hex chars is 128
#: bits — collision-free for any realistic store, and short enough that a
#: transcript reference is a rounding error next to the payload it replaced.
_DIGEST_CHARS = 32


@dataclass(frozen=True)
class AttachmentRef:
    """The reference a transcript row carries in place of inline base64."""

    digest: str
    mime_type: str
    bytes: int  # decoded bytes — what the reference stands in for


def attachments_dir() -> Path:
    """Directory holding the store. Resolved per call, never cached:
    ``config_dir()`` reads the environment on every call precisely so tests
    can relocate it after import."""
    return config_dir() / ATTACHMENTS_DIRNAME


class AttachmentStore:
    """Content-addressed binary store under the config dir.

    Instantiate per transcript directory; ``root`` is injectable for tests
    and for callers that resolve the config dir themselves.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else attachments_dir()

    # -- paths -------------------------------------------------------------

    def _content_path(self, digest: str) -> Path:
        return self.root / f"{digest}.bin"

    def _meta_path(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    # -- write -------------------------------------------------------------

    def put(self, data_b64: str, mime_type: str) -> AttachmentRef | None:
        """Store base64 ``data_b64`` and return its reference.

        ``None`` is a normal outcome, not an error: undecodable input, a
        read-only home directory, or a full disk all land here, and the
        caller's contract is to keep the inline base64 in the transcript —
        strictly worse on disk, never wrong to read. Raising instead would
        turn a degraded store into a failed message append.

        Writing the same image twice is idempotent: the digest is the
        identity, so the second write is a no-op that costs a hash and a
        stat. That is the dedup the store exists for.
        """
        try:
            raw = base64.b64decode(data_b64, validate=False)
        except (ValueError, TypeError):
            return None
        if not raw:
            return None
        digest = hashlib.sha256(raw).hexdigest()[:_DIGEST_CHARS]
        content = self._content_path(digest)
        if not content.exists():
            try:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Write content before the sidecar: a sidecar pointing at
                # absent content would resolve to nothing on read, while
                # content without a sidecar is merely unused disk.
                content.write_bytes(raw)
                meta = {"digest": digest, "mime_type": mime_type, "bytes": len(raw)}
                self._meta_path(digest).write_text(json.dumps(meta), encoding="utf-8")
            except OSError as exc:
                logger.debug("attachment store: cannot write %s: %s", digest, exc)
                return None
        return AttachmentRef(digest=digest, mime_type=mime_type, bytes=len(raw))

    # -- read --------------------------------------------------------------

    def get(self, digest: str) -> tuple[str, str] | None:
        """``(base64 data, mime_type)`` for ``digest``, or ``None``.

        ``None`` means the reference is unresolvable — an interrupted write
        or a store the user pruned by hand. Callers must treat that as
        ordinary and degrade to a placeholder, never raise: a resumed
        session must survive a missing attachment the same way it survives a
        dropped malformed transcript line.
        """
        try:
            raw = self._content_path(digest).read_bytes()
            meta = json.loads(self._meta_path(digest).read_text(encoding="utf-8"))
            mime_type = str(meta.get("mime_type", "image/png"))
        except (OSError, ValueError):
            return None
        return base64.b64encode(raw).decode("ascii"), mime_type
