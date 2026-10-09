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

The OUTPUT half of the attachment contract rides the SAME store and the same
digests, deliberately — one mechanism, two directions. A tool that produces
binary media (a generated image, a fetched video) registers the bytes once
through :func:`cache_media` and hands the transcript a pointer-shaped
``AttachmentContent`` block instead of bytes; every surface already knows how
to resolve a digest, so a second transport would be duplicate plumbing with a
second set of failure modes. The properties above are exactly what generated
media needs: never evicted (a transcript that references a video must keep
resolving it), deduplicated across sessions, and readable by anything that
can read a file. What moves between machines is the config dir; a session
moved WITHOUT it degrades to the block's ``source_url`` or to a surface's own
unavailable state — never to a broken row.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from local_operator.paths import config_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AttachmentContent

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


def store_for_transcript_dir(directory: str | Path) -> AttachmentStore:
    """The store that OWNED the journal in ``directory``, derived from its path.

    ``directory`` is a session directory — the one holding ``transcript.jsonl``,
    i.e. ``<cfg>/sessions/<id>``. A journal written there was written by a
    process whose config dir was ``<cfg>``, because that is the only place
    ``Transcript`` externalizes to (``AttachmentStore()`` ==
    ``config_dir()/attachments``). So ``<cfg>/attachments`` is the writer's
    store **by construction**, independent of what ``config_dir()`` resolves to
    in the process doing the reading.

    That independence is the whole point, and it is why this is a path
    computation rather than the env default. Two readers know the config dir
    that OWNS the transcript they are reading without being the writer: the
    sidebar's saved-preview reader and the attached viewer's cold replay. For
    them the env default happens to be the same directory in every shipped
    caller today, which is exactly the kind of unstated coincidence that let a
    second, wrong root live beside this one before (#694 — a per-session root
    that nothing writes to, so every reference resolved to ``None`` and a
    live screenshot replayed as "image unavailable"). One definition keeps the
    readers and the write path from drifting apart again.

    NEVER root this at the session directory itself: nothing ever writes under
    ``<cfg>/sessions/<id>``, so a store there can only return ``None``.
    """
    return AttachmentStore(Path(directory).parent.parent / ATTACHMENTS_DIRNAME)


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
        return self.put_bytes(raw, mime_type)

    def put_bytes(self, raw: bytes, mime_type: str) -> AttachmentRef | None:
        """Store decoded ``raw`` bytes and return their reference.

        The bytes-in counterpart of :meth:`put`, for producers that already
        hold decoded media — a generated image, a fetched video — and would
        otherwise pay a base64 round-trip only for this side to undo it.
        Same contract as :meth:`put` in every other respect: content-addressed,
        idempotent, ``None`` on any failure, never raises.
        """
        if not raw:
            return None
        digest = hashlib.sha256(raw).hexdigest()[:_DIGEST_CHARS]
        content = self._content_path(digest)
        if not content.exists():
            try:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Write content before the sidecar: a sidecar pointing at
                # absent content would resolve to nothing on read, while
                # content without a sidecar is merely unused disk. Two
                # concurrent writers of the same digest always write
                # identical bytes (the digest IS the content), so the
                # exists()-then-write race is a wasted write, never a
                # torn file. A crash between the two writes leaves
                # content without a sidecar, which ``get`` treats as
                # missing — the next ``put`` of the same image retries.
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
        except (OSError, ValueError):
            return None
        # A sidecar that PARSES but is not a JSON object (``[]``, ``null``, a
        # bare number) has no ``mime_type`` key to read, so ``meta.get`` would
        # raise AttributeError — straight through the callers whose contract is
        # "degrade to a placeholder, never raise" (this method's own docstring,
        # ``transcript._resolve_attachments``, and the readers built on them).
        # Damaged sidecars are a hand-edited or interrupted write, i.e. the same
        # class as a missing file, so they are treated the same way.
        if not isinstance(meta, dict):
            return None
        mime_type = str(meta.get("mime_type", "image/png"))
        # The filename IS the digest. A bit-rotted or truncated file
        # that still has a sidecar would otherwise flow silently-wrong
        # bytes into the model; treat a mismatch as missing so replay
        # degrades to a placeholder instead.
        if hashlib.sha256(raw).hexdigest()[:_DIGEST_CHARS] != digest:
            logger.warning("attachment store: digest mismatch for %s", digest)
            return None
        return base64.b64encode(raw).decode("ascii"), mime_type

    def get_bytes(self, digest: str) -> tuple[bytes, str] | None:
        """``(decoded bytes, mime_type)`` for ``digest``, or ``None``.

        The bytes-out counterpart of :meth:`get`, for consumers that would
        immediately decode its base64 anyway — the TUI's artifact-image mount,
        the mobile image endpoint. Same degradation contract in every other
        respect: unresolvable is ordinary (interrupted write, hand-pruned
        store), a bit-rotted file fails its own digest check, and neither
        raises.
        """
        try:
            raw = self._content_path(digest).read_bytes()
            meta = json.loads(self._meta_path(digest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict):
            return None
        if hashlib.sha256(raw).hexdigest()[:_DIGEST_CHARS] != digest:
            logger.warning("attachment store: digest mismatch for %s", digest)
            return None
        return raw, str(meta.get("mime_type", "application/octet-stream"))


#: MIME major type → the contract's ``kind`` vocabulary, for :func:`cache_media`.
#: Exact on purpose — anything else is refused rather than guessed into
#: "image", because ``kind`` decides which surface renders the block, and a
#: wrong guess shows a video where a picture belongs.
_KIND_BY_MAJOR_TYPE = {"image": "image", "video": "video", "audio": "audio"}


def cache_media(
    raw: bytes,
    content_type: str,
    *,
    kind: str | None = None,
    name: str | None = None,
    source_url: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_s: float | None = None,
) -> AttachmentContent | None:
    """Register ``raw`` as a first-class artifact and return the block to carry.

    THE one registration call of the output-attachment contract: media a
    tool produced or fetched for the USER's surfaces goes through here, and
    the returned :class:`~local_operator.harness.types.AttachmentContent`
    block (all-nullable metadata around a store digest) is what rides the
    tool result. Content-addressed through :meth:`AttachmentStore.put_bytes`,
    so the same bytes registered twice collapse to one digest, and it is the
    same store and digest format user-pasted images already resolve through
    — one mechanism, two directions, as the module docstring argues.

    ``None`` is a normal outcome, not an error: an empty payload, a
    non-media ``content_type``, a read-only home directory or a full disk all
    land here, and the caller's contract is to degrade to a text error —
    never to raise into a turn. A caller told ``None`` must NOT build an
    artifact block around it; a block with no digest renders as unavailable,
    which would be a lie about a write that never happened.

    ``kind`` is derived from ``content_type``'s major type when omitted, and
    refused when neither path lands in the contract's vocabulary. Image
    dimensions are read from the header for free (``media.sniff_image``) when
    the caller does not know them; nothing is decoded. ``content_type`` is
    trusted as given — unlike the mobile ingest path, which re-sniffs
    everything — because the caller here is a tool holding provider bytes,
    an authority the phone's wire payload does not have.
    """
    if not raw:
        return None
    resolved = (
        kind
        if kind is not None
        else _KIND_BY_MAJOR_TYPE.get(content_type.partition("/")[0].strip().lower())
    )
    # The check spells the literal tuple rather than `_KIND_BY_MAJOR_TYPE.values()`
    # deliberately: pyright narrows ``str`` through a membership test against
    # literal values, which is exactly what types ``resolved`` for the
    # ``AttachmentContent(kind=...)`` constructor below.
    if resolved not in ("image", "video", "audio"):
        return None
    if resolved == "image" and (width is None or height is None):
        from local_operator.media import sniff_image  # lazy: keep this module lean

        info = sniff_image(raw)
        if info is not None:
            width = width if width is not None else info.width
            height = height if height is not None else info.height
    ref = AttachmentStore().put_bytes(raw, content_type)
    if ref is None:
        return None
    from local_operator.harness.types import AttachmentContent  # lazy: import weight

    return AttachmentContent(
        kind=resolved,
        content_type=content_type,
        attachment=ref.digest,
        source_url=source_url,
        size_bytes=len(raw),
        width=width,
        height=height,
        duration_s=duration_s,
        name=name,
    )
