"""Bounded blob storage for preview bytes, addressed by the SCOPE that owns them.

THE ID IS AN OWNERSHIP CLAIM, NOT A CONTENT HASH. It is derived from the account,
the tenant and the canonical URL the bytes were fetched for
(:func:`~local_operator.team_bridge.link_preview.cache.artifact_id`), and the
artifact route serves a blob only when the asking scope's cache still names that
id. Two accounts that previewed the same favicon therefore hold two different ids
and two different files, which is deliberate: the operator ruled that one
account's bytes must never be served into another account's session, and a store
that shared blobs by content would make that ruling a matter of luck rather than
of design.

The id is still ONLY lowercase hex — asserted here, before any path is built, and
again at the route — so a caller cannot walk out of the directory, and no filename
can contain a URL, a query string or a token.

NO SIDECAR METADATA, DELIBERATELY. The mime type is re-sniffed from the bytes on
every read and the dimensions are the caller's to read from the same header scan,
so the directory holds bytes and nothing else. The account and tenant that
produced a blob are recoverable from nothing on disk; they live in the in-process
cache entry that names it, and die with the process.

LRU WITHOUT A CLOCK OF ITS OWN. Eviction is by file mtime, refreshed on read —
the oldest blob is the one nobody has asked for recently, which is what LRU means
for a disk cache and needs no state that a crash could desynchronise. The bounds
are checked on write only, and the sweep aims below the bound rather than at it so
a burst of writes pays for one sweep instead of one per file. This is the SECOND
bound, behind the cache's resident-byte cap: the in-memory cap is the one that
decides what a renderer can still ask for, and this one decides how much disk an
install may spend on previews that nothing can name any more.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: Directory name under the config root. Not under the web-fetch cache: that one
#: is keyed by URL and its entries are JSON pointers, while these are bytes a
#: renderer will load, and keeping them apart keeps each store's sweep honest.
STORE_DIRNAME: Final = "link_preview_artifacts"

#: Total bytes and file count the store may hold. 64 MB is ~a thousand cards'
#: worth of og:images; the count bound is the one that matters more for a disk
#: with many small icons. Both are deliberately looser than the cache's own
#: 32 MiB resident cap: the cache is the authority on what is still addressable,
#: so the disk may legitimately hold blobs that are already unreachable.
MAX_STORE_BYTES: Final = 64 * 1024 * 1024
MAX_STORE_FILES: Final = 512

#: Sweep target as a fraction of the bounds. Sweeping to exactly the bound would
#: run the directory scan on every subsequent put.
_SWEEP_TARGET: Final = 0.8

#: An artifact id is 32 lowercase hex characters and NOTHING else. Anchored, no
#: separators, no dots: this pattern is the whole reason the artifact route cannot
#: be walked out of its directory, so it is checked before any path is built. It
#: is also the one definition of the id's LENGTH — the route shapes its path
#: parameter with it and the cache truncates its digest to the same width.
ARTIFACT_ID_PATTERN: Final = r"[0-9a-f]{32}"
_ARTIFACT_ID: Final = re.compile(rf"\A{ARTIFACT_ID_PATTERN}\Z")


class InvalidArtifactId(ValueError):
    """The caller supplied something that is not an artifact id.

    Raised BEFORE any filesystem call, so a malformed id costs one regex and can
    never become a path. The route also shapes the parameter, and the belt-and-
    braces check here is the one the store itself relies on.
    """


class PreviewStore:
    """One bounded directory of scope-addressed blobs."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = MAX_STORE_BYTES,
        max_files: int = MAX_STORE_FILES,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.max_files = max_files

    @classmethod
    def from_config(cls, config_dir: Path) -> PreviewStore:
        return cls(Path(config_dir) / STORE_DIRNAME)

    def put(self, body: bytes, *, artifact_id: str) -> str:
        """Store ``body`` under ``artifact_id`` and return the id.

        WRITES THROUGH A TEMPORARY FILE and ``os.replace`` so a concurrent reader
        never sees a half-written blob, and ALWAYS REPLACES: the id names a slot
        (this scope, this page, this kind), not a body, so a card refetched after
        its TTL must show the bytes that were just fetched rather than the ones
        the first fetch happened to see. No fsync: this is a cache, and a lost
        write costs one refetch, never correctness.
        """

        path = self._path(artifact_id)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / f".incoming-{artifact_id}"
            temporary.write_bytes(body)
            temporary.replace(path)
        except OSError:
            # A cache that cannot write is a cache miss, never a failed request:
            # the artefact is optional to the preview that wanted it.
            return artifact_id
        self.sweep()
        return artifact_id

    def get(self, artifact_id: str) -> bytes | None:
        """The stored bytes for ``artifact_id``, or ``None`` if absent.

        A hit refreshes the blob's mtime, which is the LRU touch. A blob that
        vanishes between the existence check and the read (a concurrent sweep) is
        a miss, not an error.
        """

        path = self._path(artifact_id)
        try:
            body = path.read_bytes()
        except OSError:
            return None
        try:
            path.touch()
        except OSError:
            pass
        return body

    def size_of(self, artifact_id: str) -> int:
        """The bytes this store holds for ``artifact_id``, or ``0`` if absent.

        Asked by the cache so its resident-byte accounting measures what is
        actually retained rather than what was sent. A ``stat``, never a read: the
        answer must not pull a 2 MB image through memory or refresh the LRU.
        """

        try:
            return self._path(artifact_id).stat().st_size
        except OSError:
            return 0

    def sweep(self) -> None:
        """Drop the least recently used blobs until the store is under its bounds."""

        try:
            entries = [entry for entry in self.root.iterdir() if entry.is_file()]
        except OSError:
            return
        total = 0
        sized: list[tuple[float, int, Path]] = []
        for entry in entries:
            if _ARTIFACT_ID.match(entry.name) is None:
                # A stray file is not ours to delete: only blobs this store wrote
                # (a 32-hex name) are swept, so a debugging copy a human dropped
                # in the directory survives.
                continue
            try:
                info = entry.stat()
            except OSError:
                continue
            total += info.st_size
            sized.append((info.st_mtime, info.st_size, entry))
        if total <= self.max_bytes and len(sized) <= self.max_files:
            return
        sized.sort()
        byte_target = int(self.max_bytes * _SWEEP_TARGET)
        file_target = int(self.max_files * _SWEEP_TARGET)
        count = len(sized)
        for _mtime, size, entry in sized:
            if total <= byte_target and count <= file_target:
                break
            try:
                entry.unlink()
            except OSError:
                continue
            total -= size
            count -= 1

    def _path(self, artifact_id: str) -> Path:
        if _ARTIFACT_ID.match(artifact_id) is None:
            raise InvalidArtifactId(artifact_id)
        return self.root / artifact_id
