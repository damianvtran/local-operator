"""Content-addressed, bounded artifact storage for link-preview bytes.

The id IS the sha256 of the bytes, which is what makes the artifact route safe:
there is no index, no path variable and no filename to coerce, so an id that is
forty-hex-and-nothing-else cannot name anything but one blob of its own content.
A caller that guesses an id gets an image the origin already served publicly —
never another message's text, never a tenant's data.

NO SIDECAR METADATA, DELIBERATELY. The mime type is re-sniffed from the bytes on
every read and the dimensions are the caller's to read from the same header scan,
so the directory holds bytes and nothing else. That is the strongest available
form of "the cache holds no message text, sender or tenant": it holds no
metadata at all, not even the account it was fetched for. Two accounts that
preview the same favicon share one blob, which is correct for a public asset.

LRU WITHOUT A CLOCK OF ITS OWN. Eviction is by file mtime, refreshed on read —
the oldest blob is the one nobody has asked for recently, which is what LRU
means for a disk cache and needs no state that a crash could desynchronise. The
bounds are checked on write only, and the sweep aims below the bound rather than
at it so a burst of writes pays for one sweep instead of one per file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

#: Directory name under the config root. Not under the web-fetch cache: that one
#: is keyed by URL and its entries are JSON pointers, while these are bytes a
#: renderer will load, and keeping them apart keeps each store's sweep honest.
STORE_DIRNAME: Final = "link_preview_artifacts"

#: Total bytes and file count the store may hold. 64 MB is ~a thousand cards'
#: worth of og:images; the count bound is the one that matters more for a disk
#: with millions of small favicons.
MAX_STORE_BYTES: Final = 64 * 1024 * 1024
MAX_STORE_FILES: Final = 512

#: Sweep target as a fraction of the bounds. Sweeping to exactly the bound would
#: run the directory scan on every subsequent put.
_SWEEP_TARGET: Final = 0.8

#: An artifact id is a lowercase sha256 hex digest and NOTHING else. Anchored, no
#: separators, no dots: this pattern is the whole reason the artifact route cannot
#: be walked out of its directory, so it is checked before any path is built.
ARTIFACT_ID_PATTERN: Final = r"[0-9a-f]{64}"
_ARTIFACT_ID: Final = re.compile(rf"\A{ARTIFACT_ID_PATTERN}\Z")


class InvalidArtifactId(ValueError):
    """The caller supplied something that is not an artifact id.

    Raised BEFORE any filesystem call, so a malformed id costs one regex and can
    never become a path. The route also shapes the parameter, and the belt-and-
    braces check here is the one the store itself relies on.
    """


class PreviewStore:
    """One bounded directory of content-addressed blobs."""

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

    def put(self, body: bytes) -> str:
        """Store ``body`` and return its artifact id.

        Writes through a temporary file and ``os.replace`` so a concurrent reader
        never sees a half-written blob, and so two processes that fetched the
        same favicon at once both end with the same complete file. No fsync: this
        is a cache, and a lost write costs one refetch, never correctness.
        """
        artifact_id = hashlib.sha256(body).hexdigest()
        path = self._path(artifact_id)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not path.exists():
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
                # (a 64-hex name) are swept, so a debugging copy a human dropped
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
