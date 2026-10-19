"""Asset identity: the key, the opaque id, and the display-safe reference.

The design fixes three rules, and each exists to prevent a specific defect:

1. **Path key** ``p:<realpath>`` — a file named through two spellings, a
   symlink, or a relative path from two sessions is ONE asset. ``realpath``
   is resolved when the file exists; a missing file keeps its normalised
   absolute spelling, because two missing spellings cannot be proven equal.
2. **Digest key** ``a:<digest>`` — content-addressed media has no path at
   all until a surface materialises it, and the digest is already the store's
   identity.
3. **Asset id** ``la_`` + ``sha256(key)[:16]`` — opaque, stable and URL-safe.
   The id is what every route and surface carries; the key never leaves the
   backend, and an absolute path never appears in a phone projection.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Final

ASSET_ID_PREFIX: Final = "la_"
_ASSET_ID_RE: Final = re.compile(r"^la_[0-9a-f]{16}$")


def asset_id(key: str) -> str:
    """The opaque id for a canonical key. Deterministic across processes."""
    return ASSET_ID_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def is_asset_id(value: str) -> bool:
    """True for a syntactically valid opaque id; never consults the store."""
    return bool(value) and _ASSET_ID_RE.match(value) is not None


def path_key(path: str) -> str:
    """``p:<realpath>`` for a path the scanner recorded.

    ``realpath`` is best-effort and deliberately never raises: a dangling
    symlink on a removed volume still has a stable textual spelling, and a
    catalog read must not fail because one file moved.
    """
    if not path:
        raise ValueError("an empty path has no asset key")
    absolute = os.path.abspath(path)
    try:
        resolved = os.path.realpath(absolute)
    except (OSError, ValueError):
        resolved = absolute
    return f"p:{resolved}"


def digest_key(digest: str) -> str:
    if not digest:
        raise ValueError("an empty digest has no asset key")
    return f"a:{digest}"


def key_for(path: str | None, digest: str | None) -> str:
    """The canonical key for a scanned reference. Digest wins when both exist.

    A record never carries both today (a path row has no digest, an artifact
    row has no path), but the ordering is fixed so a future producer that
    writes both cannot make the identity depend on field order.
    """
    if digest:
        return digest_key(digest)
    if path:
        return path_key(path)
    raise ValueError("a reference needs a path or a digest")
