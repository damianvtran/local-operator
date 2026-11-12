"""The viewer's bounded, evictable cache of bytes pulled from a peer's session.

docs/design/remote-files-parity.md §4. Laid out in the AttachmentStore format
(``<d>.bin`` + ``<d>.json`` + an optional ``<d>.peaks.json``) so the store's own
``adopt_file`` lands into it and ``waveform_payload`` reads it unchanged — but it
is a SEPARATE directory from ``<config>/attachments``: that store is unbounded and
authoritative (a local transcript resolves its images from it), so evicting there
could delete a local session's only copy. Everything here can be re-fetched by
construction, which is what makes deleting it safe.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

#: ``desktop.remote_files_cache_bytes``'s default. Basis (§4): a typical active set
#: of 16 sessions x median 6 referenced digests x p90 187.5 KiB is ~18 MiB, so 1 GiB
#: is ~55x that, 3.8x the largest measured pad (269 MiB) and 32 files at the
#: per-file preview ceiling (32 MiB).
REMOTE_FILES_CACHE_BYTES = 1024 * 1024 * 1024

#: The configuration key and its nested path in ``config.yml``.
CACHE_BYTES_KEY = ("desktop", "remote_files_cache_bytes")

#: Where the cache lives, relative to the config root.
CACHE_PARTS: tuple[str, ...] = ("cache", "remote-files")

_SIDECARS = (".json", ".peaks.json")
_GUARD = threading.Lock()

#: The AttachmentStore's content address (``sha256[:32]``, lower-case hex).
DIGEST = re.compile(r"[0-9a-f]{32}")
#: ``desktop_sessions.SESSION_ID``'s shape, restated (as ``session_files.family``
#: does) so this module stays off the desktop module's import path.
SESSION_ID = re.compile(r"[a-f0-9]{12}")


def is_digest(value: object) -> bool:
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def is_session_id(value: object) -> bool:
    return isinstance(value, str) and SESSION_ID.fullmatch(value) is not None


def cache_root(root: Path) -> Path:
    return Path(root).joinpath(*CACHE_PARTS)


def configured_budget(root: Path) -> int:
    """The byte budget from ``config.yml``, read per pass (the key is LIVE).

    RAW YAML, never ``ConfigManager``: its constructor can move a malformed file
    aside, and an eviction pass must never be what renames a user's config (the
    rule ``desktop_feed._read_config_values`` documents). Anything unreadable or
    out of range reads as the default.
    """
    try:
        import yaml

        from local_operator.config import CONFIG_FILE_NAME

        data = yaml.safe_load((Path(root) / CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing or broken config is the default
        return REMOTE_FILES_CACHE_BYTES
    value: Any = data
    for part in CACHE_BYTES_KEY:
        value = value.get(part) if isinstance(value, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return REMOTE_FILES_CACHE_BYTES
    return value


@dataclass(frozen=True)
class CachedBlob:
    """One blob a lookup found: where, how big, and what it is."""

    path: Path
    size: int
    mime: str
    #: Which step of the lookup order found it: ``store``, ``replica`` or ``cache``.
    source: str


def _mime_beside(path: Path) -> str:
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "application/octet-stream"
    if isinstance(meta, dict) and isinstance(meta.get("mime_type"), str) and meta["mime_type"]:
        return str(meta["mime_type"])
    return "application/octet-stream"


def locations(root: Path, session_id: str, digest: str) -> list[tuple[str, Path]]:
    """The lookup order of §1, as ``(source, path)``: store, replica, cache.

    Both names become path components, and a digest can arrive from ANOTHER device
    (an owner's listing or row reply), so the shapes are checked here as well as at
    the callers: no caller can build a path out of ``../..`` (#3872 R1-1).
    """
    from local_operator.network import sync
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME

    if not is_digest(digest):
        raise ValueError("not a content digest")
    if not is_session_id(session_id):
        raise ValueError("not a session id")
    root = Path(root)
    return [
        ("store", root / ATTACHMENTS_DIRNAME / f"{digest}.bin"),
        ("replica", sync.replica_dir(root, session_id) / ATTACHMENTS_DIRNAME / f"{digest}.bin"),
        ("cache", cache_root(root) / f"{digest}.bin"),
    ]


def lookup(root: Path, session_ids: tuple[str, ...], digest: str) -> CachedBlob | None:
    """The first local copy of ``digest`` in the lookup order, by ``stat`` only.

    ``session_ids`` names every replica that may hold it (a child's bytes may sit in
    its own replica or its parent's). A cache hit is touched, because LRU is by
    ``.bin`` mtime: atime is unreliable under ``relatime``/``noatime``.

    A malformed digest names nothing on this disk, so it is a miss; malformed
    session ids are skipped (their replica cannot exist).
    """
    if not is_digest(digest):
        return None
    seen: set[Path] = set()
    for session_id in session_ids:
        if not is_session_id(session_id):
            continue
        for source, path in locations(root, session_id, digest):
            if path in seen:
                continue
            seen.add(path)
            try:
                info = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            if source == "cache":
                touch(path)
            return CachedBlob(path=path, size=info.st_size, mime=_mime_beside(path), source=source)
    return None


def touch(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.utime(path)


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """One eviction pass at a time, across processes (``flock``) and threads."""
    directory.mkdir(parents=True, exist_ok=True)
    with _GUARD:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows: the thread guard alone
            yield
            return
        with open(directory / ".lock", "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def total_bytes(root: Path) -> int:
    directory = cache_root(root)
    total = 0
    with contextlib.suppress(OSError):
        for entry in os.scandir(directory):
            if entry.name.endswith(".bin"):
                with contextlib.suppress(OSError):
                    total += entry.stat().st_size
    return total


def evict(root: Path, *, budget: int | None = None, keep: frozenset[str] = frozenset()) -> int:
    """Delete the least recently used blobs until the cache fits ``budget``.

    Returns the bytes freed. ``keep`` names digests a pass must not evict (a precache
    pass never evicts what it landed itself). No in-flight pin is needed: staging
    parts live under ``network/staging/files`` and a route serves from an fd it
    already opened, which an unlink cannot take away (POSIX keeps the inode alive).
    """
    directory = cache_root(root)
    if not directory.is_dir():
        return 0
    limit = configured_budget(root) if budget is None else budget
    freed = 0
    with _locked(directory):
        blobs: list[tuple[int, int, str]] = []
        total = 0
        with contextlib.suppress(OSError):
            for entry in os.scandir(directory):
                if not entry.name.endswith(".bin"):
                    continue
                with contextlib.suppress(OSError):
                    info = entry.stat()
                    blobs.append((info.st_mtime_ns, info.st_size, entry.name[: -len(".bin")]))
                    total += info.st_size
        for _mtime, size, digest in sorted(blobs):
            if total <= limit:
                break
            if digest in keep:
                continue
            for suffix in (".bin", *_SIDECARS):
                with contextlib.suppress(FileNotFoundError):
                    (directory / f"{digest}{suffix}").unlink()
            total -= size
            freed += size
    if freed:
        logger.debug("remote files cache: evicted %d bytes (budget %d)", freed, limit)
    return freed
