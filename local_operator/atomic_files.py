"""One temp-then-replace writer for every whole-file publish under ``local_operator/``.

WHY THIS MODULE EXISTS. Dozens of writers publish a file by writing a sibling
temp and ``os.replace``-ing it over the target, so a reader sees the old file or
the new one, never half of either. That only holds when each WRITE owns its
temp. Most of them named the temp from ``os.getpid()`` alone, and every thread of
a process shares that pid: two threads publishing the same target (the server's
worker threads, ``asyncio.to_thread`` refreshes, a TUI and its background
indexer) opened the SAME temp. The second truncates and streams into a file the
first has already renamed into place -- publishing a torn document -- and its own
``os.replace`` then raises ``FileNotFoundError`` because the temp is gone. PR
#4501 measured it in the Library (22 undecodable thumbnails across 8 threads x 50
iterations on main, 0 of 800 with per-write names) and fixed it there; this
module is that fix lifted out so the whole class closes in ONE place instead of
one patch per call site.

THE NAME. ``<target name>.<pid>.<thread id>.<random>.tmp``: the pid stays first
so a crashed process's leftovers are attributable and the existing dead-writer
sweeps (which parse it) keep working; the thread id and 32 random bits make two
writes in one process distinct even when a thread id is reused. ``hidden`` adds
a dot prefix for directories whose listing must not show a temp; ``marker``
keeps a site's legacy ``<name>.<marker>-<token>`` shape where a sweeper globs
for it (``update.py``'s ``current.tmp-*``).

CONSTRAINTS.

- The temp is a SIBLING of the target, never ``tempfile``'s directory, so the
  final ``os.replace`` is a same-filesystem atomic rename. Callers create the
  parent directory themselves, exactly as they did before; this module never
  ``mkdir``s, so a site that relied on a missing directory raising still does.
- No durability change: ``fsync`` happens only when the caller asks, i.e. only
  at sites that already fsynced.
- ``mode`` is the temp's mode FROM CREATION (``os.open`` with the mode, then
  ``fchmod`` so the umask cannot change it), never write-then-chmod: a secret
  is never readable by anyone else for even a moment. ``os.replace`` keeps it.
  ``mode=None`` is ``open(path, "w")``'s ``0o666 & ~umask``.
- On ANY failure (including ``KeyboardInterrupt``) the temp is removed and the
  error re-raised unchanged, so a caller's existing ``except OSError`` (or lack
  of one) behaves as before, minus the stranded temp.
- Text is encoded and written as bytes: no newline translation, which on POSIX
  is byte-identical to ``Path.write_text``.

HOW TO USE. Whole payload in hand: :func:`atomic_write_bytes` /
:func:`atomic_write_text`. Symlink flip: :func:`atomic_symlink`. A writer that
must fill the temp itself (SQLite, a copy fallback): name it with
:func:`unique_temp`, and on failure remove it with :func:`discard_temp` -- the one
cleanup the session-deletion guard allow-lists for this class.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

__all__ = [
    "atomic_symlink",
    "atomic_write_bytes",
    "atomic_write_text",
    "discard_temp",
    "unique_temp",
    "unique_token",
]


def unique_token() -> str:
    """``<pid>.<thread id>.<8 hex>``: distinct per write across threads and processes."""
    return f"{os.getpid()}.{threading.get_ident()}.{os.urandom(4).hex()}"


def unique_temp(
    target: str | os.PathLike[str], *, hidden: bool = False, marker: str | None = None
) -> Path:
    """A sibling of ``target`` no other write -- thread or process -- will pick.

    ``hidden`` prefixes a dot. ``marker`` yields ``<name>.<marker>-<token>``
    instead of ``<name>.<token>.tmp``, for a site whose leftovers a sweeper finds
    by that legacy shape.
    """
    path = Path(target)
    prefix = "." if hidden else ""
    if marker is not None:
        return path.with_name(f"{prefix}{path.name}.{marker}-{unique_token()}")
    return path.with_name(f"{prefix}{path.name}.{unique_token()}.tmp")


def discard_temp(temp: str | os.PathLike[str]) -> None:
    """Remove this writer's own temp, if it is still there. Never raises.

    Only ever handed a name :func:`unique_temp` built, so it cannot reach a
    target, a peer's temp, or a directory (``unlink`` refuses directories).
    """
    try:
        os.unlink(temp)
    except OSError:
        pass


def atomic_write_bytes(
    target: str | os.PathLike[str],
    data: bytes,
    *,
    mode: int | None = None,
    fsync: bool = False,
    hidden: bool = False,
    marker: str | None = None,
) -> None:
    """Publish ``data`` at ``target`` via a per-write sibling temp + ``os.replace``.

    Raises whatever the write or rename raised, after removing the temp.
    """
    temp = unique_temp(target, hidden=hidden, marker=marker)
    try:
        # O_EXCL: the name is unique, so an existing file there is never ours to
        # overwrite -- fail rather than write through someone else's inode.
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode)
        with os.fdopen(fd, "wb") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            handle.write(data)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        discard_temp(temp)
        raise


def atomic_write_text(
    target: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
    fsync: bool = False,
    hidden: bool = False,
    marker: str | None = None,
) -> None:
    """:func:`atomic_write_bytes` for ``text`` encoded with ``encoding``."""
    atomic_write_bytes(
        target, text.encode(encoding), mode=mode, fsync=fsync, hidden=hidden, marker=marker
    )


def atomic_symlink(
    link: str | os.PathLike[str],
    source: str | os.PathLike[str],
    *,
    target_is_directory: bool = False,
    hidden: bool = False,
    marker: str | None = None,
) -> None:
    """Point ``link`` at ``source`` with a rename, so no reader sees a gap.

    The link is created at a per-write sibling name and renamed over ``link``;
    two threads flipping one link each rename their own, and the later wins.
    """
    temp = unique_temp(link, hidden=hidden, marker=marker)
    try:
        os.symlink(source, temp, target_is_directory=target_is_directory)
        os.replace(temp, link)
    except BaseException:
        discard_temp(temp)
        raise
