"""Count the descriptors this process holds on SPECIFIC files.

Why not count every open fd: a test process is never single-threaded. In an
xdist worker the library indexer, file watchers and daemon threads leaked by
earlier tests open and close unrelated descriptors at any moment, so
``len(os.listdir("/proc/self/fd"))`` taken twice differs by noise even when
the code under test is perfect. PR #4539's full-suite gate failed exactly so:
run 45381339350, job 138402393965 (``test (3.13, 1)``),
``test_the_bare_with_form_really_leaves_the_descriptor_open`` with ``assert
19 == 18``. A leak test that flakes is the class of bug it exists to remove.

So the probe asks a question only the code under test can change: how many
open descriptors refer to THESE files, identified by ``(st_dev, st_ino)``. A
concurrent thread opening some other file can never match.

Constraints:

- sqlite sidecars (``-wal``, ``-shm``, ``-journal``) appear lazily, so the
  targets are re-statted on every call; a sidecar that does not exist yet
  simply matches nothing.
- A descriptor listed in the fd directory may be closed by another thread
  before it is statted (or, on macOS, be the directory handle ``listdir``
  itself used); ``OSError`` from ``fstat`` skips it. If the number was reused
  for a different file meanwhile, it does not match either.
- Linux exposes ``/proc/self/fd``, macOS ``/dev/fd``; anywhere else the
  calling test is skipped, as the process-wide counters it replaces were.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

#: Files sqlite opens next to a database, depending on its journal mode.
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def _fd_dir() -> str:
    for candidate in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(candidate):
            return candidate
    pytest.skip("no fd directory on this platform")


def _identities(paths: Iterable[Path | str]) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for path in paths:
        for suffix in ("", *SQLITE_SIDECARS):
            try:
                stat = os.stat(f"{os.fspath(path)}{suffix}")
            except OSError:
                continue
            identities.add((stat.st_dev, stat.st_ino))
    return identities


def open_fds_on(paths: Iterable[Path | str]) -> int:
    """Open descriptors on each database in ``paths`` or its sqlite sidecars."""
    fd_dir = _fd_dir()
    identities = _identities(paths)
    if not identities:
        return 0
    count = 0
    for name in os.listdir(fd_dir):
        try:
            stat = os.fstat(int(name))
        except (ValueError, OSError):
            continue
        if (stat.st_dev, stat.st_ino) in identities:
            count += 1
    return count
