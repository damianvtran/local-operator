"""Refuse a run root the operating system may purge under a live episode.

macOS purges ``/private/tmp`` on disk pressure and on a periodic sweep with no
warning and no regard for open handles. A purge mid-run destroyed a previous
paid pilot's prepared checkout, its 4.2 GB asset snapshot AND its output
directory, and left an EC2 instance running with nothing on disk naming it.
Every directory an episode must be able to read back after the fact -- the
evidence root, the artifact root, and above all the RESCUE root, which is the
only record of what to tear down -- has to live somewhere durable.

This is the one check both the adapter build script and the episode script
share, so the two cannot drift on what "volatile" means. The roots are
compared RESOLVED so ``/tmp -> /private/tmp`` on macOS cannot slip past a
prefix check on the spelled path; ``$TMPDIR`` is included because on macOS it
is a per-user directory under ``/var/folders`` that the same sweep covers.
"""

from __future__ import annotations

import os
from pathlib import Path


class VolatileRootError(ValueError):
    """The path resolves under a directory the OS may purge. Names the path."""


def volatile_roots() -> tuple[Path, ...]:
    """Directories the OS may purge under a live run, resolved."""

    roots = [Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path("/private/var/tmp")]
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        roots.append(Path(tmpdir))
    out: list[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except OSError:
            out.append(root)
    return tuple(out)


def refuse_volatile_root(path: Path, *, label: str = "root") -> None:
    """Raise ``VolatileRootError`` if ``path`` lives somewhere the OS may purge.

    ``label`` names the argument in the message (``inputs root``, ``rescue
    root``) so an operator with several roots on one command line learns
    which one to move.
    """

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for volatile in volatile_roots():
        if resolved == volatile or volatile in resolved.parents:
            raise VolatileRootError(
                f"{label} {path} resolves under {volatile}, which the OS may purge "
                "mid-run; use a durable location such as ~/worktrees/osworld"
            )
