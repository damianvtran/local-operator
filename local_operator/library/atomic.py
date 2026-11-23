"""Per-write temp names for the Library's atomic cache writers.

Every Library cache file (thumbnail WebPs, waveform JSON, the catalog index) is
published with write-to-temp + ``os.replace`` so a reader never sees a half
written file. That only holds when each WRITE owns its temp: a name built from
``os.getpid()`` alone is shared by every thread of the process, and the Library
runs several writers in one process (the ``library-thumbs`` indexer thread, the
on-demand route, concurrent grid renders). With a shared name the second writer
truncates and streams into a file the first has already renamed into place --
publishing a torn file -- and its own ``os.replace`` then fails because the temp
is gone. PR #4501's QA harness (8 caller threads plus the ``library-thumbs``
worker, 50 iterations) measured the torn case on main: 22 callers handed a
WebP PIL could not decode; with per-write names, 0 of 800 calls.

The pid stays in the name so a crashed process's leftovers are attributable;
the thread id and random suffix make concurrent writes in one process distinct.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


def unique_temp(target: Path, *, hidden: bool = False) -> Path:
    """A sibling of ``target`` no other write -- thread or process -- will pick.

    Sibling, not ``tempfile``'s directory, so ``os.replace`` stays a same-
    filesystem atomic rename. ``hidden`` keeps a dot prefix for writers whose
    directory listing must not show the temp.
    """
    prefix = "." if hidden else ""
    token = f"{os.getpid()}.{threading.get_ident()}.{os.urandom(4).hex()}"
    return target.with_name(f"{prefix}{target.name}.{token}.tmp")
