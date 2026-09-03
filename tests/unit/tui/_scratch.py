"""Tracked throwaway directories for the TUI test modules.

WHY THIS EXISTS. These modules used bare ``tempfile.mkdtemp()`` for registry
roots, and nothing ever unlinks one — measured at 65 abandoned directories per
full-suite run across the five files that use this helper (part of the 71/run
that accumulated into ~43,627 directories on one machine).

WHY A PLAIN FUNCTION AND NOT A FIXTURE. The first attempt at this fix migrated
the call sites to a ``tmp_path``-backed fixture. That added fixture
dependencies to these modules, which perturbed xdist's ``worksteal``
scheduling and landed the timing-sensitive pilot test
``test_app_pilot.py::test_a_swap_leaves_the_ledger_matching_the_new_sessions_history``
in a bad window: 4/4 failures in CI on the branch, 0/N on main, reproduced
locally in CI's exact config (``-n 4 --dist worksteal``, py3.12 + ``--cov``).
Reverting the fixture-based migration restored green. A plain function leaves
every test signature and the whole fixture graph untouched, so the scheduling
shape of the shard is exactly what it was before the leak fix.

Each call hands out a fresh directory under a tracked
``TemporaryDirectory``. The pool holds strong references so nothing is
finalized mid-test; ``TemporaryDirectory`` registers its cleanup with
``weakref.finalize``, which runs at interpreter shutdown, so every directory
is reclaimed when the (xdist worker) process exits. Directories therefore live
exactly as long as the run — long enough for every test, never longer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

#: Strong references keep the tracked directories alive for the whole worker
#: process; their finalizers remove them at interpreter shutdown.
_POOL: list[tempfile.TemporaryDirectory[str]] = []


def scratch_dir() -> Path:
    """A fresh throwaway directory, reclaimed when the worker process exits.

    Drop-in replacement for ``Path(tempfile.mkdtemp())`` at the call sites in
    this package: same shape (a unique empty directory under the temp root),
    but someone — the finalizer — actually removes it.
    """
    tracked = tempfile.TemporaryDirectory()
    _POOL.append(tracked)
    return Path(tracked.name)
