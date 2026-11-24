"""Test helpers must CLOSE the sqlite connections they open.

Why: ``with sqlite3.connect(path) as conn:`` reads like a scoped open but is
not one -- the connection's context manager only commits or rolls back; it
never closes. And CPython's ``sqlite3.Connection`` sits in a reference cycle
with its own statement cache, so an unclosed one keeps the database file and
its WAL sidecars open until a generation-2 garbage collection. Across an xdist
worker those descriptors pile up (the ``[fd-leak]`` reporter in the root
conftest named ``team_bridge/test_body_turn_e2e.py`` +346 and
``team_bridge/test_screening_status.py`` +290 in full-suite run 45360774846),
which is the headroom ``select.select`` loses at fd 1024.

The spelling used instead is ``with closing(sqlite3.connect(path)) as conn,
conn:`` -- ``closing`` closes, the inner ``conn`` keeps the commit-on-success
semantics the bare form had.

Scope: the files this rule has been applied to. Other test files still use the
bare form; add a file here when converting it, so it cannot regress.
"""

from __future__ import annotations

import gc
import os
import re
import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: A ``with`` statement whose context manager is a bare ``sqlite3.connect(...)``.
_BARE_WITH = re.compile(r"\bwith\s+sqlite3\.connect\(")

CONVERTED = (
    "tests/unit/team_bridge/test_body_turn_e2e.py",
    "tests/unit/team_bridge/test_screening_status.py",
)


def test_converted_files_never_use_the_non_closing_with_form() -> None:
    offenders = [
        f"{relative}:{number}"
        for relative in CONVERTED
        for number, line in enumerate((REPO / relative).read_text().splitlines(), start=1)
        if _BARE_WITH.search(line)
    ]
    assert offenders == [], (
        "`with sqlite3.connect(...) as conn` commits but never closes; use "
        "`with closing(sqlite3.connect(...)) as conn, conn:` instead: " + ", ".join(offenders)
    )


def _open_fd_count() -> int:
    for candidate in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(candidate):
            return len(os.listdir(candidate))
    pytest.skip("no fd directory on this platform")


def test_the_bare_with_form_really_leaves_the_descriptor_open(tmp_path: Path) -> None:
    """Negative control: the rule above guards a real leak, not a style."""
    database = tmp_path / "probe.sqlite3"
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        baseline = _open_fd_count()
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
        del conn
        assert _open_fd_count() > baseline
        gc.collect()
        assert _open_fd_count() == baseline
    finally:
        if was_enabled:
            gc.enable()
