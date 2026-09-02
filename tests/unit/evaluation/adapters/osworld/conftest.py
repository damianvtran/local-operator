"""Fixtures for the OSWorld adapter tests.

The ``episode_id`` fixture lives in the runner's conftest, but pytest does not
resolve fixtures across sibling package boundaries, so the OSWorld suite
provides its own — a fresh, unique ID per test, because the lifecycle's
authority lineage is process-global per episode ID.

The adapter source tree is prepended to ``sys.path`` at import time so the
unit tests exercise the CURRENT source, not a stale installed wheel. The wheel
itself is exercised separately and genuinely in ``test_spawn.py``, which
builds and installs the real artifact into a spawned interpreter. Testing the
source here and the wheel there are two different, both-needed guarantees: the
unit tests track edits without a rebuild; the spawn test proves the shipped
artifact loads.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[5] / "benchmarks" / "osworld_v2_adapter" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Drop any already-imported adapter modules so the source-tree copies win over
# a previously installed wheel in this interpreter.
for _name in [name for name in sys.modules if name.startswith("lop_osworld_v2_adapter")]:
    del sys.modules[_name]


@pytest.fixture(autouse=True)
def _restore_cwd() -> Iterator[None]:
    """Return every test in this package to the cwd it started in.

    ``reset_start`` moves the process into the episode scratch dir (see
    ``adapter._enter_episode_scratch``) and ``close`` moves it back — but a
    test that exercises ``reset_start`` WITHOUT reaching ``close`` leaves the
    worker parked in a ``tmp_path`` that pytest deletes immediately
    afterwards. Every later test in that worker then runs from a deleted
    directory.

    That is not hypothetical, and it escapes this package: it broke
    ``evidence/test_verify.py::test_sparse_oversized_journal_and_artifact_are_bounded``,
    which shells out with ``python -c`` and no ``cwd=``, so the child
    inherited the stale cwd and died with ``No module named 'tests'``.
    ``monkeypatch.chdir`` cannot cover this — it restores the cwd as of when
    it was called, which is before the adapter moves it again.

    Package-scoped and autouse because fifteen call sites across this
    directory call ``reset_start`` without ``close``; anchoring here is what
    keeps an adapter-internal chdir from leaking into unrelated suites.
    """

    origin = Path.cwd()
    try:
        yield
    finally:
        os.chdir(origin)


@pytest.fixture
def episode_id() -> str:
    return f"ep-{uuid.uuid4().hex[:12]}"
