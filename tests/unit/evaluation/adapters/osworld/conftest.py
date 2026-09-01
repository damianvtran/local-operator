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

import sys
import uuid
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[5] / "benchmarks" / "osworld_v2_adapter" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Drop any already-imported adapter modules so the source-tree copies win over
# a previously installed wheel in this interpreter.
for _name in [name for name in sys.modules if name.startswith("lop_osworld_v2_adapter")]:
    del sys.modules[_name]


@pytest.fixture
def episode_id() -> str:
    return f"ep-{uuid.uuid4().hex[:12]}"
