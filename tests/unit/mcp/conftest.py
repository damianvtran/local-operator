"""Shared isolation for the MCP unit tests.

One piece of state here outlives a single test and is process-global by design:
``auth.REFRESH_CONTENTION`` is the side channel that carries "we refused to
spend a refresh token without the lock" past a transport that rewrites auth-flow
failures into bare cancellations. A record left armed by a test would be
attributed to the next test's unrelated cancellation and turn it into a retry,
so it is cleared on both sides of every test rather than inside the tests that
happen to care.

Imported INSIDE the fixture, deliberately. The module-level form of this import
made the branch's first commit uncollectable on its own: that commit adds these
tests but not ``REFRESH_CONTENTION``, which the fix commit introduces, so
``pytest`` aborted during conftest collection and the failing-case evidence its
message quotes could not be reproduced from that commit at all. Deferring the
import means the pre-fix tree collects, runs, and reports the three failures the
commit documents — a repro commit has to be runnable on the tree it reproduces.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_refresh_contention_records() -> Iterator[None]:
    """Clear the contention ledger around every MCP unit test."""
    try:
        from local_operator.mcp.auth import REFRESH_CONTENTION
    except ImportError:  # pragma: no cover - the pre-fix tree has no ledger yet
        # No ledger to clear: this fixture exists to isolate process-global
        # state that does not exist on that tree.
        yield
        return
    REFRESH_CONTENTION.clear()
    yield
    REFRESH_CONTENTION.clear()
