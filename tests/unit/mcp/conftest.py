"""Shared isolation for the MCP unit tests.

One piece of state here outlives a single test and is process-global by design:
``auth.REFRESH_CONTENTION`` is the side channel that carries "we refused to
spend a refresh token without the lock" past a transport that rewrites auth-flow
failures into bare cancellations. A record left armed by a test would be
attributed to the next test's unrelated cancellation and turn it into a retry,
so it is cleared on both sides of every test rather than inside the tests that
happen to care.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from local_operator.mcp.auth import REFRESH_CONTENTION


@pytest.fixture(autouse=True)
def _no_leaked_refresh_contention_records() -> Iterator[None]:
    """Clear the contention ledger around every MCP unit test."""
    REFRESH_CONTENTION.clear()
    yield
    REFRESH_CONTENTION.clear()
