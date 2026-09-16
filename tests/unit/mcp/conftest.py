"""Shared isolation for the MCP unit tests.

Two pieces of state here outlive a single test and are process-global by
``local_operator.mcp`` design, so both are cleared around every test in this
package rather than inside the tests that happen to care.

``auth.REFRESH_CONTENTION`` is the side channel that carries "we refused to
spend a refresh token without the lock" past a transport that rewrites auth-flow
failures into bare cancellations. A record left armed by a test would be
attributed to the next test's unrelated cancellation and turn it into a retry.

``redaction``'s registry holds every resolved MCP credential for the rest of the
PROCESS, and it is what the log filter and every ``scrub()`` sink read, so a
value left in it by a test is applied to unrelated records in every later test
of the same worker.

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
def _no_leaked_redaction_registrations() -> Iterator[None]:
    """Leave the process-wide MCP redaction set as this test found it.

    ``local_operator.mcp.redaction``'s store is global BY DESIGN (a value has to
    be scrub-able from whatever thread resolves or logs it), which means a value
    a test registers outlives the test and is applied to every later record in
    that worker. Measured, not assumed: with this fixture absent, a full
    ``tests/unit/mcp`` run finishes with ``['SENTINEL-VALUE']`` still registered,
    and a three-character value registered by
    ``test_store_validates_all_ids_and_confirms_replacement`` rewrote an
    unrelated provider-failover warning two files away — the suite went
    run-order dependent (agent review R-1 / QA Q1). ``MIN_SCRUBBED_LENGTH`` and
    the bounded filter remove the damage; this removes the leakage, because a
    test that leaves global state behind is a test that reports on whichever
    test happened to run first.

    Package-wide rather than in the one file that registers today: the leak is
    the *store's* property, so the next test file to resolve a credential
    inherits the isolation without having to remember it.
    """
    from local_operator.mcp import redaction

    before = set(redaction.values())
    yield
    for value in set(redaction.values()) - before:
        redaction.unregister(value)


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
