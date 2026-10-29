"""The shared net wait's backstop, proven able to fail.

WHY THIS FILE EXISTS. The cells in this package wait on conditions that are, on a
host that is merely loaded rather than starved, already true by the time the wait is
reached — so forcing ``LO_TEST_NET_WAIT_S`` low does NOT turn
``test_membership_convergence`` red (measured here at load ~90: the whole file passes
with the budget at 0.05 s). That run is still worth having — it shows the budget is a
backstop and not the assertion — but it cannot show the backstop BITES, and a guard
that cannot go red is worse than none (AGENTS.md, "Prove the test can still fail").

These cells pin the failure path directly, off any host's load: a predicate that never
becomes true must come back falsy, and the message must carry the three things that
separate "slow machine" from "the work never happened" — the elapsed time, the budget
it expired against, and the predicate itself. The passing path is pinned too, because
that is where the drift signal lives: every satisfied wait reports its measurement.
"""

from __future__ import annotations

import pytest

from tests.unit.network import conftest as net_fixtures


def test_the_default_budget_is_the_one_shared_constant() -> None:
    # The whole point of the centralisation: a call site that says nothing gets THIS
    # number, so a new wait cannot pick its own wall-clock slice.
    assert net_fixtures.wait_for(lambda: True).budget_s == net_fixtures.NET_WAIT_BUDGET_S


def test_a_wait_that_never_holds_is_falsy_and_says_why() -> None:
    outcome = net_fixtures.wait_for(lambda: False, timeout_s=0.2)

    assert not outcome, "a predicate that never held was reported as satisfied"
    message = repr(outcome)
    assert "EXPIRED" in message
    assert "still false" in message
    # The elapsed time, the budget, and what was still false — the three facts the old
    # bare ``False`` threw away.
    assert "after 0.2" in message
    assert "budget 0.2s" in message
    assert "lambda" in message


def test_a_satisfied_wait_is_truthy_and_carries_its_measurement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("DEBUG", logger=net_fixtures.__name__):
        outcome = net_fixtures.wait_for(lambda: True)

    assert outcome
    assert outcome.elapsed_s >= 0.0
    # The drift signal: a cell creeping toward the backstop is visible on the PASSING
    # path, before it reds on a loaded host.
    assert any("net wait satisfied" in record.message for record in caplog.records)
