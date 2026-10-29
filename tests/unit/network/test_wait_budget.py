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

And the bite itself is measured: a predicate that becomes true LATE, under two budgets,
so the same wait fails on one and succeeds on the other. See
``test_the_budget_decides_a_delayed_condition_in_both_directions`` for why that cell is
the only one here that can distinguish the budget from the predicate.
"""

from __future__ import annotations

import threading
import time

import pytest

from tests.unit.network import conftest as net_fixtures

#: How long the delayed flag in the cell below takes to become true, with the two
#: budgets measured around it. 12 s leaves ~11.5 s of headroom under the long budget
#: (so "satisfied at the budget" is distinguishable from "satisfied at 12 s") and 11.5 s
#: of visible falsehood above the short one, while staying cheap enough to run per-push.
FLAG_DELAY_S = 12.0
SHORT_BUDGET_S = 0.5
LONG_BUDGET_S = 30.0


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


def test_the_budget_decides_a_delayed_condition_in_both_directions() -> None:
    """Prove the BUDGET is the mechanism: one predicate, two budgets, opposite outcomes.

    WHY THIS CELL EXISTS. The obvious negative control — run the network cells with
    ``LO_TEST_NET_WAIT_S=0.5`` and watch them red — proves nothing on a host that is
    merely loaded: those predicates are ALREADY TRUE when the wait is reached (measured:
    the whole file passes at 0.05 s), so the budget never gets a chance to bite and the
    cells stay green at any value. ``test_a_wait_that_never_holds_is_falsy_and_says_why``
    is the other half of that gap: it shows the helper CAN fail, but a predicate that is
    false forever would fail against any budget at all, so it cannot show the budget
    changed the outcome either. Without a condition that becomes true LATE, the claim
    "the 60 s budget was the mechanism that fixed the fleet reds" is unfalsifiable: a
    reader cannot tell a backstop that fired at 12 s from one that never fired and let
    the predicate win at 60 s.

    THE CELL. A driver thread sets a flag after ``FLAG_DELAY_S`` — no network, no relay,
    no fixtures, no clock assumptions about this host other than "12 s is longer than
    0.5 s". The SAME predicate is then waited on twice through the shipped
    ``wait_for``/``wait_outcome`` path (``timeout_s`` is the parameter
    ``LO_TEST_NET_WAIT_S`` feeds — one call site, so the budget cannot be smuggled in
    from elsewhere, and passing it explicitly is what makes the cell independent of
    import-time env capture).

    1. Under the short budget the wait EXPIRES while the flag is still false, and the
       message carries the elapsed time and the not-yet-true predicate — the facts a red
       shard log needs to be readable as "slow machine, real work" rather than as a
       product defect.
    2. Under the long budget the SAME wait succeeds, and it returns at ~12 s: when the
       condition became true, not when the 30 s budget ran out.
    3. The two elapsed figures are separated by the flag's delay minus the short budget
       the first wait already spent — i.e. the difference is accounted for by the
       condition, not by either bound.

    Plus the control that closes the loop: the short-budget wait run again AFTER the flag
    is set succeeds immediately, so the earlier failure was the predicate's falsehood and
    never the 0.5 s number itself.

    12 s of wall clock is the price; it buys a cell that cannot pass unless the budget
    both bites and yields.
    """
    flag = threading.Event()

    def flag_is_set() -> bool:
        """True only once the driver thread below has fired."""
        return flag.is_set()

    def set_flag_later() -> None:
        """The condition under test: false for ``FLAG_DELAY_S``, then true for good."""
        time.sleep(FLAG_DELAY_S)
        flag.set()

    driver = threading.Thread(target=set_flag_later, name="delayed-flag", daemon=True)
    driver.start()
    try:
        # 1. The short budget must expire on a predicate that is still false.
        short = net_fixtures.wait_for(flag_is_set, timeout_s=SHORT_BUDGET_S)
        assert not short, f"a wait on a still-false predicate reported success: {short!r}"
        message = repr(short)
        assert "EXPIRED" in message and "still false" in message, message
        # The elapsed time is measured, not the budget restated: the two agree only
        # because the wait really did run out, so assert both members of the report.
        assert f"after {short.elapsed_s:.2f}s" in message, message
        assert f"(budget {SHORT_BUDGET_S:.1f}s)" in message, message
        assert "flag_is_set" in message, f"the report does not name the predicate: {message}"
        assert short.elapsed_s >= SHORT_BUDGET_S, message

        # 2. The same predicate and the same wait, one budget change later: it must now
        #    succeed, and it must succeed when the flag fires rather than at the bound.
        long = net_fixtures.wait_for(flag_is_set, timeout_s=LONG_BUDGET_S)
        assert long, f"the same wait under a {LONG_BUDGET_S:.0f}s budget expired: {long!r}"
        assert short.ok is False and long.ok is True
        # Not at the budget: the flag is at 12 s and the budget at 30 s, so a wait that
        # consumed the bound would land far outside this window.
        assert FLAG_DELAY_S - 1.5 <= long.elapsed_s < LONG_BUDGET_S - 5.0, (
            f"the long wait returned at {long.elapsed_s:.2f}s, not when the flag fired "
            f"({FLAG_DELAY_S:.0f}s) nor at its {LONG_BUDGET_S:.0f}s budget"
        )

        # 3. The two figures differ by the flag delay minus the time the first wait had
        #    already spent — the gap the condition (not a bound) accounts for.
        expected_gap = FLAG_DELAY_S - SHORT_BUDGET_S
        measured_gap = long.elapsed_s - short.elapsed_s
        assert abs(measured_gap - expected_gap) <= 3.0, (
            f"short {short.elapsed_s:.2f}s, long {long.elapsed_s:.2f}s: gap "
            f"{measured_gap:.2f}s is not the ~{expected_gap:.2f}s between the flag delay "
            f"and the short budget, so neither bound explains the difference"
        )

        # The control: with the predicate now true, the SAME short budget succeeds at
        # once. The 0.5 s number was never what failed the first wait.
        after = net_fixtures.wait_for(flag_is_set, timeout_s=SHORT_BUDGET_S)
        assert after, f"a satisfied predicate still expired under the short budget: {after!r}"
        assert after.elapsed_s < SHORT_BUDGET_S, after
    finally:
        driver.join(timeout=1.0)
