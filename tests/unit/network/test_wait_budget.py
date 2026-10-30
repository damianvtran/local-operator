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

WHY THE BANDS ARE WIDE. Every numeric tolerance below separates two things that differ by
SECONDS — 0.5 s from 12 s, 12 s from 20 s or 30 s — and each one states in place what it
separates and by how much margin. None may be narrower than the 525-668 ms starvation gaps
this fleet records (``conftest.py``): a band that thin reads as load-independent and is
not, because it reds when the final poll was slow, which is the very class this PR exists
to remove. The bands are coarse on purpose; the discrimination is not. A band whose margin
is smaller than a documented starvation gap is wrong by construction.
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
    # bare ``False`` threw away. The elapsed figure is read back from the MEASURED value,
    # not written as the budget literal: ``__repr__`` renders ``:.2f`` of the measurement,
    # and the four 50 ms polls this 0.2 s budget allows cannot finish in under 0.20 s but
    # can easily land past 0.295 s — one 525-668 ms starvation gap (conftest.py, the fleet
    # figure) would red an `"after 0.2"` substring on exactly the loaded host this PR
    # exists to keep green.
    assert f"after {outcome.elapsed_s:.2f}s" in message
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
    0.5 s". The SAME predicate is then waited on three times through the shipped
    ``wait_for``/``wait_outcome`` path (``timeout_s`` is the parameter
    ``LO_TEST_NET_WAIT_S`` feeds — one call site, so the budget cannot be smuggled in
    from elsewhere, and passing it explicitly is what makes the cell independent of
    import-time env capture). Each leg asserts the BEHAVIOUR — where the wait stopped —
    rather than a wide plausible range, because a range wide enough to survive a starved
    host is also wide enough to accept the wrong mechanism.

    1. With the flag still false, the short budget EXPIRES, and the message carries the
       elapsed time and the not-yet-true predicate — the facts a red shard log needs to
       be readable as "slow machine, real work" rather than as a product defect. The
       measured elapsed must be ≈0.5 s: the wait stopped on its bound, not on the flag.
    2. The driver is started only now, and the SAME wait under the long budget succeeds
       with elapsed ≈12 s — the flag's own delay. A wait that returned at its 30 s bound,
       or that slept part of its budget before checking, lands 6-16 s away.
    3. The closing leg: the same 0.5 s budget, now against a satisfied predicate, succeeds
       — so the first failure was the predicate's falsehood and never the number itself.

    Bands are stated where they are used, with their margins; see this module's docstring
    for the rule they follow. 12 s of wall clock is the price, and it buys a cell that
    cannot pass unless the budget both bites and yields.
    """
    flag = threading.Event()

    def flag_is_set() -> bool:
        """True only once the driver thread below has fired."""
        return flag.is_set()

    def set_flag_later() -> None:
        """The condition under test: false for ``FLAG_DELAY_S``, then true for good."""
        time.sleep(FLAG_DELAY_S)
        flag.set()

    driver: threading.Thread | None = None
    try:
        # 1. The short budget must expire on a predicate that is still false, and it must
        #    expire AT its budget. BAND 0.5-2.5 s: it separates the bound from the 12 s
        #    condition by 9.5 s above the ceiling. Ten 50 ms polls are ten preemption
        #    points, so overshoot is real, and the 2 s of headroom above the bound exists
        #    because 525-668 ms starvation gaps are documented for this fleet — a 0.45 s
        #    ceiling was narrower than one of those and red a load-independent cell.
        short = net_fixtures.wait_for(flag_is_set, timeout_s=SHORT_BUDGET_S)
        assert not short, f"a wait on a still-false predicate reported success: {short!r}"
        message = repr(short)
        assert "EXPIRED" in message and "still false" in message, message
        assert f"after {short.elapsed_s:.2f}s" in message, message
        assert f"(budget {SHORT_BUDGET_S:.1f}s)" in message, message
        assert "flag_is_set" in message, f"the report does not name the predicate: {message}"
        assert SHORT_BUDGET_S <= short.elapsed_s <= SHORT_BUDGET_S + 2.0, (
            f"the short wait reported {short.elapsed_s:.2f}s against a "
            f"{SHORT_BUDGET_S:.1f}s budget: it did not stop on its bound"
        )

        # 2. Start the condition now, so the long wait's elapsed time is the flag's delay
        #    measured from its own start. BAND 12 s ± 2 s: 6 s of margin against a mutant
        #    that sleeps part of the budget before checking (the 20 s one), 16 s against
        #    the 30 s bound itself. ±1 s was narrower than one starvation gap.
        driver = threading.Thread(target=set_flag_later, name="delayed-flag", daemon=True)
        driver.start()
        long = net_fixtures.wait_for(flag_is_set, timeout_s=LONG_BUDGET_S)
        assert long, f"the same wait under a {LONG_BUDGET_S:.0f}s budget expired: {long!r}"
        assert short.ok is False and long.ok is True
        assert abs(long.elapsed_s - FLAG_DELAY_S) <= 2.0, (
            f"the long wait returned at {long.elapsed_s:.2f}s against a "
            f"{FLAG_DELAY_S:.0f}s flag delay and a {LONG_BUDGET_S:.0f}s budget: it "
            f"neither woke on the condition nor ran out of budget, so something else "
            f"ended it"
        )

        # No gap assertion between the two elapsed figures. As sized, the two bands above
        # mathematically implied it, so it could never fail; widened until it could, it
        # would only fail when the SHORT leg overshot its own ceiling — a load-dependent
        # red, the class this PR removes — while adding no discrimination that legs 1 and
        # 2 do not already carry. The long band is what rejects a wait that stopped for a
        # third reason.

        # 3. The closing leg: with the predicate now true, the SAME 0.5 s budget succeeds,
        #    so the number was never what failed the first wait. The discrimination here
        #    is the ok flip (False on leg 1, True here) under an identical budget; the
        #    elapsed ceiling only rules out "waited for the 12 s condition", and is set as
        #    wide as leg 1's because one predicate call can also be preempted.
        after = net_fixtures.wait_for(flag_is_set, timeout_s=SHORT_BUDGET_S)
        assert after, f"a satisfied predicate still expired under the short budget: {after!r}"
        assert after.elapsed_s <= SHORT_BUDGET_S + 2.0, (
            f"a satisfied predicate waited {after.elapsed_s:.2f}s: the wait spent time on "
            f"the clock instead of reading the condition"
        )
    finally:
        if driver is not None:
            driver.join(timeout=1.0)
