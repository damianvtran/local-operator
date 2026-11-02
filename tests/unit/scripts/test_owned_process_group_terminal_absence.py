"""The witness authorises a SEND; the census reports an END -- and only an END.

After a run kills the group it owns, the leader is legitimately gone and the
creation receipt can no longer be sampled. The terminal report may therefore rest
on the weaker evidence (a complete census), precisely because the stronger one is
absent by our own authorised act. The send path never gains that licence: it calls
`refresh(deadline=...)` without `terminal`, so a refusal there is still a refusal.
"""

from __future__ import annotations

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded_run_record import World


def _foreign_unreadable(world):
    """Replace the leader with a foreign pid in another group, birth unreadable."""
    world.rows[10] = (world.rows[10][0], 999, 999, None)  # type: ignore[assignment]


def _live_owned_group():
    """A captured, VERIFIED, still-live owned group: the shape a kill ends."""
    world = World()
    world.exits_at = 1e9  # the leader stays live until the kill
    cap = world.capture()
    assert cap.last.status == "VERIFIED", cap.last.reason
    return world, cap


def test_a_killed_owned_group_reports_absence_from_the_census():
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"

    # WITHOUT the terminal licence the run reports a refusal: this is what every
    # killed (and timed-out) churn run published as residual-census / rc 126.
    strict = cap.refresh(deadline=world.clock() + 1.0)
    assert strict.status == "REFUSED", strict.reason
    assert strict.refusal_class == "unproven", strict.reason

    end = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert end.status == "OBSERVED_EMPTY", end.reason


def test_a_terminal_census_that_cannot_enumerate_still_refuses():
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"

    def unreadable(_pgid, _deadline):
        raise OSError("census could not be read")

    # The capability binds the census at construction, so patch the INSTANCE --
    # `world.census = ...` would only change the fake, not what the pinned
    # capability consults.
    cap.census = unreadable
    end = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    # AN UNKNOWN POPULATION IS NEVER ABSENCE: the terminal licence buys the weaker
    # evidence, not a waiver of completeness.
    assert end.status == "REFUSED", end.reason
    assert end.refusal_class == "unproven", end.reason


def test_the_send_path_still_refuses_after_the_leader_is_gone():
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"
    # A second send after the leader is gone must NOT inherit the terminal licence.
    again = cap.signal(9, deadline=world.clock() + 1.0)
    assert again.status == "REFUSED", again.reason
    assert world.calls == [(10, 9)], world.calls


def test_foreign_reuse_with_an_unreadable_birth_without_a_kill_refuses():
    """R1/R2: empty censuses alone never buy a terminal verdict."""
    world, cap = _live_owned_group()
    # The leader is replaced by a FOREIGN process on the same pid, in a different
    # group, whose birth cannot be read -- and this run never signals anything.
    _foreign_unreadable(world)
    terminal = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert terminal.status == "REFUSED", terminal.reason
    assert "authorised kill" not in terminal.reason, terminal.reason
    assert world.calls == [], world.calls


def test_foreign_reuse_with_an_unreadable_birth_after_a_kill_does_not_leak_authority():
    """The terminal licence is for REPORTING; it never reaches the send path."""
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"
    _foreign_unreadable(world)
    terminal = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert terminal.status == "OBSERVED_EMPTY", terminal.reason
    # A SECOND SEND ON THE SAME CAPABILITY MUST STILL REFUSE.
    again = cap.signal(9, deadline=world.clock() + 1.0)
    assert again.status == "REFUSED", again.reason
    assert len(world.calls) == 1, world.calls


def test_a_partial_enumeration_refuses_rather_than_reporting_absence():
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"
    # A list carrying a non-positive row is NOT a shorter list: it is a read that
    # did not produce the enumeration it claims.
    cap.census = lambda _pgid, _deadline: [10, 0]
    terminal = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert terminal.status == "REFUSED", terminal.reason


def test_the_r2_reproduction_a_leader_removed_without_a_signal_refuses():
    world, cap = _live_owned_group()
    # NO `signal()` CALL: the leader simply goes away.
    world.rows.pop(10, None)
    strict = cap.refresh(deadline=world.clock() + 1.0)
    assert strict.status == "REFUSED", strict.reason
    terminal = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert terminal.status == "REFUSED", terminal.reason
    assert "requires a recorded kill" in terminal.reason, terminal.reason


def test_a_healthy_run_with_no_kill_still_reports_observed_empty():
    """Edge 3: the case a kill-derived gate at `_members` would have broken.

    A child that exits ON ITS OWN leaves no kill record; its held corpse plus a
    complete no-live-member census is the ordinary clean ending and must still be
    OBSERVED_EMPTY. This is why completeness may not be inferred from `kills`.
    """
    world, cap = _live_owned_group()
    world.exited = True  # the child exits by itself; nothing is signalled
    cap.census = lambda _pgid, _deadline: ()  # complete: the group holds nobody
    verdict = cap.refresh(deadline=world.clock() + 1.0)
    assert verdict.status == "OBSERVED_EMPTY", verdict.reason
    assert cap.kills == [], cap.kills


def test_an_incomplete_census_refuses_at_members_with_a_visibility_reason():
    """Edge 4: not-complete is UNKNOWN VISIBILITY, never a non-empty group."""
    world, cap = _live_owned_group()
    assert cap.signal(9, deadline=world.clock() + 1.0).status == "SIGNALED"
    cap.census = lambda _pgid, _deadline: owned.CensusResult((), False, False)
    terminal = cap.refresh(deadline=world.clock() + 1.0, terminal=True)
    assert terminal.status == "REFUSED", terminal.reason
    assert "census incomplete" in terminal.reason, terminal.reason
