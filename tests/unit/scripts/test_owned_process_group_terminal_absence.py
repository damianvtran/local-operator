"""The witness authorises a SEND; the census reports an END -- and only an END.

After a run kills the group it owns, the leader is legitimately gone and the
creation receipt can no longer be sampled. The terminal report may therefore rest
on the weaker evidence (a complete census), precisely because the stronger one is
absent by our own authorised act. The send path never gains that licence: it calls
`refresh(deadline=...)` without `terminal`, so a refusal there is still a refusal.
"""

from __future__ import annotations

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded_run_record import World


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
