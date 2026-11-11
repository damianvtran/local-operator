"""Settle the app on the PRODUCT's own condition, not on a fixed pause.

WHY THIS EXISTS. ``pilot.pause()`` pumps ONE batch of event-loop turns; it is
not a settle. A test that awaits it once and then reads widget state is betting
that the app finished everything it started during those turns, and that bet
loses under CPU contention — the app is mid-turn when the assert runs, the
assert reads a half-applied state, and the failure names the feature rather
than the race. That shape is behind the whole ``test_slash_echo`` /
``test_cut_off_notices`` / ``test_goal_panel`` / ``test_schedule_view`` /
``test_steering_approval`` / ``test_review_view`` / ``test_resume_render`` /
``test_first_paint_reveal`` / ``test_transcript_find`` family: 12+ distinct
nodes across 6 PRs and ``main``, all failing on a ``pilot.pause()`` followed by
an assertion.

WHY A TURN COUNT AND NOT A TIMEOUT. The bound here is a COUNT of pumped turns,
never a wall-clock budget. A turn count survives contention: a loaded host
inflates how long each turn takes, not how many turns the app needs to finish
work it was already doing. This is the same reasoning ``conftest.py`` records
for ``MAX_PUMP_TURNS`` and ``test_agent_class_slash._settle``'s ``tries``; this
module is that convention in one place, with the predicate written down instead
of implied by the pump count.

WHAT THE PREDICATE MUST BE. The condition the PRODUCT establishes — a latch the
implementation clears, a row it writes, a flag it sets. It must NOT be the
assertion the test makes, or the wait would make the assert vacuous. Settle the
precondition (or the state the assert is about) and keep asserting it after.

WHY NOT JUST PAUSE LONGER OR MORE TIMES. A longer fixed pump only moves the
race: it fails on the next slower host, and it spends the extra turns on every
green run. A condition wait returns the moment the app is ready and fails with
the predicate named when it never becomes ready.
"""

from __future__ import annotations

from typing import Callable

#: Turns pumped before a settle is declared failed. Sized from the app's own
#: boot and transition work, which ``conftest.py`` measures in event-loop turns:
#: ``_settle`` in ``test_agent_class_slash`` pumps 120 to cover one worker, and
#: ``_settle_loop`` in ``test_app_pilot`` pumps 60 to cover a turn retirement.
#: 240 leaves both the largest existing budget and the transitions that compose
#: two of them room, while still failing fast instead of hanging a shard.
MAX_SETTLE_TURNS = 240


async def settle(
    pilot,
    predicate: Callable[[], bool],
    *,
    what: str,
    turns: int = MAX_SETTLE_TURNS,
) -> None:
    """Pump ``pilot`` until ``predicate()`` holds, then return.

    Fails with ``what`` and the predicate named when the condition has not held
    after ``turns`` pumped turns — a settle that never settles is a product
    regression or a wrong predicate, and either way the message must say which
    condition was being waited on.

    The predicate is polled BEFORE each pump, so an app that is already settled
    costs no turns at all and a test that holds on every run pays nothing.
    """
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
    if predicate():
        return
    raise AssertionError(
        f"{what} did not settle in {turns} event-loop turns; "
        f"waited on: {getattr(predicate, '__name__', predicate)!r}"
    )
