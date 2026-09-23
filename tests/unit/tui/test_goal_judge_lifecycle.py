"""The TUI host's half of the judged-goal lifecycle.

Two rules are why this file exists rather than leaning on the driver's own tests.

**A follower terminal never judges.** ``runtime_locality`` answers whether the
loop that runs this session's turns is on THIS event loop, and a terminal
attached to somebody else's runtime answers ``"this-machine"``. Judging from
there would mint a second verdict against a conversation the owner is still
writing — and spend tokens — so the hook declines before the driver is built.

**The judge is wired to the turn's END, not to a prompt.** Every assertion below
drives a real ``TurnEnded`` through the app's own message pump, because that is
the edge the trigger is hung on: a test that called the worker directly would
pass with the hook deleted.

The judge's POLICY (cap, breaker, staleness, re-arm) is not re-tested here — it
is one object shared with the runtime, and ``tests/unit/session/test_goal_judge``
pins it. What is local to this host is the wiring, the reuse of the app's single
`complete_aside` call, and the fact that harness chrome is never painted.
"""

from __future__ import annotations

import pytest

from local_operator.session.errors import TurnInFlight
from local_operator.session.goal_judge import goal_continuation_prompt
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded, UserMessageStart

from .test_app_pilot import GoalSession, _factory

GOAL = "land the OAuth refresh fix"
CONTINUE = "VERDICT: CONTINUE\nmore to do"
ACHIEVED = "VERDICT: ACHIEVED\nthe work is done"


def _armed(verdicts: list[str] | None = None) -> GoalSession:
    """A fake holding a REAL armed record, with the judge's answers scripted."""
    session = GoalSession()
    session.arm_goal(GOAL)
    session.judge_verdicts = list(verdicts if verdicts is not None else [ACHIEVED])
    return session


async def _settle(pilot, cycles: int = 12) -> None:
    for _ in range(cycles):
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_local_turn_end_judges_once_and_settles_the_goal() -> None:
    session = _armed([ACHIEVED])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        # Exactly ONE aside for one verdict: the judge call is the app's own
        # `complete_aside`, reused rather than duplicated.
        assert session.judge_calls == 1
        assert session.goal_status == "done"
        assert [row["text"] for row in session.history_view()] == [GOAL]
        assert session.prompts == [], "an ACHIEVED verdict admits no continuation"


@pytest.mark.asyncio
async def test_a_continuation_is_admitted_once_and_then_judged() -> None:
    session = _armed([CONTINUE, ACHIEVED])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        assert session.prompts == [goal_continuation_prompt(GOAL)]
        # The continuation's own turn end is swallowed by `in_flight`, and the
        # driver judges the turn it awaited — so two verdicts, not three.
        assert session.judge_calls == 2
        assert session.goal_status == "done"


@pytest.mark.asyncio
async def test_a_follower_terminal_never_judges() -> None:
    """§3.2: the owner judges; the watcher does not spend."""
    session = _armed()
    session.runtime_locality = "this-machine"  # type: ignore[misc]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        assert session.judge_calls == 0
        assert session.prompts == []
        assert session.goal_status == "active"


@pytest.mark.asyncio
async def test_a_running_loop_owns_the_verdict() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app._loop_running = True
        try:
            app.post_message(TurnEnded(aborted=False, error=None))
            await _settle(pilot)
            assert session.judge_calls == 0
        finally:
            app._loop_running = False


@pytest.mark.asyncio
async def test_an_error_turn_waits_without_judging() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error="provider exploded"))
        await _settle(pilot)
        assert session.judge_calls == 0
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "waiting"
        # The goal stays ACTIVE: an error is not a release.
        assert session.goal_status == "active"


@pytest.mark.asyncio
async def test_a_refused_continuation_waits_and_does_not_retry() -> None:
    """`TurnInFlight` is a `waiting`, never a retry: the next turn end re-arms."""

    session = _armed([CONTINUE, CONTINUE, CONTINUE])
    attempts: list[str] = []

    async def refuse(text: str, images=None) -> None:  # noqa: ANN001
        attempts.append(text)
        raise TurnInFlight("a turn is already running")

    session.prompt = refuse  # type: ignore[method-assign]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        # ONE attempt: a retry loop here is the spin the driver must not have.
        assert len(attempts) == 1
        assert session.judge_calls == 1
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "waiting"


@pytest.mark.asyncio
async def test_the_continuation_row_is_never_painted_as_the_users_words() -> None:
    """The live-suppression leg (§4.1), asserted on the surface that paints."""
    from local_operator.tui.widgets.transcript import TranscriptView, UserBlock

    session = _armed([])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        # An owner on a build that still announces chrome, or any older
        # producer: the announcement arrives here and must paint NOTHING.
        app.post_message(UserMessageStart(goal_continuation_prompt(GOAL), 0))
        await _settle(pilot)
        painted = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, UserBlock)
        ]
        assert painted == []
        # ...while the user's own words still paint, so the leg is a filter and
        # not a mute.
        app.post_message(UserMessageStart("my own message", 0))
        await _settle(pilot)
        painted = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, UserBlock)
        ]
        assert [block.text() for block in painted] == ["my own message"]


@pytest.mark.asyncio
async def test_the_stamp_alone_suppresses_a_row_the_recogniser_would_miss() -> None:
    """Either signal: the structural marker is exact where it exists."""
    from local_operator.tui.widgets.transcript import TranscriptView, UserBlock

    session = _armed([])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(UserMessageStart("harness words in the user's shape", 0, injected=True))
        await _settle(pilot)
        painted = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, UserBlock)
        ]
        assert painted == []
