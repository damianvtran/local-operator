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

import asyncio

import pytest

from local_operator.session.errors import TurnInFlight
from local_operator.session.goal_judge import (
    MAX_GOAL_CONTINUATIONS,
    STALLED_BREAKER_NOTICE,
    STALLED_BREAKER_REASON,
    STALLED_CAP_NOTICE,
    STALLED_CAP_REASON,
    goal_continuation_prompt,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import TurnEnded, UserMessageStart
from local_operator.tui.widgets.transcript import NoticeBlock

from .test_app_pilot import GoalSession, _factory

GOAL = "land the OAuth refresh fix"
CONTINUE = "VERDICT: CONTINUE\nmore to do"
ACHIEVED = "VERDICT: ACHIEVED\nthe work is done"
#: An answer with no readable verdict: the strike the breaker counts.
UNREADABLE = "I think it is probably done, maybe?"


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
async def test_a_persistently_refused_continuation_parks_at_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The BOUND on the deferred admission, and the no-spin rule it keeps.

    The refusal is waited out only while it can clear (see QA-Q1's test above):
    a session that refuses for longer than the ceiling parks the record at
    `waiting`, which is honest — nothing is being spent — and the next turn end
    re-arms the judge. What must never happen is a SECOND judge run, and that is
    what this asserts: the driver's policy answers a refused admission with
    `waiting`, never with another verdict.

    The two windows are shortened rather than slept through, so the test pins the
    bound instead of depending on how long the suite is willing to wait.
    """
    from local_operator.tui import app as app_mod

    monkeypatch.setattr(app_mod, "_GOAL_CONTINUATION_ADMISSION_S", 0.05)
    monkeypatch.setattr(app_mod, "_GOAL_CONTINUATION_ADMISSION_RETRY_S", 0.005)

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
        await asyncio.sleep(0.2)
        await _settle(pilot, 20)
        # It ASKED again while the refusal could still clear, and gave up at the
        # ceiling rather than spinning: more than one attempt, bounded.
        assert len(attempts) > 1
        assert session.judge_calls == 1
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "waiting"
        # The goal stays ACTIVE: a refusal is not a release either.
        assert session.goal_status == "active"


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


def _stall_notices(app) -> list[str]:
    """The stall receipts actually PAINTED, off the widgets in the transcript.

    Read from the mounted blocks rather than from a recorded call, so a notice
    that was built and never shown cannot pass this.
    """
    return [
        block.text() or ""
        for block in app.query(NoticeBlock)
        if "goal stalled" in (block.text() or "")
    ]


@pytest.mark.asyncio
async def test_a_stall_is_announced_once_and_names_the_breaker() -> None:
    """Design round 1, D2 on THIS host: the TUI announced the state nowhere.

    Three unreadable verdicts are enough to fire the breaker inside ONE run —
    each strike admits a fail-safe continuation, which the fake serial advances
    so the chain keeps going (`test_three_consecutive_unreadable_verdicts_stall_
    the_goal` pins that policy) — and the state must then reach the transcript,
    naming the breaker rather than the cap.
    """
    session = _armed([UNREADABLE] * 4)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "stalled"
        assert session.goal_judge["reason"] == STALLED_BREAKER_REASON
        assert _stall_notices(app) == [STALLED_BREAKER_NOTICE]


@pytest.mark.asyncio
async def test_a_cap_stall_is_announced_once_and_names_the_cap() -> None:
    """The other bound, in the same one-word state: the receipt must say which.

    A goal that stopped at the continuation cap and one the judge gave up on send
    the user to two different places, and the reason on the record is what tells
    them apart.
    """
    session = _armed([CONTINUE] * (MAX_GOAL_CONTINUATIONS + 1))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot, 24)
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "stalled"
        assert session.goal_judge["reason"] == STALLED_CAP_REASON
        assert len(session.prompts) == MAX_GOAL_CONTINUATIONS
        assert _stall_notices(app) == [STALLED_CAP_NOTICE]


@pytest.mark.asyncio
async def test_the_stall_is_not_announced_again_by_a_later_frame() -> None:
    """ONCE PER ENTRY: a later frame beside the SAME state announces nothing.

    The frame used here is the streak reset — a turn that ended in error resets
    it, and that publish (`{"run": 0}`) lands while the state still reads
    `stalled`, which is exactly the "later frame that still reads stalled" the
    receipt must not repeat on. The state then moves to `waiting` and stays
    silent too, so what this pins is that only the ENTRY into `stalled` speaks.
    The state after is asserted as well: the silence above is the guard working,
    not the second frame failing to run.
    """
    session = _armed([UNREADABLE] * 4)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot)
        assert _stall_notices(app) == [STALLED_BREAKER_NOTICE]
        app.post_message(TurnEnded(aborted=False, error="provider exploded"))
        await _settle(pilot)
        assert session.goal_judge is not None
        assert session.goal_judge["state"] == "waiting"
        assert _stall_notices(app) == [STALLED_BREAKER_NOTICE]


@pytest.mark.asyncio
async def test_a_continuation_refused_by_a_closing_turn_is_admitted_after_it_closes() -> None:
    """QA-Q1: the edge the judge fires on is the one moment the lock is held.

    `TurnEnded` is posted from the session's HELD end event, which the turn's own
    pipeline flushes with `_turn_lock` still held — so the FIRST admission of
    every continuation meets `TurnInFlight`, and a host that gave up there left
    the goal inert with the verdict already in hand (the measured app behaviour:
    `calls=3`, `status=active`, `judge=waiting`, and `tests/e2e` `assert 3 == 5`).

    The fake raises it the way the real session does rather than accepting
    everything, which is the reason no unit test saw this before: a `prompt` that
    never refuses cannot exercise an admission path that must wait.
    """
    session = _armed([CONTINUE, ACHIEVED])
    session.turn_in_flight_prompts = 1
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        # The retry gap is real time (a thread-backed teardown resolves on the
        # clock, not on a loop-turn budget), so the settle below has to let it
        # elapse rather than only pumping messages.
        await asyncio.sleep(0.2)
        await _settle(pilot, 20)
        assert session.prompts == [goal_continuation_prompt(GOAL)]
        # ...and the chain continued: the admitted turn was judged and settled.
        assert session.judge_calls == 2
        assert session.goal_status == "done"


@pytest.mark.asyncio
async def test_a_continuation_carries_the_structural_harness_stamp() -> None:
    """Agent review MAJOR-2: the one host that did not stamp its own row.

    The TUI's live paint and its replay are covered by the text recogniser, so
    the omission was invisible here — but the DESKTOP is marker-only by contract
    (`docs/DESKTOP_API.md`), and this row is durable in the shared store, so a
    continuation written by a local TUI replayed there as the user's own words.
    """
    session = _armed([CONTINUE, ACHIEVED])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot, 2)
        app.post_message(TurnEnded(aborted=False, error=None))
        await _settle(pilot, 20)
        assert session.prompts == [goal_continuation_prompt(GOAL)]
        assert session.injected_prompts == [goal_continuation_prompt(GOAL)]


@pytest.mark.asyncio
async def test_a_restored_continuation_re_arms_at_boot_without_a_typed_turn() -> None:
    """Agent review MAJOR-1: trigger 3 had no caller on this host.

    Driven with NO `TurnEnded` at all: the restored record alone must re-engage
    the judge, which is the app-reopened case the operator asked for. Reproduced
    before the fix as `judge_calls after boot = 0`, no prompt admitted.
    """
    session = _armed([CONTINUE, ACHIEVED])
    # The exact restored shape: a continuation was in flight when the terminal
    # closed, so the record reads `continuing` with a spent streak.
    session.note_goal_judge(state="continuing", verdict="continue", run=2)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await asyncio.sleep(0.2)
        await _settle(pilot, 20)
        assert session.judge_calls >= 1, "the restored record alone re-armed the judge"
        assert session.prompts == [goal_continuation_prompt(GOAL)]
        assert session.goal_status == "done"


@pytest.mark.asyncio
async def test_a_restored_waiting_goal_is_not_re_armed_at_boot() -> None:
    """RULINGS R3's other half: checkpoint state is not an instruction to spend.

    A `waiting` goal had nothing in flight when the app closed, so a restart is
    not a reason to judge it — it re-arms on the next real turn end like any
    other. Paired with the test above so the guard cannot be satisfied by
    re-arming unconditionally.
    """
    session = _armed([ACHIEVED])
    session.note_goal_judge(state="waiting")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await asyncio.sleep(0.2)
        await _settle(pilot, 20)
        assert session.judge_calls == 0
        assert session.prompts == []
