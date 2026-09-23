"""The goal judge's policy, with no app, no provider and no event loop scaffolding.

Every collaborator the driver needs is a callable, so this file can drive the
whole lifecycle — judge, continuation, breaker, cap, staleness, re-arm — through
one recording harness, and each assertion below names the rule it pins rather than
the mechanism it happens to use.

The three rules worth reading the tests for:

* **Fail-safe CONTINUE on an unreadable verdict** is ``_parse_loop_verdict``'s own
  documented contract for its ``None`` return ("the caller treats it as a
  fail-safe CONTINUE plus a judge-failure strike, never as a release"), so a
  broken judge keeps the work moving and the STRIKE counter is what stops it.
* **The staleness guard drops a verdict whose context moved**, and the driver
  re-judges instead of leaving the goal inert — a foreign turn's own end was
  swallowed by ``in_flight``, so nothing else would re-arm it.
* **The cap and the breaker are distinguishable by their reason**, because a
  surface that reported one for the other would send the user chasing a provider
  problem that does not exist.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.session.goal import GoalJudgeState
from local_operator.session.goal_judge import (
    GOAL_JUDGE_FAILURES,
    MAX_GOAL_CONTINUATIONS,
    MAX_GOAL_REJUDGES,
    STALLED_BREAKER_REASON,
    STALLED_CAP_REASON,
    GoalJudge,
    goal_continuation_prompt,
)

ACHIEVED = "VERDICT: ACHIEVED\nAll the work is done"
CONTINUE = "VERDICT: CONTINUE\nThere is more to do"
UNREADABLE = "I think it is probably done, maybe?"


class Harness:
    """A recording double for every collaborator the driver talks to."""

    def __init__(
        self,
        *,
        goal: str = "Ship the release",
        status: str = "active",
        token: str = "goal-token",
        judge_state: GoalJudgeState | None = None,
        serial: int = 0,
        answers: list[str] | None = None,
        loop_running: bool = False,
    ) -> None:
        self.goal_text = goal
        self.status_value = status
        self.token_value = token
        self.state = judge_state or GoalJudgeState(state="waiting")
        self.serial_value = serial
        self.answers = list(answers or [])
        self.loop = loop_running
        self.judge_calls: list[str] = []
        self.prompts: list[str] = []
        self.settled: list[str] = []
        self.changed: list[dict[str, object]] = []
        #: Hooks a test uses to make the WORLD move while the judge is in flight.
        self.on_judge: object = None
        self.on_prompt: object = None
        self.prompt_error: BaseException | None = None
        self.judge_started = asyncio.Event()
        self.release_judge = asyncio.Event()

    def build(self) -> GoalJudge:
        return GoalJudge(
            judge=self.judge,
            prompt=self.prompt_turn,
            changed=self.note,
            settled=self.settle,
            goal=lambda: self.goal_text,
            status=lambda: self.status_value,
            token=lambda: self.token_value,
            serial=lambda: self.serial_value,
            judge_state=lambda: self.state,
            loop_running=lambda: self.loop,
        )

    # -- collaborators ---------------------------------------------------------

    async def judge(self, prompt_text: str) -> str:
        self.judge_calls.append(prompt_text)
        self.judge_started.set()
        if self.on_judge is not None:  # type: ignore[operator]
            self.on_judge(self)
        await self.release_judge.wait()
        if not self.answers:
            return CONTINUE
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):  # pragma: no cover - defensive
            raise answer
        return answer

    async def prompt_turn(self, text: str) -> None:
        self.prompts.append(text)
        if self.on_prompt is not None:  # type: ignore[operator]
            self.on_prompt(self)
        if self.prompt_error is not None:
            raise self.prompt_error

    def note(self, fields: dict[str, object]) -> None:
        """Apply a publish the way ``Session.note_goal_judge`` does."""
        self.changed.append(dict(fields))
        for key, value in fields.items():
            setattr(self.state, key, value)

    def settle(self, reason: str) -> None:
        self.settled.append(reason)
        self.status_value = "done"
        self.state.state = "done"

    # -- convenience -----------------------------------------------------------

    def trigger(self) -> object:
        return self.build()


async def _judge_all(h: Harness) -> None:
    """Drive one successful turn end with the judge never blocking."""
    h.release_judge.set()
    await h.build().on_turn_end(error=False, aborted=False, serial=h.serial_value)


def _continue_then_achieved(h: Harness) -> None:
    """The faithful world: each admitted continuation advances the turn serial."""
    h.serial_value += 1


# --- the happy paths ----------------------------------------------------------


@pytest.mark.asyncio
async def test_achieved_marks_done_once_and_admits_nothing():
    h = Harness(answers=[ACHIEVED])
    await _judge_all(h)
    assert h.settled == ["All the work is done"]
    assert h.prompts == []
    assert len(h.judge_calls) == 1
    # The judge's prompt is the loop's own question, asked once, in this codebase's
    # single judge voice.
    assert h.judge_calls[0] == (
        "You are judging whether a standing goal has been fully achieved, based on "
        "the conversation above (the work done so far).\n\nGOAL: Ship the release\n\n"
        "Answer with a single line, exactly one of:\n"
        "  VERDICT: ACHIEVED\n"
        "  VERDICT: CONTINUE\n"
        "Then, on the next line, one short sentence of reason. Judge strictly: "
        "answer ACHIEVED only if the goal is fully and verifiably met, not merely "
        "in progress. If unsure, answer CONTINUE. Answer in text only and do not "
        "call any tool: this is a verdict on the conversation above, and a tool "
        "call here is discarded unread."
    )
    assert h.state.state == "done"
    assert h.state.verdict == "achieved"


@pytest.mark.asyncio
async def test_continue_admits_exactly_one_continuation_then_judges_it():
    h = Harness(answers=[CONTINUE, ACHIEVED])
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert len(h.prompts) == 1
    assert h.prompts == [goal_continuation_prompt("Ship the release")]
    assert h.state.run == 1
    assert len(h.judge_calls) == 2
    assert h.settled == ["All the work is done"]


@pytest.mark.asyncio
async def test_a_readable_continuation_publishes_continuing_with_its_reason():
    h = Harness(answers=[CONTINUE, ACHIEVED])
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert any(
        fields.get("state") == "continuing" and fields.get("run") == 1 for fields in h.changed
    )


# --- the breaker and the cap --------------------------------------------------


@pytest.mark.asyncio
async def test_three_consecutive_unreadable_verdicts_stall_the_goal():
    """Fail-safe CONTINUE, and the strike is what stops it.

    ``_parse_loop_verdict`` documents its own ``None`` return as "a fail-safe
    CONTINUE plus a judge-failure strike, never as a release", so each unreadable
    verdict still sends the agent back to work — and the breaker stops it at
    ``GOAL_JUDGE_FAILURES`` rather than letting a broken judge spend without end.
    """
    h = Harness(answers=[UNREADABLE] * 10)
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert h.state.state == "stalled"
    assert h.state.reason == STALLED_BREAKER_REASON
    assert len(h.prompts) == GOAL_JUDGE_FAILURES - 1
    assert len(h.judge_calls) == GOAL_JUDGE_FAILURES
    # A stall is a stop, not a release: nothing was marked done.
    assert h.settled == []


@pytest.mark.asyncio
async def test_a_readable_verdict_clears_the_strike_count():
    """The breaker is for consecutive MALFUNCTION (goal_loop.py:128's rule)."""
    h = Harness(answers=[UNREADABLE, CONTINUE, UNREADABLE, UNREADABLE, ACHIEVED])
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert h.state.state == "done"
    assert h.settled == ["All the work is done"]
    # Two strikes never reach the breaker, and the readable CONTINUE reset them.
    assert h.state.failures == 0


@pytest.mark.asyncio
async def test_the_continuation_cap_stalls_with_the_caps_own_reason():
    h = Harness(answers=[CONTINUE] * (MAX_GOAL_CONTINUATIONS + 4))
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert h.state.state == "stalled"
    assert h.state.run == MAX_GOAL_CONTINUATIONS
    assert len(h.prompts) == MAX_GOAL_CONTINUATIONS
    assert h.state.reason == STALLED_CAP_REASON
    # The cap's reason, NOT the breaker's: a surface that mixed them up would send
    # the user looking for a provider problem that does not exist.
    assert STALLED_BREAKER_REASON not in h.state.reason


@pytest.mark.asyncio
async def test_the_cap_lets_the_last_continuation_be_judged_first():
    """A goal that finishes on its final continuation is DONE, not stalled."""
    h = Harness(answers=[CONTINUE] * (MAX_GOAL_CONTINUATIONS - 1) + [ACHIEVED])
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert h.state.state == "done"
    assert h.settled == ["All the work is done"]


@pytest.mark.asyncio
async def test_the_streak_resets_when_a_foreign_turn_ends():
    """RULINGS R4: the cap is per-streak, not a lifetime budget."""
    h = Harness(
        answers=[CONTINUE, ACHIEVED],
        judge_state=GoalJudgeState(state="waiting", run=MAX_GOAL_CONTINUATIONS - 1),
    )
    h.on_prompt = lambda _h: _continue_then_achieved(_h)
    await _judge_all(h)
    assert h.state.state == "done"
    assert {"run": 0} in h.changed
    # The reset is what let this continuation happen at all.
    assert len(h.prompts) == 1


# --- the guards ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_running_loop_owns_the_verdict():
    """The one double-judge class the design must not have."""
    h = Harness(answers=[ACHIEVED], loop_running=True)
    await _judge_all(h)
    assert h.judge_calls == []
    assert h.changed == []
    assert h.settled == []


@pytest.mark.asyncio
async def test_an_error_or_aborted_turn_waits_without_judging():
    for kwargs in ({"error": True, "aborted": False}, {"error": False, "aborted": True}):
        h = Harness(answers=[ACHIEVED])
        h.release_judge.set()
        await h.build().on_turn_end(serial=0, **kwargs)
        assert h.judge_calls == []
        assert h.settled == []
        assert h.state.state == "waiting"


@pytest.mark.asyncio
async def test_a_goal_that_is_not_active_is_left_alone():
    for status in ("done", "cleared"):
        h = Harness(answers=[ACHIEVED], status=status)
        await _judge_all(h)
        assert h.judge_calls == []


@pytest.mark.asyncio
async def test_a_goal_with_no_token_is_not_judged():
    """No token means no staleness guard, and a guard that cannot fire is worse."""
    h = Harness(answers=[ACHIEVED], token="")
    await _judge_all(h)
    assert h.judge_calls == []


# --- the staleness guard ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verdict_whose_context_moved_is_dropped_and_the_judge_retries():
    """A foreign turn ended mid-judge: neither outcome may be applied.

    The verdict was formed against a context the session has left, so marking the
    goal done or pushing a continuation under a user who is mid-conversation would
    both be dishonest. That foreign turn's OWN end was swallowed by ``in_flight``,
    so the driver is the only thing that can re-arm the goal — which is what the
    re-judge is for, bounded by ``MAX_GOAL_REJUDGES``.
    """
    h = Harness(answers=[ACHIEVED] * 10)
    h.on_judge = lambda _h: setattr(_h, "serial_value", _h.serial_value + 1)
    await _judge_all(h)
    assert len(h.judge_calls) == MAX_GOAL_REJUDGES + 1
    assert h.prompts == []
    assert h.settled == []
    assert h.state.state == "waiting"


@pytest.mark.asyncio
async def test_a_verdict_is_applied_when_the_context_moved_only_once():
    """The retry is a repair, not a second opinion: one movement, one re-judge."""
    h = Harness(answers=[UNREADABLE, ACHIEVED])
    calls = {"n": 0}

    def move_once(_h: Harness) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            _h.serial_value += 1

    h.on_judge = move_once
    await _judge_all(h)
    assert len(h.judge_calls) == 2
    assert h.settled == ["All the work is done"]


@pytest.mark.asyncio
async def test_a_replaced_goal_drops_its_verdict_without_retrying():
    """A moved TOKEN is not a moved context: that goal is simply not this one."""
    h = Harness(answers=[ACHIEVED] * 5)
    h.on_judge = lambda _h: setattr(_h, "token_value", "another-goal")
    await _judge_all(h)
    assert len(h.judge_calls) == 1
    assert h.settled == []
    assert h.prompts == []
    # The record now belongs to a DIFFERENT goal, so this driver settles nothing
    # on it: the only thing it may have published is the `judging` state of a call
    # that was already in flight when the goal changed.
    assert h.state.state != "done"


# --- admission refusals -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_admission_waits_rather_than_retrying():
    """A refusal is a `waiting`, never a retry; the next turn end re-arms."""
    from local_operator.session.errors import TurnInFlight

    h = Harness(answers=[CONTINUE, CONTINUE, CONTINUE])
    h.prompt_error = TurnInFlight("a turn is already running")
    await _judge_all(h)
    assert len(h.prompts) == 1
    assert h.state.state == "waiting"
    assert h.settled == []


# --- in_flight and re-arm -----------------------------------------------------


@pytest.mark.asyncio
async def test_in_flight_suppresses_a_second_trigger():
    h = Harness(answers=[ACHIEVED])
    driver = h.build()
    first = asyncio.ensure_future(driver.on_turn_end(error=False, aborted=False, serial=0))
    await h.judge_started.wait()
    assert driver.in_flight is True
    # A second edge — the continuation's own turn end — must not start a judge.
    await driver.on_turn_end(error=False, aborted=False, serial=1)
    assert len(h.judge_calls) == 1
    h.release_judge.set()
    await first
    assert driver.in_flight is False
    assert h.settled == ["All the work is done"]


@pytest.mark.asyncio
async def test_start_turn_end_claims_in_flight_before_the_task_exists():
    """The event-path claim is synchronous, so one loop tick cannot race it."""
    h = Harness(answers=[ACHIEVED])
    driver = h.build()
    driver.start_turn_end(error=False, aborted=False, serial=0)
    assert driver.in_flight is True
    # The second call in the SAME tick is refused, before either task has run.
    driver.start_turn_end(error=False, aborted=False, serial=0)
    h.release_judge.set()
    await asyncio.sleep(0)
    for _ in range(5):
        if not driver.in_flight:
            break
        await asyncio.sleep(0)
    assert len(h.judge_calls) == 1


@pytest.mark.asyncio
async def test_rearm_on_resume_judges_only_an_actually_in_flight_state():
    for state in ("continuing", "judging"):
        h = Harness(answers=[ACHIEVED], judge_state=GoalJudgeState(state=state, run=4))
        h.release_judge.set()
        await h.build().rearm_on_resume()
        assert len(h.judge_calls) == 1, state
    for state in ("waiting", "stalled", "done", "idle"):
        h = Harness(answers=[ACHIEVED], judge_state=GoalJudgeState(state=state))
        h.release_judge.set()
        await h.build().rearm_on_resume()
        assert h.judge_calls == [], state


@pytest.mark.asyncio
async def test_rearm_on_resume_continues_the_streak_rather_than_resetting_it():
    """A restart is not a user-authored turn: the cap must survive it."""
    h = Harness(
        answers=[ACHIEVED],
        judge_state=GoalJudgeState(state="continuing", run=MAX_GOAL_CONTINUATIONS),
    )
    h.release_judge.set()
    await h.build().rearm_on_resume()
    # The persisted streak is respected: at the cap, so this continuation is the
    # last thing the judge may admit, and an achieved verdict still settles.
    assert {"run": 0} not in h.changed
    assert h.state.state == "done"
