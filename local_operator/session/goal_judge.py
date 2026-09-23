"""The judged-goal driver: an EDGE-TRIGGERED reaction to a turn's end.

Why a sibling of :mod:`local_operator.session.goal_loop` rather than an
extension of it. ``GoalLoop`` is an owner-local TASK that runs turns in a loop
and judges each one — it owns the turns, so it can only exist while it runs, and
a restart leaves nothing behind. The goal judge reacts to a turn SOMEBODY ELSE
started (the user's, a peer delivery's, a wake's), which is the only shape that
survives everything the operator asked about: an app reopened, a runtime
restarted, a session resumed. The two share the prompts, the verdict parser and
the failure bound — ``_parse_loop_verdict``'s CONTINUE-before-ACHIEVED rule and
its token-level negation check are load-bearing and must not be forked — and
nothing else. That split is the one ``tui/app.py``'s ``_loop_worker`` vs
``_loop_goal_worker`` already made for the same reason.

The design principle this module implements: **the goal's active-ness is durable
state; the judge is triggered by an edge.** Nothing here holds a long-lived task,
so there is no in-memory driver for a restart to lose — which is what stops the
judge needing the "checkpoint state is not an instruction to spend more tokens"
exemption the retained LOOP state needs.

Bounded by construction, and every bound is a constant here rather than a config
key: this is a spend control, and a number a user can raise is a number that will
be raised before anyone has measured the distribution that would justify it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from local_operator.session.goal import MAX_GOAL_CHARS, GoalJudgeState
from local_operator.session.goal_loop import (
    LOOP_JUDGE_PROMPT,
    MAX_LOOP_JUDGE_FAILURES,
    _parse_loop_verdict,
)

logger = logging.getLogger(__name__)

#: How many auto-continuations ONE streak may admit before the judge stalls.
#:
#: The number that bounds an unattended goal, and the reason it is 12 and not
#: "unbounded until ACHIEVED": every continuation is a full billed turn, the
#: judge that admits them is a model call of its own, and the failure mode this
#: closes is a goal the judge keeps reading as "not yet" while the user is not
#: watching — a loop with no human in it. 12 is roughly a working session's worth
#: of steps toward one objective, so a genuine goal finishes inside it and a
#: mis-specified one costs a bounded sum instead of a night of tokens.
#:
#: A STREAK, not a lifetime budget: it resets to 0 when a turn the harness did
#: not admit ends (RULINGS R4), so a user who comes back and nudges the goal
#: gets a fresh streak rather than inheriting one that is already spent.
#:
#: What would revisit it: the distribution of continuations-to-ACHIEVED across
#: real goal sessions. If that distribution's tail is well inside 12 the number
#: is not doing any work and should fall; if real goals are routinely stalling on
#: it, the objective is being mis-stated rather than under-budgeted, and the
#: answer is a better continuation prompt rather than a larger cap.
MAX_GOAL_CONTINUATIONS = 12

#: Consecutive UNREADABLE verdicts that stop auto-continuation. Bound to the
#: loop's own number rather than re-chosen: the two paths ask the same model the
#: same question, so a judge that is broken here is broken there, and two
#: numbers would be two policies for one instrument.
GOAL_JUDGE_FAILURES = MAX_LOOP_JUDGE_FAILURES

#: How many times ONE trigger may re-judge because a turn it did not admit ended
#: while its verdict was in flight (see :meth:`GoalJudge._run`).
#:
#: Without a bound, a session that is never idle — a user typing continuously, a
#: peer delivering during every judge call — would have the judge pay for verdict
#: after verdict and discard each one, with no continuation ever admitted. Three
#: is the same envelope the failure breaker uses, and for the same reason: a
#: verdict that keeps being invalidated by events is a malfunction of the
#: readable path, and the answer is to stop, report `waiting`, and let the next
#: turn end start a fresh attempt.
MAX_GOAL_REJUDGES = 3

#: The continuation turn's text, split so the recogniser below can match the
#: FIXED halves around an arbitrary goal without a regex over user text.
GOAL_CONTINUATION_HEAD = "Continue working toward this goal:\n\n"

GOAL_CONTINUATION_TAIL = (
    "\n\nMake concrete progress with the tools available, then state plainly "
    "what advanced and what remains. If the goal is fully met, say so and stop."
)

GOAL_CONTINUATION_PROMPT = GOAL_CONTINUATION_HEAD + "{goal}" + GOAL_CONTINUATION_TAIL

#: What an unreadable verdict says in ``reason`` when the judge is still being
#: asked to continue. A surface reading `reason` must never find it empty beside a
#: judge state that is about to spend tokens again.
NO_VERDICT_REASON = "judge returned no verdict"

#: The two sentences a STALL carries, one per cause, and they are constants for
#: the same reason: what a surface needs from a stall is WHICH bound stopped it,
#: and the judge's own words — an unreadable answer, or a CONTINUE — do not say
#: that. A surface that reported one cause for the other would send the user
#: looking for a provider problem that does not exist.
STALLED_BREAKER_REASON = "judge could not decide"
STALLED_CAP_REASON = f"stopped after {MAX_GOAL_CONTINUATIONS} continuations"

#: The sentence a STALL owes the USER, one per bound, and keyed by the REASON the
#: judge published.
#:
#: Keyed by the reason rather than by a second cause code, so the receipt cannot
#: name a bound the record does not: the two transitions that stall already carry
#: their own sentence, and a cause added later falls to
#: :data:`STALLED_UNKNOWN_NOTICE` instead of going unannounced — which matters
#: more here than in any other receipt, because a goal the harness has stopped
#: working on and a goal quietly waiting look identical from the outside.
#:
#: The tail is the desktop's own clause for this state ("stalled — send a message
#: to continue", ``composer-status-row.tsx``/``destination-pickers.tsx``), so the
#: chip a user reads and the line they are handed say the same thing about how to
#: get the goal moving again. It is TRUE of both bounds: the breaker re-judges on
#: the next turn end and continues if the judge can now answer, and the cap's
#: streak is per-streak (RULINGS R4), so a user-authored turn resets it.
STALLED_BREAKER_NOTICE = f"goal stalled: {STALLED_BREAKER_REASON} — send a message to continue"
STALLED_CAP_NOTICE = f"goal stalled: {STALLED_CAP_REASON} — send a message to continue"

#: The reason a stall carries -> the sentence it owes the user.
STALLED_NOTICES: dict[str, str] = {
    STALLED_BREAKER_REASON: STALLED_BREAKER_NOTICE,
    STALLED_CAP_REASON: STALLED_CAP_NOTICE,
}

#: What a stall says when its reason is one this build cannot name. Never
#: nothing: an unannounced stall is the finding this vocabulary exists to close.
STALLED_UNKNOWN_NOTICE = "goal stalled: auto-continuation stopped — send a message to continue"


def goal_stalled_notice(moved: dict[str, Any]) -> str | None:
    """The sentence a STALL entry owes the user, or ``None`` when this is not one.

    THE EDGE IS THE MOVE, and the moved FIELDS carry it rather than a latch kept
    here or on a host. :meth:`GoalJudge._publish` journals and publishes only the
    fields that MOVED, so a ``state`` key in ``moved`` means the state just
    CHANGED — that, and only that, is the entry into ``stalled``. A later publish
    while the state is unchanged carries no ``state`` key at all (the diff drops
    it, and the streak reset at the next turn end is the ordinary example: it
    publishes ``{"run": 0}`` beside a state that stays ``stalled``), so a caller
    that emits on this helper's return value announces the transition once and
    cannot repeat it while the state stays put. A notice per publish is exactly
    the repeat this rule avoids.

    Asking ONE helper from both hosts is the same argument the module docstring
    makes for the shared policy: a rule applied on the runtime and forgotten on
    the TUI is the host disagreement these constants exist to remove.

    A RE-ENTRY announces again, deliberately: if the stall clears, the judge
    re-arms on the next turn end and a bound fires a second time, the harness
    really did stop twice, and silence on the second is the case where the user's
    own nudge appears to have done nothing.
    """
    if moved.get("state") != "stalled":
        return None
    return STALLED_NOTICES.get(str(moved.get("reason") or ""), STALLED_UNKNOWN_NOTICE)


def goal_continuation_prompt(goal: str) -> str:
    """The ONE producer of the continuation text.

    Clipped where the goal enters, which is ``GoalLoop``'s own discipline for the
    value it embeds (§3.1): this string is persisted as a user row, announced to
    every front end and replayed by the attached viewer, so an oversized goal
    would ride the attach frame as a dropped line.
    """
    return GOAL_CONTINUATION_PROMPT.format(goal=(goal or "").strip()[:MAX_GOAL_CHARS])


def is_goal_continuation_instruction(text: str) -> bool:
    """Whether ``text`` is a continuation prompt this module minted, for ANY goal.

    A producer-side recogniser for a FAMILY, exactly as
    :func:`local_operator.harness.loop.is_connectivity_continuation_instruction`
    is for the composed connectivity instruction — and for the same reason: the
    goal is interpolated per goal, so this family has no fixed string to list in
    ``harness_chrome_prompts()``, and a tuple that named only the empty-goal shape
    would be a lie about what it covers.

    Matched on the fixed head and the fixed tail, never as a substring search
    over the goal: a user message that merely OPENS with the head, or that quotes
    the goal text in a sentence of their own, is the user's own words and must
    paint. The empty-goal shape is accepted too, because this recognises a
    FAMILY and not a well-formedness constraint — the drive that calls the
    producer is what refuses a goal with no text, and a recogniser that second-
    guessed it would leave one member of the family paintable. The one thing it
    cannot tell apart is a user who types the whole template verbatim — the same
    inherent limit the text-equality exemption for ``_CONTINUATION_PROMPT``
    documents, and vanishingly unlikely.
    """
    stripped = text.strip()
    return stripped.startswith(GOAL_CONTINUATION_HEAD) and stripped.endswith(GOAL_CONTINUATION_TAIL)


def _never() -> bool:
    return False


def owns_the_session(session: Any) -> bool:
    """Whether the loop that runs this session's turns is on THIS event loop.

    ONE predicate, and it is already in the protocol: ``Session.runtime_locality``
    is ``"this-process"`` always, while ``AttachedSession``'s is
    ``"this-machine"`` — attached or cold. So the host that OWNS the session
    answers ``"this-process"`` and a FOLLOWER does not: every attached TUI, the
    phone, and a cold viewer must never judge, because the judge spends tokens and
    writes the goal record, and neither is a bystander's business. Without this
    the person watching a session on their laptop would mint a second verdict
    against a conversation the runtime is still writing.

    Read the same way ``tui/app.py``'s ``_session_runs_elsewhere`` reads it for the
    same question about config: the session is asked where it is BOUND rather than
    inferred from whether a socket exists.

    ``"unknown"`` (a host that does not implement the property at all) is treated
    as NOT ours: the safe direction is the one that does not spend.
    """
    return getattr(session, "runtime_locality", "unknown") == "this-process"


class GoalJudge:
    """One edge-triggered judge per OWNER of a session. Holds no long-lived task.

    Every collaborator this needs is a callable, so the whole policy is testable
    without an app, a provider or an event loop's worth of scaffolding — and so
    each host passes its own object without this module importing a UI.
    """

    def __init__(
        self,
        *,
        judge: Callable[[str], Awaitable[str]],
        prompt: Callable[[str], Awaitable[None]],
        changed: Callable[[dict[str, Any]], None],
        settled: Callable[[str], None],
        goal: Callable[[], str],
        status: Callable[[], str],
        token: Callable[[], str],
        serial: Callable[[], int],
        judge_state: Callable[[], GoalJudgeState],
        loop_running: Callable[[], bool] = _never,
    ) -> None:
        self.judge = judge
        #: Admits ONE continuation turn and AWAITS it, so this driver never has
        #: to correlate its own turn's end with an event.
        self.prompt = prompt
        #: Publishes the judge fields that MOVED, and journals them (the host's
        #: ``Session.note_goal_judge``). A tick that moved nothing is not
        #: published at all — see :meth:`_publish`.
        self.changed = changed
        #: Marks the active goal done with the judge's reason. Reaching the
        #: record through the host rather than through this module is what lets
        #: the mark-done and delete commands stay the same call the user's own
        #: ``/goal --done`` makes.
        self.settled = settled
        self.goal = goal
        self.status = status
        self.token = token
        self.serial = serial
        self.judge_state = judge_state
        #: Whether a ``/loop`` driver currently owns this session's verdicts.
        #: A CALLBACK, not a captured value: the rule is a check at tick time, so
        #: a loop that ends re-enables the goal judge with no bookkeeping
        #: anywhere.
        self.loop_running = loop_running
        self._in_flight = False
        self._task: asyncio.Task[None] | None = None
        self._mirror: dict[str, Any] = {}

    @property
    def in_flight(self) -> bool:
        """Whether a judge run is active — claimed SYNCHRONOUSLY, see start_turn_end."""
        return self._in_flight

    # -- trigger 1 and 2: a turn this judge did not admit ended -----------------

    async def on_turn_end(self, *, error: bool, aborted: bool, serial: int) -> None:
        """Awaitable form of the trigger, for an in-process caller or a test.

        ``serial`` is the ended turn's own counter (``Session._generation`` at
        its end). It is accepted but not used as the staleness baseline: the
        baseline has to be captured when THIS judge starts its call, which can be
        several turns later.
        """
        if not self._claim():
            return
        try:
            await self._drive(error=error, aborted=aborted, reset_streak=True)
        finally:
            self._release()

    def start_turn_end(self, *, error: bool, aborted: bool, serial: int) -> None:
        """The event-path trigger: schedule the run and return.

        ``in_flight`` is claimed HERE, synchronously, before the task exists.
        That is the whole reason this is not just ``ensure_future(on_turn_end)``
        at the call site: the claim and the scheduling have to be one atomic step
        on the event path, or a second ``AgentEndEvent`` delivered in the same
        loop tick could start a second judge and pay for a second verdict.
        """
        if not self._claim():
            return
        task = asyncio.ensure_future(self._drive(error=error, aborted=aborted, reset_streak=True))
        self._task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._release()
        if task.cancelled():
            return
        # An exception here is a bug in the policy, not a provider failure (those
        # are caught in `_ask`): swallow it into a log rather than let it become
        # an unretrieved task exception, and release the claim either way so the
        # next turn end can judge again rather than the goal going inert.
        error = task.exception()
        if error is not None:  # pragma: no cover - defensive
            logger.warning("goal judge failed", exc_info=error)

    def _claim(self) -> bool:
        if self._in_flight:
            return False
        self._in_flight = True
        return True

    def _release(self) -> None:
        self._in_flight = False
        self._task = None

    # -- trigger 3: session start / resume / adopt -----------------------------

    async def rearm_on_resume(self) -> None:
        """Re-engage ONCE when the restored record says a continuation was in flight.

        RULINGS R3, and it is the whole of the operator's "the judge must
        re-engage after a stop/resume" reconciled with this codebase's own rule
        that *checkpoint state is not an instruction to spend more tokens*: only
        the two judge states that mean "work was already being spent" re-arm
        (``continuing``, ``judging``). A ``waiting`` or ``stalled`` goal is left
        alone — nothing was in flight, so a restart is not a reason to spend —
        and it re-arms on the next turn end like any other.

        The streak is NOT reset: a resume is not a user-authored turn, and
        zeroing it here would hand a resumed goal a fresh 12 continuations every
        time the app is reopened.
        """
        if not self._claim():
            return
        try:
            record = self.judge_state()
            if record.state not in {"continuing", "judging"}:
                return
            await self._drive(error=False, aborted=False, reset_streak=False)
        finally:
            self._release()

    # -- the policy ------------------------------------------------------------

    async def _drive(self, *, error: bool, aborted: bool, reset_streak: bool) -> None:
        """One trigger's decision: judge, or report why it will not."""
        if not self._enabled():
            return
        # MUTUAL EXCLUSION, checked at tick time at both call sites of this
        # module and stated once here: while a `/loop` driver runs, THAT is the
        # judge of this session, and a second judge on the same goal would be the
        # one double-judge class this design must not have.
        if self.loop_running():
            return
        record = self.judge_state()
        self._mirror = {
            "state": record.state,
            "run": int(record.run),
            "verdict": record.verdict,
            "reason": record.reason,
            "failures": int(record.failures),
        }
        if reset_streak and self._mirror["run"]:
            # RULINGS R4: a turn the harness did not admit BREAKS the streak, so
            # the cap is per-streak rather than a lifetime budget. Published only
            # when it moved, because a turn end with nothing to say must not
            # journal.
            self._publish(run=0)
        if error or aborted:
            # Trigger 2. No continuation, no judge call, and honest state: the
            # goal is still active but nothing is being spent on it until the
            # next turn end. `verdict` keeps whatever the last readable answer
            # was — it is the parser's record of what a model SAID.
            self._publish(state="waiting")
            return
        goal = self.goal()
        if not goal:  # pragma: no cover - `_enabled` already refused this
            return
        await self._run(goal, self.token())

    def _enabled(self) -> bool:
        """Whether this trigger may judge at all.

        A caller-side guard duplicated nowhere: the runtime's ``_maybe_judge_goal``
        and the TUI's turn hook both delegate the whole question here, so a rule
        added once cannot be applied on one host and forgotten on the other.

        ``""`` counts as active, deliberately: the frontend fold reads a goal with
        no status as ``active`` (RULINGS R9), because a goal restored from a build
        that predates the record, or set through the mobile relay's plain
        ``set_goal``, is standing work the user expects pursued. The TOKEN is what
        is not negotiable — it is this judge's whole staleness guard — so a goal
        without one is not judged rather than judged with a guard that cannot
        fire.
        """
        return bool(self.goal()) and self.status() in {"", "active"} and bool(self.token())

    async def _run(self, goal: str, token: str) -> None:
        """One judge call plus the decision it earns — and the chain it starts.

        This is ``GoalLoop.run``'s shape without the loop control: judge →
        ACHIEVED ⇒ mark done and stop; CONTINUE ⇒ ONE continuation turn, awaited,
        then judge again. The chain lives HERE rather than in the event path,
        which is what makes the ordering deterministic instead of a race between
        an event handler and a task: a continuation's own turn end is ignored
        (``in_flight``), and its verdict is the next iteration of this loop.
        """
        rearms = 0
        while True:
            if self.token() != token:
                # The goal was replaced or deleted before this iteration: the run
                # is over, and the record now belongs to a DIFFERENT goal, so this
                # driver must not write its state onto the new one.
                return
            # Captured IMMEDIATELY before the call, so the comparison below is
            # against the context this verdict is about.
            captured = self.serial()
            self._publish(state="judging", verdict="")
            verdict, reason = await self._ask(goal)
            if self.token() != token:
                # The goal was replaced or deleted under the call. The verdict
                # belongs to an objective nobody is pursuing: drop it, and leave
                # the record alone — the new goal's own trigger publishes its
                # state.
                return
            if self.serial() != captured:
                # A turn this judge did NOT admit ended while the verdict was in
                # flight: the context the judge read is not the context the
                # session is in, so neither marking the goal done nor pushing a
                # continuation under the user would be honest. Its own turn end
                # was suppressed by `in_flight`, so the re-judge has to happen
                # here or the goal would sit inert; `MAX_GOAL_REJUDGES` bounds a
                # session that never goes quiet.
                rearms += 1
                if rearms > MAX_GOAL_REJUDGES:
                    self._publish(state="waiting")
                    return
                continue
            if verdict is True:
                # SETTLE FIRST, then publish, and the order is deliberate rather
                # than incidental: `settled` is what moves `goal_status` to
                # `done`, so the judge state must not reach a surface as `done`
                # while the goal itself still reads as active — `done` in this
                # vocabulary MEANS "verdict ACHIEVED and the goal has been marked
                # done". Each push is a complete, honest frame: the first says
                # "settled, with no model verdict recorded yet", the second adds
                # the verdict the model gave.
                self.settled(reason)
                # `failures=0` rides along because a readable verdict is what
                # clears the strike count, and ACHIEVED is the most readable
                # verdict there is: leaving two strikes on a settled goal would
                # hand the next arming a breaker that is already two thirds spent
                # (the record is the authority `_drive` reads its counters from).
                self._publish(state="done", verdict="achieved", reason=reason, failures=0)
                return
            if verdict is False:
                # A readable verdict clears the strike counter: the breaker is
                # for consecutive MALFUNCTION, not for a judge healthily telling
                # us to keep going (goal_loop.py:128's rule, reused).
                self._publish(failures=0)
            else:
                failures = int(self._mirror["failures"]) + 1
                # Fail-safe CONTINUE, which is the contract `_parse_loop_verdict`
                # states for its own `None` return: an answer with no readable
                # verdict is never a release and never a stop on its own, so the
                # work continues — and the STRIKE is what stops a broken judge
                # from spending indefinitely.
                if failures >= GOAL_JUDGE_FAILURES:
                    # The BREAKER's own sentence, and NOT the parser's words:
                    # this verdict was unreadable, so quoting it as the reason a
                    # surface shows would present a misreading as the judge's
                    # opinion. The state and the reason together are what let a
                    # surface tell this stall from the cap's.
                    self._publish(
                        state="stalled",
                        verdict="unknown",
                        failures=failures,
                        reason=STALLED_BREAKER_REASON,
                    )
                    return
                self._publish(verdict="unknown", failures=failures, reason=NO_VERDICT_REASON)
            if int(self._mirror["run"]) >= MAX_GOAL_CONTINUATIONS:
                # The cap, NOT the breaker, and the reason says which: a surface
                # that reported one for the other would send the user looking at
                # a provider problem that does not exist. Reached only after the
                # last admitted continuation has been JUDGED, so a goal that
                # finished on its final continuation is marked done rather than
                # stalled.
                self._publish(state="stalled", reason=STALLED_CAP_REASON)
                return
            run = int(self._mirror["run"]) + 1
            if verdict is False:
                self._publish(state="continuing", verdict="continue", run=run, reason=reason)
            else:
                self._publish(state="continuing", verdict="unknown", run=run)
            try:
                await self.prompt(goal_continuation_prompt(goal))
            except Exception:  # noqa: BLE001 -- any refusal is a `waiting`, never a retry
                # A REJECTED ADMISSION IS A `WAITING`, NEVER A RETRY. The
                # interesting rejection is `TurnInFlight` — the session is
                # genuinely still closing a turn — and retrying it here would
                # spin; the next turn end re-arms the judge instead.
                self._publish(state="waiting")
                return

    async def _ask(self, goal: str) -> tuple[bool | None, str]:
        """One judge call, parsed. Any provider failure is an unreadable verdict."""
        try:
            answer = await self.judge(LOOP_JUDGE_PROMPT.format(goal=goal))
        except Exception:  # noqa: BLE001 — a judge that cannot answer is a strike
            return None, ""
        return _parse_loop_verdict(answer)

    def _publish(self, **fields: Any) -> None:
        """Journal and publish ONLY the fields that moved.

        The diff matters: this is driven by every turn end and by every judge
        tick, and a write per tick for a value that did not move is pure I/O on
        the hot path (the rule ``Session._persist_goal_record`` documents, which
        binds harder here than it does for the attachment).
        """
        moved = {key: value for key, value in fields.items() if self._mirror.get(key) != value}
        if not moved:
            return
        if moved.get("state") == "stalled" and "reason" in fields:
            # A STALL'S REASON ALWAYS RIDES ITS TRANSITION, even when it repeats.
            # Everything else here is a write-thrift rule, but the reason is the
            # ONE fact the stall's receipt exists to carry — WHICH bound stopped
            # auto-continuation — and the same bound firing twice publishes the
            # same string, which the diff above would drop. The receipt would
            # then fall through to the sentence for a cause it cannot name, i.e.
            # announce the stop and forget the thing it was announcing about.
            #
            # Carried HERE rather than patched in each reader: a host that had to
            # reach past the published diff for the reason would be a second
            # opinion about what the judge published, and there are two hosts.
            moved["reason"] = fields["reason"]
        self._mirror.update(moved)
        self.changed(moved)
