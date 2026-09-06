"""Shared loop prompts and verdict semantics, without importing a UI host."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any


class GoalLoop:
    """One owner-local orchestration task; viewer reconnects cannot restart it.

    Prompt waits for completion, not merely admission. Gate futures are never
    answered here. Cancelling the driver also interrupts its current turn, while
    ordinary viewer detachment leaves both driver and owner alone.
    """

    def __init__(
        self,
        prompt: Callable[[str], Awaitable[None]],
        judge: Callable[[str], Awaitable[str]],
        abort: Callable[[], None],
        changed: Callable[[dict[str, Any]], None],
    ):
        self.prompt = prompt
        self.judge = judge
        self.abort = abort
        self.changed = changed
        self.task: asyncio.Task[None] | None = None
        self.state: dict[str, Any] = {"status": "idle", "completed": 0}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def publish(self, **values: Any) -> None:
        self.state.update(values)
        self.changed(dict(self.state))

    async def cancel(self) -> None:
        if self.running:
            assert self.task is not None
            self.task.cancel()
            self.abort()
            await asyncio.gather(self.task, return_exceptions=True)

    def start(self, args: str, standing_goal: str) -> dict[str, Any]:
        if self.running:
            raise ValueError("A loop is already running")
        goal = ""
        try:
            count: int | None = int(args) if args else DEFAULT_LOOP_ITERATIONS
        except ValueError:
            if _BOTCHED_COUNT_RE.fullmatch(args):
                raise ValueError("Enter a whole iteration count or a goal") from None
            # Clipped HERE, where the user's slash argument enters the loop
            # state, for the same reason `reason` is clipped at its parse: this
            # is the one place the value arrives, so a later writer cannot
            # forget. Without it `goal` rode the attach frame bounded only by
            # the desktop route's `max_length=200_000`, and an oversized frame
            # is a DROPPED LINE -- a session that simply cannot be attached to.
            goal, count = args[:LOOP_GOAL_CHARS], None
        if count is not None and not 1 <= count <= MAX_LOOP_ITERATIONS:
            raise ValueError(f"Iterations must be between 1 and {MAX_LOOP_ITERATIONS}")
        if not goal and not standing_goal:
            raise ValueError("Set a standing goal or enter a goal for this loop")
        self.state = {"status": "running", "completed": 0, "goal": goal, "iterations": count}
        self.publish()
        self.task = asyncio.create_task(self.run(goal, count))
        return dict(self.state)

    async def run(self, goal: str, count: int | None) -> None:
        failures = 0
        try:
            while count is None or self.state["completed"] < count:
                await self.prompt(LOOP_GOAL_PROMPT.format(goal=goal) if goal else LOOP_PROMPT)
                self.publish(completed=self.state["completed"] + 1)
                if goal:
                    self.publish(status="judging")
                    try:
                        verdict, reason = _parse_loop_verdict(
                            await self.judge(LOOP_JUDGE_PROMPT.format(goal=goal))
                        )
                    except Exception:
                        verdict, reason = None, "Judge unavailable"
                    if verdict is True:
                        self.publish(status="achieved", reason=reason)
                        return
                    failures = failures + 1 if verdict is None else 0
                    if failures >= MAX_LOOP_JUDGE_FAILURES:
                        self.publish(status="failed", reason="Judge could not decide")
                        return
                    self.publish(status="running", reason=reason)
            self.publish(status="completed")
        except asyncio.CancelledError:
            self.publish(status="cancelled")
            raise
        except Exception:
            # Provider exception strings may contain private upstream bodies.
            self.publish(status="failed", reason="The loop turn failed")


#: Upper bound on the judge's free-text reason, which is the ONLY unbounded
#: value in the loop state.
#:
#: That state rides the attach frame, and an oversized frame is a DROPPED LINE
#: rather than an error anybody reports -- a session that simply cannot be
#: attached to. Every other key the loop publishes is a scalar with a fixed
#: name, so clipping here is what lets the frame guard classify the whole field
#: as bounded rather than having to cap it at the wire.
#:
#: Clipped at the parse, not at the publish, because this is the one place the
#: model's text enters the state: a later writer cannot forget to do it. One
#: sentence of explanation is the whole purpose of the value, and the bound
#: matches ``JOB_PROMPT_WIRE_CHARS`` so two frontends do not disagree about how
#: much explanatory text is "a preview".
LOOP_REASON_CHARS = 1_000

#: Upper bound on the user's typed loop goal, the state's other free-text value.
#:
#: It rides the same attach frame as `reason` and was previously bounded only by
#: the desktop route's `max_length=200_000` -- two orders of magnitude past what
#: the frame guard asserted against, so the guard's 2,000-char fixture was
#: testing a limit nothing enforced. Clipped at :meth:`GoalLoop.start`, where the
#: value enters the state.
#:
#: 2,000 rather than LOOP_REASON_CHARS: a goal is the user's own instruction and
#: the loop prompts with it every iteration, so it earns more room than a judge's
#: one-sentence explanation. It is also the bound the frame guard already
#: asserts, which is now a real bound rather than a hopeful one.
LOOP_GOAL_CHARS = 2_000

_BOTCHED_COUNT_RE = re.compile(r"\d[A-Za-z0-9.]*$")

_NEGATION_TOKEN_RE = re.compile(r"[A-Z0-9_]+(?:['\u2019][A-Z]+)?")


def _parse_loop_verdict(text: str) -> tuple[bool | None, str]:
    """Parse a goal-mode judge answer into ``(achieved, reason)``.

    ``achieved`` is ``True`` (ACHIEVED), ``False`` (CONTINUE), or ``None`` when
    no ``VERDICT:`` line is readable — which the caller treats as a fail-safe
    CONTINUE plus a judge-failure strike, never as a release. A pure module-level
    function so a unit test can hammer it with garbage without standing up the app.

    Two robustness rules are load-bearing:

    - ``CONTINUE`` is checked BEFORE ``ACHIEVED`` on the verdict payload, so a
      "not achieved" style answer (or any line that names both tokens) reads as
      continue, never as a false release. A false release is the one
      trust-breaking error under the notify-once contract.
    - Negation is matched at the TOKEN level, not as a substring. `NOT` is a
      substring of ordinary words a model naturally writes on the verdict line
      (`NOTHING`, `ANOTHER`, `CANNOT`, `NOTE`), and `N'T` of `CAN'T`/`DON'T`;
      an unbounded substring scan turned `VERDICT: ACHIEVED, nothing else
      remains` into a false CONTINUE, spinning the loop forever (goal mode has
      no iteration ceiling and a readable CONTINUE resets the failure breaker).
      So we split the payload into alphanumeric-plus-apostrophe tokens and only
      treat a WHOLE-word negator (`NOT`, a contraction ending in `N'T`, or the
      compound `NOT_ACHIEVED`) as flipping ACHIEVED to CONTINUE.

    We do NOT infer a verdict from free prose: an answer with no ``VERDICT:``
    line at all is a judge failure (``None``), not a guess.
    """
    reason = ""
    achieved: bool | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if "VERDICT:" in upper:
            payload = upper.split("VERDICT:", 1)[1]
            # CONTINUE first: a stray "not achieved" must never read as achieved.
            if "CONTINUE" in payload:
                achieved = False
            elif "ACHIEVED" in payload:
                # Token-level negation check (see docstring): only a whole-word
                # negator flips a genuine ACHIEVED to CONTINUE, so an ordinary
                # word that merely CONTAINS "not"/"n't" (nothing, cannot, another)
                # no longer causes an unbounded false-continue.
                tokens = _NEGATION_TOKEN_RE.findall(payload)
                # The contraction suffix is checked for BOTH apostrophe forms:
                # the tokeniser accepts a curly apostrophe (U+2019), which most
                # editors and phones autocorrect a straight one into, so
                # `can't`/`can’t` must both count as a negator. Missing the curly
                # form would let `VERDICT: ACHIEVED but can’t verify` read as a
                # false RELEASE — the one trust-breaking outcome R2 closed.
                negated = any(
                    tok in ("NOT", "NOT_ACHIEVED") or tok.endswith(("N'T", "N\u2019T"))
                    for tok in tokens
                )
                achieved = not negated
            continue
        # The first non-empty line after a readable verdict is the reason.
        if achieved is not None and not reason:
            reason = stripped[:LOOP_REASON_CHARS]
    return achieved, reason


DEFAULT_LOOP_ITERATIONS = 3

MAX_LOOP_ITERATIONS = 25

LOOP_PROMPT = (
    "Continue working toward the standing goal. Make concrete progress with "
    "the tools available, then briefly state what advanced and what remains. "
    "If the goal is already fully met, say so plainly and stop."
)

LOOP_GOAL_PROMPT = (
    "Work toward this goal:\n\n{goal}\n\n"
    "Make concrete progress with the tools available, then briefly state what "
    "advanced and what remains. If the goal is already fully met, say so plainly."
)

LOOP_JUDGE_PROMPT = (
    "You are judging whether a standing goal has been fully achieved, based on "
    "the conversation above (the work done so far).\n\n"
    "GOAL: {goal}\n\n"
    "Answer with a single line, exactly one of:\n"
    "  VERDICT: ACHIEVED\n"
    "  VERDICT: CONTINUE\n"
    "Then, on the next line, one short sentence of reason. Judge strictly: "
    "answer ACHIEVED only if the goal is fully and verifiably met, not merely "
    "in progress. If unsure, answer CONTINUE. Answer in text only and do not "
    "call any tool: this is a verdict on the conversation above, and a tool "
    "call here is discarded unread."
)

MAX_LOOP_JUDGE_FAILURES = 3
