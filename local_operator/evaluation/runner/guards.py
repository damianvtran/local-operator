"""Benchmark-agnostic episode guards: when to stop an episode that is not going anywhere.

A guard is a pure function over a :class:`GuardInput` snapshot of what the
runner already knows -- steps, cycles, cost, elapsed time, the budget, and the
last few turns -- returning a :class:`GuardVerdict`. No I/O, no adapter or
benchmark knowledge, no provider vocabulary: a guard reads protocol objects
and integers, which is what keeps this module inside the runner core's import
isolation and keeps every guard usable on any benchmark.

**A verdict is a TRUNCATION, never a failure and never a cancel.** The runner
applies it exactly where ``max_steps`` truncation is applied (on the last
executed step, before its event is written), so the episode is scored on the
state it reached and the reason is recorded in
``EnvironmentStepPayload.truncation_reason``. This is deliberate: an episode
stopped for floundering is a *scored partial* an operator can compare
("stopped at 40% after repeating one click 4 times"), whereas a cancel seals
unscored and a fabricated failure lies about what the environment saw. A
budget cap is the same shape -- the budget was reported as overrun after the
fact and never enforced; :class:`BudgetCapGuard` is what makes it real.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import ConfigDict, Field

from local_operator.evaluation.protocol import AskUserAction, ProtocolModel, WaitAction
from local_operator.evaluation.receipts import (
    BudgetAuthorization,
    BudgetResource,
    CappedAllowance,
    SafeCount,
    StrictIdentifier,
)
from local_operator.evaluation.runner.model import EpisodeTurn

__all__ = [
    "RECENT_TURNS_WINDOW",
    "AskLoopGuard",
    "BudgetCapGuard",
    "CostRateGuard",
    "EpisodeGuard",
    "GuardInput",
    "GuardVerdict",
    "NoChangeGuard",
    "RepeatedBatchGuard",
    "default_guards",
]


class GuardInput(ProtocolModel):
    """Everything a guard may read, snapshotted by the runner after a step.

    ``recent_turns`` are the newest turns, oldest first, each with the batch
    the model chose for it; ``recent_costs_micros`` are the per-cycle provider
    costs in the same order (one per model cycle, so they may outnumber the
    turns kept). ``usage_totals`` mirrors what the runner will reconcile.

    ``max_steps`` is the episode's own step budget when the caller states one
    (``EpisodeConfig.max_steps``), and ``None`` when it does not -- which is
    what lets a guard tell a bounded episode, whose remaining steps are known,
    from one that is bounded only by cost. A guard that reads it must treat
    ``None`` as "no step budget", never as zero steps.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False, validate_default=True)

    steps_taken: SafeCount
    max_steps: SafeCount | None = None
    model_cycles: SafeCount
    provider_cost_micros: SafeCount
    elapsed_ms: SafeCount
    usage_totals: Mapping[str, int] = Field(default_factory=dict)
    recent_turns: tuple[EpisodeTurn, ...] = ()
    recent_costs_micros: tuple[int, ...] = ()
    budget: BudgetAuthorization


class GuardVerdict(ProtocolModel):
    """``continue`` or ``truncate`` with a stable code and a human detail.

    ``code`` becomes ``truncation_reason`` in the bundle, so it must be a
    ``StrictIdentifier`` and must be stable across releases -- consumers
    compare runs on it.
    """

    kind: Literal["continue", "truncate"]
    code: StrictIdentifier
    detail: str = Field(min_length=1, max_length=2000)


CONTINUE = GuardVerdict(kind="continue", code="ok", detail="no guard fired")


def _capped_usd_cap(snapshot: GuardInput) -> int | None:
    """The episode's explicit provider-cost cap in micro-USD, if it declared one.

    Only a CAPPED allowance is an explicit cap -- the same rule
    :class:`BudgetCapGuard` enforces -- because an uncapped allowance is the
    *absence* of one (a named person removed it, and ``scripts/run_episode.py``
    leaves the resources it cannot size up front uncapped for the same reason).
    An episode without an explicit cost cap keeps the ratio check as its only
    cost authority.
    """

    for allowance in snapshot.budget.allowances:
        if allowance.resource == "provider_usd_micros" and isinstance(allowance, CappedAllowance):
            return int(allowance.value)
    return None


#: How many of the newest turns the runner snapshots into
#: ``GuardInput.recent_turns``. A guard that needs to see more turns than
#: this can never fire, so the turn-window guards validate their parameters
#: against it at construction rather than silently going inert.
RECENT_TURNS_WINDOW = 16


@runtime_checkable
class EpisodeGuard(Protocol):
    def evaluate(self, snapshot: GuardInput) -> GuardVerdict: ...


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

#: How a ``GuardInput`` field maps onto a budget resource. The four resources
#: the runner measures continuously; the rest (cloud cost, instance time,
#: guest actions, simulator turns) are reconciled at the end and cannot be
#: enforced mid-episode without a signal the runner does not have.
_ENFORCED: tuple[tuple[BudgetResource, str], ...] = (
    ("provider_usd_micros", "provider_cost_micros"),
    ("wall_milliseconds", "elapsed_ms"),
    ("model_cycles", "model_cycles"),
    ("provider_input_tokens", "input_tokens"),
    ("provider_output_tokens", "output_tokens"),
)


class BudgetCapGuard:
    """Truncate once any CAPPED allowance the runner can measure is reached.

    Only capped allowances count: an uncapped one was authorised by a named
    person for a reason, and enforcing a cap they removed would override
    them. Reaching the cap (``>=``) fires, not exceeding it -- the next cycle
    would exceed it and the budget is an authority, not a suggestion.
    """

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        caps = {
            allowance.resource: allowance.value
            for allowance in snapshot.budget.allowances
            if isinstance(allowance, CappedAllowance)
        }
        for resource, field in _ENFORCED:
            cap = caps.get(resource)
            if cap is None:
                continue
            if field in ("input_tokens", "output_tokens"):
                used = int(snapshot.usage_totals.get(field, 0))
            else:
                used = int(getattr(snapshot, field))
            if used >= cap:
                return GuardVerdict(
                    kind="truncate",
                    code="budget-cap",
                    detail=f"{resource} reached its cap ({used} >= {cap})",
                )
        return CONTINUE


# ---------------------------------------------------------------------------
# Cost rate
# ---------------------------------------------------------------------------


class CostRateGuard:
    """Truncate on a cost SPIKE: the last ``window`` cycles cost more than
    ``ratio`` times the ``window`` before them, or any single cycle cost more
    than ``max_cycle_cost_micros``.

    A spike is the signature of the context blowing up (a frame policy that
    stopped pruning, a summary that failed and left the window full): the
    per-cycle price climbs even though the episode is doing the same work. The
    ratio needs two full windows so a cheap first turn cannot trip it; the
    absolute cap is optional because a sane value depends on the model's
    price, which is the episode config's to know.

    The ratio check needs REPORTED cost. A provider that reports no
    ``usd_cost`` folds in as 0 (the evidence payload has no "unknown"
    encoding), and a free tier is genuinely 0; either way a zero previous
    window has no rate to exceed. Rather than silently returning the generic
    continue, the guard says so in its verdict (``code="cost-unreported"``)
    so a reader of the guard's decisions can see the ratio check was skipped,
    and only ``max_cycle_cost_micros`` remains in force.

    **The ratio is inapplicable to a doubly-capped episode.** When the episode
    declares BOTH an explicit step budget and an explicit provider-cost cap,
    those caps are the authority and the ratio is not judged at all -- it is
    replaced by a per-cycle ceiling prorated from what is actually left
    (``remaining cost budget / remaining steps * bounded_cycle_margin``). Two
    things make the ratio the wrong instrument there: it cannot tell a
    legitimate expensive cycle from a runaway one, and it is blind to how much
    of the budget remains. Its dominant false positive in practice is a
    PROMPT-CACHE MISS, which multiplies the input price of one cycle while the
    context is unchanged -- measured on an OSWorld batch, cycles at 2744/1686/
    1908 micro-USD against a ~1000-1400 baseline cut episodes at a median of 11
    of a 500-step budget, so the guard, not the benchmark, was being measured.
    The bounded ceiling is instead anchored to the episode's own budget: it
    fires only on a cycle that costs more than the remaining budget can afford
    per remaining step, and it tightens as the budget is spent.

    A bounded-but-spent episode (no steps left) has no allowance to derive;
    the step budget is already the binding authority, so the guard continues
    without judging the ratio. An episode missing either cap -- including every
    caller that snapshots no step budget at all -- keeps the ratio check
    exactly as it was.
    """

    def __init__(
        self,
        *,
        window: int = 10,
        ratio: float = 3.0,
        max_cycle_cost_micros: int | None = None,
        bounded_cycle_margin: float = 4.0,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if ratio <= 1.0:
            raise ValueError("ratio must exceed 1.0")
        if bounded_cycle_margin <= 0:
            raise ValueError("bounded_cycle_margin must be positive")
        self._window = window
        self._ratio = ratio
        self._max_cycle = max_cycle_cost_micros
        self._bounded_margin = bounded_cycle_margin

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        costs = snapshot.recent_costs_micros
        if self._max_cycle is not None and costs and costs[-1] > self._max_cycle:
            return GuardVerdict(
                kind="truncate",
                code="cost-spike",
                detail=f"one model cycle cost {costs[-1]} micro-USD (cap {self._max_cycle})",
            )
        bounded, ceiling = self._bounded_ceiling(snapshot)
        if bounded:
            if ceiling is not None and costs and costs[-1] > ceiling:
                return GuardVerdict(
                    kind="truncate",
                    code="cost-spike",
                    detail=(
                        f"one model cycle cost {costs[-1]} micro-USD against the {ceiling} "
                        "the episode's remaining cost budget allows per remaining step"
                    ),
                )
            return GuardVerdict(
                kind="continue",
                code="cost-bounded",
                detail=(
                    "the episode's step budget is spent, so no per-cycle allowance is "
                    "derived and the cost-rate ratio is not judged"
                    if ceiling is None
                    else (
                        "the episode declares an explicit step budget and an explicit cost "
                        f"cap, so the caps are the authority and the cost-rate ratio is not "
                        f"judged; {ceiling} micro-USD per remaining step is affordable"
                    )
                ),
            )
        if len(costs) < 2 * self._window:
            return CONTINUE
        recent = sum(costs[-self._window :])
        previous = sum(costs[-2 * self._window : -self._window])
        if previous <= 0:
            return GuardVerdict(
                kind="continue",
                code="cost-unreported",
                detail=(
                    f"the previous {self._window} cycles reported no cost, so the"
                    " cost-rate ratio cannot be judged; only the per-cycle cap applies"
                ),
            )
        if recent > self._ratio * previous:
            return GuardVerdict(
                kind="truncate",
                code="cost-spike",
                detail=(
                    f"the last {self._window} cycles cost {recent} micro-USD against "
                    f"{previous} for the {self._window} before (ratio {self._ratio})"
                ),
            )
        return CONTINUE

    def _bounded_ceiling(self, snapshot: GuardInput) -> tuple[bool, int | None]:
        """The per-cycle ceiling a doubly-capped episode is judged by.

        ``(False, None)`` -- the episode lacks an explicit step budget or an
        explicit provider-cost cap, so the ratio check applies as before.
        ``(True, None)`` -- it is bounded, but its step budget is spent, so
        there is no allowance to prorate and the step budget itself is the
        binding authority.
        ``(True, ceiling)`` -- the episode is bounded and still has steps to
        spend: the ceiling is the FLOOR of the remaining cost budget divided by
        the remaining steps, times the margin.

        ``bounded_cycle_margin`` default (4.0) is sized from the runner's own
        budget model: ``scripts/run_episode.py`` allows up to TWO model cycles
        per step (``model_cycles = max_steps * 2``), so a per-cycle ceiling
        prorated from a per-step allowance is unreachable below 2x, and the
        second 2x absorbs a legitimately expensive cycle (a large frame set on
        a long turn) without turning the ceiling into a proxy for the ratio it
        replaced. A remaining budget already at or below zero yields a ceiling
        at or below zero: the episode is at its cap, and ``BudgetCapGuard``
        (which runs ahead of this guard in ``default_guards``) fires on that
        same snapshot. Firing here too is deliberate -- this guard must bound a
        runaway when it is used on its own.
        """

        usd_cap = _capped_usd_cap(snapshot)
        if usd_cap is None or snapshot.max_steps is None:
            return False, None
        remaining_steps = snapshot.max_steps - snapshot.steps_taken
        if remaining_steps <= 0:
            return True, None
        remaining_usd = usd_cap - snapshot.provider_cost_micros
        ceiling = remaining_usd * self._bounded_margin // remaining_steps
        return True, int(ceiling)


# ---------------------------------------------------------------------------
# Floundering
# ---------------------------------------------------------------------------


def _decided(turns: Sequence[EpisodeTurn]) -> list[EpisodeTurn]:
    """Turns that have a batch; the current, undecided turn is excluded."""
    return [turn for turn in turns if turn.batch is not None]


def _is_waiting(turn: EpisodeTurn) -> bool:
    """A batch made only of ``wait`` actions.

    Waiting is NOT acting. ``wait`` is the one legal way the protocol gives a
    model to let a slow page or app finish loading (bounded at ``MAX_WAIT_MS``
    per action, so a 25 s load is legitimately five 5 s waits on a static
    screen), and both floundering guards would otherwise read that as the
    model repeating itself while the screen does not move. So a wait-only
    batch counts neither as a repeat nor as an action that should have changed
    the screen; it is transparent to both guards.
    """
    return turn.batch is not None and all(
        isinstance(action, WaitAction) for action in turn.batch.actions
    )


def _acted(turns: Sequence[EpisodeTurn]) -> list[EpisodeTurn]:
    """Decided turns whose batch actually acted (non-empty, not wait-only)."""
    return [
        turn
        for turn in turns
        if turn.batch is not None and turn.batch.actions and not _is_waiting(turn)
    ]


def _action_span(turns: Sequence[EpisodeTurn], repeats: int) -> Sequence[EpisodeTurn]:
    """Include the result of every counted action, including intervening waits.

    Turn i's batch produces turn i+1's observation. Without that final,
    undecided observation, a guard could stop on the very action that moved
    the screen. Waits do not count as attempts, but progress during a wait
    still disproves a stationary loop.
    """
    if not turns or turns[-1].batch is not None:
        return ()
    acting = 0
    for index in range(len(turns) - 2, -1, -1):
        turn = turns[index]
        if turn.batch is None:
            return ()
        if turn.batch.actions and not _is_waiting(turn):
            acting += 1
            if acting == repeats:
                return turns[index:]
    return ()


def _unchanged_observations(turns: Sequence[EpisodeTurn]) -> bool:
    """Exact public text and ordered frame digests, never capture IDs or sequence.

    Digests are already verified by the runner, so equality needs no image I/O
    or visual-similarity threshold. Preserve every frame's order and count:
    an unchanged first frame cannot hide progress in a later one. Missing
    both text and frames is unknown state, not evidence of no progress.
    """
    if not turns:
        return False
    keys = []
    for turn in turns:
        observation = turn.observation
        if observation.text is None and not observation.frames:
            return False
        keys.append(
            (observation.text, tuple(frame.artifact.sha256 for frame in observation.frames))
        )
    return len(set(keys)) == 1


class RepeatedBatchGuard:
    """Truncate when ``repeats`` identical batches leave observable state unchanged.

    Repeated input can legitimately advance pages or move through a view, so
    action identity alone proves nothing. Compare canonical actions without
    observation IDs, and require exact public text and frame equality across
    the entire span including the last result. Wait-only batches are
    transparent (see :func:`_is_waiting`), but their observations still count
    as evidence of progress.
    """

    def __init__(self, repeats: int = 4) -> None:
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        if repeats > RECENT_TURNS_WINDOW:
            raise ValueError(f"repeats must not exceed RECENT_TURNS_WINDOW ({RECENT_TURNS_WINDOW})")
        self._repeats = repeats

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        span = _action_span(snapshot.recent_turns, self._repeats)
        keys = {_batch_key(turn) for turn in _acted(span)}
        if len(keys) == 1 and _unchanged_observations(span):
            return GuardVerdict(
                kind="truncate",
                code="repeated-batch",
                detail=(
                    f"the same action batch was issued {self._repeats} times "
                    "without observable change"
                ),
            )
        return CONTINUE


def _batch_key(turn: EpisodeTurn) -> str:
    assert turn.batch is not None
    actions = []
    for action in turn.batch.actions:
        payload = action.model_dump(mode="json")
        payload.pop("observation_id", None)
        actions.append(payload)
    return json.dumps(actions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class NoChangeGuard:
    """Truncate when ``repeats`` NON-EMPTY batches leave frames AND text unchanged.

    Share exact observation equality with :class:`RepeatedBatchGuard` so text
    progress beside a static image cannot trigger a conflicting stop. This
    guard still requires frames throughout; a text-only benchmark never trips
    it, since different actions can legitimately produce the same text.

    Wait-only batches are not actions (see :func:`_is_waiting`): a model
    waiting on a static screen is loading, not floundering. Waits interleaved
    with real actions do not reset the count either -- if the screen stayed
    byte-identical through ``repeats`` real actions and whatever waits sat
    between them, the actions changed nothing.
    """

    def __init__(self, repeats: int = 6) -> None:
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        if repeats + 1 > RECENT_TURNS_WINDOW:
            raise ValueError(
                f"repeats + 1 must not exceed RECENT_TURNS_WINDOW ({RECENT_TURNS_WINDOW})"
            )
        self._repeats = repeats

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        span = _action_span(snapshot.recent_turns, self._repeats)
        if any(not turn.observation.frames for turn in span):
            return CONTINUE
        if _unchanged_observations(span):
            return GuardVerdict(
                kind="truncate",
                code="no-change",
                detail=f"{self._repeats} consecutive actions left frames and text unchanged",
            )
        return CONTINUE


class AskLoopGuard:
    """Truncate after ``asks`` consecutive ``ask_user`` batches.

    Asking is legitimate once; asking again immediately after being answered
    means the model is not acting on answers, and every ask costs a simulator
    turn (or a real person's attention).
    """

    def __init__(self, asks: int = 3) -> None:
        if asks < 1:
            raise ValueError("asks must be positive")
        if asks > RECENT_TURNS_WINDOW:
            raise ValueError(f"asks must not exceed RECENT_TURNS_WINDOW ({RECENT_TURNS_WINDOW})")
        self._asks = asks

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        decided = _decided(snapshot.recent_turns)
        if len(decided) < self._asks:
            return CONTINUE
        tail = decided[-self._asks :]
        if all(
            turn.batch is not None
            and len(turn.batch.actions) == 1
            and isinstance(turn.batch.actions[0], AskUserAction)
            for turn in tail
        ):
            return GuardVerdict(
                kind="truncate",
                code="ask-loop",
                detail=f"{self._asks} consecutive ask_user turns",
            )
        return CONTINUE


def default_guards(config: Any) -> tuple[EpisodeGuard, ...]:
    """The guard set an episode runs with when none is configured.

    ``config`` is duck-typed (``max_cycle_cost_micros`` is the only field
    read) so this module does not import ``episode`` and ``episode`` can
    import it. Every guard here is on by default because each one converts a
    reported-after-the-fact failure into a scored stop; the cost-rate cap is
    the one that needs a configured number, and its ratio check is judged only
    for an episode that does not declare BOTH a step budget and a cost cap
    (see :class:`CostRateGuard`).
    """

    max_cycle = getattr(config, "max_cycle_cost_micros", None)
    return (
        BudgetCapGuard(),
        CostRateGuard(max_cycle_cost_micros=max_cycle),
        RepeatedBatchGuard(),
        NoChangeGuard(),
        AskLoopGuard(),
    )
