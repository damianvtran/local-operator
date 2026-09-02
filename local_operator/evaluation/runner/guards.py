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

from local_operator.evaluation.protocol import AskUserAction, ProtocolModel
from local_operator.evaluation.receipts import (
    BudgetAuthorization,
    BudgetResource,
    CappedAllowance,
    SafeCount,
    StrictIdentifier,
)
from local_operator.evaluation.runner.model import EpisodeTurn

__all__ = [
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
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False, validate_default=True)

    steps_taken: SafeCount
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
    """

    def __init__(
        self,
        *,
        window: int = 10,
        ratio: float = 3.0,
        max_cycle_cost_micros: int | None = None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if ratio <= 1.0:
            raise ValueError("ratio must exceed 1.0")
        self._window = window
        self._ratio = ratio
        self._max_cycle = max_cycle_cost_micros

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        costs = snapshot.recent_costs_micros
        if self._max_cycle is not None and costs and costs[-1] > self._max_cycle:
            return GuardVerdict(
                kind="truncate",
                code="cost-spike",
                detail=f"one model cycle cost {costs[-1]} micro-USD (cap {self._max_cycle})",
            )
        if len(costs) < 2 * self._window:
            return CONTINUE
        recent = sum(costs[-self._window :])
        previous = sum(costs[-2 * self._window : -self._window])
        if previous > 0 and recent > self._ratio * previous:
            return GuardVerdict(
                kind="truncate",
                code="cost-spike",
                detail=(
                    f"the last {self._window} cycles cost {recent} micro-USD against "
                    f"{previous} for the {self._window} before (ratio {self._ratio})"
                ),
            )
        return CONTINUE


# ---------------------------------------------------------------------------
# Floundering
# ---------------------------------------------------------------------------


def _decided(turns: Sequence[EpisodeTurn]) -> list[EpisodeTurn]:
    """Turns that have a batch; the current, undecided turn is excluded."""
    return [turn for turn in turns if turn.batch is not None]


class RepeatedBatchGuard:
    """Truncate when the last ``repeats`` batches are canonically identical.

    Identical bytes, not "similar": the model issuing the exact same action
    list on consecutive turns is doing nothing new. Batches are compared by
    their canonical JSON with the observation id removed -- the id changes
    every turn by construction, so with it every batch is unique.
    """

    def __init__(self, repeats: int = 4) -> None:
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        self._repeats = repeats

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        decided = _decided(snapshot.recent_turns)
        if len(decided) < self._repeats:
            return CONTINUE
        tail = decided[-self._repeats :]
        keys = {_batch_key(turn) for turn in tail}
        if len(keys) == 1:
            return GuardVerdict(
                kind="truncate",
                code="repeated-batch",
                detail=f"the same action batch was issued {self._repeats} times in a row",
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
    """Truncate when ``repeats`` consecutive NON-EMPTY batches produced
    byte-identical frames -- the model keeps acting and the screen does not
    move.

    Frames are compared by their content digest (``artifact.sha256``), which
    the adapter already computed and the runner already verified, so this
    costs no bytes. An observation with no frames cannot be judged and does
    not count; a text-only benchmark never trips this guard.
    """

    def __init__(self, repeats: int = 6) -> None:
        if repeats < 2:
            raise ValueError("repeats must be at least 2")
        self._repeats = repeats

    def evaluate(self, snapshot: GuardInput) -> GuardVerdict:
        turns = list(snapshot.recent_turns)
        # The frame produced by turn i's batch is turn i+1's observation, so
        # ``repeats`` acted-upon turns need ``repeats + 1`` observations.
        if len(turns) < self._repeats + 1:
            return CONTINUE
        window = turns[-(self._repeats + 1) :]
        digests: list[tuple[str, ...]] = []
        for turn in window:
            if not turn.observation.frames:
                return CONTINUE
            digests.append(tuple(frame.artifact.sha256 for frame in turn.observation.frames))
        acted = window[:-1]
        if any(turn.batch is None or not turn.batch.actions for turn in acted):
            return CONTINUE
        if len(set(digests)) == 1:
            return GuardVerdict(
                kind="truncate",
                code="no-change",
                detail=f"{self._repeats} consecutive actions left the screen byte-identical",
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
    the one that needs a configured number.
    """

    max_cycle = getattr(config, "max_cycle_cost_micros", None)
    return (
        BudgetCapGuard(),
        CostRateGuard(max_cycle_cost_micros=max_cycle),
        RepeatedBatchGuard(),
        NoChangeGuard(),
        AskLoopGuard(),
    )
