"""The model-facing statement of an episode's step budget.

WHY THIS EXISTS. An evaluation episode is bounded by ``max_steps`` (the OSWorld
2.1 protocol runs at 500), and until this module the model was never told. On
sealed real-model records (Kimi K3.1 Pro, arm ``main25``) the model declared
``finish(done)`` after 16-57 action batches -- 3-11 % of its budget -- and the
episodes ended on incomplete states. A model that cannot see how much budget
it has cannot weigh whether to keep working.

CONSTRAINTS, each load-bearing:

* TRUTHFUL. The position is never counted here. Each caller passes the counter
  it ENFORCES (the runner's ``_steps_taken`` against ``EpisodeConfig.max_steps``;
  the session bridge's admitted and attempted clocks against its
  ``max_steps``, which the session driver's stop gate reads too), read at the
  moment the observation is shown, so a cap changed mid-run is stated as the
  value now enforced and the line cannot drift from the check that ends the
  episode. A screen RE-ATTACHED later (the completion challenge) is restated
  with the position at that later moment (:func:`restate_step_budget`), never
  the count it carried when first shown.
* STEPS ONLY. A wall-clock or dollar bound is not a step count; it is never
  phrased as one. Those bounds produce no position and therefore no line.
* NEVER A CRASH. A cap below 1 (``max_steps=0``) is a run that ends at its
  first call; it has no position to state, so :func:`enforced_position` gives
  ``None`` and the screen renders with no line, exactly as before this module.
* NEUTRAL. The line states the numbers and that the budget ends the episode,
  without claiming that is the only ending (a time, cost or no-progress bound
  can end it too). It never tells
  the model to keep going, never judges the work, and never names a benchmark,
  application or task family -- it is a fact about the episode, not advice.
* ZERO COST WHEN UNBOUNDED. ``None`` in, ``None`` out: a turn with no budget
  renders byte-identically to the render before this module existed (pinned
  in ``tests/unit/evaluation/runner/test_step_budget.py``). Ordinary lop
  sessions have no step budget and never reach this code.
"""

from __future__ import annotations

from typing import Any

from local_operator.evaluation.protocol import ProtocolModel
from local_operator.evaluation.receipts import PositiveSafeCount, SafeCount


class StepPosition(ProtocolModel):
    """Steps already charged against the enforced budget, and that budget."""

    used: SafeCount
    budget: PositiveSafeCount


def enforced_position(used: int, budget: int) -> StepPosition | None:
    """The position on an enforced counter, or ``None`` when no step is allowed.

    Both arms build their position here so a degenerate cap is handled once:
    ``budget < 1`` means the episode is refused at its first call, and a
    ``0 of 0`` line would be a nonsense count, so it states nothing.
    """

    if budget < 1:
        return None
    return StepPosition(used=used, budget=budget)


def step_budget_line(position: StepPosition | None) -> str | None:
    """One sentence stating the position, or ``None`` when there is no budget.

    ``remaining`` floors at zero: the enforced checks are ``used >= budget``,
    so a cap lowered below what was already spent leaves nothing, never a
    negative count the model would have to interpret.
    """

    if position is None:
        return None
    remaining = max(position.budget - position.used, 0)
    return (
        f"Step budget: {position.used} of {position.budget} steps used, "
        f"{remaining} remaining. Spending the budget ends the episode; it can "
        "also end earlier, for example when you finish."
    )


def restate_step_budget(
    blocks: list[Any], shown: StepPosition | None, current: StepPosition | None
) -> list[Any]:
    """Re-attached blocks with the line they were shown with made current.

    A screen re-attached after a step was charged (the completion challenge
    consumes one) would otherwise state the count from when it was first
    shown -- one behind the enforced position at the moment the model reads
    it. Only the EXACT line rendered from ``shown`` is replaced, first
    occurrence only: it sits in the header, ahead of the observation's own
    text, so nothing the environment wrote can be rewritten. A screen rendered
    without a line (no budget wired) is returned unchanged; one whose current
    position is ``None`` loses the line rather than keep a stale count.
    """

    old = step_budget_line(shown)
    if old is None:
        return blocks
    new = step_budget_line(current)
    restated: list[Any] = []
    replaced = False
    for block in blocks:
        text = getattr(block, "text", None)
        if not replaced and isinstance(text, str) and old in text:
            replaced = True
            fresh = text.replace(old + "\n", "", 1) if new is None else text.replace(old, new, 1)
            block = block.model_copy(update={"text": fresh})
        restated.append(block)
    return restated
