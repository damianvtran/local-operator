"""The model-facing statement of an episode's step budget.

WHY THIS EXISTS. An evaluation episode is bounded by ``max_steps`` (the OSWorld
2.1 protocol runs at 500), and until this module the model was never told. On
sealed real-model records (Kimi K3.1 Pro, arm ``main25``) the model declared
``finish(done)`` after 16-57 action batches -- 3-11 % of its budget -- and the
episodes ended on incomplete states. A model that cannot see how much budget
it has cannot weigh whether to keep working; every harness with strong
computer-use numbers states the position.

CONSTRAINTS, each load-bearing:

* TRUTHFUL. The position is never counted here. Each caller passes the counter
  it ENFORCES (the runner's ``_steps_taken`` against ``EpisodeConfig.max_steps``;
  the session bridge's admitted and attempted clocks against its
  ``max_steps``), read at the moment the observation is shown, so a cap changed
  mid-run is stated as the value now enforced and the line cannot drift from
  the check that ends the episode.
* STEPS ONLY. A wall-clock or dollar bound is not a step count; it is never
  phrased as one. Those bounds produce no position and therefore no line.
* NEUTRAL. The line states the numbers and how the episode ends. It never tells
  the model to keep going, never judges the work, and never names a benchmark,
  application or task family -- it is a fact about the episode, not advice.
* ZERO COST WHEN UNBOUNDED. ``None`` in, ``None`` out: a turn with no budget
  renders byte-identically to the render before this module existed (pinned
  in ``tests/unit/evaluation/runner/test_step_budget.py``). Ordinary lop
  sessions have no step budget and never reach this code.
"""

from __future__ import annotations

from local_operator.evaluation.protocol import ProtocolModel
from local_operator.evaluation.receipts import PositiveSafeCount, SafeCount


class StepPosition(ProtocolModel):
    """Steps already charged against the enforced budget, and that budget."""

    used: SafeCount
    budget: PositiveSafeCount


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
        f"{remaining} remaining. The episode ends when the budget is spent or "
        "when you finish, whichever comes first."
    )
