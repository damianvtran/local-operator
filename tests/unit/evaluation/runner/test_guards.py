"""Each guard's trigger and non-trigger on a synthetic snapshot.

Guards are pure functions over ``GuardInput``; nothing here touches an
adapter or a writer. The runner-level consequence of a verdict (a scored
truncation carrying the code) is asserted in ``test_episode.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.protocol import (
    ActionBatch,
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    BudgetAuthorization,
    CappedAllowance,
)
from local_operator.evaluation.runner.guards import (
    AskLoopGuard,
    BudgetCapGuard,
    CostRateGuard,
    EpisodeGuard,
    GuardInput,
    NoChangeGuard,
    RepeatedBatchGuard,
    default_guards,
)
from local_operator.evaluation.runner.model import EpisodeTurn


def _budget(**caps: int) -> BudgetAuthorization:
    return BudgetAuthorization(
        episode_id="episode-1",
        allowances=tuple(
            CappedAllowance(
                resource=resource, value=caps.get(resource, 1_000_000), reporting="optional"
            )
            for resource in BUDGET_RESOURCES
        ),
    )


def _observation(sequence: int, *, frame_digest: str | None = None) -> Observation:
    frames: tuple[FrameRef, ...] = ()
    if frame_digest is not None:
        frames = (
            FrameRef(
                frame_id=f"frame-{sequence}",
                artifact=ArtifactRef(sha256=frame_digest, media_type="image/png", byte_count=8),
                geometry=FrameGeometry(
                    native=FrameSize(width=1, height=1),
                    model_visible=FrameSize(width=1, height=1),
                ),
            ),
        )
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text=f"state-{sequence}",
        frames=frames,
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def _batch(current: Observation, actions: Sequence[dict[str, Any]]) -> ActionBatch:
    return ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": current.task_id,
            "episode_id": current.episode_id,
            "observation_id": current.observation_id,
            "actions": [{**action, "observation_id": current.observation_id} for action in actions],
        },
        strict=True,
    )


WAIT = {"kind": "wait", "duration_ms": 1}
CLICK = {"kind": "click", "frame_id": "frame-0", "x": 1, "y": 1}
ASK = {"kind": "ask_user", "request_id": "ask-1", "question": "what?"}


def _turns(
    kinds: Sequence[Sequence[dict[str, Any]] | None], *, digests: Sequence[str] | None = None
):
    """Turns oldest-first; ``None`` is the undecided current turn."""

    turns: list[EpisodeTurn] = []
    for index, actions in enumerate(kinds):
        digest = digests[index] if digests is not None else None
        observation = _observation(index, frame_digest=digest)
        batch = _batch(observation, actions) if actions is not None else None
        turns.append(EpisodeTurn(observation=observation, batch=batch))
    return tuple(turns)


def _snapshot(**overrides: Any) -> GuardInput:
    values: dict[str, Any] = {
        "steps_taken": 3,
        "model_cycles": 3,
        "provider_cost_micros": 21,
        "elapsed_ms": 1000,
        "usage_totals": {"input_tokens": 30, "output_tokens": 15},
        "recent_turns": (),
        "recent_costs_micros": (7, 7, 7),
        "budget": _budget(),
    }
    values.update(overrides)
    return GuardInput(**values)


def test_budget_cap_fires_at_the_cap_and_not_below() -> None:
    guard = BudgetCapGuard()
    assert guard.evaluate(_snapshot(budget=_budget(provider_usd_micros=22))).kind == "continue"
    verdict = guard.evaluate(_snapshot(budget=_budget(provider_usd_micros=21)))
    assert verdict.kind == "truncate" and verdict.code == "budget-cap"
    assert "provider_usd_micros" in verdict.detail
    # Token caps read the usage totals, not the cost.
    verdict = guard.evaluate(_snapshot(budget=_budget(provider_input_tokens=30)))
    assert verdict.kind == "truncate"
    assert guard.evaluate(_snapshot(budget=_budget(model_cycles=4))).kind == "continue"
    assert guard.evaluate(_snapshot(budget=_budget(model_cycles=3))).kind == "truncate"


def test_cost_rate_needs_two_full_windows_then_fires_on_a_spike() -> None:
    guard = CostRateGuard(window=3, ratio=2.0)
    assert guard.evaluate(_snapshot(recent_costs_micros=(1, 1, 1, 9, 9))).kind == "continue"
    assert guard.evaluate(_snapshot(recent_costs_micros=(1, 1, 1, 2, 2, 2))).kind == "continue"
    verdict = guard.evaluate(_snapshot(recent_costs_micros=(1, 1, 1, 3, 3, 3)))
    assert verdict.kind == "truncate" and verdict.code == "cost-spike"
    # A zero-cost previous window cannot be "exceeded by ratio".
    assert guard.evaluate(_snapshot(recent_costs_micros=(0, 0, 0, 5, 5, 5))).kind == "continue"


def test_cost_rate_absolute_cap_fires_on_one_expensive_cycle() -> None:
    guard = CostRateGuard(max_cycle_cost_micros=100)
    assert guard.evaluate(_snapshot(recent_costs_micros=(7, 100))).kind == "continue"
    verdict = guard.evaluate(_snapshot(recent_costs_micros=(7, 101)))
    assert verdict.kind == "truncate" and verdict.code == "cost-spike"


def test_repeated_batch_ignores_the_observation_id_and_the_undecided_turn() -> None:
    guard = RepeatedBatchGuard(repeats=3)
    same = _turns([[WAIT], [WAIT], [WAIT], None])
    verdict = guard.evaluate(_snapshot(recent_turns=same))
    assert verdict.kind == "truncate" and verdict.code == "repeated-batch"
    varied = _turns([[WAIT], [WAIT], [CLICK], None])
    assert guard.evaluate(_snapshot(recent_turns=varied)).kind == "continue"
    short = _turns([[WAIT], [WAIT], None])
    assert guard.evaluate(_snapshot(recent_turns=short)).kind == "continue"


def test_no_change_needs_identical_frames_after_non_empty_batches() -> None:
    guard = NoChangeGuard(repeats=2)
    still = _turns([[CLICK], [CLICK], None], digests=["a" * 64, "a" * 64, "a" * 64])
    verdict = guard.evaluate(_snapshot(recent_turns=still))
    assert verdict.kind == "truncate" and verdict.code == "no-change"
    moved = _turns([[CLICK], [CLICK], None], digests=["a" * 64, "a" * 64, "b" * 64])
    assert guard.evaluate(_snapshot(recent_turns=moved)).kind == "continue"
    # No frames at all: a text-only benchmark can never trip this guard.
    blind = _turns([[CLICK], [CLICK], None])
    assert guard.evaluate(_snapshot(recent_turns=blind)).kind == "continue"


def test_ask_loop_counts_consecutive_ask_batches_only() -> None:
    guard = AskLoopGuard(asks=2)
    looping = _turns([[ASK], [ASK], None])
    verdict = guard.evaluate(_snapshot(recent_turns=looping))
    assert verdict.kind == "truncate" and verdict.code == "ask-loop"
    interleaved = _turns([[ASK], [CLICK], [ASK], None])
    assert guard.evaluate(_snapshot(recent_turns=interleaved)).kind == "continue"


def test_default_guards_composition_and_config_knob() -> None:
    class _Config:
        max_cycle_cost_micros = 5

    guards = default_guards(_Config())
    assert [type(guard).__name__ for guard in guards] == [
        "BudgetCapGuard",
        "CostRateGuard",
        "RepeatedBatchGuard",
        "NoChangeGuard",
        "AskLoopGuard",
    ]
    assert all(isinstance(guard, EpisodeGuard) for guard in guards)
    # The configured per-cycle cap reached the cost guard.
    verdict = guards[1].evaluate(_snapshot(recent_costs_micros=(6,)))
    assert verdict.kind == "truncate"
    assert default_guards(object())[1].evaluate(_snapshot(recent_costs_micros=(6,))).kind == (
        "continue"
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CostRateGuard(window=0),
        lambda: CostRateGuard(ratio=1.0),
        lambda: RepeatedBatchGuard(repeats=1),
        lambda: NoChangeGuard(repeats=1),
        lambda: AskLoopGuard(asks=0),
    ],
)
def test_degenerate_guard_parameters_are_rejected(factory: Any) -> None:
    with pytest.raises(ValueError):
        factory()
