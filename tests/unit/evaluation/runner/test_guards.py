"""Each guard's trigger and non-trigger on a synthetic snapshot.

Guards are pure functions over ``GuardInput``; nothing here touches an
adapter or a writer. The runner-level consequence of a verdict (a scored
truncation carrying the code) is asserted in ``test_episode.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest
from pydantic import ValidationError

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
    RECENT_TURNS_WINDOW,
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


def _observation(
    sequence: int, *, frame_digest: str | tuple[str, ...] | None = None, text: str | None = "state"
) -> Observation:
    digests = (frame_digest,) if isinstance(frame_digest, str) else (frame_digest or ())
    frames = tuple(
        FrameRef(
            frame_id=f"frame-{sequence}-{index}",
            artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=8),
            geometry=FrameGeometry(
                native=FrameSize(width=1, height=1),
                model_visible=FrameSize(width=1, height=1),
            ),
        )
        for index, digest in enumerate(digests)
    )
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text=text,
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


WAIT = {"kind": "wait", "duration_ms": 5000}
CLICK = {"kind": "click", "frame_id": "frame-0", "x": 1, "y": 1}
TYPE = {"kind": "type", "text": "hello"}
FORWARD = {"kind": "key", "keys": ["RIGHT"]}
ASK = {"kind": "ask_user", "request_id": "ask-1", "question": "what?"}


def _turns(
    kinds: Sequence[Sequence[dict[str, Any]] | None],
    *,
    digests: Sequence[str | tuple[str, ...] | None] | None = None,
    texts: Sequence[str | None] | None = None,
) -> tuple[EpisodeTurn, ...]:
    """Turns oldest-first; ``None`` is the undecided current turn."""

    turns: list[EpisodeTurn] = []
    for index, actions in enumerate(kinds):
        digest = digests[index] if digests is not None else None
        observation = _observation(
            index, frame_digest=digest, text=texts[index] if texts is not None else "state"
        )
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


def test_cost_rate_says_so_when_cost_is_unreported() -> None:
    """A zero previous window (unreported cost, or a free tier) has no rate to
    exceed. The guard must not silently continue as if it had judged; it
    names the skip so the ratio check's inertness is visible."""

    guard = CostRateGuard(window=3, ratio=2.0)
    verdict = guard.evaluate(_snapshot(recent_costs_micros=(0, 0, 0, 5, 5, 5)))
    assert verdict.kind == "continue"
    assert verdict.code == "cost-unreported"
    assert "reported no cost" in verdict.detail
    # The absolute cap still applies without reported ratio data.
    capped = CostRateGuard(window=3, ratio=2.0, max_cycle_cost_micros=4)
    assert capped.evaluate(_snapshot(recent_costs_micros=(0, 0, 0, 5, 5, 5))).kind == "truncate"


def test_cost_rate_absolute_cap_fires_on_one_expensive_cycle() -> None:
    guard = CostRateGuard(max_cycle_cost_micros=100)
    assert guard.evaluate(_snapshot(recent_costs_micros=(7, 100))).kind == "continue"
    verdict = guard.evaluate(_snapshot(recent_costs_micros=(7, 101)))
    assert verdict.kind == "truncate" and verdict.code == "cost-spike"


def test_repeated_batch_ignores_observation_ids_but_requires_unchanged_state() -> None:
    guard = RepeatedBatchGuard(repeats=3)
    same = _turns([[CLICK], [CLICK], [CLICK], None])
    verdict = guard.evaluate(_snapshot(recent_turns=same))
    assert verdict.kind == "truncate" and verdict.code == "repeated-batch"
    varied = _turns([[CLICK], [CLICK], [TYPE], None])
    assert guard.evaluate(_snapshot(recent_turns=varied)).kind == "continue"
    short = _turns([[CLICK], [CLICK], None])
    assert guard.evaluate(_snapshot(recent_turns=short)).kind == "continue"


@pytest.mark.parametrize("digests", [["a", "b", "c", "d", "e"], ["a"] * 4 + ["b"]])
def test_repeated_forward_input_with_visible_progress_continues(digests: list[str]) -> None:
    # The newest, undecided observation contains the final forward input's result.
    turns = _turns([[FORWARD]] * 4 + [None], digests=[digest * 64 for digest in digests])
    assert RepeatedBatchGuard().evaluate(_snapshot(recent_turns=turns)).kind == "continue"


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


def test_waiting_on_a_static_screen_is_loading_not_floundering() -> None:
    """The review's reproduction (M1): a model issuing ``wait`` on a static
    screen -- a page or app still loading, the protocol's one legal way to
    wait -- must not be scored as repeating itself or as acting to no effect.
    Seven wait-only turns on one frame digest trip neither default guard."""

    static = ["a" * 64] * 8
    loading = _turns([[WAIT]] * 7 + [None], digests=static)
    assert RepeatedBatchGuard().evaluate(_snapshot(recent_turns=loading)).kind == "continue"
    assert NoChangeGuard().evaluate(_snapshot(recent_turns=loading)).kind == "continue"
    # A batch mixing a wait with a real action IS acting.
    mixed = _turns([[CLICK, WAIT]] * 7 + [None], digests=static)
    assert RepeatedBatchGuard().evaluate(_snapshot(recent_turns=mixed)).code == "repeated-batch"
    assert NoChangeGuard().evaluate(_snapshot(recent_turns=mixed)).code == "no-change"


def test_waits_are_transparent_to_a_run_of_real_actions() -> None:
    """Waits interleaved with identical real actions neither break the
    repeat count nor reset the no-change span: the actions changed nothing
    whatever waits sat between them."""

    interleaved = _turns([[CLICK], [WAIT], [CLICK], [WAIT], [CLICK], None])
    verdict = RepeatedBatchGuard(repeats=3).evaluate(_snapshot(recent_turns=interleaved))
    assert verdict.code == "repeated-batch"

    static = ["a" * 64] * 6
    still = _turns([[CLICK], [WAIT], [CLICK], [WAIT], [CLICK], None], digests=static)
    assert NoChangeGuard(repeats=3).evaluate(_snapshot(recent_turns=still)).code == "no-change"
    # Two real actions plus waits is below a repeats=3 span.
    fewer = _turns([[CLICK], [WAIT], [WAIT], [CLICK], None], digests=static[:5])
    assert NoChangeGuard(repeats=3).evaluate(_snapshot(recent_turns=fewer)).kind == "continue"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
def test_loop_guards_allow_changed_text_beside_identical_frames(guard_type: Any) -> None:
    turns = _turns(
        [[FORWARD], [FORWARD], None],
        digests=["a" * 64] * 3,
        texts=["page 1", "page 1", "page 2"],
    )
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_loop_guards_need_observable_state_throughout(guard_type: Any, missing_index: int) -> None:
    digests: list[str | None] = ["a" * 64] * 3
    texts: list[str | None] = ["state"] * 3
    digests[missing_index] = None
    texts[missing_index] = None
    turns = _turns([[TYPE], [TYPE], None], digests=digests, texts=texts)
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"
    blind = _turns([[TYPE], [TYPE], None], texts=[None] * 3)
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=blind)).kind == "continue"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
@pytest.mark.parametrize("final_frames", [("a", "c"), ("b", "a"), ("a",)])
def test_loop_guards_compare_every_frame_in_order(
    guard_type: Any, final_frames: tuple[str, ...]
) -> None:
    frames = ("a" * 64, "b" * 64)
    turns = _turns(
        [[TYPE], [TYPE], None],
        digests=[frames, frames, tuple(digest * 64 for digest in final_frames)],
    )
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
def test_loop_guards_ignore_capture_ids_but_stop_real_stationary_loops(guard_type: Any) -> None:
    frames = ("a" * 64, "b" * 64)
    turns = _turns([[FORWARD], [FORWARD], None], digests=[frames] * 3, texts=[None] * 3)
    assert len({turn.observation.observation_id for turn in turns}) == 3
    assert len({turn.observation.frames[0].frame_id for turn in turns}) == 3
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "truncate"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
@pytest.mark.parametrize("progress_index", [1, 2, 3])
def test_waits_do_not_hide_observation_progress(guard_type: Any, progress_index: int) -> None:
    # The view may move during a wait then return before the next action.
    digests = ["a" * 64] * 4
    digests[progress_index] = "b" * 64
    turns = _turns([[TYPE], [WAIT], [TYPE], None], digests=digests)
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"


@pytest.mark.parametrize("guard_type", [RepeatedBatchGuard, NoChangeGuard])
def test_last_action_needs_its_result_observation(guard_type: Any) -> None:
    turns = _turns([[TYPE]] * 3, digests=["a" * 64] * 3)
    assert guard_type(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"


@pytest.mark.parametrize("texts", [["state"] * 3, ["state", "state", None]])
def test_repeated_batch_handles_text_only_state_conservatively(texts: list[str | None]) -> None:
    turns = _turns([[TYPE], [TYPE], None], texts=texts)
    verdict = RepeatedBatchGuard(repeats=2).evaluate(_snapshot(recent_turns=turns))
    assert verdict.kind == ("truncate" if texts[-1] is not None else "continue")
    assert NoChangeGuard(repeats=2).evaluate(_snapshot(recent_turns=turns)).kind == "continue"


def test_repeated_batch_maximum_fits_completed_turns_plus_latest_observation() -> None:
    # EpisodeRunner supplies the full completed-turn window PLUS the new result.
    turns = _turns([[TYPE]] * RECENT_TURNS_WINDOW + [None])
    verdict = RepeatedBatchGuard(repeats=RECENT_TURNS_WINDOW).evaluate(
        _snapshot(recent_turns=turns)
    )
    assert verdict.code == "repeated-batch"


@pytest.mark.parametrize("text", ["", " "])
def test_empty_text_is_rejected_by_observation_contract(text: str) -> None:
    with pytest.raises(ValidationError):
        _observation(0, text=text)


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
        # A guard needing more turns than the runner snapshots could never
        # fire; it is refused at construction rather than left inert.
        lambda: RepeatedBatchGuard(repeats=RECENT_TURNS_WINDOW + 1),
        lambda: NoChangeGuard(repeats=RECENT_TURNS_WINDOW),
        lambda: AskLoopGuard(asks=RECENT_TURNS_WINDOW + 1),
    ],
)
def test_degenerate_guard_parameters_are_rejected(factory: Any) -> None:
    with pytest.raises(ValueError):
        factory()
