"""The session arm's opt-in no-progress policy: the runner's own guards, at the bridge.

THE MEASURED GAP (OSWorld task030, session arm, 2026-10-28). After step 7 the
model repeated one no-op key batch; the bridge accepted 53 remote calls and the
model then blamed the transport. Replaying the recorded batches through the
runner's EXISTING ``RepeatedBatchGuard``/``NoChangeGuard`` (thresholds 4 and 6)
fires at step 14 and step 16 respectively -- the reply arm would have stopped
it, the session arm had no guard at all.

These tests pin the parity, not a new detector: the bridge runs the SAME guard
objects at the SAME point the runner does (after an executed step, before the
model's next decision), with the runner's thresholds, and a firing guard ends
the episode as the runner's truncation does -- a scored stop with the guard's
stable code as the terminal reason. Off by default: an unset policy runs no
guard code at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ExecuteResult,
    ExecutionReceipt,
    observation_content_id,
)
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.runner.guards import (
    NoChangeGuard,
    RepeatedBatchGuard,
    progress_guards,
)
from local_operator.evaluation.session_arm import (
    PROGRESS_STOP_ACK,
    ActionBridge,
)
from local_operator.harness.types import TextContent
from tests.unit.evaluation.runner.conftest import build_spec

#: The episode's own budget, as the driver passes ``spec.budget``. The
#: turn-window guards never read it; ``GuardInput`` requires a real one.
BUDGET = build_spec("ep-progress").budget

#: Two distinct "screens". Only the digest is compared (the guards never read
#: image bytes), so the artifact does not have to exist on disk here.
STATIC = "a" * 64
MOVED = "b" * 64


def _frame(sha: str) -> FrameRef:
    size = FrameSize(width=1280, height=720)
    return FrameRef(
        frame_id="screen",
        artifact=ArtifactRef(sha256=sha, media_type="image/png", byte_count=1),
        geometry=FrameGeometry(native=size, model_visible=size),
    )


def _observation(sequence: int, sha: str = STATIC, text: str = "terminal") -> Observation:
    draft = Observation(
        task_id="task_001",
        episode_id="ep-progress",
        sequence=sequence,
        observation_id="draft",
        text=text,
        frames=(_frame(sha),),
    )
    return draft.model_copy(update={"observation_id": observation_content_id(draft)})


class _World:
    """A scripted environment: each executed batch yields the next screen.

    ``screens`` is a function of the 1-based step, so a test states exactly
    which steps changed the screen. Observation IDs differ every step (the
    sequence is part of the content id) -- the real clock-only case: an ID
    changing is never progress, the digests and text are.
    """

    def __init__(self, screens: Any) -> None:
        self.screens = screens
        self.current = _observation(0)
        self.batches: list[Any] = []

    async def execute(self, batch: Any) -> ExecuteResult:
        self.batches.append(batch)
        step = len(self.batches)
        output = _observation(step, *self.screens(step))
        receipt = ExecutionReceipt(
            operation_id=f"op-{step}",
            action_batch_id=hashlib.sha256(str(step).encode()).hexdigest(),
            input_observation_id=self.current.observation_id,
            output_observation_id=output.observation_id,
            sequence=output.sequence,
        )
        self.current = output
        return ExecuteResult(receipt=receipt, observation=output)


def _bridge(
    tmp_path: Path, world: _World, *, guards: Any = None, max_steps: int = 60
) -> tuple[ActionBridge, list[tuple[str, dict[str, Any]]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    bridge = ActionBridge(
        endpoint=tmp_path / "b.sock",
        surface=ActionSurface(),
        render=lambda observation: [TextContent(text=f"seen {observation.sequence}")],
        execute=world.execute,
        max_steps=max_steps,
        record=lambda kind, payload: records.append((kind, payload)),
        progress_guards=guards,
        progress_budget=BUDGET if guards else None,
    )
    bridge.arm(world.current)
    return bridge, records


async def _drive(bridge: ActionBridge, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One call per turn, exactly as the session delivers them."""

    from local_operator.harness.types import TurnEndEvent

    replies = []
    for arguments in batches:
        replies.append(await bridge.call(arguments))
        bridge.fold(TurnEndEvent())
    return replies


#: The task030 shape: the same no-op chord, every turn, on a frozen screen.
NOOP = {"actions": [{"kind": "key", "keys": ["ctrl", "l"]}]}


def _last_fire(replies: list[dict[str, Any]]) -> int | None:
    for index, reply in enumerate(replies):
        if reply["details"].get("terminal") == "progress-guard":
            return index
    return None


@pytest.mark.asyncio
async def test_repeated_noop_batches_stop_at_the_runner_threshold(tmp_path: Path) -> None:
    """RepeatedBatchGuard(4): the 4th identical no-op batch is the last one run."""

    world = _World(lambda step: (STATIC,))
    bridge, records = _bridge(tmp_path, world, guards=progress_guards())
    replies = await _drive(bridge, [NOOP] * 10)

    fired = _last_fire(replies)
    assert fired == 3  # the 4th call: its batch ran, and the guard read its result
    assert len(world.batches) == 4
    assert bridge.steps == 4
    assert bridge.end_requested == "repeated-batch"
    assert bridge.end_diagnostic is not None
    assert "4 times without observable change" in bridge.end_diagnostic
    # The stopping call still returns the screen its batch produced, plus the
    # stop sentence -- the model is told WHY, and that nothing failed.
    text = " ".join(c.text for c in replies[fired]["content"])
    assert PROGRESS_STOP_ACK.split(":")[0] in text
    assert "seen 4" in text
    assert replies[fired]["is_error"] is False
    # Every later call is refused before execute: the token is terminal.
    assert all(r["is_error"] for r in replies[fired + 1 :])
    assert [kind for kind, _ in records].count("progress_stop") == 1
    stop = dict(records)["progress_stop"]
    assert stop["code"] == "repeated-batch" and stop["steps"] == 4


@pytest.mark.asyncio
async def test_varied_noop_actions_stop_through_no_change_at_six(tmp_path: Path) -> None:
    """Different actions, frozen screen: NoChangeGuard(6), not RepeatedBatch."""

    world = _World(lambda step: (STATIC,))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    batches = [
        {"actions": [{"kind": "key", "keys": [key]}]} for key in ("a", "b", "c", "d", "e", "f", "g")
    ]
    replies = await _drive(bridge, batches)

    assert _last_fire(replies) == 5
    assert bridge.steps == 6
    assert bridge.end_requested == "no-change"


@pytest.mark.asyncio
async def test_repeated_keys_that_advance_the_screen_never_trip(tmp_path: Path) -> None:
    """PageDown x12 turning a page each time is progress, however repetitive."""

    world = _World(lambda step: (hashlib.sha256(str(step).encode()).hexdigest(),))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    replies = await _drive(bridge, [{"actions": [{"kind": "key", "keys": ["pagedown"]}]}] * 12)

    assert _last_fire(replies) is None
    assert bridge.end_requested is None
    assert bridge.steps == 12


@pytest.mark.asyncio
async def test_waiting_on_a_static_screen_never_trips(tmp_path: Path) -> None:
    """Wait-only batches are loading, not floundering (the guards' own rule)."""

    world = _World(lambda step: (STATIC,))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    replies = await _drive(bridge, [{"actions": [{"kind": "wait", "duration_ms": 500}]}] * 12)

    assert _last_fire(replies) is None
    assert bridge.steps == 12


@pytest.mark.asyncio
async def test_text_progress_beside_a_static_frame_is_progress(tmp_path: Path) -> None:
    """A clock-only frame and a CHANGING public text: not unchanged state."""

    world = _World(lambda step: (STATIC, f"typed {step} chars"))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    replies = await _drive(bridge, [{"actions": [{"kind": "type", "text": "x"}]}] * 10)

    assert _last_fire(replies) is None


@pytest.mark.asyncio
async def test_one_change_resets_the_count(tmp_path: Path) -> None:
    """Three no-ops, the screen moves, then the window starts over."""

    world = _World(lambda step: (MOVED if step >= 3 else STATIC,))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    replies = await _drive(bridge, [NOOP] * 12)

    # Step 3 PRODUCED the new screen, so batches 4-7 are the first four decided
    # on it; the span includes the screen each was decided on, so the window
    # completes at step 7 -- three steps later than an unbroken run (step 4).
    assert _last_fire(replies) == 6
    assert bridge.steps == 7


@pytest.mark.asyncio
async def test_disabled_by_default_runs_every_batch(tmp_path: Path) -> None:
    """Unset policy: the bridge behaves exactly as before (the 53-call shape)."""

    world = _World(lambda step: (STATIC,))
    bridge, records = _bridge(tmp_path, world, guards=None)
    replies = await _drive(bridge, [NOOP] * 20)

    assert _last_fire(replies) is None
    assert bridge.steps == 20
    assert bridge.end_requested is None
    assert not any(kind == "progress_stop" for kind, _ in records)
    # No turn window is even kept when disabled: zero added work per call.
    assert bridge._progress_turns == []


@pytest.mark.asyncio
async def test_a_refused_call_is_not_a_turn(tmp_path: Path) -> None:
    """Malformed input never reaches the window: a refusal ran nothing."""

    world = _World(lambda step: (STATIC,))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards())
    bad = {"actions": [{"kind": "key", "keys": ["NOSUCHKEY"]}]}
    replies = await _drive(bridge, [NOOP, bad, bad, NOOP, bad, NOOP, NOOP])

    assert [r["is_error"] for r in replies] == [False, True, True, False, True, False, False]
    assert _last_fire(replies) == 6
    assert bridge.steps == 4


@pytest.mark.asyncio
async def test_the_budget_still_wins_when_it_is_reached_first(tmp_path: Path) -> None:
    world = _World(lambda step: (STATIC,))
    bridge, _ = _bridge(tmp_path, world, guards=progress_guards(), max_steps=3)
    replies = await _drive(bridge, [NOOP] * 5)

    assert _last_fire(replies) is None
    assert bridge.end_requested == "max-steps"
    assert bridge.steps == 3


@pytest.mark.asyncio
async def test_two_bridges_never_share_a_window(tmp_path: Path) -> None:
    """Guard state is per bridge (per episode), even with shared guard objects."""

    shared = progress_guards()
    first_world = _World(lambda step: (STATIC,))
    second_world = _World(lambda step: (STATIC,))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, _ = _bridge(tmp_path / "a", first_world, guards=shared)
    second, _ = _bridge(tmp_path / "b", second_world, guards=shared)
    for _ in range(3):
        await _drive(first, [NOOP])
        await _drive(second, [NOOP])
    assert first.end_requested is None and second.end_requested is None
    await _drive(first, [NOOP])
    assert first.end_requested == "repeated-batch"
    assert second.end_requested is None


def test_the_policy_is_the_runners_guard_set() -> None:
    """No second detector: same types, same thresholds as ``default_guards``."""

    from types import SimpleNamespace

    from local_operator.evaluation.runner.guards import default_guards

    shared = progress_guards()
    runner = default_guards(SimpleNamespace(max_cycle_cost_micros=None))
    assert [type(g) for g in shared] == [type(g) for g in runner[-len(shared) :]]
    assert [vars(g) for g in shared] == [vars(g) for g in runner[-len(shared) :]]
    assert isinstance(shared[0], RepeatedBatchGuard) and isinstance(shared[1], NoChangeGuard)


def test_a_budget_guard_is_refused_by_the_bridge(tmp_path: Path) -> None:
    """The bridge cannot feed cost/budget counters; a guard needing them is refused."""

    from local_operator.evaluation.runner.guards import BudgetCapGuard
    from local_operator.evaluation.session_arm import SessionArmError

    world = _World(lambda step: (STATIC,))
    with pytest.raises(SessionArmError, match="turn-window"):
        _bridge(tmp_path, world, guards=(BudgetCapGuard(),))
