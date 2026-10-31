"""The model-facing step-budget position, on every engagement that has one.

WHY THIS EXISTS. Kimi K3.1 Pro on the OSWorld 2.1 protocol (``max_steps=500``)
declared ``finish(done)`` after 16-57 action batches -- 3-11 % of its budget --
and the episodes ended incomplete. Nothing it was shown said a budget existed.
These tests pin that the position is stated, that it is the ENFORCED counter
(not a second one), that it follows a changed cap, and that a turn without a
budget renders byte-identically to before.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.protocol import Observation
from local_operator.evaluation.runner.model import EpisodeTurn
from local_operator.evaluation.runner.provider_client import _ContextBuilder
from local_operator.evaluation.runner.step_budget import (
    StepPosition,
    enforced_position,
    restate_step_budget,
    step_budget_line,
)
from local_operator.harness.types import TextContent
from tests.unit.evaluation.runner.conftest import FakeAdapter, ScriptedModel
from tests.unit.evaluation.runner.test_episode import _runner


def _obs(sequence: int) -> Observation:
    draft = Observation(
        task_id="t",
        episode_id="e",
        sequence=sequence,
        observation_id="x",
        text=f"state-{sequence}",
    )
    return draft.model_copy(update={"observation_id": observation_content_id(draft)})


def _builder() -> _ContextBuilder:
    return _ContextBuilder(
        artifact_root=Path("/nonexistent"), keep_recent_frames=2, rebuild_every_frames=8
    )


def _first_text(message: Any) -> str:
    """The rendered header block's text, narrowed for the type checker."""

    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _dump(builder: _ContextBuilder) -> str:
    return json.dumps(
        [m.model_dump(mode="json", exclude={"id", "timestamp"}) for m in builder.messages],
        sort_keys=True,
    )


#: The two-turn render as ``origin/main`` (d592f6d47f4) produced it, captured
#: BEFORE this change from the same builder inputs. Pinned as text so a
#: no-budget turn is proved byte-identical to the base, not merely to itself.
BASE_TWO_TURNS = [
    "Task: t\nStep: 0\nObservation ID: "
    "a97c83c255dfdca1a6a0c295ecf28e3f348615feccce2539108522498e08123d\nFrames: none\n\nstate-0",
    "Step: 1\nObservation ID: "
    "97eb096d68612407ea0141904bf7682be726d14dbb50c02622a0998e6aeeb64b\nFrames: none\n\nstate-1",
]


def test_the_line_states_used_budget_and_remaining_neutrally() -> None:
    line = step_budget_line(StepPosition(used=16, budget=500))
    assert line == (
        "Step budget: 16 of 500 steps used, 484 remaining. Spending the budget "
        "ends the episode; it can also end earlier, for example when you finish."
    )
    # Not exhaustive by construction (review r1 M2): a time, cost or
    # no-progress bound can also end an episode, so the line never claims to
    # list every ending.
    assert "whichever comes first" not in line
    assert "only" not in line.lower()
    # Neutral: no instruction to continue, no benchmark/app vocabulary.
    lowered = line.lower()
    for word in ("keep", "continue", "must", "should", "osworld", "task"):
        assert word not in lowered


def test_no_position_means_no_line() -> None:
    assert step_budget_line(None) is None


def test_remaining_never_goes_negative() -> None:
    # A cap lowered below what was already spent: the enforced check is
    # ``used >= budget``, so the truthful remaining count is zero.
    assert "0 remaining" in (step_budget_line(StepPosition(used=7, budget=5)) or "")


@pytest.mark.parametrize("budget", [0, -1])
def test_a_cap_below_one_has_no_position(budget: int) -> None:
    # Review r1 M3: ``max_steps=0`` is an episode refused at its first call.
    # It must produce no position (so no line), never a validation crash and
    # never a ``0 of 0`` count.
    assert enforced_position(0, budget) is None
    assert step_budget_line(enforced_position(0, budget)) is None
    assert enforced_position(3, 5) == StepPosition(used=3, budget=5)


def test_a_restated_screen_carries_the_current_position_only_in_the_header() -> None:
    shown = StepPosition(used=0, budget=5)
    old = step_budget_line(shown)
    assert old is not None
    # The observation's own text happens to contain the old line verbatim; only
    # the header copy (the first) may change.
    blocks = [TextContent(text=f"Step: 0\n{old}\nFrames: none\n\n{old}")]
    current = StepPosition(used=1, budget=5)
    (restated,) = restate_step_budget(blocks, shown, current)
    new = step_budget_line(current)
    assert restated.text == f"Step: 0\n{new}\nFrames: none\n\n{old}"
    # The input is never mutated: the shown blocks stay as first delivered.
    assert blocks[0].text.startswith(f"Step: 0\n{old}")
    # A current position of ``None`` (cap dropped below 1) removes the line.
    (dropped,) = restate_step_budget(blocks, shown, None)
    assert dropped.text == f"Step: 0\nFrames: none\n\n{old}"
    # A screen rendered without a budget is untouched.
    assert restate_step_budget(blocks, None, current) == blocks


def test_a_turn_without_a_budget_renders_byte_identically_to_base() -> None:
    builder = _builder()
    builder.append_new_turns([EpisodeTurn(observation=_obs(0)), EpisodeTurn(observation=_obs(1))])
    texts = [_first_text(m) for m in builder.messages]
    assert texts == BASE_TWO_TURNS


def test_a_bounded_turn_carries_the_position_after_the_header() -> None:
    builder = _builder()
    builder.append_new_turns(
        [
            EpisodeTurn(observation=_obs(0), step_budget=StepPosition(used=0, budget=500)),
            EpisodeTurn(observation=_obs(1), step_budget=StepPosition(used=1, budget=500)),
        ]
    )
    first, second = (_first_text(m) for m in builder.messages)
    expected = step_budget_line(StepPosition(used=1, budget=500))
    assert expected is not None and expected in second
    # The line sits in the header block, never between the task text and the
    # model: the observation's own text is still the message's last line.
    assert second.endswith("\n\nstate-1")
    assert first.startswith("Task: t\n")
    # Removing the line leaves exactly the base render: it is an insertion
    # and nothing else moved.
    assert second.replace(expected + "\n", "") == BASE_TWO_TURNS[1]


@pytest.mark.asyncio
async def test_the_runner_attaches_its_enforced_counter(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model, max_steps=4)

    await runner.run()

    positions = [turn.step_budget for turn in model.histories[2]]
    assert positions == [
        StepPosition(used=0, budget=4),
        StepPosition(used=1, budget=4),
        StepPosition(used=2, budget=4),
    ]
    # The SAME counter the truncation check reads.
    assert runner._steps_taken == 2


@pytest.mark.asyncio
async def test_the_runner_line_follows_a_changed_cap(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    seen: list[Any] = []

    class _Raising(ScriptedModel):
        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            seen.append(history[-1].step_budget)
            if self.calls == 0:
                # A cap raised mid-run: the next observation must state it.
                runner._config = replace(runner._config, max_steps=9)
            return await super().decide(observation, history, **kwargs)

    model = _Raising(["step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model, max_steps=4)

    await runner.run()

    assert seen[0] == StepPosition(used=0, budget=4)
    assert seen[1] == StepPosition(used=1, budget=9)
