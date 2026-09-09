"""The guest must receive a batch's actions in the order the model wrote them.

WHY THIS FILE EXISTS. The adapter used to split a compiled batch into two
buckets -- every guest statement, then every wait -- and run the buckets in
that order. For a batch the model wrote as ``[click, wait 2000, paste_text]``
that executed click, paste, sleep: the paste landed on a UI that had not
finished responding to the click, and the sleep happened afterwards where it
could no longer do anything.

It was not rare. Across the 64 evidence bundles of the 2026-09 Kimi K3 cohort,
425 of 1746 wait-bearing batches (24%) contained a wait followed by a later
action, so in a quarter of the cases where the model deliberately paced itself
against a slow UI the harness silently disobeyed it. On a benchmark whose tasks
are largely about driving real applications, that is a correctness bug that
costs score, not a stylistic one.

These tests drive the REAL ``OSWorldV2Adapter.execute`` against a recording
provider and assert on the order the guest saw. They fail on the two-bucket
implementation, which is the only thing that makes them worth having.
"""

from __future__ import annotations

from typing import Any

import pytest
from lop_osworld_v2_adapter import adapter as adapter_module

from local_operator.evaluation.adapters.api import ExecuteParams
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import (
    PROTOCOL_VERSION,
    ActionBatch,
    ArtifactRef,
    ClickAction,
    FrameGeometry,
    FrameRef,
    FrameSize,
    KeyAction,
    Observation,
    TypeAction,
    WaitAction,
)


class _RecordingProvider:
    """Records every guest interaction in the order it actually happened."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def execute(self, statements: list[str]) -> None:
        self.calls.append(("exec", list(statements)))

    async def observe(self) -> dict[str, Any]:
        self.calls.append(("observe", None))
        return {"screenshot": _PNG}


# A 1x1 PNG: the observation builder only needs something decodable.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082"
)


def _observation(observation_id: str, sequence: int) -> Observation:
    """A real protocol ``Observation`` -- ``ExecuteResult`` accepts nothing less."""
    size = FrameSize(width=1920, height=1080)
    return Observation(
        task_id="task_001",
        episode_id="ep-1",
        sequence=sequence,
        observation_id=observation_id,
        frames=(
            FrameRef(
                frame_id="screen",
                artifact=ArtifactRef(sha256="0" * 64, media_type="image/png", byte_count=len(_PNG)),
                geometry=FrameGeometry(native=size, model_visible=size),
            ),
        ),
    )


class _StubBuilder:
    def build(self, raw: Any, **kwargs: Any) -> Observation:
        return _observation("obs-2", kwargs.get("sequence", 2))


def _adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _RecordingProvider]:
    provider = _RecordingProvider()
    inst = adapter_module.OSWorldV2Adapter.__new__(adapter_module.OSWorldV2Adapter)
    inst._provider = provider  # type: ignore[attr-defined]
    inst._observation_builder = _StubBuilder()  # type: ignore[attr-defined]
    inst._current_observation = _observation("obs-1", 1)  # type: ignore[attr-defined]
    inst._sequence = 1  # type: ignore[attr-defined]

    async def fake_sleep(seconds: float) -> None:
        provider.calls.append(("sleep", round(seconds, 3)))

    monkeypatch.setattr(adapter_module.asyncio, "sleep", fake_sleep)
    return inst, provider


def _params(*actions: Any) -> ExecuteParams:
    batch = ActionBatch(
        protocol_version=PROTOCOL_VERSION,
        task_id="task_001",
        episode_id="ep-1",
        observation_id="obs-1",
        actions=tuple(actions),
    )
    return ExecuteParams(
        operation_id="op-1",
        action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
        action_batch=batch,
    )


def _click(x: int = 10, y: int = 20) -> ClickAction:
    return ClickAction(observation_id="obs-1", frame_id="screen", x=x, y=y)


def _guest_calls(provider: _RecordingProvider) -> list[tuple[str, Any]]:
    """Drop the trailing read-back so assertions describe the mutation only."""
    return [c for c in provider.calls if c[0] != "observe"]


@pytest.mark.asyncio
async def test_a_wait_between_two_actions_runs_between_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[click, wait, type]`` must not become ``click, type, wait``.

    This is the exact shape seen most often in the failing cohort: the model
    clicks something, waits for the UI to catch up, then types into it.
    """

    inst, provider = _adapter(monkeypatch)
    await inst.execute(
        _params(
            _click(),
            WaitAction(observation_id="obs-1", duration_ms=2000),
            TypeAction(observation_id="obs-1", text="hello"),
        )
    )

    calls = _guest_calls(provider)
    kinds = [c[0] for c in calls]
    assert kinds == ["exec", "sleep", "exec"], calls
    assert calls[1] == ("sleep", 2.0)
    # The click travelled before the sleep and the text after it.
    assert any("click" in s for s in calls[0][1])
    assert any("hello" in s for s in calls[2][1])


@pytest.mark.asyncio
async def test_consecutive_actions_still_share_one_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering is preserved WITHOUT paying a round trip per action.

    Honouring order naively -- one ``execute`` call per statement -- would add
    a guest HTTP round trip for every action, and batches average ~2.9 actions.
    Consecutive statements must still travel together.
    """

    inst, provider = _adapter(monkeypatch)
    await inst.execute(
        _params(
            _click(1, 1),
            _click(2, 2),
            _click(3, 3),
            WaitAction(observation_id="obs-1", duration_ms=500),
            _click(4, 4),
        )
    )

    calls = _guest_calls(provider)
    assert [c[0] for c in calls] == ["exec", "sleep", "exec"], calls
    assert len(calls[0][1]) == 3, "three consecutive clicks must share one call"
    assert len(calls[2][1]) == 1


@pytest.mark.asyncio
async def test_a_leading_wait_precedes_the_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait the model put FIRST must happen before anything is typed."""

    inst, provider = _adapter(monkeypatch)
    await inst.execute(
        _params(
            WaitAction(observation_id="obs-1", duration_ms=1000),
            TypeAction(observation_id="obs-1", text="hi"),
        )
    )

    calls = _guest_calls(provider)
    assert [c[0] for c in calls] == ["sleep", "exec"], calls
    assert calls[0] == ("sleep", 1.0)


@pytest.mark.asyncio
async def test_a_pure_wait_batch_still_advances_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch of only waits sleeps and then pokes the guest.

    The empty ``execute`` is what advances the environment's own clock; without
    it a pure-wait batch would be invisible to the guest.
    """

    inst, provider = _adapter(monkeypatch)
    await inst.execute(_params(WaitAction(observation_id="obs-1", duration_ms=1500)))

    calls = _guest_calls(provider)
    assert calls == [("sleep", 1.5), ("exec", [])], calls


@pytest.mark.asyncio
async def test_multiple_waits_keep_their_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several waits interleaved with actions each land where they were written."""

    inst, provider = _adapter(monkeypatch)
    await inst.execute(
        _params(
            KeyAction(observation_id="obs-1", keys=("ctrl", "s")),
            WaitAction(observation_id="obs-1", duration_ms=300),
            KeyAction(observation_id="obs-1", keys=("enter",)),
            WaitAction(observation_id="obs-1", duration_ms=700),
            KeyAction(observation_id="obs-1", keys=("esc",)),
        )
    )

    calls = _guest_calls(provider)
    assert [c[0] for c in calls] == ["exec", "sleep", "exec", "sleep", "exec"], calls
    assert calls[1] == ("sleep", 0.3)
    assert calls[3] == ("sleep", 0.7)
