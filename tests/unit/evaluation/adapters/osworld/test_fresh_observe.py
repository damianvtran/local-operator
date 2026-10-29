"""Live read-only recovery must not masquerade as a cached action result.

The production worker/wheel path is covered in test_spawn. These tests isolate
capture/build failures (including cancellation) and the parent's fail-closed
validation, without the cost of another worker install for every negative.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter.observation import (
    ObservationBuilder,
    ObservationError,
    write_png_rgb,
)

from local_operator.evaluation.adapters.api import (
    ObservationResult,
    ObserveParams,
    observation_content_id,
)
from local_operator.evaluation.adapters.supervisor import (
    HostVerifier,
    VerifiedAdapterSession,
)
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
)
from tests.unit.evaluation.adapters.osworld.test_not_applied_declaration import (
    _adapter,
    _ScriptedProvider,
)
from tests.unit.evaluation.adapters.test_supervisor import RawSupervisor, observation


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["capture", "cancel", "build"])
async def test_failed_live_capture_preserves_both_adapter_state_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    provider = _ScriptedProvider([])
    adapter = _adapter(monkeypatch, provider)
    initial = adapter._current_observation
    capture_calls: list[int] = []

    async def capture() -> dict[str, Any]:
        capture_calls.append(1)
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "capture":
            raise RuntimeError("capture unavailable")
        return {"screenshot": None}  # The real builder must reject frameless output.

    monkeypatch.setattr(provider, "observe", capture)
    adapter._observation_builder = ObservationBuilder(tmp_path)
    expected = {
        "capture": RuntimeError,
        "cancel": asyncio.CancelledError,
        "build": ObservationError,
    }[failure]
    with pytest.raises(expected):
        await adapter.observe(ObserveParams(episode_id="ep-1", fresh=True))
    assert capture_calls == [1]
    assert adapter._sequence == 1 and adapter._current_observation is initial
    assert provider.calls == []


@pytest.mark.asyncio
async def test_live_capture_is_new_content_and_binding_without_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _ScriptedProvider([])
    adapter = _adapter(monkeypatch, provider)
    initial = adapter._current_observation
    capture_calls: list[int] = []

    async def capture() -> dict[str, Any]:
        capture_calls.append(1)
        return {"screenshot": write_png_rgb(1920, 1080, b"\xff\x00\x00" * (1920 * 1080))}

    monkeypatch.setattr(provider, "observe", capture)
    adapter._observation_builder = ObservationBuilder(tmp_path)
    cached = await adapter.observe(ObserveParams(episode_id="ep-1"))
    assert cached.observation is initial and capture_calls == []
    live = await adapter.observe(ObserveParams(episode_id="ep-1", fresh=True))
    assert live.observation.sequence == 2
    assert live.observation.observation_id != initial.observation_id
    assert live.observation.observation_id == observation_content_id(live.observation)
    assert capture_calls == [1] and provider.calls == []
    assert adapter._current_observation is live.observation and adapter._sequence == 2
    assert (await adapter.observe(ObserveParams(episode_id="ep-1"))).observation == live.observation
    assert capture_calls == [1]  # Default observe still does not capture.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["cache", "skip", "task", "episode", "identity", "missing-artifact", "capture-error"]
)
async def test_fresh_parent_validation_is_fail_closed(tmp_path: Path, mode: str) -> None:
    initial = observation("task", "episode", 0)
    output = observation("task", "episode", 1)
    if mode == "cache":
        output = initial  # Includes an adapter that silently ignores fresh=True.
    elif mode == "skip":
        output = observation("task", "episode", 2)
    elif mode == "task":
        output = observation("foreign", "episode", 1)
    elif mode == "episode":
        output = observation("task", "foreign", 1)
    elif mode == "identity":
        output = output.model_copy(update={"observation_id": "forged"})
    elif mode == "missing-artifact":
        size = FrameSize(width=2, height=2)
        output = output.model_copy(
            update={
                "frames": (
                    FrameRef(
                        frame_id="screen",
                        artifact=ArtifactRef(
                            sha256="a" * 64, byte_count=12, media_type="image/png"
                        ),
                        geometry=FrameGeometry(native=size, model_visible=size),
                    ),
                )
            }
        )
        output = output.model_copy(update={"observation_id": observation_content_id(output)})
    response = (
        RuntimeError("capture failed")
        if mode == "capture-error"
        else ObservationResult(observation=output)
    )
    raw = RawSupervisor([response])
    verifier = HostVerifier("task", "episode", tmp_path)
    verifier.accept_initial(initial)
    session = VerifiedAdapterSession(raw, verifier)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        await session.observe(ObserveParams(episode_id="episode", fresh=True), timeout=1)
    assert verifier.current_observation is initial
    assert raw.calls == ["observe"], "the failed response must reach the actual validation path"
    assert raw.terminated
    with pytest.raises(Exception, match="poisoned"):
        session._ensure_usable()


@pytest.mark.asyncio
async def test_fresh_parent_accepts_exact_next_and_cached_default_stays_equal(
    tmp_path: Path,
) -> None:
    initial = observation("task", "episode", 0)
    live = observation("task", "episode", 1, text="autonomously changed screen")
    raw = RawSupervisor([ObservationResult(observation=live), ObservationResult(observation=live)])
    verifier = HostVerifier("task", "episode", tmp_path)
    verifier.accept_initial(initial)
    session = VerifiedAdapterSession(raw, verifier)  # type: ignore[arg-type]
    assert (
        await session.observe(ObserveParams(episode_id="episode", fresh=True), timeout=1)
    ).observation == live
    assert verifier.current_observation == live
    assert (
        await session.observe(ObserveParams(episode_id="episode"), timeout=1)
    ).observation == live
    assert not raw.terminated
