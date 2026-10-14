"""Exercise the existing provider/runner paths with negotiated capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.evidence.models import (
    ModelRequestPayload,
    canonical_digest,
)
from local_operator.evaluation.protocol import Observation
from local_operator.evaluation.runner.model import DecisionRejected
from tests.unit.evaluation.runner.conftest import FakeAdapter, ScriptedModel, payloads
from tests.unit.evaluation.runner.test_episode import _runner
from tests.unit.evaluation.runner.test_provider_client import (
    ScriptedStream,
    _client,
    _turns,
    observation,
)


def _payload(kind: str, observation_id: str | None = None) -> str:
    action: dict[str, Any] = {
        "kind": kind,
        "observation_id": observation_id or observation().observation_id,
    }
    if kind == "paste_text":
        action.update(text="café 東京🙂", keys=["ctrl", "v"], clipboard_policy="overwrite")
    elif kind == "drag":
        action.update(frame_id="screen", start_x=1, start_y=2, end_x=3, end_y=4)
    else:
        action.update(text="café 東京🙂")
    return json.dumps({"actions": [action]})


def _drag_observation(root: Path) -> Observation:
    """A framed observation with real geometry so drag coordinates validate."""
    import hashlib

    from local_operator.evaluation.adapters.api import observation_content_id
    from local_operator.evaluation.protocol import (
        ArtifactRef,
        FrameGeometry,
        FrameRef,
        FrameSize,
    )
    from tests.unit.evaluation.runner.test_provider_client import _png

    data = _png()
    digest = hashlib.sha256(data).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    (root / digest).write_bytes(data)
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=0,
        observation_id="provisional",
        text="a screen",
        frames=(
            FrameRef(
                frame_id="screen",
                artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
                geometry=FrameGeometry(
                    native=FrameSize(width=1280, height=720),
                    model_visible=FrameSize(width=1280, height=720),
                ),
            ),
        ),
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


@pytest.mark.asyncio
async def test_provider_sees_supported_unicode_path_and_rejects_lossy_type() -> None:
    surface = ActionSurface(paste_text=True, type_text_mode="ascii")
    stream = ScriptedStream(_payload("type"))
    client = _client(stream)
    with pytest.raises(DecisionRejected, match="use paste_text"):
        await client.decide(observation(), _turns(observation()), action_surface=surface)
    assert '"paste_text"' in stream.requests[0].system_blocks[0]
    stream.text = _payload("paste_text")
    decision = await client.decide(observation(), _turns(observation()), action_surface=surface)
    assert decision.action_batch.actions[0].kind == "paste_text"
    assert len(stream.requests) == 2


@pytest.mark.asyncio
async def test_unsupported_paste_is_a_correctable_decision_not_dispatch() -> None:
    stream = ScriptedStream(_payload("paste_text"))
    client = _client(stream)
    with pytest.raises(DecisionRejected, match="not supported"):
        await client.decide(observation(), _turns(observation()), action_surface=ActionSurface())
    assert '"paste_text"' not in stream.requests[0].system_blocks[0]


@pytest.mark.asyncio
async def test_drag_is_offered_and_dispatched_only_with_the_capability(tmp_path: Path) -> None:
    current = _drag_observation(tmp_path)
    offered = ActionSurface(drag=True)
    stream = ScriptedStream(_payload("drag", current.observation_id))
    client = _client(stream, tmp_path)
    decision = await client.decide(current, _turns(current), action_surface=offered)
    assert decision.action_batch.actions[0].kind == "drag"
    assert '"drag"' in stream.requests[0].system_blocks[0]


@pytest.mark.asyncio
async def test_unsupported_drag_is_a_correctable_decision_not_dispatch(tmp_path: Path) -> None:
    current = _drag_observation(tmp_path)
    stream = ScriptedStream(_payload("drag", current.observation_id))
    client = _client(stream, tmp_path)
    with pytest.raises(DecisionRejected, match="not supported"):
        await client.decide(current, _turns(current), action_surface=ActionSurface())
    assert '"drag"' not in stream.requests[0].system_blocks[0]


@pytest.mark.asyncio
async def test_runner_records_the_same_negotiated_schema_it_passes_to_model(
    tmp_path: Path, episode_id: str
) -> None:
    seen: list[ActionSurface] = []

    class CapturingModel(ScriptedModel):
        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["action_surface"])
            return await super().decide(observation, history, **kwargs)

    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=FakeAdapter(tmp_path, episode_id),
        model=CapturingModel(["finish"]),
    ).run()
    assert outcome.status == "completed"
    assert outcome.bundle_root is not None
    manifest = json.loads((outcome.bundle_root / "manifest.json").read_text())
    recorded = json.loads(manifest["metadata"]["action_surface"])
    assert recorded == seen[0].schema()
    requests = payloads(outcome.bundle_root, ModelRequestPayload)
    assert requests[0].tool_schema_digest == canonical_digest("runner-tool-schema-v1", recorded)
    assert requests[0].tool_schema_digest != canonical_digest(
        "runner-tool-schema-v1", {"episode_id": episode_id}
    )
