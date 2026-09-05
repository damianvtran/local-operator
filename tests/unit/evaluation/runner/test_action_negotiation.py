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
from local_operator.evaluation.runner.model import DecisionRejected
from tests.unit.evaluation.runner.conftest import FakeAdapter, ScriptedModel, payloads
from tests.unit.evaluation.runner.test_episode import _runner
from tests.unit.evaluation.runner.test_provider_client import (
    ScriptedStream,
    _client,
    _turns,
    observation,
)


def _payload(kind: str) -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "kind": kind,
                    "observation_id": observation().observation_id,
                    "text": "café 東京🙂",
                    **(
                        {"keys": ["ctrl", "v"], "clipboard_policy": "overwrite"}
                        if kind == "paste_text"
                        else {}
                    ),
                }
            ]
        }
    )


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
