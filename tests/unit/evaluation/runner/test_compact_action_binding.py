"""Evaluation-only compact observation binding contract.

The compact schema saves one repeated content ID per action while retaining the
legacy ActionBatch and adapter protocol as the canonical internal boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import cast

import pytest

from local_operator.evaluation.action_surface import LEGACY_ACTION_SURFACE
from local_operator.evaluation.evidence.models import (
    ModelResponsePayload,
    RouteIdentity,
    canonical_digest,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.protocol import ActionBatch
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner
from local_operator.evaluation.runner.provider_client import (
    DecisionParseError,
    build_system_prompt,
    parse_decision,
)
from local_operator.evaluation.runner.public_reply import (
    COMPACT_ACTION_BINDING,
    LEGACY_ACTION_BINDING,
    public_reply_contract,
    public_reply_schema,
)
from scripts import run_episode
from tests.unit.evaluation.runner.conftest import FakeAdapter, build_config, build_spec
from tests.unit.evaluation.runner.conftest import observation as runner_observation
from tests.unit.evaluation.runner.conftest import payloads, selector
from tests.unit.evaluation.runner.test_episode import _rescue_ok
from tests.unit.evaluation.runner.test_provider_client import (
    ScriptedStream,
    _client,
    _turns,
    observation,
    type_payload,
)

ROUTE = RouteIdentity(provider_id="provider", route_id="route", model_id="model")


def _compact(current, *, top=None, action_id=None) -> str:
    action = {"kind": "type", "text": "hello"}
    if action_id is not None:
        action["observation_id"] = action_id
    value = {
        "observation_id": current.observation_id if top is None else top,
        "actions": [action],
    }
    return json.dumps(value)


def test_compact_schema_and_prompt_omit_repeated_action_ids() -> None:
    legacy = public_reply_schema()
    compact = public_reply_schema(action_binding=COMPACT_ACTION_BINDING)
    assert '"observation_id"' in json.dumps(legacy["properties"]["actions"])
    assert "observation_id" not in json.dumps(compact["properties"]["actions"])
    assert compact["required"] == ["observation_id", "actions"]
    first_action_binding = LEGACY_ACTION_SURFACE.models[0].model_json_schema()["properties"][
        "observation_id"
    ]
    assert compact["properties"]["observation_id"] == first_action_binding
    assert compact["properties"]["observation_id"]["pattern"] == r"\S"
    assert public_reply_schema() == legacy
    assert build_system_prompt() == build_system_prompt(action_binding=LEGACY_ACTION_BINDING)
    # The legacy prompt is byte-frozen on purpose, and the digest moved once:
    # the "finish" bullet gained the sentence that states the completion gate's
    # contract (the reason the model is told to check the newest observation
    # against the task before declaring done). Re-pin deliberately, never by
    # pasting whatever the run printed.
    assert hashlib.sha256(build_system_prompt().encode()).hexdigest() == (
        "6c94d6494b605e791b77af73d0bbb1cbf411773d730670f2f255f3cf19fa2b29"
    )
    assert '"observation_id": "<current observation id>"' in build_system_prompt(
        action_binding=COMPACT_ACTION_BINDING
    )
    assert '"observation_id": "<id>"' not in build_system_prompt(
        action_binding=COMPACT_ACTION_BINDING
    )


def test_compact_reply_materializes_the_same_canonical_batch() -> None:
    current = observation()
    compact = parse_decision(
        _compact(current), current, route=ROUTE, action_binding=COMPACT_ACTION_BINDING
    )
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    assert isinstance(compact.action_batch, ActionBatch)
    assert compact.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    compact.action_batch.validate_for(current)


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda current: _compact(current, top="stale-observation"),
        lambda current: _compact(current, top=""),
        lambda current: _compact(current, action_id="stale-action-observation"),
        lambda current: json.dumps({"actions": [{"kind": "type", "text": "hello"}]}),
        lambda current: json.dumps(
            {
                "observation_id": current.observation_id,
                "action_batch": {
                    "observation_id": "stale-batch-observation",
                    "actions": [{"kind": "type", "text": "hello"}],
                },
            }
        ),
    ],
    ids=["stale-top", "empty-top", "stale-legacy-action", "missing-top", "stale-batch"],
)
def test_compact_mismatched_or_missing_binding_is_rejected(payload_factory) -> None:
    current = observation()
    with pytest.raises(DecisionParseError):
        parse_decision(
            payload_factory(current),
            current,
            route=ROUTE,
            action_binding=COMPACT_ACTION_BINDING,
        )


def test_compact_binding_is_checked_against_episode_and_task() -> None:
    current = observation()
    changed = current.model_copy(
        update={"episode_id": "other-episode", "observation_id": "other-observation"}
    )
    with pytest.raises(DecisionParseError):
        parse_decision(
            _compact(current), changed, route=ROUTE, action_binding=COMPACT_ACTION_BINDING
        )


def test_legacy_binding_remains_required_and_current() -> None:
    current = observation()
    parse_decision(type_payload(current), current, route=ROUTE)
    stale = json.loads(type_payload(current))
    stale["actions"][0]["observation_id"] = "stale"
    with pytest.raises(DecisionParseError):
        parse_decision(json.dumps(stale), current, route=ROUTE)


def test_reply_contract_identity_is_stable_and_mode_specific() -> None:
    legacy = public_reply_contract()
    compact = public_reply_contract(action_binding=COMPACT_ACTION_BINDING)
    assert legacy == public_reply_contract(action_binding=LEGACY_ACTION_BINDING)
    assert compact == public_reply_contract(action_binding=COMPACT_ACTION_BINDING)
    assert legacy["model_reply_contract_digest"] != compact["model_reply_contract_digest"]
    assert legacy["model_reply_contract_digest"] == canonical_digest(
        "runner-model-reply-v1", json.loads(legacy["model_reply_contract"])
    )
    assert compact["model_reply_contract_digest"] == canonical_digest(
        "runner-model-reply-v1", json.loads(compact["model_reply_contract"])
    )


@pytest.mark.asyncio
async def test_provider_client_sends_selected_schema_prompt_and_contract(tmp_path) -> None:
    current = observation()
    stream = ScriptedStream(_compact(current))
    client = _client(stream, tmp_path, action_binding=COMPACT_ACTION_BINDING)

    await client.decide(current, _turns(current))

    request = stream.requests[0]
    request_schema = request.tools[0].parameters
    assert request_schema == public_reply_schema(
        LEGACY_ACTION_SURFACE, action_binding=COMPACT_ACTION_BINDING
    )
    assert '"observation_id": "<current observation id>"' in request.system_blocks[0]
    assert '"observation_id": "<id>"' not in request.system_blocks[0]
    assert client.model_reply_metadata == public_reply_contract(
        action_binding=COMPACT_ACTION_BINDING
    )


@pytest.mark.asyncio
async def test_compact_contract_is_sealed_in_verified_runner_bundle(
    tmp_path, episode_id: str
) -> None:
    current = runner_observation(episode_id, 0)
    compact_reply = json.dumps(
        {
            "observation_id": current.observation_id,
            "actions": [
                {
                    "kind": "finish",
                    "status": "done",
                    "reason": "task complete",
                }
            ],
        }
    )
    stream = ScriptedStream(compact_reply)
    client = _client(stream, tmp_path / "artifacts", action_binding=COMPACT_ACTION_BINDING)
    run_root = tmp_path / "runner"
    # Control arm: this test is about which reply CONTRACT is sealed, and the
    # completion gate's own extra cycle is covered by
    # ``tests/unit/evaluation/runner/test_completion_gate.py``.
    config = build_config(run_root, completion_gate=False)
    episode_spec = build_spec(episode_id)
    adapter_selector = selector(tmp_path)

    outcome = await EpisodeRunner(
        episode_spec,
        config,
        selector=adapter_selector,
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values(()),
    ).run()

    assert outcome.status == "completed", outcome
    assert outcome.bundle_root is not None
    report = verify_bundle(outcome.bundle_root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.manifest is not None
    expected = public_reply_contract(action_binding=COMPACT_ACTION_BINDING)
    assert report.manifest.metadata["model_reply_contract"] == expected["model_reply_contract"]
    assert (
        report.manifest.metadata["model_reply_contract_digest"]
        == expected["model_reply_contract_digest"]
    )
    contract = json.loads(cast(str, report.manifest.metadata["model_reply_contract"]))
    assert contract["action_binding"] == COMPACT_ACTION_BINDING
    responses = payloads(outcome.bundle_root, ModelResponsePayload)
    assert len(responses) == 1


def test_run_episode_exposes_and_forwards_action_binding(monkeypatch, tmp_path) -> None:
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "local_operator.evaluation.runner.provider_client.create_provider_model_client",
        capture,
    )
    base = {
        "auth_store": object(),
        "settings": {},
        "provider": "provider",
        "model": "model",
        "route": ROUTE,
        "artifact_root": tmp_path,
        "episode_id": "episode-1",
        "task_id": "task-1",
        "keep_recent_frames": 3,
    }

    run_episode._model_client(**base)
    assert captured["action_binding"] == LEGACY_ACTION_BINDING
    run_episode._model_client(**base, action_binding=COMPACT_ACTION_BINDING)
    assert captured["action_binding"] == COMPACT_ACTION_BINDING

    defaults = run_episode.build_parser().parse_args(
        ["--selector", "s.json", "--task-id", "t", "--route", "test/model", "--run-root", "r"]
    )
    assert defaults.action_binding == LEGACY_ACTION_BINDING
    compact = run_episode.build_parser().parse_args(
        [
            "--selector",
            "s.json",
            "--task-id",
            "t",
            "--route",
            "test/model",
            "--run-root",
            "r",
            "--action-binding",
            COMPACT_ACTION_BINDING,
        ]
    )
    assert compact.action_binding == COMPACT_ACTION_BINDING


@pytest.mark.asyncio
async def test_provider_client_legacy_schema_prompt_and_contract_stay_default(tmp_path) -> None:
    current = observation()
    stream = ScriptedStream(type_payload(current))
    client = _client(stream, tmp_path)

    await client.decide(current, _turns(current))

    request = stream.requests[0]
    assert request.tools[0].parameters == public_reply_schema(LEGACY_ACTION_SURFACE)
    assert client.model_reply_metadata == public_reply_contract()
