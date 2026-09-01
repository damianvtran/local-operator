"""Offline behavioural coverage for the provider-backed model client.

This module needs no credentials and spends no tokens: it drives the real
``ProviderModelClient`` against a scripted async stream that yields the same
event shapes ``local_operator.harness.types`` defines. That is enough to prove
the thing the runner actually depends on -- a provider reply becomes a
protocol-valid ``ActionBatch`` -- which is the property that was previously
asserted nowhere.

Its absence is what let a dead provider path ship: the module type-checked and
imported cleanly while being incapable of parsing any real reply.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import (
    ActionBatch,
    FinishAction,
    Observation,
    TypeAction,
)
from local_operator.evaluation.runner.provider_client import (
    DecisionParseError,
    ProviderModelClient,
    parse_decision,
)
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    Usage,
)

ROUTE = RouteIdentity(provider_id="provider", route_id="route", model_id="model")


def observation(sequence: int = 0) -> Observation:
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text="a screen",
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def finish_payload(current: Observation) -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "kind": "finish",
                    "observation_id": current.observation_id,
                    "status": "done",
                    "reason": "the task is complete",
                }
            ]
        }
    )


def type_payload(current: Observation) -> str:
    """A non-terminal, frameless action.

    Pointer actions are deliberately excluded here: ``validate_for`` resolves
    ``frame_id`` against the observation's frames, so covering a click would
    mean asserting frame plumbing rather than decision parsing.
    """

    return json.dumps(
        {
            "actions": [
                {
                    "kind": "type",
                    "observation_id": current.observation_id,
                    "text": "hello",
                }
            ]
        }
    )


class ScriptedStream:
    """Stands in for ``SessionStreamFn`` with a fixed event script."""

    def __init__(
        self,
        text: str,
        *,
        usage: Usage | None = None,
        stop_reason: str = "stop",
        chunk: int = 7,
    ) -> None:
        self.text = text
        self.usage = usage
        self.stop_reason = stop_reason
        self.chunk = chunk
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        # Delivered in fragments because a provider streams text in pieces; a
        # client that reads only the first delta would still pass a whole-string
        # fake and then fail against every real provider.
        for start in range(0, len(self.text), self.chunk):
            yield StreamTextDelta(delta=self.text[start : start + self.chunk])
        if self.usage is not None:
            yield StreamUsageEvent(usage=self.usage)
        yield StreamEndEvent(stop_reason=self.stop_reason, usage=self.usage)


def _client(stream: ScriptedStream) -> ProviderModelClient:
    spec = ModelSpec(provider="provider", model_id="model")
    return ProviderModelClient(stream, route=ROUTE, model_spec=spec)


# ---------------------------------------------------------------------------
# parse_decision
# ---------------------------------------------------------------------------


def test_parse_decision_accepts_a_protocol_correct_finish() -> None:
    current = observation()

    decision = parse_decision(finish_payload(current), current, route=ROUTE)

    assert isinstance(decision.action_batch, ActionBatch)
    assert decision.action_batch.protocol_version == "1.0"
    assert len(decision.action_batch.actions) == 1
    assert isinstance(decision.action_batch.actions[0], FinishAction)
    # The batch must survive the same validation the adapter boundary applies.
    decision.action_batch.validate_for(current)


def test_parse_decision_accepts_a_protocol_correct_non_terminal_action() -> None:
    current = observation()

    decision = parse_decision(type_payload(current), current, route=ROUTE)

    assert isinstance(decision.action_batch.actions[0], TypeAction)
    decision.action_batch.validate_for(current)


def test_parse_decision_binds_the_batch_to_the_current_observation() -> None:
    current = observation()

    decision = parse_decision(finish_payload(current), current, route=ROUTE)

    batch = decision.action_batch
    assert batch.task_id == current.task_id
    assert batch.episode_id == current.episode_id
    assert batch.observation_id == current.observation_id


def test_parse_decision_rejects_a_batch_bound_to_a_stale_observation() -> None:
    """A decision about a screen the environment has moved past is not usable."""

    stale = observation(0)
    current = observation(1)

    with pytest.raises(DecisionParseError):
        parse_decision(finish_payload(stale), current, route=ROUTE)


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        '{"actions": []}',
        '{"actions": "click"}',
        '{"no_actions_key": true}',
        "[]",
    ],
)
def test_parse_decision_rejects_malformed_replies(payload: str) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision(payload, observation(), route=ROUTE)


def test_parse_decision_rejects_a_wrong_discriminator_key() -> None:
    """A model emitting ``type`` instead of ``kind`` must fail loudly."""

    current = observation()
    payload = json.dumps(
        {
            "actions": [
                {
                    "type": "finish",
                    "observation_id": current.observation_id,
                    "status": "done",
                    "reason": "done",
                }
            ]
        }
    )

    with pytest.raises(DecisionParseError):
        parse_decision(payload, current, route=ROUTE)


# ---------------------------------------------------------------------------
# ProviderModelClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_turns_a_streamed_reply_into_a_valid_action_batch() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current))

    decision = await _client(stream).decide(current, [current])

    assert isinstance(decision.action_batch, ActionBatch)
    decision.action_batch.validate_for(current)
    assert decision.route == ROUTE


@pytest.mark.asyncio
async def test_client_sends_the_system_prompt_as_a_cacheable_block() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream).decide(current, [current])

    request = stream.requests[0]
    assert len(request.system_blocks) == 1
    assert request.messages[0].role == "user"
    # The episode drives the environment through the protocol, not through
    # harness tools, so the provider must not be offered any.
    assert request.tool_choice == "none"


@pytest.mark.asyncio
async def test_system_prompt_states_the_schema_the_protocol_enforces() -> None:
    """An under-specified prompt is a correctness defect: a parse failure is terminal."""

    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream).decide(current, [current])

    prompt = stream.requests[0].system_blocks[0]
    # The discriminator key, and every finish literal the protocol accepts.
    assert '"kind"' in prompt
    for status in ("done", "failed", "infeasible"):
        assert status in prompt
    assert "reason" in prompt
    # Derived from the protocol models, so a new action kind cannot drift out.
    for kind in ("click", "type", "key", "scroll", "wait", "finish", "ask_user"):
        assert kind in prompt


@pytest.mark.asyncio
async def test_client_carries_provider_usage_and_cost_into_the_decision() -> None:
    """Cost is authoritative provider data; dropping it reports every run as free."""

    current = observation()
    usage = Usage(
        input_tokens=1234,
        output_tokens=56,
        reasoning_tokens=7,
        cache_read_tokens=8,
        cache_write_tokens=9,
        usd_cost=0.0421,
    )
    stream = ScriptedStream(finish_payload(current), usage=usage)

    decision = await _client(stream).decide(current, [current])

    assert decision.usage.input_tokens == 1234
    assert decision.usage.output_tokens == 56
    assert decision.usage.reasoning_tokens == 7
    assert decision.usage.cache_read_tokens == 8
    assert decision.usage.cache_write_tokens == 9
    assert decision.cost_micros == 42100


@pytest.mark.asyncio
async def test_unreported_cost_is_zero_rather_than_a_guess() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current), usage=Usage(input_tokens=5, output_tokens=5))

    decision = await _client(stream).decide(current, [current])

    assert decision.cost_micros == 0


@pytest.mark.asyncio
async def test_client_reports_the_provider_stop_reason() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current), stop_reason="length")

    decision = await _client(stream).decide(current, [current])

    assert decision.stop_reason == "length"


@pytest.mark.asyncio
async def test_client_raises_on_an_unparseable_reply() -> None:
    """The runner converts this to a provider failure, so it must surface."""

    current = observation()
    stream = ScriptedStream("I'm afraid I can't do that.")

    with pytest.raises(DecisionParseError):
        await _client(stream).decide(current, [current])
