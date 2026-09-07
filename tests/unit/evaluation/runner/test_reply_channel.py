"""The decision envelope offered as the model's own tool channel.

A model whose tool channel is the strongest thing in its post-training answers
on that channel under pressure, emitting native call syntax as TEXT that the
strict prose decoder throws away — a full paid round trip for a reply whose
intent was right and whose channel was wrong (43.9% of one measured cohort's
replies against ~10% for two others).

These tests hold the four properties that make the fix a fix rather than a
loosening: the channel is offered on capability alone, both channels converge
on ONE validated envelope, a malformed reply is still rejected identically on
either channel, and the schema that rides in the prompt-cache prefix is stable
across turns.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from local_operator.evaluation.protocol import ActionBatch
from local_operator.evaluation.runner.model import DecisionRejected
from local_operator.evaluation.runner.public_reply import (
    public_reply_contract,
    public_reply_schema,
)
from local_operator.harness.reply_channel import (
    REPLY_CHANNEL_TOOL_NAME,
    build_reply_channel_tool,
    envelope_from_tool_call,
    reply_channel_tools,
)
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from tests.unit.evaluation.runner.test_provider_client import (
    _client,
    _turns,
    finish_payload,
    observation,
)
from tests.unit.evaluation.runner.test_public_reply import envelope


class ChannelStream:
    """A provider that answers by CALLING the reply channel instead of writing prose.

    Arguments arrive in fragments, as a real provider streams them: a client
    that read only the first delta would pass a whole-string fake and then fail
    against every live provider.
    """

    def __init__(
        self,
        arguments: str,
        *,
        name: str = REPLY_CHANNEL_TOOL_NAME,
        text: str = "",
        chunk: int = 5,
        stop_reason: str = "toolUse",
    ) -> None:
        self.arguments = arguments
        self.name = name
        self.text = text
        self.chunk = chunk
        self.stop_reason = stop_reason
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        if self.text:
            yield StreamTextDelta(delta=self.text)
        yield StreamToolCallDelta(index=0, id="call-1", name=self.name)
        for start in range(0, len(self.arguments), self.chunk):
            yield StreamToolCallDelta(
                index=0, argument_delta=self.arguments[start : start + self.chunk]
            )
        yield StreamEndEvent(stop_reason=self.stop_reason)


def _spec(*, supports_tools: bool) -> ModelSpec:
    return ModelSpec(provider="provider", model_id="model", supports_tools=supports_tools)


# ---------------------------------------------------------------------------
# The channel is offered on capability alone
# ---------------------------------------------------------------------------


def test_reply_channel_is_offered_only_when_the_spec_supports_tools() -> None:
    """Capability is the ONLY question asked — never the model or provider name."""

    schema = public_reply_schema()
    offered = reply_channel_tools(_spec(supports_tools=True), schema, description="d")
    withheld = reply_channel_tools(_spec(supports_tools=False), schema, description="d")

    assert [tool.name for tool in offered] == [REPLY_CHANNEL_TOOL_NAME]
    assert withheld == []


@pytest.mark.asyncio
async def test_a_tools_capable_model_is_offered_the_channel() -> None:
    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream, model_spec=_spec(supports_tools=True)).decide(current, _turns(current))

    request = stream.requests[0]
    assert [tool.name for tool in request.tools] == [REPLY_CHANNEL_TOOL_NAME]
    # Offered, never forced: the prose path still works for the cohorts that
    # already use it successfully, and a forced call is a worse failure when a
    # model has genuinely nothing to say.
    assert request.tool_choice == "auto"


@pytest.mark.asyncio
async def test_a_model_without_tool_support_gets_the_request_it_always_got() -> None:
    """No tools and no wire ``tool_choice`` key — offering a function to a model
    that cannot take one is at best ignored and at worst a 400 on the whole
    request."""

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream, model_spec=_spec(supports_tools=False)).decide(current, _turns(current))

    request = stream.requests[0]
    assert request.tools == []
    assert request.tool_choice == "none"


def test_the_channel_is_a_reply_not_a_capability() -> None:
    """Nothing may execute it: a dispatched reply channel would run the model's
    answer as if it were an action, so it fails loudly rather than benignly."""

    tool = build_reply_channel_tool(public_reply_schema(), description="d")

    assert tool.hidden is True
    assert tool.approval_tier == "read"
    with pytest.raises(RuntimeError, match="never be executed"):
        import asyncio

        asyncio.run(tool.execute(tool.name, {}, None, None, None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Both channels converge on one validated envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_channels_produce_the_identical_validated_envelope() -> None:
    """The whole point of the fix: same bytes in, same decision out."""

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")

    prose = await _client(ScriptedStream(body), model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )
    channel = await _client(ChannelStream(body), model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )

    assert isinstance(channel.action_batch, ActionBatch)
    assert channel.action_batch.to_canonical_json() == prose.action_batch.to_canonical_json()
    assert channel.public_reply == prose.public_reply
    channel.action_batch.validate_for(current)


@pytest.mark.asyncio
async def test_a_legacy_bare_batch_is_accepted_on_the_channel_too() -> None:
    """The channel carries whatever the prose path carries, envelope or not."""

    current = observation()

    decision = await _client(
        ChannelStream(finish_payload(current)), model_spec=_spec(supports_tools=True)
    ).decide(current, _turns(current))

    assert decision.public_reply is None
    decision.action_batch.validate_for(current)


def test_the_channel_hands_back_raw_bytes_rather_than_a_repaired_object() -> None:
    """Re-serializing would quietly REPAIR duplicate keys the strict decoder
    exists to reject, turning a transparent channel into a lenient one."""

    duplicated = '{"reply_version": "1.0", "reply_version": "1.0"}'
    call = ToolCall(name=REPLY_CHANNEL_TOOL_NAME, raw_arguments=duplicated)

    assert envelope_from_tool_call([call], name=REPLY_CHANNEL_TOOL_NAME) == duplicated


def test_an_unused_channel_is_distinguishable_from_an_empty_one() -> None:
    """``None`` means read the prose; ``""`` means the model used the channel
    and sent nothing usable, which is a rejection to report."""

    unused = envelope_from_tool_call(
        [ToolCall(name="something_else", raw_arguments="{}")], name=REPLY_CHANNEL_TOOL_NAME
    )
    empty = envelope_from_tool_call(
        [ToolCall(name=REPLY_CHANNEL_TOOL_NAME, raw_arguments=None)],
        name=REPLY_CHANNEL_TOOL_NAME,
    )

    assert unused is None
    assert empty == ""


# ---------------------------------------------------------------------------
# Rejection is unchanged — no new salvage path, no loosened validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "not json at all",
        '{"reply_version": "9.9", "action_batch": {"actions": []}, "public_observations": ""}',
        '{"actions": []}',
        "",
    ],
)
async def test_a_malformed_reply_is_rejected_on_the_channel_exactly_as_in_prose(
    body: str,
) -> None:
    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()

    with pytest.raises(DecisionRejected) as prose:
        await _client(ScriptedStream(body), model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )
    with pytest.raises(DecisionRejected) as channel:
        await _client(ChannelStream(body), model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert str(channel.value) == str(prose.value)


@pytest.mark.asyncio
async def test_native_tool_syntax_in_prose_is_still_rejected() -> None:
    """The marker sequence that motivated this work is NOT salvaged. Offering
    the real channel is the fix; sniffing for a vendor's syntax would be a
    per-model rule that ages badly and silently widens what is accepted."""

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    body = '<|open|>tools<|sep|><|open|>call tool="terminal"{"cmd": "ls"}'

    with pytest.raises(DecisionRejected):
        await _client(ScriptedStream(body), model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )


@pytest.mark.asyncio
async def test_a_call_to_a_name_we_never_offered_is_not_read_as_a_reply() -> None:
    """Only THE channel is a reply. Any other call is counted for the evidence
    bundle and otherwise ignored, so the prose path still decides."""

    current = observation()
    stream = ChannelStream(finish_payload(current), name="terminal", text="")

    with pytest.raises(DecisionRejected):
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )


# ---------------------------------------------------------------------------
# Prompt-cache stability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_prefix_stays_cache_stable_across_turns() -> None:
    """The tools array is the FRONT of the provider cache prefix. A schema that
    varied per turn would re-write the whole prefix on every call and cost more
    than the rejections it saves."""

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    first = observation(0)
    second = observation(1)
    stream = ScriptedStream(finish_payload(first))
    client = _client(stream, model_spec=_spec(supports_tools=True))

    await client.decide(first, _turns(first))
    stream.text = finish_payload(second)
    await client.decide(second, _turns(first, second))

    tools = [request.tools for request in stream.requests]
    assert len(tools) == 2
    assert _serialized(tools[0]) == _serialized(tools[1])
    # System block too: the channel must not have made the prompt turn-dependent.
    assert stream.requests[0].system_blocks == stream.requests[1].system_blocks


def _serialized(tools: list[Any]) -> str:
    return json.dumps(
        [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools],
        sort_keys=True,
    )


def test_the_channel_schema_is_the_published_contract_schema() -> None:
    """One definition, two readers. If these ever diverge, the tool a model may
    call and the envelope the decoder validates are different shapes."""

    published = json.loads(public_reply_contract()["model_reply_contract"])["schema"]
    tool = build_reply_channel_tool(public_reply_schema(), description="d")

    assert tool.parameters == published
