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
from collections.abc import Sequence
from typing import Any, AsyncIterator

import pytest

from local_operator.evaluation.protocol import ActionBatch
from local_operator.evaluation.runner.model import DecisionRejected
from local_operator.evaluation.runner.provider_client import parse_decision
from local_operator.evaluation.runner.public_reply import (
    decode_public_reply,
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
from tests.unit.evaluation.runner.test_public_reply import _wrapped, envelope


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
        calls: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.arguments = arguments
        self.name = name
        self.text = text
        self.chunk = chunk
        self.stop_reason = stop_reason
        # More than one call in one stream, as ``(name, arguments)`` pairs. A
        # second call is how the ambiguity refusals are exercised (two batches in
        # one reply), and it is deliberately a SEPARATE field rather than a
        # `name` list: the single-call shape stays the one every other test in
        # this file uses, unchanged.
        self.calls = calls
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        if self.text:
            yield StreamTextDelta(delta=self.text)
        if self.calls is None:
            yield StreamToolCallDelta(index=0, id="call-1", name=self.name)
            for start in range(0, len(self.arguments), self.chunk):
                yield StreamToolCallDelta(
                    index=0, argument_delta=self.arguments[start : start + self.chunk]
                )
        else:
            for index, (name, arguments) in enumerate(self.calls):
                yield StreamToolCallDelta(index=index, id=f"call-{index}", name=name)
                for start in range(0, len(arguments), self.chunk):
                    yield StreamToolCallDelta(
                        index=index, argument_delta=arguments[start : start + self.chunk]
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
@pytest.mark.parametrize(
    "variant", ["tool_name-parameters", "tool_call-input-string", "input-object"]
)
async def test_a_wrapped_call_on_the_channel_decodes_like_the_unwrapped_one(
    variant: str,
) -> None:
    """The channel MOVED the refusals, so it has to carry the wrapped ones too.

    65 of the canary arm's 104 refusals arrived on the tool channel, and the
    class it recorded as ``batch-shape`` is exactly this: a complete envelope
    serialized as one generic tool call, handed to a decoder that wanted the
    envelope itself. The channel supplies the bytes and the same decoder judges
    them, so a wrapped call must reach the batch the unwrapped call reaches.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")

    plain = await _client(ChannelStream(body), model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )
    wrapped = await _client(
        ChannelStream(_wrapped(variant, body)), model_spec=_spec(supports_tools=True)
    ).decide(current, _turns(current))

    assert wrapped.action_batch.to_canonical_json() == plain.action_batch.to_canonical_json()
    assert decode_public_reply(wrapped.public_reply or "")["public_observations"] == (
        "Visible status: ready"
    )
    wrapped.action_batch.validate_for(current)


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
        # NOTE: the empty body is deliberately absent. It does not encode the
        # same model behaviour on both paths, so it cannot test parity between
        # them: in prose it means the model emitted NOTHING (no tool call, no
        # text) and is now reported as the silence it is, while on the channel
        # it means the model DID select the channel and sent empty arguments --
        # a malformed call, not an absent reply. Requiring one diagnostic to
        # cover both would force the silent case back to reporting a JSON
        # complaint, which is the misdiagnosis that spends an episode's retry
        # bound re-prompting a model to fix JSON it never wrote. The silent case
        # is covered directly by ``test_a_silent_reply_is_a_correctable_rejection``
        # and the channel case by
        # ``test_an_empty_channel_call_is_rejected_like_an_empty_prose_body``.
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
async def test_a_call_named_anything_is_the_reply_when_it_is_the_only_tool_offered() -> None:
    """The sole offered name IS the reply channel, so a call under any other name
    is still the model answering -- and 178 of the arm's 204 ``leading-delimiter``
    refusal artifacts were exactly this (recounted 2026-09-25 over
    ``~/worktrees/osworld/runs``; see ``harness/reply_channel``), refused with no
    reply text published at all.

    The widening rests on what the REQUEST advertised, which is why the reason
    the empty prose was refused before is now the reason the call is read: the
    request put one function on the wire and that function's parameters ARE the
    decision envelope, so there is nothing else a call could be.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    stream = ChannelStream(body, name="submit_action_batch", text="")

    decision = await _client(stream, model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )

    # The decoded decision IS the reply's own batch, read from the same bytes by
    # the one canonical decoder -- not a reconstruction and not a second
    # parser's opinion of them.
    from tests.unit.evaluation.runner.test_provider_client import ROUTE

    direct = parse_decision(body, current, route=ROUTE)
    assert isinstance(decision.action_batch, ActionBatch)
    assert decision.action_batch.to_canonical_json() == direct.action_batch.to_canonical_json()
    assert decision.public_reply == body
    decision.action_batch.validate_for(current)
    # And the same bytes under the name we offered reach the same decision, so
    # the name changes nothing about what is executed.
    offered = await _client(ChannelStream(body), model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )
    assert decision.action_batch.to_canonical_json() == offered.action_batch.to_canonical_json()


@pytest.mark.asyncio
async def test_a_foreign_named_call_is_not_a_reply_when_other_tools_were_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widening is gated on the OFFER, not on a name blacklist.

    A request that put a real capability on the wire is one where a call under a
    name we did not choose may be that capability being invoked, so the strict
    name test stays. This is the arm's own defect in reverse: the reason 178
    refusals were misread is that nobody recorded what had been offered, so this
    pins the reading to the offer rather than to a guess about intent.
    """

    from local_operator.evaluation.runner import provider_client as client_module
    from local_operator.harness.types import AgentTool

    current = observation()
    body = envelope(finish_payload(current), "")

    def with_a_second_tool(spec: Any, schema: Any, *, description: str) -> list[AgentTool]:
        # A second tool built from the SAME builder, so the test cannot pass by
        # handing the client a shape it would not accept from a real caller.
        return [
            *reply_channel_tools(spec, schema, description=description),
            build_reply_channel_tool(schema, description=description, name="terminal"),
        ]

    monkeypatch.setattr(client_module, "reply_channel_tools", with_a_second_tool)
    stream = ChannelStream(body, name="terminal", text="")

    with pytest.raises(DecisionRejected):
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert [tool.name for tool in stream.requests[0].tools] == [
        REPLY_CHANNEL_TOOL_NAME,
        "terminal",
    ]


@pytest.mark.asyncio
async def test_the_older_name_test_still_holds_when_the_offer_is_not_passed() -> None:
    """``envelope_from_tool_call`` without ``offered_names`` is byte-for-byte the
    reader it was, so a caller that has not adopted the sole-offer rule cannot be
    changed by it."""

    call = ToolCall(name="submit_action_batch", raw_arguments='{"actions": []}')

    assert envelope_from_tool_call([call], name=REPLY_CHANNEL_TOOL_NAME) is None
    assert (
        envelope_from_tool_call(
            [call],
            name=REPLY_CHANNEL_TOOL_NAME,
            offered_names=[REPLY_CHANNEL_TOOL_NAME],
        )
        == '{"actions": []}'
    )
    assert (
        envelope_from_tool_call(
            [call],
            name=REPLY_CHANNEL_TOOL_NAME,
            offered_names=[REPLY_CHANNEL_TOOL_NAME, "terminal"],
        )
        is None
    )


# ---------------------------------------------------------------------------
# The widening is bounded: every refusal the channel could already earn, it
# still earns. Each case is its own test so a future widening has to break one
# of them by name rather than slip through a shared assertion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_foreign_named_calls_for_one_observation_still_refuse() -> None:
    """Widening WHICH calls are the channel must not widen how many may execute.

    Two calls carrying two complete batches are the ambiguity the prose decoder
    refuses, and the channel may not resolve by order what prose refuses.
    """

    current = observation()
    first = envelope(finish_payload(current), "first")
    second = envelope(finish_payload(current), "second")
    stream = ChannelStream("", calls=[("apply", first), ("apply", second)], text="")

    with pytest.raises(DecisionRejected):
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )


@pytest.mark.asyncio
async def test_duplicate_keys_in_a_foreign_named_call_still_refuse() -> None:
    """The channel hands back RAW bytes precisely so this cannot be repaired."""

    current = observation()
    duplicated = (
        '{"reply_version": "1.0", "reply_version": "1.0", "action_batch": '
        + json.dumps(json.loads(finish_payload(current)))
        + "}"
    )

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    with pytest.raises(DecisionRejected) as on_channel:
        await _client(
            ChannelStream(duplicated, name="apply", text=""),
            model_spec=_spec(supports_tools=True),
        ).decide(current, _turns(current))
    with pytest.raises(DecisionRejected) as in_prose:
        await _client(ScriptedStream(duplicated), model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    # PARITY is the property, not any particular message: the raw bytes reach the
    # decoder UNREPAIRED on both channels, so the same text fails identically
    # whether it arrived as prose or as a call. A re-serialized object would have
    # deduplicated these keys on the channel and this reply would have executed.
    assert on_channel.value.class_key == in_prose.value.class_key == "incomplete-json"


@pytest.mark.asyncio
async def test_a_call_carrying_no_decision_still_refuses() -> None:
    """A call to the channel whose arguments are not a decision is not one."""

    current = observation()
    stream = ChannelStream('{"action": "search"}', name="search", text="")

    with pytest.raises(DecisionRejected):
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )


@pytest.mark.asyncio
async def test_a_foreign_named_call_that_fails_validation_still_refuses() -> None:
    """A located-but-invalid batch keeps the class its own defect earns: the
    channel moves where the bytes came from, never what counts as valid."""

    current = observation()
    # A field NO kind declares. #1554's tolerance drops a field that belongs to a
    # sibling KIND -- the kind tag is explicit, so the action chosen is not in
    # doubt -- and deliberately does not reach this: nothing states what the model
    # meant by a name no kind has.
    bad = json.dumps(
        {
            "actions": [
                {
                    "kind": "key",
                    "observation_id": current.observation_id,
                    "keys": ["enter"],
                    "bogus": 1,
                }
            ]
        }
    )
    stream = ChannelStream(bad, name="apply", text="")

    with pytest.raises(DecisionRejected) as raised:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert raised.value.class_key == "extra-action-key"


@pytest.mark.asyncio
async def test_no_call_and_no_text_still_refuses() -> None:
    """The one shape that never reaches the decoder at all: a turn that said
    nothing on either channel is refused before parsing, so no widening here can
    turn silence into a decision."""

    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    stream = ScriptedStream("", stop_reason="toolUse")

    with pytest.raises(DecisionRejected) as raised:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert raised.value.class_key == "empty-reply"


# ---------------------------------------------------------------------------
# The refusal names its channel and its call, so a sealed bundle is diagnosable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_call_records_which_channel_was_read_and_what_it_was_called() -> None:
    """The one thing the sealed corpus could not answer, recorded from now on.

    178 of the arm's 204 ``leading-delimiter`` refusal artifacts published no
    reply text AND recorded no call name (recounted 2026-09-25 over
    ``~/worktrees/osworld/runs``), so nothing in the bundle said the model had
    answered on the tool channel at all -- every reader was sent down the
    framing path, and the class could not be diagnosed without paying for the
    run again. This pins that a refusal now states both.
    """

    from local_operator.evaluation.receipts import RedactionSet
    from local_operator.evaluation.runner.episode import _rejection_detail

    current = observation()
    # A real sealed reply: the model called with another tool's parameters.
    stream = ChannelStream('{"action": "search"}', name="search", text="")

    with pytest.raises(DecisionRejected) as info:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    rejected = info.value
    # The count was a 0 default on this path, so a bundle read as "the model
    # called nothing" on the very refusals where it called something.
    assert rejected.tool_call_count == 1
    assert rejected.channel_read is True

    artifact = _rejection_detail(rejected, RedactionSet.from_resolved_values([]))
    assert "channel=read(prose=0)" in artifact
    # Quoted, so a name carrying the separator (``apply,apply``) cannot read as
    # two calls -- see ``_bounded_call_names``.
    assert 'tool_calls=["search"]' in artifact
    assert "class: leading-delimiter" in artifact
    # The call's own arguments are what was judged, and they are in the artifact:
    # the diagnostic alone could never say that.
    assert '{"action": "search"}' in artifact


@pytest.mark.asyncio
async def test_a_call_name_cannot_forge_an_artifact_header_or_grow_it() -> None:
    """The name is model-authored text on a line a script parses field by field.

    The artifact's reader relies on a fixed section order, so a name carrying a
    newline must not be able to open a line that reads like another header — the
    same reason ``stop`` is escaped, one field over.
    """

    from local_operator.evaluation.receipts import RedactionSet
    from local_operator.evaluation.runner.episode import (
        _header_value,
        _rejection_detail,
    )
    from local_operator.evaluation.runner.model import StreamShape
    from local_operator.evaluation.runner.provider_client import _bounded_call_names

    class _Call:
        def __init__(self, name: str) -> None:
            self.name = name

    current = observation()
    stream = ChannelStream('{"action": "search"}', name="x\nclass: second-batch", text="")
    with pytest.raises(DecisionRejected) as info:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    artifact = _rejection_detail(info.value, RedactionSet.from_resolved_values([]))
    stream_lines = [line for line in artifact.splitlines() if line.startswith("stream: ")]
    # One line: the forged newline opened nothing, and the name is not lost either
    # -- the field carries it through the same escape every other model-authored
    # value takes, so the assertion is against the rendered field rather than a
    # fragment of it.
    assert len(stream_lines) == 1
    rendered = _header_value(_bounded_call_names([_Call("x\nclass: second-batch")]))
    assert f"tool_calls=[{rendered}]" in stream_lines[0]
    assert sum(1 for line in artifact.splitlines() if line.startswith("class: ")) == 1

    # Quoting is what makes the field readable as NAMES: one name carrying the
    # separator renders as ONE token, and two names render as two.
    assert _bounded_call_names([_Call("apply,apply")]) == '"apply,apply"'
    assert _bounded_call_names([_Call("a"), _Call("b")]) == '"a","b"'

    # Bounded: a model that names its call a kilobyte of prose cannot grow the
    # record, and an empty stream stays a reading rather than an absence. The
    # bound is on the raw NAME (64 characters) and the quoting around it is not
    # part of it -- the long-name case is asserted as the quoted whole so the two
    # are not confused.
    many = _bounded_call_names([_Call(f"call-{n}") for n in range(20)])
    assert len(many.split(",")) <= 10
    assert "+12 more" in many
    assert _bounded_call_names([_Call("n" * 5000)]) == '"' + "n" * 64 + '"'
    assert _bounded_call_names([]) == ""
    assert StreamShape().tool_call_names == ""


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


# ---------------------------------------------------------------------------
# Round 1 review: the channel must not accept what prose refuses, and the
# schema it offers must be one every provider can actually read.
# ---------------------------------------------------------------------------


class TwoCallStream:
    """A provider that names the reply channel TWICE for one observation."""

    def __init__(self, first: str, second: str) -> None:
        self.first = first
        self.second = second
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        for index, arguments in enumerate((self.first, self.second)):
            yield StreamToolCallDelta(index=index, id=f"call-{index}", name=REPLY_CHANNEL_TOOL_NAME)
            yield StreamToolCallDelta(index=index, argument_delta=arguments)
        yield StreamEndEvent(stop_reason="toolUse")


class NamedButEmptyCallStream:
    """Prose carries a COMPLETE envelope; the channel is named with no arguments.

    A length stop mid-call produces the same shape, which is why this is a
    regression risk rather than a curiosity: the model did answer, in prose.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        yield StreamTextDelta(delta=self.text)
        yield StreamToolCallDelta(index=0, id="call-1", name=REPLY_CHANNEL_TOOL_NAME)
        yield StreamEndEvent(stop_reason="toolUse")


@pytest.mark.asyncio
async def test_two_channel_calls_are_refused_exactly_as_two_prose_batches_are() -> None:
    """The channel may not accept an ambiguity the prose decoder rejects.

    Taking the first call would execute a decision the model may have
    superseded — the precise reason the prose path refuses a second batch for
    the same observation. Both channels must reach the same verdict, or the
    channel has become a lenient second contract.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")

    client = _client(TwoCallStream(body, body), model_spec=_spec(supports_tools=True))
    with pytest.raises(DecisionRejected) as raised:
        await client.decide(current, _turns(current))

    assert "second action batch for the same observation" in str(raised.value)


@pytest.mark.asyncio
async def test_an_empty_channel_call_leaves_a_valid_prose_reply_standing() -> None:
    """Naming the channel without arguments must not destroy a good prose answer.

    Preferring the channel unconditionally turned a turn the harness would have
    accepted into "not valid JSON: Expecting value" — the very rejection the
    channel exists to remove.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    client = _client(NamedButEmptyCallStream(body), model_spec=_spec(supports_tools=True))

    decision = await client.decide(current, _turns(current))

    assert isinstance(decision.action_batch, ActionBatch)
    decision.action_batch.validate_for(current)


@pytest.mark.asyncio
async def test_a_foreign_named_call_does_not_override_a_prose_decision() -> None:
    """A turn that answered on BOTH channels is judged on the prose it wrote.

    This is the widening's bound rather than a preference: reading the call as
    the channel DISCARDS a complete decision the model wrote and refuses the
    turn on the call's bytes instead -- measured on this branch, a foreign-named
    call carrying ``{"query": "weather"}`` turned an ACCEPTED reply into a
    ``batch-shape`` refusal. The widening is a recovery of a decision that would
    otherwise be lost, so it applies only where there was none to lose.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    stream = ChannelStream('{"query": "weather"}', name="web_search", text=body)

    decision = await _client(stream, model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )

    # The PROSE is what was judged: neither the call's bytes nor a repair of them
    # may reach the decision.
    assert decision.public_reply == body
    decision.action_batch.validate_for(current)


@pytest.mark.asyncio
async def test_a_foreign_named_call_is_still_read_when_the_prose_states_no_decision() -> None:
    """The gate is "the prose stated a decision", not "the prose was empty".

    A chatty prose reply carries no decision to lose, so the measured class
    stays fixed: 178 of the arm's 204 ``leading-delimiter`` refusal artifacts
    published no reply text at all (recounted 2026-09-25 over
    ``~/worktrees/osworld/runs``), but the question the gate asks is whether a
    decision would be LOST, and prose that states none loses none.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    stream = ChannelStream(body, name="web_search", text="Let me look at the screen first.")

    decision = await _client(stream, model_spec=_spec(supports_tools=True)).decide(
        current, _turns(current)
    )

    assert decision.public_reply == body
    decision.action_batch.validate_for(current)


@pytest.mark.asyncio
async def test_a_refused_prose_decision_withholds_the_widening() -> None:
    """A prose decision the decoder REFUSES still owns the turn.

    The gate asks whether a decision is THERE, not whether it is usable:
    deciding this on the call's valid envelope would swap the model's answer for
    one the harness has no reason to prefer, and would erase the refusal the
    model needs to see.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    duplicated = (
        '{"reply_version": "1.0", "reply_version": "1.0", "action_batch": '
        + json.dumps(json.loads(finish_payload(current)))
        + "}"
    )
    stream = ChannelStream(body, name="web_search", text=duplicated)

    with pytest.raises(DecisionRejected) as raised:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert raised.value.class_key == "incomplete-json"
    assert raised.value.channel_read is False


@pytest.mark.asyncio
async def test_prose_that_is_json_but_not_a_decision_still_withholds_the_widening() -> None:
    """The gate is about a decision being THERE, not about what it decodes to.

    A prose reply that opens a JSON value the decoder reads and refuses (a
    non-object, here) is a decision the model wrote; widening over it would
    replace a refusal the model needs to see with an acceptance it never asked
    for. This is the boundary of ``_states_a_decision``: only "nothing could be
    read at all" counts as decisionless.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    stream = ChannelStream(body, name="web_search", text="[1, 2, 3]")

    with pytest.raises(DecisionRejected) as raised:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    assert raised.value.class_key == "batch-shape"
    assert raised.value.channel_read is False


@pytest.mark.asyncio
async def test_a_read_channel_publishes_how_much_prose_it_set_aside() -> None:
    """The refusal artifact carries the prose count, so the collateral is countable.

    A sealed refusal showed ``channel=read`` and nothing about what the channel
    decision was made OVER, so how often prose and a call co-occur was
    unmeasurable from a bundle. The count rides on the READ branch only: on the
    prose branch the reply section already IS that prose.
    """

    from local_operator.evaluation.receipts import RedactionSet
    from local_operator.evaluation.runner.episode import _rejection_detail

    current = observation()
    prose = "Let me look at the screen first."
    stream = ChannelStream('{"query": "weather"}', name="web_search", text=prose)

    with pytest.raises(DecisionRejected) as raised:
        await _client(stream, model_spec=_spec(supports_tools=True)).decide(
            current, _turns(current)
        )

    artifact = _rejection_detail(raised.value, RedactionSet.from_resolved_values([]))
    assert raised.value.channel_read is True
    assert f"channel=read(prose={len(prose)})" in artifact


def test_the_offered_schema_avoids_constructs_providers_reject() -> None:
    """The schema goes on the wire as a function declaration, so it must be portable.

    Gemini's ``FunctionDeclaration.parameters`` accepts a narrow OpenAPI subset:
    ``$ref``, ``$defs``, ``oneOf`` and ``discriminator`` are rejected with 400
    INVALID_ARGUMENT, which would fail EVERY request of a Gemini-routed episode
    rather than degrading. This was previously unreachable only because the
    runner sent no tools at all.
    """

    serialized = json.dumps(public_reply_schema())

    for construct in ("$ref", "$defs", "oneOf", "discriminator"):
        assert construct not in serialized, f"{construct} reaches the provider verbatim"


def test_the_offered_schema_advertises_only_what_the_surface_accepts() -> None:
    """Offering an action the surface rejects invites the model into a rejection.

    The prose prompt is surface-aware, so an unfiltered schema would present a
    different contract than the prompt — and a model following the schema would
    be refused by ``validate_batch``.
    """

    from dataclasses import replace

    from local_operator.evaluation.runner.provider_client import LEGACY_ACTION_SURFACE

    legacy = json.dumps(public_reply_schema(LEGACY_ACTION_SURFACE))
    assert "paste_text" not in legacy

    without_ask = replace(LEGACY_ACTION_SURFACE, ask_user=False)
    assert "ask_user" not in json.dumps(public_reply_schema(without_ask))

    # Unchanged for the published contract, which describes the protocol rather
    # than one adapter's negotiated subset.
    assert "paste_text" in json.dumps(public_reply_schema())


def test_the_surface_scoped_schema_is_still_byte_stable_across_turns() -> None:
    """The surface is fixed for an episode, so the cache prefix must not move."""

    from local_operator.evaluation.runner.provider_client import LEGACY_ACTION_SURFACE

    first = json.dumps(public_reply_schema(LEGACY_ACTION_SURFACE), sort_keys=True)
    second = json.dumps(public_reply_schema(LEGACY_ACTION_SURFACE), sort_keys=True)

    assert first == second


# ---------------------------------------------------------------------------
# Round 2 review: the round 1 fixes reopened one defect and widened the schema.
# ---------------------------------------------------------------------------


class TwoEmptyCallStream:
    """Prose carries a COMPLETE envelope; the channel is named twice, emptily.

    The single-empty-call case was already guarded. Joining several empty calls
    produced a string of separators, which is TRUTHY, so the guard was bypassed
    and the good prose reply was destroyed anyway.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        yield StreamTextDelta(delta=self.text)
        for index in range(2):
            yield StreamToolCallDelta(index=index, id=f"call-{index}", name=REPLY_CHANNEL_TOOL_NAME)
        yield StreamEndEvent(stop_reason="toolUse")


@pytest.mark.asyncio
async def test_several_empty_channel_calls_still_leave_the_prose_reply_standing() -> None:
    """Joining empty calls must not manufacture a truthy reply out of separators."""

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    client = _client(TwoEmptyCallStream(body), model_spec=_spec(supports_tools=True))

    decision = await client.decide(current, _turns(current))

    assert isinstance(decision.action_batch, ActionBatch)
    decision.action_batch.validate_for(current)


def test_the_flattened_schema_admits_exactly_what_the_validator_admits() -> None:
    """Flattening a discriminated union must not widen the admitted set.

    ``kind`` carries the discriminator and pydantic leaves it out of
    ``required`` because each member defaults it. That is safe under a
    discriminated union, where the tag selects the member before its fields are
    checked, and unsafe under a bare ``anyOf``, where a tag-less action matches
    whichever member its other fields happen to satisfy. Admitting at the
    schema what the validator rejects invites the model into a rejection, which
    is the failure class this contract exists to prevent.
    """

    import jsonschema

    items = public_reply_schema()["properties"]["actions"]["items"]
    base = {
        "protocol_version": "1.0",
        "task_id": "t",
        "episode_id": "e",
        "observation_id": "o",
        "kind": "action_batch",
    }

    def admitted_by_schema(action: dict[str, Any]) -> bool:
        try:
            jsonschema.validate(action, items)
            return True
        except jsonschema.ValidationError:
            return False

    def admitted_by_validator(action: dict[str, Any]) -> bool:
        try:
            ActionBatch.model_validate({**base, "actions": [action]})
            return True
        except Exception:
            return False

    cases = [
        # The exact widening round 2 found: no ``kind``, but a click's fields.
        {"observation_id": "o", "frame_id": "f", "x": 1, "y": 2},
        {"kind": "click", "observation_id": "o", "frame_id": "f", "x": 1, "y": 2},
        {"kind": "wait", "observation_id": "o", "duration_ms": 10},
        {"kind": "type", "observation_id": "o", "text": "hi"},
        {"kind": "finish", "observation_id": "o", "status": "done", "reason": "r"},
        {"kind": "nope", "observation_id": "o"},
        {"kind": "wait", "observation_id": "o", "duration_ms": 999999},
    ]

    for action in cases:
        assert admitted_by_schema(action) == admitted_by_validator(action), action

    # The agreement above is about SHAPE: presence, type, and per-field range.
    # It is deliberately not total, and saying so here matters because a green
    # test otherwise reads as "the schema and the validator agree, full stop".
    #
    # A cross-field invariant cannot be expressed in the JSON Schema subset
    # these providers accept, so the validator stays strictly stricter for
    # those. ``ScrollAction`` requires some motion; the schema admits a scroll
    # with both deltas zero and pydantic then rejects it. This is INHERENT, not
    # a consequence of flattening: the pre-flattening discriminated schema
    # admitted it too. What the flattening must not do is widen the set beyond
    # that pre-existing boundary, which is what the loop above pins.
    motionless_scroll = {
        "kind": "scroll",
        "observation_id": "o",
        "frame_id": "f",
        "x": 1,
        "y": 2,
        "delta_x": 0,
        "delta_y": 0,
    }
    assert admitted_by_schema(motionless_scroll) is True
    assert admitted_by_validator(motionless_scroll) is False


@pytest.mark.asyncio
async def test_a_rejected_reply_still_records_the_tools_the_request_offered() -> None:
    """The rejection path is where the offered count matters most.

    A reply rejected WHILE the channel was on offer is the measurement that
    says whether offering it is paying off. Recording 0 there would describe a
    request the wire never sent, and would understate exactly the population
    worth counting.
    """

    from local_operator.evaluation.runner.model import DecisionRejected
    from tests.unit.evaluation.runner.test_provider_client import ScriptedStream

    current = observation()
    client = _client(ScriptedStream("not json at all"), model_spec=_spec(supports_tools=True))

    with pytest.raises(DecisionRejected) as raised:
        await client.decide(current, _turns(current))

    assert raised.value.offered_tool_count == 1


@pytest.mark.asyncio
async def test_a_channel_answer_without_prose_is_not_read_as_silence() -> None:
    """Pins the ``tool_call_count == 0`` half of the silent-reply guard.

    ``provider_client`` rejects a reply that emitted no tool call AND no text,
    because that model said nothing on either channel. The tool-call half of
    that condition is load-bearing: a model that answers ON the channel and
    writes no prose arrives with empty ``text``, so keying the guard on silence
    alone would reject the very replies the channel exists to carry.

    This test asserts the ACCEPTANCE side of that -- a channel answer is
    decoded normally. It does NOT by itself fail when ``tool_call_count == 0``
    is deleted, because a valid channel call sets ``text`` from the channel
    reply and so never reaches the guard. The mutation is caught by
    ``test_an_empty_channel_call_is_rejected_like_an_empty_prose_body``, whose
    empty-argument call leaves ``text`` empty WITH a tool call present -- the
    one shape that distinguishes the two conditions. Both are kept: this one
    pins the behaviour, that one pins the clause.
    """

    current = observation()
    body = envelope(finish_payload(current), "Visible status: ready")
    client = _client(ChannelStream(body), model_spec=_spec(supports_tools=True))

    decision = await client.decide(current, _turns(current))

    assert isinstance(decision.action_batch, ActionBatch)
    assert decision.tool_call_count == 1


@pytest.mark.asyncio
async def test_an_empty_channel_call_is_rejected_like_an_empty_prose_body() -> None:
    """A channel call carrying EMPTY arguments is malformed, not silent.

    This is the half of the old empty-body parity case that still belongs to
    the malformed family. The model selected the channel and sent nothing in
    it, so a tool call really was emitted -- the silent-reply guard must not
    claim the model said nothing, and the reply must still be rejected on the
    correctable path.

    Kept as its own test because the two halves of the old parametrised case
    encode DIFFERENT model behaviours that the fixture spelled with the same
    empty string.
    """

    current = observation()
    client = _client(ChannelStream(""), model_spec=_spec(supports_tools=True))

    with pytest.raises(DecisionRejected) as info:
        await client.decide(current, _turns(current))

    message = str(info.value)
    assert "no tool call and no text" not in message
    # The reply is reported as what it is: a reply that did not come through as
    # one JSON object, bucketed as that class -- and specifically as the
    # OFFSET-ZERO half of it, because an empty reply has no first byte to read.
    # The wording moved when the model-facing text became a shape hint instead
    # of the decoder's prose, and the key moved when the old ``malformed-json``
    # class was split into the half that cannot start and the half that starts
    # and breaks; the distinction this test draws -- not silent, but unreadable
    # -- is unchanged.
    assert "did not begin with the JSON object" in message
    assert info.value.class_key == "leading-delimiter"
