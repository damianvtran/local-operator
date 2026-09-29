"""Input-refusal recovery: a content screen's refusal of the request's INPUT is
re-asked with a NARROWED request, bounded, recorded, and legible.

The class under test is what the OSWorld arm hit: HTTP 400
``data_inspection_failed`` ("Input text data may contain inappropriate
content"). These tests drive the REAL :func:`stream_with_failover` with
scripted wires that replay the captured body, so the recovery, its ceiling,
and the terminal message are exercised through the same walk every consumer
(session loop, evaluation runner, errands) uses.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    Content,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.providers.failover import (
    INPUT_REFUSAL_IMAGE_PLACEHOLDER,
    INPUT_REFUSAL_OBSERVATION_PLACEHOLDER,
    MAX_INPUT_REFUSAL_RETRIES,
    ProviderError,
    is_input_refusal,
    stream_with_failover,
)
from tests.unit.providers.test_failover import FakeAuth, _FnClient

pytestmark = pytest.mark.asyncio

#: The field's terminal body, verbatim apart from the relay id: what
#: ``_extract_error_message`` hands the walk when Alibaba's content screen
#: refuses an input (captured from runs/a1748-w2-task_006-20260929-130146).
FIELD_BODY = (
    'invalid_request: data: {"error":{"code":"data_inspection_failed","param":null,'
    '"message":"Input text data may contain inappropriate content.",'
    '"type":"data_inspection_failed"},'
    '"id":"chatcmpl-8d1ef3ff-dfa9-98ef-854f-e2942931508b"}'
)


def _refusal() -> ProviderError:
    return ProviderError(400, FIELD_BODY)


def _image_count(request: ChatRequest) -> int:
    return sum(
        1
        for message in request.messages
        for block in message.content
        if isinstance(block, ImageContent)
    )


def _long_request(*, images: bool = True, rows: int = 40) -> ChatRequest:
    """> the shed rung's recent window, so both rungs are in play."""
    messages: list[Message] = [Message.user("Task: find the candidates")]
    for index in range(rows):
        messages.append(Message.assistant(text=f"thinking {index}"))
        content: list[Content] = [TextContent(text=f"Step: {index} observation " + "x" * 40)]
        if images:
            content.append(ImageContent(data="aGVsbG8=", mime_type="image/png"))
        messages.append(
            Message(
                role="tool",
                content=content,
                tool_call_id=f"call_{index}",
                tool_name="apply_actions",
            )
        )
    return ChatRequest(
        model=ModelSpec(provider="openrouter", model_id="qwen/qwen3.8-max-0902"),
        messages=messages,
    )


def _short_request() -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider="openrouter", model_id="qwen/qwen3.8-max-0902"),
        messages=[
            Message.user("Task: find the candidates"),
            Message(
                role="tool",
                content=[TextContent(text="Step: 0 observation")],
                tool_call_id="call_0",
                tool_name="apply_actions",
            ),
        ],
    )


def _scripted(
    refusals: int, seen: list[ChatRequest]
) -> tuple[Any, dict[str, int], list[ChatRequest]]:
    """A client_for + bookkeeping: refuse the first ``refusals`` attempts."""

    calls = {"n": 0}

    async def client_for(spec: ModelSpec):
        def run(request: ChatRequest, api_key: str | None, oauth_access=None):
            calls["n"] += 1
            seen.append(request)

            async def gen():
                if calls["n"] <= refusals:
                    raise _refusal()
                yield StreamTextDelta(delta="continued")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

        return _FnClient(run)

    return client_for, calls, seen


async def test_the_predicate_recognizes_the_field_refusal_and_only_input_side() -> None:
    refusal = _refusal()
    assert refusal.kind == "request"
    assert is_input_refusal(refusal)
    assert is_input_refusal(f"invalid request (HTTP 400): {FIELD_BODY}")

    # The OUTPUT-side sentence seen in the same field run arrives as a
    # retryable 5xx and must keep the ordinary retry ladder, where a fresh
    # generation can genuinely clear it.
    output_side = ProviderError(
        502, "Upstream error from Alibaba: Output data may contain inappropriate content."
    )
    assert not is_input_refusal(output_side)
    assert not is_input_refusal(str(output_side))

    # A malformed 400 is still a terminal request defect, and an image-FORMAT
    # rejection belongs to the sibling predicate.
    assert not is_input_refusal(
        ProviderError(400, "messages: text content blocks must be non-empty")
    )
    assert not is_input_refusal(ProviderError(400, "image could not be processed"))


async def test_an_input_refusal_is_re_asked_without_screenshots_and_recorded() -> None:
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(refusals=1, seen=seen)

    events = [
        event
        async for event in stream_with_failover(
            _long_request(), FakeAuth({"openrouter": ["k"]}), None, client_for
        )
    ]

    assert calls["n"] == 2
    # The retry changed the request: every image block became a placeholder,
    # and nothing else about the shape moved.
    retired = seen[1]
    assert _image_count(seen[0]) == 40 and _image_count(retired) == 0
    assert len(retired.messages) == len(seen[0].messages)
    assert any(
        isinstance(block, TextContent) and block.text == INPUT_REFUSAL_IMAGE_PLACEHOLDER
        for message in retired.messages
        for block in message.content
    )
    # Structure and pairing survive the copy.
    assert [m.tool_call_id for m in retired.messages] == [m.tool_call_id for m in seen[0].messages]
    assert all(m.tool_calls == o.tool_calls for m, o in zip(retired.messages, seen[0].messages))

    # The recovery rides the end event's payload -- the durable record the
    # loop turns into a notice.
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.stop_reason == "stop"
    assert end.provider_payload == {
        "input_refusal_recovery": {"degradations": ["screenshots_removed"]}
    }


async def test_the_ladder_runs_screenshots_then_observations_and_is_bounded() -> None:
    """All attempts refuse: two degraded re-asks, then a legible terminal.

    The ceiling is ``MAX_INPUT_REFUSAL_RETRIES`` degraded attempts (three wire
    requests total): one per rung, each request SMALLER than the last, and
    never a re-send of identical bytes.
    """
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(refusals=99, seen=seen)

    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _long_request(), FakeAuth({"openrouter": ["k"]}), None, client_for
            )
        ]

    assert calls["n"] == 1 + MAX_INPUT_REFUSAL_RETRIES
    first, second, third = seen
    assert _image_count(first) == 40
    assert _image_count(second) == 0  # rung 1: screenshots removed
    assert _image_count(third) == 0  # rung 2: ... plus older observations
    # Rung 2 shed only OLD tool rows: the newest window is byte-identical to
    # the post-rung-1 request, and the model's own turns and the task
    # statement are untouched.
    keep = 24
    assert third.messages[:-keep] != second.messages[:-keep]
    for old, new in zip(third.messages[-keep:], second.messages[-keep:]):
        assert old == new
    assert any(
        isinstance(block, TextContent) and block.text == INPUT_REFUSAL_OBSERVATION_PLACEHOLDER
        for message in third.messages[:-keep]
        if message.role == "tool"
        for block in message.content
    )
    assert all(
        message.content == original.content
        for message, original in zip(third.messages[:-keep], second.messages[:-keep])
        if message.role != "tool"
    )

    # The terminal error keeps the provider's words in front and names what
    # was tried, the never-processed fact, and the no-identical-resend rule.
    error = caught.value
    assert error.kind == "request" and error.status == 400 and not error.retryable
    assert "data_inspection_failed" in error.message
    assert "the provider refused the request's input as inappropriate content" in error.message
    assert "screenshots removed, then older observations removed" in error.message
    assert "was never processed" in error.message
    assert "not re-sent unchanged" in error.message


async def test_nothing_is_re_sent_when_no_rung_can_change_the_request() -> None:
    """A refused request with nothing removable earns no attempt at all -- the
    one thing the refusal already ruled out is re-sending identical bytes."""
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(refusals=99, seen=seen)

    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _short_request(), FakeAuth({"openrouter": ["k"]}), None, client_for
            )
        ]

    assert calls["n"] == 1
    assert "No degraded retry was available" in caught.value.message


async def test_a_retry_disabled_call_is_not_re_asked_but_stays_legible() -> None:
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(refusals=99, seen=seen)

    with pytest.raises(ProviderError) as caught:
        [
            event
            async for event in stream_with_failover(
                _long_request(),
                FakeAuth({"openrouter": ["k"]}),
                {"retry": {"enabled": False}},
                client_for,
            )
        ]

    assert calls["n"] == 1
    assert (
        "the provider refused the request's input as inappropriate content" in caught.value.message
    )


async def test_a_normal_call_is_untouched_and_carries_no_recovery() -> None:
    """The happy path: one attempt, the request object unmoved byte for byte,
    and no recovery key anywhere on the stream."""
    seen: list[ChatRequest] = []
    client_for, calls, _ = _scripted(refusals=0, seen=seen)
    request = _long_request()
    before = request.model_dump()

    events = [
        event
        async for event in stream_with_failover(
            request, FakeAuth({"openrouter": ["k"]}), None, client_for
        )
    ]

    assert calls["n"] == 1
    assert seen[0].model_dump() == before
    end = [event for event in events if isinstance(event, StreamEndEvent)][-1]
    assert end.provider_payload is None
