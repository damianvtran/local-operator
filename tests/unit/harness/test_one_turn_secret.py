"""One-turn, non-persisted native ToolResult secret lifecycle tests.

All values are synthetic canaries assembled at runtime. No credential store or
producer tool is involved; these tests drive the reviewed internal prerequisite.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.parse
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from pydantic import BaseModel, SecretStr
from pydantic_core import PydanticSerializationError

from local_operator.harness.loop import (
    AgentLoop,
    LoopContext,
    _ModelTurnResult,
    _OneTurnSecretLease,
    _PlannedCall,
)
from local_operator.harness.types import (
    ONE_TURN_SECRET_RECEIPT,
    AbortSignal,
    AgentEndEvent,
    AgentMessage,
    AgentTool,
    AgentToolUpdate,
    ChatRequest,
    LoopConfig,
    Message,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSpec,
    OneTurnSecretScopeError,
    OneTurnSecretSerializationError,
    ProviderTurnStartEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamReasoningDelta,
    StreamStartEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolContext,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
)
from local_operator.variables import VariableStore

MODEL = ModelSpec(provider="test", model_id="m")


def _canary(label: str = "value") -> str:
    return "v0rn-" + label + "-7d4c"


def _secret_tool(name: str, value: str) -> AgentTool:
    async def execute(tool_call_id: str, args: dict[str, Any], *_args: Any) -> ToolResult:
        return ToolResult.one_turn_secret(tool_call_id=tool_call_id, tool_name=name, value=value)

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _echo_tool(name: str, executed: list[dict[str, Any]]) -> AgentTool:
    async def execute(tool_call_id: str, args: dict[str, Any], *_args: Any) -> ToolResult:
        executed.append(args)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=name,
            content=[TextContent(text=json.dumps(args))],
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )


def _config(
    stream_fn: Any,
    variables: VariableStore,
    **kwargs: Any,
) -> LoopConfig:
    values: dict[str, Any] = {
        "model": MODEL,
        "convert_to_llm": lambda messages: [
            message for message in messages if isinstance(message, Message)
        ],
        "stream_fn": stream_fn,
        "redact_tool_result": variables.redact,
        "redact_tool_arguments": variables.redact_arguments,
    }
    values.update(kwargs)
    return LoopConfig(**values)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def _tool_delta(call_id: str, name: str, arguments: str) -> list[Any]:
    midpoint = len(arguments) // 2
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name),
        StreamToolCallDelta(index=0, argument_delta=arguments[:midpoint]),
        StreamToolCallDelta(index=0, argument_delta=arguments[midpoint:]),
    ]


def _public_tool_result(call_id: str = "call-a", value: str | None = None) -> ToolResult:
    if value is not None:
        return ToolResult.one_turn_secret(
            tool_call_id=call_id, tool_name="native-secret", value=value
        )
    return ToolResult(
        tool_call_id=call_id,
        tool_name="native-secret",
        content=[TextContent(text=ONE_TURN_SECRET_RECEIPT)],
    )


def _request(*call_ids: str) -> ChatRequest:
    # A retry rebuilds a request over the SAME durable history, including its
    # stable message ids. Fresh random ids here would be a genuinely new request.
    user = Message.user("go")
    user.id = "user-go"
    rows = []
    for call_id in call_ids:
        row = Message.tool_result(_public_tool_result(call_id))
        row.id = "tool-" + call_id
        rows.append(row)
    return ChatRequest(model=MODEL, messages=[user, *rows])


def test_unmarked_result_and_request_keep_identity_and_public_shape() -> None:
    result = ToolResult(
        tool_call_id="ordinary",
        tool_name="echo",
        content=[TextContent(text="ok")],
        details=None,
    )
    before = result.model_dump(mode="json")
    assert before == {
        "tool_call_id": "ordinary",
        "tool_name": "echo",
        "content": [{"type": "text", "text": "ok"}],
        "details": None,
        "duration_s": None,
        "is_error": False,
        "useless": False,
    }
    assert not any("one_turn" in key for key in before)
    request = ChatRequest(model=MODEL, messages=[Message.tool_result(result)])
    assert request.wire_messages() is request.messages
    assert request.wire_messages()[0] is request.messages[0]
    assert not request.has_one_turn_secrets


def test_armed_models_fail_direct_and_nested_serialization_without_canary() -> None:
    canary = _canary("serialization")
    marked = _public_tool_result(value=canary)
    request = _request("call-a")
    request.arm_one_turn_secrets({"call-a": SecretStr(canary)})
    wire_messages = request.wire_messages()
    raw_message = wire_messages[-1]
    assert raw_message is not request.messages[-1]
    assert wire_messages[0] is request.messages[0]

    class Envelope(BaseModel):
        result: ToolResult
        message: Message
        request: ChatRequest

    for model in (marked, raw_message, request):
        for dump in (lambda: model.model_dump(mode="json"), model.model_dump_json):
            with pytest.raises(OneTurnSecretSerializationError) as caught:
                dump()
            assert canary not in str(caught.value)
            assert "armed one-turn secret" in str(caught.value)

    with pytest.raises(PydanticSerializationError) as nested:
        Envelope(result=marked, message=raw_message, request=request).model_dump(mode="json")
    assert canary not in str(nested.value)
    assert isinstance(nested.value.__cause__, OneTurnSecretSerializationError)


@pytest.mark.parametrize("bad_value", ["", None, 17])
def test_secret_producer_rejects_empty_or_non_string_values(bad_value: Any) -> None:
    with pytest.raises(ValueError) as caught:
        ToolResult.one_turn_secret(
            tool_call_id="call-a", tool_name="native-secret", value=bad_value
        )
    assert str(caught.value) == "One-turn secret value must be a non-empty string."


@pytest.mark.asyncio
async def test_native_capture_wire_echo_execution_and_cleanup() -> None:
    canary = _canary("echo")
    variables = VariableStore()
    executed: list[dict[str, Any]] = []
    hook_results: list[dict[str, Any]] = []
    reporter_results: list[dict[str, Any]] = []
    requests: list[ChatRequest] = []

    def capture_write(_call: ToolCall, result: ToolResult) -> tuple[list[str], bool]:
        reporter_results.append(result.model_dump(mode="json"))
        return [], True

    async def post_hook(
        _name: str, _args: dict[str, Any], _call_id: str, result: ToolResult
    ) -> list[str]:
        hook_results.append(result.model_dump(mode="json"))
        return []

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            if index == 2:
                assert request.has_one_turn_secrets
                assert request.messages[-1].text == ONE_TURN_SECRET_RECEIPT
                wire_text = json.dumps(
                    [message.model_dump for message in []]
                )  # negative control: no generic armed dump
                assert wire_text == "[]"
                wire = request.wire_messages()
                assert wire[-1].text == canary
                yield StreamTextDelta(delta=canary[:5])
                yield StreamReasoningDelta(delta="think " + canary)
                yield StreamTextDelta(delta=canary[5:])
                for event in _tool_delta("echo-call", "echo", json.dumps({"text": canary})):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert not request.has_one_turn_secrets
            assert request.wire_messages() is request.messages
            assert canary not in json.dumps(request.model_dump(mode="json"))
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary), _echo_tool("echo", executed)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("go")],
            context,
            _config(
                stream_fn,
                variables,
                post_tool_hooks=post_hook,
                capture_prepared_write=capture_write,
            ),
            None,
        )
    ]

    assert executed == [{"text": canary}]
    assert len(requests) == 3
    assert [event.delta for event in events if isinstance(event, MessageUpdateEvent)] == ["done"]
    assert not any(isinstance(event, ReasoningDeltaEvent) for event in events)
    assert {event.tool_call_id for event in events if isinstance(event, ToolCallComposeEvent)} == {
        "secret-call"
    }
    starts = [event for event in events if isinstance(event, ToolExecutionStartEvent)]
    echo_start = next(event for event in starts if event.tool_call_id == "echo-call")
    assert canary not in json.dumps(echo_start.model_dump(mode="json"))

    serialized_events = json.dumps([event.model_dump(mode="json") for event in events])
    assert canary not in serialized_events
    assert canary not in json.dumps(hook_results)
    assert canary not in json.dumps(reporter_results)
    assert canary not in json.dumps(
        [message.model_dump(mode="json") for message in context.messages]
    )
    tool_rows = [
        message
        for message in context.messages
        if isinstance(message, Message) and message.role == "tool"
    ]
    assert tool_rows[0].text == ONE_TURN_SECRET_RECEIPT
    assert context.one_turn_secret_leases == {}
    assert not context.one_turn_secret_run_active
    assert context.original_call_args == {}
    assert canary not in variables.redaction_values()
    end = next(event for event in events if isinstance(event, AgentEndEvent))
    assert end.error is None


@pytest.mark.asyncio
async def test_parallel_sequential_retry_and_duplicate_lease_scope() -> None:
    first = _canary("parallel-a")
    second = _canary("parallel-b")
    variables = VariableStore()
    context = LoopContext(tool_context=ToolContext(cwd=".", variables=variables))
    loop = AgentLoop()
    call_a = _tool_call("call-a", "native-secret", {})
    call_b = _tool_call("call-b", "native-secret", {})

    public_a = loop._capture_one_turn_secret(
        context, call_a, "native-secret", _public_tool_result(value=first), allow=True
    )
    assert public_a.text == ONE_TURN_SECRET_RECEIPT
    assert public_a.model_dump(mode="json")
    loop._capture_one_turn_secret(
        context,
        call_b,
        "native-secret",
        _public_tool_result(call_id="call-b", value=second),
        allow=True,
    )
    both = loop._arm_one_turn_secrets(context, _request("call-a", "call-b"))
    assert [message.text for message in both.wire_messages()[-2:]] == [first, second]

    # An identical outer retry differs only in timing/retry observations.
    retry = _request("call-a", "call-b")
    retry.preparation_ms = 999.0
    retry.same_route_retry = True
    assert [
        message.text for message in loop._arm_one_turn_secrets(context, retry).wire_messages()[-2:]
    ] == [first, second]

    # A bad duplicate cannot overwrite the still-live A lease.
    duplicate = loop._capture_one_turn_secret(
        context, call_a, "native-secret", _public_tool_result(value=_canary("bad")), allow=True
    )
    assert duplicate.is_error
    assert context.one_turn_secret_leases["call-a"].value.get_secret_value() == first

    loop._clear_one_turn_secrets(context)

    # Separate sequential lifecycle: A was already sent, THEN B was produced.
    # B is therefore pending (not an already-armed lease) when the changed later
    # request expires A and arms B independently.
    sequential = LoopContext(tool_context=ToolContext(cwd=".", variables=variables))
    loop._capture_one_turn_secret(
        sequential, call_a, "native-secret", _public_tool_result(value=first), allow=True
    )
    loop._arm_one_turn_secrets(sequential, _request("call-a"))
    loop._capture_one_turn_secret(
        sequential,
        call_b,
        "native-secret",
        _public_tool_result(call_id="call-b", value=second),
        allow=True,
    )
    later_request = _request("call-a", "call-b")
    intermediate = Message(role="assistant", content=[TextContent(text="next step")])
    intermediate.id = "assistant-intermediate"
    later_request.messages.insert(2, intermediate)
    later = loop._arm_one_turn_secrets(sequential, later_request)
    later_tool_rows = [message for message in later.wire_messages() if message.role == "tool"]
    assert [message.text for message in later_tool_rows] == [
        ONE_TURN_SECRET_RECEIPT,
        second,
    ]
    assert "call-a" not in sequential.one_turn_secret_leases
    assert first not in variables.redaction_values()
    assert second in variables.redaction_values()
    loop._clear_one_turn_secrets(sequential)
    assert context.one_turn_secret_leases == {}
    assert sequential.one_turn_secret_leases == {}
    assert first not in variables.redaction_values()
    assert second not in variables.redaction_values()


def test_missing_or_duplicate_wire_target_fails_before_provider_use() -> None:
    canary = _canary("scope")
    request = _request("call-a", "call-a")
    request.arm_one_turn_secrets({"call-a": SecretStr(canary)})
    with pytest.raises(OneTurnSecretScopeError):
        request.wire_messages()
    request = _request("call-b")
    request.arm_one_turn_secrets({"call-a": SecretStr(canary)})
    with pytest.raises(OneTurnSecretScopeError):
        request.wire_messages()


def _context_with_lease(loop: AgentLoop, variables: VariableStore, canary: str) -> LoopContext:
    request = _request("call-a")
    context = LoopContext(
        system_blocks=[],
        messages=cast(list[AgentMessage], request.messages),
        tools=[],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    loop._capture_one_turn_secret(
        context,
        _tool_call("call-a", "native-secret", {}),
        "native-secret",
        _public_tool_result(value=canary),
        allow=True,
    )
    return context


def _public_event_json(events: list[Any]) -> str:
    return json.dumps(
        [event.model_dump(mode="json") for event in events if isinstance(event, BaseModel)]
    )


@pytest.mark.asyncio
async def test_preabort_never_iterates_provider_and_cleans_on_run_finally() -> None:
    canary = _canary("preabort")
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)
    started: list[bool] = []

    def stream_fn(_request: ChatRequest, _signal: AbortSignal | None) -> Any:
        async def generate() -> Any:
            started.append(True)
            yield StreamTextDelta(delta=canary)

        return generate()

    signal = AbortSignal()
    signal.abort("test abort")
    events = [
        event async for event in loop._model_turn(context, _config(stream_fn, variables), signal)
    ]
    assert started == []
    assert canary not in _public_event_json(events)
    loop._clear_one_turn_secrets(context)
    assert context.one_turn_secret_leases == {}
    assert canary not in variables.redaction_values()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["text", "tool"])
async def test_hard_cancel_scrubs_shared_placeholder(partial: str) -> None:
    canary = _canary("cancel-" + partial)
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)

    def stream_fn(_request: ChatRequest, _signal: AbortSignal | None) -> Any:
        async def generate() -> Any:
            if partial == "text":
                yield StreamTextDelta(delta=canary)
            else:
                yield StreamToolCallDelta(
                    index=0,
                    id="echo-call",
                    name="echo",
                    argument_delta=json.dumps({"text": canary}),
                )
            await asyncio.Event().wait()

        return generate()

    turn = cast(
        AsyncGenerator[Any, Any],
        loop._model_turn(context, _config(stream_fn, variables), None),
    )
    await anext(turn)  # TurnStartEvent precedes the shared assistant placeholder.
    start = await anext(turn)
    assert isinstance(start, MessageStartEvent)
    pending = asyncio.ensure_future(anext(turn))
    for _ in range(4):
        await asyncio.sleep(0)
    assert not pending.done()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert canary not in start.message.model_dump_json()
    assert isinstance(start.message, Message)
    assert start.message.provider_payload is None
    loop._clear_one_turn_secrets(context)
    assert canary not in variables.redaction_values()


@pytest.mark.asyncio
async def test_generator_close_runs_the_same_final_scrub() -> None:
    canary = _canary("generator-close")
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)

    def stream_fn(_request: ChatRequest, _signal: AbortSignal | None) -> Any:
        async def generate() -> Any:
            yield StreamTextDelta(delta=canary)
            yield StreamStartEvent(response_id="synthetic")
            await asyncio.Event().wait()

        return generate()

    turn = cast(
        AsyncGenerator[Any, Any],
        loop._model_turn(context, _config(stream_fn, variables), None),
    )
    await anext(turn)  # TurnStartEvent precedes the shared assistant placeholder.
    start = await anext(turn)
    boundary = await anext(turn)
    assert isinstance(boundary, ProviderTurnStartEvent)
    await turn.aclose()
    assert isinstance(start, MessageStartEvent)
    assert canary not in start.message.model_dump_json()
    assert isinstance(start.message, Message)
    assert start.message.provider_payload is None
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
async def test_provider_exception_is_scrubbed_from_events_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = _canary("provider-error")
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)

    def stream_fn(_request: ChatRequest, _signal: AbortSignal | None) -> Any:
        async def generate() -> Any:
            yield StreamTextDelta(delta=canary)
            raise RuntimeError("provider failed while sending " + canary)

        return generate()

    with caplog.at_level("WARNING"):
        events = [
            event async for event in loop._model_turn(context, _config(stream_fn, variables), None)
        ]
    assert canary not in _public_event_json(events)
    assert canary not in caplog.text
    result = events[-1]
    assert isinstance(result, _ModelTurnResult)
    assert result.error and canary not in result.error
    assert result.message.provider_payload is None
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
async def test_assistant_scrub_failure_blocks_tool_execution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = _canary("scrub-failure")
    variables = VariableStore()
    executed: list[dict[str, Any]] = []
    requests: list[ChatRequest] = []

    def failing_redact(text: str) -> str:
        if canary in text:
            raise RuntimeError("synthetic scrub failure")
        return variables.redact(text)

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            yield StreamTextDelta(delta=canary)
            for event in _tool_delta("echo-call", "echo", json.dumps({"text": canary})):
                yield event
            yield StreamEndEvent(stop_reason="toolUse")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary), _echo_tool("echo", executed)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    config = _config(stream_fn, variables)
    config.redact_tool_result = failing_redact
    with caplog.at_level("WARNING"):
        events = [
            event async for event in AgentLoop().run([Message.user("go")], context, config, None)
        ]
    assert executed == []
    assert len(requests) == 2
    assert canary not in _public_event_json(events)
    assert canary not in caplog.text
    assert canary not in json.dumps(
        [message.model_dump(mode="json") for message in context.messages]
    )
    end = next(event for event in events if isinstance(event, AgentEndEvent))
    assert end.error and canary not in end.error
    assert context.one_turn_secret_leases == {}
    assert canary not in variables.redaction_values()


@pytest.mark.asyncio
async def test_nested_eval_refuses_marker_without_capturing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = _canary("nested")
    variables = VariableStore()
    loop = AgentLoop()
    context = LoopContext(tool_context=ToolContext(cwd=".", variables=variables))
    call = _tool_call("nested-call", "native-secret", {})
    planned = _PlannedCall(call=call, tool=_secret_tool("native-secret", canary), args={})
    with caplog.at_level("WARNING"):
        result = await loop._runner_result(
            planned,
            context,
            _config(lambda *_args: None, variables),
            None,
            asyncio.Queue(),
            allow_one_turn_secret=False,
        )
    assert result.is_error
    assert result.text != ONE_TURN_SECRET_RECEIPT
    assert canary not in result.model_dump_json()
    assert canary not in caplog.text
    assert context.one_turn_secret_leases == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["task", "hub", "send"])
async def test_agent_forwarding_refuses_raw_but_allows_safe_arguments(
    tool_name: str,
) -> None:
    canary = _canary("forward-" + tool_name)
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)
    loop._arm_one_turn_secrets(context, _request("call-a"))
    dispatched: list[dict[str, Any]] = []

    async def execute(tool_call_id: str, args: dict[str, Any], *_args: Any) -> ToolResult:
        dispatched.append(args)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=[TextContent(text="safe result")],
        )

    tool = AgentTool(
        name=tool_name,
        parameters={"type": "object", "properties": {"prompt": {"type": "string"}}},
        execute=execute,
    )
    config = _config(lambda *_args: None, variables)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    raw_call = _tool_call("forward-raw", tool_name, {"prompt": canary})
    refusal = await loop._runner_result(
        _PlannedCall(call=raw_call, tool=tool, args={"prompt": canary}),
        context,
        config,
        None,
        queue,
    )
    assert refusal.is_error
    assert dispatched == []
    assert canary not in refusal.model_dump_json()

    safe_call = _tool_call("forward-safe", tool_name, {"prompt": "safe"})
    allowed = await loop._runner_result(
        _PlannedCall(call=safe_call, tool=tool, args={"prompt": "safe"}),
        context,
        config,
        None,
        queue,
    )
    assert not allowed.is_error
    assert dispatched == [{"prompt": "safe"}]
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("sessions", {"op": "spawn", "prompt": None}),
        ("sessions", {"op": "resume", "session": "child", "prompt": None}),
        ("network", {"action": "sessions", "peer": "peer", "create": True, "prompt": None}),
        ("network", {"action": "sessions", "peer": "peer", "send": "child", "text": None}),
        ("network", {"action": "sessions", "peer": "peer", "steer": "child", "text": None}),
        ("network", {"action": "sessions", "peer": "peer", "slash": "/run", "text": None}),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
async def test_sessions_and_network_forwarding_fail_before_dispatch(
    tool_name: str,
    args: dict[str, Any],
    nested: bool,
    encoded: bool,
) -> None:
    canary = 'v0rn-"quoted\\path\n' + tool_name + "-7d4c"
    payload = json.dumps(canary)[1:-1] if encoded else canary
    args = dict(args)
    carrier = "prompt" if "prompt" in args else "text"
    args[carrier] = payload
    variables = VariableStore()
    variables.acquire_scoped_redaction(canary)
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)
    loop._arm_one_turn_secrets(context, _request("call-a"))
    dispatched: list[dict[str, Any]] = []

    async def execute(tool_call_id: str, call_args: dict[str, Any], *_args: Any) -> ToolResult:
        dispatched.append(call_args)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=[TextContent(text="safe")],
        )

    tool = AgentTool(
        name=tool_name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    call = _tool_call("forward-operation", tool_name, args)
    result = await loop._runner_result(
        _PlannedCall(call=call, tool=tool, args=args),
        context,
        _config(lambda *_args: None, variables),
        None,
        asyncio.Queue(),
        allow_one_turn_secret=not nested,
    )
    assert result.is_error
    assert dispatched == []
    assert canary not in result.model_dump_json()
    assert payload not in result.model_dump_json()
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("sessions", {"op": "spawn", "prompt": "safe"}),
        ("sessions", {"op": "resume", "session": "child", "prompt": "safe"}),
        ("network", {"action": "sessions", "peer": "peer", "create": True, "prompt": "safe"}),
        ("network", {"action": "sessions", "peer": "peer", "send": "child", "text": "safe"}),
        ("network", {"action": "sessions", "peer": "peer", "steer": "child", "text": "safe"}),
        ("network", {"action": "sessions", "peer": "peer", "slash": "/run", "text": "safe"}),
    ],
)
async def test_sessions_and_network_safe_forwarding_remains_allowed(
    tool_name: str, args: dict[str, Any]
) -> None:
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, _canary("safe-forwarding"))
    loop._arm_one_turn_secrets(context, _request("call-a"))
    dispatched: list[dict[str, Any]] = []

    async def execute(tool_call_id: str, call_args: dict[str, Any], *_args: Any) -> ToolResult:
        dispatched.append(call_args)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=[TextContent(text="safe")],
        )

    tool = AgentTool(
        name=tool_name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    call = _tool_call("forward-safe-operation", tool_name, args)
    result = await loop._runner_result(
        _PlannedCall(call=call, tool=tool, args=args),
        context,
        _config(lambda *_args: None, variables),
        None,
        asyncio.Queue(),
    )
    assert not result.is_error
    assert dispatched == [args]
    loop._clear_one_turn_secrets(context)


def test_parent_and_child_lease_and_final_output_are_isolated() -> None:
    parent_value = _canary("parent")
    child_value = _canary("child")
    parent_variables = VariableStore()
    child_variables = VariableStore()
    loop = AgentLoop()
    parent = _context_with_lease(loop, parent_variables, parent_value)
    child = _context_with_lease(loop, child_variables, child_value)
    parent_request = loop._arm_one_turn_secrets(parent, _request("call-a"))
    child_request = loop._arm_one_turn_secrets(child, _request("call-a"))
    assert parent_request.wire_messages()[-1].text == parent_value
    assert child_request.wire_messages()[-1].text == child_value

    child_assistant = Message(
        role="assistant",
        content=[TextContent(text="child echoes " + child_value)],
        provider_payload={"native": child_value},
    )
    _executing, public, _error, failed = loop._finalize_one_turn_secret_assistant(
        child_assistant,
        None,
        _config(lambda *_args: None, child_variables),
        [child.one_turn_secret_leases["call-a"].value],
    )
    assert not failed
    assert child_value not in public.model_dump_json()
    assert public.provider_payload is None
    loop._clear_one_turn_secrets(parent)
    loop._clear_one_turn_secrets(child)


def test_scoped_redaction_release_preserves_parallel_and_preexisting_owners() -> None:
    canary = _canary("owners")
    variables = VariableStore()
    variables.register_redaction(canary)
    first = variables.acquire_scoped_redaction(canary)
    second = variables.acquire_scoped_redaction(canary)
    assert variables.release_scoped_redaction(first)
    assert canary in variables.redaction_values()
    assert variables.release_scoped_redaction(second)
    assert not variables.release_scoped_redaction(second)
    assert canary in variables.redaction_values()
    variables.unregister_redaction(canary)
    assert canary not in variables.redaction_values()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: setattr(result, "details", {"unexpected": True}),
        lambda result: setattr(result, "is_error", True),
        lambda result: setattr(result, "useless", True),
        lambda result: setattr(result, "tool_call_id", "wrong"),
        lambda result: result.content.append(TextContent(text="extra")),
        lambda result: setattr(result.content[0], "text", "producer-controlled"),
    ],
)
def test_invalid_marker_shapes_become_static_safe_errors(
    mutation: Any, caplog: pytest.LogCaptureFixture
) -> None:
    canary = _canary("invalid")
    result = _public_tool_result(value=canary)
    mutation(result)
    context = LoopContext()
    with caplog.at_level("WARNING"):
        public = AgentLoop()._capture_one_turn_secret(
            context,
            _tool_call("call-a", "native-secret", {}),
            "native-secret",
            result,
            allow=True,
        )
    assert public.is_error
    assert public.details is None
    assert canary not in public.model_dump_json()
    assert canary not in caplog.text
    assert context.one_turn_secret_leases == {}


def test_empty_value_and_missing_target_fail_before_provider_call() -> None:
    with pytest.raises(ValueError) as caught:
        ToolResult.one_turn_secret(tool_call_id="call-a", tool_name="native-secret", value="")
    assert "non-empty" in str(caught.value)

    canary = _canary("missing-target")
    loop = AgentLoop()
    context = _context_with_lease(loop, VariableStore(), canary)
    with pytest.raises(OneTurnSecretScopeError):
        loop._arm_one_turn_secrets(context, _request("other-call"))
    assert not context.one_turn_secret_leases["call-a"].redaction_token
    loop._clear_one_turn_secrets(context)


def test_encoded_model_echo_is_scrubbed_but_executor_keeps_arguments() -> None:
    canary = 'v0rn-"quoted\\path\n' + "encoded-7d4c"
    encoded = json.dumps(canary)[1:-1]
    variables = VariableStore()
    token = variables.acquire_scoped_redaction(canary)
    loop = AgentLoop()
    raw_arguments = json.dumps({"text": canary})
    assistant = Message(
        role="assistant",
        content=[TextContent(text="model " + json.dumps({"echo": canary}))],
        tool_calls=[
            ToolCall(
                id="echo-call",
                name="echo",
                arguments={"text": canary},
                raw_arguments=raw_arguments,
            )
        ],
        provider_payload={"native": canary},
    )
    executing, public, _error, failed = loop._finalize_one_turn_secret_assistant(
        assistant,
        None,
        _config(lambda *_args: None, variables),
        [SecretStr(canary)],
    )
    assert not failed
    assert canary not in public.model_dump_json()
    assert encoded not in public.model_dump_json()
    assert public.provider_payload is None
    assert executing.tool_calls[0].arguments == {"text": canary}
    assert executing.tool_calls[0].raw_arguments == raw_arguments
    assert executing.provider_payload is None
    assert variables.release_scoped_redaction(token)


@pytest.mark.asyncio
async def test_full_run_cancellation_cleans_lease_token_and_raw_arguments() -> None:
    canary = _canary("full-cancel")
    variables = VariableStore()
    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    requests: list[ChatRequest] = []
    second_placeholder = asyncio.Event()

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            yield StreamTextDelta(delta=canary)
            await asyncio.Event().wait()

        return generate()

    loop = AgentLoop()
    events: list[Any] = []

    async def collect() -> None:
        starts = 0
        async for event in loop.run(
            [Message.user("go")], context, _config(stream_fn, variables), None
        ):
            events.append(event)
            if isinstance(event, MessageStartEvent):
                starts += 1
                if starts == 2:
                    second_placeholder.set()

    task = asyncio.create_task(collect())
    await asyncio.wait_for(second_placeholder.wait(), 2)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert task.done()
    assert context.one_turn_secret_leases == {}
    assert not context.one_turn_secret_run_active
    assert context.original_call_args == {}
    assert canary not in variables.redaction_values()
    assert canary not in _public_event_json(events)


@pytest.mark.asyncio
async def test_full_run_generator_close_cleans_lease_and_placeholder() -> None:
    canary = _canary("full-close")
    variables = VariableStore()
    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    requests: list[ChatRequest] = []

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            yield StreamTextDelta(delta=canary)
            yield StreamStartEvent(response_id="close-boundary")
            await asyncio.Event().wait()

        return generate()

    run = cast(
        AsyncGenerator[Any, Any],
        AgentLoop().run([Message.user("go")], context, _config(stream_fn, variables), None),
    )
    events: list[Any] = []
    starts = 0
    while True:
        event = await anext(run)
        events.append(event)
        if isinstance(event, MessageStartEvent):
            starts += 1
        if starts == 2 and isinstance(event, ProviderTurnStartEvent):
            break
    await run.aclose()
    assert context.one_turn_secret_leases == {}
    assert not context.one_turn_secret_run_active
    assert context.original_call_args == {}
    assert canary not in variables.redaction_values()
    assert canary not in _public_event_json(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["json", "quote", "quote_plus"])
async def test_noop_sink_cannot_publish_an_encoded_assistant_echo(spelling: str) -> None:
    """S-10 / Q-5: the published carriers use the coverage primitive.

    A disconnected or no-op ``redact_tool_result`` leaves an encoded echo
    unchanged, and idempotence then proves nothing. Under the previous
    raw-substring check these echoes were ACCEPTED into assistant events and
    durable history; the turn must instead fail closed. The canary contains a
    space and a slash on purpose, so ``quote`` and ``quote_plus`` are genuinely
    different spellings.
    """
    canary = 'v0rn-encoded value/with slash-"q"\\path-9f2a'
    if spelling == "json":
        rendered = json.dumps(canary)[1:-1]
    elif spelling == "quote":
        rendered = urllib.parse.quote(canary, safe="")
    else:
        rendered = urllib.parse.quote_plus(canary)
    variables = VariableStore()
    requests: list[ChatRequest] = []

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            yield StreamTextDelta(delta=rendered)
            yield StreamEndEvent(stop_reason="stop")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    config = _config(
        stream_fn,
        variables,
        redact_tool_result=lambda text: text,
        redact_tool_arguments=lambda text: text,
    )
    loop = AgentLoop()
    events = [event async for event in loop.run([Message.user("go")], context, config, None)]
    published = _public_event_json(events)
    from local_operator.harness.loop import _ONE_TURN_SECRET_SCRUB_ERROR

    assert _ONE_TURN_SECRET_SCRUB_ERROR in published, "the turn must refuse, not merely drop"
    assert canary not in published
    assert rendered not in published
    history = json.dumps(
        [
            message.model_dump(mode="json")
            for message in context.messages
            if isinstance(message, Message)
        ]
    )
    assert canary not in history and rendered not in history
    loop._clear_one_turn_secrets(context)
    assert context.one_turn_secret_leases == {}


@pytest.mark.asyncio
async def test_noop_sink_cannot_execute_url_encoded_arguments() -> None:
    """Q-5 argument arm: an encoded-only argument must not run or be replayed."""
    canary = 'v0rn-argument secret/path-"q"-4b71'
    rendered = urllib.parse.quote(canary, safe="")
    variables = VariableStore()
    executed: list[dict[str, Any]] = []
    requests: list[ChatRequest] = []
    payload = json.dumps({"text": rendered})

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            for event in _tool_delta("echo-call", "echo", payload):
                yield event
            yield StreamEndEvent(stop_reason="toolUse")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary), _echo_tool("echo", executed)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    config = _config(
        stream_fn,
        variables,
        redact_tool_result=lambda text: text,
        redact_tool_arguments=lambda text: text,
    )
    loop = AgentLoop()
    events = [event async for event in loop.run([Message.user("go")], context, config, None)]
    assert executed == [], "an unprovable encoded argument must not reach the executor"
    published = _public_event_json(events)
    assert rendered not in published
    later = ChatRequest(
        model=MODEL,
        messages=[message for message in context.messages if isinstance(message, Message)],
    )
    from tests.unit.providers.test_one_turn_secret import _BUILDERS

    assert all(rendered not in json.dumps(build(later)) for build in _BUILDERS.values())
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("field", ["id", "name"])
async def test_model_echoed_tool_identity_never_executes_or_replays(
    field: str, encoded: bool, caplog: pytest.LogCaptureFixture
) -> None:
    canary = 'v0rn-"quoted\\path\n' + field + "-7d4c"
    identity = json.dumps(canary)[1:-1] if encoded else canary
    variables = VariableStore()
    executed: list[dict[str, Any]] = []
    requests: list[ChatRequest] = []

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            call_id = identity if field == "id" else "echo-call"
            name = identity if field == "name" else "echo"
            for event in _tool_delta(call_id, name, json.dumps({"text": "safe"})):
                yield event
            yield StreamEndEvent(stop_reason="toolUse")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary), _echo_tool("echo", executed)],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    with caplog.at_level("WARNING"):
        events = [
            event
            async for event in AgentLoop().run(
                [Message.user("go")], context, _config(stream_fn, variables), None
            )
        ]
    assert executed == []
    assert canary not in _public_event_json(events)
    assert canary not in caplog.text
    assert identity not in caplog.text
    history = json.dumps(
        [
            message.model_dump(mode="json")
            for message in context.messages
            if isinstance(message, Message)
        ]
    )
    assert canary not in history
    later = ChatRequest(
        model=MODEL,
        messages=[message for message in context.messages if isinstance(message, Message)],
    )
    assert not later.has_one_turn_secrets
    from tests.unit.providers.test_one_turn_secret import _BUILDERS

    assert all(canary not in json.dumps(build(later)) for build in _BUILDERS.values())
    assert context.one_turn_secret_leases == {}


@pytest.mark.asyncio
async def test_nameless_call_warning_withholds_provider_identity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = _canary("nameless-id")
    variables = VariableStore()
    loop = AgentLoop()
    context = _context_with_lease(loop, variables, canary)

    def stream_fn(_request: ChatRequest, _signal: AbortSignal | None) -> Any:
        async def generate() -> Any:
            yield StreamToolCallDelta(index=0, id=canary, argument_delta="{}")
            yield StreamEndEvent(stop_reason="toolUse")

        return generate()

    with caplog.at_level("WARNING"):
        events = [
            event async for event in loop._model_turn(context, _config(stream_fn, variables), None)
        ]
    assert canary not in caplog.text
    assert canary not in _public_event_json(events)
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
async def test_terminal_compose_and_duplicate_identity_are_safe() -> None:
    canary = _canary("terminal-compose")
    variables = VariableStore()
    context = LoopContext(tool_context=ToolContext(cwd=".", variables=variables))
    token = variables.acquire_scoped_redaction(canary)
    context.one_turn_secret_leases["call-a"] = _OneTurnSecretLease(
        value=SecretStr(canary), redaction_token=token
    )
    context.tools = [_echo_tool("echo", [])]
    calls = [
        _tool_call("duplicate", "echo", {"text": canary}),
        _tool_call("duplicate", "echo", {"text": "second"}),
    ]
    loop = AgentLoop()
    results: list[ToolResult] = []
    frames = [
        frame
        async for frame in loop._execute_tool_calls(
            calls, context, _config(lambda *_args: None, variables), None, results
        )
    ]
    public = json.dumps(
        [frame.model_dump(mode="json") for frame in frames if isinstance(frame, BaseModel)]
    )
    assert canary not in public
    compose = [frame for frame in frames if isinstance(frame, ToolCallComposeEvent)]
    assert compose and all(frame.argument_bytes == 0 for frame in compose)
    loop._clear_one_turn_secrets(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_secret_active_tool_progress_is_never_public(nested: bool) -> None:
    canary = _canary("live-progress")
    variables = VariableStore()
    requests: list[ChatRequest] = []
    executed: list[dict[str, Any]] = []

    async def progress(tool_call_id: str, args: dict[str, Any], *_args: Any) -> ToolResult:
        executed.append(args)
        on_update = _args[2]
        on_update(
            AgentToolUpdate(
                content=[TextContent(text=args["text"])], details={"echo": args["text"]}
            )
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="echo",
            content=[TextContent(text=args["text"])],
        )

    echo = AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=progress,
    )

    def stream_fn(request: ChatRequest, _signal: AbortSignal | None) -> Any:
        requests.append(request)
        index = len(requests)

        async def generate() -> Any:
            if index == 1:
                for event in _tool_delta("secret-call", "native-secret", "{}"):
                    yield event
                yield StreamEndEvent(stop_reason="toolUse")
                return
            assert request.has_one_turn_secrets
            for event in _tool_delta("echo-call", "echo", json.dumps({"text": canary})):
                yield event
            yield StreamEndEvent(stop_reason="toolUse")

        return generate()

    context = LoopContext(
        system_blocks=[],
        messages=[],
        tools=[_secret_tool("native-secret", canary), echo],
        tool_context=ToolContext(cwd=".", variables=variables),
    )
    config = _config(stream_fn, variables)
    if nested:
        loop = AgentLoop()
        loop._capture_one_turn_secret(
            context,
            _tool_call("secret-call", "native-secret", {}),
            "native-secret",
            _public_tool_result(value=canary),
            allow=True,
        )
        from local_operator.harness.loop import _BatchDone, _ToolDone

        queue: asyncio.Queue[Any] = asyncio.Queue()
        planned = _PlannedCall(
            call=_tool_call("echo-call", "echo", {"text": canary}),
            tool=echo,
            args={"text": canary},
        )
        result = await loop._runner_result(
            planned, context, config, None, queue, allow_one_turn_secret=False
        )
        queued: list[Any] = []
        while not queue.empty():
            queued.append(queue.get_nowait())
        assert canary not in result.model_dump_json()
        assert not any(isinstance(event, ToolExecutionUpdateEvent) for event in queued)
        assert all(
            not isinstance(event, (ToolExecutionUpdateEvent, _ToolDone, _BatchDone))
            or canary not in str(event)
            for event in queued
        )
    else:
        events = [
            event async for event in AgentLoop().run([Message.user("go")], context, config, None)
        ]
        assert executed == [{"text": canary}]
        assert not any(isinstance(event, ToolExecutionUpdateEvent) for event in events)
        assert canary not in _public_event_json(events)
        assert canary not in json.dumps(
            [
                message.model_dump(mode="json")
                for message in context.messages
                if isinstance(message, Message)
            ]
        )
    loop = AgentLoop()
    loop._clear_one_turn_secrets(context)
