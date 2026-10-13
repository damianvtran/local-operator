"""Google's final-user requirement is a wire invariant, not a session repair."""

from __future__ import annotations

import copy

import pytest

from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import (
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    ToolCall,
)
from local_operator.providers.clients import GoogleClient, OpenAICompatClient


@pytest.mark.parametrize(
    "provider,model_id", [("google", "gemini-4-argon"), ("openrouter", "google/gemini-4-argon")]
)
@pytest.mark.parametrize("tail", ["assistant", "empty", "nameless", "none", "system"])
def test_google_rendered_request_has_user_boundary(provider: str, model_id: str, tail: str) -> None:
    messages = (
        [] if tail in ("none", "system") else [Message.user("Say OK."), Message.assistant("OK.")]
    )
    if tail == "empty":
        messages.append(Message.assistant("  "))
    if tail == "nameless":
        messages.extend(
            [
                Message(role="assistant", tool_calls=[ToolCall(id="bad", name="", arguments={})]),
                Message(role="tool", tool_call_id="bad", tool_name="", content=[]),
            ]
        )
    request = ChatRequest(
        model=ModelSpec(provider=provider, model_id=model_id),
        messages=messages,
        system_blocks=["Harness context."] if tail == "system" else [],
    )
    original = copy.deepcopy(request)
    client = (
        GoogleClient()
        if provider == "google"
        else OpenAICompatClient("https://openrouter.ai/api/v1")
    )
    body = client._build_body(request)
    wire = body["contents" if provider == "google" else "messages"]
    assert wire[-1]["role"] == "user"
    text = wire[-1]["parts"][0]["text"] if provider == "google" else wire[-1]["content"]
    assert "harness" in text.lower() and "not" in text and "operator" in text
    assert request == original


@pytest.mark.parametrize(
    "provider,model_id", [("google", "gemini-4-argon"), ("openrouter", "google/gemini-4-argon")]
)
@pytest.mark.parametrize("tool_result", [False, True])
def test_google_valid_user_or_tool_boundary_is_unchanged(
    provider: str, model_id: str, tool_result: bool
) -> None:
    messages = [Message.user("Say OK.")]
    if tool_result:
        messages.extend(
            [
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="safe", name="safe_probe", arguments={})],
                ),
                Message(
                    role="tool",
                    tool_call_id="safe",
                    tool_name="safe_probe",
                    content=Message.user("OK").content,
                ),
            ]
        )
    body = (
        GoogleClient()
        if provider == "google"
        else OpenAICompatClient("https://openrouter.ai/api/v1")
    )._build_body(
        ChatRequest(model=ModelSpec(provider=provider, model_id=model_id), messages=messages)
    )
    wire = body["contents" if provider == "google" else "messages"]
    assert len(wire) == len(messages)
    assert wire[-1]["role"] == ("tool" if provider == "openrouter" and tool_result else "user")


@pytest.mark.parametrize("assistant_tail", [False, True])
def test_compaction_neighbor_and_model_switch_normalize_only_new_route(
    assistant_tail: bool,
) -> None:
    marker = CustomMessage(
        custom_type="compaction_summary", details={"summary": "Synthetic pending task: say OK."}
    )
    history = _default_convert_to_llm([marker])
    if assistant_tail:
        history.append(Message.assistant("OK."))
    original = ChatRequest(model=ModelSpec(provider="radient", model_id="auto"), messages=history)
    client = OpenAICompatClient("https://synthetic.invalid/v1")
    before = client._build_body(original)
    switched = original.model_copy(
        update={"model": ModelSpec(provider="openrouter", model_id="google/gemini-4-argon")}
    )
    after = client._build_body(switched)
    assert after["messages"][-1]["role"] == "user"
    assert len(after["messages"]) == len(before["messages"]) + int(assistant_tail)
    assert after["messages"][: len(before["messages"])] == before["messages"]
    assert original.model.provider == "radient"
    assert len(history) == 1 + int(assistant_tail)


@pytest.mark.parametrize(
    "provider,model_id",
    [("openai", "gpt-4o"), ("openrouter", "anthropic/claude-sonnet-4"), ("radient", "auto")],
)
def test_other_providers_keep_assistant_tail(provider: str, model_id: str) -> None:
    body = OpenAICompatClient("https://example.invalid/v1")._build_body(
        ChatRequest(
            model=ModelSpec(provider=provider, model_id=model_id),
            messages=[Message.user("Say OK."), Message.assistant("OK.")],
        )
    )
    assert body["messages"][-1] == {"role": "assistant", "content": "OK."}
