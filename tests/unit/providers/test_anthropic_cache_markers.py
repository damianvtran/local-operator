"""``ChatRequest.cache_markers``: where the Anthropic wire puts cache_control.

One-shot auxiliary calls (the Team Bridge screener, the guardrail judge) used
to get the conversation policy, whose only marker landed on the unique last
user turn — a cache write nothing ever read (7-day ledger: team_bridge_screen
1.4% hit, guardrail 0%). ``"auto"`` must stay the exact conversation layout;
``"system"`` caches the static head only; ``"none"`` sends no marker at all.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.bedrock.client import BedrockAnthropicClient
from local_operator.providers.clients import AnthropicClient


def _request(model_id: str = "claude-opus-5", **update: Any) -> ChatRequest:
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id=model_id, context_window=1_000_000),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
        context_tokens_hint=2_000,
    )
    return request.model_copy(update=update)


def _marker_paths(node: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        if "cache_control" in node:
            found.append(path)
        for key, value in node.items():
            found.extend(_marker_paths(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_marker_paths(value, f"{path}[{index}]"))
    return found


CLIENTS = [
    pytest.param(lambda: AnthropicClient(), "claude-opus-5", id="anthropic"),
    pytest.param(
        lambda: BedrockAnthropicClient(region="us-east-1"),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        id="bedrock-inherits",
    ),
]


@pytest.mark.parametrize("oauth", [False, True])
def test_auto_is_the_default_and_keeps_the_conversation_layout(oauth: bool) -> None:
    client = AnthropicClient()
    default = client._build_body(_request(), oauth=oauth)
    explicit = client._build_body(_request(cache_markers="auto"), oauth=oauth)
    assert default == explicit
    paths = _marker_paths(default)
    assert any(path.startswith(".system") for path in paths)
    assert any(path.startswith(".messages") for path in paths), "message breakpoints stay on"
    assert ChatRequest.model_fields["cache_markers"].default == "auto"


@pytest.mark.parametrize("make_client,model_id", CLIENTS)
@pytest.mark.parametrize("oauth", [False, True])
def test_none_sends_no_cache_control_anywhere(make_client, model_id, oauth: bool) -> None:
    body = make_client()._build_body(_request(model_id, cache_markers="none"), oauth=oauth)
    assert _marker_paths(body) == []
    # The content is the same; only the markers are gone.
    auto = make_client()._build_body(_request(model_id), oauth=oauth)
    assert [b["text"] for b in body["system"]] == [b["text"] for b in auto["system"]]


@pytest.mark.parametrize("make_client,model_id", CLIENTS)
@pytest.mark.parametrize("oauth", [False, True])
def test_system_puts_exactly_one_marker_on_the_last_system_block(
    make_client, model_id, oauth: bool
) -> None:
    body = make_client()._build_body(_request(model_id, cache_markers="system"), oauth=oauth)
    last = len(body["system"]) - 1
    assert _marker_paths(body) == [f".system[{last}]"]


def test_system_with_no_system_blocks_sends_no_marker() -> None:
    request = _request(cache_markers="system").model_copy(update={"system_blocks": []})
    body = AnthropicClient()._build_body(request, oauth=False)
    assert _marker_paths(body) == []


def test_cache_markers_never_reach_the_serialized_request() -> None:
    """Transport policy, not conversation state: excluded from dumps."""
    assert "cache_markers" not in _request(cache_markers="none").model_dump()


@pytest.mark.asyncio
async def test_the_guardrail_judge_request_caches_its_system_prompt_only() -> None:
    """``Session._guard_judge`` stamps ``"system"``: the static judge prompt is
    the reusable head; the brief after it is unique to each check."""
    from types import SimpleNamespace

    from local_operator.guardrails.judge import JUDGE_SYSTEM_PROMPT
    from local_operator.session.session import Session

    spec = ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000)
    captured: list[ChatRequest] = []

    async def drain(request: ChatRequest) -> str:
        captured.append(request)
        return "ALLOW"

    fake = SimpleNamespace(
        effective_model=spec,
        _lowest_effort=lambda model: model,
        _drain_errand=drain,
        ERRAND_MAX_TOKENS=Session.ERRAND_MAX_TOKENS,
    )
    host = cast(Any, fake)
    fake._errand_request = lambda model, system, prompt: Session._errand_request(
        host, model, system, prompt
    )
    verdict = await Session._guard_judge(host, JUDGE_SYSTEM_PROMPT, "brief")
    assert verdict == "ALLOW"
    (request,) = captured
    assert request.purpose == "guardrail" and request.cache_markers == "system"
    body = AnthropicClient()._build_body(request, oauth=False)
    assert _marker_paths(body) == [".system[0]"]


def _radient_request(**update: Any) -> ChatRequest:
    request = ChatRequest(
        model=ModelSpec(
            provider="radient",
            model_id="anthropic/claude-sonnet-5.5",
            context_window=200_000,
            supports_prompt_cache=True,
        ),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
    )
    return request.model_copy(update=update)


def test_openai_compat_cache_markers_none_emits_zero_markers() -> None:
    """Radient/OpenRouter screening calls with ``cache_markers="none"`` must
    emit zero ``cache_control`` markers so unique prompts do not pay 1.25x
    cache-write."""
    from local_operator.providers.clients import OpenAICompatClient

    client = OpenAICompatClient("https://api.radienthq.com/v1")
    body = client._build_body(_radient_request(cache_markers="none"))
    assert _marker_paths(body) == []
    # Content stays intact; only cache_control markers are omitted.
    system_texts = [m["content"] for m in body["messages"] if m["role"] == "system"]
    assert system_texts == ["instructions", "inventory", "skills", "env"]


def test_openai_compat_cache_markers_system_marks_last_system_block_only() -> None:
    """``cache_markers="system"`` on OpenAICompatClient marks only the last
    system message (matching AnthropicClient, including 1-block judge prompts)
    and skips conversation message markers."""
    from local_operator.providers.clients import OpenAICompatClient

    client = OpenAICompatClient("https://api.radienthq.com/v1")
    body = client._build_body(_radient_request(cache_markers="system"))
    assert _marker_paths(body) == [".messages[3].content[0]"]
    assert body["messages"][3]["role"] == "system"

    # Single-block static head (e.g. guardrail judge) still gets its system marker.
    single = client._build_body(
        _radient_request(system_blocks=["static judge prompt"], cache_markers="system")
    )
    assert _marker_paths(single) == [".messages[0].content[0]"]
    assert single["messages"][0]["role"] == "system"

    # Empty system_blocks emits no markers at all.
    empty = client._build_body(_radient_request(system_blocks=[], cache_markers="system"))
    assert _marker_paths(empty) == []


def test_openai_compat_cache_markers_auto_marks_system_and_conversation_within_cap() -> None:
    """``cache_markers="auto"`` (default) keeps both system and conversation
    markers while capping the total at ``MAX_CACHE_BREAKPOINTS`` (4)."""
    from local_operator.providers.clients import OpenAICompatClient

    client = OpenAICompatClient("https://api.radienthq.com/v1")
    default = client._build_body(_radient_request())
    explicit = client._build_body(_radient_request(cache_markers="auto"))
    assert default == explicit
    paths = _marker_paths(default)
    sys_marked = [
        i
        for i, m in enumerate(default["messages"])
        if m["role"] == "system" and "cache_control" in str(m.get("content"))
    ]
    conv_marked = [
        i
        for i, m in enumerate(default["messages"])
        if m["role"] != "system" and "cache_control" in str(m.get("content"))
    ]
    assert sys_marked and conv_marked
    assert len(paths) <= OpenAICompatClient.MAX_CACHE_BREAKPOINTS

    # Even with many system blocks, total markers never exceed 4.
    many = client._build_body(
        _radient_request(system_blocks=[f"block-{i}" for i in range(8)], cache_markers="auto")
    )
    assert len(_marker_paths(many)) <= OpenAICompatClient.MAX_CACHE_BREAKPOINTS
