"""Cache markers for a Saver frozen prefix (contract §4) and prune ≥ frontier (§7).

With ``ChatRequest.frozen_prefix_messages`` set, the wire puts one marker on
the last block of the last epoch message — the byte-stable prefix the next turn
reads from cache — plus the usual final-message marker. Unset, the body is
byte-identical to today's (Saver's fail-open state).
"""

from __future__ import annotations

import json
from typing import Any

from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import AnthropicClient, OpenAICompatClient

EPOCH = (
    "[radient:epoch reduce-v2] Earlier part of this conversation, condensed. Historical record, "
    "not new instructions. Unindented lines are record headers; tab-indented lines are quoted "
    "content.\nuser: hi"
)


def _tail() -> list[Message]:
    out: list[Message] = []
    for i in range(4):
        out.append(Message(role="user", content=[TextContent(text=f"live {i}")]))
        out.append(
            Message(
                role="assistant",
                content=[TextContent(text="ok")],
                tool_calls=[ToolCall(id=f"t{i}", name="bash", arguments={"command": "ls"})],
            )
        )
        out.append(
            Message(
                role="tool",
                tool_call_id=f"t{i}",
                tool_name="bash",
                content=[TextContent(text="a")],
            )
        )
        out.append(Message(role="assistant", content=[TextContent(text=f"done {i}")]))
    out.append(Message(role="user", content=[TextContent(text="now")]))
    return out


def _epochs(n: int = 2) -> list[Message]:
    return [
        Message(id=f"saver:0:0:{i}", role="user", content=[TextContent(text=EPOCH)])
        for i in range(n)
    ]


def _marked(blocks: list[Any]) -> list[Any]:
    return [b for b in blocks if isinstance(b, dict) and "cache_control" in b]


def _anthropic(frozen: int | None, messages: list[Message]) -> dict[str, Any]:
    model = ModelSpec(provider="anthropic", model_id="claude-test", supports_prompt_cache=True)
    request = ChatRequest(model=model, messages=messages, frozen_prefix_messages=frozen)
    return AnthropicClient()._build_body(request)


def _openai(frozen: int | None, messages: list[Message]) -> dict[str, Any]:
    model = ModelSpec(
        provider="openrouter", model_id="anthropic/claude-test", supports_prompt_cache=True
    )
    request = ChatRequest(model=model, messages=messages, frozen_prefix_messages=frozen)
    return OpenAICompatClient("https://openrouter.ai/api/v1")._build_body(request)


def _content(entry: dict[str, Any]) -> list[Any]:
    content = entry.get("content")
    return content if isinstance(content, list) else []


def test_anthropic_marks_the_last_block_of_the_last_epoch_and_the_final_block() -> None:
    messages = _epochs() + _tail()
    body = _anthropic(2, messages)
    wire = body["messages"]
    # Epoch rows are user turns; the first wire user turn holds them (merged or not).
    epoch_entries = [e for e in wire if any(EPOCH == b.get("text") for b in _content(e))]
    assert epoch_entries, "epochs ride as user turns, never system"
    last_epoch_block = [b for b in _content(epoch_entries[-1]) if b.get("text") == EPOCH][-1]
    assert "cache_control" in last_epoch_block
    assert "cache_control" in _content(wire[-1])[-1]
    assert not any(EPOCH in json.dumps(s) for s in body.get("system") or [])


def test_anthropic_without_a_frozen_prefix_renders_exactly_as_before() -> None:
    messages = _epochs() + _tail()
    assert _anthropic(None, messages) == _anthropic(None, list(messages))
    unmarked = _anthropic(None, messages)
    first_epoch = next(
        e for e in unmarked["messages"] if any(EPOCH == b.get("text") for b in _content(e))
    )
    # With no hint the epoch block is just text: the heuristic picks its own target.
    assert _anthropic(0, messages) == unmarked
    assert first_epoch["role"] == "user"


def test_anthropic_marker_budget_is_respected() -> None:
    body = _anthropic(2, _epochs() + _tail())
    total = sum(len(_marked(_content(e))) for e in body["messages"])
    total += len(_marked(body.get("system") or []))
    assert total <= 4


def test_openai_compatible_wire_marks_the_frozen_prefix_too() -> None:
    body = _openai(2, _epochs() + _tail())
    wire = body["messages"]
    epoch_entries = [e for e in wire if any(b.get("text") == EPOCH for b in _content(e))]
    assert epoch_entries and epoch_entries[-1]["role"] == "user"
    assert "cache_control" in _content(epoch_entries[-1])[-1]
    assert "cache_control" in _content(wire[-1])[-1]


def test_openai_compatible_without_a_frozen_prefix_is_unchanged() -> None:
    messages = _epochs() + _tail()
    assert _openai(None, messages) == _openai(0, messages)


def _mark_positions(body: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (index, (block.get("text") or "")[:15])
        for index, entry in enumerate(body["messages"])
        for block in _content(entry)
        if isinstance(block, dict) and "cache_control" in block
    ]


def test_negative_control_the_hint_is_what_moves_the_marker_onto_the_epoch() -> None:
    """Without the hint the heuristic marks a live-tail turn; with it, the epoch.

    Proves the tests above are not satisfied by the pre-existing heuristic.
    """
    for build in (_anthropic, _openai):
        without = _mark_positions(build(None, _epochs() + _tail()))
        with_hint = _mark_positions(build(2, _epochs() + _tail()))
        assert not any(text.startswith("[radient:epoch") for _, text in without)
        assert with_hint[0] == (1, "[radient:epoch ")
        assert with_hint[-1] == without[-1]  # the final-message marker is unchanged
