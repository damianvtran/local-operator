"""Cross-session cache breakpoint on the session-agnostic system head.

WHY: the frozen-epoch policy marked only ``system[-1]``, a prefix that ends
AFTER the env block (``Today is <date>``, working directory), so a new session
or subagent in another cwd or on another day could never read the identical
tools+persona bytes another session had just written. A second marker now sits
on the last block before the env block (``prompts_api.STABLE_HEAD_BLOCK_COUNT``)
and is the first marker shed when the 4-marker cap binds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamTextDelta,
)
from local_operator.prompts_api import STABLE_HEAD_BLOCK_COUNT, build_system_blocks
from local_operator.providers.bedrock.client import BedrockAnthropicClient
from local_operator.providers.clients import AnthropicClient, OpenAICompatClient
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

_SYS = ["instructions", "inventory", "env", "tail"]
_PREAMBLE = "Shared rubric.\n" + ("context line " * 300) + "\n---\n"


def _spec(provider: str = "anthropic", model_id: str = "claude-opus-5") -> ModelSpec:
    return ModelSpec(provider=provider, model_id=model_id, context_window=1_000_000)


def _marker_paths(body: dict[str, Any]) -> list[str]:
    paths = [
        f".system[{i}]" for i, b in enumerate(body.get("system") or []) if "cache_control" in b
    ]
    for m_index, message in enumerate(body["messages"]):
        content = message.get("content")
        if isinstance(content, list):
            for c_index, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    paths.append(f".messages[{m_index}].content[{c_index}]")
    return paths


def _all_markers(body: dict[str, Any]) -> list[dict[str, Any]]:
    out = [b["cache_control"] for b in body.get("system") or [] if "cache_control" in b]
    for message in body["messages"]:
        if isinstance(message.get("content"), list):
            out += [b["cache_control"] for b in message["content"] if "cache_control" in b]
    return out


def _request(**update: Any) -> ChatRequest:
    base: dict[str, Any] = {
        "model": _spec(),
        "system_blocks": list(_SYS),
        "system_blocks_frozen": True,
        "system_stable_head_blocks": STABLE_HEAD_BLOCK_COUNT,
        "messages": [Message.user("hi")],
    }
    base.update(update)
    return ChatRequest(**base)


# ---------------------------------------------------------------------------
# Marker positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "head", "expected"),
    [
        (4, 2, [1, 3]),
        (5, 2, [1, 4]),  # output contract appended after the tail
        (4, 0, [3]),  # unknown head: historical single tail marker
        (2, 2, [1]),  # head IS the last block: one marker, never a duplicate
        (1, 2, [0]),
        (0, 2, []),
    ],
)
def test_frozen_system_blocks_mark_head_and_tail(
    count: int, head: int, expected: list[int]
) -> None:
    rendered = AnthropicClient._system_blocks(
        [f"b{i}" for i in range(count)], frozen=True, stable_head=head
    )
    assert [i for i, e in enumerate(rendered) if "cache_control" in e] == expected


def test_non_frozen_ignores_stable_head_and_keeps_legacy_layout() -> None:
    # Legacy non-frozen callers already mark every block before the last two,
    # which on the four-block layout IS the head (0, 1) and never the env block.
    body = AnthropicClient()._build_body(_request(system_blocks_frozen=False))
    assert _marker_paths(body)[:2] == [".system[0]", ".system[1]"]
    assert "cache_control" not in body["system"][2]


@pytest.mark.parametrize("oauth", [False, True])
def test_build_body_head_marker_sits_before_env_block(oauth: bool) -> None:
    body = AnthropicClient()._build_body(_request(), oauth=oauth)
    texts = [b["text"] for b in body["system"]]
    env = texts.index("env")
    marked = [i for i, b in enumerate(body["system"]) if "cache_control" in b]
    # Head marker on the inventory (the block right before env), tail on the last.
    assert marked == [env - 1, len(texts) - 1]
    if oauth:
        # The constant identity block is part of the shared head, unmarked itself.
        assert texts[0] == AnthropicClient.CLAUDE_CODE_IDENTITY
        assert marked[0] == 2


def test_cache_markers_system_and_none_are_untouched() -> None:
    client = AnthropicClient()
    assert _marker_paths(client._build_body(_request(cache_markers="system"))) == [".system[3]"]
    assert _marker_paths(client._build_body(_request(cache_markers="none"))) == []


@pytest.mark.parametrize("ttl_min", [0, 1])
def test_every_marker_carries_one_ttl(ttl_min: int) -> None:
    # A 1h marker may not follow a 5m one: head, tail and message markers share it.
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=ttl_min)
    body = client._build_body(_request(context_tokens_hint=50_000))
    markers = _all_markers(body)
    assert len(markers) >= 3
    expected = {"type": "ephemeral", "ttl": "1h"} if ttl_min else {"type": "ephemeral"}
    assert all(marker == expected for marker in markers)


# ---------------------------------------------------------------------------
# Cap and drop order
# ---------------------------------------------------------------------------


def test_two_message_markers_keep_both_system_markers() -> None:
    history = [Message.user("one"), Message.assistant("ok"), Message.user("two")]
    body = AnthropicClient()._build_body(_request(messages=history))
    assert _marker_paths(body) == [
        ".system[1]",
        ".system[3]",
        ".messages[0].content[0]",
        ".messages[2].content[0]",
    ]


@pytest.mark.parametrize(
    ("oauth", "head", "tail"),
    [
        (False, ".system[1]", ".system[3]"),
        # OAuth prepends the identity block, shifting head and tail by one.
        (True, ".system[2]", ".system[4]"),
    ],
)
def test_cap_drops_head_marker_before_tail_and_message_markers(
    oauth: bool, head: str, tail: str
) -> None:
    # Static preamble split + settled image + rolling tail = 3 message markers,
    # leaving ONE system slot: the tail marker must win it, not the head.
    history = [
        Message.user(_PREAMBLE + "task"),
        Message.assistant("step 1"),
        Message.user("shot", [ImageContent(data="aW1n", mime_type="image/png")]),
        Message.assistant("step 2"),
        Message.user("latest"),
    ]
    body = AnthropicClient()._build_body(_request(messages=history), oauth=oauth)
    paths = _marker_paths(body)
    assert len(paths) == AnthropicClient.MAX_CACHE_BREAKPOINTS
    assert tail in paths
    assert head not in paths
    assert sum(p.startswith(".messages") for p in paths) == 3


def test_prev_user_fallback_outranks_head_marker() -> None:
    # Saver frozen_block absent, no frame: tail + prev_user; with the static
    # preamble that is three message markers and the head is the one shed.
    history = [
        Message.user(_PREAMBLE + "task"),
        Message.assistant("a"),
        Message.user("two"),
        Message.assistant("b"),
        Message.user("three"),
    ]
    body = AnthropicClient()._build_body(_request(messages=history))
    paths = _marker_paths(body)
    assert len(paths) <= AnthropicClient.MAX_CACHE_BREAKPOINTS
    assert ".system[3]" in paths
    assert ".messages[2].content[0]" in paths  # prev_user kept
    assert ".system[1]" not in paths


def test_bedrock_inherits_head_marker_and_ttl_downgrade() -> None:
    client = BedrockAnthropicClient(region="us-east-1", cache_ttl_1h_min_context_tokens=1)
    unknown = _request(
        model=_spec("bedrock", "us.anthropic.claude-unknown-v1:0"), context_tokens_hint=50_000
    )
    body = client._build_body(unknown)
    assert [i for i, b in enumerate(body["system"]) if "cache_control" in b] == [1, 3]
    # A model without 1h support downgrades EVERY marker, head included.
    assert {tuple(sorted(m.items())) for m in _all_markers(body)} == {(("type", "ephemeral"),)}


def test_openai_compat_auto_markers_already_end_before_env_block() -> None:
    # The gateway wire does not use the frozen policy: "auto" marks every block
    # but the last two, which on the four-block layout is the shared head.
    client = OpenAICompatClient(
        "https://gateway.example.invalid/v1", http_client=httpx.AsyncClient()
    )
    request = _request(
        model=ModelSpec(
            provider="openrouter",
            model_id="anthropic/claude-opus-5",
            context_window=1_000_000,
            supports_prompt_cache=True,
        )
    )
    rows = client._system_messages(request)
    marked = [i for i, row in enumerate(rows) if isinstance(row["content"], list)]
    assert marked == [0, 1]


# ---------------------------------------------------------------------------
# Structural seam: the builder's head is session-agnostic
# ---------------------------------------------------------------------------


def test_build_system_blocks_head_is_identical_across_cwd_and_date() -> None:
    a = build_system_blocks([], "", "Working directory: /a", "2026-11-24")
    b = build_system_blocks([], "", "Working directory: /b", "2026-11-25")
    assert a[:STABLE_HEAD_BLOCK_COUNT] == b[:STABLE_HEAD_BLOCK_COUNT]
    # The first volatile block is exactly the one after the head.
    assert a[STABLE_HEAD_BLOCK_COUNT] != b[STABLE_HEAD_BLOCK_COUNT]
    assert a[STABLE_HEAD_BLOCK_COUNT].startswith("Today is 2026-11-24.")


class _CapturingStream:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def _gen():
            yield StreamTextDelta(delta="done.")

        return _gen()


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [True, False])
async def test_session_stamps_stable_head_only_when_provider_declares_it(
    tmp_path: Path, declared: bool
) -> None:
    stream = _CapturingStream()

    def provider() -> list[str]:
        return list(_SYS)

    setattr(provider, "append_only_state", True)
    if declared:
        setattr(provider, "stable_head_blocks", STABLE_HEAD_BLOCK_COUNT)
    session = Session(
        model=_spec("bedrock", "us.anthropic.claude-sonnet-4-6"),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=provider,
    )
    await session.prompt("hello")
    await session.complete_aside([Message.user("side")])
    expected = STABLE_HEAD_BLOCK_COUNT if declared else 0
    assert [r.system_stable_head_blocks for r in stream.requests] == [expected, expected]
    body = BedrockAnthropicClient(region="us-east-1")._build_body(stream.requests[0])
    marked = [i for i, b in enumerate(body["system"]) if "cache_control" in b]
    assert marked == ([1, 3] if declared else [3])


@pytest.mark.asyncio
async def test_unfrozen_provider_reports_no_stable_head(tmp_path: Path) -> None:
    stream = _CapturingStream()

    def provider() -> list[str]:
        return list(_SYS)

    # Declares a head but not the persisted-prefix protocol: no byte-stable
    # promise, so the wire must not be told about a head.
    setattr(provider, "stable_head_blocks", STABLE_HEAD_BLOCK_COUNT)
    session = Session(
        model=_spec(),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "plain"),
        system_blocks_provider=provider,
    )
    await session.prompt("hello")
    assert stream.requests[0].system_stable_head_blocks == 0
    assert "system_stable_head_blocks" not in stream.requests[0].model_dump()
