"""Freeze-aware Anthropic/Bedrock system & initial-message static-prefix caching.

Verifies:
1. ``ChatRequest.system_blocks_frozen`` transport metadata and real ``Session``
   -> ``AgentLoop`` plumbing (``append_only_state`` frozen epochs vs standalone
   dynamic callers).
2. ``AnthropicClient._system_blocks(..., frozen=True/False)`` across block
   counts, ``cache_markers`` modes, OAuth identity prepend, and 5m/1h TTLs.
3. ``_split_reusable_prefix`` fenced-code-block tracking, exact ``\\n---\\n``
   boundary matching, minimum length / non-empty suffix guards, and exact byte
   preservation.
4. Render-only wire splitting on the first original persistent user message,
   turn-to-turn byte stability, multi-block preservation, coalescing/ephemeral
   edge cases, centralized 4-breakpoint priority budgeting, and Bedrock
   inheritance (``BedrockAnthropicClient`` + ``transform_body``).
"""

from __future__ import annotations

import base64
import copy
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    DocumentContent,
    ImageContent,
    Message,
    ModelSpec,
    StreamTextDelta,
    TextContent,
)
from local_operator.providers.bedrock.client import (
    BedrockAnthropicClient,
    transform_body,
)
from local_operator.providers.clients import (
    _REUSABLE_PREFIX_MIN_CHARS,
    AnthropicClient,
    _split_reusable_prefix,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

_PREAMBLE = "Shared rubric and guidelines.\n" + ("context line " * 300) + "\n---\n"
assert len(_PREAMBLE) >= _REUSABLE_PREFIX_MIN_CHARS
_TASK = "Solve item 42 concisely."
_LONG_PROMPT = _PREAMBLE + _TASK


def _anthropic_spec(
    model_id: str = "claude-opus-5",
    *,
    provider: str = "anthropic",
    context_window: int = 1_000_000,
) -> ModelSpec:
    return ModelSpec(provider=provider, model_id=model_id, context_window=context_window)


def _bedrock_spec(model_id: str = "us.anthropic.claude-sonnet-4-6") -> ModelSpec:
    return ModelSpec(provider="bedrock", model_id=model_id, context_window=1_000_000)


def _marker_entries(node: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        if "cache_control" in node:
            found.append((path, node["cache_control"]))
        # Traverse system before messages for deterministic prefix ordering.
        keys = sorted(
            node.keys(), key=lambda k: (0 if k == "system" else 1 if k == "messages" else 2, k)
        )
        for key in keys:
            found.extend(_marker_entries(node[key], f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_marker_entries(value, f"{path}[{index}]"))
    return found


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(v) for v in value]
    return value


def _tiny_image(data: str = "aW1n") -> ImageContent:
    return ImageContent(data=data, mime_type="image/png")


# ---------------------------------------------------------------------------
# 1. _system_blocks: frozen vs legacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 5, 9])
def test_system_blocks_frozen_marks_only_trailing_block(count: int) -> None:
    blocks = [f"sys-{i}" for i in range(count)]
    rendered = AnthropicClient._system_blocks(blocks, frozen=True)
    assert [entry["text"] for entry in rendered] == blocks
    marked = [i for i, entry in enumerate(rendered) if "cache_control" in entry]
    if count == 0:
        assert marked == []
    else:
        assert marked == [count - 1]
        assert rendered[-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    ("count", "expected_marked"),
    [
        (0, []),
        (1, []),
        (2, []),
        (5, [0, 1, 2]),
        (9, [0, 1, 2, 3]),
    ],
)
def test_system_blocks_non_frozen_preserves_legacy_stable_head(
    count: int, expected_marked: list[int]
) -> None:
    blocks = [f"sys-{i}" for i in range(count)]
    rendered = AnthropicClient._system_blocks(blocks, frozen=False)
    marked = [i for i, entry in enumerate(rendered) if "cache_control" in entry]
    assert marked == expected_marked


@pytest.mark.parametrize("ttl", [None, "1h"])
def test_system_blocks_frozen_propagates_ttl(ttl: str | None) -> None:
    rendered = AnthropicClient._system_blocks(["a", "b", "c", "d", "e"], ttl=ttl, frozen=True)
    expected_cc = {"type": "ephemeral", "ttl": "1h"} if ttl else {"type": "ephemeral"}
    assert [i for i, e in enumerate(rendered) if "cache_control" in e] == [4]
    assert rendered[4]["cache_control"] == expected_cc


@pytest.mark.parametrize("oauth", [False, True])
def test_build_body_frozen_respects_cache_markers_and_oauth(oauth: bool) -> None:
    client = AnthropicClient()
    sys_blocks = ["instructions", "inventory", "skills", "env", "output-contract"]

    # cache_markers="auto" + frozen=True -> single marker on final system block
    # (even after OAuth identity prepend)
    auto_body = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[Message.user("hi")],
        ),
        oauth=oauth,
    )
    sys_marked = [i for i, b in enumerate(auto_body["system"]) if "cache_control" in b]
    assert sys_marked == [len(auto_body["system"]) - 1]
    if oauth:
        assert auto_body["system"][0]["text"] == AnthropicClient.CLAUDE_CODE_IDENTITY
        assert "cache_control" not in auto_body["system"][0]

    # cache_markers="none" -> zero markers anywhere, even with frozen=True + splittable prompt
    none_body = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            cache_markers="none",
            messages=[Message.user(_LONG_PROMPT)],
        ),
        oauth=oauth,
    )
    assert _marker_entries(none_body) == []
    # Also does not split the user text when message caching is disabled.
    assert len(none_body["messages"][0]["content"]) == 1
    assert none_body["messages"][0]["content"][0]["text"] == _LONG_PROMPT

    # cache_markers="system" -> single marker on system[-1], zero message markers
    system_only = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            cache_markers="system",
            messages=[Message.user(_LONG_PROMPT)],
        ),
        oauth=oauth,
    )
    last_sys = len(system_only["system"]) - 1
    assert [path for path, _ in _marker_entries(system_only)] == [f".system[{last_sys}]"]
    assert len(system_only["messages"][0]["content"]) == 1


def test_system_blocks_frozen_field_defaults_false_and_excluded_from_dump() -> None:
    req = ChatRequest(model=_anthropic_spec(), messages=[Message.user("hi")])
    assert req.system_blocks_frozen is False
    assert "system_blocks_frozen" not in req.model_dump()
    frozen_req = req.model_copy(update={"system_blocks_frozen": True})
    assert frozen_req.system_blocks_frozen is True
    assert "system_blocks_frozen" not in frozen_req.model_dump()


# ---------------------------------------------------------------------------
# 2. _split_reusable_prefix
# ---------------------------------------------------------------------------


def test_split_reusable_prefix_basic_and_exact_bytes() -> None:
    res = _split_reusable_prefix(_LONG_PROMPT)
    assert res is not None
    prefix, suffix = res
    assert prefix == _PREAMBLE
    assert suffix == _TASK
    assert prefix + suffix == _LONG_PROMPT
    assert prefix.endswith("\n---\n")


def test_split_reusable_prefix_no_delimiter_and_edge_positions() -> None:
    assert _split_reusable_prefix("x" * 5000) is None
    # Delimiter at very start (no preceding '\n') or very end (no trailing '\n')
    assert _split_reusable_prefix("---\n" + "x" * 5000) is None
    assert _split_reusable_prefix("x" * 5000 + "\n---") is None
    # Trailing spaces on --- line do not match exact '\n---\n'
    assert _split_reusable_prefix("x" * 4000 + "\n--- \n" + "task") is None


def test_split_reusable_prefix_enforces_min_length_and_non_empty_suffix() -> None:
    # 3594 + len("\n---\n") (5) = 3599 < 3600 -> rejected
    too_short = ("a" * 3594) + "\n---\n" + "task"
    assert _split_reusable_prefix(too_short) is None

    # 3595 + len("\n---\n") (5) = 3600 == 3600 -> accepted
    exact_min = ("a" * 3595) + "\n---\n" + "task"
    split = _split_reusable_prefix(exact_min)
    assert split is not None
    assert len(split[0]) == 3600
    assert split[1] == "task"
    assert split[0] + split[1] == exact_min

    # Empty / whitespace-only suffix rejected
    assert _split_reusable_prefix("a" * 4000 + "\n---\n") is None
    assert _split_reusable_prefix("a" * 4000 + "\n---\n   \n\t  ") is None


def test_split_reusable_prefix_ignores_fenced_code_blocks() -> None:
    # Backtick fence containing ---
    backtick_only = ("a" * 4000) + "\n```yaml\n---\nkey: val\n```\nfinal task"
    assert _split_reusable_prefix(backtick_only) is None

    # Tilde fence containing --- (closed by longer tilde run)
    tilde_only = ("a" * 4000) + "\n~~~markdown\n---\n~~~~\nfinal task"
    assert _split_reusable_prefix(tilde_only) is None

    # 4-backtick outer fence containing a 3-backtick fence and ---
    nested_fence = ("a" * 4000) + "\n````md\n```\n---\n```\n---\n````\nfinal task"
    assert _split_reusable_prefix(nested_fence) is None

    # Unclosed fence swallows trailing ---
    unclosed = ("a" * 4000) + "\n```text\n---\ntask"
    assert _split_reusable_prefix(unclosed) is None

    # Backtick fence inside tilde fence does NOT prematurely close the tilde fence
    mixed = ("a" * 4000) + "\n~~~\n```\n---\n~~~\ntask"
    assert _split_reusable_prefix(mixed) is None

    # Inline triple-backtick code span does NOT open a fence
    inline_span = ("a" * 4000) + "\n```inline```\n---\nreal task"
    split = _split_reusable_prefix(inline_span)
    assert split is not None
    assert split[1] == "real task"
    assert split[0] + split[1] == inline_span


def test_split_reusable_prefix_picks_last_delimiter_outside_fences() -> None:
    text = (
        ("part1 " * 400)
        + "\n---\n"
        + ("part2 " * 400)
        + "\n---\n"
        + "```yaml\n---\n```\n"
        + "actual per-item task"
    )
    split = _split_reusable_prefix(text)
    assert split is not None
    prefix, suffix = split
    assert prefix + suffix == text
    assert suffix == "```yaml\n---\n```\nactual per-item task"


# ---------------------------------------------------------------------------
# 3. Render-only split & 4-breakpoint centralized budgeting
# ---------------------------------------------------------------------------


def test_first_user_split_is_render_only_and_preserves_multiblock_order() -> None:
    img = _tiny_image("cG5n")
    # Non-PDF document renders through fallback_document_text (a {"type": "text"} dict)
    # and MUST NOT be split even if it contains '\n---\n'.
    doc_text = ("doc " * 1200) + "\n---\n" + "after doc hr"
    doc = DocumentContent(
        data=base64.b64encode(doc_text.encode("utf-8")).decode("ascii"),
        mime_type="text/plain",
        filename="notes.txt",
    )
    msg = Message(
        role="user",
        content=[img, TextContent(text=_LONG_PROMPT), doc, TextContent(text="trailing note")],
    )
    snapshot = copy.deepcopy(msg)

    body = AnthropicClient()._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=["instructions", "inventory", "skills", "env"],
            system_blocks_frozen=True,
            messages=[msg],
        )
    )
    # Stored Message is never mutated.
    assert msg == snapshot
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text", "text", "text", "text"]
    assert content[1]["text"] == _PREAMBLE
    assert content[2]["text"] == _TASK
    assert "after doc hr" in content[3]["text"]
    assert content[4]["text"] == "trailing note"
    # Breakpoints: system[-1], static_prefix (content[1]), tail (content[4])
    assert "cache_control" in content[1]
    assert "cache_control" not in content[2]
    assert "cache_control" in content[4]
    assert len(_marker_entries(body)) == 3

    # Document-only message (no ordinary TextContent) is NEVER split.
    doc_only_msg = Message(role="user", content=[doc])
    doc_body = AnthropicClient()._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=["instructions", "env"],
            system_blocks_frozen=True,
            messages=[doc_only_msg],
        )
    )
    assert len(doc_body["messages"][0]["content"]) == 1


def test_split_blocks_are_byte_identical_across_turn_1_and_turn_2_plus() -> None:
    client = AnthropicClient()
    sys_blocks = ["instructions", "inventory", "skills", "env", "output-contract"]
    user0 = Message.user(_LONG_PROMPT)

    turn1 = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[user0],
        )
    )
    turn2 = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[user0, Message.assistant("working"), Message.user("follow-up")],
        )
    )
    assert _strip_cache_control(turn1["system"]) == _strip_cache_control(turn2["system"])
    assert _strip_cache_control(turn1["messages"][0]) == _strip_cache_control(turn2["messages"][0])
    # Turn 1: system[4], messages[0].content[0] (static_prefix), messages[0].content[1] (tail) = 3
    assert [p for p, _ in _marker_entries(turn1)] == [
        ".system[4]",
        ".messages[0].content[0]",
        ".messages[0].content[1]",
    ]
    # Turn 2: system[4], static_prefix, prev_user (messages[0].content[1]), tail (messages[2]) = 4
    assert [p for p, _ in _marker_entries(turn2)] == [
        ".system[4]",
        ".messages[0].content[0]",
        ".messages[0].content[1]",
        ".messages[2].content[0]",
    ]


def test_target_selection_handles_leading_and_coalesced_tool_results_and_ephemerals() -> None:
    client = AnthropicClient()
    sys_blocks = ["instructions", "inventory", "skills", "env"]

    # Case A: Leading tool result before first persistent user message (e.g. spliced history).
    leading_tool_body = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[
                Message(role="tool", tool_call_id="t0", content=[TextContent(text="prior result")]),
                Message.user(_LONG_PROMPT),
                Message.assistant("ack"),
                Message.user("next"),
            ],
        )
    )
    # messages[0] is the tool_result user message; messages[1] is the first original user message!
    assert leading_tool_body["messages"][0]["content"][0]["type"] == "tool_result"
    assert leading_tool_body["messages"][1]["content"][0]["text"] == _PREAMBLE
    assert leading_tool_body["messages"][1]["content"][1]["text"] == _TASK
    assert "cache_control" in leading_tool_body["messages"][1]["content"][0]
    assert len(_marker_entries(leading_tool_body)) <= AnthropicClient.MAX_CACHE_BREAKPOINTS

    # Case B: Tool result coalesced onto the first user message still marks static_prefix_block.
    coalesced_body = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[
                Message.user(_LONG_PROMPT),
                Message(role="tool", tool_call_id="t1", content=[TextContent(text="tool out")]),
            ],
        )
    )
    assert len(coalesced_body["messages"]) == 1
    blocks = coalesced_body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text", "tool_result"]
    assert "cache_control" in blocks[0]
    assert "cache_control" in blocks[2]
    assert len(_marker_entries(coalesced_body)) == 3

    # Case C: Leading ephemeral frame skipped for split; trailing ephemeral frame skipped for tail.
    lead_eph = Message.user(_LONG_PROMPT)
    lead_eph._ephemeral = True
    trail_eph = Message.user("transient pending frame")
    trail_eph._ephemeral = True
    eph_body = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=[lead_eph, Message.user(_LONG_PROMPT), trail_eph],
        )
    )
    # Leading ephemeral is untouched (unsplit, unmarked)
    assert len(eph_body["messages"][0]["content"]) == 1
    assert "cache_control" not in eph_body["messages"][0]["content"][0]
    # First persistent user (index 1) is split and carries static_prefix + persistent tail
    assert len(eph_body["messages"][1]["content"]) == 2
    assert "cache_control" in eph_body["messages"][1]["content"][0]
    assert "cache_control" in eph_body["messages"][1]["content"][1]
    # Trailing ephemeral (index 2) is unmarked
    assert "cache_control" not in eph_body["messages"][2]["content"][0]
    assert len(_marker_entries(eph_body)) == 3


def test_four_breakpoint_cap_across_saver_settled_image_and_prev_user_combinations() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=1)
    sys_blocks = ["instructions", "inventory", "skills", "env", "output-contract"]

    # Deep settled image + prev_user strictly after settled_idx + static_prefix + frozen system
    deep_image_history = [
        Message.user(_LONG_PROMPT),
        Message.assistant("step 1"),
        Message.user("screenshot", [_tiny_image("b2xk")]),
        Message.assistant("step 2"),
        Message.user("mid user turn"),
        Message.assistant("step 3"),
        Message.user("latest turn"),
    ]

    # 1. Frozen system: system[-1] MUST survive; prev_user ("mid user turn") is omitted.
    body_frozen_img = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=deep_image_history,
            context_tokens_hint=50_000,
        )
    )
    entries = _marker_entries(body_frozen_img)
    assert len(entries) == 4
    assert [p for p, _ in entries] == [
        ".system[4]",  # Priority 4: sole frozen system marker preserved
        ".messages[0].content[0]",  # Priority 3: static_prefix_block
        ".messages[1].content[0]",  # Priority 2: settled boundary before image at index 2
        ".messages[6].content[0]",  # Priority 1: persistent rolling tail
    ]
    # Every marker carries the exact same request-wide 1h TTL.
    assert {tuple(sorted(cc.items())) for _, cc in entries} == {
        (("ttl", "1h"), ("type", "ephemeral"))
    }

    # 2. Radient Saver frozen_prefix_messages + static_prefix + frozen system
    body_saver = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=sys_blocks,
            system_blocks_frozen=True,
            messages=deep_image_history,
            frozen_prefix_messages=4,  # epoch ends at messages[3] ("step 2")
            context_tokens_hint=50_000,
        )
    )
    saver_entries = _marker_entries(body_saver)
    assert len(saver_entries) == 4
    assert [p for p, _ in saver_entries] == [
        ".system[4]",
        ".messages[0].content[0]",  # static_prefix_block
        ".messages[3].content[0]",  # Saver frozen_block
        ".messages[6].content[0]",  # rolling tail
    ]
    assert {tuple(sorted(cc.items())) for _, cc in saver_entries} == {
        (("ttl", "1h"), ("type", "ephemeral"))
    }

    # 3. Legacy (non-frozen) 6 system blocks + deep settled image + static_prefix:
    #    trailing legacy system markers trimmed, system[0] preserved, prev_user omitted.
    body_legacy_img = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=[f"sys-{i}" for i in range(6)],
            system_blocks_frozen=False,
            messages=deep_image_history,
        )
    )
    legacy_entries = _marker_entries(body_legacy_img)
    assert len(legacy_entries) == 4
    assert [p for p, _ in legacy_entries] == [
        ".system[0]",
        ".messages[0].content[0]",
        ".messages[1].content[0]",
        ".messages[6].content[0]",
    ]

    # 4. No system blocks at all: prev_user fits in the 4th slot without evicting anything.
    body_no_sys = client._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=[],
            system_blocks_frozen=True,
            messages=deep_image_history,
        )
    )
    no_sys_entries = _marker_entries(body_no_sys)
    assert len(no_sys_entries) == 4
    assert [p for p, _ in no_sys_entries] == [
        ".messages[0].content[0]",  # static_prefix_block
        ".messages[1].content[0]",  # settled boundary
        ".messages[4].content[0]",  # prev_user fits since no system markers exist
        ".messages[6].content[0]",  # rolling tail
    ]


def test_native_provider_replay_on_first_user_message_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the first persistent user message carries a native provider payload,
    never mutate or split it."""
    import local_operator.providers.clients as clients_mod

    first = Message.user(_LONG_PROMPT)
    second = Message.user("follow-up")
    native_payload = [{"type": "text", "text": _LONG_PROMPT}]
    monkeypatch.setattr(
        clients_mod,
        "_anthropic_native_blocks",
        lambda msg, _model, _url, _scope: native_payload if msg.id == first.id else None,
    )
    body = AnthropicClient()._build_body(
        ChatRequest(
            model=_anthropic_spec(),
            system_blocks=["instructions", "env"],
            system_blocks_frozen=True,
            messages=[first, Message.assistant("ok"), second],
        )
    )
    assert len(body["messages"][0]["content"]) == 1
    assert body["messages"][0]["content"][0]["text"] == _LONG_PROMPT


# ---------------------------------------------------------------------------
# 4. BedrockAnthropicClient wire inheritance (Turn 1 + Turn 2+ + TTL)
# ---------------------------------------------------------------------------


def test_bedrock_anthropic_client_inherits_frozen_system_and_static_prefix() -> None:
    client = BedrockAnthropicClient(region="us-east-1", cache_ttl_1h_min_context_tokens=1)
    sys_blocks = ["instructions", "inventory", "skills", "env", "output-contract"]
    user0 = Message.user(_LONG_PROMPT)

    # Turn 1 on 1h-capable Bedrock model
    req1 = ChatRequest(
        model=_bedrock_spec("us.anthropic.claude-sonnet-4-6"),
        system_blocks=sys_blocks,
        system_blocks_frozen=True,
        messages=[user0],
        context_tokens_hint=50_000,
    )
    body1 = client._build_body(req1)
    wire1 = transform_body(body1, client._headers(None))
    entries1 = _marker_entries(wire1)
    assert [p for p, _ in entries1] == [
        ".system[4]",
        ".messages[0].content[0]",
        ".messages[0].content[1]",
    ]
    assert {tuple(sorted(cc.items())) for _, cc in entries1} == {
        (("ttl", "1h"), ("type", "ephemeral"))
    }

    # Turn 2 on same Bedrock model: messages[0] content stays byte-identical
    req2 = req1.model_copy(
        update={"messages": [user0, Message.assistant("working"), Message.user("next")]}
    )
    body2 = client._build_body(req2)
    wire2 = transform_body(body2, client._headers(None))
    entries2 = _marker_entries(wire2)
    assert _strip_cache_control(wire1["messages"][0]) == _strip_cache_control(wire2["messages"][0])
    assert [p for p, _ in entries2] == [
        ".system[4]",
        ".messages[0].content[0]",
        ".messages[0].content[1]",
        ".messages[2].content[0]",
    ]
    assert {tuple(sorted(cc.items())) for _, cc in entries2} == {
        (("ttl", "1h"), ("type", "ephemeral"))
    }

    # Unknown Bedrock model id falls back to 5m TTL consistently on ALL markers
    req_5m = req1.model_copy(update={"model": _bedrock_spec("us.anthropic.claude-unknown-v1:0")})
    body_5m = client._build_body(req_5m)
    entries_5m = _marker_entries(body_5m)
    assert len(entries_5m) == 3
    assert {tuple(sorted(cc.items())) for _, cc in entries_5m} == {(("type", "ephemeral"),)}


# ---------------------------------------------------------------------------
# 5. Real Session -> AgentLoop -> ChatRequest.system_blocks_frozen plumbing
# ---------------------------------------------------------------------------


class _CapturingStream:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def _gen():
            yield StreamTextDelta(delta="done.")

        return _gen()


@pytest.mark.asyncio
async def test_session_append_only_state_sets_system_blocks_frozen_and_wires_bedrock(
    tmp_path: Path,
) -> None:
    stream = _CapturingStream()

    def provider() -> list[str]:
        return ["instructions", "inventory", "skills", "env", "output-contract"]

    setattr(provider, "append_only_state", True)

    session = Session(
        model=_bedrock_spec(),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "frozen-sess"),
        system_blocks_provider=provider,
    )
    await session.prompt(_LONG_PROMPT)
    await session.prompt("follow-up question")
    await session.complete_aside([Message.user("quick side question")])

    assert len(stream.requests) == 3
    # Turn 1, Turn 2, and complete_aside all carry explicit system_blocks_frozen=True
    assert [req.system_blocks_frozen for req in stream.requests] == [True, True, True]

    # Feed real captured requests directly into BedrockAnthropicClient._build_body
    bedrock = BedrockAnthropicClient(region="us-east-1")
    body1 = bedrock._build_body(stream.requests[0])
    body2 = bedrock._build_body(stream.requests[1])
    aside_body = bedrock._build_body(stream.requests[2])

    assert [i for i, b in enumerate(body1["system"]) if "cache_control" in b] == [4]
    assert body1["messages"][0]["content"][0]["text"] == _PREAMBLE
    assert "cache_control" in body1["messages"][0]["content"][0]
    assert len(_marker_entries(body1)) <= AnthropicClient.MAX_CACHE_BREAKPOINTS

    assert [i for i, b in enumerate(body2["system"]) if "cache_control" in b] == [4]
    assert _strip_cache_control(body1["messages"][0]) == _strip_cache_control(body2["messages"][0])
    assert len(_marker_entries(body2)) <= AnthropicClient.MAX_CACHE_BREAKPOINTS

    # Aside rides the exact same frozen system[-1] breakpoint!
    assert [i for i, b in enumerate(aside_body["system"]) if "cache_control" in b] == [4]


@pytest.mark.asyncio
async def test_session_without_append_only_state_keeps_system_blocks_frozen_false(
    tmp_path: Path,
) -> None:
    stream = _CapturingStream()

    def plain_provider() -> list[str]:
        return ["instructions", "inventory", "skills", "env"]

    session = Session(
        model=_anthropic_spec(),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "plain-sess"),
        system_blocks_provider=plain_provider,
    )
    await session.prompt("hello")
    assert len(stream.requests) == 1
    assert stream.requests[0].system_blocks_frozen is False

    body = AnthropicClient()._build_body(stream.requests[0])
    # Legacy 4-block non-frozen behavior: blocks 0 and 1 marked, blocks 2 and 3 unmarked
    assert [i for i, b in enumerate(body["system"]) if "cache_control" in b] == [0, 1]
