"""The analytics data model: component attribution and the aggregate math.

Two things are proven here because they are the two easy things to get wrong:
the estimated component split must SUM to the authoritative context total (no
tokens invented, none lost to rounding), and the derived rates on
``UsageAggregate`` (thinking/generation, cache hit) must read straight off the
provider counts rather than off the estimate.
"""

from __future__ import annotations

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    CallSnapshot,
    UsageAggregate,
    apportion_components,
    snapshot_component_chars,
    split_system_prompt,
)
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolResult,
)


async def _noop(tool_call_id: str, *_args: object) -> ToolResult:
    return ToolResult(tool_call_id=tool_call_id, tool_name="stub", content=[])


def test_split_system_prompt_no_custom_section():
    # A block with no custom-instructions header is all packaged persona.
    block = "You are the assistant.\n\nAGENTS.md guidance here."
    system, custom = split_system_prompt(block)
    assert system == len(block)
    assert custom == 0


def test_split_system_prompt_with_custom_section():
    persona = "You are the assistant.\n\n"
    custom = "## User's custom instructions\n\n<user_instructions>be terse</user_instructions>"
    block = persona + custom
    system_chars, custom_chars = split_system_prompt(block)
    assert system_chars == len(persona)
    assert custom_chars == len(custom)
    assert system_chars + custom_chars == len(block)


def test_apportion_sums_to_context_total():
    # An awkward ratio that does not divide evenly: the largest-remainder
    # rounding must still hand out exactly ``context_tokens`` with no drift.
    chars = {
        "system_prompt": 333,
        "custom_instructions": 111,
        "tool_schemas": 777,
        "conversation": 1234,
        "tool_results": 555,
    }
    for total in (1, 7, 100, 9999, 1_000_003):
        split = apportion_components(chars, total)
        assert sum(split.values()) == total, total
        # Every component key is present, even the zero ones.
        assert set(split) == set(COMPONENT_KEYS)


def test_apportion_zero_context_is_all_zero():
    chars = {"conversation": 500, "system_prompt": 500}
    split = apportion_components(chars, 0)
    assert set(split) == set(COMPONENT_KEYS)
    assert all(v == 0 for v in split.values())


def test_apportion_is_deterministic():
    chars = {"system_prompt": 500, "conversation": 500, "tool_results": 500}
    a = apportion_components(chars, 1000)
    b = apportion_components(chars, 1000)
    assert a == b


def test_snapshot_component_chars_separates_regions():
    persona = "You are the assistant. " * 5
    custom = "## User's custom instructions\n\nbe terse"
    # The whole custom section starts at its header; everything before the
    # header (persona + the blank-line separator) is system_prompt.
    separator = "\n\n"
    block0 = persona + separator + custom
    tools = [
        AgentTool(
            name="bash",
            description="run a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            execute=_noop,
        )
    ]
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="m"),
        system_blocks=[
            block0,
            "## Available tools\ninventory prose",
            "env facts",
            "<skills>k</skills>",
        ],
        messages=[
            Message(role="user", content=[TextContent(text="hello there")]),
            Message(
                role="tool",
                tool_call_id="t1",
                tool_name="bash",
                content=[TextContent(text="command output")],
            ),
        ],
        tools=tools,
    )
    chars = snapshot_component_chars(request)
    # system_prompt is persona + separator (everything up to the header).
    assert chars["system_prompt"] == len(persona) + len(separator)
    assert chars["custom_instructions"] == len(custom)
    assert chars["system_prompt"] + chars["custom_instructions"] == len(block0)
    assert chars["tool_inventory"] == len("## Available tools\ninventory prose")
    assert chars["environment"] == len("env facts")
    assert chars["knowledge"] == len("<skills>k</skills>")
    assert chars["conversation"] == len("hello there")
    assert chars["tool_results"] == len("command output")
    # Tool schema is name + description + compact params JSON, and is nonzero.
    assert chars["tool_schemas"] > len("bash") + len("run a shell command")
    # No image blocks here, so the images bucket is empty.
    assert chars["images"] == 0


def test_snapshot_component_chars_splits_images_from_text():
    # An image block in a message must attribute to ``images``, NOT to the text
    # bucket the message sits in — and the text chars beside it must stay in
    # that text bucket. This is the double-counting guard: apportionment splits
    # a fixed total by chars, so an image counted in both would over-weight its
    # text bucket. Images come from BOTH conversation and tool messages.
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="m"),
        system_blocks=["", "", "", ""],
        messages=[
            Message(
                role="user",
                content=[
                    TextContent(text="look at this"),
                    ImageContent(data="deadbeef", mime_type="image/png"),
                ],
            ),
            Message(
                role="tool",
                tool_call_id="t1",
                tool_name="screenshot",
                content=[
                    TextContent(text="captured"),
                    ImageContent(data="cafe", mime_type="image/png"),
                ],
            ),
        ],
        tools=[],
    )
    chars = snapshot_component_chars(request)
    # Text chars stay in their own buckets, unaffected by the images beside them.
    assert chars["conversation"] == len("look at this")
    assert chars["tool_results"] == len("captured")
    # Two image blocks (one conversation, one tool) sum into ``images`` and
    # nowhere else — the flat char proxy per image, so a positive multiple of 2.
    assert chars["images"] > 0
    assert chars["images"] % 2 == 0
    # The image chars are NOT also folded into the text buckets (the guard).
    assert chars["conversation"] + chars["tool_results"] == len("look at this") + len("captured")


def test_aggregate_generation_and_cache_rate():
    agg = UsageAggregate(
        calls=3,
        ok_calls=3,
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=8000,
        cache_write_tokens=200,
        reasoning_tokens=120,
        context_tokens=9200,
    )
    # generation = output - reasoning (thinking is a subset of output).
    assert agg.generation_tokens == 380
    # total billed = full input context (incl. cache) + output. context_tokens
    # is the normalised full input, so 9200 + 500 = 9700 — NOT input+output,
    # which would drop the cache volume on a cache-exclusive provider (A1).
    assert agg.total_tokens == 9700
    # cache hit rate = cache_read / context, capped at 1.0.
    assert agg.cache_hit_rate is not None
    assert round(agg.cache_hit_rate, 4) == round(8000 / 9200, 4)


def test_total_tokens_includes_cache_on_cache_exclusive_provider():
    # Anthropic reports input_tokens EXCLUDING cache; context_tokens normalises
    # to the full input. total_tokens must use context (A1 regression): a 100k
    # cached turn must not report ~7k.
    agg = UsageAggregate(
        input_tokens=5000,  # fresh input only (Anthropic shape)
        cache_read_tokens=90_000,
        cache_write_tokens=5000,
        output_tokens=2000,
        context_tokens=100_000,  # input + cache_read + cache_write
    )
    assert agg.total_tokens == 102_000  # context + output, not input + output


def test_cost_usd_from_micro():
    agg = UsageAggregate(calls=2, cost_micro=8_340_000, cost_known_calls=2)
    assert agg.cost_usd == 8.34
    assert agg.cost_is_known is True
    assert agg.cost_is_partial is False


def test_cost_partial_when_some_calls_unpriced():
    # 10 calls, only 7 priceable -> the dollar figure is a lower bound.
    agg = UsageAggregate(calls=10, cost_micro=1_500_000, cost_known_calls=7)
    assert agg.cost_is_partial is True
    assert agg.cost_is_known is True


def test_cost_unknown_when_nothing_priced():
    # A local-model-only run: no call had a price.
    agg = UsageAggregate(calls=5, cost_micro=0, cost_known_calls=0)
    assert agg.cost_is_known is False
    assert agg.cost_is_partial is True  # 0 < 5
    assert agg.cost_usd == 0.0


def test_callsnapshot_carries_cost():
    snap = CallSnapshot(
        ts_ms=1,
        session_id="s",
        provider="anthropic",
        model_id="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=2,
        component_chars={},
        cost_micro=12345,
        cost_known=True,
    )
    assert snap.cost_micro == 12345
    assert snap.cost_known is True


def test_aggregate_cache_rate_none_without_context():
    agg = UsageAggregate(calls=1, context_tokens=0, cache_read_tokens=0)
    assert agg.cache_hit_rate is None


def test_aggregate_generation_never_negative():
    # A provider that reports reasoning >= output (shouldn't happen, but the
    # math must stay non-negative rather than showing a negative generation).
    agg = UsageAggregate(output_tokens=100, reasoning_tokens=150)
    assert agg.generation_tokens == 0


def test_fresh_tokens_independent_of_provider_input_shape():
    # Anthropic: input already excludes cache, so fresh == input.
    anthropic = UsageAggregate(
        input_tokens=5_000,
        cache_read_tokens=90_000,
        cache_write_tokens=5_000,
        context_tokens=100_000,
    )
    assert anthropic.fresh_tokens == 5_000
    # OpenAI: input already includes cache, so fresh is the remainder.
    openai = UsageAggregate(
        input_tokens=100_000,
        cache_read_tokens=80_000,
        cache_write_tokens=0,
        context_tokens=100_000,
    )
    assert openai.fresh_tokens == 20_000
    # Never negative even if a provider over-reports cache against context.
    over = UsageAggregate(
        input_tokens=10,
        cache_read_tokens=80,
        cache_write_tokens=30,
        context_tokens=100,
    )
    assert over.fresh_tokens == 0


def test_callsnapshot_is_frozen_scalars():
    # The snapshot is handed to a background thread; it must be a plain value.
    snap = CallSnapshot(
        ts_ms=1,
        session_id="s",
        provider="p",
        model_id="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=2,
        component_chars={"conversation": 4},
        ok=True,
    )
    assert snap.session_id == "s"
    # frozen dataclass: attribute assignment raises.
    try:
        snap.session_id = "x"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised
