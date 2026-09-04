"""Efficiency reports must count provider usage once and preserve wire content."""

import asyncio
import json

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    Usage,
)
from local_operator.providers.clients import AnthropicClient
from scripts import bench_cache_rate, bench_complex_tasks


def test_cache_denominators_follow_the_provider_counting_contract():
    assert (
        bench_cache_rate._prompt_tokens(
            "anthropic", Usage(input_tokens=100, cache_read_tokens=800, cache_write_tokens=100)
        )
        == 1000
    )
    assert (
        bench_cache_rate._prompt_tokens(
            "openrouter", Usage(input_tokens=1000, cache_read_tokens=800, cache_write_tokens=100)
        )
        == 1000
    )


def test_complex_tally_deduplicates_message_and_turn_end_and_prices_cached_usage(
    tmp_path, monkeypatch
):
    from local_operator.model import configure, registry

    calls = []

    def price(provider, info, usage):
        calls.append(usage)
        return usage.cache_read_tokens / 1000

    monkeypatch.setattr(configure, "cost_for_usage", price)
    monkeypatch.setattr(registry, "get_model_info", lambda *args: object())
    message = {
        "id": "message-1",
        "role": "assistant",
        "usage": {
            "input_tokens": 1000,
            "cache_read_tokens": 900,
            "output_tokens": 50,
            "context_tokens": 1000,
        },
    }
    path = tmp_path / "capture.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"type": kind, "message": message})
            for kind in ["message_end", "turn_end", "message_end"]
        )
    )
    tokens, cost = bench_complex_tasks.tally_cost(path)
    assert tokens == {"input": 1000, "output": 50, "max_context": 1000}
    assert cost == 0.9
    assert len(calls) == 1


def test_structural_probe_uses_real_schemas_and_hierarchy_without_claiming_hits():
    async def noop(*args):
        raise AssertionError("wire construction must never execute tools")

    async def run():
        client = AnthropicClient()
        try:
            tool = AgentTool(
                name="read", description="first description", parameters={}, execute=noop
            )
            request = ChatRequest(
                model=ModelSpec(provider="anthropic", model_id="test"),
                tools=[tool],
                system_blocks=["system"],
                messages=[Message.user("hello")],
            )
            before = bench_cache_rate._serialize_request(request, client)
            assert (
                before.index(b'"tools"') < before.index(b'"system"') < before.index(b'"messages"')
            )
            tool.description = "changed description"
            after = bench_cache_rate._serialize_request(request, client)
            assert before != after
            assert b"first description" in before
            assert b"cache_control" not in before
            assert request.messages[0].id.encode() not in before
        finally:
            await client.aclose()

    asyncio.run(run())


def test_unknown_benchmark_pricing_is_not_reported_as_free(tmp_path, monkeypatch):
    from local_operator.model import registry

    def unknown(*args):
        raise ValueError("unknown model price")

    monkeypatch.setattr(registry, "get_model_info", unknown)
    path = tmp_path / "capture.jsonl"
    path.write_text(
        json.dumps({"type": "message_end", "message": {"id": "1", "usage": {"input_tokens": 100}}})
    )
    _, cost = bench_complex_tasks.tally_cost(path)
    assert cost is None
