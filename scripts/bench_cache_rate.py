#!/usr/bin/env python3
"""Cache-rate benchmark for the harness rewrite.

Runs a multi-turn agentic task and reports the prompt-cache hit rate, per the
performance contract in docs/REWRITE.md (>= 90% target; omp runs ~95%).

Two measurements:
  1. Structural prefix stability (always, no network): for each turn, the
     actual Anthropic content is serialized in cache hierarchy order. Matching
     bytes measure prefix eligibility, not provider cache hits: tokenization,
     expiry, account routing and server policy are not simulated.
  2. Live cache rate (when OPENROUTER_API_KEY is available): sum of
     cache_read_tokens / prompt-side tokens across turns from provider usage.

The structural number is a regression aid; only provider usage measures hits.

Run: .venv/bin/python scripts/bench_cache_rate.py [--turns N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.harness.types import (  # noqa: E402
    AbortSignal,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
    Usage,
)
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.providers.clients import AnthropicClient  # noqa: E402
from local_operator.tools.registry import create_tools  # noqa: E402

TASK_PROMPTS = [
    "Create a file todo_app.py with a Todo dataclass (id, text, done).",
    "Add a TodoList class with add/complete/remove methods and a len().",
    "Write a test file test_todo_app.py exercising all three methods.",
    "Read both files back and confirm they are consistent.",
]


def _serialize_request(req: ChatRequest, client: AnthropicClient) -> bytes:
    wire = client._build_body(req)

    def content(value):
        # Breakpoint placement is cache policy, not model input. Preserve real
        # schemas, descriptions, images and function-call ordering; omit only
        # cache markers so moving the write boundary doesn't look like changed
        # conversation content. Separate live usage validates actual reuse.
        if isinstance(value, dict):
            return {key: content(item) for key, item in value.items() if key != "cache_control"}
        if isinstance(value, list):
            return [content(item) for item in value]
        return value

    body = {key: content(wire.get(key, [])) for key in ("tools", "system", "messages")}
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()


def _prompt_tokens(provider: str, usage: Usage) -> int:
    """Count input once under each wire's documented usage convention."""
    if provider == "anthropic":
        return usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    return usage.input_tokens


def _common_prefix_len(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    step = 65536
    pos = 0
    while pos < limit:
        end = min(pos + step, limit)
        if a[pos:end] != b[pos:end]:
            for i in range(pos, end):
                if a[i] != b[i]:
                    return i
            return end
        pos = end
    return limit


def _prefix_stability(requests: list[bytes]) -> float:
    """Fraction of each request's bytes that were a prefix of the next."""
    if len(requests) < 2:
        return 1.0
    stable = 0
    total = 0
    for prev, nxt in zip(requests, requests[1:]):
        stable += _common_prefix_len(prev, nxt)
        total += len(nxt)
    return stable / total if total else 1.0


async def _mock_stream(req: ChatRequest, signal: AbortSignal | None):
    yield StreamTextDelta(delta=f"Done: {req.messages[-1].text[:40]}\n")
    yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=0, output_tokens=10))


async def run_structural(turns: int) -> float:
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    requests: list[bytes] = []
    wire_client = AnthropicClient()

    async def capturing_stream(req: ChatRequest, signal: AbortSignal | None):
        requests.append(_serialize_request(req, wire_client))
        async for ev in _mock_stream(req, signal):
            yield ev

    tools = create_tools(ToolContext(cwd=str(REPO), session_id="bench"))
    # The skills block is selected ONCE (the first prompt) and frozen for the
    # session — per-turn re-selection would invalidate the whole conversation
    # prefix on every change. The bench models the live contract: the first
    # provider call carries the selected block, later calls reuse it
    # byte-identically, so the measured stability is the stability the wire
    # actually gets.
    frozen: dict[str, str | None] = {"block": None}

    def provider() -> list[str]:
        block = frozen["block"]
        if block is None:
            block = "<skills>\nminerva-observability: Datadog playbooks\n</skills>"
            frozen["block"] = block
        return build_system_blocks(tools, block, "bench env", "2026-08-04")

    transcript = Transcript(Path(tempfile.mkdtemp(prefix="lo-bench-")))
    session = Session(
        model=ModelSpec(provider="mock", model_id="mock"),
        stream_fn=capturing_stream,
        tools=tools,
        transcript=transcript,
        session_id="bench-cache",
        yolo=True,
        system_blocks_provider=provider,
    )
    for prompt in TASK_PROMPTS[:turns]:
        await session.prompt(prompt)
    await session.dispose()
    await wire_client.aclose()
    return _prefix_stability(requests)


async def run_live(turns: int) -> float | None:
    # Direct Anthropic reports cache_read/cache_write; OpenRouter's shared
    # pool does not surface cache stats (verified 2026-08-04), so prefer a
    # direct key and note the limitation otherwise.
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        spec = ModelSpec(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            context_window=200_000,
            supports_prompt_cache=True,
        )
        client = AnthropicClient()
    elif os.environ.get("OPENROUTER_API_KEY"):
        from local_operator.providers.clients import OpenAICompatClient

        key = os.environ["OPENROUTER_API_KEY"]
        spec = ModelSpec(
            provider="openrouter",
            model_id="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            context_window=200_000,
            supports_prompt_cache=True,
        )
        client = OpenAICompatClient(spec.base_url or "https://openrouter.ai/api/v1")
    else:
        return None
    tools = create_tools(ToolContext(cwd=str(REPO), session_id="bench-live"))
    blocks = build_system_blocks(tools, "", "bench env", "2026-08-04")
    messages: list[Message] = []
    cache_read = 0
    prompt_total = 0
    for prompt in TASK_PROMPTS[:turns]:
        messages.append(Message.user(prompt))
        req = ChatRequest(model=spec, system_blocks=blocks, messages=list(messages), tools=tools)
        last_usage = None
        async for ev in client.stream(req, api_key=key):
            if isinstance(ev, StreamEndEvent):
                last_usage = ev.usage
        if last_usage:
            cache_read += last_usage.cache_read_tokens
            prompt_total += _prompt_tokens(spec.provider, last_usage)
        messages.append(Message.assistant("ok"))
    await client.aclose()
    return (cache_read / prompt_total) if prompt_total else None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument(
        "--live", action="store_true", help="Also make billable cache-probe requests"
    )
    args = parser.parse_args()

    stability = await run_structural(args.turns)
    print(f"structural prefix eligibility: {stability:.1%} (not a measured hit rate)")

    try:
        live = await run_live(args.turns) if args.live else None
    except Exception as exc:  # live path is evidence, not the contract
        print(f"live cache rate: skipped ({type(exc).__name__}: {exc})")
        live = None
    if live is not None:
        print(f"live cache rate: {live:.1%}")
    else:
        print("live cache rate: skipped (--live and a supported credential required)")

    # A small cold-start sample cannot guarantee a fleet cache target. Gate
    # local prefix regressions, and report observed cache usage independently.
    ok = stability >= 0.90
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
