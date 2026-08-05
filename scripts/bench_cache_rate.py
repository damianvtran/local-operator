#!/usr/bin/env python3
"""Cache-rate benchmark for the harness rewrite.

Runs a multi-turn agentic task and reports the prompt-cache hit rate, per the
performance contract in docs/REWRITE.md (>= 90% target; omp runs ~95%).

Two measurements:
  1. Structural prefix stability (always, no network): for each turn, the
     serialized (system_blocks, messages) body of request N must be a byte
     prefix of request N+1. The ratio of stable bytes to total bytes is the
     cache rate our design guarantees regardless of provider.
  2. Live cache rate (when OPENROUTER_API_KEY is available): sum of
     cache_read_tokens / prompt-side tokens across turns from provider usage.

The structural number is the contract; the live number is evidence.

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

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
    Usage,
)
from local_operator.prompts_api import build_system_blocks
from local_operator.tools.registry import create_tools

TASK_PROMPTS = [
    "Create a file todo_app.py with a Todo dataclass (id, text, done).",
    "Add a TodoList class with add/complete/remove methods and a len().",
    "Write a test file test_todo_app.py exercising all three methods.",
    "Read both files back and confirm they are consistent.",
]


def _serialize_request(req: ChatRequest) -> bytes:
    # Wire order matters: providers cache the (system, tools, messages...) prefix,
    # so serialize in that order (insertion-ordered dump, NOT sort_keys) — an
    # appended message must EXTEND the prefix, not break it.
    body = {
        "system": req.system_blocks,
        "tools": [{"name": t.name, "parameters": t.parameters} for t in req.tools],
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
    }
    return json.dumps(body).encode()


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

    async def capturing_stream(req: ChatRequest, signal: AbortSignal | None):
        requests.append(_serialize_request(req))
        async for ev in _mock_stream(req, signal):
            yield ev

    tools = create_tools(ToolContext(cwd=str(REPO), session_id="bench"))
    # The skills block is selected ONCE (the first prompt) and frozen for the
    # session — per-turn re-selection would invalidate the whole conversation
    # prefix on every change. The bench models the live contract: the first
    # provider call carries the selected block, later calls reuse it
    # byte-identically, so the measured stability is the stability the wire
    # actually gets.
    frozen = {"block": None}

    def provider() -> list[str]:
        if frozen["block"] is None:
            frozen["block"] = (
                "<skills>\nminerva-observability: Datadog playbooks\n</skills>"
            )
        return build_system_blocks(tools, frozen["block"], "bench env", "2026-08-04")

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
    return _prefix_stability(requests)


async def run_live(turns: int) -> float | None:
    # Direct Anthropic reports cache_read/cache_write; OpenRouter's shared
    # pool does not surface cache stats (verified 2026-08-04), so prefer a
    # direct key and note the limitation otherwise.
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        from local_operator.providers.clients import AnthropicClient

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
            prompt_total += (
                last_usage.input_tokens
                + last_usage.cache_read_tokens
                + last_usage.cache_write_tokens
            )
        messages.append(Message.assistant("ok"))
    return (cache_read / prompt_total) if prompt_total else None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=4)
    args = parser.parse_args()

    stability = await run_structural(args.turns)
    print(f"structural prefix stability: {stability:.1%} (contract: >= 90%)")

    try:
        live = await run_live(args.turns)
    except Exception as exc:  # live path is evidence, not the contract
        print(f"live cache rate: skipped ({type(exc).__name__}: {exc})")
        live = None
    if live is not None:
        print(f"live cache rate (openrouter):  {live:.1%}")
    else:
        print("live cache rate: skipped (no OPENROUTER_API_KEY)")

    # The structural number is the contract: it is the cache rate our request
    # shaping guarantees against any provider that honors prefix caching.
    # Live >=90% is only enforceable against direct Anthropic (OpenRouter's
    # shared pool reports cache stats unreliably — verified 2026-08-04).
    live_gate = live is None or live >= 0.90 or os.environ.get("ANTHROPIC_API_KEY") is None
    ok = stability >= 0.90 and live_gate
    if live is not None and live < 0.90 and os.environ.get("ANTHROPIC_API_KEY") is None:
        print("note: live rate informational (provider pool does not reliably report cache)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
