"""Live measurement of the compaction advisor's prompt-cache economics.

The advisor (BETA) reads the WHOLE conversation on every call. That is only
affordable if the read hits the provider's cache: on the session that motivated
the feature, 92.9% of prompt-side tokens were cache reads, and an advisor call
riding that warm prefix costs roughly 2.6% of the bill. The same call on a COLD
prefix costs about 25.6% and makes the feature a net loss. The design therefore
sets ``isolated=False`` specifically to keep the session's ``prompt_cache_key``,
and that claim must not ship unmeasured.

This script makes four REAL streamed calls and prints the provider's own cache
counters for each:

1. WARM      — a working-turn-shaped request that writes the prefix.
2. ADVISOR   — the advisor request shape (same system + tools + conversation,
               plus the advisor's own short question, ``tool_choice="none"``),
               ``isolated=False``: what ships.
3. RE-WARM   — re-writes the prefix so (4) is not measured against (2)'s.
4. ADVISOR-ISOLATED — the same request with ``isolated=True``: the variant the
               design rejected. On the OpenAI wire this drops
               ``prompt_cache_key``; on Anthropic, caching is keyed by prefix
               CONTENT rather than by that key, so this arm is what tells us
               whether isolation actually costs anything on this provider.
5. ADVISOR-SYSBLOCK — the advisor's instructions carried as an extra SYSTEM
               block instead of inside the appended user turn. System sits in
               the cache prefix ahead of the messages, so this arm measures
               whether that placement costs the whole prefix.

Both wires are reported because the answer differs by provider and the design
comment should be able to name which one it is talking about.

Run (needs the configured Anthropic credential):
    .venv/bin/python scripts/measure_advisor_cache.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.advisor import ADVISOR_SYSTEM_PROMPT  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    ChatRequest,
    Message,
    StreamUsageEvent,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.providers.auth_store import AuthStore, default_db_path  # noqa: E402
from local_operator.providers.clients import client_for_spec  # noqa: E402


def _noop(*_a, **_k):
    raise AssertionError("tool must not run")


def _tools() -> list[AgentTool]:
    """A realistic core-tool surface, so the tools block is a real chunk of the
    cache prefix (it sits at position 0 on every wire)."""
    names = ["bash", "read", "write", "edit", "grep", "glob", "eval", "task"]
    return [
        AgentTool(
            name=name,
            description=f"{name} tool: does {name} things with useful parameters",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "target path"},
                    "content": {"type": "string", "description": "payload"},
                },
            },
            execute=_noop,
        )
        for name in names
    ]


def _conversation() -> list[Message]:
    """A long exchange, so cache read/write counts are unambiguous."""
    convo: list[Message] = []
    for i in range(12):
        convo.append(
            Message.user((f"Step {i}: please explain concept number {i} in detail. " * 10).strip())
        )
        convo.append(Message.assistant((f"Concept {i} explained thoroughly. " * 60).strip()))
    convo.append(Message.user(("Now continue with the next part of the task. " * 6).strip()))
    return convo


SYSTEM = [
    "You are a helpful coding agent. " * 30,
    "Environment: macOS, python 3.14. " * 20,
]

#: The advisor's own appended turn. Short by design — the conversation is
#: already in the message list, so restating it would pay twice AND break the
#: prefix this measurement is about.
ADVISOR_QUESTION = (
    "A compaction decision is pending for the conversation above.\n"
    "Context size: 480,000 tokens.\nAutomatic compaction threshold: 600,000 tokens.\n"
    "Answer with the fenced JSON block described in your instructions and nothing else."
)


async def _run(client, oauth, request, label):
    usage = None
    async for event in client.stream(request, None, oauth_access=oauth):
        if isinstance(event, StreamUsageEvent):
            usage = event.usage
    if usage is None:
        print(f"{label}: no usage reported")
        return None
    total = (
        (usage.cache_read_tokens or 0) + (usage.cache_write_tokens or 0) + (usage.input_tokens or 0)
    )
    rate = (usage.cache_read_tokens or 0) / total * 100 if total else 0.0
    print(
        f"{label}: cache_read={usage.cache_read_tokens} "
        f"cache_write={usage.cache_write_tokens} input={usage.input_tokens} "
        f"output={usage.output_tokens} context={usage.context_tokens} "
        f"cache_hit={rate:.1f}%"
    )
    return usage


async def main() -> None:
    spec = build_model_spec("anthropic", "claude-opus-4-8")
    store = AuthStore(default_db_path())
    oauth = await store.get_oauth_access("anthropic")
    client = client_for_spec(spec)

    tools = _tools()
    convo = _conversation()
    advisor_messages = [*convo, Message.user(ADVISOR_QUESTION)]

    warm = ChatRequest(model=spec, system_blocks=list(SYSTEM), messages=list(convo), tools=tools)

    def advisor(isolated: bool, system_block: bool = False) -> ChatRequest:
        return ChatRequest(
            model=spec,
            # SHIPPED SHAPE: the session's system blocks UNCHANGED. The
            # advisor's instructions ride inside the appended user turn
            # instead, because system sits in the cache prefix ahead of the
            # messages and adding a block there diverges the prefix — arm 5
            # measures exactly that.
            system_blocks=[*SYSTEM, ADVISOR_SYSTEM_PROMPT] if system_block else list(SYSTEM),
            messages=(
                list(advisor_messages)
                if system_block
                else [*convo, Message.user(f"{ADVISOR_SYSTEM_PROMPT}\n\n{ADVISOR_QUESTION}")]
            ),
            tools=tools,
            tool_choice="none",
            replayable=True,
            isolated=isolated,
        )

    await _run(client, oauth, warm, "1 WARM (turn, tools)         ")
    await asyncio.sleep(1)
    shipped = await _run(client, oauth, advisor(False), "2 ADVISOR isolated=False    ")
    await asyncio.sleep(1)
    await _run(client, oauth, warm, "3 RE-WARM (turn, tools)     ")
    await asyncio.sleep(1)
    isolated = await _run(client, oauth, advisor(True), "4 ADVISOR isolated=True     ")
    await asyncio.sleep(1)
    await _run(client, oauth, warm, "5 RE-WARM (turn, tools)     ")
    await asyncio.sleep(1)
    sysblock = await _run(client, oauth, advisor(False, True), "6 ADVISOR system-block      ")

    if shipped and isolated:
        print(
            f"\nRESULT: advisor cache_read shipped={shipped.cache_read_tokens} "
            f"isolated={isolated.cache_read_tokens} "
            f"(delta={shipped.cache_read_tokens - isolated.cache_read_tokens})"
        )
    if sysblock:
        print(
            f"        system-block placement cache_read={sysblock.cache_read_tokens} "
            f"cache_write={sysblock.cache_write_tokens}"
        )


if __name__ == "__main__":
    asyncio.run(main())
